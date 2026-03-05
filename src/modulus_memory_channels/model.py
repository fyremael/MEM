from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .config import MemoryModelConfig


Array = jax.Array
Params = dict[str, Any]


def _init_linear(key: Array, in_dim: int, out_dim: int, scale: float = 1.0) -> Array:
    std = scale / jnp.sqrt(float(in_dim))
    return jax.random.normal(key, (in_dim, out_dim)) * std


def rms_norm(x: Array, eps: float) -> Array:
    scale = jnp.reciprocal(jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps))
    return x * scale


def _split_heads(x: Array, num_heads: int) -> Array:
    batch, seq_len, width = x.shape
    head_dim = width // num_heads
    return x.reshape(batch, seq_len, num_heads, head_dim)


def _merge_heads(x: Array) -> Array:
    batch, seq_len, num_heads, head_dim = x.shape
    return x.reshape(batch, seq_len, num_heads * head_dim)


def _split_state(state: Array, d_model: int) -> tuple[Array, Array]:
    return state[..., :d_model], state[..., d_model:]


def _merge_state(tokens: Array, memory: Array) -> Array:
    return jnp.concatenate([tokens, memory], axis=-1)


def _state_norm(x: Array) -> Array:
    return jnp.mean(jnp.linalg.norm(x, axis=-1))


def _stabilize_update(x: Array, eps: float) -> Array:
    return rms_norm(jnp.tanh(x), eps)


def _memory_activation(x: Array) -> Array:
    return jnp.tanh(x)


def _covariance_basis(states: Array, k: int) -> Array:
    samples = states.reshape(-1, states.shape[-1])
    centered = samples - jnp.mean(samples, axis=0, keepdims=True)
    denom = jnp.maximum(centered.shape[0] - 1, 1)
    cov = centered.T @ centered / denom
    eigvals, eigvecs = jnp.linalg.eigh(cov + 1e-6 * jnp.eye(cov.shape[0], dtype=cov.dtype))
    order = jnp.argsort(eigvals)[::-1]
    return eigvecs[:, order[:k]]


def _basis_drift(basis_a: Array, basis_b: Array) -> Array:
    overlap = basis_a.T @ basis_b
    singular_values = jnp.linalg.svd(overlap, compute_uv=False)
    sigma_min = singular_values[-1]
    return jnp.sqrt(jnp.maximum(1.0 - sigma_min**2, 0.0))


def _causal_mask(seq_len: int) -> Array:
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    return mask[None, None, :, :]


def _attention_mask_bias(mask: Array | None, seq_len: int, causal: bool) -> Array:
    attn_mask = None
    if causal:
        attn_mask = _causal_mask(seq_len)
    if mask is not None:
        token_mask = mask[:, None, None, :]
        query_mask = mask[:, None, :, None]
        full_mask = token_mask & query_mask
        attn_mask = full_mask if attn_mask is None else (attn_mask & full_mask)
    if attn_mask is None:
        return 0.0
    return jnp.where(attn_mask, 0.0, -1.0e9)


def init_memory_model_params(key: Array, config: MemoryModelConfig) -> Params:
    per_layer = 13
    keys = jax.random.split(key, config.num_layers * per_layer + 2)
    embed_key = keys[0]
    lm_head_key = keys[1]
    layer_keys = keys[2:]

    params: Params = {
        "token_embed": jax.random.normal(embed_key, (config.vocab_size, config.d_model))
        * (1.0 / jnp.sqrt(float(config.d_model))),
        "lm_head": _init_linear(lm_head_key, config.d_model, config.vocab_size),
        "layers": [],
    }

    for layer_idx in range(config.num_layers):
        offset = layer_idx * per_layer
        (
            wq_key,
            wk_key,
            wv_key,
            wo_key,
            w1_key,
            w2_key,
            write_q_key,
            write_k_key,
            write_v_key,
            write_o_key,
            mem_keep_key,
            read_key,
            read_gate_key,
        ) = layer_keys[
            offset : offset + per_layer
        ]
        memory_identity = jnp.eye(config.memory_dim, dtype=jnp.float32)
        memory_keep = 0.5 * memory_identity + 0.05 * _init_linear(
            mem_keep_key,
            config.memory_dim,
            config.memory_dim,
        )

        params["layers"].append(
            {
                "token_wq": _init_linear(wq_key, config.d_model, config.d_model),
                "token_wk": _init_linear(wk_key, config.d_model, config.d_model),
                "token_wv": _init_linear(wv_key, config.d_model, config.d_model),
                "token_wo": _init_linear(wo_key, config.d_model, config.d_model),
                "w1": _init_linear(w1_key, config.d_model, config.mlp_dim),
                "w2": _init_linear(w2_key, config.mlp_dim, config.d_model),
                "write_wq": _init_linear(write_q_key, config.d_model, config.memory_dim),
                "write_wk": _init_linear(write_k_key, config.d_model, config.memory_dim),
                "write_wv": _init_linear(write_v_key, config.d_model, config.memory_dim),
                "write_wo": _init_linear(write_o_key, config.memory_dim, config.memory_dim),
                "memory_keep": memory_keep,
                "memory_read": _init_linear(read_key, config.memory_dim, config.d_model),
                "read_gate": _init_linear(read_gate_key, config.d_model, 1),
                "alpha": jnp.array(config.alpha_init, dtype=jnp.float32),
                "beta": jnp.array(config.beta_init, dtype=jnp.float32),
                "memory_write_gain": jnp.array(config.memory_write_init, dtype=jnp.float32),
                "memory_keep_gain": jnp.array(config.memory_keep_init, dtype=jnp.float32),
                "memory_read_gain": jnp.array(config.memory_read_init, dtype=jnp.float32),
            }
        )
    return params


def _attention(
    x_q: Array,
    x_kv: Array,
    *,
    wq: Array,
    wk: Array,
    wv: Array,
    wo: Array,
    num_heads: int,
    head_width: int,
    causal: bool,
    attention_mask: Array | None,
) -> Array:
    seq_len = x_q.shape[1]
    q = (x_q @ wq).reshape(x_q.shape[0], x_q.shape[1], num_heads, head_width)
    k = (x_kv @ wk).reshape(x_kv.shape[0], x_kv.shape[1], num_heads, head_width)
    v = (x_kv @ wv).reshape(x_kv.shape[0], x_kv.shape[1], num_heads, head_width)
    scale = 1.0 / jnp.sqrt(jnp.array(head_width, dtype=x_q.dtype))
    scores = jnp.einsum("bthd,bshd->bhts", q, k) * scale
    scores = scores + _attention_mask_bias(attention_mask, seq_len, causal)
    probs = jax.nn.softmax(scores, axis=-1)
    heads = jnp.einsum("bhts,bshd->bthd", probs, v)
    return _merge_heads(heads) @ wo


def _mlp(layer: Params, x: Array) -> Array:
    return jax.nn.gelu(x @ layer["w1"]) @ layer["w2"]


@dataclass(frozen=True)
class MemoryChannelsModel:
    config: MemoryModelConfig

    def init(self, key: Array) -> Params:
        return init_memory_model_params(key, self.config)

    @property
    def state_dim(self) -> int:
        return self.config.d_model + self.config.memory_dim

    def embed_tokens(self, params: Params, tokens: Array) -> Array:
        return params["token_embed"][tokens]

    def initial_state(self, params: Params, tokens: Array) -> Array:
        token_state = self.embed_tokens(params, tokens)
        memory_state = jnp.zeros(
            token_state.shape[:-1] + (self.config.memory_dim,),
            dtype=token_state.dtype,
        )
        return _merge_state(token_state, memory_state)

    def view_state(self, state: Array, state_view: str) -> Array:
        if state_view == "full":
            return state
        tokens, memory = _split_state(state, self.config.d_model)
        if state_view == "token":
            return tokens
        if state_view == "memory":
            return memory
        raise ValueError(f"Unsupported state_view: {state_view}")

    def layer_role_scales(self, layer_index: int) -> dict[str, float]:
        is_write_layer = (
            (layer_index % self.config.memory_write_interval) == 0
            and (layer_index // self.config.memory_write_interval) < self.config.memory_write_layers
        )
        if is_write_layer:
            return {
                "is_write_layer": 1.0,
                "write_scale": 1.0,
                "read_scale": 1.0,
                "mix_scale": 1.0,
            }
        return {
            "is_write_layer": 0.0,
            "write_scale": self.config.keep_write_scale,
            "read_scale": self.config.keep_read_scale,
            "mix_scale": self.config.keep_token_mix_scale,
        }

    def block_apply(
        self,
        layer: Params,
        state: Array,
        *,
        layer_index: int,
        attention_mask: Array | None = None,
        return_stats: bool = False,
    ) -> tuple[Array, dict[str, Array]] | Array:
        scales = self.layer_role_scales(layer_index)
        tokens, memory = _split_state(state, self.config.d_model)
        norm_tokens = rms_norm(tokens, self.config.rms_norm_eps)
        token_attn_out = _attention(
            norm_tokens,
            norm_tokens,
            wq=layer["token_wq"],
            wk=layer["token_wk"],
            wv=layer["token_wv"],
            wo=layer["token_wo"],
            num_heads=self.config.num_heads,
            head_width=self.config.d_model // self.config.num_heads,
            causal=self.config.causal,
            attention_mask=attention_mask,
        )
        token_attn_out = _stabilize_update(token_attn_out, self.config.rms_norm_eps)
        token_mix_delta = scales["mix_scale"] * layer["alpha"] * token_attn_out
        token_after_mix = tokens + token_mix_delta

        if scales["is_write_layer"]:
            write_out = _attention(
                norm_tokens,
                norm_tokens,
                wq=layer["write_wq"],
                wk=layer["write_wk"],
                wv=layer["write_wv"],
                wo=layer["write_wo"],
                num_heads=self.config.num_memory_heads,
                head_width=self.config.memory_dim // self.config.num_memory_heads,
                causal=self.config.causal,
                attention_mask=attention_mask,
            )
            write_out = _stabilize_update(write_out, self.config.rms_norm_eps)
            norm_memory = _memory_activation(memory)
            memory_delta = (
                layer["memory_keep_gain"] * _memory_activation(norm_memory @ layer["memory_keep"])
                + layer["memory_write_gain"] * write_out
            )
            next_memory = memory + memory_delta
        else:
            write_out = jnp.zeros_like(memory)
            memory_delta = jnp.zeros_like(memory)
            next_memory = memory

        read_gate = jax.nn.sigmoid(norm_tokens @ layer["read_gate"])
        memory_read = read_gate * (
            _stabilize_update(
                _memory_activation(next_memory) @ layer["memory_read"],
                self.config.rms_norm_eps,
            )
        )
        token_with_memory = token_after_mix + (
            scales["read_scale"] * layer["memory_read_gain"] * memory_read
        )
        mlp_delta = scales["mix_scale"] * layer["beta"] * _stabilize_update(
            _mlp(
                layer,
                rms_norm(token_with_memory, self.config.rms_norm_eps),
            ),
            self.config.rms_norm_eps,
        )
        token_output = token_with_memory + mlp_delta
        next_state = _merge_state(token_output, next_memory)
        if not return_stats:
            return next_state
        stats = {
            "is_write_layer": jnp.array(scales["is_write_layer"], dtype=token_output.dtype),
            "write_norm": _state_norm(write_out),
            "read_norm": _state_norm(memory_read),
            "memory_delta_norm": _state_norm(memory_delta),
            "token_mix_norm": _state_norm(token_mix_delta + mlp_delta),
        }
        return next_state, stats

    def memory_regularization(
        self,
        params: Params,
        layer_stats: dict[str, Array],
        *,
        subspace_rank: int = 4,
        include_subspace_terms: bool = True,
    ) -> dict[str, Array]:
        identity = jnp.eye(self.config.memory_dim, dtype=jnp.float32)
        keep_identity = jnp.mean(
            jnp.stack(
                [
                    jnp.mean(jnp.square(layer["memory_keep"] - identity))
                    for layer in params["layers"]
                ]
            )
        )
        keep_mask = 1.0 - layer_stats["is_write_layer"]
        keep_count = jnp.maximum(jnp.sum(keep_mask), 1.0)
        keep_write = jnp.sum(keep_mask * jnp.square(layer_stats["write_norm"])) / keep_count
        keep_mix = jnp.sum(keep_mask * jnp.square(layer_stats["token_mix_norm"])) / keep_count
        keep_delta = jnp.sum(keep_mask * jnp.square(layer_stats["memory_delta_norm"])) / keep_count
        if not include_subspace_terms:
            zero = jnp.array(0.0, dtype=keep_write.dtype)
            return {
                "keep_write": keep_write,
                "keep_mix": keep_mix,
                "keep_delta": keep_delta,
                "keep_identity": keep_identity,
                "memory_subspace_drift": zero,
                "memory_subspace_leakage": zero,
            }
        memory_states = layer_stats["memory_states"]
        rank = min(max(subspace_rank, 1), self.config.memory_dim)
        drift_terms = []
        leakage_terms = []
        for layer_index in range(len(params["layers"])):
            prev_memory = memory_states[layer_index]
            next_memory = memory_states[layer_index + 1]
            basis_prev = _covariance_basis(prev_memory, rank)
            basis_next = _covariance_basis(next_memory, rank)
            drift_terms.append(_basis_drift(basis_prev, basis_next))

            projector = basis_prev @ basis_prev.T
            delta = next_memory - prev_memory
            projected = jnp.einsum("ij,btj->bti", projector, delta)
            leakage = delta - projected
            leakage_norm = jnp.linalg.norm(leakage.reshape(leakage.shape[0], -1), axis=-1)
            delta_norm = jnp.linalg.norm(delta.reshape(delta.shape[0], -1), axis=-1)
            leakage_terms.append(jnp.mean(leakage_norm / (delta_norm + 1e-8)))

        drift_terms = jnp.stack(drift_terms)
        leakage_terms = jnp.stack(leakage_terms)
        memory_subspace_drift = jnp.sum(keep_mask * drift_terms) / keep_count
        memory_subspace_leakage = jnp.sum(keep_mask * leakage_terms) / keep_count
        return {
            "keep_write": keep_write,
            "keep_mix": keep_mix,
            "keep_delta": keep_delta,
            "keep_identity": keep_identity,
            "memory_subspace_drift": memory_subspace_drift,
            "memory_subspace_leakage": memory_subspace_leakage,
        }

    def apply_embeddings(
        self,
        params: Params,
        state: Array,
        *,
        attention_mask: Array | None = None,
        return_states: bool = False,
    ) -> tuple[Array, list[Array]] | Array:
        current_state = state
        states = [current_state] if return_states else []
        for layer_index, layer in enumerate(params["layers"]):
            current_state = self.block_apply(
                layer,
                current_state,
                layer_index=layer_index,
                attention_mask=attention_mask,
            )
            if return_states:
                states.append(current_state)
        return (current_state, states) if return_states else current_state

    def forward_with_aux(
        self,
        params: Params,
        tokens: Array,
        *,
        attention_mask: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        current_state = self.initial_state(params, tokens)
        stats_list: list[dict[str, Array]] = []
        memory_states = [self.view_state(current_state, "memory")]
        for layer_index, layer in enumerate(params["layers"]):
            current_state, stats = self.block_apply(
                layer,
                current_state,
                layer_index=layer_index,
                attention_mask=attention_mask,
                return_stats=True,
            )
            stats_list.append(stats)
            memory_states.append(self.view_state(current_state, "memory"))
        final_tokens, _ = _split_state(current_state, self.config.d_model)
        logits = final_tokens @ params["lm_head"]
        stacked_stats = {
            key: jnp.stack([stats[key] for stats in stats_list])
            for key in stats_list[0]
        }
        stacked_stats["memory_states"] = jnp.stack(memory_states)
        return logits, stacked_stats

    def apply(
        self,
        params: Params,
        tokens: Array,
        *,
        attention_mask: Array | None = None,
        return_states: bool = False,
    ) -> tuple[Array, list[Array]] | Array:
        initial_state = self.initial_state(params, tokens)
        output = self.apply_embeddings(
            params,
            initial_state,
            attention_mask=attention_mask,
            return_states=return_states,
        )
        if not return_states:
            return output
        final_state, states = output
        final_tokens, _ = _split_state(final_state, self.config.d_model)
        logits = final_tokens @ params["lm_head"]
        return logits, states

    def logits(
        self,
        params: Params,
        tokens: Array,
        *,
        attention_mask: Array | None = None,
    ) -> Array:
        initial_state = self.initial_state(params, tokens)
        final_state = self.apply_embeddings(params, initial_state, attention_mask=attention_mask)
        final_tokens, _ = _split_state(final_state, self.config.d_model)
        return final_tokens @ params["lm_head"]

    def layer_map(
        self,
        params: Params,
        layer_index: int,
        *,
        attention_mask: Array | None = None,
        state_view: str = "full",
        reference_state: Array | None = None,
    ):
        if layer_index == len(params["layers"]):
            return lambda x: x
        layer = params["layers"][layer_index]

        def f(x: Array) -> Array:
            if state_view == "full":
                state = x
            else:
                if reference_state is None:
                    raise ValueError("reference_state is required for token or memory views.")
                ref_tokens, ref_memory = _split_state(reference_state, self.config.d_model)
                if state_view == "token":
                    state = _merge_state(x, ref_memory)
                elif state_view == "memory":
                    state = _merge_state(ref_tokens, x)
                else:
                    raise ValueError(f"Unsupported state_view: {state_view}")
            batch_x = state[None, ...]
            batch_mask = None if attention_mask is None else attention_mask[None, ...]
            y = self.block_apply(
                layer,
                batch_x,
                layer_index=layer_index,
                attention_mask=batch_mask,
            )
            return self.view_state(y[0], state_view)

        return f
