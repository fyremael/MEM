from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np

from .config import ProbeConfig
from .dk_predictor import (
    DavisKahanResult,
    davis_kahan_bound,
    davis_kahan_from_components,
    symmetric_operator_norm,
)
from .gap import GapSummary, summarize_gaps
from .jacobian_norm import jacobian_spectral_norm
from .model import MemoryChannelsModel, Params
from .subspace import (
    SubspaceResult,
    covariance_operator,
    principal_angle_drift,
    projector_from_basis,
    sensitivity_operator,
    subspace_from_sensitivity,
    top_eigenspace,
)


Array = jax.Array


@dataclass(frozen=True)
class LayerProbeMetrics:
    jacobian_norms: Array
    subspace: SubspaceResult
    gap: GapSummary
    drift_to_next: Array | None
    state_drift_to_next: Array | None
    state_delta_norm: Array | None
    leakage: Array | None
    dk_result: DavisKahanResult | None


@dataclass(frozen=True)
class ProbeRun:
    probe_layers: tuple[int, ...]
    layer_metrics: dict[int, LayerProbeMetrics]
    state_view: str = "memory"

    def to_serializable(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "probe_layers": list(self.probe_layers),
            "state_view": self.state_view,
            "layers": {},
        }
        for layer, metrics in self.layer_metrics.items():
            payload["layers"][str(layer)] = {
                "jacobian_norms": np.asarray(metrics.jacobian_norms).tolist(),
                "operator_type": metrics.subspace.operator_type,
                "ambient_shape": list(metrics.subspace.ambient_shape),
                "eigenvalues": np.asarray(metrics.subspace.eigenvalues).tolist(),
                "eigengap": float(np.asarray(metrics.gap.eigengap)),
                "drift_to_next": None
                if metrics.drift_to_next is None
                else float(np.asarray(metrics.drift_to_next)),
                "state_drift_to_next": None
                if metrics.state_drift_to_next is None
                else float(np.asarray(metrics.state_drift_to_next)),
                "state_delta_norm": None
                if metrics.state_delta_norm is None
                else float(np.asarray(metrics.state_delta_norm)),
                "leakage": None if metrics.leakage is None else float(np.asarray(metrics.leakage)),
                "dk_result": None
                if metrics.dk_result is None
                else {
                    "perturbation_norm": float(np.asarray(metrics.dk_result.perturbation_norm)),
                    "gap": float(np.asarray(metrics.dk_result.gap)),
                    "observed_drift": float(np.asarray(metrics.dk_result.observed_drift)),
                    "predicted_drift": float(np.asarray(metrics.dk_result.predicted_drift)),
                    "ratio": float(np.asarray(metrics.dk_result.ratio)),
                },
            }
        return payload

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_serializable(), indent=2), encoding="ascii")

    def save_npz(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {}
        for layer, metrics in self.layer_metrics.items():
            arrays[f"layer_{layer}_jacobian_norms"] = np.asarray(metrics.jacobian_norms)
            arrays[f"layer_{layer}_eigenvalues"] = np.asarray(metrics.subspace.eigenvalues)
        np.savez(path, **arrays)


def _layer_subspace(states: Array, k: int) -> tuple[SubspaceResult, GapSummary]:
    operator = covariance_operator(states)
    subspace = top_eigenspace(operator, k)
    gap = summarize_gaps(subspace.eigenvalues, k)
    return subspace, gap


def _state_delta_norm(previous_states: Array, next_states: Array) -> Array:
    delta = next_states - previous_states
    delta_norm = jnp.linalg.norm(delta.reshape(delta.shape[0], -1), axis=-1)
    previous_norm = jnp.linalg.norm(previous_states.reshape(previous_states.shape[0], -1), axis=-1)
    next_norm = jnp.linalg.norm(next_states.reshape(next_states.shape[0], -1), axis=-1)
    scale = jnp.maximum(jnp.maximum(previous_norm, next_norm), 1e-8)
    return jnp.mean(delta_norm / scale)


def _project_updates(delta: Array, basis: Array, ambient_shape: tuple[int, ...]) -> Array:
    projector = projector_from_basis(basis)
    if ambient_shape == (delta.shape[-1],):
        return jnp.einsum("ij,btj->bti", projector, delta)
    flat_delta = delta.reshape(delta.shape[0], -1)
    projected = flat_delta @ projector.T
    return projected.reshape(delta.shape)


def _write_leakage(previous_states: Array, next_states: Array, basis: Array) -> Array:
    delta = next_states - previous_states
    ambient_shape = (basis.shape[0],) if basis.shape[0] == delta.shape[-1] else tuple(delta.shape[1:])
    projected = _project_updates(delta, basis, ambient_shape)
    leakage = delta - projected
    leakage_norm = jnp.linalg.norm(leakage.reshape(leakage.shape[0], -1), axis=-1)
    delta_norm = jnp.linalg.norm(delta.reshape(delta.shape[0], -1), axis=-1)
    return jnp.mean(leakage_norm / (delta_norm + 1e-8))


def _layer_map_factory(
    model: MemoryChannelsModel,
    params: Params,
    layer: int,
    *,
    attention_mask: Array | None,
    state_view: str,
):
    return lambda sample_full_state: model.layer_map(
        params,
        layer,
        attention_mask=attention_mask,
        state_view=state_view,
        reference_state=sample_full_state,
    )


def _batched_probe_jacobian_norms(
    model: MemoryChannelsModel,
    params: Params,
    layer: int,
    state_view: str,
    probe_state: Array,
    full_state: Array,
    *,
    attention_mask: Array | None,
    key: Array,
    num_iters: int,
) -> Array:
    keys = jax.random.split(key, probe_state.shape[0])

    def per_sample(sample_state: Array, sample_full_state: Array, sample_key: Array) -> Array:
        layer_fn = model.layer_map(
            params,
            layer,
            attention_mask=attention_mask,
            state_view=state_view,
            reference_state=sample_full_state,
        )
        return jacobian_spectral_norm(
            layer_fn,
            sample_state,
            key=sample_key,
            num_iters=num_iters,
        )

    return jax.vmap(per_sample)(probe_state, full_state, keys)


def _probe_subspace(
    probe_config: ProbeConfig,
    *,
    state: Array,
    full_state: Array,
    layer_fn_factory,
    key: Array,
) -> tuple[SubspaceResult, GapSummary]:
    if probe_config.operator == "covariance":
        return _layer_subspace(state, probe_config.k_eigs)
    if probe_config.operator == "sensitivity":
        subspace = subspace_from_sensitivity(
            None,
            state,
            probe_config.k_eigs,
            key=key,
            num_iters=probe_config.subspace_iters,
            oversample=probe_config.subspace_oversample,
            reference_states=full_state,
            layer_map_factory=layer_fn_factory,
        )
        return subspace, summarize_gaps(subspace.eigenvalues, probe_config.k_eigs)
    raise NotImplementedError(f"Unsupported operator: {probe_config.operator}")


def run_probe_suite(
    model: MemoryChannelsModel,
    params: Params,
    tokens: Array,
    probe_config: ProbeConfig,
    *,
    attention_mask: Array | None = None,
    perturbed_tokens: Array | None = None,
) -> ProbeRun:
    _, full_states = model.apply(params, tokens, attention_mask=attention_mask, return_states=True)
    states = [model.view_state(state, probe_config.state_view) for state in full_states]
    perturbed_full_states = None
    perturbed_states = None
    if perturbed_tokens is not None:
        _, perturbed_full_states = model.apply(
            params,
            perturbed_tokens,
            attention_mask=attention_mask,
            return_states=True,
        )
        perturbed_states = [
            model.view_state(state, probe_config.state_view)
            for state in perturbed_full_states
        ]

    layer_metrics: dict[int, LayerProbeMetrics] = {}
    rng = jax.random.PRNGKey(probe_config.seed)
    for idx, layer in enumerate(probe_config.probe_layers):
        state = states[layer]
        layer_key = jax.random.fold_in(rng, layer)
        layer_fn_factory = _layer_map_factory(
            model,
            params,
            layer,
            attention_mask=attention_mask,
            state_view=probe_config.state_view,
        )
        jacobian_norms = _batched_probe_jacobian_norms(
            model,
            params,
            layer,
            probe_config.state_view,
            state,
            full_states[layer],
            attention_mask=attention_mask,
            key=layer_key,
            num_iters=probe_config.power_iters,
        )
        subspace, gap = _probe_subspace(
            probe_config,
            state=state,
            full_state=full_states[layer],
            layer_fn_factory=layer_fn_factory,
            key=layer_key,
        )

        drift = None
        state_drift = None
        state_delta = None
        leakage = None
        if idx + 1 < len(probe_config.probe_layers):
            next_layer = probe_config.probe_layers[idx + 1]
            next_state = states[next_layer]
            next_subspace, _ = _probe_subspace(
                probe_config,
                state=next_state,
                full_state=full_states[next_layer],
                layer_fn_factory=_layer_map_factory(
                    model,
                    params,
                    next_layer,
                    attention_mask=attention_mask,
                    state_view=probe_config.state_view,
                ),
                key=jax.random.fold_in(rng, next_layer),
            )
            drift = principal_angle_drift(subspace.basis, next_subspace.basis)
            state_subspace, _ = _layer_subspace(state, probe_config.k_eigs)
            next_state_subspace, _ = _layer_subspace(next_state, probe_config.k_eigs)
            state_drift = principal_angle_drift(state_subspace.basis, next_state_subspace.basis)
            state_delta = _state_delta_norm(state, next_state)
            leakage = _write_leakage(state, next_state, subspace.basis)

        dk_result = None
        if perturbed_states is not None:
            if probe_config.operator == "covariance":
                perturbed_operator = covariance_operator(perturbed_states[layer])
                dk_result = davis_kahan_bound(
                    subspace.operator,
                    perturbed_operator,
                    k=probe_config.k_eigs,
                )
            else:
                perturbed_subspace = subspace_from_sensitivity(
                    None,
                    perturbed_states[layer],
                    probe_config.k_eigs,
                    key=jax.random.fold_in(layer_key, 1000),
                    num_iters=probe_config.subspace_iters,
                    oversample=probe_config.subspace_oversample,
                    reference_states=perturbed_full_states[layer],
                    layer_map_factory=_layer_map_factory(
                        model,
                        params,
                        layer,
                        attention_mask=attention_mask,
                        state_view=probe_config.state_view,
                    ),
                )
                observed = principal_angle_drift(subspace.basis, perturbed_subspace.basis)
                base_operator, flat_dim, _ = sensitivity_operator(
                    None,
                    state,
                    reference_states=full_states[layer],
                    layer_map_factory=layer_fn_factory,
                )
                perturbed_operator, _, _ = sensitivity_operator(
                    None,
                    perturbed_states[layer],
                    reference_states=perturbed_full_states[layer],
                    layer_map_factory=_layer_map_factory(
                        model,
                        params,
                        layer,
                        attention_mask=attention_mask,
                        state_view=probe_config.state_view,
                    ),
                )
                diff_operator = lambda vector: perturbed_operator(vector) - base_operator(vector)
                perturbation_norm = symmetric_operator_norm(
                    diff_operator,
                    flat_dim,
                    key=jax.random.fold_in(layer_key, 2000),
                    num_iters=probe_config.power_iters,
                )
                dk_result = davis_kahan_from_components(
                    perturbation_norm=perturbation_norm,
                    gap=gap.eigengap,
                    observed_drift=observed,
                )

        layer_metrics[layer] = LayerProbeMetrics(
            jacobian_norms=jacobian_norms,
            subspace=subspace,
            gap=gap,
            drift_to_next=drift,
            state_drift_to_next=state_drift,
            state_delta_norm=state_delta,
            leakage=leakage,
            dk_result=dk_result,
        )

    return ProbeRun(
        probe_layers=probe_config.probe_layers,
        layer_metrics=layer_metrics,
        state_view=probe_config.state_view,
    )
