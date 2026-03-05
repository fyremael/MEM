from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from .bench_write_keep_read import (
    WriteKeepReadBatch,
    WriteKeepReadConfig,
    compute_retrieval_accuracy,
    generate_write_keep_read_batch,
)
from .model import MemoryChannelsModel, Params


Array = jax.Array


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 200
    learning_rate: float = 3e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    keep_write_loss_weight: float = 1e-3
    keep_mix_loss_weight: float = 1e-4
    memory_delta_loss_weight: float = 1e-4
    keep_identity_loss_weight: float = 1e-4
    memory_subspace_loss_weight: float = 0.0
    memory_leakage_loss_weight: float = 0.0
    memory_subspace_rank: int = 4
    eval_distances: tuple[int, ...] = (4, 8, 16, 32)
    eval_batches: int = 8

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive.")
        if self.eval_batches < 1:
            raise ValueError("eval_batches must be positive.")


@dataclass(frozen=True)
class AdamState:
    step: int
    first_moment: Params
    second_moment: Params


@dataclass(frozen=True)
class TrainResult:
    params: Params
    history: dict[str, list[float]]
    eval_accuracy_by_distance: dict[int, float]
    optimizer_state: AdamState


def _zeros_like_tree(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def init_adam_state(params: Params) -> AdamState:
    zeros = _zeros_like_tree(params)
    return AdamState(step=0, first_moment=zeros, second_moment=zeros)


def retrieval_loss(
    logits: Array,
    readout_positions: Array,
    targets: Array,
) -> Array:
    readout_logits = logits[jnp.arange(logits.shape[0]), readout_positions]
    log_probs = jax.nn.log_softmax(readout_logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(log_probs.shape[0]), targets])


def batch_loss(
    model: MemoryChannelsModel,
    params: Params,
    batch: WriteKeepReadBatch,
) -> tuple[Array, Array]:
    logits = model.logits(params, batch.tokens)
    loss = retrieval_loss(logits, batch.readout_positions, batch.targets)
    accuracy = compute_retrieval_accuracy(logits, batch.readout_positions, batch.targets)
    return loss, accuracy


def _adam_update(
    params: Params,
    grads: Params,
    state: AdamState,
    config: TrainConfig,
) -> tuple[Params, AdamState]:
    step = state.step + 1
    first_moment = jax.tree_util.tree_map(
        lambda m, g: config.beta1 * m + (1.0 - config.beta1) * g,
        state.first_moment,
        grads,
    )
    second_moment = jax.tree_util.tree_map(
        lambda v, g: config.beta2 * v + (1.0 - config.beta2) * jnp.square(g),
        state.second_moment,
        grads,
    )
    bias_correction1 = 1.0 - (config.beta1**step)
    bias_correction2 = 1.0 - (config.beta2**step)

    def update_param(param, grad, m, v):
        m_hat = m / bias_correction1
        v_hat = v / bias_correction2
        update = m_hat / (jnp.sqrt(v_hat) + config.eps)
        if config.weight_decay:
            update = update + config.weight_decay * param
        return param - config.learning_rate * update

    new_params = jax.tree_util.tree_map(update_param, params, grads, first_moment, second_moment)
    return new_params, AdamState(
        step=step,
        first_moment=first_moment,
        second_moment=second_moment,
    )


def evaluate_write_keep_read_model(
    model: MemoryChannelsModel,
    params: Params,
    task_config: WriteKeepReadConfig,
    distances: tuple[int, ...],
    *,
    key: Array,
    num_batches: int = 8,
) -> dict[int, float]:
    results: dict[int, float] = {}
    for distance in distances:
        distance_config = WriteKeepReadConfig(
            vocab_size=task_config.vocab_size,
            num_pairs=task_config.num_pairs,
            batch_size=task_config.batch_size,
            distance=distance,
            num_distractors=task_config.num_distractors,
            num_memories=task_config.num_memories,
            pad_token_id=task_config.pad_token_id,
            bos_token_id=task_config.bos_token_id,
            query_token_id=task_config.query_token_id,
            distractor_token_low=task_config.distractor_token_low,
        )
        accuracies = []
        for batch_idx in range(num_batches):
            batch_key = jax.random.fold_in(jax.random.fold_in(key, distance), batch_idx)
            batch = generate_write_keep_read_batch(batch_key, distance_config)
            _, accuracy = batch_loss(model, params, batch)
            accuracies.append(float(accuracy))
        results[distance] = sum(accuracies) / len(accuracies)
    return results


def train_write_keep_read_model(
    model: MemoryChannelsModel,
    params: Params,
    task_config: WriteKeepReadConfig,
    train_config: TrainConfig,
    *,
    key: Array,
    optimizer_state: AdamState | None = None,
    history: dict[str, list[float]] | None = None,
    step_callback: Callable[[int, Params, AdamState, dict[str, list[float]]], None] | None = None,
) -> TrainResult:
    optimizer_state = init_adam_state(params) if optimizer_state is None else optimizer_state
    history = (
        {"loss": list(history["loss"]), "accuracy": list(history["accuracy"])}
        if history is not None
        else {"loss": [], "accuracy": []}
    )

    def loss_only(
        current_params: Params,
        tokens: Array,
        readout_positions: Array,
        targets: Array,
    ) -> Array:
        logits, layer_stats = model.forward_with_aux(current_params, tokens)
        task_loss = retrieval_loss(logits, readout_positions, targets)
        reg_loss = jnp.array(0.0, dtype=task_loss.dtype)
        if (
            train_config.keep_write_loss_weight
            or train_config.keep_mix_loss_weight
            or train_config.memory_delta_loss_weight
            or train_config.keep_identity_loss_weight
            or train_config.memory_subspace_loss_weight
            or train_config.memory_leakage_loss_weight
        ):
            include_subspace_terms = bool(
                train_config.memory_subspace_loss_weight
                or train_config.memory_leakage_loss_weight
            )
            reg_terms = model.memory_regularization(
                current_params,
                layer_stats,
                subspace_rank=train_config.memory_subspace_rank,
                include_subspace_terms=include_subspace_terms,
            )
            reg_loss = (
                train_config.keep_write_loss_weight * reg_terms["keep_write"]
                + train_config.keep_mix_loss_weight * reg_terms["keep_mix"]
                + train_config.memory_delta_loss_weight * reg_terms["keep_delta"]
                + train_config.keep_identity_loss_weight * reg_terms["keep_identity"]
                + train_config.memory_subspace_loss_weight * reg_terms["memory_subspace_drift"]
                + train_config.memory_leakage_loss_weight * reg_terms["memory_subspace_leakage"]
            )
        return task_loss + reg_loss

    loss_and_grad = jax.jit(jax.value_and_grad(loss_only, argnums=0, allow_int=True))

    current_params = params
    for step in range(train_config.steps):
        batch_key = jax.random.fold_in(key, step)
        batch = generate_write_keep_read_batch(batch_key, task_config)
        loss, grads = loss_and_grad(
            current_params,
            batch.tokens,
            batch.readout_positions,
            batch.targets,
        )
        current_params, optimizer_state = _adam_update(
            current_params,
            grads,
            optimizer_state,
            train_config,
        )
        logits = model.logits(current_params, batch.tokens)
        accuracy = compute_retrieval_accuracy(logits, batch.readout_positions, batch.targets)
        history["loss"].append(float(loss))
        history["accuracy"].append(float(accuracy))
        if step_callback is not None:
            step_callback(step + 1, current_params, optimizer_state, history)

    eval_key = jax.random.fold_in(key, train_config.steps + 1)
    eval_accuracy = evaluate_write_keep_read_model(
        model,
        current_params,
        task_config,
        train_config.eval_distances,
        key=eval_key,
        num_batches=train_config.eval_batches,
    )
    return TrainResult(
        params=current_params,
        history=history,
        eval_accuracy_by_distance=eval_accuracy,
        optimizer_state=optimizer_state,
    )
