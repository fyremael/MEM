from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


Array = jax.Array


@dataclass(frozen=True)
class WriteKeepReadConfig:
    vocab_size: int
    num_pairs: int = 16
    batch_size: int = 8
    distance: int = 16
    num_distractors: int = 8
    num_memories: int = 1
    pad_token_id: int = 0
    bos_token_id: int = 1
    query_token_id: int = 2
    distractor_token_low: int = 3

    def __post_init__(self) -> None:
        minimum_vocab = self.distractor_token_low + (2 * self.num_pairs) + 8
        if self.vocab_size < minimum_vocab:
            raise ValueError(
                "vocab_size is too small for the requested pair inventory and specials."
            )


@dataclass(frozen=True)
class WriteKeepReadBatch:
    tokens: Array
    readout_positions: Array
    targets: Array
    keys: Array
    values: Array
    memory_indices: Array

    @property
    def query_positions(self) -> Array:
        return self.readout_positions


def _key_token(pair_index: Array, config: WriteKeepReadConfig) -> Array:
    return config.distractor_token_low + pair_index


def _value_token(pair_index: Array, config: WriteKeepReadConfig) -> Array:
    return config.distractor_token_low + config.num_pairs + pair_index


def generate_write_keep_read_batch(
    key: Array,
    config: WriteKeepReadConfig,
) -> WriteKeepReadBatch:
    batch_size = config.batch_size
    total_len = (
        1
        + (2 * config.num_memories)
        + config.distance
        + config.num_distractors
        + 2
    )
    tokens = jnp.full((batch_size, total_len), config.pad_token_id, dtype=jnp.int32)
    tokens = tokens.at[:, 0].set(config.bos_token_id)

    pair_keys = jax.random.randint(key, (batch_size, config.num_memories), 0, config.num_pairs)
    key_tokens = _key_token(pair_keys, config)
    value_tokens = _value_token(pair_keys, config)

    for memory_idx in range(config.num_memories):
        base = 1 + (2 * memory_idx)
        tokens = tokens.at[:, base].set(key_tokens[:, memory_idx])
        tokens = tokens.at[:, base + 1].set(value_tokens[:, memory_idx])

    distractor_len = config.distance + config.num_distractors
    distractor_key = jax.random.fold_in(key, 1)
    distractors = jax.random.randint(
        distractor_key,
        (batch_size, distractor_len),
        config.distractor_token_low + (2 * config.num_pairs),
        config.vocab_size,
    )
    distractor_start = 1 + (2 * config.num_memories)
    tokens = tokens.at[:, distractor_start : distractor_start + distractor_len].set(distractors)

    readout_positions = jnp.full((batch_size,), total_len - 1, dtype=jnp.int32)
    query_positions = jnp.full((batch_size,), total_len - 2, dtype=jnp.int32)
    query_index_key = jax.random.fold_in(key, 2)
    query_memory_indices = jax.random.randint(
        query_index_key,
        (batch_size,),
        0,
        config.num_memories,
    )
    query_keys = key_tokens[jnp.arange(batch_size), query_memory_indices]
    targets = value_tokens[jnp.arange(batch_size), query_memory_indices]
    tokens = tokens.at[jnp.arange(batch_size), query_positions].set(config.query_token_id)
    tokens = tokens.at[jnp.arange(batch_size), readout_positions].set(query_keys)

    return WriteKeepReadBatch(
        tokens=tokens,
        readout_positions=readout_positions,
        targets=targets,
        keys=query_keys,
        values=targets,
        memory_indices=query_memory_indices,
    )


def compute_retrieval_accuracy(
    logits: Array,
    query_positions: Array,
    targets: Array,
) -> Array:
    query_logits = logits[jnp.arange(logits.shape[0]), query_positions]
    predictions = jnp.argmax(query_logits, axis=-1)
    return jnp.mean(predictions == targets)
