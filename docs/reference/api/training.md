# `modulus_memory_channels.training`

Source: `src/modulus_memory_channels/training.py`

## Module Summary
No module-level documentation provided.

## Public Constants
No public constants detected.

## Public Classes
### `TrainConfig`

Signature: `class TrainConfig`

No documentation provided.

Methods:
- No public methods detected.

### `AdamState`

Signature: `class AdamState`

No documentation provided.

Methods:
- No public methods detected.

### `TrainResult`

Signature: `class TrainResult`

No documentation provided.

Methods:
- No public methods detected.

## Public Functions
### `init_adam_state`

Signature: `def init_adam_state(params: Params) -> AdamState`

No documentation provided.

### `retrieval_loss`

Signature: `def retrieval_loss(logits: Array, readout_positions: Array, targets: Array) -> Array`

No documentation provided.

### `batch_loss`

Signature: `def batch_loss(model: MemoryChannelsModel, params: Params, batch: WriteKeepReadBatch) -> tuple[Array, Array]`

No documentation provided.

### `evaluate_write_keep_read_model`

Signature: `def evaluate_write_keep_read_model(model: MemoryChannelsModel, params: Params, task_config: WriteKeepReadConfig, distances: tuple[int, ...], *, key: Array, num_batches: int = 8) -> dict[int, float]`

No documentation provided.

### `train_write_keep_read_model`

Signature: `def train_write_keep_read_model(model: MemoryChannelsModel, params: Params, task_config: WriteKeepReadConfig, train_config: TrainConfig, *, key: Array, optimizer_state: AdamState | None = None, history: dict[str, list[float]] | None = None, step_callback: Callable[[int, Params, AdamState, dict[str, list[float]]], None] | None = None) -> TrainResult`

No documentation provided.
