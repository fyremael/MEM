# `modulus_memory_channels.model`

Source: `src/modulus_memory_channels/model.py`

## Module Summary
No module-level documentation provided.

## Public Constants
No public constants detected.

## Public Classes
### `MemoryChannelsModel`

Signature: `class MemoryChannelsModel`

No documentation provided.

Methods:
- `def init(self, key: Array) -> Params`: No documentation provided.
- `def state_dim(self) -> int`: No documentation provided.
- `def embed_tokens(self, params: Params, tokens: Array) -> Array`: No documentation provided.
- `def initial_state(self, params: Params, tokens: Array) -> Array`: No documentation provided.
- `def view_state(self, state: Array, state_view: str) -> Array`: No documentation provided.
- `def layer_role_scales(self, layer_index: int) -> dict[str, float]`: No documentation provided.
- `def block_apply(self, layer: Params, state: Array, *, layer_index: int, attention_mask: Array | None = None, return_stats: bool = False) -> tuple[Array, dict[str, Array]] | Array`: No documentation provided.
- `def memory_regularization(self, params: Params, layer_stats: dict[str, Array], *, subspace_rank: int = 4, include_subspace_terms: bool = True) -> dict[str, Array]`: No documentation provided.
- `def apply_embeddings(self, params: Params, state: Array, *, attention_mask: Array | None = None, return_states: bool = False) -> tuple[Array, list[Array]] | Array`: No documentation provided.
- `def forward_with_aux(self, params: Params, tokens: Array, *, attention_mask: Array | None = None) -> tuple[Array, dict[str, Array]]`: No documentation provided.
- `def apply(self, params: Params, tokens: Array, *, attention_mask: Array | None = None, return_states: bool = False) -> tuple[Array, list[Array]] | Array`: No documentation provided.
- `def logits(self, params: Params, tokens: Array, *, attention_mask: Array | None = None) -> Array`: No documentation provided.
- `def layer_map(self, params: Params, layer_index: int, *, attention_mask: Array | None = None, state_view: str = 'full', reference_state: Array | None = None)`: No documentation provided.

## Public Functions
### `rms_norm`

Signature: `def rms_norm(x: Array, eps: float) -> Array`

No documentation provided.

### `init_memory_model_params`

Signature: `def init_memory_model_params(key: Array, config: MemoryModelConfig) -> Params`

No documentation provided.
