# `modulus_memory_channels.subspace`

Source: `src/modulus_memory_channels/subspace.py`

## Module Summary
No module-level documentation provided.

## Public Constants
No public constants detected.

## Public Classes
### `SubspaceResult`

Signature: `class SubspaceResult`

No documentation provided.

Methods:
- No public methods detected.

## Public Functions
### `flatten_samples`

Signature: `def flatten_samples(states: Array) -> Array`

No documentation provided.

### `covariance_operator`

Signature: `def covariance_operator(states: Array, eps: float = 1e-06) -> Array`

No documentation provided.

### `top_eigenspace`

Signature: `def top_eigenspace(operator: Array, k: int) -> SubspaceResult`

No documentation provided.

### `projector_from_basis`

Signature: `def projector_from_basis(basis: Array) -> Array`

No documentation provided.

### `principal_angle_drift`

Signature: `def principal_angle_drift(basis_a: Array, basis_b: Array) -> Array`

No documentation provided.

### `subspace_from_covariance`

Signature: `def subspace_from_covariance(states: Array, k: int) -> SubspaceResult`

No documentation provided.

### `implicit_symmetric_eigenspace`

Signature: `def implicit_symmetric_eigenspace(operator: Callable[[Array], Array], dim: int, k: int, *, key: Array, num_iters: int = 12, oversample: int = 4) -> tuple[Array, Array]`

No documentation provided.

### `sensitivity_operator`

Signature: `def sensitivity_operator(layer_map, states: Array, *, reference_states: Array | None = None, layer_map_factory = None) -> tuple[Callable[[Array], Array], int, tuple[int, ...]]`

No documentation provided.

### `subspace_from_sensitivity`

Signature: `def subspace_from_sensitivity(layer_map, states: Array, k: int, *, key: Array, num_iters: int = 12, oversample: int = 4, reference_states: Array | None = None, layer_map_factory = None) -> SubspaceResult`

No documentation provided.
