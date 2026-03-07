# `modulus_memory_channels.dk_predictor`

Source: `src/modulus_memory_channels/dk_predictor.py`

## Module Summary
No module-level documentation provided.

## Public Constants
No public constants detected.

## Public Classes
### `DavisKahanResult`

Signature: `class DavisKahanResult`

No documentation provided.

Methods:
- No public methods detected.

## Public Functions
### `spectral_norm`

Signature: `def spectral_norm(operator: Array) -> Array`

No documentation provided.

### `symmetric_operator_norm`

Signature: `def symmetric_operator_norm(operator: Callable[[Array], Array], dim: int, *, key: Array, num_iters: int = 20) -> Array`

No documentation provided.

### `davis_kahan_from_components`

Signature: `def davis_kahan_from_components(*, perturbation_norm: Array, gap: Array, observed_drift: Array) -> DavisKahanResult`

No documentation provided.

### `davis_kahan_bound`

Signature: `def davis_kahan_bound(operator: Array, perturbed_operator: Array, *, k: int) -> DavisKahanResult`

No documentation provided.
