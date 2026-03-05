from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from .gap import eigengap
from .subspace import principal_angle_drift, top_eigenspace


Array = jax.Array


@dataclass(frozen=True)
class DavisKahanResult:
    perturbation_norm: Array
    gap: Array
    observed_drift: Array
    predicted_drift: Array
    ratio: Array


def spectral_norm(operator: Array) -> Array:
    singular_values = jnp.linalg.svd(operator, compute_uv=False)
    return singular_values[0]


def symmetric_operator_norm(
    operator: Callable[[Array], Array],
    dim: int,
    *,
    key: Array,
    num_iters: int = 20,
) -> Array:
    vector = jax.random.normal(key, (dim,), dtype=jnp.float32)
    vector = vector / (jnp.linalg.norm(vector) + 1e-8)
    for _ in range(num_iters):
        vector = operator(vector)
        vector = vector / (jnp.linalg.norm(vector) + 1e-8)
    rayleigh = jnp.vdot(vector, operator(vector))
    return jnp.abs(rayleigh)


def davis_kahan_from_components(
    *,
    perturbation_norm: Array,
    gap: Array,
    observed_drift: Array,
) -> DavisKahanResult:
    ratio = perturbation_norm / jnp.maximum(gap, 1e-8)
    predicted = jnp.minimum(1.0, ratio)
    return DavisKahanResult(
        perturbation_norm=perturbation_norm,
        gap=gap,
        observed_drift=observed_drift,
        predicted_drift=predicted,
        ratio=ratio,
    )


def davis_kahan_bound(
    operator: Array,
    perturbed_operator: Array,
    *,
    k: int,
) -> DavisKahanResult:
    base = top_eigenspace(operator, k)
    perturbed = top_eigenspace(perturbed_operator, k)
    gap = eigengap(base.eigenvalues, k)
    perturbation = perturbed_operator - operator
    perturbation_norm = spectral_norm(perturbation)
    observed = principal_angle_drift(base.basis, perturbed.basis)
    return davis_kahan_from_components(
        perturbation_norm=perturbation_norm,
        gap=gap,
        observed_drift=observed,
    )
