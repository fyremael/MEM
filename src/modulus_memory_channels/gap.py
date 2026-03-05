from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


Array = jax.Array


@dataclass(frozen=True)
class GapSummary:
    top_k: int
    eigengap: Array
    leading_eigenvalues: Array


def eigengap(eigenvalues: Array, k: int) -> Array:
    if k >= eigenvalues.shape[0]:
        raise ValueError("k must be smaller than the number of eigenvalues.")
    return eigenvalues[k - 1] - eigenvalues[k]


def summarize_gaps(eigenvalues: Array, k: int) -> GapSummary:
    return GapSummary(
        top_k=k,
        eigengap=eigengap(eigenvalues, k),
        leading_eigenvalues=eigenvalues[: k + 1],
    )
