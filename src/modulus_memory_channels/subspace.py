from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Callable

import jax
import jax.numpy as jnp


Array = jax.Array


@dataclass(frozen=True)
class SubspaceResult:
    operator: Array | None
    eigenvalues: Array
    basis: Array
    operator_type: str
    ambient_shape: tuple[int, ...]


def flatten_samples(states: Array) -> Array:
    return states.reshape(-1, states.shape[-1])


def covariance_operator(states: Array, eps: float = 1e-6) -> Array:
    samples = flatten_samples(states)
    centered = samples - jnp.mean(samples, axis=0, keepdims=True)
    denom = jnp.maximum(centered.shape[0] - 1, 1)
    cov = centered.T @ centered / denom
    return cov + eps * jnp.eye(cov.shape[0], dtype=cov.dtype)


def top_eigenspace(operator: Array, k: int) -> SubspaceResult:
    eigenvalues, eigenvectors = jnp.linalg.eigh(operator)
    order = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    basis = eigenvectors[:, :k]
    return SubspaceResult(
        operator=operator,
        eigenvalues=eigenvalues,
        basis=basis,
        operator_type="covariance",
        ambient_shape=(operator.shape[0],),
    )


def projector_from_basis(basis: Array) -> Array:
    return basis @ basis.T


def principal_angle_drift(basis_a: Array, basis_b: Array) -> Array:
    overlap = basis_a.T @ basis_b
    singular_values = jnp.linalg.svd(overlap, compute_uv=False)
    sigma_min = singular_values[-1]
    return jnp.sqrt(jnp.maximum(1.0 - sigma_min**2, 0.0))


def subspace_from_covariance(states: Array, k: int) -> SubspaceResult:
    return top_eigenspace(covariance_operator(states), k)


def _orthonormalize(matrix: Array) -> Array:
    q, _ = jnp.linalg.qr(matrix, mode="reduced")
    return q


def _apply_operator_to_matrix(
    operator: Callable[[Array], Array],
    matrix: Array,
) -> Array:
    columns = jnp.swapaxes(matrix, 0, 1)
    applied = jax.vmap(operator)(columns)
    return jnp.swapaxes(applied, 0, 1)


def implicit_symmetric_eigenspace(
    operator: Callable[[Array], Array],
    dim: int,
    k: int,
    *,
    key: Array,
    num_iters: int = 12,
    oversample: int = 4,
) -> tuple[Array, Array]:
    sketch_dim = min(dim, max(k + 1, k + oversample))
    basis = _orthonormalize(jax.random.normal(key, (dim, sketch_dim), dtype=jnp.float32))
    for _ in range(num_iters):
        basis = _orthonormalize(_apply_operator_to_matrix(operator, basis))

    projected = _apply_operator_to_matrix(operator, basis)
    small = 0.5 * ((basis.T @ projected) + (projected.T @ basis))
    eigenvalues, eigenvectors = jnp.linalg.eigh(small)
    order = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    lifted_basis = _orthonormalize(basis @ eigenvectors[:, :k])
    return eigenvalues, lifted_basis


def sensitivity_operator(
    layer_map,
    states: Array,
    *,
    reference_states: Array | None = None,
    layer_map_factory=None,
) -> tuple[Callable[[Array], Array], int, tuple[int, ...]]:
    sample_shape = tuple(states.shape[1:])
    flat_dim = prod(sample_shape)

    def operator(flat_vector: Array) -> Array:
        tangent = flat_vector.reshape(sample_shape)

        def per_sample(x: Array, ref_state: Array | None = None) -> Array:
            sample_layer_map = layer_map if layer_map_factory is None else layer_map_factory(ref_state)
            _, jv = jax.jvp(sample_layer_map, (x,), (tangent,))
            _, vjp_fun = jax.vjp(sample_layer_map, x)
            jt_j_v = vjp_fun(jv)[0]
            return jt_j_v.reshape(-1)

        if layer_map_factory is None:
            return jnp.mean(jax.vmap(per_sample)(states), axis=0)
        return jnp.mean(jax.vmap(per_sample)(states, reference_states), axis=0)

    return operator, flat_dim, sample_shape


def subspace_from_sensitivity(
    layer_map,
    states: Array,
    k: int,
    *,
    key: Array,
    num_iters: int = 12,
    oversample: int = 4,
    reference_states: Array | None = None,
    layer_map_factory=None,
) -> SubspaceResult:
    operator, flat_dim, sample_shape = sensitivity_operator(
        layer_map,
        states,
        reference_states=reference_states,
        layer_map_factory=layer_map_factory,
    )
    eigenvalues, basis = implicit_symmetric_eigenspace(
        operator,
        flat_dim,
        k,
        key=key,
        num_iters=num_iters,
        oversample=oversample,
    )
    return SubspaceResult(
        operator=None,
        eigenvalues=eigenvalues,
        basis=basis,
        operator_type="sensitivity",
        ambient_shape=sample_shape,
    )
