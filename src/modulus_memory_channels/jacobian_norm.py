from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


Array = jax.Array


def _normalize(x: Array, eps: float = 1e-8) -> Array:
    norm = jnp.linalg.norm(x.reshape(-1))
    return x / (norm + eps)


def jacobian_spectral_norm(
    f,
    x: Array,
    *,
    key: Array,
    num_iters: int = 20,
    eps: float = 1e-8,
) -> Array:
    v0 = _normalize(jax.random.normal(key, x.shape, dtype=x.dtype), eps)

    def body(v: Array, _: None) -> tuple[Array, Array]:
        _, jv = jax.jvp(f, (x,), (v,))
        _, vjp_fun = jax.vjp(f, x)
        jt_j_v = vjp_fun(jv)[0]
        return _normalize(jt_j_v, eps), jt_j_v

    v = v0
    jt_j_v = v0
    for _ in range(num_iters):
        v, jt_j_v = body(v, None)
    rayleigh = jnp.vdot(v.reshape(-1), jt_j_v.reshape(-1))
    return jnp.sqrt(jnp.maximum(rayleigh, 0.0))


def batched_jacobian_spectral_norm(
    f,
    xs: Array,
    *,
    key: Array,
    num_iters: int = 20,
) -> Array:
    keys = jax.random.split(key, xs.shape[0])
    vmapped = jax.vmap(
        lambda sample, sample_key: jacobian_spectral_norm(
            f,
            sample,
            key=sample_key,
            num_iters=num_iters,
        )
    )
    return vmapped(xs, keys)
