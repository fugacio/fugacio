"""Finite-difference gradient checks for the differentiable core.

Fugacio's defining feature is that *everything* is differentiable, so a unique
class of oracle is available for free: compare automatic-differentiation
derivatives against finite differences. These helpers turn that comparison into a
single number a test can assert on, covering the analytic ``custom_jvp`` /
``custom_vjp`` rules in `fugacio.thermo.eos`,
`fugacio.thermo.equilibrium`, and `fugacio.thermo.implicit`.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

ScalarFn = Callable[[Array], Array]
VectorFn = Callable[[Array], Array]


def finite_difference_gradient(f: ScalarFn, x: Array, eps: float = 1e-6) -> Array:
    """Central-difference gradient of a scalar-valued ``f`` at ``x``.

    Works for scalar or 1-D array ``x``; returns an array shaped like ``x``.
    """
    x = jnp.asarray(x, dtype=float)
    if x.ndim == 0:
        return (f(x + eps) - f(x - eps)) / (2.0 * eps)
    eye = jnp.eye(x.shape[0])
    cols = [(f(x + eps * eye[i]) - f(x - eps * eye[i])) / (2.0 * eps) for i in range(x.shape[0])]
    return jnp.stack(cols)


def finite_difference_jacobian(f: VectorFn, x: Array, eps: float = 1e-6) -> Array:
    """Central-difference Jacobian of a vector-valued ``f`` at ``x`` (shape ``(out, in)``)."""
    x = jnp.asarray(x, dtype=float)
    eye = jnp.eye(x.shape[0])
    cols = [(f(x + eps * eye[i]) - f(x - eps * eye[i])) / (2.0 * eps) for i in range(x.shape[0])]
    return jnp.stack(cols, axis=1)


def max_gradient_error(f: ScalarFn, x: Array, eps: float = 1e-6) -> Array:
    """Max absolute difference between ``jax.grad(f)`` and finite differences."""
    ad = jax.grad(f)(jnp.asarray(x, dtype=float))
    fd = finite_difference_gradient(f, x, eps)
    return jnp.max(jnp.abs(ad - fd))


def max_jacobian_error(f: VectorFn, x: Array, eps: float = 1e-6) -> Array:
    """Max absolute difference between ``jax.jacobian(f)`` and finite differences."""
    ad = jax.jacobian(f)(jnp.asarray(x, dtype=float))
    fd = finite_difference_jacobian(f, x, eps)
    return jnp.max(jnp.abs(ad - fd))
