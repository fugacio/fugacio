"""Implicit differentiation of fixed-point solvers.

Phase-equilibrium calculations are *iterative*: a flash, a bubble point, or a
saturation pressure is the solution of a fixed-point or root-finding loop.
Back-propagating through the individual iterations would be wasteful and
numerically noisy. Instead Fugacio differentiates the *converged solution*
directly, via the implicit function theorem.

For a fixed point ``x* = g(x*, theta)`` the sensitivity to the parameters
``theta`` satisfies::

    (I - dg/dx) dx*/dtheta = dg/dtheta

so a single linear solve (here a contraction iteration that reuses ``g``'s own
vector-Jacobian product) yields exact gradients regardless of how many
iterations the forward solve took. This is the same trick used by the cubic-root
:func:`fugacio.thermo.eos.compress_factor`, generalized to vector unknowns.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def fixed_point(
    g: Callable[[Array, Any], Array],
    x0: Array,
    theta: Any,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Array:
    """Solve ``x = g(x, theta)`` and return the fixed point ``x*``.

    Args:
        g: Update map ``g(x, theta) -> x`` (must be a contraction near ``x*``).
        x0: Initial guess.
        theta: Differentiable parameter pytree passed through to ``g``.
        tol: Convergence tolerance on the max-norm of successive iterates.
        max_iter: Iteration cap.

    Returns:
        The converged fixed point. Gradients with respect to ``theta`` are
        computed by implicit differentiation (see module docstring).
    """

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        x_prev, x, i = carry
        return (jnp.max(jnp.abs(x - x_prev)) > tol) & (i < max_iter)

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        _, x, i = carry
        return x, g(x, theta), i + 1

    x1 = g(x0, theta)
    init = (x0, x1, jnp.asarray(1))
    _, x_star, _ = jax.lax.while_loop(cond, body, init)
    return x_star


def _fixed_point_fwd(
    g: Callable[[Array, Any], Array],
    x0: Array,
    theta: Any,
    tol: float,
    max_iter: int,
) -> tuple[Array, tuple[Array, Any]]:
    x_star = fixed_point(g, x0, theta, tol, max_iter)
    return x_star, (x_star, theta)


def _fixed_point_bwd(
    g: Callable[[Array, Any], Array],
    tol: float,
    max_iter: int,
    res: tuple[Array, Any],
    x_bar: Array,
) -> tuple[Array, Any]:
    x_star, theta = res
    _, vjp_x = jax.vjp(lambda x: g(x, theta), x_star)

    def w_cond(carry: tuple[Array, Array, Array]) -> Array:
        w_prev, w, i = carry
        return (jnp.max(jnp.abs(w - w_prev)) > tol) & (i < max_iter)

    def w_body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        _, w, i = carry
        return w, x_bar + vjp_x(w)[0], i + 1

    w1 = x_bar + vjp_x(x_bar)[0]
    _, w_star, _ = jax.lax.while_loop(w_cond, w_body, (x_bar, w1, jnp.asarray(1)))

    _, vjp_theta = jax.vjp(lambda th: g(x_star, th), theta)
    theta_bar = vjp_theta(w_star)[0]
    return jnp.zeros_like(x_star), theta_bar


fixed_point.defvjp(_fixed_point_fwd, _fixed_point_bwd)
