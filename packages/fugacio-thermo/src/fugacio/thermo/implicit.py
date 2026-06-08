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

ResidualFn = Callable[[Array, Any], Array]


@partial(jax.custom_jvp, nondiff_argnums=(0, 4, 5))
def bracketed_root(
    residual: ResidualFn,
    params: Any,
    lo: Array,
    hi: Array,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Array:
    """Solve a scalar ``residual(x, params) = 0`` for ``x`` in ``[lo, hi]`` by bisection.

    The forward pass uses only residual *values*, so it is robust through the
    poles and kinks that scalar equilibrium residuals (bubble/dew temperature,
    Underwood roots, saturation lines) routinely exhibit at the bracket ends. The
    root is differentiated with respect to the parameter pytree ``params`` by the
    implicit function theorem in the ``custom_jvp`` rule below; the locators
    ``lo``/``hi`` carry no gradient.

    Args:
        residual: Scalar function ``residual(x, params) -> r`` with a single sign
            change on ``[lo, hi]``.
        params: Differentiable parameter pytree forwarded to ``residual``.
        lo: Lower bracket (``residual`` must straddle zero across ``[lo, hi]``).
        hi: Upper bracket.
        tol: Absolute width of the final bracket.
        max_iter: Bisection iteration cap.

    Returns:
        The bracketed root ``x*``; differentiable with respect to ``params``.
    """

    def cond(carry: tuple[Array, Array, Array, Array]) -> Array:
        lo_, hi_, _flo, i = carry
        return ((hi_ - lo_) > tol) & (i < max_iter)

    def body(carry: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
        lo_, hi_, flo, i = carry
        mid = 0.5 * (lo_ + hi_)
        fmid = residual(mid, params)
        same = jnp.sign(fmid) == jnp.sign(flo)
        lo_new = jnp.where(same, mid, lo_)
        hi_new = jnp.where(same, hi_, mid)
        flo_new = jnp.where(same, fmid, flo)
        return lo_new, hi_new, flo_new, i + 1

    flo0 = residual(lo, params)
    init = (lo, hi, flo0, jnp.asarray(0))
    lo_star, hi_star, _, _ = jax.lax.while_loop(cond, body, init)
    return 0.5 * (lo_star + hi_star)


@bracketed_root.defjvp
def _bracketed_root_jvp(
    residual: ResidualFn,
    tol: float,
    max_iter: int,
    primals: tuple[Any, Array, Array],
    tangents: tuple[Any, Array, Array],
) -> tuple[Array, Array]:
    params, lo, hi = primals
    params_dot, _, _ = tangents
    root = bracketed_root(residual, params, lo, hi, tol, max_iter)
    r_root = jax.grad(lambda xx: residual(xx, params))(root)
    grad_params = jax.grad(lambda pp: residual(root, pp))(params)
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda g, d: jnp.vdot(g, d), grad_params, params_dot)
    )
    r_dot = sum(leaves, jnp.asarray(0.0))
    return root, -r_dot / r_root


@partial(jax.custom_jvp, nondiff_argnums=(0, 3, 4, 5))
def newton_root(
    residual: ResidualFn,
    params: Any,
    x0: Array,
    tol: float = 1e-12,
    max_iter: int = 100,
    damping: float = 1.0,
) -> Array:
    """Solve a scalar ``residual(x, params) = 0`` by a damped Newton iteration.

    The forward Newton step uses the *autodiff* slope ``dr/dx`` and an optional
    ``damping`` (step multiplier in ``(0, 1]``) for stability; the converged root
    is differentiated with respect to ``params`` by the implicit function theorem
    (the iteration itself is not traced). Prefer :func:`bracketed_root` when a
    reliable bracket is available; ``newton_root`` is for smooth residuals where a
    good initial guess is cheap (saturation updates, Poynting corrections).

    Returns:
        The root ``x*``; differentiable with respect to ``params``.
    """

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        _x, i, err = carry
        return (err > tol) & (i < max_iter)

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        x, i, _ = carry
        r, dr = jax.value_and_grad(lambda xx: residual(xx, params))(x)
        dr = jnp.where(jnp.abs(dr) < 1e-30, 1e-30, dr)
        x_new = x - damping * r / dr
        return x_new, i + 1, jnp.abs(x_new - x)

    x_star, _, _ = jax.lax.while_loop(
        cond, body, (jnp.asarray(x0, dtype=float), jnp.asarray(0), jnp.asarray(jnp.inf))
    )
    return x_star


@newton_root.defjvp
def _newton_root_jvp(
    residual: ResidualFn,
    tol: float,
    max_iter: int,
    damping: float,
    primals: tuple[Any, Array],
    tangents: tuple[Any, Array],
) -> tuple[Array, Array]:
    params, x0 = primals
    params_dot, _ = tangents
    x_star = newton_root(residual, params, x0, tol, max_iter, damping)
    r_x = jax.grad(lambda xx: residual(xx, params))(x_star)
    grad_params = jax.grad(lambda pp: residual(x_star, pp))(params)
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda g, d: jnp.vdot(g, d), grad_params, params_dot)
    )
    r_dot = sum(leaves, jnp.asarray(0.0))
    return x_star, -r_dot / r_x


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def newton_system(
    residual: ResidualFn,
    x0: Array,
    theta: Any,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> Array:
    """Solve a *vector* root ``residual(x, theta) = 0`` by a damped Newton iteration.

    The forward pass takes full Newton steps ``dx = -J^{-1} F`` with the autodiff
    Jacobian ``J = dF/dx`` (a dense solve, intended for the small systems that
    multi-reaction equilibrium and multi-phase flashes produce). The converged
    root is differentiated with respect to the parameter pytree ``theta`` by the
    implicit function theorem -- ``dx*/dtheta = -J^{-1} dF/dtheta`` -- so gradients
    are exact and independent of the iteration count.

    Args:
        residual: Vector function ``residual(x, theta) -> r`` with ``r.shape == x.shape``.
        x0: Initial guess (should be interior to any feasible region).
        theta: Differentiable parameter pytree forwarded to ``residual``.
        tol: Convergence tolerance on the max-norm of the Newton step.
        max_iter: Iteration cap.

    Returns:
        The converged root ``x*``; differentiable with respect to ``theta``.
    """

    # Damping candidates for the backtracking line search (full step first).
    alphas = jnp.array([1.0, 0.5, 0.25, 0.1, 0.03, 0.01])

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        _x, i, err = carry
        return (err > tol) & (i < max_iter)

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        x, i, _ = carry
        f = residual(x, theta)
        jac = jax.jacobian(lambda xx: residual(xx, theta))(x)
        dx = jnp.linalg.solve(jac, -f)

        # Backtracking line search: take the damped step that most reduces the
        # residual norm. This keeps the Newton iteration from overshooting into
        # infeasible regions (e.g. negative mole numbers in a log-activity
        # residual) where a full step would diverge.
        def norm_at(alpha: Array) -> Array:
            r = residual(x + alpha * dx, theta)
            return jnp.sqrt(jnp.sum(r * r))

        norms = jax.vmap(norm_at)(alphas)
        norms = jnp.where(jnp.isfinite(norms), norms, jnp.inf)
        step = alphas[jnp.argmin(norms)] * dx
        return x + step, i + 1, jnp.max(jnp.abs(step))

    x_star, _, _ = jax.lax.while_loop(
        cond, body, (jnp.asarray(x0, dtype=float), jnp.asarray(0), jnp.asarray(jnp.inf))
    )
    return x_star


def _newton_system_fwd(
    residual: ResidualFn, x0: Array, theta: Any, tol: float, max_iter: int
) -> tuple[Array, tuple[Array, Any]]:
    x_star = newton_system(residual, x0, theta, tol, max_iter)
    return x_star, (x_star, theta)


def _newton_system_bwd(
    residual: ResidualFn, tol: float, max_iter: int, res: tuple[Array, Any], x_bar: Array
) -> tuple[Array, Any]:
    x_star, theta = res
    jac = jax.jacobian(lambda xx: residual(xx, theta))(x_star)
    w = jnp.linalg.solve(jac.T, x_bar)
    _, vjp_theta = jax.vjp(lambda th: residual(x_star, th), theta)
    theta_bar = vjp_theta(-w)[0]
    return jnp.zeros_like(x_star), theta_bar


newton_system.defvjp(_newton_system_fwd, _newton_system_bwd)


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
