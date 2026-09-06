"""Implicit differentiation of fixed-point solvers.

Phase-equilibrium calculations are *iterative*: a flash, a bubble point, or a
saturation pressure is the solution of a fixed-point or root-finding loop.
Back-propagating through the individual iterations would be wasteful and
numerically noisy. Instead Fugacio differentiates the *converged solution*
directly, via the implicit function theorem.

For a fixed point ``x* = g(x*, theta)`` the sensitivity to the parameters
``theta`` satisfies::

    (I - dg/dx) dx*/dtheta = dg/dtheta

so a linear solve of the residual Jacobian yields implicit sensitivities
regardless of how many iterations the forward solve took. This is the same trick
used by the cubic-root `fugacio.thermo.eos.compress_factor`, generalized to vector unknowns.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.diagnostics import SolveResult, SolveStatus, residual_report

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
    (the iteration itself is not traced). Prefer `bracketed_root` when a
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


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def implicit_solution(residual: ResidualFn, value: Array, theta: Any, valid: Array) -> Array:
    """Attach an implicit derivative to a detached, independently solved root.

    The initial guess and iteration history carry no derivative. Both forward
    and reverse differentiation solve the linearized residual system. A failed
    primal has a nonfinite sensitivity, preventing optimization from silently
    consuming derivatives at an unconverged iterate.
    """
    return value


@implicit_solution.defjvp
def _implicit_solution_jvp(
    residual: ResidualFn,
    primals: tuple[Array, Any, Array],
    tangents: tuple[Array, Any, Any],
) -> tuple[Array, Array]:
    value, theta, valid = primals
    _, theta_dot, _ = tangents
    root = implicit_solution(residual, value, theta, valid)
    jac = jax.jacrev(lambda x: residual(x, theta))(root)
    _, rhs = jax.jvp(lambda th: residual(root, th), (theta,), (theta_dot,))
    tangent = jnp.linalg.solve(jnp.reshape(jac, (root.size, root.size)), -jnp.ravel(rhs)).reshape(
        root.shape
    )
    # The gate depends only on primals, so this remains linear in tangents and
    # JAX can transpose it to obtain the independently solved adjoint.
    return root, tangent * jnp.where(valid, 1.0, jnp.nan)


def _newton_iterations(
    residual: ResidualFn,
    x0: Array,
    theta: Any,
    tol: float,
    max_iter: int,
    scale: Array,
    residual_scale: Array,
    lower: Array,
    upper: Array,
) -> SolveResult:
    """Scaled, bounded Newton with an actual residual-decreasing line search."""
    alphas = jnp.asarray([1.0, 0.5, 0.25, 0.1, 0.03, 0.01, 0.003, 0.001])

    def f(y: Array) -> Array:
        return residual(y * scale, theta) / residual_scale

    def norm(r: Array) -> Array:
        return jnp.max(jnp.abs(r), initial=0.0)

    y0 = jnp.clip(jnp.asarray(x0, dtype=float), lower, upper) / scale
    r0 = f(y0)

    def cond(carry: tuple[Array, Array, Array, Array, Array]) -> Array:
        _, r, i, _, stalled = carry
        return (norm(r) > tol) & jnp.all(jnp.isfinite(r)) & (i < max_iter) & ~stalled

    def body(carry: tuple[Array, Array, Array, Array, Array]) -> tuple:
        y, r, i, _, _ = carry
        jac = jax.jacrev(f)(y)
        dy = jnp.linalg.solve(jac, -r)

        def regularized(_: None) -> Array:
            # A singular Newton matrix can still have a useful descent direction.
            # This changes the search step, never the equations or their derivative.
            jt = jac.T
            damping = 1e-8 * jnp.maximum(jnp.max(jnp.abs(jt @ jac)), 1.0)
            return jnp.linalg.solve(jt @ jac + damping * jnp.eye(y.size), -(jt @ r))

        dy = jax.lax.cond(jnp.all(jnp.isfinite(dy)), lambda _: dy, regularized, None)

        def trial(alpha: Array) -> tuple[Array, Array, Array]:
            candidate = jnp.clip((y + alpha * dy) * scale, lower, upper) / scale
            r_new = f(candidate)
            merit = jnp.sum(r_new * r_new)
            merit = jnp.where(jnp.all(jnp.isfinite(r_new)), merit, jnp.inf)
            return candidate, r_new, merit

        # Sequential trials keep phase-selective residuals selective and avoid
        # evaluating every expensive flowsheet residual when a full step works.
        candidate, trial_r, merit = trial(alphas[0])

        def search_cond(state: tuple) -> Array:
            index, _, _, score = state
            return (index < alphas.size) & (score > 0.01 * jnp.sum(r * r))

        def search_body(state: tuple) -> tuple:
            index, best_point, best_values, best_score = state
            point, values, score = trial(alphas[index])
            better = score < best_score
            return (
                index + 1,
                jnp.where(better, point, best_point),
                jnp.where(better, values, best_values),
                jnp.minimum(score, best_score),
            )

        _, candidate, trial_r, merit = jax.lax.while_loop(
            search_cond, search_body, (jnp.asarray(1), candidate, trial_r, merit)
        )
        accept = merit < jnp.sum(r * r)
        y_new = jnp.where(accept, candidate, y)
        r_new = jnp.where(accept, trial_r, r)
        step = norm(y_new - y)
        return y_new, r_new, i + 1, step, ~accept

    y, r, iterations, step, stalled = jax.lax.while_loop(
        cond, body, (y0, r0, jnp.asarray(0), jnp.asarray(0.0), jnp.asarray(False))
    )
    report = residual_report(r, tol, iterations=iterations, step_norm=step)
    status = jnp.where(stalled & ~report.converged, SolveStatus.STALLED, report.status)
    return SolveResult(y * scale, report._replace(status=status))


def newton_system_with_info(
    residual: ResidualFn,
    x0: Array,
    theta: Any,
    tol: float = 1e-10,
    max_iter: int = 50,
    *,
    scale: Array | None = None,
    residual_scale: Array | None = None,
    lower: Array | None = None,
    upper: Array | None = None,
) -> SolveResult:
    """Solve a square residual system and report convergence independently.

    Args:
        residual: Vector residual ``F(x, theta)`` with the same shape as ``x``.
        x0: Starting vector; may be a previously converged solution.
        theta: Differentiable parameters. Pass all varying quantities here.
        tol: Maximum scaled residual accepted as converged.
        max_iter: Maximum number of Newton steps.
        scale: Positive characteristic variable magnitudes; defaults to one.
        residual_scale: Positive equation scales; defaults to one.
        lower: Optional lower bounds used only during initialization and search.
        upper: Optional upper bounds used only during initialization and search.

    Returns:
        Best iterate and a :class:`SolveReport`. Bounds and scales are numerical
        aids, not additional equations. Sensitivities are defined only when the
        original residual converges to a locally nonsingular root.

    Raises:
        ValueError: If the tolerance or iteration cap is invalid.
    """
    if tol <= 0 or max_iter < 0:
        raise ValueError("tol must be positive and max_iter must be nonnegative")
    x0 = jnp.asarray(x0, dtype=float)
    ones = jnp.ones_like(x0)
    sc = ones if scale is None else jnp.broadcast_to(jnp.asarray(scale), x0.shape)
    rs = ones if residual_scale is None else jnp.broadcast_to(jnp.asarray(residual_scale), x0.shape)
    lo = -jnp.full_like(x0, jnp.inf) if lower is None else jnp.broadcast_to(lower, x0.shape)
    hi = jnp.full_like(x0, jnp.inf) if upper is None else jnp.broadcast_to(upper, x0.shape)
    args = jax.lax.stop_gradient((x0, theta, sc, rs, lo, hi))
    raw = _newton_iterations(residual, args[0], args[1], tol, max_iter, *args[2:])
    valid_input = jnp.all((sc > 0) & (rs > 0) & jnp.isfinite(sc) & jnp.isfinite(rs) & (lo <= hi))
    report = raw.report._replace(
        status=jnp.where(valid_input, raw.report.status, SolveStatus.INVALID_INPUT)
    )
    report = jax.lax.stop_gradient(report)
    value = implicit_solution(residual, jax.lax.stop_gradient(raw.value), theta, report.converged)
    return SolveResult(value, report)


def newton_system(
    residual: ResidualFn,
    x0: Array,
    theta: Any,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> Array:
    """Solve a vector root with implicit forward and reverse derivatives.

    Returns the best iterate for compatibility. Use :func:`newton_system_with_info`
    when accepting a result; a finite iterate alone does not prove convergence.
    Derivatives of an unconverged solution are nonfinite.
    """
    return newton_system_with_info(residual, x0, theta, tol, max_iter).value


def fixed_point_with_info(
    g: Callable[[Array, Any], Array],
    x0: Array,
    theta: Any,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> SolveResult:
    """Converge a contraction and solve its derivative as a linear system.

    The derivative does not repeat the forward fixed-point iteration, so a
    slowly converging adjoint cannot silently exhaust a separate iteration cap.
    """
    if tol <= 0 or max_iter < 0:
        raise ValueError("tol must be positive and max_iter must be nonnegative")
    params = jax.lax.stop_gradient(theta)
    start = jax.lax.stop_gradient(jnp.asarray(x0, dtype=float))

    def cond(carry: tuple[Array, Array, Array, Array]) -> Array:
        _, r, i, _ = carry
        return (jnp.max(jnp.abs(r)) > tol) & jnp.all(jnp.isfinite(r)) & (i < max_iter)

    def body(carry: tuple[Array, Array, Array, Array]) -> tuple:
        x, r, i, _ = carry
        x_new = x + r
        return x_new, g(x_new, params) - x_new, i + 1, jnp.max(jnp.abs(r))

    x, r, iterations, step = jax.lax.while_loop(
        cond, body, (start, g(start, params) - start, jnp.asarray(0), jnp.asarray(0.0))
    )
    report = residual_report(r, tol, iterations=iterations, step_norm=step)
    value = implicit_solution(
        lambda x, th: g(x, th) - x, jax.lax.stop_gradient(x), theta, report.converged
    )
    return SolveResult(value, report)


def fixed_point(
    g: Callable[[Array, Any], Array],
    x0: Array,
    theta: Any,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Array:
    """Return a contraction's fixed point with implicit forward/reverse derivatives.

    Use :func:`fixed_point_with_info` to inspect termination. An unconverged
    iterate has nonfinite derivatives.
    """
    return fixed_point_with_info(g, x0, theta, tol, max_iter).value


def bracketed_root_with_info(
    residual: ResidualFn,
    params: Any,
    lo: Array,
    hi: Array,
    tol: float = 1e-12,
    max_iter: int = 200,
    *,
    residual_tol: float = 1e-8,
) -> SolveResult:
    """Bisect a validated scalar bracket and check the resulting residual.

    ``tol`` limits bracket width; ``residual_tol`` independently limits the
    function residual in its own units. A sign change across a discontinuity
    can reduce the width without satisfying the equation and is reported as a
    failure. Endpoints that already solve the equation take zero iterations.

    Raises:
        ValueError: If tolerances or the iteration limit are invalid.
    """
    if tol <= 0 or residual_tol <= 0 or max_iter < 0:
        raise ValueError("tolerances must be positive and max_iter nonnegative")
    lower, upper, theta = jax.lax.stop_gradient(
        (jnp.asarray(lo, dtype=float), jnp.asarray(hi, dtype=float), params)
    )
    fl, fu = residual(lower, theta), residual(upper, theta)
    valid = (
        jnp.isfinite(fl)
        & jnp.isfinite(fu)
        & (lower <= upper)
        & ((jnp.sign(fl) != jnp.sign(fu)) | (fl == 0) | (fu == 0))
    )
    endpoint = (jnp.abs(fl) <= residual_tol) | (jnp.abs(fu) <= residual_tol)

    def cond(state: tuple) -> Array:
        left, right, _, _, i = state
        return valid & ~endpoint & ((right - left) > tol) & (i < max_iter)

    def body(state: tuple) -> tuple:
        left, right, fleft, _, i = state
        mid = (left + right) / 2
        fm = residual(mid, theta)
        same = jnp.sign(fm) == jnp.sign(fleft)
        return (
            jnp.where(same, mid, left),
            jnp.where(same, right, mid),
            jnp.where(same, fm, fleft),
            fm,
            i + 1,
        )

    left, right, _, _, iterations = jax.lax.while_loop(
        cond, body, (lower, upper, fl, fu, jnp.asarray(0))
    )
    root = jnp.where(
        endpoint, jnp.where(jnp.abs(fl) <= residual_tol, lower, upper), (left + right) / 2
    )
    report = residual_report(
        jnp.atleast_1d(residual(root, theta)),
        residual_tol,
        iterations=iterations,
        step_norm=right - left,
    )
    report = report._replace(status=jnp.where(valid, report.status, SolveStatus.INVALID_INPUT))
    value = implicit_solution(
        lambda x, th: jnp.atleast_1d(residual(x[0], th)),
        jnp.atleast_1d(root),
        params,
        report.converged,
    )[0]
    return SolveResult(value, report)


def newton_root_with_info(
    residual: ResidualFn,
    params: Any,
    x0: Array,
    tol: float = 1e-10,
    max_iter: int = 100,
    *,
    lower: Array | None = None,
    upper: Array | None = None,
) -> SolveResult:
    """Solve a scalar root with residual-decreasing steps and a checked report."""
    result = newton_system_with_info(
        lambda x, th: jnp.atleast_1d(residual(x[0], th)),
        jnp.atleast_1d(x0),
        params,
        tol,
        max_iter,
        lower=lower,
        upper=upper,
    )
    return SolveResult(result.value[0], result.report)
