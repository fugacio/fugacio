"""A small, differentiable convex quadratic-program solver for MPC.

Every linear and offset-free MPC step is a convex quadratic program (QP), and a
*differentiable* one is what lets Fugacio do the thing a closed simulator cannot:
take a gradient of a closed-loop performance index straight through the
controller's optimization, so the MPC weights themselves can be tuned by descent.

The solver follows the design that has made `OSQP
<https://osqp.org>`_ the de-facto embedded QP method, specialised here for the
small dense problems an MPC horizon produces:

* **Forward**: the operator-splitting (ADMM) iteration on the canonical form
  ``min 0.5 z^T P z + q^T z  s.t.  l <= A z <= u``. The KKT coefficient matrix is
  constant across iterations, so it is factorised once and the fixed-length march
  is a cheap `jax.lax.scan`. A final *polish* solves the equality-constrained
  KKT system on the identified active set, recovering a solution accurate to
  linear-solve precision (and the constraint multipliers).
* **Backward**: a hand-written ``custom_vjp`` differentiates the *solution*, not
  the iteration, by the implicit function theorem applied to the active-set KKT
  conditions (the OptNet rule). Gradients with respect to every datum ``P, q, A,
  l, u`` cost a single linear solve and are independent of the ADMM iteration
  count, exactly the "differentiate the converged solution" philosophy of
  `fugacio.sim.argmin` and `fugacio.sim.tear_solve`.

`solve_qp` is the ergonomic entry point (objective plus optional equality,
inequality, and box constraints); `solve_qp_canonical` exposes the bare
``l <= A z <= u`` form that the MPC layer builds directly.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import lu_factor, lu_solve

ArrayLike = Array | float


class QPSettings(NamedTuple):
    """Algorithmic settings for the ADMM forward solve (all static, never differentiated).

    Attributes:
        rho: ADMM penalty on the consistency constraint (step size).
        sigma: Tiny proximal regularization that makes the KKT matrix nonsingular
            even when ``P`` is only positive *semi*-definite.
        alpha: Over-relaxation factor in ``(0, 2)`` (1.6 is the OSQP default).
        iters: Number of ADMM iterations (fixed, for a differentiable scan).
        active_tol: Distance below which a constraint counts as active in the polish.
    """

    rho: float = 1.0
    sigma: float = 1e-6
    alpha: float = 1.6
    iters: int = 800
    active_tol: float = 1e-6


_DEFAULT_SETTINGS = QPSettings()


class QPSolution(NamedTuple):
    """Outcome of a QP solve.

    Attributes:
        x: The optimal decision vector ``(n,)``.
        obj: Objective value ``0.5 x^T P x + q^T x`` at ``x``.
        primal_residual: Max constraint violation ``max(l - Ax, Ax - u, 0)`` (inf-norm).
        dual_residual: Max-norm stationarity residual ``P x + q + A^T y``.
        converged: Whether both residuals are below tolerance.
    """

    x: Array
    obj: Array
    primal_residual: Array
    dual_residual: Array
    converged: Array


# --------------------------------------------------------------------------- #
# Active-set KKT solve (used by both the polish and the implicit gradient)
# --------------------------------------------------------------------------- #
def _active_masks(a: Array, x: Array, low: Array, up: Array, tol: float) -> tuple[Array, Array]:
    """Boolean masks of lower- and upper-active constraint rows at ``x`` (no gradient)."""
    con = a @ x
    scale = 1.0 + jnp.abs(con)
    lower = jnp.abs(con - low) <= tol * scale
    upper = jnp.abs(con - up) <= tol * scale
    # An equality row (l == u) is always active; attribute it to the lower side.
    lower = lower | (low == up)
    upper = upper & ~lower
    return jax.lax.stop_gradient(lower), jax.lax.stop_gradient(upper)


def _kkt_matrix(p: Array, a: Array, active: Array) -> Array:
    """Active-set KKT matrix ``[[P, A^T], [diag(act) A, diag(1 - act)]]``.

    Inactive rows are replaced by ``lambda_i = 0`` so the system keeps a fixed
    ``(n + m)`` size regardless of which constraints bind.
    """
    m = a.shape[0]
    act = active.astype(p.dtype)
    top = jnp.concatenate([p, a.T], axis=1)
    bot = jnp.concatenate([act[:, None] * a, jnp.diag(1.0 - act)], axis=1)
    return jnp.concatenate([top, bot], axis=0) if m > 0 else p


def _kkt_rhs(q: Array, b_active: Array, active: Array) -> Array:
    """Right-hand side ``[-q; act * b]`` for the active-set KKT polish solve."""
    if active.shape[0] == 0:
        return -q
    return jnp.concatenate([-q, active.astype(q.dtype) * b_active])


def _active_bound(low: Array, up: Array, lower: Array, upper: Array) -> Array:
    """The bound value targeted by each active row (``l`` if lower-active, ``u`` if upper)."""
    safe_low = jnp.where(jnp.isfinite(low), low, 0.0)
    safe_up = jnp.where(jnp.isfinite(up), up, 0.0)
    return jnp.where(lower, safe_low, jnp.where(upper, safe_up, 0.0))


# --------------------------------------------------------------------------- #
# ADMM forward solve
# --------------------------------------------------------------------------- #
def _osqp(p: Array, q: Array, a: Array, low: Array, up: Array, s: QPSettings) -> Array:
    """ADMM (OSQP) march returning the primal iterate ``x`` (forward solve only).

    The objective is internally normalized by the larger of the mean Hessian
    diagonal and the gradient magnitude, so the fixed penalty ``rho`` is well matched
    across problem scales and the Lagrange multipliers it must build up stay O(1)
    (otherwise a large linear term would need as many iterations as its magnitude to
    enforce the constraints). This only affects the (gradient-invisible) convergence
    path: scaling the objective leaves the minimizer unchanged, and the polish /
    implicit gradient use the original data.
    """
    n = p.shape[0]
    m = a.shape[0]
    rho, sigma, alpha = s.rho, s.sigma, s.alpha
    scale = jnp.maximum(jnp.maximum(jnp.mean(jnp.diag(p)), jnp.max(jnp.abs(q))), 1e-12)
    p = p / scale
    q = q / scale
    kkt = jnp.block([[p + sigma * jnp.eye(n), a.T], [a, -(1.0 / rho) * jnp.eye(m)]])
    lu_piv = lu_factor(kkt)

    def body(
        carry: tuple[Array, Array, Array], _: object
    ) -> tuple[tuple[Array, Array, Array], None]:
        x, z, y = carry
        rhs = jnp.concatenate([sigma * x - q, z - (1.0 / rho) * y])
        sol = lu_solve(lu_piv, rhs)
        x_t = sol[:n]
        nu = sol[n:]
        z_t = z + (1.0 / rho) * (nu - y)
        x_next = alpha * x_t + (1.0 - alpha) * x
        z_relaxed = alpha * z_t + (1.0 - alpha) * z
        z_next = jnp.clip(z_relaxed + (1.0 / rho) * y, low, up)
        y_next = y + rho * (z_relaxed - z_next)
        return (x_next, z_next, y_next), None

    x0 = jnp.zeros((n,))
    z0 = jnp.zeros((m,))
    y0 = jnp.zeros((m,))
    (x_star, _, _), _ = jax.lax.scan(body, (x0, z0, y0), None, length=s.iters)
    return x_star


def _polish(
    p: Array, q: Array, a: Array, low: Array, up: Array, x_admm: Array, s: QPSettings
) -> tuple[Array, Array, Array, Array]:
    """Recover an accurate ``(x, lambda)`` by solving the active-set KKT system.

    Returns ``(x, lam, lower_active, upper_active)``. The active set is frozen from
    the ADMM iterate, so the polish is a single linear solve. The solve uses a
    least-squares pseudoinverse so a rank-deficient active set (a degenerate
    constraint vertex, where more rows are active than there are variables) yields
    a bounded minimum-norm answer instead of an exploding one; the caller's merit
    guard then discards it in favour of the feasible ADMM iterate when needed.
    """
    m = a.shape[0]
    lower, upper = _active_masks(a, x_admm, low, up, s.active_tol)
    if m == 0:
        x = jnp.linalg.solve(p, -q)
        return x, jnp.zeros((0,)), lower, upper
    active = lower | upper
    b_act = _active_bound(low, up, lower, upper)
    kkt = _kkt_matrix(p, a, active)
    rhs = _kkt_rhs(q, b_act, active)
    sol = jnp.linalg.lstsq(kkt, rhs)[0]
    n = p.shape[0]
    return sol[:n], sol[n:], lower, upper


def _merit(p: Array, q: Array, a: Array, low: Array, up: Array, x: Array) -> tuple[Array, Array]:
    """Scale-free merit ``(normalized objective + heavy infeasibility penalty, violation)``."""
    obj = 0.5 * jnp.vdot(x, p @ x) + jnp.vdot(q, x)
    if a.shape[0] == 0:
        return obj, jnp.asarray(0.0)
    con = a @ x
    viol = jnp.max(jnp.maximum(jnp.maximum(low - con, con - up), 0.0))
    return obj, viol


def _select(
    p: Array, q: Array, a: Array, low: Array, up: Array, x_admm: Array, x_polish: Array
) -> Array:
    """Pick the better of the ADMM iterate and the polished point by merit.

    The polish is accurate when the active set is well determined but can be wildly
    infeasible at a degenerate vertex; the (clamped) ADMM iterate is always close to
    feasible. Comparing a scale-normalized objective plus a large feasibility penalty
    keeps the accurate polish in the common case and falls back safely otherwise.
    """
    obj_a, viol_a = _merit(p, q, a, low, up, x_admm)
    obj_p, viol_p = _merit(p, q, a, low, up, x_polish)
    scale = 1.0 + jnp.abs(obj_a) + jnp.abs(obj_p)
    merit_a = obj_a / scale + 1e6 * viol_a
    merit_p = obj_p / scale + 1e6 * viol_p
    return jnp.where(merit_p <= merit_a, x_polish, x_admm)


# --------------------------------------------------------------------------- #
# Differentiable primal solve (implicit-function-theorem custom_vjp)
# --------------------------------------------------------------------------- #
def _make_solver(s: QPSettings) -> Any:
    """Build a ``custom_vjp`` primal QP solver for fixed (static) settings ``s``."""

    @jax.custom_vjp
    def solve(params: tuple[Array, Array, Array, Array, Array]) -> Array:
        p, q, a, low, up = params
        x_admm = _osqp(p, q, a, low, up, s)
        x_polish, _lam, _lo, _up = _polish(p, q, a, low, up, x_admm, s)
        return _select(p, q, a, low, up, x_admm, x_polish)

    def solve_fwd(
        params: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, tuple[Any, ...]]:
        p, q, a, low, up = params
        x_admm = _osqp(p, q, a, low, up, s)
        x_polish, lam, lower, upper = _polish(p, q, a, low, up, x_admm, s)
        x = _select(p, q, a, low, up, x_admm, x_polish)
        return x, (params, x, lam, lower, upper)

    def solve_bwd(res: tuple[Any, ...], x_bar: Array) -> tuple[tuple[Array, ...]]:
        params, x_star, lam, lower, upper = res
        p, _q, a, _low, _up = params
        m = a.shape[0]
        active = lower | upper

        if m == 0:
            dx = jnp.linalg.solve(p.T, x_bar)

            def residual0(prm: tuple[Array, ...]) -> Array:
                pp, qq, _a, _l, _u = prm
                return pp @ x_star + qq

            _, vjp = jax.vjp(residual0, params)
            return (vjp(-dx)[0],)

        kkt = _kkt_matrix(p, a, active)
        rhs = jnp.concatenate([x_bar, jnp.zeros((m,))])
        y = jnp.linalg.lstsq(kkt.T, rhs)[0]

        def residual(prm: tuple[Array, ...]) -> Array:
            pp, qq, aa, ll, uu = prm
            act = active.astype(pp.dtype)
            b_act = _active_bound(ll, uu, lower, upper)
            stat = pp @ x_star + qq + aa.T @ lam
            cons = act * (aa @ x_star - b_act) + (1.0 - act) * lam
            return jnp.concatenate([stat, cons])

        _, vjp = jax.vjp(residual, params)
        return (vjp(-y)[0],)

    solve.defvjp(solve_fwd, solve_bwd)
    return solve


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _residuals(
    p: Array, q: Array, a: Array, low: Array, up: Array, x: Array
) -> tuple[Array, Array]:
    con = a @ x
    viol = jnp.maximum(jnp.maximum(low - con, con - up), 0.0)
    primal = jnp.max(viol) if con.shape[0] > 0 else jnp.asarray(0.0)
    # Least-squares dual estimate for reporting only.
    grad = p @ x + q
    if a.shape[0] > 0:
        lam = jnp.linalg.lstsq(a.T, -grad)[0]
        dual = jnp.max(jnp.abs(grad + a.T @ lam))
    else:
        dual = jnp.max(jnp.abs(grad))
    return primal, dual


def solve_qp_canonical(
    p: ArrayLike,
    q: ArrayLike,
    a: ArrayLike,
    low: ArrayLike,
    up: ArrayLike,
    *,
    settings: QPSettings = _DEFAULT_SETTINGS,
) -> Array:
    """Differentiable minimizer of ``0.5 z^T P z + q^T z`` s.t. ``low <= A z <= up``.

    The bare canonical form the MPC layer assembles directly. Returns just the
    optimal ``z`` (a la `fugacio.sim.argmin`), carrying exact implicit-diff
    gradients with respect to ``P, q, A, low, up``.
    """
    p = jnp.atleast_2d(jnp.asarray(p, dtype=float))
    q = jnp.atleast_1d(jnp.asarray(q, dtype=float))
    a = jnp.asarray(a, dtype=float).reshape(-1, p.shape[0])
    low = jnp.broadcast_to(jnp.asarray(low, dtype=float), (a.shape[0],))
    up = jnp.broadcast_to(jnp.asarray(up, dtype=float), (a.shape[0],))
    solver = _make_solver(settings)
    return solver((p, q, a, low, up))


def _assemble(
    n: int,
    a_eq: Array | None,
    b_eq: Array | None,
    g_ineq: Array | None,
    h_ineq: Array | None,
    lb: Array | None,
    ub: Array | None,
) -> tuple[Array, Array, Array]:
    """Stack equality / inequality / box constraints into one ``low <= A z <= up`` block."""
    rows: list[Array] = []
    lows: list[Array] = []
    ups: list[Array] = []
    if a_eq is not None and b_eq is not None:
        rows.append(a_eq)
        lows.append(b_eq)
        ups.append(b_eq)
    if g_ineq is not None and h_ineq is not None:
        rows.append(g_ineq)
        lows.append(jnp.full(h_ineq.shape, -jnp.inf))
        ups.append(h_ineq)
    if lb is not None or ub is not None:
        rows.append(jnp.eye(n))
        lows.append(lb if lb is not None else jnp.full((n,), -jnp.inf))
        ups.append(ub if ub is not None else jnp.full((n,), jnp.inf))
    if not rows:
        return jnp.zeros((0, n)), jnp.zeros((0,)), jnp.zeros((0,))
    return jnp.concatenate(rows, axis=0), jnp.concatenate(lows), jnp.concatenate(ups)


def solve_qp(
    p: ArrayLike,
    q: ArrayLike,
    *,
    a_eq: ArrayLike | None = None,
    b_eq: ArrayLike | None = None,
    g_ineq: ArrayLike | None = None,
    h_ineq: ArrayLike | None = None,
    lb: ArrayLike | None = None,
    ub: ArrayLike | None = None,
    settings: QPSettings = _DEFAULT_SETTINGS,
) -> QPSolution:
    """Solve ``min 0.5 z^T P z + q^T z`` with optional equality/inequality/box constraints.

    Args:
        p: Symmetric positive-semidefinite Hessian ``(n, n)``.
        q: Linear term ``(n,)``.
        a_eq: Optional equality-constraint matrix ``A_eq`` in ``A_eq z = b_eq``.
        b_eq: Optional equality-constraint vector ``b_eq``.
        g_ineq: Optional inequality-constraint matrix ``G`` in ``G z <= h``.
        h_ineq: Optional inequality-constraint vector ``h``.
        lb: Optional lower box bound ``lb <= z`` (scalars broadcast).
        ub: Optional upper box bound ``z <= ub`` (scalars broadcast).
        settings: ADMM `QPSettings`.

    Returns:
        A `QPSolution`. The decision vector ``x`` is differentiable with
        respect to every datum (``p, q``, the constraint matrices/vectors) by the
        active-set implicit function theorem.
    """
    p = jnp.atleast_2d(jnp.asarray(p, dtype=float))
    n = p.shape[0]
    q = jnp.broadcast_to(jnp.asarray(q, dtype=float), (n,))
    a_eq_a = None if a_eq is None else jnp.asarray(a_eq, dtype=float).reshape(-1, n)
    b_eq_a = None if b_eq is None else jnp.atleast_1d(jnp.asarray(b_eq, dtype=float))
    g_a = None if g_ineq is None else jnp.asarray(g_ineq, dtype=float).reshape(-1, n)
    h_a = None if h_ineq is None else jnp.atleast_1d(jnp.asarray(h_ineq, dtype=float))
    lb_a = None if lb is None else jnp.broadcast_to(jnp.asarray(lb, dtype=float), (n,))
    ub_a = None if ub is None else jnp.broadcast_to(jnp.asarray(ub, dtype=float), (n,))

    a, low, up = _assemble(n, a_eq_a, b_eq_a, g_a, h_a, lb_a, ub_a)
    if a.shape[0] == 0:
        x = jnp.linalg.solve(p, -q)
    else:
        x = solve_qp_canonical(p, q, a, low, up, settings=settings)
    obj = 0.5 * jnp.vdot(x, p @ x) + jnp.vdot(q, x)
    primal, dual = _residuals(p, q, a, low, up, x)
    converged = (primal <= 1e-5) & (dual <= 1e-4)
    return QPSolution(x=x, obj=obj, primal_residual=primal, dual_residual=dual, converged=converged)


__all__ = [
    "QPSettings",
    "QPSolution",
    "solve_qp",
    "solve_qp_canonical",
]
