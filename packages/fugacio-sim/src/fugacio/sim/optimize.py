"""Differentiable numerical optimization for process design.

Fugacio's whole premise is that a flowsheet is *end-to-end differentiable*, so a
design problem (minimize a cost, maximize a yield, hit a purity at least
operating cost) is a smooth optimization that gradients can solve directly.
This module supplies the optimizers, written against `jax.numpy` so they
compose with the rest of the engine, and (crucially) it differentiates
*through the optimum*: the solution ``x*(theta)`` of a parametric optimization
problem carries exact derivatives with respect to the parameters ``theta`` by the
implicit function theorem applied to the optimality (KKT) conditions, exactly as
`fugacio.thermo.implicit` differentiates a converged flash.

The numeric core operates on a flat parameter vector, but every public entry
point accepts an arbitrary JAX pytree as the decision variable (a dict of
operating conditions, a `Stream`, ...) and flattens it
internally with `jax.flatten_util.ravel_pytree`, so you optimize in the
natural shape of your problem.

Algorithms
----------
* **BFGS** (dense inverse-Hessian quasi-Newton): the robust default for smooth
  unconstrained problems of modest dimension, with an Armijo backtracking line
  search and a curvature-safeguarded update.
* **Gradient descent** with optional momentum and a line search: a simple,
  dependable fallback.
* **Newton**: full-Hessian steps with a line search, for cheap, well-behaved
  Hessians (small design problems).
* **Spectral projected gradient (SPG)**: Barzilai-Borwein steps projected onto
  box bounds with a non-monotone line search, for bound-constrained problems.
* **Augmented Lagrangian**: equality and inequality constraints wrapped around
  any of the inner solvers above (the workhorse for constrained design).
* **Levenberg-Marquardt**: damped Gauss-Newton for nonlinear least squares
  (data fitting, multi-spec reconciliation).

Differentiation
---------------
`argmin` returns just the optimal decision variable and attaches an
implicit-function-theorem ``custom_vjp``: for an unconstrained minimum the
stationarity condition ``grad_x f(x*, theta) = 0`` is differentiated; with box
bounds the active variables are held fixed and the reduced Hessian system is
solved on the free set; with equality constraints the full KKT system is
differentiated. The forward solve (however many iterations it took) never appears
in the backward pass, so a gradient of an optimized design with respect to a
price, a feed spec, or a model parameter costs a single linear solve.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

ArrayLike = Array | float

#: A scalar objective ``f(x, theta) -> ()`` of the decision pytree ``x`` and a
#: differentiable parameter pytree ``theta``.
Objective = Callable[[Any, Any], Array]
#: A vector constraint map ``c(x, theta) -> (m,)`` (``= 0`` for equalities,
#: ``<= 0`` for inequalities).
Constraint = Callable[[Any, Any], Array]
#: A residual map ``r(x, theta) -> (m,)`` for least squares.
Residual = Callable[[Any, Any], Array]

_FlatObjective = Callable[[Array], Array]


class OptimizeResult(NamedTuple):
    """Outcome of an optimization run.

    Attributes:
        x: The optimal decision variable, in the pytree structure of ``x0``.
        fun: Objective value at ``x``.
        grad_norm: Max-norm of the (projected) gradient at ``x``, the
            first-order optimality residual.
        n_iter: Number of outer iterations taken.
        converged: Whether the optimality/feasibility tolerances were met.
        constraint_violation: Max constraint violation (``0`` when unconstrained).
    """

    x: Any
    fun: Array
    grad_norm: Array
    n_iter: Array
    converged: Array
    constraint_violation: Array


# --------------------------------------------------------------------------- #
# Line search
# --------------------------------------------------------------------------- #
def _armijo(
    f: _FlatObjective,
    x: Array,
    p: Array,
    f0: Array,
    g0: Array,
    *,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    shrink: float = 0.5,
    max_ls: int = 40,
) -> tuple[Array, Array]:
    """Backtracking line search satisfying the Armijo sufficient-decrease rule.

    Shrinks the step ``alpha`` from ``alpha0`` by ``shrink`` until
    ``f(x + alpha p) <= f0 + c1 alpha (g0 . p)`` or ``max_ls`` trials elapse.
    Returns ``(alpha, f_new)``. The search uses only function values, so it is
    safe across the kinks an EOS flash can produce; it is never differentiated
    (the optimum is differentiated implicitly).
    """
    slope = jnp.vdot(g0, p)

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        alpha, f_new, i = carry
        armijo = f_new <= f0 + c1 * alpha * slope
        return (~armijo & jnp.isfinite(f0)) & (i < max_ls)

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        alpha, _f_new, i = carry
        alpha_new = alpha * shrink
        return alpha_new, f(x + alpha_new * p), i + 1

    a0 = jnp.asarray(alpha0)
    init = (a0, f(x + a0 * p), jnp.asarray(0))
    alpha, f_new, _ = jax.lax.while_loop(cond, body, init)
    return alpha, f_new


# --------------------------------------------------------------------------- #
# Unconstrained inner solvers (operate on a flat vector)
# --------------------------------------------------------------------------- #
def _bfgs(
    f: _FlatObjective,
    x0: Array,
    *,
    tol: float,
    max_iter: int,
) -> tuple[Array, Array, Array]:
    """Dense BFGS with Armijo line search. Returns ``(x*, n_iter, grad_norm)``.

    Maintains an explicit inverse-Hessian approximation ``H`` and takes the
    quasi-Newton step ``p = -H g``. The rank-two BFGS update is applied only when
    the curvature condition ``s . y > 0`` holds (otherwise ``H`` is kept), which
    keeps ``H`` positive definite without a trust region.
    """
    n = x0.shape[0]
    grad = jax.grad(f)

    def cond(carry: tuple[Array, Array, Array, Array, Array]) -> Array:
        _x, g, _h, i, _ = carry
        return (jnp.max(jnp.abs(g)) > tol) & (i < max_iter)

    def body(
        carry: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        x, g, h, i, _ = carry
        p = -h @ g
        # Guard against a non-descent direction from a stale Hessian estimate.
        p = jnp.where(jnp.vdot(g, p) < 0.0, p, -g)
        f0 = f(x)
        alpha, _ = _armijo(f, x, p, f0, g)
        s = alpha * p
        x_new = x + s
        g_new = grad(x_new)
        y = g_new - g
        sy = jnp.vdot(s, y)
        rho = jnp.where(jnp.abs(sy) > 1e-12, 1.0 / sy, 0.0)
        eye = jnp.eye(n)
        v = eye - rho * jnp.outer(s, y)
        h_new = v @ h @ v.T + rho * jnp.outer(s, s)
        # Only accept the update when curvature is positive (Wolfe-safe).
        h_new = jnp.where(sy > 1e-12, h_new, h)
        return x_new, g_new, h_new, i + 1, jnp.max(jnp.abs(g_new))

    g0 = grad(x0)
    init = (x0, g0, jnp.eye(n), jnp.asarray(0), jnp.max(jnp.abs(g0)))
    x_star, g_star, _, n_iter, _ = jax.lax.while_loop(cond, body, init)
    return x_star, n_iter, jnp.max(jnp.abs(g_star))


def _gradient_descent(
    f: _FlatObjective,
    x0: Array,
    *,
    tol: float,
    max_iter: int,
    momentum: float = 0.9,
) -> tuple[Array, Array, Array]:
    """Momentum gradient descent with a line search. Returns ``(x*, n_iter, grad_norm)``."""
    grad = jax.grad(f)

    def cond(carry: tuple[Array, Array, Array, Array]) -> Array:
        _x, _v, g, i = carry
        return (jnp.max(jnp.abs(g)) > tol) & (i < max_iter)

    def body(
        carry: tuple[Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array]:
        x, v, g, i = carry
        v_new = momentum * v - g
        p = jnp.where(jnp.vdot(g, v_new) < 0.0, v_new, -g)
        alpha, _ = _armijo(f, x, p, f(x), g)
        x_new = x + alpha * p
        return x_new, v_new, grad(x_new), i + 1

    g0 = grad(x0)
    init = (x0, jnp.zeros_like(x0), g0, jnp.asarray(0))
    x_star, _, g_star, n_iter = jax.lax.while_loop(cond, body, init)
    return x_star, n_iter, jnp.max(jnp.abs(g_star))


def _newton_min(
    f: _FlatObjective,
    x0: Array,
    *,
    tol: float,
    max_iter: int,
) -> tuple[Array, Array, Array]:
    """Newton minimization with Hessian regularization and a line search."""
    grad = jax.grad(f)
    hess = jax.hessian(f)
    n = x0.shape[0]

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        _x, g, i = carry
        return (jnp.max(jnp.abs(g)) > tol) & (i < max_iter)

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        x, g, i = carry
        hmat = hess(x)
        # Levenberg-style regularization to keep the step a descent direction.
        lam = 1e-8 * (1.0 + jnp.max(jnp.abs(hmat)))
        p = jnp.linalg.solve(hmat + lam * jnp.eye(n), -g)
        p = jnp.where(jnp.vdot(g, p) < 0.0, p, -g)
        alpha, _ = _armijo(f, x, p, f(x), g)
        x_new = x + alpha * p
        return x_new, grad(x_new), i + 1

    g0 = grad(x0)
    x_star, g_star, n_iter = jax.lax.while_loop(cond, body, (x0, g0, jnp.asarray(0)))
    return x_star, n_iter, jnp.max(jnp.abs(g_star))


def _spg(
    f: _FlatObjective,
    x0: Array,
    lower: Array,
    upper: Array,
    *,
    tol: float,
    max_iter: int,
    mem: int = 10,
) -> tuple[Array, Array, Array]:
    """Spectral projected gradient (Barzilai-Borwein + non-monotone line search).

    Bound-constrained minimization on the box ``[lower, upper]``. The projected
    gradient max-norm is the optimality residual. Robust and matrix-free, so it
    scales to large bound-constrained design vectors.
    """
    grad = jax.grad(f)

    def project(x: Array) -> Array:
        return jnp.clip(x, lower, upper)

    def pg_norm(x: Array, g: Array) -> Array:
        return jnp.max(jnp.abs(project(x - g) - x))

    def cond(carry: tuple[Array, Array, Array, Array, Array]) -> Array:
        x, g, _bb, i, _fhist = carry
        return (pg_norm(x, g) > tol) & (i < max_iter)

    def body(
        carry: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        x, g, bb, i, fhist = carry
        d = project(x - bb * g) - x
        # A non-finite gradient component would make ``x + alpha d`` non-finite
        # for *every* ``alpha``, defeating the line search; freeze those entries.
        d = jnp.where(jnp.isfinite(d), d, 0.0)
        f_ref = jnp.max(fhist)
        slope = jnp.vdot(g, d)

        def ls_cond(c: tuple[Array, Array]) -> Array:
            alpha, fa = c
            # Accept only a *finite* point that meets the non-monotone Armijo
            # rule; a NaN/Inf objective (the EOS enthalpy integral evaluated in a
            # non-physical region the step overshot into) must be rejected, not
            # silently accepted because ``NaN > x`` is ``False``. Keep shrinking
            # toward ``x`` (which is finite) until the step is admissible.
            accept = (fa <= f_ref + 1e-4 * alpha * slope) & jnp.isfinite(fa)
            return (~accept) & (alpha > 1e-10)

        def ls_body(c: tuple[Array, Array]) -> tuple[Array, Array]:
            alpha, _ = c
            a_new = 0.5 * alpha
            return a_new, f(x + a_new * d)

        alpha, _ = jax.lax.while_loop(ls_cond, ls_body, (jnp.asarray(1.0), f(x + d)))
        x_new = x + alpha * d
        g_new = grad(x_new)
        s = x_new - x
        y = g_new - g
        sy = jnp.vdot(s, y)
        ss = jnp.vdot(s, s)
        bb_new = jnp.where(sy > 1e-12, jnp.clip(ss / sy, 1e-10, 1e10), 1.0)
        fhist_new = jnp.concatenate([fhist[1:], f(x_new)[None]])
        return x_new, g_new, bb_new, i + 1, fhist_new

    x_init = project(x0)
    g0 = grad(x_init)
    f0 = f(x_init)
    fhist0 = jnp.full((mem,), f0)
    init = (x_init, g0, jnp.asarray(1.0), jnp.asarray(0), fhist0)
    x_star, g_star, _, n_iter, _ = jax.lax.while_loop(cond, body, init)
    return x_star, n_iter, pg_norm(x_star, g_star)


# --------------------------------------------------------------------------- #
# Constrained: augmented Lagrangian
# --------------------------------------------------------------------------- #
def _auglag(
    f: _FlatObjective,
    x0: Array,
    eq: Callable[[Array], Array] | None,
    ineq: Callable[[Array], Array] | None,
    lower: Array | None,
    upper: Array | None,
    *,
    tol: float,
    max_iter: int,
    inner_iter: int,
    penalty0: float = 10.0,
    penalty_growth: float = 5.0,
) -> tuple[Array, Array, Array, Array, Array]:
    """Bound-aware augmented-Lagrangian solver for equality/inequality constraints.

    Minimizes ``f`` subject to ``eq(x) = 0`` and ``ineq(x) <= 0`` (and optional
    box bounds) by minimizing the augmented Lagrangian

        ``L_A = f + lam_eq . h + (mu/2)||h||^2``
              ``+ (1/2mu) sum[max(0, lam_in + mu g)^2 - lam_in^2]``

    over ``x`` for fixed multipliers, then updating the multipliers and growing
    the penalty ``mu`` until the constraints are satisfied. Inner minimizations
    use SPG when bounds are present, BFGS otherwise. Termination requires both
    feasibility *and* a settled inner solution, so a feasible-but-suboptimal
    start does not stop the iteration prematurely.

    Returns ``(x*, n_outer, grad_norm, constraint_violation, converged)``.
    """
    n_eq = 0 if eq is None else int(eq(x0).shape[0])
    n_in = 0 if ineq is None else int(ineq(x0).shape[0])
    has_bounds = lower is not None and upper is not None

    def violation(x: Array) -> Array:
        v = jnp.asarray(0.0)
        if eq is not None:
            v = jnp.maximum(v, jnp.max(jnp.abs(eq(x))))
        if ineq is not None:
            v = jnp.maximum(v, jnp.max(jnp.maximum(ineq(x), 0.0)))
        return v

    def make_la(lam_eq: Array, lam_in: Array, mu: Array) -> _FlatObjective:
        def la(x: Array) -> Array:
            val = f(x)
            if eq is not None:
                h = eq(x)
                val = val + jnp.vdot(lam_eq, h) + 0.5 * mu * jnp.vdot(h, h)
            if ineq is not None:
                g = ineq(x)
                t = jnp.maximum(0.0, lam_in + mu * g)
                val = val + (0.5 / mu) * jnp.sum(t**2 - lam_in**2)
            return val

        return la

    def inner_solve(la: _FlatObjective, x_start: Array) -> Array:
        if has_bounds:
            assert lower is not None and upper is not None
            x_new, _, _ = _spg(la, x_start, lower, upper, tol=tol, max_iter=inner_iter)
        else:
            x_new, _, _ = _bfgs(la, x_start, tol=tol, max_iter=inner_iter)
        return x_new

    def cond(carry: tuple[Array, Array, Array, Array, Array, Array]) -> Array:
        _x, _le, _li, _mu, i, done = carry
        return (~done) & (i < max_iter)

    def body(
        carry: tuple[Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        x, lam_eq, lam_in, mu, i, _done = carry
        la = make_la(lam_eq, lam_in, mu)
        x_new = inner_solve(la, x)
        lam_eq_new = lam_eq + mu * eq(x_new) if eq is not None else lam_eq
        lam_in_new = jnp.maximum(0.0, lam_in + mu * ineq(x_new)) if ineq is not None else lam_in
        mu_new = jnp.minimum(mu * penalty_growth, 1e12)
        step = jnp.max(jnp.abs(x_new - x))
        done = (violation(x_new) <= tol) & (step <= jnp.sqrt(tol))
        return x_new, lam_eq_new, lam_in_new, mu_new, i + 1, done

    init = (
        x0,
        jnp.zeros((n_eq,)),
        jnp.zeros((n_in,)),
        jnp.asarray(penalty0),
        jnp.asarray(0),
        jnp.asarray(False),
    )
    x_star, _, _, _, n_outer, done = jax.lax.while_loop(cond, body, init)
    grad_norm = jnp.max(jnp.abs(jax.grad(f)(x_star)))
    return x_star, n_outer, grad_norm, violation(x_star), done


def _levenberg_marquardt(
    r: Callable[[Array], Array],
    x0: Array,
    *,
    tol: float,
    max_iter: int,
    lam0: float = 1e-2,
) -> tuple[Array, Array, Array]:
    """Damped Gauss-Newton for ``min ||r(x)||^2``. Returns ``(x*, n_iter, grad_norm)``."""
    n = x0.shape[0]
    jac = jax.jacobian(r)

    def cost(x: Array) -> Array:
        rr = r(x)
        return 0.5 * jnp.vdot(rr, rr)

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        x, lam, i = carry
        g = jac(x).T @ r(x)
        return (jnp.max(jnp.abs(g)) > tol) & (i < max_iter) & (lam < 1e12)

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        x, lam, i = carry
        j = jac(x)
        rr = r(x)
        jtj = j.T @ j
        g = j.T @ rr
        step = jnp.linalg.solve(jtj + lam * jnp.eye(n), -g)
        x_try = x + step
        improved = cost(x_try) < cost(x)
        x_new = jnp.where(improved, x_try, x)
        lam_new = jnp.where(improved, jnp.maximum(lam / 3.0, 1e-12), jnp.minimum(lam * 3.0, 1e12))
        return x_new, lam_new, i + 1

    x_star, _, n_iter = jax.lax.while_loop(cond, body, (x0, jnp.asarray(lam0), jnp.asarray(0)))
    grad_norm = jnp.max(jnp.abs(jac(x_star).T @ r(x_star)))
    return x_star, n_iter, grad_norm


# --------------------------------------------------------------------------- #
# Flat dispatch
# --------------------------------------------------------------------------- #
def _solve_flat(
    f: _FlatObjective,
    x0: Array,
    *,
    method: str,
    lower: Array | None,
    upper: Array | None,
    eq: Callable[[Array], Array] | None,
    ineq: Callable[[Array], Array] | None,
    tol: float,
    max_iter: int,
    inner_iter: int,
) -> tuple[Array, Array, Array, Array, Array]:
    """Run the requested flat solver. Returns ``(x*, n_iter, grad_norm, violation, converged)``."""
    constrained = eq is not None or ineq is not None
    has_bounds = lower is not None and upper is not None
    opt_tol = jnp.sqrt(jnp.asarray(tol))
    if constrained:
        return _auglag(
            f, x0, eq, ineq, lower, upper, tol=tol, max_iter=max_iter, inner_iter=inner_iter
        )
    if has_bounds:
        assert lower is not None and upper is not None
        x_star, n_iter, gnorm = _spg(f, x0, lower, upper, tol=tol, max_iter=max_iter)
        return x_star, n_iter, gnorm, jnp.asarray(0.0), gnorm <= opt_tol
    if method == "bfgs":
        x_star, n_iter, gnorm = _bfgs(f, x0, tol=tol, max_iter=max_iter)
    elif method == "gradient-descent":
        x_star, n_iter, gnorm = _gradient_descent(f, x0, tol=tol, max_iter=max_iter)
    elif method == "newton":
        x_star, n_iter, gnorm = _newton_min(f, x0, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(
            f"unknown method {method!r}; choose 'bfgs', 'gradient-descent', or 'newton'"
        )
    return x_star, n_iter, gnorm, jnp.asarray(0.0), gnorm <= opt_tol


def _bounds_to_flat(
    bounds: tuple[Any, Any] | None,
    unravel: Callable[[Array], Any],
    n: int,
) -> tuple[Array | None, Array | None]:
    """Flatten a ``(lower, upper)`` bound pytree pair to flat vectors (or ``None``)."""
    if bounds is None:
        return None, None
    lo_tree, hi_tree = bounds
    lo = _broadcast_bound(lo_tree, unravel, n, -jnp.inf)
    hi = _broadcast_bound(hi_tree, unravel, n, jnp.inf)
    return lo, hi


def _broadcast_bound(
    bound: Any,
    unravel: Callable[[Array], Any],
    n: int,
    fill: ArrayLike,
) -> Array:
    """Turn a scalar or pytree bound into a flat vector aligned with the decision vector.

    A single scalar broadcasts to every variable; a pytree bound (matching the
    decision structure) is flattened in the same order as the decision vector.
    """
    if bound is None:
        return jnp.full((n,), fill)
    flat, _ = ravel_pytree(bound)
    if flat.shape[0] == 1 and n != 1:
        return jnp.full((n,), flat[0])
    return flat


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def minimize(
    fun: Objective,
    x0: Any,
    theta: Any = None,
    *,
    method: str = "bfgs",
    bounds: tuple[Any, Any] | None = None,
    eq_constraints: Constraint | None = None,
    ineq_constraints: Constraint | None = None,
    tol: float = 1e-6,
    max_iter: int = 200,
    inner_iter: int = 100,
) -> OptimizeResult:
    """Minimize ``fun(x, theta)`` over the decision pytree ``x``.

    Args:
        fun: Scalar objective ``fun(x, theta) -> ()``.
        x0: Initial decision pytree (its structure defines the unknown).
        theta: Optional parameter pytree forwarded to ``fun`` and the constraints.
        method: Unconstrained inner method, one of ``"bfgs"`` (default),
            ``"gradient-descent"``, or ``"newton"``. Ignored when bounds or
            constraints are present (SPG / augmented Lagrangian take over).
        bounds: Optional ``(lower, upper)`` box. Each side may be a scalar or a
            pytree matching ``x0``; ``None`` on a side means unbounded.
        eq_constraints: Optional ``h(x, theta) -> (m,)`` enforced ``= 0``.
        ineq_constraints: Optional ``g(x, theta) -> (k,)`` enforced ``<= 0``.
        tol: First-order optimality / feasibility tolerance.
        max_iter: Outer iteration cap.
        inner_iter: Inner-solve iteration cap (constrained problems only).

    Returns:
        An `OptimizeResult`. For gradients of the *solution* with respect
        to ``theta``, use `argmin`.
    """
    flat0, unravel = ravel_pytree(x0)
    n = flat0.shape[0]
    lower, upper = _bounds_to_flat(bounds, unravel, n)

    def f_flat(x: Array) -> Array:
        return fun(unravel(x), theta)

    eq_flat = (lambda x: eq_constraints(unravel(x), theta)) if eq_constraints else None
    ineq_flat = (lambda x: ineq_constraints(unravel(x), theta)) if ineq_constraints else None

    x_star, n_iter, grad_norm, violation, converged = _solve_flat(
        f_flat,
        flat0,
        method=method,
        lower=lower,
        upper=upper,
        eq=eq_flat,
        ineq=ineq_flat,
        tol=tol,
        max_iter=max_iter,
        inner_iter=inner_iter,
    )
    return OptimizeResult(
        x=unravel(x_star),
        fun=f_flat(x_star),
        grad_norm=grad_norm,
        n_iter=n_iter,
        converged=converged & (violation <= jnp.sqrt(jnp.asarray(tol))),
        constraint_violation=violation,
    )


def argmin(
    fun: Objective,
    x0: Any,
    theta: Any,
    *,
    method: str = "bfgs",
    bounds: tuple[Any, Any] | None = None,
    eq_constraints: Constraint | None = None,
    ineq_constraints: Constraint | None = None,
    tol: float = 1e-7,
    max_iter: int = 200,
    inner_iter: int = 100,
) -> Any:
    """The minimizer ``x*(theta) = argmin_x fun(x, theta)``, differentiable in ``theta``.

    Identical problem setup to `minimize`, but returns only the optimal
    decision pytree and (the point of this function) carries exact gradients
    with respect to ``theta`` by implicit differentiation of the optimality
    conditions. Use it to differentiate an optimized design with respect to
    prices, feed specifications, or thermodynamic-model parameters.

    The implicit rule differentiates: the stationarity condition
    ``grad_x f(x*, theta) = 0`` (unconstrained); the same restricted to the free
    variables, holding active box bounds fixed (bound-constrained); or the full
    KKT system ``[grad_x L; c] = 0`` with active inequalities promoted to
    equalities (constrained). Gradients are independent of the iteration count.
    """
    flat0, unravel = ravel_pytree(x0)
    n = flat0.shape[0]
    lower, upper = _bounds_to_flat(bounds, unravel, n)

    def f_flat(x: Array, th: Any) -> Array:
        return fun(unravel(x), th)

    eq_flat = (lambda x, th: eq_constraints(unravel(x), th)) if eq_constraints else None
    ineq_flat = (lambda x, th: ineq_constraints(unravel(x), th)) if ineq_constraints else None
    free = jnp.ones((n,), dtype=bool) if bounds is None else None

    @jax.custom_vjp
    def solve(th: Any) -> Array:
        x_star, *_ = _solve_flat(
            lambda x: f_flat(x, th),
            flat0,
            method=method,
            lower=lower,
            upper=upper,
            eq=(lambda x: eq_flat(x, th)) if eq_flat else None,
            ineq=(lambda x: ineq_flat(x, th)) if ineq_flat else None,
            tol=tol,
            max_iter=max_iter,
            inner_iter=inner_iter,
        )
        return x_star

    def solve_fwd(th: Any) -> tuple[Array, tuple[Array, Any]]:
        x_star = solve(th)
        return x_star, (x_star, th)

    def solve_bwd(res: tuple[Array, Any], x_bar: Array) -> tuple[Any]:
        x_star, th = res
        return (_argmin_adjoint(f_flat, eq_flat, ineq_flat, lower, upper, free, x_star, th, x_bar),)

    solve.defvjp(solve_fwd, solve_bwd)
    return unravel(solve(theta))


def _free_mask(x: Array, lower: Array | None, upper: Array | None) -> Array:
    """Boolean mask of *free* (inactive-bound) variables at the solution ``x``."""
    n = x.shape[0]
    if lower is None or upper is None:
        return jnp.ones((n,), dtype=bool)
    at_lo = jnp.abs(x - lower) <= 1e-6 * (1.0 + jnp.abs(lower))
    at_hi = jnp.abs(x - upper) <= 1e-6 * (1.0 + jnp.abs(upper))
    return ~(at_lo | at_hi)


def _argmin_adjoint(
    f_flat: Callable[[Array, Any], Array],
    eq_flat: Callable[[Array, Any], Array] | None,
    ineq_flat: Callable[[Array, Any], Array] | None,
    lower: Array | None,
    upper: Array | None,
    free_all: Array | None,
    x_star: Array,
    theta: Any,
    x_bar: Array,
) -> Any:
    """Implicit-function-theorem cotangent ``theta_bar`` of the optimum ``x*(theta)``.

    Solves the adjoint of the optimality conditions in flat space and pushes it
    back to the parameter pytree. The forward iteration never appears here.
    """
    n = x_star.shape[0]

    def grad_x(x: Array, th: Any) -> Array:
        return jax.grad(lambda xx: f_flat(xx, th))(x)

    def constraints(x: Array, th: Any) -> Array:
        parts = []
        if eq_flat is not None:
            parts.append(eq_flat(x, th))
        if ineq_flat is not None:
            g = ineq_flat(x, th)
            active = jax.lax.stop_gradient(g > -1e-6)
            parts.append(jnp.where(active, g, 0.0))
        if not parts:
            return jnp.zeros((0,))
        return jnp.concatenate(parts)

    m = constraints(x_star, theta).shape[0]
    free = free_all if free_all is not None else _free_mask(x_star, lower, upper)
    hxx = jax.hessian(lambda x: f_flat(x, theta))(x_star)

    if m == 0:
        proj = jnp.diag(free.astype(x_star.dtype))
        h_eff = proj @ hxx @ proj + (jnp.eye(n) - proj)
        w = jnp.linalg.solve(h_eff.T, x_bar * free)
        _, vjp_theta = jax.vjp(lambda th: grad_x(x_star, th), theta)
        return vjp_theta(-w)[0]

    jac_c = jax.jacobian(lambda x: constraints(x, theta))(x_star)
    lam = jnp.linalg.lstsq(jac_c.T, -grad_x(x_star, theta))[0]
    # KKT (1,1) block is the Hessian of the Lagrangian (exact for nonlinear c).
    hxx_l = jax.hessian(lambda x: f_flat(x, theta) + jnp.vdot(lam, constraints(x, theta)))(x_star)
    kkt = jnp.block([[hxx_l, jac_c.T], [jac_c, jnp.zeros((m, m))]])
    rhs = jnp.concatenate([x_bar, jnp.zeros((m,))])
    y = jnp.linalg.solve(kkt.T, rhs)

    def kkt_residual(th: Any) -> Array:
        _, vjp_c = jax.vjp(lambda xx: constraints(xx, th), x_star)
        gx = grad_x(x_star, th) + vjp_c(lam)[0]
        return jnp.concatenate([gx, constraints(x_star, th)])

    _, vjp_theta = jax.vjp(kkt_residual, theta)
    return vjp_theta(-y)[0]


def least_squares(
    residual: Residual,
    x0: Any,
    theta: Any = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> OptimizeResult:
    """Solve ``min_x 0.5 ||residual(x, theta)||^2`` by Levenberg-Marquardt.

    A damped Gauss-Newton method for nonlinear least squares: parameter
    reconciliation, fitting a model to several measurements at once, or driving a
    set of design residuals to zero. Returns an `OptimizeResult` whose
    ``fun`` is the half-sum-of-squares.
    """
    flat0, unravel = ravel_pytree(x0)

    def r_flat(x: Array) -> Array:
        return residual(unravel(x), theta)

    x_star, n_iter, grad_norm = _levenberg_marquardt(r_flat, flat0, tol=tol, max_iter=max_iter)
    rr = r_flat(x_star)
    return OptimizeResult(
        x=unravel(x_star),
        fun=0.5 * jnp.vdot(rr, rr),
        grad_norm=grad_norm,
        n_iter=n_iter,
        converged=grad_norm <= jnp.sqrt(jnp.asarray(tol)),
        constraint_violation=jnp.asarray(0.0),
    )
