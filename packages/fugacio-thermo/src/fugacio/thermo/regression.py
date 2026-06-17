"""Differentiable parameter estimation for activity-coefficient models.

Because every equilibrium output is differentiable with respect to the
activity-model parameters (see `fugacio.thermo.gammaphi`,
`fugacio.thermo.lle`), fitting a model to data is *plain* gradient-based
optimisation -- no finite-difference parameter sweeps, no black-box derivatives.
This module supplies:

* two self-contained optimisers -- `levenberg_marquardt` for nonlinear
  least squares (exact Gauss-Newton Hessian, adaptive damping) and a simple
  `gradient_descent` -- that operate on an arbitrary parameter *pytree*;
* residual builders that turn experimental data into a residual vector:
  `bubble_pressure_residuals` (isothermal/isobaric P-x-y VLE),
  `activity_residuals` (measured ``ln gamma``), and
  `lle_residuals` (mutual-solubility / tie-line data); and
* convenience fitters (`fit_nrtl_binary`, `fit_uniquac_binary`) that
  wire a model factory to the optimiser and return a ready model object.

A "model factory" is any ``theta -> ActivityModel`` mapping; the optimiser fits
``theta`` (the differentiable leaves you choose to expose), so you control which
parameters are free and which are fixed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from fugacio.thermo.activity.models import NRTL, UNIQUAC, ActivityModel
from fugacio.thermo.data import uniquac_rq
from fugacio.thermo.gammaphi import bubble_pressure_gamma
from fugacio.thermo.groupcontrib.dortmund import modified_unifac_activity
from fugacio.thermo.groupcontrib.unifac import unifac_activity

ArrayLike = Array | float
ResidualFn = Callable[[Any], Array]
ModelFactory = Callable[[Any], ActivityModel]


def levenberg_marquardt(
    residual: ResidualFn,
    theta0: Any,
    *,
    max_iter: int = 100,
    lambda0: float = 1e-2,
    factor: float = 5.0,
    tol: float = 1e-12,
) -> tuple[Any, Array]:
    """Minimise ``0.5 * sum(residual(theta)**2)`` by Levenberg-Marquardt.

    A trust-region blend of Gauss-Newton and gradient descent: each step solves
    ``(J^T J + lambda diag(J^T J)) delta = -J^T r`` with the exact Jacobian ``J``
    (via `jax.jacobian`), shrinking ``lambda`` after an accepted step and
    growing it after a rejected one. Operates on any parameter pytree ``theta0``.

    Returns:
        ``(theta, cost)`` -- the fitted parameter pytree and the final
        half-sum-of-squares cost.
    """
    flat0, unravel = ravel_pytree(theta0)

    # Compile the residual and its Jacobian *once*: both are called repeatedly on
    # identically-shaped inputs, so jitting here turns ``max_iter`` recompilations
    # of a (potentially expensive, implicit-diff) graph into a single compile.
    r_of_vec = jax.jit(lambda v: residual(unravel(v)))
    jac_of_vec = jax.jit(jax.jacobian(lambda v: residual(unravel(v))))

    def cost_of(v: Array) -> float:
        r = r_of_vec(v)
        return 0.5 * float(jnp.sum(r * r))

    v = flat0
    lam = lambda0
    cost = cost_of(v)
    for _ in range(max_iter):
        r = r_of_vec(v)
        jac = jac_of_vec(v)
        g = jac.T @ r
        h = jac.T @ jac
        diag = jnp.diag(jnp.clip(jnp.diag(h), 1e-12, None))
        step = jnp.linalg.solve(h + lam * diag, -g)
        v_new = v + step
        cost_new = cost_of(v_new)
        if cost_new < cost:
            improvement = cost - cost_new
            v, cost = v_new, cost_new
            lam = lam / factor
            if improvement < tol:
                break
        else:
            lam = lam * factor
            if lam > 1e12:
                break
    return unravel(v), jnp.asarray(cost)


def gradient_descent(
    objective: Callable[[Any], Array],
    theta0: Any,
    *,
    learning_rate: float = 1e-2,
    max_iter: int = 500,
) -> tuple[Any, Array]:
    """Minimise a scalar ``objective(theta)`` by fixed-step gradient descent.

    A dependency-free fallback for objectives that are not least-squares; returns
    ``(theta, objective(theta))``.
    """
    flat0, unravel = ravel_pytree(theta0)

    def obj(v: Array) -> Array:
        return objective(unravel(v))

    grad = jax.jit(jax.grad(obj))
    v = flat0
    for _ in range(max_iter):
        v = v - learning_rate * grad(v)
    return unravel(v), obj(v)


def bubble_pressure_residuals(
    make_model: ModelFactory,
    t: Array,
    x: Array,
    p_exp: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    y_exp: Array | None = None,
    p_scale: ArrayLike | None = None,
    y_weight: float = 1.0,
    **opts: Any,
) -> ResidualFn:
    """Residuals of predicted vs. measured bubble pressure (and optionally vapour).

    Args:
        make_model: ``theta -> ActivityModel`` factory.
        t: Temperatures (K), shape ``(m,)``.
        x: Liquid compositions, shape ``(m, n)``.
        p_exp: Measured bubble pressures (Pa), shape ``(m,)``.
        tc: Component critical temperatures (K).
        pc: Component critical pressures (Pa).
        omega: Component acentric factors.
        y_exp: Optional measured vapour compositions, shape ``(m, n)``.
        p_scale: Pressure normaliser (defaults to ``mean(p_exp)``).
        y_weight: Relative weight on the vapour-composition residuals.
        **opts: Forwarded to `bubble_pressure_gamma` (``vapor``, ``poynting``, ...).

    Returns:
        ``residual(theta) -> 1-D array`` for use with `levenberg_marquardt`.
    """
    t = jnp.asarray(t)
    x = jnp.asarray(x)
    p_exp = jnp.asarray(p_exp)
    scale = jnp.mean(p_exp) if p_scale is None else jnp.asarray(p_scale)

    def residual(theta: Any) -> Array:
        model = make_model(theta)

        def one(t_i: Array, x_i: Array) -> tuple[Array, Array]:
            return bubble_pressure_gamma(model, t_i, x_i, tc, pc, omega, **opts)

        p_pred, y_pred = jax.vmap(one)(t, x)
        parts = [(p_pred - p_exp) / scale]
        if y_exp is not None:
            parts.append(y_weight * (y_pred - jnp.asarray(y_exp)).reshape(-1))
        return jnp.concatenate(parts)

    return residual


def activity_residuals(
    make_model: ModelFactory, t: Array, x: Array, ln_gamma_exp: Array
) -> ResidualFn:
    """Residuals of predicted vs. measured log activity coefficients.

    ``t`` is shape ``(m,)``, ``x`` and ``ln_gamma_exp`` are shape ``(m, n)``.
    """
    t = jnp.asarray(t)
    x = jnp.asarray(x)
    ln_gamma_exp = jnp.asarray(ln_gamma_exp)

    def residual(theta: Any) -> Array:
        model = make_model(theta)
        pred = jax.vmap(lambda t_i, x_i: model.ln_gamma(x_i, t_i))(t, x)
        return (pred - ln_gamma_exp).reshape(-1)

    return residual


def lle_residuals(
    make_model: ModelFactory, t: Array, x_i_exp: Array, x_ii_exp: Array
) -> ResidualFn:
    """Isoactivity residuals at measured liquid-liquid tie-line ends.

    A consistent model makes each experimental conjugate pair iso-active:
    ``x_i^I gamma_i^I = x_i^II gamma_i^II``. ``t`` is ``(m,)``; the compositions
    are ``(m, n)``.
    """
    t = jnp.asarray(t)
    x_i_exp = jnp.asarray(x_i_exp)
    x_ii_exp = jnp.asarray(x_ii_exp)

    def residual(theta: Any) -> Array:
        model = make_model(theta)

        def one(t_i: Array, xi: Array, xii: Array) -> Array:
            a_i = xi * jnp.exp(model.ln_gamma(xi, t_i))
            a_ii = xii * jnp.exp(model.ln_gamma(xii, t_i))
            return jnp.log(a_i) - jnp.log(a_ii)

        return jax.vmap(one)(t, x_i_exp, x_ii_exp).reshape(-1)

    return residual


def fit_nrtl_binary(
    t: Array,
    x: Array,
    p_exp: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    alpha: float = 0.3,
    y_exp: Array | None = None,
    b0: tuple[float, float] = (0.0, 0.0),
    max_iter: int = 80,
    **opts: Any,
) -> tuple[NRTL, Array]:
    """Fit binary NRTL ``b`` parameters (fixed ``alpha``) to bubble-point data.

    The free parameters are the two ``1/T`` interaction coefficients
    ``b12``, ``b21`` (Kelvin); ``a = 0`` and the non-randomness ``alpha`` are held
    fixed. Returns ``(fitted NRTL, final cost)``.
    """
    alpha_m = jnp.array([[0.0, alpha], [alpha, 0.0]])

    def make_model(theta: Array) -> NRTL:
        b = jnp.array([[0.0, theta[0]], [theta[1], 0.0]])
        zeros = jnp.zeros((2, 2))
        return NRTL(a=zeros, b=b, alpha=alpha_m, e=zeros)

    residual = bubble_pressure_residuals(
        make_model, t, x, p_exp, tc, pc, omega, y_exp=y_exp, **opts
    )
    theta, cost = levenberg_marquardt(residual, jnp.asarray(b0), max_iter=max_iter)
    return make_model(theta), cost


def unifac_ln_gamma_grid(
    components: list[str],
    t: ArrayLike,
    *,
    points: int = 11,
    dortmund: bool = False,
    x_min: float = 0.02,
) -> tuple[Array, Array, Array]:
    """Sample (modified) UNIFAC ``ln gamma`` on a binary composition/temperature grid.

    This turns a *predictive* group-contribution model into pseudo-data for fitting
    a *correlative* NRTL/UNIQUAC model -- the standard way to obtain binary
    interaction parameters for a pair that has no measured VLE.

    Args:
        components: Exactly two component names with UNIFAC group assignments.
        t: Temperature(s) (K); a scalar or 1-D array. The grid is the outer product
            of the temperatures with the composition samples.
        points: Number of liquid compositions sampled in ``(x_min, 1 - x_min)``.
        dortmund: Use modified UNIFAC (Dortmund) instead of classic UNIFAC.
        x_min: Smallest mole fraction sampled (kept away from the pure limits).

    Returns:
        ``(t_grid, x_grid, ln_gamma_grid)`` with shapes ``(m,)``, ``(m, 2)``,
        ``(m, 2)`` where ``m = points * n_temperatures``.

    Raises:
        ValueError: if ``components`` is not a binary pair.
    """
    if len(components) != 2:
        raise ValueError("UNIFAC->binary-parameter prediction expects exactly 2 components")
    temps = jnp.atleast_1d(jnp.asarray(t, dtype=float))
    x1 = jnp.linspace(x_min, 1.0 - x_min, points)
    activity = modified_unifac_activity if dortmund else unifac_activity
    t_list: list[Array] = []
    x_list: list[Array] = []
    g_list: list[Array] = []
    for t_i in temps:
        for x1_i in x1:
            x_vec = jnp.array([x1_i, 1.0 - x1_i])
            t_list.append(t_i)
            x_list.append(x_vec)
            g_list.append(activity(components, x_vec, t_i))
    return jnp.stack(t_list), jnp.stack(x_list), jnp.stack(g_list)


def predict_nrtl_from_unifac(
    components: list[str],
    t: ArrayLike,
    *,
    alpha: float = 0.3,
    dortmund: bool = False,
    points: int = 11,
    x_min: float = 0.02,
    b0: tuple[float, float] = (0.0, 0.0),
    max_iter: int = 120,
) -> tuple[NRTL, Array]:
    """Predict binary NRTL ``b`` parameters by fitting to UNIFAC activity coefficients.

    UNIFAC supplies ``ln gamma`` over a composition (and temperature) grid; the two
    NRTL ``1/T`` coefficients ``b12``, ``b21`` (with ``a = 0`` and fixed ``alpha``)
    are fitted to it by `levenberg_marquardt`. Use it to bootstrap a
    correlative model for a pair without measured data.

    Returns:
        ``(fitted NRTL, final cost)``.
    """
    t_grid, x_grid, ln_gamma = unifac_ln_gamma_grid(
        components, t, points=points, dortmund=dortmund, x_min=x_min
    )
    alpha_m = jnp.array([[0.0, alpha], [alpha, 0.0]])

    def make_model(theta: Array) -> NRTL:
        b = jnp.array([[0.0, theta[0]], [theta[1], 0.0]])
        zeros = jnp.zeros((2, 2))
        return NRTL(a=zeros, b=b, alpha=alpha_m, e=zeros)

    residual = activity_residuals(make_model, t_grid, x_grid, ln_gamma)
    theta, cost = levenberg_marquardt(residual, jnp.asarray(b0), max_iter=max_iter)
    return make_model(theta), cost


def predict_uniquac_from_unifac(
    components: list[str],
    t: ArrayLike,
    *,
    r: Array | None = None,
    q: Array | None = None,
    dortmund: bool = False,
    points: int = 11,
    x_min: float = 0.02,
    b0: tuple[float, float] = (0.0, 0.0),
    max_iter: int = 120,
) -> tuple[UNIQUAC, Array]:
    """Predict binary UNIQUAC ``b`` parameters by fitting to UNIFAC activity coefficients.

    Like `predict_nrtl_from_unifac`, but for UNIQUAC. The surface/volume
    parameters ``r``, ``q`` default to the curated values
    (`fugacio.thermo.data.uniquac_rq`); the free parameters are the ``1/T``
    coefficients of ``tau = exp(b/T)``.

    Returns:
        ``(fitted UNIQUAC, final cost)``.
    """
    if r is None or q is None:
        r, q = uniquac_rq(components)
    r = jnp.asarray(r)
    q = jnp.asarray(q)
    t_grid, x_grid, ln_gamma = unifac_ln_gamma_grid(
        components, t, points=points, dortmund=dortmund, x_min=x_min
    )

    def make_model(theta: Array) -> UNIQUAC:
        b = jnp.array([[0.0, theta[0]], [theta[1], 0.0]])
        return UNIQUAC(r=r, q=q, a=jnp.zeros((2, 2)), b=b)

    residual = activity_residuals(make_model, t_grid, x_grid, ln_gamma)
    theta, cost = levenberg_marquardt(residual, jnp.asarray(b0), max_iter=max_iter)
    return make_model(theta), cost


def fit_uniquac_binary(
    t: Array,
    x: Array,
    p_exp: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    r: Array,
    q: Array,
    *,
    y_exp: Array | None = None,
    b0: tuple[float, float] = (0.0, 0.0),
    max_iter: int = 80,
    **opts: Any,
) -> tuple[UNIQUAC, Array]:
    """Fit binary UNIQUAC ``b`` parameters (with given ``r``, ``q``) to bubble-point data.

    Free parameters are the ``1/T`` coefficients of ``ln tau = a + b/T`` with
    ``a = 0``. Returns ``(fitted UNIQUAC, final cost)``.
    """
    r = jnp.asarray(r)
    q = jnp.asarray(q)

    def make_model(theta: Array) -> UNIQUAC:
        b = jnp.array([[0.0, theta[0]], [theta[1], 0.0]])
        return UNIQUAC(r=r, q=q, a=jnp.zeros((2, 2)), b=b)

    residual = bubble_pressure_residuals(
        make_model, t, x, p_exp, tc, pc, omega, y_exp=y_exp, **opts
    )
    theta, cost = levenberg_marquardt(residual, jnp.asarray(b0), max_iter=max_iter)
    return make_model(theta), cost
