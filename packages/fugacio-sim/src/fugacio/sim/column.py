"""Distillation columns: shortcut (Fenske-Underwood-Gilliland) and rigorous stages.

Two complementary, fully differentiable column models:

* **Shortcut** -- the Fenske-Underwood-Gilliland (FUG) method: minimum stages at
  total reflux (`fenske_min_stages`), minimum reflux (`underwood_min_reflux`),
  the actual stage count at a working reflux (`gilliland_stages`), and the
  feed-stage location (`kirkbride_feed_stage`), tied together by
  `shortcut_column`. Cheap, robust, and ideal for screening or as an
  initial guess for the rigorous model.
* **Rigorous** -- a multistage equilibrium-stage column solved by the Wang-Henke
  bubble-point method under constant molar overflow, with EOS K-values on every
  stage (`solve_column`). The converged profile is differentiable through
  the fixed-point iteration by implicit differentiation.

Both expose exact gradients of their outputs (stage count, reflux, product
purities) with respect to the design variables, so a column can be embedded in a
gradient-based optimisation alongside the rest of a flowsheet.

Component ordering convention: relative volatilities ``alpha`` are given relative
to a common reference (any component); the *light key* ``lk`` is more volatile
than the *heavy key* ``hk`` (``alpha[lk] > alpha[hk]``).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.flowsheet import tear_solve
from fugacio.sim.properties import _resolve
from fugacio.sim.stream import Stream
from fugacio.thermo import (
    PR,
    CubicEOS,
    ln_phi_mixture,
    mixture_enthalpy,
    molar_enthalpy,
)

ArrayLike = Array | float


@partial(jax.custom_jvp, nondiff_argnums=(0, 4, 5))
def _bracketed_root(
    residual: Callable[[Array, Any], Array],
    params: Any,
    lo: Array,
    hi: Array,
    tol: float,
    max_iter: int,
) -> Array:
    """Find the single root of ``residual(., params)`` in ``[lo, hi]`` by bisection.

    The forward pass uses only residual *values* (robust through poles/kinks at the
    bracket ends); the root is differentiated with respect to ``params`` by the
    implicit function theorem in the ``custom_jvp`` rule (the bracket ``lo, hi`` are
    treated as plain locators and carry no gradient).
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


@_bracketed_root.defjvp
def _bracketed_root_jvp(
    residual: Callable[[Array, Any], Array],
    tol: float,
    max_iter: int,
    primals: tuple[Any, Array, Array],
    tangents: tuple[Any, Array, Array],
) -> tuple[Array, Array]:
    params, lo, hi = primals
    params_dot, _, _ = tangents
    root = _bracketed_root(residual, params, lo, hi, tol, max_iter)
    r_root = jax.grad(lambda tt: residual(tt, params))(root)
    grad_params = jax.grad(lambda pp: residual(root, pp))(params)
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda g, d: jnp.vdot(g, d), grad_params, params_dot)
    )
    r_dot = sum(leaves, jnp.asarray(0.0))
    return root, -r_dot / r_root


def relative_volatility(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    z: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    ref: int,
    kij: Array | None = None,
) -> Array:
    """Relative volatilities ``alpha_i = K_i / K_ref`` at ``(T, P, z)`` from the EOS.

    K-values are evaluated as ``phi_i^L(z) / phi_i^V(z)`` at the given composition,
    a standard shortcut estimate that is well defined whether or not the feed is
    two-phase. Differentiable in ``(T, P, z)``.
    """
    n = z.shape[0]
    kij_arr = jnp.zeros((n, n)) if kij is None else jnp.asarray(kij)
    ln_phi_l, _ = ln_phi_mixture(eos, t, p, z, tc, pc, omega, phase="liquid", kij=kij_arr)
    ln_phi_v, _ = ln_phi_mixture(eos, t, p, z, tc, pc, omega, phase="vapor", kij=kij_arr)
    k = jnp.exp(ln_phi_l - ln_phi_v)
    return k / k[ref]


def fenske_min_stages(d: Array, b: Array, lk: int, hk: int, alpha: Array) -> Array:
    """Fenske minimum number of equilibrium stages at total reflux.

    ``N_min = ln[(d_LK/d_HK)(b_HK/b_LK)] / ln(alpha_LK/alpha_HK)`` where ``d`` and
    ``b`` are the distillate and bottoms component molar flows. Includes the
    reboiler as a stage (the classic Fenske count).
    """
    alpha_lk_hk = alpha[lk] / alpha[hk]
    separation = (d[lk] / d[hk]) * (b[hk] / b[lk])
    return jnp.log(separation) / jnp.log(alpha_lk_hk)


def underwood_min_reflux(
    z: Array,
    x_d: Array,
    alpha: Array,
    q: ArrayLike,
    lk: int,
    hk: int,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> tuple[Array, Array]:
    """Underwood minimum reflux ratio ``R_min`` (constant relative volatility).

    Solves the first Underwood equation ``sum_i alpha_i z_i / (alpha_i - theta) =
    1 - q`` for the common root ``theta`` between ``alpha_HK`` and ``alpha_LK``,
    then evaluates ``R_min + 1 = sum_i alpha_i x_{i,D} / (alpha_i - theta)``.

    Returns ``(R_min, theta)``; both are differentiable in ``(z, x_d, alpha, q)``.
    """
    z = jnp.asarray(z)
    x_d = jnp.asarray(x_d)
    alpha = jnp.asarray(alpha)

    def equation(theta: Array, params: tuple[Array, Array, Array]) -> Array:
        z_, alpha_, q_ = params
        return jnp.sum(alpha_ * z_ / (alpha_ - theta)) - (1.0 - q_)

    lo = alpha[hk]
    hi = alpha[lk]
    gap = hi - lo
    theta = _bracketed_root(
        equation,
        (z, alpha, jnp.asarray(q, dtype=float)),
        lo + 1e-6 * gap,
        hi - 1e-6 * gap,
        tol,
        max_iter,
    )
    r_min = jnp.sum(alpha * x_d / (alpha - theta)) - 1.0
    return r_min, theta


def gilliland_stages(n_min: ArrayLike, r_min: ArrayLike, r: ArrayLike) -> Array:
    """Actual equilibrium-stage count from the Gilliland (Eduljee) correlation.

    With ``X = (R - R_min)/(R + 1)`` and ``Y = 0.75 (1 - X^0.5668)``, the stage
    count follows from ``(N - N_min)/(N + 1) = Y``, i.e. ``N = (N_min + Y)/(1 - Y)``.
    """
    x = (jnp.asarray(r) - jnp.asarray(r_min)) / (jnp.asarray(r) + 1.0)
    y = 0.75 * (1.0 - x**0.5668)
    return (jnp.asarray(n_min) + y) / (1.0 - y)


def kirkbride_feed_stage(
    n: ArrayLike,
    z: Array,
    x_d: Array,
    x_b: Array,
    d_total: ArrayLike,
    b_total: ArrayLike,
    lk: int,
    hk: int,
) -> Array:
    """Kirkbride correlation for the number of stages *above* the feed.

    ``log10(N_R/N_S) = 0.206 log10[(z_HK/z_LK)(x_{LK,B}/x_{HK,D})^2 (B/D)]``;
    returns ``N_R`` given the total stage count ``N = N_R + N_S``.
    """
    flow_ratio = jnp.asarray(b_total) / jnp.asarray(d_total)
    ratio = (z[hk] / z[lk]) * (x_b[lk] / x_d[hk]) ** 2 * flow_ratio
    nr_over_ns = ratio**0.206
    n_s = jnp.asarray(n) / (1.0 + nr_over_ns)
    return jnp.asarray(n) - n_s


class ShortcutResult(NamedTuple):
    """Summary of a Fenske-Underwood-Gilliland shortcut design.

    Attributes:
        n_min: Minimum equilibrium stages at total reflux (Fenske).
        r_min: Minimum reflux ratio (Underwood).
        theta: Underwood common root.
        r: Working reflux ratio used for the Gilliland step.
        n_stages: Actual equilibrium stages (Gilliland).
        feed_stage: Number of stages above the feed (Kirkbride).
    """

    n_min: Array
    r_min: Array
    theta: Array
    r: Array
    n_stages: Array
    feed_stage: Array


def shortcut_column(
    z: Array,
    d: Array,
    b: Array,
    alpha: Array,
    q: ArrayLike,
    lk: int,
    hk: int,
    *,
    reflux: ArrayLike | None = None,
    reflux_factor: ArrayLike = 1.3,
) -> ShortcutResult:
    """Full FUG shortcut design from a feed and a specified product split.

    Args:
        z: Feed mole fractions.
        d: Distillate component molar flows (the chosen split of the feed).
        b: Bottoms component molar flows (``z * F - d`` on a consistent basis).
        alpha: Relative volatilities (to any common reference).
        q: Feed thermal quality (1 = saturated liquid, 0 = saturated vapour).
        lk: Light-key component index.
        hk: Heavy-key component index.
        reflux: Working reflux ratio. If ``None``, ``reflux_factor * R_min`` is used.
        reflux_factor: Multiplier on ``R_min`` when ``reflux`` is not given.

    Returns:
        A `ShortcutResult`; every field is differentiable in the inputs.
    """
    d = jnp.asarray(d)
    b = jnp.asarray(b)
    d_total = jnp.sum(d)
    b_total = jnp.sum(b)
    x_d = d / d_total
    x_b = b / b_total
    n_min = fenske_min_stages(d, b, lk, hk, alpha)
    r_min, theta = underwood_min_reflux(z, x_d, alpha, q, lk, hk)
    r = reflux_factor * r_min if reflux is None else jnp.asarray(reflux)
    n_stages = gilliland_stages(n_min, r_min, r)
    feed_stage = kirkbride_feed_stage(n_stages, z, x_d, x_b, d_total, b_total, lk, hk)
    return ShortcutResult(
        n_min=n_min,
        r_min=r_min,
        theta=theta,
        r=jnp.asarray(r),
        n_stages=n_stages,
        feed_stage=feed_stage,
    )


class ColumnResult(NamedTuple):
    """Converged profile and products of a rigorous equilibrium-stage column.

    Attributes:
        t: Stage temperatures (K), top stage first.
        x: Liquid mole fractions, shape ``(n_stages, n_components)``.
        y: Vapour mole fractions, shape ``(n_stages, n_components)``.
        distillate: Distillate product `Stream`.
        bottoms: Bottoms product `Stream`.
        reflux: Reflux ratio used.
        condenser_duty: Condenser heat removed (W, positive).
        reboiler_duty: Reboiler heat added (W, positive).
    """

    t: Array
    x: Array
    y: Array
    distillate: Stream
    bottoms: Stream
    reflux: Array
    condenser_duty: Array
    reboiler_duty: Array


def solve_column(
    feed: Stream,
    n_stages: int,
    feed_stage: int,
    reflux: ArrayLike,
    distillate_rate: ArrayLike,
    *,
    eos: CubicEOS = PR,
    q: ArrayLike = 1.0,
    kij: Array | None = None,
    t_top: ArrayLike | None = None,
    t_bottom: ArrayLike | None = None,
    t_min: float = 100.0,
    t_max: float = 800.0,
    tol: float = 1e-9,
    max_iter: int = 400,
) -> ColumnResult:
    """Rigorous multistage column by the Wang-Henke bubble-point method (CMO).

    A total condenser sits above stage 1 (distillate and reflux share the stage-1
    vapour composition) and a partial reboiler is stage ``n_stages``; a single feed
    of quality ``q`` enters at ``feed_stage`` (1-indexed from the top). Under
    constant molar overflow the section flows are fixed by ``reflux`` and
    ``distillate_rate``, and each outer sweep (i) solves the tridiagonal component
    balances for the liquid profile with the current EOS K-values, (ii) takes a
    bubble-point Newton step on every stage temperature, and (iii) refreshes the
    vapour compositions. The sweep is iterated to a fixed point by
    `tear_solve`, so the converged profile, products,
    and duties are all differentiable (implicit differentiation) with respect to
    ``reflux``, ``distillate_rate``, the feed, and model parameters.

    Args:
        feed: Feed `Stream` (its temperature sets the
            feed enthalpy used for the duty balance).
        n_stages: Number of equilibrium stages including the reboiler.
        feed_stage: 1-indexed feed stage (``2 <= feed_stage <= n_stages - 1``).
        reflux: Reflux ratio ``L/D``.
        distillate_rate: Distillate molar flow (mol/s); bottoms is the remainder.
        eos: Cubic equation of state (default Peng-Robinson).
        q: Feed thermal quality (1 = saturated liquid).
        kij: Optional binary interaction matrix.
        t_top: Optional initial top-stage temperature for the linear starting
            profile (default ``feed.t`` minus a small spread).
        t_bottom: Optional initial bottom-stage temperature for the linear starting
            profile (default ``feed.t`` plus a small spread).
        t_min: Lower bracket clamp for the per-stage temperature updates.
        t_max: Upper bracket clamp for the per-stage temperature updates.
        tol: Convergence tolerance for the outer fixed point.
        max_iter: Maximum number of outer sweeps.

    Returns:
        A `ColumnResult`.
    """
    components = feed.components
    n = n_stages
    f_idx = feed_stage - 1
    tc, pc, omega, _, cp = _resolve(components)
    n_c = len(components)
    kij_arr = jnp.zeros((n_c, n_c)) if kij is None else jnp.asarray(kij)
    p = feed.p
    z = feed.z
    big_f = feed.total
    feed_comp = jnp.zeros((n, n_c)).at[f_idx].set(big_f * z)
    q_arr = jnp.asarray(q, dtype=float)
    idx = jnp.arange(n)

    def stage_k(t_j: Array, x_j: Array, y_j: Array) -> Array:
        ln_phi_l, _ = ln_phi_mixture(eos, t_j, p, x_j, tc, pc, omega, phase="liquid", kij=kij_arr)
        ln_phi_v, _ = ln_phi_mixture(eos, t_j, p, y_j, tc, pc, omega, phase="vapor", kij=kij_arr)
        return jnp.exp(ln_phi_l - ln_phi_v)

    def k_profile(t: Array, x: Array, y: Array) -> Array:
        return jax.vmap(stage_k)(t, x, y)

    def cmo_flows(r: Array, d: Array) -> tuple[Array, Array]:
        b = big_f - d
        v_rect = (r + 1.0) * d
        v_strip = (r + 1.0) * d - (1.0 - q_arr) * big_f
        l_rect = r * d
        l_strip = r * d + q_arr * big_f
        v = jnp.where(idx + 1 <= feed_stage, v_rect, v_strip)
        liq = jnp.where(idx + 1 < feed_stage, l_rect, jnp.where(idx + 1 < n, l_strip, b))
        return v, liq

    def tridiag_component(k_col: Array, f_col: Array, v: Array, liq: Array, r: Array) -> Array:
        diag = -(1.0 + v * k_col / liq)
        diag = diag.at[0].set(-1.0 - k_col[0] / r)
        sub = jnp.ones(n - 1)
        sup = v[1:] * k_col[1:] / liq[1:]
        mat = jnp.diag(diag) + jnp.diag(sub, -1) + jnp.diag(sup, 1)
        return jnp.linalg.solve(mat, -f_col)

    def sweep(state: tuple[Array, Array, Array], theta: dict[str, Array]) -> tuple[Array, ...]:
        t, x, y = state
        r, d = theta["R"], theta["D"]
        v, liq_flows = cmo_flows(r, d)
        k = k_profile(t, x, y)
        liq = jax.vmap(tridiag_component, in_axes=(1, 1, None, None, None), out_axes=1)(
            k, feed_comp, v, liq_flows, r
        )
        liq = jnp.maximum(liq, 1e-12)
        x_new = liq / jnp.sum(liq, axis=1, keepdims=True)

        def bubble_residual(t_j: Array, x_j: Array, y_j: Array) -> Array:
            return jnp.sum(stage_k(t_j, x_j, y_j) * x_j) - 1.0

        r_bp = jax.vmap(bubble_residual)(t, x_new, y)
        dr_bp = jax.vmap(jax.grad(bubble_residual))(t, x_new, y)
        step = jnp.clip(r_bp / dr_bp, -25.0, 25.0)
        t_new = jnp.clip(t - step, t_min, t_max)
        k_new = k_profile(t_new, x_new, y)
        y_unnorm = k_new * x_new
        y_new = y_unnorm / jnp.sum(y_unnorm, axis=1, keepdims=True)
        return t_new, x_new, y_new

    t_hi = feed.t + 25.0 if t_bottom is None else jnp.asarray(t_bottom)
    t_lo = feed.t - 5.0 if t_top is None else jnp.asarray(t_top)
    t0 = jnp.linspace(t_lo, t_hi, n)
    x0 = jnp.broadcast_to(z, (n, n_c))
    theta = {"R": jnp.asarray(reflux, dtype=float), "D": jnp.asarray(distillate_rate, dtype=float)}
    t_star, x_star, y_star = tear_solve(
        sweep, (t0, x0, x0), theta, q_min=-5.0, q_max=0.0, tol=tol, max_iter=max_iter
    )

    big_d = jnp.asarray(distillate_rate, dtype=float)
    big_b = big_f - big_d
    y_dist = y_star[0]
    x_bot = x_star[-1]
    distillate = Stream(n=big_d * y_dist, t=t_star[0], p=p, components=components)
    bottoms = Stream(n=big_b * x_bot, t=t_star[-1], p=p, components=components)

    r_arr = jnp.asarray(reflux, dtype=float)
    h_vap_top = molar_enthalpy(
        t_star[0], p, y_dist, tc, pc, omega, cp, eos=eos, phase="vapor", kij=kij_arr
    )
    h_liq_dist = molar_enthalpy(
        t_star[0], p, y_dist, tc, pc, omega, cp, eos=eos, phase="liquid", kij=kij_arr
    )
    condenser_duty = (r_arr + 1.0) * big_d * (h_vap_top - h_liq_dist)
    h_feed = mixture_enthalpy(eos, feed.t, p, z, tc, pc, omega, cp, kij=kij_arr)
    h_bottoms = molar_enthalpy(
        t_star[-1], p, x_bot, tc, pc, omega, cp, eos=eos, phase="liquid", kij=kij_arr
    )
    reboiler_duty = big_d * h_liq_dist + big_b * h_bottoms + condenser_duty - big_f * h_feed

    return ColumnResult(
        t=t_star,
        x=x_star,
        y=y_star,
        distillate=distillate,
        bottoms=bottoms,
        reflux=r_arr,
        condenser_duty=condenser_duty,
        reboiler_duty=reboiler_duty,
    )
