"""Shortcut distillation design: Fenske-Underwood-Gilliland (FUG).

The FUG method gives the minimum stages at total reflux
(`fenske_min_stages`), the minimum reflux (`underwood_min_reflux`), the actual
stage count at a working reflux (`gilliland_stages`), and the feed-stage
location (`kirkbride_feed_stage`), tied together by `shortcut_column`. It is
cheap, robust, and ideal for screening or for initializing the rigorous MESH
column (`fugacio.sim.distillation.rigorous_column`). Every output is exactly
differentiable with respect to the design variables, so a shortcut design can
be embedded in a gradient-based optimisation alongside the rest of a flowsheet.

Component ordering convention: relative volatilities ``alpha`` are given relative
to a common reference (any component); the *light key* ``lk`` is more volatile
than the *heavy key* ``hk`` (``alpha[lk] > alpha[hk]``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from fugacio.thermo import PropertyPackage
from fugacio.thermo.diagnostics import nan_unless_converged
from fugacio.thermo.implicit import bracketed_root_with_info

ArrayLike = Array | float


def relative_volatility(
    model: PropertyPackage, t: ArrayLike, p: ArrayLike, z: Array, *, ref: int
) -> Array:
    """Relative volatilities ``alpha_i = K_i / K_ref`` at ``(T, P, z)``.

    K-values are the package's ``phi_i^L(z) / phi_i^V(z)`` at the given
    composition, a standard shortcut estimate that is well defined whether or
    not the feed is two-phase. Differentiable in ``(T, P, z)`` and the package
    parameters.
    """
    z = jnp.asarray(z)
    k = model.k_values(t, p, z, z)
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

    Returns ``(R_min, theta)``; both are differentiable in ``(z, x_d, alpha, q)``
    and NaN when no root lies between the key volatilities.
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
    solved = bracketed_root_with_info(
        equation,
        (z, alpha, jnp.asarray(q, dtype=float)),
        lo + 1e-6 * gap,
        hi - 1e-6 * gap,
        tol,
        max_iter,
    )
    theta = nan_unless_converged(solved.value, solved.report)
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
