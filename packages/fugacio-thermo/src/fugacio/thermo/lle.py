"""Liquid-liquid equilibrium (LLE): two-liquid flash, tie-lines, and binodals.

Partially miscible liquids split into two phases ``I`` and ``II`` whose component
fugacities are equal. With an activity-coefficient model that equality is the
*isoactivity* condition

    x_i^I gamma_i(x^I, T) = x_i^II gamma_i(x^II, T),

i.e. ``K_i = x_i^II / x_i^I = gamma_i^I / gamma_i^II``,
closed by the same Rachford-Rice material balance as a vapour-liquid flash, with
the phase fraction ``psi`` now the mole fraction in phase ``II``:

    sum_i z_i (K_i - 1) / (1 + psi (K_i - 1)) = 0,
    x_i^I = z_i / (1 + psi (K_i - 1)),   x_i^II = K_i x_i^I.

The flash is stability-first. The tangent-plane search of
`fugacio.thermo.stability.liquid_stability` decides whether a split exists: an
established stable liquid is returned as one phase (``psi = 0``), and an unstable
one seeds the isoactivity iteration from its most negative trial phase, which
avoids the ever-present trivial solution ``x^I = x^II = z``. An iteration that
still collapses onto it is reported as ``TRIVIAL``. The converged split is
differentiable in temperature, feed, and the model parameters via implicit
differentiation of the fixed point, so tie-lines move smoothly under a gradient,
which matters for solvent-selection optimisation and parameter fitting to
mutual-solubility data.

Only an activity model can describe an LLE; Wilson's model is excluded by
construction (it has no miscibility gap), so use NRTL, UNIQUAC, or UNIFAC.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.activity.models import ActivityModel
from fugacio.thermo.diagnostics import (
    SolveReport,
    SolveStatus,
    canonical_report,
    nan_unless_converged,
    residual_report,
    with_status,
)
from fugacio.thermo.equilibrium import input_report, rachford_rice
from fugacio.thermo.implicit import fixed_point_with_info
from fugacio.thermo.stability import liquid_stability

ArrayLike = Array | float


class LLEResult(NamedTuple):
    """Result of a two-liquid (LLE) flash.

    Attributes:
        psi: Mole fraction of the feed in liquid phase ``II``.
        x_i: Phase ``I`` (raffinate-like) mole fractions.
        x_ii: Phase ``II`` (extract-like) mole fractions.
        k: Distribution ratios ``K_i = x_i^II / x_i^I`` at the solution.
    """

    psi: Array
    x_i: Array
    x_ii: Array
    k: Array


class LLESolveResult(NamedTuple):
    """Two-liquid flash and its report."""

    value: LLEResult
    report: SolveReport


def _k_from_compositions(model: ActivityModel, t: ArrayLike, x_i: Array, x_ii: Array) -> Array:
    g_i = model.ln_gamma(x_i, t)
    g_ii = model.ln_gamma(x_ii, t)
    return jnp.exp(g_i - g_ii)


def _split(model: ActivityModel, t: Array, z: Array, k: Array) -> LLEResult:
    psi = rachford_rice(z, k)
    denom = 1.0 + psi * (k - 1.0)
    x_i = z / denom
    x_ii = k * x_i
    return LLEResult(psi=psi, x_i=x_i / jnp.sum(x_i), x_ii=x_ii / jnp.sum(x_ii), k=k)


def flash_lle_with_info(
    model: ActivityModel,
    t: ArrayLike,
    z: Array,
    *,
    k_guess: Array | None = None,
    tol: float = 1e-12,
    max_iter: int = 400,
) -> LLESolveResult:
    """Split feed ``z`` into two liquids at temperature ``t`` (isoactivity flash).

    Args:
        model: Liquid activity-coefficient model (must admit a miscibility gap).
        t: Temperature (K).
        z: Overall (feed) mole fractions; absent components stay absent.
        k_guess: Optional initial distribution ratios ``K_i``. If omitted, the
            tangent-plane stability test decides whether a split exists and seeds
            it.
        tol: Fixed-point convergence tolerance.
        max_iter: Maximum number of fixed-point iterations.

    Returns:
        The split and its report. An established stable liquid converges to one
        phase (``psi = 0``, both compositions equal to ``z``); a collapse onto the
        trivial solution from an unstable feed is ``TRIVIAL``.
    """
    z = jnp.asarray(z, dtype=float)
    t = jnp.asarray(t, dtype=float)
    support = z > 0
    if k_guess is None:
        stability = liquid_stability(model, t, z)
        seed = jnp.where(support, stability.trial / jnp.where(support, z, 1.0), 1.0)
        single = stability.stable & stability.converged
    else:
        seed = jnp.asarray(k_guess, dtype=float)
        single = jnp.asarray(False)
    k0 = jnp.clip(seed, 1e-10, 1e10)
    theta = (model, t, z)

    def g(ln_k: Array, theta: Any) -> Array:
        model_, t_, z_ = theta
        state = _split(model_, t_, z_, jnp.exp(ln_k))
        return jnp.log(_k_from_compositions(model_, t_, state.x_i, state.x_ii))

    def one_liquid(_: None) -> LLESolveResult:
        value = LLEResult(psi=jnp.zeros_like(t), x_i=z, x_ii=z, k=jnp.ones_like(z))
        return LLESolveResult(value, canonical_report(residual_report(jnp.zeros(1))))

    def two_liquids(_: None) -> LLESolveResult:
        solved = fixed_point_with_info(g, jnp.log(k0), theta, tol, max_iter)
        k = jnp.exp(solved.value)
        trivial = jnp.max(jnp.where(support, jnp.abs(solved.value), 0.0)) < 1e-6
        report = with_status(solved.report, trivial, SolveStatus.TRIVIAL)
        return LLESolveResult(_split(model, t, z, k), canonical_report(report))

    # Only the branch that applies is differentiated (a discarded failed split
    # would otherwise contribute a NaN derivative).
    result = jax.lax.cond(jax.lax.stop_gradient(single), one_liquid, two_liquids, None)
    report = with_status(result.report, ~input_report(t, 1.0, z), SolveStatus.INVALID_INPUT)
    return LLESolveResult(result.value, report)


def flash_lle(
    model: ActivityModel,
    t: ArrayLike,
    z: Array,
    *,
    k_guess: Array | None = None,
    tol: float = 1e-12,
    max_iter: int = 400,
) -> LLEResult:
    """Two-liquid flash value; NaN when `flash_lle_with_info` reports failure."""
    solved = flash_lle_with_info(model, t, z, k_guess=k_guess, tol=tol, max_iter=max_iter)
    return nan_unless_converged(solved.value, solved.report)


def binary_binodal(
    model: ActivityModel,
    t: ArrayLike,
    *,
    feed: ArrayLike = 0.5,
    tol: float = 1e-12,
    max_iter: int = 400,
) -> tuple[Array, Array]:
    """Mutual-solubility (binodal) compositions of a binary at temperature ``t``.

    Returns ``(x1_phase_I, x1_phase_II)``, the mole fraction of component 1 in
    each conjugate liquid (the tie-line ends). Any ``feed`` inside the gap gives the
    same pair; the default 50/50 feed sits squarely in a symmetric gap. Outside
    the gap (no split) both are NaN.
    """
    z = jnp.asarray([feed, 1.0 - feed])
    solved = flash_lle_with_info(model, t, z, tol=tol, max_iter=max_iter)
    split = solved.report.converged & (solved.value.psi > 0) & (solved.value.psi < 1)
    x1 = jnp.where(split, solved.value.x_i[0], jnp.nan)
    x2 = jnp.where(split, solved.value.x_ii[0], jnp.nan)
    return x1, x2


def tie_line(
    model: ActivityModel,
    t: ArrayLike,
    z: Array,
    *,
    tol: float = 1e-12,
    max_iter: int = 400,
) -> tuple[Array, Array, Array]:
    """One ternary tie-line through feed ``z``: ``(x_raffinate, x_extract, psi)``.

    A thin wrapper over `flash_lle` returning the two conjugate-phase
    compositions and the phase fraction, the unit of a ternary LLE diagram.
    """
    res = flash_lle(model, t, z, tol=tol, max_iter=max_iter)
    return res.x_i, res.x_ii, res.psi


def binodal_curve(
    model: ActivityModel,
    temperatures: Array,
    *,
    feed: ArrayLike = 0.5,
    tol: float = 1e-12,
    max_iter: int = 400,
) -> tuple[Array, Array]:
    """Binary binodal branches over a temperature range.

    Maps `binary_binodal` across ``temperatures`` and returns
    ``(x1_phase_I, x1_phase_II)`` arrays aligned with the input, the two branches
    of the solubility envelope that meet at the upper (or lower) critical solution
    temperature (NaN beyond it).
    """
    temperatures = jnp.asarray(temperatures)

    def one(t: Array) -> tuple[Array, Array]:
        return binary_binodal(model, t, feed=feed, tol=tol, max_iter=max_iter)

    return jax.vmap(one)(temperatures)
