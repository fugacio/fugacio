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

The catch is the ever-present *trivial* solution ``x^I = x^II = z`` (``K = 1``); we
avoid it by seeding the iteration from the unstable trial phase found by the
tangent-plane test in :mod:`fugacio.thermo.stability`. The converged split is
differentiable in temperature, feed, and the model parameters via implicit
differentiation of the fixed point -- so tie-lines move smoothly under a gradient,
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
from fugacio.thermo.equilibrium import rachford_rice
from fugacio.thermo.implicit import fixed_point
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


def _k_from_compositions(model: ActivityModel, t: ArrayLike, x_i: Array, x_ii: Array) -> Array:
    g_i = model.ln_gamma(x_i, t)
    g_ii = model.ln_gamma(x_ii, t)
    return jnp.exp(g_i - g_ii)


def flash_lle(
    model: ActivityModel,
    t: ArrayLike,
    z: Array,
    *,
    k_guess: Array | None = None,
    tol: float = 1e-12,
    max_iter: int = 400,
) -> LLEResult:
    """Split feed ``z`` into two liquids at temperature ``t`` (isoactivity flash).

    Args:
        model: Liquid activity-coefficient model (must admit a miscibility gap).
        t: Temperature (K).
        z: Overall (feed) mole fractions.
        k_guess: Optional initial distribution ratios ``K_i``. If omitted, the
            tangent-plane stability test seeds a non-trivial split automatically.
        tol, max_iter: Fixed-point convergence controls.

    Returns:
        An :class:`LLEResult`. If the feed is actually miscible the iteration
        collapses toward the trivial ``K = 1`` (``psi`` at a bound); call
        :func:`fugacio.thermo.stability.liquid_stability` first to decide whether a
        split exists.
    """
    z = jnp.asarray(z)
    if k_guess is None:
        split = liquid_stability(model, t, z).split
        k0 = split / z
    else:
        k0 = jnp.asarray(k_guess)
    theta = (model, jnp.asarray(t, dtype=float), z)

    def g(ln_k: Array, theta: Any) -> Array:
        model_, t_, z_ = theta
        k = jnp.exp(ln_k)
        psi = rachford_rice(z_, k)
        denom = 1.0 + psi * (k - 1.0)
        x_i = z_ / denom
        x_ii = k * x_i
        x_i = x_i / jnp.sum(x_i)
        x_ii = x_ii / jnp.sum(x_ii)
        return jnp.log(_k_from_compositions(model_, t_, x_i, x_ii))

    ln_k_star = fixed_point(g, jnp.log(k0), theta, tol, max_iter)
    k = jnp.exp(ln_k_star)
    psi = rachford_rice(z, k)
    denom = 1.0 + psi * (k - 1.0)
    x_i = z / denom
    x_ii = k * x_i
    x_i = x_i / jnp.sum(x_i)
    x_ii = x_ii / jnp.sum(x_ii)
    return LLEResult(psi=psi, x_i=x_i, x_ii=x_ii, k=k)


def binary_binodal(
    model: ActivityModel,
    t: ArrayLike,
    *,
    feed: ArrayLike = 0.5,
    tol: float = 1e-12,
    max_iter: int = 400,
) -> tuple[Array, Array]:
    """Mutual-solubility (binodal) compositions of a binary at temperature ``t``.

    Returns ``(x1_phase_I, x1_phase_II)`` -- the mole fraction of component 1 in
    each conjugate liquid (the tie-line ends). Any ``feed`` inside the gap gives the
    same pair; the default 50/50 feed sits squarely in a symmetric gap.
    """
    z = jnp.asarray([feed, 1.0 - feed])
    res = flash_lle(model, t, z, tol=tol, max_iter=max_iter)
    return res.x_i[0], res.x_ii[0]


def tie_line(
    model: ActivityModel,
    t: ArrayLike,
    z: Array,
    *,
    tol: float = 1e-12,
    max_iter: int = 400,
) -> tuple[Array, Array, Array]:
    """One ternary tie-line through feed ``z``: ``(x_raffinate, x_extract, psi)``.

    A thin wrapper over :func:`flash_lle` returning the two conjugate-phase
    compositions and the phase fraction -- the unit of a ternary LLE diagram.
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

    Maps :func:`binary_binodal` across ``temperatures`` and returns
    ``(x1_phase_I, x1_phase_II)`` arrays aligned with the input -- the two branches
    of the solubility envelope that meet at the upper (or lower) critical solution
    temperature.
    """
    temperatures = jnp.asarray(temperatures)

    def one(t: Array) -> tuple[Array, Array]:
        return binary_binodal(model, t, feed=feed, tol=tol, max_iter=max_iter)

    return jax.vmap(one)(temperatures)
