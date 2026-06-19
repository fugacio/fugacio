"""PC-SAFT phase equilibrium: flash, bubble/dew points, saturation, stability.

These routines mirror the cubic-EOS equilibrium layer
(`fugacio.thermo.equilibrium`) one-for-one, swapping the cubic fugacity
coefficient for the PC-SAFT one (`fugacio.thermo.saft.properties`). The
vapour-liquid iterations reuse `fugacio.thermo.implicit.fixed_point`, so the
converged phase split is differentiable, by the implicit function theorem, with
respect to temperature, pressure, composition, *and* the PC-SAFT parameters,
which is what lets `fugacio.thermo.saft.regression` fit a binary ``k_ij`` to VLE
data by gradient descent.

Wilson K-values seed the flashes, so the routines accept the critical constants
``(tc, pc, omega)`` for the seed only; the equilibrium itself is entirely PC-SAFT.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.equilibrium import FlashResult, StabilityResult, rachford_rice, wilson_k
from fugacio.thermo.implicit import fixed_point
from fugacio.thermo.saft.parameters import SaftParameters
from fugacio.thermo.saft.properties import ln_fugacity_coefficients

ArrayLike = Array | float


def flash_pt_saft(
    params: SaftParameters,
    t: ArrayLike,
    p: ArrayLike,
    z: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    tol: float = 1e-12,
    max_iter: int = 300,
) -> FlashResult:
    """Isothermal-isobaric two-phase flash on PC-SAFT by accelerated substitution.

    Solves the equal-fugacity conditions ``phi_i^L x_i = phi_i^V y_i`` with the
    Rachford-Rice material balance, seeded from Wilson K-values. The converged
    ``(beta, x, y)`` is differentiable in ``(T, P, z)`` and the PC-SAFT
    parameters through implicit differentiation of the fixed point.
    """
    z = jnp.asarray(z, dtype=float)
    k0 = wilson_k(t, p, tc, pc, omega)
    theta = (params, jnp.asarray(t, dtype=float), jnp.asarray(p, dtype=float), z)

    def g(ln_k: Array, theta: Any) -> Array:
        params_, t_, p_, z_ = theta
        k = jnp.exp(ln_k)
        beta = rachford_rice(z_, k)
        denom = 1.0 + beta * (k - 1.0)
        x = z_ / denom
        y = k * x
        # Normalise the trial phases before the (composition-sensitive) PC-SAFT
        # density solve. At the interior Rachford-Rice root both phases already
        # sum to one, so this is a no-op there; it only regularises the
        # incipient phase when the feed is single-phase (beta pinned at 0 or 1),
        # where the unnormalised ``z / K`` would otherwise drive the density
        # root - and the K-iteration - to diverge.
        x = x / jnp.sum(x)
        y = y / jnp.sum(y)
        ln_phi_l = ln_fugacity_coefficients(params_, t_, p_, x, phase="liquid")
        ln_phi_v = ln_fugacity_coefficients(params_, t_, p_, y, phase="vapor")
        return ln_phi_l - ln_phi_v

    ln_k_star = fixed_point(g, jnp.log(k0), theta, tol, max_iter)
    k = jnp.exp(ln_k_star)
    beta = rachford_rice(z, k)
    denom = 1.0 + beta * (k - 1.0)
    x = z / denom
    y = k * x
    return FlashResult(beta=beta, x=x, y=y, k=k)


def bubble_pressure_saft(
    params: SaftParameters,
    t: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    tol: float = 1e-12,
    max_iter: int = 300,
) -> tuple[Array, Array]:
    """Bubble-point pressure and incipient vapour ``(P, y)`` at fixed ``T``, ``x``.

    Solved as a coupled fixed point in ``(ln P, y)`` so the result is
    differentiable in temperature, composition, and the PC-SAFT parameters.
    """
    x = jnp.asarray(x, dtype=float)
    k0 = wilson_k(t, jnp.sum(x * pc), tc, pc, omega)
    p0 = jnp.sum(x * pc * jnp.exp(5.373 * (1.0 + omega) * (1.0 - tc / jnp.asarray(t, float))))
    y0 = x * k0 / jnp.sum(x * k0)
    state0 = jnp.concatenate([jnp.log(p0)[None], y0])
    theta = (params, jnp.asarray(t, dtype=float), x)

    def g(state: Array, theta: Any) -> Array:
        params_, t_, x_ = theta
        p = jnp.exp(state[0])
        y = state[1:]
        ln_phi_l = ln_fugacity_coefficients(params_, t_, p, x_, phase="liquid")
        ln_phi_v = ln_fugacity_coefficients(params_, t_, p, y, phase="vapor")
        k = jnp.exp(ln_phi_l - ln_phi_v)
        y_unnorm = k * x_
        s = jnp.sum(y_unnorm)
        return jnp.concatenate([(state[0] + jnp.log(s))[None], y_unnorm / s])

    state = fixed_point(g, state0, theta, tol, max_iter)
    return jnp.exp(state[0]), state[1:]


def dew_pressure_saft(
    params: SaftParameters,
    t: ArrayLike,
    y: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    tol: float = 1e-12,
    max_iter: int = 300,
) -> tuple[Array, Array]:
    """Dew-point pressure and incipient liquid ``(P, x)`` at fixed ``T``, ``y``.

    Differentiable in temperature, composition, and the PC-SAFT parameters.
    """
    y = jnp.asarray(y, dtype=float)
    k0 = wilson_k(t, jnp.sum(y * pc), tc, pc, omega)
    p0 = 1.0 / jnp.sum(
        y / (pc * jnp.exp(5.373 * (1.0 + omega) * (1.0 - tc / jnp.asarray(t, float))))
    )
    x0 = (y / k0) / jnp.sum(y / k0)
    state0 = jnp.concatenate([jnp.log(p0)[None], x0])
    theta = (params, jnp.asarray(t, dtype=float), y)

    def g(state: Array, theta: Any) -> Array:
        params_, t_, y_ = theta
        p = jnp.exp(state[0])
        x = state[1:]
        ln_phi_l = ln_fugacity_coefficients(params_, t_, p, x, phase="liquid")
        ln_phi_v = ln_fugacity_coefficients(params_, t_, p, y_, phase="vapor")
        k = jnp.exp(ln_phi_l - ln_phi_v)
        x_unnorm = y_ / k
        s = jnp.sum(x_unnorm)
        return jnp.concatenate([(state[0] - jnp.log(s))[None], x_unnorm / s])

    state = fixed_point(g, state0, theta, tol, max_iter)
    return jnp.exp(state[0]), state[1:]


def _psat_residual(params: SaftParameters, t: Array, p: Array) -> Array:
    x = jnp.ones(1)
    ln_phi_l = ln_fugacity_coefficients(params, t, p, x, phase="liquid")[0]
    ln_phi_v = ln_fugacity_coefficients(params, t, p, x, phase="vapor")[0]
    return ln_phi_l - ln_phi_v


def psat_saft(
    params: SaftParameters,
    t: ArrayLike,
    p_guess: ArrayLike,
    *,
    tol: float = 1e-11,
    max_iter: int = 100,
) -> Array:
    """Pure-component saturation pressure (Pa) by equifugacity, from a guess ``p_guess``.

    Solves ``ln phi^L(T, P) = ln phi^V(T, P)`` for ``P`` with a Newton iteration in
    ``ln P`` (keeping the pressure positive). ``params`` must hold a *single*
    component. Differentiable in ``T`` and the PC-SAFT parameters through the
    Clapeyron-like implicit derivative.

    Args:
        params: Single-component PC-SAFT parameter set.
        t: Temperature (K).
        p_guess: Initial pressure estimate (Pa); a Wilson/Antoine value is fine.
        tol: Residual tolerance on ``ln phi^L - ln phi^V``.
        max_iter: Newton iteration cap.

    Returns:
        The saturation pressure (Pa).
    """
    from fugacio.thermo.implicit import newton_root

    t = jnp.asarray(t, dtype=float)

    def residual(ln_p: Array, theta: tuple[SaftParameters, Array]) -> Array:
        params_, t_ = theta
        return _psat_residual(params_, t_, jnp.exp(ln_p))

    ln_p = newton_root(
        residual, (params, t), jnp.log(jnp.asarray(p_guess, dtype=float)), tol, max_iter, 0.7
    )
    return jnp.exp(ln_p)


def stability_saft(
    params: SaftParameters,
    t: ArrayLike,
    p: ArrayLike,
    z: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    iters: int = 40,
) -> StabilityResult:
    """Michelsen tangent-plane stability test for feed ``z`` at ``(T, P)`` on PC-SAFT.

    Runs vapour-like and liquid-like trial-phase searches; a modified
    tangent-plane distance below zero for either means the feed splits.
    """
    z = jnp.asarray(z, dtype=float)
    ln_phi_zl = ln_fugacity_coefficients(params, t, p, z, phase="liquid")
    ln_phi_zv = ln_fugacity_coefficients(params, t, p, z, phase="vapor")
    g_l = jnp.sum(z * (jnp.log(z) + ln_phi_zl))
    g_v = jnp.sum(z * (jnp.log(z) + ln_phi_zv))
    d = jnp.log(z) + jnp.where(g_l < g_v, ln_phi_zl, ln_phi_zv)
    k_wilson = wilson_k(t, p, tc, pc, omega)

    def run_trial(w0: Array, phase: str) -> Array:
        def body(_: Array, w: Array) -> Array:
            wn = w / jnp.sum(w)
            ln_phi_w = ln_fugacity_coefficients(params, t, p, wn, phase=phase)
            return jnp.exp(d - ln_phi_w)

        w = jax.lax.fori_loop(0, iters, body, w0)
        wn = w / jnp.sum(w)
        ln_phi_w = ln_fugacity_coefficients(params, t, p, wn, phase=phase)
        return 1.0 + jnp.sum(w * (jnp.log(w) + ln_phi_w - d - 1.0))

    # A trial phase whose density branch does not exist on this isotherm (e.g.
    # the vapour-like trial deep in the compressed liquid) yields a non-finite
    # tangent-plane distance; it offers no evidence of a split, so score it +inf.
    tm_vapor = jnp.nan_to_num(run_trial(z * k_wilson, "vapor"), nan=jnp.inf, posinf=jnp.inf)
    tm_liquid = jnp.nan_to_num(run_trial(z / k_wilson, "liquid"), nan=jnp.inf, posinf=jnp.inf)
    tpd = jnp.minimum(tm_vapor, tm_liquid)
    return StabilityResult(stable=tpd >= -1e-8, tpd=tpd)


__all__ = [
    "bubble_pressure_saft",
    "dew_pressure_saft",
    "flash_pt_saft",
    "psat_saft",
    "stability_saft",
]
