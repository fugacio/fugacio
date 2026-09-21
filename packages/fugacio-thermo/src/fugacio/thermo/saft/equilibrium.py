"""PC-SAFT phase equilibrium: flash, bubble/dew points, and saturation.

These routines mirror the cubic-EOS equilibrium layer
(`fugacio.thermo.equilibrium`) one-for-one, swapping the cubic fugacity
coefficient for the PC-SAFT one (`fugacio.thermo.saft.properties`). Each has a
checked ``_with_info`` form returning a `fugacio.thermo.diagnostics.SolveReport`
and a value-only form that returns NaN when the report fails. Converged results
are differentiable, by the implicit function theorem, with respect to
temperature, pressure, composition, *and* the PC-SAFT parameters, which is
what lets `fugacio.thermo.saft.regression` fit a binary ``k_ij`` to VLE data by
gradient descent.

Wilson K-values seed the flashes, so the routines accept the critical constants
``(tc, pc, omega)`` for the seed only; the equilibrium itself is entirely
PC-SAFT. Phase stability is tested by every property package's ``stability``
method (`fugacio.thermo.stability`).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import R
from fugacio.thermo.diagnostics import (
    SolveResult,
    SolveStatus,
    nan_unless_converged,
    with_status,
)
from fugacio.thermo.equilibrium import (
    FlashResult,
    FlashSolveResult,
    SaturationResult,
    SaturationSolveResult,
    classify_trivial,
    input_report,
    phase_compositions,
    rachford_rice,
    wilson_k,
    wilson_psat,
)
from fugacio.thermo.implicit import fixed_point_with_info, gate_tree, newton_root_with_info
from fugacio.thermo.saft.parameters import SaftParameters
from fugacio.thermo.saft.properties import ln_fugacity_coefficients, molar_density

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
    """Isothermal flash value; NaN when `flash_pt_saft_with_info` reports failure."""
    solved = flash_pt_saft_with_info(params, t, p, z, tc, pc, omega, tol=tol, max_iter=max_iter)
    return nan_unless_converged(solved.value, solved.report)


def flash_pt_saft_with_info(
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
) -> FlashSolveResult:
    """Isothermal-isobaric vapour-liquid flash on PC-SAFT by successive substitution.

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

    solved = fixed_point_with_info(g, jnp.log(k0), theta, tol, max_iter)
    k = jnp.exp(solved.value)
    beta = rachford_rice(z, k)
    t_arr, p_arr = theta[1], theta[2]
    z_single = p_arr / (molar_density(params, t_arr, p_arr, z, phase="vapor") * R * t_arr)
    beta = classify_trivial(z, k, beta, k0, z_single)
    x, y = phase_compositions(z, k, beta)
    report = with_status(solved.report, ~input_report(t, p, z), SolveStatus.INVALID_INPUT)
    return FlashSolveResult(FlashResult(beta=beta, x=x, y=y, k=k), report)


def _trivial_saturation(
    params: SaftParameters, t: ArrayLike, p: Array, liquid: Array, vapor: Array
) -> Array:
    """Whether a converged saturation point collapsed onto one density root."""
    rho_l = molar_density(params, t, p, liquid, phase="liquid")
    rho_v = molar_density(params, t, p, vapor, phase="vapor")
    same_density = jnp.abs(rho_l - rho_v) <= 1e-6 * jnp.maximum(jnp.abs(rho_l), 1e-12)
    return jax.lax.stop_gradient(same_density & (jnp.max(jnp.abs(vapor - liquid)) <= 1e-6))


def bubble_pressure_saft_with_info(
    params: SaftParameters,
    t: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    tol: float = 1e-12,
    max_iter: int = 300,
) -> SaturationSolveResult:
    """Bubble-point pressure and incipient vapour at fixed ``T``, ``x``.

    Solved as a coupled fixed point in ``(ln P, y)``. A collapse onto one
    density root with ``y = x`` is reported as ``TRIVIAL``.
    """
    x = jnp.asarray(x, dtype=float)
    k0 = wilson_k(t, jnp.sum(x * pc), tc, pc, omega)
    p0 = jnp.sum(x * wilson_psat(t, tc, pc, omega))
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

    solved = fixed_point_with_info(g, state0, theta, tol, max_iter)
    p, y = jnp.exp(solved.value[0]), solved.value[1:]
    report = with_status(
        solved.report, _trivial_saturation(params, t, p, x, y), SolveStatus.TRIVIAL
    )
    report = with_status(report, ~input_report(t, 1.0, x), SolveStatus.INVALID_INPUT)
    return SaturationSolveResult(gate_tree(SaturationResult(p, y), report.converged), report)


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
) -> SaturationResult:
    """Bubble pressure and incipient vapour ``(P, y)``; NaN on failure."""
    solved = bubble_pressure_saft_with_info(params, t, x, tc, pc, omega, tol=tol, max_iter=max_iter)
    return nan_unless_converged(solved.value, solved.report)


def dew_pressure_saft_with_info(
    params: SaftParameters,
    t: ArrayLike,
    y: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    tol: float = 1e-12,
    max_iter: int = 300,
) -> SaturationSolveResult:
    """Dew-point pressure and incipient liquid at fixed ``T``, ``y``.

    Coupled fixed point in ``(ln P, x)``; a trivial collapse is ``TRIVIAL``.
    """
    y = jnp.asarray(y, dtype=float)
    k0 = wilson_k(t, jnp.sum(y * pc), tc, pc, omega)
    p0 = 1.0 / jnp.sum(y / wilson_psat(t, tc, pc, omega))
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

    solved = fixed_point_with_info(g, state0, theta, tol, max_iter)
    p, x = jnp.exp(solved.value[0]), solved.value[1:]
    report = with_status(
        solved.report, _trivial_saturation(params, t, p, x, y), SolveStatus.TRIVIAL
    )
    report = with_status(report, ~input_report(t, 1.0, y), SolveStatus.INVALID_INPUT)
    return SaturationSolveResult(gate_tree(SaturationResult(p, x), report.converged), report)


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
) -> SaturationResult:
    """Dew pressure and incipient liquid ``(P, x)``; NaN on failure."""
    solved = dew_pressure_saft_with_info(params, t, y, tc, pc, omega, tol=tol, max_iter=max_iter)
    return nan_unless_converged(solved.value, solved.report)


def _psat_residual(params: SaftParameters, t: Array, p: Array) -> Array:
    x = jnp.ones(1)
    ln_phi_l = ln_fugacity_coefficients(params, t, p, x, phase="liquid")[0]
    ln_phi_v = ln_fugacity_coefficients(params, t, p, x, phase="vapor")[0]
    return ln_phi_l - ln_phi_v


def psat_saft_with_info(
    params: SaftParameters,
    t: ArrayLike,
    p_guess: ArrayLike,
    *,
    tol: float = 1e-11,
    max_iter: int = 100,
) -> SolveResult:
    """Pure-component saturation pressure (Pa) by equifugacity, from a guess ``p_guess``.

    Solves ``ln phi^L(T, P) = ln phi^V(T, P)`` for ``P`` with a checked Newton
    iteration in ``ln P``. ``params`` must hold a *single* component. A pressure
    where the liquid and vapour density branches coincide satisfies the
    residual trivially and is reported as ``TRIVIAL``, not as a saturation
    point. Converged values are differentiable in ``T`` and the PC-SAFT
    parameters through the Clapeyron-like implicit derivative.

    Args:
        params: Single-component PC-SAFT parameter set.
        t: Temperature (K).
        p_guess: Initial pressure estimate (Pa); a Wilson/Antoine value is fine.
        tol: Residual tolerance on ``ln phi^L - ln phi^V``.
        max_iter: Newton iteration cap.

    Returns:
        The saturation pressure and its report.
    """
    t = jnp.asarray(t, dtype=float)

    def residual(ln_p: Array, theta: tuple[SaftParameters, Array]) -> Array:
        params_, t_ = theta
        return _psat_residual(params_, t_, jnp.exp(ln_p))

    solved = newton_root_with_info(
        residual, (params, t), jnp.log(jnp.asarray(p_guess, dtype=float)), tol, max_iter
    )
    p = jnp.exp(solved.value)
    one = jnp.ones(1)
    trivial = _trivial_saturation(params, t, p, one, one)
    report = with_status(solved.report, trivial, SolveStatus.TRIVIAL)
    return SolveResult(gate_tree(p, report.converged), report)


def psat_saft(
    params: SaftParameters,
    t: ArrayLike,
    p_guess: ArrayLike,
    *,
    tol: float = 1e-11,
    max_iter: int = 100,
) -> Array:
    """Saturation pressure (Pa); NaN when `psat_saft_with_info` reports failure."""
    solved = psat_saft_with_info(params, t, p_guess, tol=tol, max_iter=max_iter)
    return nan_unless_converged(solved.value, solved.report)


__all__ = [
    "bubble_pressure_saft",
    "bubble_pressure_saft_with_info",
    "dew_pressure_saft",
    "dew_pressure_saft_with_info",
    "flash_pt_saft",
    "flash_pt_saft_with_info",
    "psat_saft",
    "psat_saft_with_info",
]
