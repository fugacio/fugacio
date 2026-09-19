"""Gamma-phi vapour-liquid equilibrium (activity-coefficient liquid model).

The cubic-EOS flash in `fugacio.thermo.equilibrium` describes both phases
with one equation of state (the *phi-phi* approach). For the low-pressure, polar,
strongly non-ideal mixtures that dominate real separations (ethanol/water and
other azeotropes, alcohol/ketone/water systems), a cubic EOS with zero binary
interaction parameters is simply the wrong tool. The standard answer is the
*gamma-phi* approach: model the liquid with an activity-coefficient model and the
vapour with an equation of state (or as an ideal gas at low pressure).

Equilibrium equates the component fugacities

    x_i gamma_i(x, T) f_i^{0,L}(T, P) = y_i phi_i^V(y, T, P) P

so the K-values are

    K_i = y_i / x_i = gamma_i f_i^{0,L} / (phi_i^V P).

With an ideal vapour (``phi^V = 1``) and the plain saturation reference
(``f^{0,L} = Psat``), this collapses to modified Raoult's law
``K_i = gamma_i Psat_i / P``, enough to reproduce azeotropes that the
zero-``kij`` cubic cannot. The richer reference (saturation fugacity coefficient +
Poynting, see `fugacio.thermo.reference`) and an EOS vapour are available via
keyword flags.

The saturation-based reference exists only below each component's critical
temperature. Every calculation here reports ``OUT_OF_DOMAIN`` when a component
present in the relevant phase is supercritical, instead of returning a number
built on an extrapolated vapour pressure; absent components are harmless.

Each routine has a checked ``_with_info`` form and a value-only form that
returns NaN on failure. Converged results are differentiable end-to-end with
respect to ``T``, ``P``, composition, *and* the activity-model parameters, which
turns parameter regression (`fugacio.thermo.regression`) into plain gradient
descent.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.activity.models import ActivityModel
from fugacio.thermo.diagnostics import SolveReport, SolveStatus, nan_unless_converged, with_status
from fugacio.thermo.eos import PR, CubicEOS, ln_phi_mixture
from fugacio.thermo.equilibrium import (
    FlashResult,
    FlashSolveResult,
    SaturationResult,
    SaturationSolveResult,
    input_report,
    phase_compositions,
    rachford_rice,
)
from fugacio.thermo.implicit import fixed_point_with_info, gate_tree, scanned_root_with_info
from fugacio.thermo.reference import liquid_reference_fugacity, saturation_pressures_with_info

ArrayLike = Array | float


def _ln_phi_vapor(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    y: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    kij: Array | None,
    vapor: str,
) -> Array:
    """Log vapour fugacity coefficients, or zeros for an ideal-gas vapour."""
    if vapor == "ideal":
        return jnp.zeros_like(jnp.asarray(y))
    if vapor == "eos":
        ln_phi, _ = ln_phi_mixture(eos, t, p, y, tc, pc, omega, phase="vapor", kij=kij)
        return ln_phi
    raise ValueError(f"unknown vapor model {vapor!r}; use 'ideal' or 'eos'")


def reference_domain(
    t: ArrayLike, composition: Array, tc: Array, eos: CubicEOS, pc: Array, omega: Array
) -> Array:
    """Whether every component present in ``composition`` has a saturation reference at ``t``."""
    _, converged = saturation_pressures_with_info(eos, t, tc, pc, omega)
    return jax.lax.stop_gradient(jnp.all(jnp.where(jnp.asarray(composition) > 0, converged, True)))


def gamma_phi_k_values(
    model: ActivityModel,
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    y: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
) -> Array:
    """Gamma-phi K-values ``K_i = gamma_i f_i^{0,L} / (phi_i^V P)``.

    Args:
        model: Liquid activity-coefficient model.
        t: Temperature (K).
        p: Pressure (Pa).
        x: Liquid mole fractions.
        y: Vapour mole fractions (only matters for an EOS vapour).
        tc: Component critical temperatures (K).
        pc: Component critical pressures (Pa).
        omega: Component acentric factors.
        eos: Cubic EOS for the saturation reference and (if selected) the vapour.
        kij: Optional binary interaction matrix for the vapour EOS.
        vapor: ``"ideal"`` (phi^V = 1) or ``"eos"``.
        poynting: Include the Poynting pressure correction in the reference.
        phi_saturation: Include the saturation fugacity coefficient in the reference.

    Returns:
        K-values aligned with ``x``. Supercritical components carry values built
        on an extrapolated vapour pressure; see `reference_domain`.
    """
    f_ref, _ = liquid_reference_fugacity(
        eos, t, p, tc, pc, omega, poynting=poynting, phi_saturation=phi_saturation
    )
    ln_gamma = model.ln_gamma(x, t)
    ln_phi_v = _ln_phi_vapor(eos, t, p, y, tc, pc, omega, kij, vapor)
    return jnp.exp(ln_gamma) * f_ref / (jnp.exp(ln_phi_v) * jnp.asarray(p))


def _domain_report(
    report: SolveReport,
    t: ArrayLike,
    composition: Array,
    tc: Array,
    eos: CubicEOS,
    pc: Array,
    omega: Array,
) -> SolveReport:
    """Mark a report out of domain when a present component lacks a reference."""
    valid = reference_domain(t, composition, tc, eos, pc, omega)
    return with_status(report, ~valid, SolveStatus.OUT_OF_DOMAIN)


def bubble_pressure_gamma_with_info(
    model: ActivityModel,
    t: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> SaturationSolveResult:
    """Bubble-point pressure and incipient vapour composition at fixed ``T``, ``x``.

    Solved as a coupled fixed point in ``(ln P, y)``: K-values give an unnormalised
    vapour ``y* = K x`` whose sum scales the pressure until it is one.
    """
    x = jnp.asarray(x)
    psat, _ = saturation_pressures_with_info(eos, t, tc, pc, omega)
    gamma0 = jnp.exp(model.ln_gamma(x, t))
    p0 = jnp.sum(x * gamma0 * psat)
    y0 = x * gamma0 * psat / p0
    state0 = jnp.concatenate([jnp.log(p0)[None], y0])
    theta = (model, jnp.asarray(t, dtype=float), x, tc, pc, omega)

    def g(state: Array, theta: Any) -> Array:
        model_, t_, x_, tc_, pc_, omega_ = theta
        p = jnp.exp(state[0])
        y = state[1:]
        k = gamma_phi_k_values(
            model_,
            t_,
            p,
            x_,
            y,
            tc_,
            pc_,
            omega_,
            eos=eos,
            kij=kij,
            vapor=vapor,
            poynting=poynting,
            phi_saturation=phi_saturation,
        )
        y_unnorm = k * x_
        s = jnp.sum(y_unnorm)
        return jnp.concatenate([(state[0] + jnp.log(s))[None], y_unnorm / s])

    solved = fixed_point_with_info(g, state0, theta, tol, max_iter)
    report = _domain_report(solved.report, t, x, tc, eos, pc, omega)
    report = with_status(report, ~input_report(t, 1.0, x), SolveStatus.INVALID_INPUT)
    value = SaturationResult(jnp.exp(solved.value[0]), solved.value[1:])
    return SaturationSolveResult(gate_tree(value, report.converged), report)


def bubble_pressure_gamma(
    model: ActivityModel, t: ArrayLike, x: Array, *args: Any, **kwargs: Any
) -> SaturationResult:
    """Bubble pressure and incipient vapour ``(P, y)``; NaN on failure."""
    solved = bubble_pressure_gamma_with_info(model, t, x, *args, **kwargs)
    return nan_unless_converged(solved.value, solved.report)


def dew_pressure_gamma_with_info(
    model: ActivityModel,
    t: ArrayLike,
    y: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> SaturationSolveResult:
    """Dew-point pressure and incipient liquid composition at fixed ``T``, ``y``.

    Coupled fixed point in ``(ln P, x)``: with ``x_i = y_i / K_i`` (and the
    activity coefficients re-evaluated at the updated ``x``), the pressure is
    scaled until the liquid sums to one.
    """
    y = jnp.asarray(y)
    psat, _ = saturation_pressures_with_info(eos, t, tc, pc, omega)
    p0 = 1.0 / jnp.sum(y / psat)
    x0 = y * p0 / psat
    x0 = x0 / jnp.sum(x0)
    state0 = jnp.concatenate([jnp.log(p0)[None], x0])
    theta = (model, jnp.asarray(t, dtype=float), y, tc, pc, omega)

    def g(state: Array, theta: Any) -> Array:
        model_, t_, y_, tc_, pc_, omega_ = theta
        p = jnp.exp(state[0])
        x = state[1:]
        k = gamma_phi_k_values(
            model_,
            t_,
            p,
            x,
            y_,
            tc_,
            pc_,
            omega_,
            eos=eos,
            kij=kij,
            vapor=vapor,
            poynting=poynting,
            phi_saturation=phi_saturation,
        )
        x_unnorm = y_ / k
        s = jnp.sum(x_unnorm)
        return jnp.concatenate([(state[0] - jnp.log(s))[None], x_unnorm / s])

    solved = fixed_point_with_info(g, state0, theta, tol, max_iter)
    report = _domain_report(solved.report, t, y, tc, eos, pc, omega)
    report = with_status(report, ~input_report(t, 1.0, y), SolveStatus.INVALID_INPUT)
    value = SaturationResult(jnp.exp(solved.value[0]), solved.value[1:])
    return SaturationSolveResult(gate_tree(value, report.converged), report)


def dew_pressure_gamma(
    model: ActivityModel, t: ArrayLike, y: Array, *args: Any, **kwargs: Any
) -> SaturationResult:
    """Dew pressure and incipient liquid ``(P, x)``; NaN on failure."""
    solved = dew_pressure_gamma_with_info(model, t, y, *args, **kwargs)
    return nan_unless_converged(solved.value, solved.report)


def default_temperature_bracket(tc: Array, composition: Array) -> tuple[Array, Array]:
    """A saturation-temperature bracket from the present components' critical points.

    ``[0.2 min Tc, max Tc]`` over the components present in ``composition``: wide
    enough for cryogenic and heavy systems, and scanned for the first finite
    sign change rather than assumed valid.
    """
    present = jnp.asarray(composition) > 0
    tc = jnp.asarray(tc)
    lo = 0.2 * jnp.min(jnp.where(present, tc, jnp.inf))
    hi = jnp.max(jnp.where(present, tc, -jnp.inf))
    return jax.lax.stop_gradient(lo), jax.lax.stop_gradient(hi)


def _incipient_vapor(
    model_: Any,
    t: Array,
    p_: Array,
    x_: Array,
    tc_: Array,
    pc_: Array,
    omega_: Array,
    options: dict[str, Any],
    inner_iter: int,
) -> Array:
    """Incipient vapour at ``(T, P, x)``, settled by ``inner_iter`` sweeps for an EOS vapour."""

    def step(_: int, y_cur: Array) -> Array:
        k = gamma_phi_k_values(model_, t, p_, x_, y_cur, tc_, pc_, omega_, **options)
        yn = k * x_
        return yn / jnp.sum(yn)

    return jax.lax.fori_loop(0, inner_iter, step, step(0, x_))


def bubble_temperature_gamma_with_info(
    model: ActivityModel,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
    t_min: ArrayLike | None = None,
    t_max: ArrayLike | None = None,
    tol: float = 1e-9,
    max_iter: int = 200,
    inner_iter: int | None = None,
) -> SaturationSolveResult:
    """Bubble-point temperature and incipient vapour at fixed ``P``, ``x``.

    The bubble temperature is the root of ``sum_i K_i(T) x_i = 1`` (monotone in
    ``T``), located by `fugacio.thermo.implicit.scanned_root_with_info` on
    ``[t_min, t_max]`` (by default `default_temperature_bracket`) and
    differentiated by the implicit function theorem. For an EOS vapour the
    incipient ``y`` is settled by ``inner_iter`` sweeps (default 30) at every
    trial temperature; an ideal vapour needs none.
    """
    x = jnp.asarray(x)
    options: dict[str, Any] = {
        "eos": eos,
        "kij": kij,
        "vapor": vapor,
        "poynting": poynting,
        "phi_saturation": phi_saturation,
    }
    sweeps = (0 if vapor == "ideal" else 30) if inner_iter is None else inner_iter
    lo, hi = default_temperature_bracket(tc, x)
    lo = lo if t_min is None else jnp.asarray(t_min, dtype=float)
    hi = hi if t_max is None else jnp.asarray(t_max, dtype=float)

    def residual(t: Array, params: Any) -> Array:
        model_, p_, x_, tc_, pc_, omega_ = params
        y = _incipient_vapor(model_, t, p_, x_, tc_, pc_, omega_, options, sweeps)
        k = gamma_phi_k_values(model_, t, p_, x_, y, tc_, pc_, omega_, **options)
        # Undefined wherever a present component has no saturation reference.
        valid = reference_domain(t, x_, tc_, eos, pc_, omega_)
        return jnp.where(valid, jnp.log(jnp.sum(k * x_)), jnp.nan)

    params = (model, jnp.asarray(p, dtype=float), x, tc, pc, omega)
    solved = scanned_root_with_info(residual, params, lo, hi, tol, max_iter)
    t_star = solved.value
    y = _incipient_vapor(
        model, t_star, jnp.asarray(p, dtype=float), x, tc, pc, omega, options, sweeps
    )
    report = _domain_report(solved.report, t_star, x, tc, eos, pc, omega)
    report = with_status(report, ~input_report(1.0, p, x), SolveStatus.INVALID_INPUT)
    return SaturationSolveResult(gate_tree(SaturationResult(t_star, y), report.converged), report)


def bubble_temperature_gamma(
    model: ActivityModel, p: ArrayLike, x: Array, *args: Any, **kwargs: Any
) -> SaturationResult:
    """Bubble temperature and incipient vapour ``(T, y)``; NaN on failure."""
    solved = bubble_temperature_gamma_with_info(model, p, x, *args, **kwargs)
    return nan_unless_converged(solved.value, solved.report)


def dew_temperature_gamma_with_info(
    model: ActivityModel,
    p: ArrayLike,
    y: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
    t_min: ArrayLike | None = None,
    t_max: ArrayLike | None = None,
    tol: float = 1e-9,
    max_iter: int = 200,
    inner_iter: int = 30,
) -> SaturationSolveResult:
    """Dew-point temperature and incipient liquid at fixed ``P``, ``y``.

    Root of ``sum_i (y_i / K_i(T)) = 1`` in ``T`` with an inner sweep that settles
    the incipient liquid ``x`` (on which the activity coefficients depend) at each
    trial temperature, on a scanned bracket (see
    `bubble_temperature_gamma_with_info`).
    """
    y = jnp.asarray(y)
    options: dict[str, Any] = {
        "eos": eos,
        "kij": kij,
        "vapor": vapor,
        "poynting": poynting,
        "phi_saturation": phi_saturation,
    }
    lo, hi = default_temperature_bracket(tc, y)
    lo = lo if t_min is None else jnp.asarray(t_min, dtype=float)
    hi = hi if t_max is None else jnp.asarray(t_max, dtype=float)

    def liquid_at(t: Array, params: Any) -> Array:
        model_, p_, y_, tc_, pc_, omega_ = params
        psat, _ = saturation_pressures_with_info(eos, t, tc_, pc_, omega_)
        x = y_ * (1.0 / jnp.sum(y_ / psat)) / psat
        x = x / jnp.sum(x)

        def step(_: int, x_cur: Array) -> Array:
            k = gamma_phi_k_values(model_, t, p_, x_cur, y_, tc_, pc_, omega_, **options)
            xn = y_ / k
            return xn / jnp.sum(xn)

        return jax.lax.fori_loop(0, inner_iter, step, x)

    def residual(t: Array, params: Any) -> Array:
        model_, p_, y_, tc_, pc_, omega_ = params
        x = liquid_at(t, params)
        k = gamma_phi_k_values(model_, t, p_, x, y_, tc_, pc_, omega_, **options)
        valid = reference_domain(t, y_, tc_, eos, pc_, omega_)
        # ln(sum y/K) decreases with T; negate it so the root is a rising crossing.
        return jnp.where(valid, -jnp.log(jnp.sum(y_ / k)), jnp.nan)

    params = (model, jnp.asarray(p, dtype=float), y, tc, pc, omega)
    solved = scanned_root_with_info(residual, params, lo, hi, tol, max_iter)
    t_star = solved.value
    x = liquid_at(t_star, params)
    report = _domain_report(solved.report, t_star, y, tc, eos, pc, omega)
    report = with_status(report, ~input_report(1.0, p, y), SolveStatus.INVALID_INPUT)
    return SaturationSolveResult(gate_tree(SaturationResult(t_star, x), report.converged), report)


def dew_temperature_gamma(
    model: ActivityModel, p: ArrayLike, y: Array, *args: Any, **kwargs: Any
) -> SaturationResult:
    """Dew temperature and incipient liquid ``(T, x)``; NaN on failure."""
    solved = dew_temperature_gamma_with_info(model, p, y, *args, **kwargs)
    return nan_unless_converged(solved.value, solved.report)


def flash_pt_gamma(
    model: ActivityModel, t: ArrayLike, p: ArrayLike, z: Array, *args: Any, **kwargs: Any
) -> FlashResult:
    """Isothermal flash value; NaN when `flash_pt_gamma_with_info` reports failure."""
    solved = flash_pt_gamma_with_info(model, t, p, z, *args, **kwargs)
    return nan_unless_converged(solved.value, solved.report)


def flash_pt_gamma_with_info(
    model: ActivityModel,
    t: ArrayLike,
    p: ArrayLike,
    z: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
    tol: float = 1e-12,
    max_iter: int = 300,
) -> FlashSolveResult:
    """Isothermal-isobaric gamma-phi flash by successive substitution.

    Iterates the gamma-phi K-values to a fixed point in ``ln K`` with the
    Rachford-Rice material balance closing the phase split at each step. The
    converged ``beta``, ``x``, ``y`` are differentiable with respect to
    ``(T, P, z)`` and the activity-model parameters. A present supercritical
    component is ``OUT_OF_DOMAIN``.
    """
    z = jnp.asarray(z)
    psat, _ = saturation_pressures_with_info(eos, t, tc, pc, omega)
    gamma0 = jnp.exp(model.ln_gamma(z, t))
    k0 = gamma0 * psat / jnp.asarray(p)
    theta = (model, jnp.asarray(t, dtype=float), jnp.asarray(p, dtype=float), z, tc, pc, omega)

    def g(ln_k: Array, theta: Any) -> Array:
        model_, t_, p_, z_, tc_, pc_, omega_ = theta
        k = jnp.exp(ln_k)
        beta = rachford_rice(z_, k)
        # Normalized trial phases (a no-op at an interior root); activity
        # coefficients are only defined for mole fractions.
        x, y = phase_compositions(z_, k, beta)
        x, y = x / jnp.sum(x), y / jnp.sum(y)
        k_new = gamma_phi_k_values(
            model_,
            t_,
            p_,
            x,
            y,
            tc_,
            pc_,
            omega_,
            eos=eos,
            kij=kij,
            vapor=vapor,
            poynting=poynting,
            phi_saturation=phi_saturation,
        )
        return jnp.log(k_new)

    solved = fixed_point_with_info(g, jnp.log(k0), theta, tol, max_iter)
    k = jnp.exp(solved.value)
    beta = rachford_rice(z, k)
    x, y = phase_compositions(z, k, beta)
    report = _domain_report(solved.report, t, z, tc, eos, pc, omega)
    report = with_status(report, ~input_report(t, p, z), SolveStatus.INVALID_INPUT)
    return FlashSolveResult(FlashResult(beta=beta, x=x, y=y, k=k), report)
