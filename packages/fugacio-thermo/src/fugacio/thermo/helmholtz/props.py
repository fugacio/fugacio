"""Thermodynamic properties from a reference Helmholtz EOS, at ``(rho, T)``.

Every function here is an algebraic combination of the autodiff partials of
``alpha(delta, tau)`` (`fugacio.thermo.helmholtz.terms`), evaluated at a
molar density ``rho`` (mol/m^3) and temperature ``t`` (K) -- the natural
variables of a Helmholtz EOS. The standard property relations are (Span,
*Multiparameter Equations of State*, 2000):

    P           = rho R T (1 + delta ar_d)
    u / (R T)   = tau (a0_t + ar_t)
    h / (R T)   = 1 + tau (a0_t + ar_t) + delta ar_d
    s / R       = tau (a0_t + ar_t) - a0 - ar
    cv / R      = -tau^2 (a0_tt + ar_tt)
    cp / R      = cv/R + (1 + delta ar_d - delta tau ar_dt)^2
                          / (1 + 2 delta ar_d + delta^2 ar_dd)
    w^2 M/(R T) = 1 + 2 delta ar_d + delta^2 ar_dd
                  - (1 + delta ar_d - delta tau ar_dt)^2 / (tau^2 (a0_tt + ar_tt))
    ln(phi)     = ar + Z - 1 - ln(Z)

Enthalpy and entropy are on each fluid's published reference state (for water
the IAPWS-95 convention: zero internal energy and entropy of the saturated
liquid at the triple point), *not* on the formation basis used by
`fugacio.thermo.ideal`; differences cancel in the Delta-H / Delta-S that
energy balances consume. Use `fugacio.thermo.helmholtz.states.state_tp`
and friends when starting from ``(T, P)`` instead of ``(rho, T)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.helmholtz.fluids import HelmholtzFluid
from fugacio.thermo.helmholtz.terms import alpha_derivatives, first_derivatives, residual_alpha

ArrayLike = Array | float


def pressure(fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike) -> Array:
    """Pressure (Pa) at molar density ``rho`` (mol/m^3) and ``t`` (K)."""
    rho = jnp.asarray(rho, dtype=float)
    t = jnp.asarray(t, dtype=float)
    delta = rho / fluid.rho_reducing
    tau = fluid.t_reducing / t

    def ar(d: Array) -> Array:
        return residual_alpha(fluid, d, tau)

    ar_d = jax.grad(ar)(delta)
    return rho * fluid.gas_constant * t * (1.0 + delta * ar_d)


def compressibility_factor(fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike) -> Array:
    """Compressibility factor ``Z = P/(rho R T)``."""
    rho = jnp.asarray(rho, dtype=float)
    return pressure(fluid, rho, t) / (rho * fluid.gas_constant * jnp.asarray(t, dtype=float))


def internal_energy(fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike) -> Array:
    """Molar internal energy (J/mol)."""
    _a0, a0_t, _ar, _ar_d, ar_t = first_derivatives(fluid, rho, t)
    t = jnp.asarray(t, dtype=float)
    tau = fluid.t_reducing / t
    return fluid.gas_constant * t * tau * (a0_t + ar_t)


def enthalpy(fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike) -> Array:
    """Molar enthalpy (J/mol) on the fluid's published reference state."""
    _a0, a0_t, _ar, ar_d, ar_t = first_derivatives(fluid, rho, t)
    t = jnp.asarray(t, dtype=float)
    delta = jnp.asarray(rho, dtype=float) / fluid.rho_reducing
    tau = fluid.t_reducing / t
    return fluid.gas_constant * t * (1.0 + tau * (a0_t + ar_t) + delta * ar_d)


def entropy(fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike) -> Array:
    """Molar entropy (J/mol/K) on the fluid's published reference state."""
    a0, a0_t, ar, _ar_d, ar_t = first_derivatives(fluid, rho, t)
    tau = fluid.t_reducing / jnp.asarray(t, dtype=float)
    return fluid.gas_constant * (tau * (a0_t + ar_t) - a0 - ar)


def gibbs_energy(fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike) -> Array:
    """Molar Gibbs energy (J/mol); equal in both phases at saturation."""
    a0, _a0_t, ar, ar_d, _ar_t = first_derivatives(fluid, rho, t)
    t = jnp.asarray(t, dtype=float)
    delta = jnp.asarray(rho, dtype=float) / fluid.rho_reducing
    return fluid.gas_constant * t * (1.0 + a0 + ar + delta * ar_d)


def helmholtz_energy(fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike) -> Array:
    """Molar Helmholtz energy ``a = R T alpha`` (J/mol)."""
    a0, _a0_t, ar, _ar_d, _ar_t = first_derivatives(fluid, rho, t)
    return fluid.gas_constant * jnp.asarray(t, dtype=float) * (a0 + ar)


def isochoric_heat_capacity(fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike) -> Array:
    """Molar isochoric heat capacity ``cv`` (J/mol/K)."""
    d = alpha_derivatives(fluid, rho, t)
    tau = fluid.t_reducing / jnp.asarray(t, dtype=float)
    return -fluid.gas_constant * tau**2 * (d.a0_tt + d.ar_tt)


def isobaric_heat_capacity(fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike) -> Array:
    """Molar isobaric heat capacity ``cp`` (J/mol/K); diverges at the critical point."""
    d = alpha_derivatives(fluid, rho, t)
    delta = jnp.asarray(rho, dtype=float) / fluid.rho_reducing
    tau = fluid.t_reducing / jnp.asarray(t, dtype=float)
    cv = -(tau**2) * (d.a0_tt + d.ar_tt)
    numerator = (1.0 + delta * d.ar_d - delta * tau * d.ar_dt) ** 2
    denominator = 1.0 + 2.0 * delta * d.ar_d + delta**2 * d.ar_dd
    return fluid.gas_constant * (cv + numerator / denominator)


def speed_of_sound(fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike) -> Array:
    """Speed of sound (m/s)."""
    d = alpha_derivatives(fluid, rho, t)
    t = jnp.asarray(t, dtype=float)
    delta = jnp.asarray(rho, dtype=float) / fluid.rho_reducing
    tau = fluid.t_reducing / t
    w_sq = (
        1.0
        + 2.0 * delta * d.ar_d
        + delta**2 * d.ar_dd
        - (1.0 + delta * d.ar_d - delta * tau * d.ar_dt) ** 2 / (tau**2 * (d.a0_tt + d.ar_tt))
    )
    return jnp.sqrt(jnp.maximum(w_sq, 0.0) * fluid.gas_constant * t / fluid.molar_mass)


def ln_fugacity_coefficient(fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike) -> Array:
    """Log fugacity coefficient ``ln(phi) = ar + Z - 1 - ln(Z)`` of the pure fluid."""
    rho = jnp.asarray(rho, dtype=float)
    t = jnp.asarray(t, dtype=float)
    delta = rho / fluid.rho_reducing
    tau = fluid.t_reducing / t

    def ar(d: Array) -> Array:
        return residual_alpha(fluid, d, tau)

    ar_value = ar(delta)
    z = 1.0 + delta * jax.grad(ar)(delta)
    return ar_value + z - 1.0 - jnp.log(z)


def fugacity(fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike) -> Array:
    """Fugacity ``f = phi * P`` (Pa) of the pure fluid."""
    return jnp.exp(ln_fugacity_coefficient(fluid, rho, t)) * pressure(fluid, rho, t)


def joule_thomson(fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike) -> Array:
    """Joule-Thomson coefficient ``(dT/dP)_h`` (K/Pa)."""
    d = alpha_derivatives(fluid, rho, t)
    rho = jnp.asarray(rho, dtype=float)
    delta = rho / fluid.rho_reducing
    tau = fluid.t_reducing / jnp.asarray(t, dtype=float)
    a = 1.0 + delta * d.ar_d - delta * tau * d.ar_dt
    numerator = -(delta * d.ar_d + delta**2 * d.ar_dd + delta * tau * d.ar_dt)
    denominator = a**2 - tau**2 * (d.a0_tt + d.ar_tt) * (
        1.0 + 2.0 * delta * d.ar_d + delta**2 * d.ar_dd
    )
    return numerator / denominator / (fluid.gas_constant * rho)


def isothermal_compressibility(fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike) -> Array:
    """Isothermal compressibility ``kappa_T = -(dV/dP)_T / V`` (1/Pa)."""
    d = alpha_derivatives(fluid, rho, t)
    rho = jnp.asarray(rho, dtype=float)
    delta = rho / fluid.rho_reducing
    denominator = 1.0 + 2.0 * delta * d.ar_d + delta**2 * d.ar_dd
    return 1.0 / (rho * fluid.gas_constant * jnp.asarray(t, dtype=float) * denominator)


def isobaric_expansivity(fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike) -> Array:
    """Volumetric thermal expansion coefficient ``alpha_V = (dV/dT)_P / V`` (1/K)."""
    d = alpha_derivatives(fluid, rho, t)
    t = jnp.asarray(t, dtype=float)
    delta = jnp.asarray(rho, dtype=float) / fluid.rho_reducing
    tau = fluid.t_reducing / t
    numerator = 1.0 + delta * d.ar_d - delta * tau * d.ar_dt
    denominator = 1.0 + 2.0 * delta * d.ar_d + delta**2 * d.ar_dd
    return numerator / (denominator * t)


def second_virial(fluid: HelmholtzFluid, t: ArrayLike) -> Array:
    """Second virial coefficient ``B(T) = ar_d(delta -> 0) / rho_reducing`` (m^3/mol)."""
    tau = fluid.t_reducing / jnp.asarray(t, dtype=float)

    def ar(d: Array) -> Array:
        return residual_alpha(fluid, d, tau)

    return jax.grad(ar)(jnp.asarray(0.0)) / fluid.rho_reducing


def third_virial(fluid: HelmholtzFluid, t: ArrayLike) -> Array:
    """Third virial coefficient ``C(T) = ar_dd(delta -> 0) / rho_reducing^2`` (m^6/mol^2).

    Evaluated at a vanishing reduced density rather than exactly zero: the
    second power-rule derivative of the ``d = 1`` terms is the indeterminate
    ``0 * delta**-1`` at the origin, which autodiff cannot cancel symbolically.
    The offset changes the result by O(1e-12).
    """
    tau = fluid.t_reducing / jnp.asarray(t, dtype=float)

    def ar(d: Array) -> Array:
        return residual_alpha(fluid, d, tau)

    return jax.grad(jax.grad(ar))(jnp.asarray(1e-12)) / fluid.rho_reducing**2


__all__ = [
    "compressibility_factor",
    "enthalpy",
    "entropy",
    "fugacity",
    "gibbs_energy",
    "helmholtz_energy",
    "internal_energy",
    "isobaric_expansivity",
    "isobaric_heat_capacity",
    "isochoric_heat_capacity",
    "isothermal_compressibility",
    "joule_thomson",
    "ln_fugacity_coefficient",
    "pressure",
    "second_virial",
    "speed_of_sound",
    "third_virial",
]
