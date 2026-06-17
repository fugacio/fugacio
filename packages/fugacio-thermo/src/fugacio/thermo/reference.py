"""Pure-component liquid reference fugacity for the gamma-phi approach.

In the *gamma-phi* (activity-coefficient) description of vapour-liquid
equilibrium the liquid-phase fugacity of component ``i`` is written

    f_i^L = x_i * gamma_i(x, T) * f_i^{0,L}(T, P)

where ``f_i^{0,L}`` is the fugacity of *pure liquid* ``i`` at the mixture ``T, P``.
That reference fugacity is

    f_i^{0,L}(T, P) = phi_i^sat(T) * Psat_i(T) * Poynting_i(T, P)

with the saturation fugacity coefficient ``phi_i^sat`` (vapour non-ideality at the
pure saturation point), the saturation pressure ``Psat_i`` and the Poynting factor

    Poynting_i = exp[ v_i^L (P - Psat_i) / (R T) ]

correcting the liquid fugacity from ``Psat_i`` up to the system pressure ``P``. At
low pressure ``phi_i^sat -> 1`` and ``Poynting -> 1``, recovering the familiar
modified Raoult's law ``p_i = x_i gamma_i Psat_i``. This module supplies each
piece: saturation pressures (from the EOS or an Antoine fit), the Poynting
factor, and Henry's-law constants for non-condensable gases, all differentiable
in temperature, pressure, and the underlying parameters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import BAR, R
from fugacio.thermo.eos import CubicEOS, ln_phi_pure, molar_volume
from fugacio.thermo.equilibrium import psat_eos

ArrayLike = Array | float


def saturation_pressures(eos: CubicEOS, t: ArrayLike, tc: Array, pc: Array, omega: Array) -> Array:
    """Vector of pure-component saturation pressures ``Psat_i(T)`` (Pa) from the EOS.

    Maps the differentiable `fugacio.thermo.equilibrium.psat_eos` over every
    component, so the result carries Clapeyron ``dPsat/dT`` derivatives.
    """
    tc = jnp.asarray(tc)
    pc = jnp.asarray(pc)
    omega = jnp.asarray(omega)
    return jax.vmap(lambda a, b, c: psat_eos(eos, t, a, b, c))(tc, pc, omega)


def antoine_psat(t: ArrayLike, a: ArrayLike, b: ArrayLike, c: ArrayLike) -> Array:
    """Saturation pressure (Pa) from NIST-form Antoine ``log10(P/bar) = a - b/(T + c)``."""
    t = jnp.asarray(t)
    return BAR * jnp.power(10.0, a - b / (t + c))


def saturation_fugacity_coefficient(
    eos: CubicEOS, t: ArrayLike, psat: ArrayLike, tc: ArrayLike, pc: ArrayLike, omega: ArrayLike
) -> Array:
    """Saturation fugacity coefficient ``phi_i^sat`` of pure ``i`` (vapour root at ``Psat``)."""
    ln_phi, _ = ln_phi_pure(eos, t, psat, tc, pc, omega, phase="vapor")
    return jnp.exp(ln_phi)


def poynting_factor(v_liquid: Array, p: ArrayLike, psat: Array, t: ArrayLike) -> Array:
    """Poynting correction ``exp[v_L (P - Psat) / (R T)]`` (per component).

    Args:
        v_liquid: Pure-liquid molar volumes ``v_i^L`` (m^3/mol), shape ``(n,)``.
        p: System pressure (Pa).
        psat: Saturation pressures ``Psat_i`` (Pa), shape ``(n,)``.
        t: Temperature (K).
    """
    v_liquid = jnp.asarray(v_liquid)
    psat = jnp.asarray(psat)
    return jnp.exp(v_liquid * (jnp.asarray(p) - psat) / (R * jnp.asarray(t)))


def pure_liquid_volumes(
    eos: CubicEOS, t: ArrayLike, psat: Array, tc: Array, pc: Array, omega: Array
) -> Array:
    """Pure-liquid molar volumes ``v_i^L`` at each component's saturation point (m^3/mol)."""
    tc = jnp.asarray(tc)
    pc = jnp.asarray(pc)
    omega = jnp.asarray(omega)
    psat = jnp.asarray(psat)

    def one(ps: Array, a: Array, b: Array, c: Array) -> Array:
        return molar_volume(
            eos, t, ps, jnp.asarray([1.0]), a[None], b[None], c[None], phase="liquid"
        )

    return jax.vmap(one)(psat, tc, pc, omega)


def liquid_reference_fugacity(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    poynting: bool = True,
    phi_saturation: bool = True,
) -> tuple[Array, Array]:
    """Pure-liquid reference fugacity ``f_i^{0,L}(T, P)`` (Pa) for every component.

    Assembles ``phi^sat * Psat * Poynting``. Disable ``phi_saturation`` and
    ``poynting`` (both default on) to recover the plain ``Psat`` reference of
    elementary modified Raoult's law.

    Returns:
        ``(f_ref, psat)``: the reference fugacities and the saturation pressures
        used to build them (the latter is handy for K-value initialisation).
    """
    psat = saturation_pressures(eos, t, tc, pc, omega)
    f_ref = psat
    if phi_saturation:
        f_ref = f_ref * jax.vmap(
            lambda ps, a, b, c: saturation_fugacity_coefficient(eos, t, ps, a, b, c)
        )(psat, jnp.asarray(tc), jnp.asarray(pc), jnp.asarray(omega))
    if poynting:
        v_l = pure_liquid_volumes(eos, t, psat, tc, pc, omega)
        f_ref = f_ref * poynting_factor(v_l, p, psat, t)
    return f_ref, psat


def henry_constant(t: ArrayLike, a: ArrayLike, b: ArrayLike, c: ArrayLike, d: ArrayLike) -> Array:
    """Henry's-law constant ``H(T) = exp(a + b/T + c ln T + d T)`` (Pa).

    The standard four-parameter correlation for the solubility of a
    non-condensable gas in a solvent; the liquid fugacity of a Henry component is
    ``f_i^L = x_i H_i(T)`` (an unsymmetric reference), replacing the
    saturation-based reference that does not exist above the gas's critical
    temperature.
    """
    t = jnp.asarray(t)
    return jnp.exp(a + b / t + c * jnp.log(t) + d * t)
