"""Stream property bridge: enthalpy, entropy, flows, and transport for a `Stream`.

Unit operations close *material and energy* balances, so they need a stream's
enthalpy and entropy, not just its composition. This module resolves a stream's
(static) component names to the array constants the `fugacio.thermo` kernels
expect -- caching that lookup, since names never change during a solve -- and
exposes the resulting molar and total-flow properties.

Enthalpy and entropy are two-phase aware: they run the equilibrium flash at the
stream's ``(T, P)`` and blend the phase properties, so a subcooled liquid, a
superheated vapour, and a flashing two-phase stream are all handled by the same
call. Everything stays differentiable with respect to the stream's flows,
temperature, and pressure (the component constants are not differentiated, which
is exactly right -- they are reference data, not decision variables).

Sizing-grade physical properties are surfaced too: phase densities and
volumetric flows (`liquid_density`, `vapor_volumetric_flow`),
viscosities, thermal conductivities, and surface tension, all evaluated at the
stream's state through the curated correlations and mixture rules in
`fugacio.thermo`. A stream-aware Souders-Brown helper
(`column_diameter_for`) wires them straight into the equipment-sizing
correlations of `fugacio.sim.economics`.
"""

from __future__ import annotations

from functools import cache

import jax.numpy as jnp
from jax import Array

from fugacio.sim.economics import column_diameter
from fugacio.sim.stream import Stream
from fugacio.thermo import (
    PR,
    CubicEOS,
    component_arrays,
    get,
    ideal_gas_coeffs,
)
from fugacio.thermo import (
    gas_mixture_thermal_conductivity as _gas_k,
)
from fugacio.thermo import (
    gas_mixture_viscosity as _gas_mu,
)
from fugacio.thermo import (
    liquid_density as _liquid_density,
)
from fugacio.thermo import (
    liquid_mixture_thermal_conductivity as _liquid_k,
)
from fugacio.thermo import (
    liquid_mixture_viscosity as _liquid_mu,
)
from fugacio.thermo import (
    mixture_surface_tension as _surface_tension,
)
from fugacio.thermo import (
    vapor_density as _vapor_density,
)
from fugacio.thermo.energy import mixture_enthalpy, mixture_entropy

ArrayLike = Array | float
CpCoeffs = tuple[Array, Array, Array, Array, Array]


@cache
def _resolve(components: tuple[str, ...]) -> tuple[Array, Array, Array, Array, CpCoeffs]:
    """Resolve component names to ``(tc, pc, omega, mw, cp)`` array constants (cached)."""
    arr = component_arrays(list(components))
    cp = ideal_gas_coeffs([get(c) for c in components])
    return arr["tc"], arr["pc"], arr["omega"], arr["mw"], cp


def molar_enthalpy(stream: Stream, *, eos: CubicEOS = PR, kij: Array | None = None) -> Array:
    """Molar enthalpy of the stream (J/mol), relative to the ideal-gas reference."""
    tc, pc, omega, _, cp = _resolve(stream.components)
    return mixture_enthalpy(eos, stream.t, stream.p, stream.z, tc, pc, omega, cp, kij=kij)


def molar_entropy(stream: Stream, *, eos: CubicEOS = PR, kij: Array | None = None) -> Array:
    """Molar entropy of the stream (J/mol/K), relative to the ideal-gas reference."""
    tc, pc, omega, _, cp = _resolve(stream.components)
    return mixture_entropy(eos, stream.t, stream.p, stream.z, tc, pc, omega, cp, kij=kij)


def enthalpy_flow(stream: Stream, *, eos: CubicEOS = PR, kij: Array | None = None) -> Array:
    """Total enthalpy flow of the stream (W = J/s)."""
    return stream.total * molar_enthalpy(stream, eos=eos, kij=kij)


def entropy_flow(stream: Stream, *, eos: CubicEOS = PR, kij: Array | None = None) -> Array:
    """Total entropy flow of the stream (W/K)."""
    return stream.total * molar_entropy(stream, eos=eos, kij=kij)


def molar_mass(stream: Stream) -> Array:
    """Mole-fraction-averaged molar mass of the stream (g/mol)."""
    _, _, _, mw, _ = _resolve(stream.components)
    return jnp.sum(stream.z * mw)


def mass_flow(stream: Stream) -> Array:
    """Total mass flow of the stream (kg/s)."""
    return stream.total * molar_mass(stream) * 1.0e-3


# --------------------------------------------------------------------------- #
# Volumetric and transport properties at the stream state
# --------------------------------------------------------------------------- #


def _names(stream: Stream) -> list[str]:
    return list(stream.components)


def liquid_density(stream: Stream) -> Array:
    """Saturated-liquid mass density at the stream's ``T`` and composition (kg/m^3)."""
    return _liquid_density(_names(stream), stream.t, stream.z)


def vapor_density(stream: Stream, *, eos: CubicEOS = PR) -> Array:
    """Vapour mass density from the EOS at the stream's ``(T, P)`` (kg/m^3)."""
    return _vapor_density(_names(stream), stream.t, stream.p, stream.z, eos=eos)


def liquid_volumetric_flow(stream: Stream) -> Array:
    """Volumetric flow if the stream is all liquid (m^3/s)."""
    return mass_flow(stream) / liquid_density(stream)


def vapor_volumetric_flow(stream: Stream, *, eos: CubicEOS = PR) -> Array:
    """Volumetric flow if the stream is all vapour (m^3/s)."""
    return mass_flow(stream) / vapor_density(stream, eos=eos)


def liquid_viscosity(stream: Stream) -> Array:
    """Liquid-mixture viscosity at the stream's ``T`` (Pa*s), Grunberg-Nissan."""
    return _liquid_mu(_names(stream), stream.t, stream.z)


def vapor_viscosity(stream: Stream) -> Array:
    """Dilute-gas mixture viscosity at the stream's ``T`` (Pa*s), Wilke."""
    return _gas_mu(_names(stream), stream.t, stream.z)


def liquid_thermal_conductivity(stream: Stream) -> Array:
    """Liquid-mixture thermal conductivity at the stream's ``T`` (W/m/K), DIPPR9H."""
    return _liquid_k(_names(stream), stream.t, stream.z)


def vapor_thermal_conductivity(stream: Stream) -> Array:
    """Gas-mixture thermal conductivity at the stream's ``T`` (W/m/K), Wassiljewa."""
    return _gas_k(_names(stream), stream.t, stream.z)


def surface_tension(stream: Stream) -> Array:
    """Liquid-mixture surface tension at the stream's ``T`` (N/m)."""
    return _surface_tension(_names(stream), stream.t, stream.z)


def column_diameter_for(
    vapor: Stream,
    liquid: Stream | None = None,
    *,
    k_drum: ArrayLike = 0.07,
    flooding: ArrayLike = 0.8,
) -> Array:
    """Souders-Brown column/drum diameter sized from the actual stream states (m).

    The vapour density, molar mass, and flow come from ``vapor``; the liquid
    density from ``liquid`` (defaulting to the vapour stream's composition at
    its own temperature -- the saturated-liquid view of the same material, a
    sensible drum approximation).
    """
    rho_v = vapor_density(vapor)
    rho_l = liquid_density(liquid if liquid is not None else vapor)
    return column_diameter(
        vapor.total,
        rho_v,
        rho_l,
        molar_mass=molar_mass(vapor) * 1.0e-3,
        k_drum=k_drum,
        flooding=flooding,
    )
