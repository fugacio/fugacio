"""Stream property bridge: enthalpy, entropy, and mass flows for a :class:`Stream`.

Unit operations close *material and energy* balances, so they need a stream's
enthalpy and entropy, not just its composition. This module resolves a stream's
(static) component names to the array constants the :mod:`fugacio.thermo` kernels
expect -- caching that lookup, since names never change during a solve -- and
exposes the resulting molar and total-flow properties.

Enthalpy and entropy are two-phase aware: they run the equilibrium flash at the
stream's ``(T, P)`` and blend the phase properties, so a subcooled liquid, a
superheated vapour, and a flashing two-phase stream are all handled by the same
call. Everything stays differentiable with respect to the stream's flows,
temperature, and pressure (the component constants are not differentiated, which
is exactly right -- they are reference data, not decision variables).
"""

from __future__ import annotations

from functools import cache

import jax.numpy as jnp
from jax import Array

from fugacio.sim.stream import Stream
from fugacio.thermo import (
    PR,
    CubicEOS,
    component_arrays,
    get,
    ideal_gas_coeffs,
)
from fugacio.thermo.energy import mixture_enthalpy, mixture_entropy

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
