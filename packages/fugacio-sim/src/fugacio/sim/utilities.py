"""Steam and cooling-water utilities on reference-grade (IAPWS-95) water.

The economics layer prices utilities per gigajoule; this module supplies the
*physical* side: how many kg/s of steam a reboiler actually condenses, what a
cooling-water circuit's return flow is, how much shaft power a backpressure
turbine extracts from a header. All of it runs on the IAPWS-95 steam tables
from `fugacio.thermo.helmholtz` -- real latent heats at the header
pressure (not a constant 2257 kJ/kg), real liquid heat capacities, real
isentropic enthalpy drops -- and stays differentiable end to end, so a
total-annual-cost objective can take exact gradients through the utility
balances with respect to column duties, header pressures, or approach
temperatures.

Sign conventions match `fugacio.sim.units.heater`: duties are signed
heat *added to the process* (so a condenser's duty is negative); utility
functions accept either sign and size on the magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.helmholtz import (
    reference_fluid,
    saturation_state,
    state_ph,
    state_ps,
    state_tp,
    state_tq,
)

ArrayLike = Array | float

#: Typical plant steam headers (absolute pressure, Pa), Turton-style levels.
STEAM_LEVELS: dict[str, float] = {
    "lp": 5e5,
    "mp": 11e5,
    "hp": 42e5,
}


def _water_molar_mass() -> float:
    return reference_fluid("water").molar_mass


@dataclass(frozen=True)
class SteamHeatingResult:
    """Steam consumption of a heating duty.

    Attributes:
        mass_flow: Steam consumed (kg/s).
        molar_flow: Steam consumed (mol/s).
        t_steam: Steam supply temperature (K) -- saturation plus any superheat.
        t_condensate: Condensate return temperature (K).
        p: Header pressure (Pa).
        duty: Heat delivered to the process (W, positive).
        dh_specific: Specific heat released per kilogram of steam (J/kg).
    """

    mass_flow: Array
    molar_flow: Array
    t_steam: Array
    t_condensate: Array
    p: Array
    duty: Array
    dh_specific: Array


jax.tree_util.register_dataclass(
    SteamHeatingResult,
    data_fields=["mass_flow", "molar_flow", "t_steam", "t_condensate", "p", "duty", "dh_specific"],
    meta_fields=[],
)


def steam_heating(
    duty: ArrayLike,
    *,
    pressure: ArrayLike = STEAM_LEVELS["lp"],
    superheat: ArrayLike = 0.0,
    condensate_subcooling: ArrayLike = 0.0,
) -> SteamHeatingResult:
    """Steam flow required to deliver a heating duty from a saturated header.

    Steam arrives at ``pressure`` with optional ``superheat`` (K above
    saturation) and leaves as condensate ``condensate_subcooling`` K below
    saturation; the released enthalpy is evaluated from IAPWS-95, so the
    latent heat shrinks correctly as the header pressure rises toward the
    critical point.

    Args:
        duty: Heat to deliver to the process (W); the sign is ignored.
        pressure: Steam header pressure (Pa, absolute).
        superheat: Supply superheat above saturation (K).
        condensate_subcooling: Condensate return subcooling below saturation (K).

    Returns:
        A `SteamHeatingResult`; differentiable in every argument.
    """
    water = reference_fluid("water")
    duty = jnp.abs(jnp.asarray(duty, dtype=float))
    p = jnp.asarray(pressure, dtype=float)
    sat = saturation_state(water, p=p)
    t_steam = sat.t + jnp.asarray(superheat, dtype=float)
    t_condensate = sat.t - jnp.asarray(condensate_subcooling, dtype=float)

    h_in = jnp.where(
        jnp.asarray(superheat, dtype=float) > 0.0,
        state_tp(water, t_steam, p, phase="vapor").h,
        sat.h_vapor,
    )
    h_out = jnp.where(
        jnp.asarray(condensate_subcooling, dtype=float) > 0.0,
        state_tp(water, t_condensate, p, phase="liquid").h,
        sat.h_liquid,
    )
    dh_molar = h_in - h_out  # J/mol released
    molar_flow = duty / dh_molar
    return SteamHeatingResult(
        mass_flow=molar_flow * water.molar_mass,
        molar_flow=molar_flow,
        t_steam=t_steam,
        t_condensate=t_condensate,
        p=p,
        duty=duty,
        dh_specific=dh_molar / water.molar_mass,
    )


@dataclass(frozen=True)
class CoolingWaterResult:
    """Cooling-water circulation for a cooling duty.

    Attributes:
        mass_flow: Circulating water (kg/s).
        molar_flow: Circulating water (mol/s).
        t_supply: Supply temperature (K).
        t_return: Return temperature (K).
        duty: Heat removed from the process (W, positive).
    """

    mass_flow: Array
    molar_flow: Array
    t_supply: Array
    t_return: Array
    duty: Array


jax.tree_util.register_dataclass(
    CoolingWaterResult,
    data_fields=["mass_flow", "molar_flow", "t_supply", "t_return", "duty"],
    meta_fields=[],
)


def cooling_water(
    duty: ArrayLike,
    *,
    t_supply: ArrayLike = 303.15,
    t_return: ArrayLike = 318.15,
    pressure: ArrayLike = 4e5,
) -> CoolingWaterResult:
    """Cooling-water flow to absorb a duty over a supply/return temperature rise.

    Enthalpies of the liquid water are taken from IAPWS-95 at the circuit
    pressure (no constant-``cp`` approximation, however small the difference).

    Args:
        duty: Heat to remove from the process (W); the sign is ignored.
        t_supply: Cooling-water supply temperature (K).
        t_return: Cooling-water return temperature (K); must exceed supply.
        pressure: Circuit pressure (Pa).

    Returns:
        A `CoolingWaterResult`; differentiable in every argument.
    """
    water = reference_fluid("water")
    duty = jnp.abs(jnp.asarray(duty, dtype=float))
    t_supply = jnp.asarray(t_supply, dtype=float)
    t_return = jnp.asarray(t_return, dtype=float)
    p = jnp.asarray(pressure, dtype=float)
    dh = (
        state_tp(water, t_return, p, phase="liquid").h
        - state_tp(water, t_supply, p, phase="liquid").h
    )
    molar_flow = duty / dh
    return CoolingWaterResult(
        mass_flow=molar_flow * water.molar_mass,
        molar_flow=molar_flow,
        t_supply=t_supply,
        t_return=t_return,
        duty=duty,
    )


@dataclass(frozen=True)
class SteamTurbineResult:
    """Expansion of steam through a turbine with an isentropic efficiency.

    Attributes:
        power: Shaft power extracted (W, positive).
        mass_flow: Steam flow (kg/s).
        t_out: Outlet temperature (K).
        q_out: Outlet vapor quality (``nan`` if the outlet is single-phase).
        h_in: Inlet molar enthalpy (J/mol).
        h_out: Outlet molar enthalpy (J/mol).
        h_out_isentropic: Isentropic outlet molar enthalpy (J/mol).
        two_phase: Whether the outlet sits inside the dome (wet steam).
    """

    power: Array
    mass_flow: Array
    t_out: Array
    q_out: Array
    h_in: Array
    h_out: Array
    h_out_isentropic: Array
    two_phase: Array


jax.tree_util.register_dataclass(
    SteamTurbineResult,
    data_fields=[
        "power",
        "mass_flow",
        "t_out",
        "q_out",
        "h_in",
        "h_out",
        "h_out_isentropic",
        "two_phase",
    ],
    meta_fields=[],
)


def steam_turbine(
    mass_flow: ArrayLike,
    *,
    p_in: ArrayLike,
    t_in: ArrayLike,
    p_out: ArrayLike,
    isentropic_efficiency: ArrayLike = 0.75,
) -> SteamTurbineResult:
    """Expand steam from ``(p_in, t_in)`` to ``p_out`` and extract shaft work.

    The ideal outlet comes from an isentropic ``(P, s)`` resolution of
    IAPWS-95 (wet outlets handled by the two-phase dome logic of
    `fugacio.thermo.helmholtz.state_ps`); the real outlet enthalpy is
    ``h_in - eta * (h_in - h_s)``. Gradients flow through both state solves,
    so turbine power can be differentiated with respect to throttle pressure,
    inlet superheat, or backpressure -- the classic Rankine design knobs.

    Args:
        mass_flow: Steam flow (kg/s).
        p_in: Inlet pressure (Pa).
        t_in: Inlet temperature (K); must be superheated vapor.
        p_out: Outlet (back)pressure (Pa).
        isentropic_efficiency: Fraction of the ideal enthalpy drop extracted.

    Returns:
        A `SteamTurbineResult`; differentiable in every argument.
    """
    water = reference_fluid("water")
    mass_flow = jnp.asarray(mass_flow, dtype=float)
    inlet = state_tp(water, t_in, p_in, phase="vapor")
    ideal = state_ps(water, p_out, inlet.s)
    h_out = inlet.h - jnp.asarray(isentropic_efficiency, dtype=float) * (inlet.h - ideal.h)
    outlet = state_ph(water, p_out, h_out)
    molar_flow = mass_flow / water.molar_mass
    return SteamTurbineResult(
        power=molar_flow * (inlet.h - h_out),
        mass_flow=mass_flow,
        t_out=outlet.t,
        q_out=outlet.q,
        h_in=inlet.h,
        h_out=h_out,
        h_out_isentropic=ideal.h,
        two_phase=outlet.two_phase,
    )


def saturated_steam_temperature(pressure: ArrayLike) -> Array:
    """Saturation temperature (K) of steam at ``pressure`` (Pa) from IAPWS-95.

    The driving-force side of reboiler design: pinch margins against a
    header's condensing temperature, differentiable in the pressure.
    """
    water = reference_fluid("water")
    return saturation_state(water, p=jnp.asarray(pressure, dtype=float)).t


def steam_quality_after_letdown(
    p_supply: ArrayLike, p_use: ArrayLike, *, superheat: ArrayLike = 0.0
) -> Array:
    """Vapor quality after an isenthalpic letdown from one header to another.

    Saturated (or slightly superheated) steam throttled across a letdown valve
    flashes to the lower header pressure at constant enthalpy; the result is
    the outlet quality (> 1 is impossible -- ``nan`` marks a superheated,
    single-phase outlet, matching `fugacio.thermo.helmholtz.FluidState`).
    """
    water = reference_fluid("water")
    p_supply = jnp.asarray(p_supply, dtype=float)
    sat = saturation_state(water, p=p_supply)
    h_in = jnp.where(
        jnp.asarray(superheat, dtype=float) > 0.0,
        state_tp(water, sat.t + jnp.asarray(superheat, dtype=float), p_supply, phase="vapor").h,
        sat.h_vapor,
    )
    return state_ph(water, jnp.asarray(p_use, dtype=float), h_in).q


def condensate_flash_fraction(p_trap: ArrayLike, p_flash: ArrayLike) -> Array:
    """Fraction of saturated condensate that flashes when let down in pressure.

    Saturated liquid at ``p_trap`` flashed isenthalpically to ``p_flash``
    (a flash-steam recovery drum); returns the vapor fraction.
    """
    water = reference_fluid("water")
    sat = saturation_state(water, p=jnp.asarray(p_trap, dtype=float))
    return state_ph(water, jnp.asarray(p_flash, dtype=float), sat.h_liquid).q


def steam_enthalpy(
    pressure: ArrayLike, *, quality: ArrayLike | None = None, temperature: ArrayLike | None = None
) -> Array:
    """Molar enthalpy (J/mol) of steam/water at a header condition.

    Provide ``quality`` for a state on the dome or ``temperature`` for a
    single-phase state (exactly one of the two).
    """
    water = reference_fluid("water")
    p = jnp.asarray(pressure, dtype=float)
    if (quality is None) == (temperature is None):
        raise ValueError("specify exactly one of quality or temperature")
    if quality is not None:
        return state_tq(water, saturation_state(water, p=p).t, jnp.asarray(quality)).h
    return state_tp(water, jnp.asarray(temperature, dtype=float), p).h


__all__ = [
    "STEAM_LEVELS",
    "CoolingWaterResult",
    "SteamHeatingResult",
    "SteamTurbineResult",
    "condensate_flash_fraction",
    "cooling_water",
    "saturated_steam_temperature",
    "steam_enthalpy",
    "steam_heating",
    "steam_quality_after_letdown",
    "steam_turbine",
]
