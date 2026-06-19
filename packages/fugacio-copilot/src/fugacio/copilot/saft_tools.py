"""Copilot tools for the PC-SAFT molecular-based equation of state.

These expose `fugacio.thermo.saft` (through the `fugacio.sim.saft_model_for`
bridge) to the LLM design agent as deterministic, JSON-in/JSON-out calculations:

* ``saft_flash``: an isothermal-isobaric vapour-liquid flash on PC-SAFT;
* ``saft_density``: molar/mass density and compressibility factor on a phase
  branch;
* ``saft_saturation_pressure``: a pure-component saturation pressure;
* ``saft_bubble_pressure``: a mixture bubble pressure and incipient vapour.

PC-SAFT is the method of choice for associating fluids (water, alcohols), so the
tools accept the same component names as the rest of the registry but route them
through the molecular-based model rather than a cubic EOS. Everything returned is
plain Python (floats / lists) for trivial serialisation.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from fugacio.sim import saft_model_for
from fugacio.thermo import component_arrays, saft_parameters_for, saft_residual_properties
from fugacio.thermo.saft import compressibility_factor, molar_density, psat_saft

JsonDict = dict[str, Any]


def _wilson_pressure(component: str, temperature: float) -> float:
    """Wilson saturation-pressure estimate (Pa), a seed for the PC-SAFT solve."""
    arr = component_arrays([component])
    tc = float(arr["tc"][0])
    pc = float(arr["pc"][0])
    omega = float(arr["omega"][0])
    return pc * float(jnp.exp(5.373 * (1.0 + omega) * (1.0 - tc / temperature)))


def _saft_flash(
    components: list[str],
    z: list[float],
    temperature: float,
    pressure: float,
) -> JsonDict:
    """Isothermal-isobaric two-phase flash on PC-SAFT."""
    model = saft_model_for(components)
    result = model.flash_pt(float(temperature), float(pressure), jnp.asarray(z, dtype=float))
    return {
        "components": list(components),
        "temperature_k": float(temperature),
        "pressure_pa": float(pressure),
        "vapor_fraction": float(result.beta),
        "liquid_composition": [float(v) for v in result.x],
        "vapor_composition": [float(v) for v in result.y],
        "k_values": [float(v) for v in result.k],
    }


def _saft_density(
    components: list[str],
    x: list[float],
    temperature: float,
    pressure: float = 101325.0,
    phase: str = "liquid",
) -> JsonDict:
    """Molar/mass density and compressibility factor of a mixture on a phase branch."""
    params = saft_parameters_for(components)
    arr = component_arrays(components)
    x_arr = jnp.asarray(x, dtype=float)
    t = float(temperature)
    p = float(pressure)
    rho = molar_density(params, t, p, x_arr, phase=phase)
    z = compressibility_factor(params, rho, t, x_arr)
    molar_mass = float(jnp.sum(x_arr * arr["mw"]))  # g/mol
    return {
        "components": list(components),
        "x": [float(v) for v in x],
        "temperature_k": t,
        "pressure_pa": p,
        "phase": phase,
        "molar_density_mol_m3": float(rho),
        "mass_density_kg_m3": float(rho) * molar_mass / 1000.0,
        "compressibility_factor": float(z),
    }


def _saft_saturation_pressure(
    component: str, temperature: float, pressure_guess: float | None = None
) -> JsonDict:
    """Pure-component saturation pressure (Pa) from PC-SAFT by equifugacity."""
    params = saft_parameters_for([component])
    t = float(temperature)
    guess = _wilson_pressure(component, t) if pressure_guess is None else float(pressure_guess)
    psat = float(psat_saft(params, t, guess))
    return {"component": component, "temperature_k": t, "psat_pa": psat}


def _saft_bubble_pressure(components: list[str], x: list[float], temperature: float) -> JsonDict:
    """Bubble pressure (Pa) and incipient vapour of a mixture from PC-SAFT."""
    model = saft_model_for(components)
    p, y = model.bubble_pressure(float(temperature), jnp.asarray(x, dtype=float))
    return {
        "components": list(components),
        "temperature_k": float(temperature),
        "x": [float(v) for v in x],
        "bubble_pressure_pa": float(p),
        "vapor_composition": [float(v) for v in y],
    }


def saft_residual_enthalpy(
    components: list[str],
    x: list[float],
    temperature: float,
    pressure: float,
    phase: str = "liquid",
) -> JsonDict:
    """Residual (departure) molar enthalpy/entropy of a mixture from PC-SAFT."""
    params = saft_parameters_for(components)
    res = saft_residual_properties(
        params, float(temperature), float(pressure), jnp.asarray(x), phase=phase
    )
    return {
        "components": list(components),
        "temperature_k": float(temperature),
        "pressure_pa": float(pressure),
        "phase": phase,
        "compressibility_factor": float(res.z),
        "residual_enthalpy_j_mol": float(res.enthalpy),
        "residual_entropy_j_mol_k": float(res.entropy),
        "residual_cp_j_mol_k": float(res.cp),
    }


def saft_tool_specs() -> list[Any]:
    """ToolSpecs for the PC-SAFT layer (folded into ``default_registry``)."""
    from fugacio.copilot.tools import ToolSpec

    components_schema = {"type": "array", "items": {"type": "string"}}
    fractions_schema = {"type": "array", "items": {"type": "number"}}
    return [
        ToolSpec(
            name="saft_flash",
            description=(
                "Isothermal-isobaric vapour-liquid flash on the PC-SAFT equation of "
                "state (the molecular-based method preferred for associating fluids "
                "such as water and alcohols). Returns the vapour fraction and the "
                "liquid/vapour compositions and K-values."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": components_schema,
                    "z": {**fractions_schema, "description": "Feed mole fractions."},
                    "temperature": {"type": "number", "description": "Temperature (K)."},
                    "pressure": {"type": "number", "description": "Pressure (Pa)."},
                },
                "required": ["components", "z", "temperature", "pressure"],
            },
            run=_saft_flash,
        ),
        ToolSpec(
            name="saft_density",
            description=(
                "Molar and mass density and the compressibility factor of a mixture "
                "from PC-SAFT on a chosen phase branch ('liquid', 'vapor', or "
                "'stable')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": components_schema,
                    "x": {**fractions_schema, "description": "Mole fractions."},
                    "temperature": {"type": "number", "description": "Temperature (K)."},
                    "pressure": {"type": "number", "description": "Pressure (Pa)."},
                    "phase": {
                        "type": "string",
                        "enum": ["liquid", "vapor", "stable"],
                        "description": "Density branch to return.",
                    },
                },
                "required": ["components", "x", "temperature"],
            },
            run=_saft_density,
        ),
        ToolSpec(
            name="saft_saturation_pressure",
            description=(
                "Pure-component saturation (vapour) pressure from PC-SAFT by "
                "equifugacity. A Wilson estimate seeds the solve when no guess is "
                "given."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "component": {"type": "string"},
                    "temperature": {"type": "number", "description": "Temperature (K)."},
                    "pressure_guess": {
                        "type": "number",
                        "description": "Optional initial pressure (Pa).",
                    },
                },
                "required": ["component", "temperature"],
            },
            run=_saft_saturation_pressure,
        ),
        ToolSpec(
            name="saft_bubble_pressure",
            description=(
                "Bubble-point pressure and incipient-vapour composition of a liquid "
                "mixture from PC-SAFT at a fixed temperature."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": components_schema,
                    "x": {**fractions_schema, "description": "Liquid mole fractions."},
                    "temperature": {"type": "number", "description": "Temperature (K)."},
                },
                "required": ["components", "x", "temperature"],
            },
            run=_saft_bubble_pressure,
        ),
        ToolSpec(
            name="saft_residual_enthalpy",
            description=(
                "Residual (departure) molar enthalpy, entropy and heat capacity, plus "
                "the compressibility factor, of a mixture from PC-SAFT on a phase "
                "branch."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": components_schema,
                    "x": {**fractions_schema, "description": "Mole fractions."},
                    "temperature": {"type": "number", "description": "Temperature (K)."},
                    "pressure": {"type": "number", "description": "Pressure (Pa)."},
                    "phase": {
                        "type": "string",
                        "enum": ["liquid", "vapor", "stable"],
                        "description": "Density branch to evaluate.",
                    },
                },
                "required": ["components", "x", "temperature", "pressure"],
            },
            run=saft_residual_enthalpy,
        ),
    ]


__all__ = ["saft_tool_specs"]
