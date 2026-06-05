"""A tool registry exposing the Fugacio engine to an LLM design agent.

The copilot layer turns natural-language design goals into engineering
calculations. The bridge between a language model and the differentiable engine
is a set of *tools*: deterministic, JSON-in/JSON-out functions with machine-
readable schemas (the same shape OpenAI / Anthropic function-calling expects).

This module defines that registry over :mod:`fugacio.sim` and
:mod:`fugacio.thermo`. The tools are plain Python (floats and lists, not JAX
arrays) so they are trivial to serialise; the LLM-planning loop that selects and
sequences them lives behind the optional ``llm`` extra (:mod:`fugacio.copilot.agent`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from fugacio.sim import Stream, flash_drum
from fugacio.thermo import (
    PR,
    bubble_pressure_eos,
    component_arrays,
    get,
    names,
    psat_eos,
)

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    """A callable tool with an LLM-facing JSON schema.

    Attributes:
        name: Unique tool name.
        description: One-line description for the planner.
        parameters: JSON-schema object describing the arguments.
        run: The implementation, taking keyword arguments and returning a dict.
    """

    name: str
    description: str
    parameters: JsonDict
    run: Callable[..., JsonDict]


def _list_components() -> JsonDict:
    return {"components": names()}


def _component_properties(component: str) -> JsonDict:
    c = get(component)
    return {
        "name": c.name,
        "formula": c.formula,
        "molar_mass_g_mol": c.mw,
        "critical_temperature_k": c.tc,
        "critical_pressure_pa": c.pc,
        "acentric_factor": c.omega,
        "normal_boiling_point_k": c.tb,
    }


def _saturation_pressure(component: str, temperature: float) -> JsonDict:
    c = get(component)
    psat = float(psat_eos(PR, temperature, c.tc, c.pc, c.omega))
    return {"component": c.name, "temperature_k": temperature, "psat_pa": psat}


def _bubble_pressure(components: list[str], x: list[float], temperature: float) -> JsonDict:
    arr = component_arrays(components)
    p, y = bubble_pressure_eos(PR, temperature, jnp.asarray(x), arr["tc"], arr["pc"], arr["omega"])
    return {
        "temperature_k": temperature,
        "bubble_pressure_pa": float(p),
        "vapor_composition": [float(v) for v in y],
    }


def _flash(
    components: list[str],
    z: list[float],
    flow: float,
    temperature: float,
    pressure: float,
) -> JsonDict:
    feed = Stream.from_fractions(tuple(components), jnp.asarray(z), flow, temperature, pressure)
    vapor, liquid = flash_drum(feed, temperature, pressure)
    return {
        "vapor_fraction": float(vapor.total / feed.total),
        "vapor": {"flow_mol_s": float(vapor.total), "composition": [float(v) for v in vapor.z]},
        "liquid": {"flow_mol_s": float(liquid.total), "composition": [float(v) for v in liquid.z]},
    }


def default_registry() -> dict[str, ToolSpec]:
    """Return the built-in tool registry keyed by tool name."""
    specs = [
        ToolSpec(
            name="list_components",
            description="List the components available in the Fugacio database.",
            parameters={"type": "object", "properties": {}, "required": []},
            run=_list_components,
        ),
        ToolSpec(
            name="component_properties",
            description="Look up critical constants and basic properties of one component.",
            parameters={
                "type": "object",
                "properties": {"component": {"type": "string"}},
                "required": ["component"],
            },
            run=_component_properties,
        ),
        ToolSpec(
            name="saturation_pressure",
            description="Compute the pure-component saturation pressure (Pa) at a temperature.",
            parameters={
                "type": "object",
                "properties": {
                    "component": {"type": "string"},
                    "temperature": {"type": "number", "description": "Temperature in K"},
                },
                "required": ["component", "temperature"],
            },
            run=_saturation_pressure,
        ),
        ToolSpec(
            name="bubble_pressure",
            description="Bubble-point pressure and vapor composition of a liquid mixture.",
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "x": {"type": "array", "items": {"type": "number"}},
                    "temperature": {"type": "number"},
                },
                "required": ["components", "x", "temperature"],
            },
            run=_bubble_pressure,
        ),
        ToolSpec(
            name="flash_drum",
            description="Isothermal-isobaric flash of a feed; returns vapor/liquid products.",
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "flow": {"type": "number", "description": "Total feed flow (mol/s)"},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                },
                "required": ["components", "z", "flow", "temperature", "pressure"],
            },
            run=_flash,
        ),
    ]
    return {spec.name: spec for spec in specs}


def tool_schemas(registry: dict[str, ToolSpec] | None = None) -> list[JsonDict]:
    """Return the function-calling schemas for every tool in the registry."""
    registry = default_registry() if registry is None else registry
    return [
        {"name": s.name, "description": s.description, "parameters": s.parameters}
        for s in registry.values()
    ]


def call_tool(
    name: str,
    arguments: JsonDict,
    registry: dict[str, ToolSpec] | None = None,
) -> JsonDict:
    """Dispatch a tool call by name with a JSON argument dict.

    Raises:
        KeyError: if ``name`` is not a registered tool.
    """
    registry = default_registry() if registry is None else registry
    if name not in registry:
        raise KeyError(f"unknown tool {name!r}; available: {sorted(registry)}")
    return registry[name].run(**arguments)
