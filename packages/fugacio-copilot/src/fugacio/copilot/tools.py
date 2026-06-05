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

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from fugacio.sim import (
    Stream,
    compressor,
    flash_drum,
    heater,
    pump,
    relative_volatility,
    shortcut_column,
    solve_column,
    turbine,
    valve,
)
from fugacio.thermo import (
    PR,
    bubble_pressure_eos,
    component_arrays,
    flash_pt,
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


def _stream(components: list[str], z: list[float], flow: float, t: float, p: float) -> Stream:
    return Stream.from_fractions(tuple(components), jnp.asarray(z), flow, t, p)


def _heat_exchanger(
    components: list[str],
    z: list[float],
    flow: float,
    temperature: float,
    pressure: float,
    t_out: float | None = None,
    duty: float | None = None,
) -> JsonDict:
    feed = _stream(components, z, flow, temperature, pressure)
    res = heater(feed, t_out=t_out, duty=duty)
    return {
        "outlet_temperature_k": float(res.outlet.t),
        "duty_w": float(res.duty),
        "outlet_pressure_pa": float(res.outlet.p),
    }


def _compressor(
    components: list[str],
    z: list[float],
    flow: float,
    temperature: float,
    pressure: float,
    pressure_out: float,
    efficiency: float = 0.75,
    machine: str = "compressor",
) -> JsonDict:
    feed = _stream(components, z, flow, temperature, pressure)
    model = turbine if machine == "turbine" else compressor
    res = model(feed, pressure_out, efficiency=efficiency)
    return {
        "outlet_temperature_k": float(res.outlet.t),
        "outlet_pressure_pa": float(res.outlet.p),
        "work_w": float(res.work),
        "ideal_work_w": float(res.ideal_work),
    }


def _pump(
    components: list[str],
    z: list[float],
    flow: float,
    temperature: float,
    pressure: float,
    pressure_out: float,
    efficiency: float = 0.75,
) -> JsonDict:
    feed = _stream(components, z, flow, temperature, pressure)
    res = pump(feed, pressure_out, efficiency=efficiency)
    return {
        "outlet_temperature_k": float(res.outlet.t),
        "outlet_pressure_pa": float(res.outlet.p),
        "work_w": float(res.work),
    }


def _valve(
    components: list[str],
    z: list[float],
    flow: float,
    temperature: float,
    pressure: float,
    pressure_out: float,
) -> JsonDict:
    feed = _stream(components, z, flow, temperature, pressure)
    out = valve(feed, pressure_out)
    return {
        "outlet_temperature_k": float(out.t),
        "outlet_pressure_pa": float(out.p),
        "temperature_drop_k": float(temperature - out.t),
    }


def _distillation_recoveries(
    alpha: list[float], lk: int, hk: int, lk_rec: float, hk_rec: float
) -> list[float]:
    """Per-component recovery to the distillate via the key recoveries and volatility."""
    a_lk, a_hk = alpha[lk], alpha[hk]
    recoveries = []
    for i, a_i in enumerate(alpha):
        if i == lk:
            recoveries.append(lk_rec)
        elif i == hk:
            recoveries.append(hk_rec)
        elif a_i >= a_lk:
            recoveries.append(0.999)
        elif a_i <= a_hk:
            recoveries.append(0.001)
        else:
            frac = (a_i - a_hk) / (a_lk - a_hk)
            recoveries.append(hk_rec + frac * (lk_rec - hk_rec))
    return recoveries


def _shortcut_distillation(
    components: list[str],
    z: list[float],
    feed_flow: float,
    temperature: float,
    pressure: float,
    light_key: int,
    heavy_key: int,
    lk_recovery: float = 0.99,
    hk_recovery: float = 0.01,
    q: float = 1.0,
) -> JsonDict:
    arr = component_arrays(components)
    z_arr = jnp.asarray(z)
    alpha = relative_volatility(
        PR, temperature, pressure, z_arr, arr["tc"], arr["pc"], arr["omega"], ref=heavy_key
    )
    recoveries = _distillation_recoveries(
        [float(a) for a in alpha], light_key, heavy_key, lk_recovery, hk_recovery
    )
    rec = jnp.asarray(recoveries)
    d = rec * feed_flow * z_arr
    b = (1.0 - rec) * feed_flow * z_arr
    res = shortcut_column(z_arr, d, b, alpha, q, light_key, heavy_key)
    return {
        "minimum_stages": float(res.n_min),
        "minimum_reflux": float(res.r_min),
        "reflux_ratio": float(res.r),
        "actual_stages": float(res.n_stages),
        "stages_above_feed": float(res.feed_stage),
        "relative_volatilities": [float(a) for a in alpha],
    }


def _rigorous_distillation(
    components: list[str],
    z: list[float],
    feed_flow: float,
    feed_temperature: float,
    pressure: float,
    n_stages: int,
    feed_stage: int,
    reflux: float,
    distillate_rate: float,
    q: float = 1.0,
) -> JsonDict:
    feed = _stream(components, z, feed_flow, feed_temperature, pressure)
    res = solve_column(feed, n_stages, feed_stage, reflux, distillate_rate, q=q)
    return {
        "distillate": {
            "flow_mol_s": float(res.distillate.total),
            "composition": [float(v) for v in res.distillate.z],
            "temperature_k": float(res.distillate.t),
        },
        "bottoms": {
            "flow_mol_s": float(res.bottoms.total),
            "composition": [float(v) for v in res.bottoms.z],
            "temperature_k": float(res.bottoms.t),
        },
        "condenser_duty_w": float(res.condenser_duty),
        "reboiler_duty_w": float(res.reboiler_duty),
        "top_temperature_k": float(res.t[0]),
        "bottom_temperature_k": float(res.t[-1]),
    }


def _safeguarded_newton(
    f: Callable[[float], Any], lo: float, hi: float, iters: int = 40, tol: float = 1e-7
) -> float:
    """Find an increasing scalar's root in ``[lo, hi]`` by gradient (Newton) + bisection.

    Uses ``jax.value_and_grad`` for the Newton step and falls back to bisection when
    the step leaves the bracket or the slope is unusable -- robust and still
    gradient-driven where the function is well behaved.
    """
    lo, hi = float(lo), float(hi)
    x = 0.5 * (lo + hi)
    for _ in range(iters):
        val, grad = jax.value_and_grad(f)(x)
        val, grad = float(val), float(grad)
        if val > 0.0:
            hi = x
        else:
            lo = x
        if abs(val) < tol:
            break
        step_ok = grad != 0.0 and math.isfinite(grad)
        x_newton = x - val / grad if step_ok else math.inf
        x = x_newton if (step_ok and lo < x_newton < hi) else 0.5 * (lo + hi)
    return x


def _optimize_flash_temperature(
    components: list[str],
    z: list[float],
    pressure: float,
    target_vapor_fraction: float,
    t_min: float = 100.0,
    t_max: float = 800.0,
) -> JsonDict:
    arr = component_arrays(components)
    z_arr = jnp.asarray(z)

    def beta_of_t(t: float) -> Any:
        result = flash_pt(PR, t, pressure, z_arr, arr["tc"], arr["pc"], arr["omega"])
        return result.beta

    t_opt = _safeguarded_newton(lambda t: beta_of_t(t) - target_vapor_fraction, t_min, t_max)
    return {
        "temperature_k": t_opt,
        "achieved_vapor_fraction": float(beta_of_t(t_opt)),
        "target_vapor_fraction": target_vapor_fraction,
    }


def _optimize_column_reflux(
    components: list[str],
    z: list[float],
    feed_flow: float,
    feed_temperature: float,
    pressure: float,
    n_stages: int,
    feed_stage: int,
    distillate_rate: float,
    light_key: int,
    target_purity: float,
    q: float = 1.0,
    iters: int = 8,
) -> JsonDict:
    feed = _stream(components, z, feed_flow, feed_temperature, pressure)

    def purity_of_reflux(reflux: float) -> Any:
        res = solve_column(feed, n_stages, feed_stage, reflux, distillate_rate, q=q)
        return res.distillate.z[light_key]

    reflux_opt = _safeguarded_newton(
        lambda r: purity_of_reflux(r) - target_purity, 0.2, 25.0, iters=iters
    )
    return {
        "reflux_ratio": reflux_opt,
        "achieved_purity": float(purity_of_reflux(reflux_opt)),
        "target_purity": target_purity,
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
        ToolSpec(
            name="heat_exchanger",
            description=(
                "Heat or cool a stream. Provide exactly one of t_out (target outlet "
                "temperature, K) or duty (signed heat added, W); returns the other."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "flow": {"type": "number"},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "t_out": {"type": "number", "description": "Target outlet temperature (K)"},
                    "duty": {"type": "number", "description": "Signed heat added (W)"},
                },
                "required": ["components", "z", "flow", "temperature", "pressure"],
            },
            run=_heat_exchanger,
        ),
        ToolSpec(
            name="compressor",
            description=(
                "Isentropic-efficiency compressor or turbine. Set machine='turbine' to "
                "expand; returns outlet temperature and shaft work."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "flow": {"type": "number"},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "pressure_out": {"type": "number"},
                    "efficiency": {"type": "number"},
                    "machine": {"type": "string", "enum": ["compressor", "turbine"]},
                },
                "required": ["components", "z", "flow", "temperature", "pressure", "pressure_out"],
            },
            run=_compressor,
        ),
        ToolSpec(
            name="pump",
            description=(
                "Pump an incompressible liquid to a higher pressure; returns work and outlet T."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "flow": {"type": "number"},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "pressure_out": {"type": "number"},
                    "efficiency": {"type": "number"},
                },
                "required": ["components", "z", "flow", "temperature", "pressure", "pressure_out"],
            },
            run=_pump,
        ),
        ToolSpec(
            name="valve",
            description=(
                "Isenthalpic (Joule-Thomson) pressure letdown; returns the outlet temperature."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "flow": {"type": "number"},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "pressure_out": {"type": "number"},
                },
                "required": ["components", "z", "flow", "temperature", "pressure", "pressure_out"],
            },
            run=_valve,
        ),
        ToolSpec(
            name="shortcut_distillation",
            description=(
                "Fenske-Underwood-Gilliland shortcut design from key recoveries: returns "
                "minimum stages, minimum reflux, actual stages, and feed location."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "feed_flow": {"type": "number"},
                    "temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "light_key": {"type": "integer", "description": "Index of the light key"},
                    "heavy_key": {"type": "integer", "description": "Index of the heavy key"},
                    "lk_recovery": {"type": "number", "description": "Light-key recovery to top"},
                    "hk_recovery": {"type": "number", "description": "Heavy-key recovery to top"},
                    "q": {"type": "number", "description": "Feed quality (1 = sat. liquid)"},
                },
                "required": [
                    "components",
                    "z",
                    "feed_flow",
                    "temperature",
                    "pressure",
                    "light_key",
                    "heavy_key",
                ],
            },
            run=_shortcut_distillation,
        ),
        ToolSpec(
            name="rigorous_distillation",
            description=(
                "Rigorous multistage column (Wang-Henke, constant molar overflow) with EOS "
                "K-values; returns product compositions, temperatures, and column duties."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "feed_flow": {"type": "number"},
                    "feed_temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "n_stages": {"type": "integer"},
                    "feed_stage": {"type": "integer"},
                    "reflux": {"type": "number"},
                    "distillate_rate": {"type": "number"},
                    "q": {"type": "number"},
                },
                "required": [
                    "components",
                    "z",
                    "feed_flow",
                    "feed_temperature",
                    "pressure",
                    "n_stages",
                    "feed_stage",
                    "reflux",
                    "distillate_rate",
                ],
            },
            run=_rigorous_distillation,
        ),
        ToolSpec(
            name="optimize_flash_temperature",
            description=(
                "Gradient-based solve for the flash temperature (K) that hits a target vapor "
                "fraction, differentiating the equilibrium flash."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "pressure": {"type": "number"},
                    "target_vapor_fraction": {"type": "number"},
                },
                "required": ["components", "z", "pressure", "target_vapor_fraction"],
            },
            run=_optimize_flash_temperature,
        ),
        ToolSpec(
            name="optimize_column_reflux",
            description=(
                "Gradient-based solve for the reflux ratio that achieves a target distillate "
                "purity of the light key, differentiating through the rigorous column."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "components": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "array", "items": {"type": "number"}},
                    "feed_flow": {"type": "number"},
                    "feed_temperature": {"type": "number"},
                    "pressure": {"type": "number"},
                    "n_stages": {"type": "integer"},
                    "feed_stage": {"type": "integer"},
                    "distillate_rate": {"type": "number"},
                    "light_key": {"type": "integer"},
                    "target_purity": {"type": "number"},
                    "q": {"type": "number"},
                },
                "required": [
                    "components",
                    "z",
                    "feed_flow",
                    "feed_temperature",
                    "pressure",
                    "n_stages",
                    "feed_stage",
                    "distillate_rate",
                    "light_key",
                    "target_purity",
                ],
            },
            run=_optimize_column_reflux,
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
