"""Copilot tools for the heat-integration / pinch-analysis layer.

These expose `fugacio.sim.integration` to the LLM design agent as
deterministic, JSON-in/JSON-out calculations: compute the minimum-utility and
pinch targets for a set of process streams, return the composite / grand
composite curves, find the cost-optimal minimum approach temperature, and
synthesise a heat-exchanger network by the pinch design method. They are kept in
their own module (rather than the already-large `fugacio.copilot.tools`) and
folded into the registry there.

A stream is supplied as ``{"t_supply", "t_target"}`` plus either a heat-capacity
flowrate ``"cp"`` (W/K) or a ``"duty"`` (W), and an optional film coefficient
``"h"``; everything else follows from the differentiable targets.
"""

from __future__ import annotations

from typing import Any

from fugacio.sim.integration import (
    DEFAULT_FILM_COEFFICIENT,
    area_target,
    composite_curves,
    grand_composite_curve,
    make_stream,
    optimal_dt_min,
    pinch_analysis,
    synthesize_network,
    total_annual_cost_target,
    units_target,
)

JsonDict = dict[str, Any]


def _build_streams(streams: list[JsonDict], default_h: float) -> list[Any]:
    """Turn a list of stream dicts into `HeatStream` objects."""
    if not streams:
        raise ValueError("provide at least one stream")
    out = []
    for i, s in enumerate(streams):
        t_supply = float(s["t_supply"])
        t_target = float(s["t_target"])
        h = float(s.get("h", default_h))
        name = str(s.get("name", f"S{i + 1}"))
        if s.get("cp") is not None:
            cp = float(s["cp"])
        elif s.get("duty") is not None:
            span = abs(t_supply - t_target)
            cp = float(s["duty"]) / span if span > 0.0 else 0.0
        else:
            raise ValueError(f"stream {name!r} needs either 'cp' or 'duty'")
        out.append(make_stream(t_supply, t_target, cp, h=h, name=name))
    return out


def _thin(values: list[float], points: int) -> list[float]:
    n = len(values)
    if n <= points:
        return [float(v) for v in values]
    idx = [round(i * (n - 1) / (points - 1)) for i in range(points)]
    return [float(values[j]) for j in idx]


def _heat_integration_targets(
    streams: list[JsonDict],
    dt_min: float,
    film_coefficient: float = DEFAULT_FILM_COEFFICIENT,
    hot_utility: str = "hp_steam",
    cold_utility: str = "cooling_water",
    area_cost_b: float = 1200.0,
    area_cost_c: float = 0.6,
) -> JsonDict:
    """Minimum-utility, pinch, area, unit-count and total-annual-cost targets."""
    ss = _build_streams(streams, film_coefficient)
    dt = float(dt_min)
    res = pinch_analysis(ss, dt)
    ut = units_target(ss, dt)
    area = float(area_target(ss, dt))
    tac = total_annual_cost_target(
        ss,
        dt,
        hot_utility=hot_utility,
        cold_utility=cold_utility,
        area_cost=(0.0, area_cost_b, area_cost_c),
    )
    return {
        "dt_min": dt,
        "hot_utility_w": float(res.hot_utility),
        "cold_utility_w": float(res.cold_utility),
        "heat_recovery_w": float(res.heat_recovery),
        "pinch": {
            "exists": bool(res.has_pinch),
            "hot_temperature_k": float(res.hot_pinch_temperature),
            "cold_temperature_k": float(res.cold_pinch_temperature),
        },
        "minimum_units": int(ut.units),
        "area_target_m2": area,
        "capital_target_usd": float(tac.capital),
        "utility_cost_usd_yr": float(tac.utility_cost),
        "total_annual_cost_usd_yr": float(tac.total_annual_cost),
    }


def _composite_curves(
    streams: list[JsonDict],
    dt_min: float,
    film_coefficient: float = DEFAULT_FILM_COEFFICIENT,
    points: int = 41,
) -> JsonDict:
    """Hot/cold composite curves and the grand composite curve for plotting."""
    ss = _build_streams(streams, film_coefficient)
    dt = float(dt_min)
    cc = composite_curves(ss, dt)
    gcc = grand_composite_curve(ss, dt)
    return {
        "dt_min": dt,
        "minimum_approach_k": float(cc.min_approach),
        "hot_composite": {
            "temperature_k": _thin([float(v) for v in cc.hot_t], points),
            "enthalpy_w": _thin([float(v) for v in cc.hot_h], points),
        },
        "cold_composite": {
            "temperature_k": _thin([float(v) for v in cc.cold_t], points),
            "enthalpy_w": _thin([float(v) for v in cc.cold_h], points),
        },
        "grand_composite": {
            "shifted_temperature_k": _thin([float(v) for v in gcc.shifted_temperature], points),
            "net_heat_flow_w": _thin([float(v) for v in gcc.net_heat_flow], points),
        },
    }


def _optimum_dt_min(
    streams: list[JsonDict],
    film_coefficient: float = DEFAULT_FILM_COEFFICIENT,
    dt_min_low: float = 1.0,
    dt_min_high: float = 60.0,
    hot_utility: str = "hp_steam",
    cold_utility: str = "cooling_water",
    area_cost_b: float = 1200.0,
    area_cost_c: float = 0.6,
) -> JsonDict:
    """Cost-optimal minimum approach temperature (supertargeting) and its breakdown."""
    ss = _build_streams(streams, film_coefficient)
    opt = optimal_dt_min(
        ss,
        bounds=(float(dt_min_low), float(dt_min_high)),
        hot_utility=hot_utility,
        cold_utility=cold_utility,
        area_cost=(0.0, area_cost_b, area_cost_c),
    )
    t = opt.target
    return {
        "optimal_dt_min": float(opt.dt_min),
        "total_annual_cost_usd_yr": float(opt.total_annual_cost),
        "hot_utility_w": float(t.hot_utility),
        "cold_utility_w": float(t.cold_utility),
        "area_target_m2": float(t.area),
        "capital_target_usd": float(t.capital),
        "utility_cost_usd_yr": float(t.utility_cost),
        "annualized_capital_usd_yr": float(t.annualized_capital),
    }


def _heat_exchanger_network(
    streams: list[JsonDict],
    dt_min: float,
    film_coefficient: float = DEFAULT_FILM_COEFFICIENT,
) -> JsonDict:
    """Synthesise a heat-exchanger network by the pinch design method and verify it."""
    ss = _build_streams(streams, film_coefficient)
    net = synthesize_network(ss, float(dt_min))
    return {
        "dt_min": float(net.dt_min),
        "feasible": bool(net.feasible),
        "achieves_minimum_utilities": bool(net.achieves_mer),
        "hot_utility_w": float(net.hot_utility),
        "cold_utility_w": float(net.cold_utility),
        "number_of_units": int(net.n_units),
        "total_area_m2": float(net.total_area),
        "minimum_approach_k": float(net.min_approach),
        "exchangers": [
            {
                "kind": e.kind,
                "hot": e.hot,
                "cold": e.cold,
                "duty_w": float(e.duty),
                "area_m2": float(e.area),
                "approach_a_k": float(e.dt_a),
                "approach_b_k": float(e.dt_b),
            }
            for e in net.exchangers
        ],
    }


def integration_tool_specs() -> list[Any]:
    """ToolSpecs for the heat-integration layer (folded into ``default_registry``)."""
    from fugacio.copilot.tools import ToolSpec

    stream_schema = {
        "type": "array",
        "description": "Process streams to integrate.",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "t_supply": {"type": "number", "description": "Supply temperature (K)"},
                "t_target": {"type": "number", "description": "Target temperature (K)"},
                "cp": {"type": "number", "description": "Heat-capacity flowrate CP (W/K)"},
                "duty": {"type": "number", "description": "Duty (W); alternative to cp"},
                "h": {"type": "number", "description": "Film coefficient (W/m^2/K)"},
            },
            "required": ["t_supply", "t_target"],
        },
    }
    return [
        ToolSpec(
            name="heat_integration_targets",
            description=(
                "Compute pinch-analysis targets for a set of hot and cold process "
                "streams: the minimum hot/cold utility duties, the pinch "
                "temperatures, the maximum heat recovery, the minimum number of "
                "exchanger units, the heat-transfer area target, and the resulting "
                "capital and total annual cost. Streams give t_supply/t_target plus "
                "cp or duty."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "streams": stream_schema,
                    "dt_min": {"type": "number", "description": "Minimum approach temperature (K)"},
                    "film_coefficient": {
                        "type": "number",
                        "description": "Default film coefficient (W/m^2/K)",
                    },
                    "hot_utility": {"type": "string", "description": "Hot-utility key"},
                    "cold_utility": {"type": "string", "description": "Cold-utility key"},
                },
                "required": ["streams", "dt_min"],
            },
            run=_heat_integration_targets,
        ),
        ToolSpec(
            name="composite_curves",
            description=(
                "Return the hot and cold composite curves and the grand composite "
                "curve (temperature-enthalpy data) for a set of process streams at "
                "a given minimum approach temperature -- the diagrams behind utility "
                "selection and heat-recovery targeting."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "streams": stream_schema,
                    "dt_min": {"type": "number", "description": "Minimum approach temperature (K)"},
                    "film_coefficient": {"type": "number"},
                    "points": {"type": "integer", "description": "Samples per curve to return"},
                },
                "required": ["streams", "dt_min"],
            },
            run=_composite_curves,
        ),
        ToolSpec(
            name="optimum_dt_min",
            description=(
                "Find the minimum approach temperature that minimises total annual "
                "cost (the capital-energy trade-off / supertargeting), returning the "
                "optimal dt_min and the utility, area, capital and cost breakdown "
                "there."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "streams": stream_schema,
                    "film_coefficient": {"type": "number"},
                    "dt_min_low": {"type": "number", "description": "Lower search bound (K)"},
                    "dt_min_high": {"type": "number", "description": "Upper search bound (K)"},
                    "hot_utility": {"type": "string"},
                    "cold_utility": {"type": "string"},
                },
                "required": ["streams"],
            },
            run=_optimum_dt_min,
        ),
        ToolSpec(
            name="heat_exchanger_network",
            description=(
                "Synthesise a minimum-utility heat-exchanger network for the process "
                "streams by the pinch design method (tick-off with CP feasibility) "
                "and verify it: returns the exchangers (process matches, heaters, "
                "coolers) with their duties and areas, whether every approach "
                "respects dt_min, and whether the design hits the minimum-utility "
                "targets."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "streams": stream_schema,
                    "dt_min": {"type": "number", "description": "Minimum approach temperature (K)"},
                    "film_coefficient": {"type": "number"},
                },
                "required": ["streams", "dt_min"],
            },
            run=_heat_exchanger_network,
        ),
    ]


__all__ = ["integration_tool_specs"]
