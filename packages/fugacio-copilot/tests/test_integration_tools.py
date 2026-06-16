"""Copilot tools for the heat-integration layer (targets, curves, optimum, network)."""

import pytest

from fugacio.copilot import call_tool, summarize_heat_integration, tool_schemas

# The textbook four-stream problem, expressed as tool-call JSON.
FOUR_STREAM = [
    {"name": "C1", "t_supply": 20.0, "t_target": 135.0, "cp": 2.0, "h": 1.0},
    {"name": "H1", "t_supply": 170.0, "t_target": 60.0, "cp": 3.0, "h": 1.0},
    {"name": "C2", "t_supply": 80.0, "t_target": 140.0, "cp": 4.0, "h": 1.0},
    {"name": "H2", "t_supply": 150.0, "t_target": 30.0, "cp": 1.5, "h": 1.0},
]


def test_integration_tools_are_registered() -> None:
    names = {s["name"] for s in tool_schemas()}
    assert names >= {
        "heat_integration_targets",
        "composite_curves",
        "optimum_dt_min",
        "heat_exchanger_network",
    }


def test_heat_integration_targets_four_stream() -> None:
    out = call_tool("heat_integration_targets", {"streams": FOUR_STREAM, "dt_min": 10.0})
    assert out["hot_utility_w"] == pytest.approx(20.0, abs=1e-6)
    assert out["cold_utility_w"] == pytest.approx(60.0, abs=1e-6)
    assert out["pinch"]["exists"]
    assert out["pinch"]["hot_temperature_k"] == pytest.approx(90.0, abs=1e-6)
    assert out["minimum_units"] == 7
    assert out["area_target_m2"] > 0.0
    assert out["total_annual_cost_usd_yr"] > 0.0
    # The Markdown report renders the headline targets.
    md = summarize_heat_integration(out)
    assert "Heat integration" in md
    assert "Pinch" in md


def test_targets_accept_duty_instead_of_cp() -> None:
    streams = [
        {"name": "H", "t_supply": 200.0, "t_target": 100.0, "duty": 200.0},
        {"name": "C", "t_supply": 80.0, "t_target": 180.0, "duty": 200.0},
    ]
    out = call_tool("heat_integration_targets", {"streams": streams, "dt_min": 20.0})
    # Equal-duty, equal-CP streams fully overlap: a threshold problem.
    assert not out["pinch"]["exists"]


def test_composite_curves_tool() -> None:
    out = call_tool("composite_curves", {"streams": FOUR_STREAM, "dt_min": 10.0, "points": 25})
    assert out["minimum_approach_k"] == pytest.approx(10.0, abs=1e-6)
    assert len(out["hot_composite"]["temperature_k"]) <= 25
    assert len(out["grand_composite"]["net_heat_flow_w"]) >= 2


def test_optimum_dt_min_tool() -> None:
    out = call_tool(
        "optimum_dt_min",
        {"streams": FOUR_STREAM, "dt_min_low": 1.0, "dt_min_high": 40.0, "area_cost_b": 800.0},
    )
    assert 1.0 <= out["optimal_dt_min"] <= 40.0
    assert out["total_annual_cost_usd_yr"] > 0.0


def test_heat_exchanger_network_tool() -> None:
    out = call_tool("heat_exchanger_network", {"streams": FOUR_STREAM, "dt_min": 10.0})
    assert out["feasible"]
    assert out["achieves_minimum_utilities"]
    assert out["hot_utility_w"] == pytest.approx(20.0, abs=1e-6)
    assert out["minimum_approach_k"] >= 10.0 - 1e-6
    assert len(out["exchangers"]) == out["number_of_units"]
    kinds = {e["kind"] for e in out["exchangers"]}
    assert kinds <= {"process", "heater", "cooler"}


def test_network_tool_rejects_streams_without_cp_or_duty() -> None:
    with pytest.raises(ValueError, match="needs either 'cp' or 'duty'"):
        call_tool(
            "heat_exchanger_network",
            {"streams": [{"name": "X", "t_supply": 100.0, "t_target": 50.0}], "dt_min": 10.0},
        )
