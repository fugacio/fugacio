"""Copilot tool registry: schemas, dispatch, and engine-backed results."""

import pytest

from fugacio.copilot import call_tool, default_registry, tool_schemas


def test_schemas_are_function_calling_shaped() -> None:
    schemas = tool_schemas()
    assert {s["name"] for s in schemas} >= {
        "list_components",
        "component_properties",
        "saturation_pressure",
        "bubble_pressure",
        "flash_drum",
    }
    for s in schemas:
        assert set(s) == {"name", "description", "parameters"}
        assert s["parameters"]["type"] == "object"


def test_list_and_lookup_components() -> None:
    listing = call_tool("list_components", {})
    assert "water" in listing["components"]
    props = call_tool("component_properties", {"component": "water"})
    assert props["critical_temperature_k"] == pytest.approx(647.1, rel=0.02)


def test_saturation_pressure_tool() -> None:
    result = call_tool("saturation_pressure", {"component": "propane", "temperature": 300.0})
    assert result["psat_pa"] > 0.0
    assert result["component"].lower() == "propane"


def test_flash_drum_tool_balances() -> None:
    result = call_tool(
        "flash_drum",
        {
            "components": ["methane", "propane", "n-pentane"],
            "z": [0.5, 0.3, 0.2],
            "flow": 100.0,
            "temperature": 320.0,
            "pressure": 20e5,
        },
    )
    assert 0.0 <= result["vapor_fraction"] <= 1.0
    total = result["vapor"]["flow_mol_s"] + result["liquid"]["flow_mol_s"]
    assert total == pytest.approx(100.0, rel=1e-6)


def test_new_engine_tools_are_registered() -> None:
    names = {s["name"] for s in tool_schemas()}
    assert names >= {
        "heat_exchanger",
        "compressor",
        "pump",
        "valve",
        "shortcut_distillation",
        "rigorous_distillation",
        "optimize_flash_temperature",
        "optimize_column_reflux",
    }


def test_heat_exchanger_temperature_and_duty_modes() -> None:
    base = {
        "components": ["methane", "propane"],
        "z": [0.5, 0.5],
        "flow": 100.0,
        "temperature": 300.0,
        "pressure": 5e5,
    }
    hot = call_tool("heat_exchanger", {**base, "t_out": 350.0})
    assert hot["duty_w"] > 0.0
    assert hot["outlet_temperature_k"] == pytest.approx(350.0)
    back = call_tool("heat_exchanger", {**base, "duty": hot["duty_w"]})
    assert back["outlet_temperature_k"] == pytest.approx(350.0, abs=0.3)


def test_compressor_tool_reports_work() -> None:
    result = call_tool(
        "compressor",
        {
            "components": ["methane", "propane"],
            "z": [0.5, 0.5],
            "flow": 100.0,
            "temperature": 300.0,
            "pressure": 5e5,
            "pressure_out": 20e5,
            "efficiency": 0.75,
        },
    )
    assert result["work_w"] > result["ideal_work_w"] > 0.0
    assert result["outlet_temperature_k"] > 300.0


def test_valve_tool_drops_pressure() -> None:
    result = call_tool(
        "valve",
        {
            "components": ["methane", "propane"],
            "z": [0.5, 0.5],
            "flow": 100.0,
            "temperature": 300.0,
            "pressure": 50e5,
            "pressure_out": 5e5,
        },
    )
    assert result["outlet_pressure_pa"] == pytest.approx(5e5)


def test_shortcut_distillation_tool() -> None:
    result = call_tool(
        "shortcut_distillation",
        {
            "components": ["propane", "n-butane"],
            "z": [0.5, 0.5],
            "feed_flow": 100.0,
            "temperature": 330.0,
            "pressure": 10e5,
            "light_key": 0,
            "heavy_key": 1,
            "lk_recovery": 0.98,
            "hk_recovery": 0.02,
        },
    )
    assert result["minimum_stages"] > 0.0
    assert result["actual_stages"] > result["minimum_stages"]
    assert result["minimum_reflux"] > 0.0


def test_rigorous_distillation_tool_balances_and_separates() -> None:
    result = call_tool(
        "rigorous_distillation",
        {
            "components": ["propane", "n-butane"],
            "z": [0.5, 0.5],
            "feed_flow": 100.0,
            "feed_temperature": 320.0,
            "pressure": 10e5,
            "n_stages": 10,
            "feed_stage": 5,
            "reflux": 2.0,
            "distillate_rate": 50.0,
        },
    )
    assert result["distillate"]["composition"][0] > 0.9  # propane overhead
    assert result["bottoms"]["composition"][1] > 0.9  # butane bottoms
    total = result["distillate"]["flow_mol_s"] + result["bottoms"]["flow_mol_s"]
    assert total == pytest.approx(100.0, rel=1e-5)
    assert result["reboiler_duty_w"] > 0.0


def test_optimize_flash_temperature_hits_target() -> None:
    result = call_tool(
        "optimize_flash_temperature",
        {
            "components": ["methane", "propane", "n-pentane"],
            "z": [0.3, 0.3, 0.4],
            "pressure": 20e5,
            "target_vapor_fraction": 0.5,
        },
    )
    assert result["achieved_vapor_fraction"] == pytest.approx(0.5, abs=1e-3)


def test_optimize_column_reflux_hits_target_purity() -> None:
    result = call_tool(
        "optimize_column_reflux",
        {
            "components": ["propane", "n-butane"],
            "z": [0.5, 0.5],
            "feed_flow": 100.0,
            "feed_temperature": 320.0,
            "pressure": 10e5,
            "n_stages": 8,
            "feed_stage": 4,
            "distillate_rate": 50.0,
            "light_key": 0,
            "target_purity": 0.85,
            "iters": 5,
        },
    )
    assert 0.2 <= result["reflux_ratio"] <= 25.0
    assert result["achieved_purity"] == pytest.approx(0.85, abs=0.05)


def test_unknown_tool_raises() -> None:
    with pytest.raises(KeyError):
        call_tool("not_a_tool", {})


def test_registry_is_overridable() -> None:
    registry = default_registry()
    del registry["flash_drum"]
    with pytest.raises(KeyError):
        call_tool("flash_drum", {}, registry)
