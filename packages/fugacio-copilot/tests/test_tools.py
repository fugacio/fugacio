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


def test_unknown_tool_raises() -> None:
    with pytest.raises(KeyError):
        call_tool("not_a_tool", {})


def test_registry_is_overridable() -> None:
    registry = default_registry()
    del registry["flash_drum"]
    with pytest.raises(KeyError):
        call_tool("flash_drum", {}, registry)
