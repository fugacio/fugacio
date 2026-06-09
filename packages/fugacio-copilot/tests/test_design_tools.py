"""Optimization / economics / sensitivity tools added to the copilot registry.

Each tool is JSON-in/JSON-out and deterministic; these checks confirm the new
tools are registered, return the expected keys, and produce physically sensible
numbers (a hotter flash makes more vapour, installed cost exceeds purchased, TAC
combines capital and operating cost correctly).
"""

import pytest

from fugacio.copilot import call_tool, default_registry, tool_schemas

REGISTRY = default_registry()


def _call(name: str, **kwargs: object) -> dict:
    return call_tool(name, kwargs, REGISTRY)


def test_new_tools_registered_and_schemas_valid() -> None:
    expected = {
        "flash_sensitivity",
        "equipment_cost",
        "heat_exchanger_cost",
        "utility_cost",
        "annual_cost",
        "net_present_value",
        "size_column",
    }
    assert expected <= set(REGISTRY)
    # Every schema is a JSON-schema object with the required envelope.
    for schema in tool_schemas(REGISTRY):
        assert schema["parameters"]["type"] == "object"
        assert "properties" in schema["parameters"]


def test_flash_sensitivity_signs() -> None:
    out = _call(
        "flash_sensitivity",
        components=["propane", "n-butane", "n-pentane"],
        z=[0.4, 0.35, 0.25],
        flow=100.0,
        temperature=330.0,
        pressure=8e5,
    )
    assert 0.0 < out["vapor_fraction"] < 1.0
    assert out["d_vapor_fraction_dT_per_k"] > 0.0  # hotter -> more vapour
    assert out["d_vapor_fraction_dP_per_pa"] < 0.0  # higher pressure -> less vapour


def test_equipment_cost_material_and_install() -> None:
    cs = _call("equipment_cost", kind="heat_exchanger", size=100.0, material="CS")
    ss = _call("equipment_cost", kind="heat_exchanger", size=100.0, material="SS")
    assert cs["bare_module_usd"] > cs["purchased_usd"] > 0.0
    assert ss["bare_module_usd"] > cs["bare_module_usd"]


def test_heat_exchanger_cost_keys() -> None:
    out = _call("heat_exchanger_cost", duty=1.0e6, u=500.0, dt_hot=60.0, dt_cold=40.0)
    assert out["area_m2"] > 0.0
    assert out["bare_module_usd"] > 0.0


def test_utility_cost_ordering() -> None:
    steam = _call("utility_cost", duty=1e6, utility="hp_steam")
    water = _call("utility_cost", duty=1e6, utility="cooling_water")
    assert steam["annual_cost_usd"] > water["annual_cost_usd"] > 0.0


def test_annual_cost_and_npv() -> None:
    tac = _call("annual_cost", capex=1.0e6, opex=2.0e5)
    assert tac["total_annual_cost_usd_per_year"] == pytest.approx(
        tac["annualized_capital_usd_per_year"] + 2.0e5, rel=1e-9
    )
    pv = _call("net_present_value", cash_flows=[-1.0e6] + [2.5e5] * 10, discount_rate=0.1)
    assert pv["npv_usd"] > 0.0


def test_size_column_geometry() -> None:
    out = _call("size_column", vapor_molar_flow=50.0, temperature=350.0, pressure=2e5, n_stages=20)
    assert out["diameter_m"] > 0.0
    assert out["height_m"] == pytest.approx(20 * 0.6 + 4.0, abs=1e-6)
    assert out["installed_cost_usd"] == pytest.approx(
        out["shell_cost_usd"] + out["trays_cost_usd"], rel=1e-9
    )
