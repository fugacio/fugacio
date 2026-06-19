"""Copilot tools for the PC-SAFT molecular-based equation of state."""

import pytest

from fugacio.copilot import call_tool, tool_schemas


def test_saft_tools_are_registered() -> None:
    names = {s["name"] for s in tool_schemas()}
    assert names >= {
        "saft_flash",
        "saft_density",
        "saft_saturation_pressure",
        "saft_bubble_pressure",
        "saft_residual_enthalpy",
    }


def test_saft_tool_schemas_are_well_formed() -> None:
    by_name = {s["name"]: s for s in tool_schemas()}
    flash = by_name["saft_flash"]
    assert flash["parameters"]["type"] == "object"
    assert set(flash["parameters"]["required"]) == {"components", "z", "temperature", "pressure"}
    assert flash["description"]


def test_saft_saturation_pressure_of_water_near_one_atmosphere() -> None:
    out = call_tool("saft_saturation_pressure", {"component": "water", "temperature": 373.15})
    assert out["component"] == "water"
    assert out["psat_pa"] == pytest.approx(101325.0, rel=0.08)


def test_saft_flash_returns_a_valid_two_phase_split() -> None:
    out = call_tool(
        "saft_flash",
        {
            "components": ["propane", "n-butane"],
            "z": [0.5, 0.5],
            "temperature": 320.0,
            "pressure": 8e5,
        },
    )
    assert 0.0 < out["vapor_fraction"] < 1.0
    assert sum(out["liquid_composition"]) == pytest.approx(1.0, abs=1e-6)
    assert sum(out["vapor_composition"]) == pytest.approx(1.0, abs=1e-6)
    # The more volatile propane concentrates in the vapour.
    assert out["vapor_composition"][0] > out["liquid_composition"][0]


def test_saft_density_of_liquid_water_is_reasonable() -> None:
    out = call_tool(
        "saft_density",
        {"components": ["water"], "x": [1.0], "temperature": 298.15, "pressure": 1e5},
    )
    assert 850.0 < out["mass_density_kg_m3"] < 1050.0
    assert 0.0 < out["compressibility_factor"] < 0.05


def test_saft_bubble_pressure_orders_vapor_enrichment() -> None:
    out = call_tool(
        "saft_bubble_pressure",
        {"components": ["propane", "n-butane"], "x": [0.4, 0.6], "temperature": 320.0},
    )
    assert out["bubble_pressure_pa"] > 0.0
    assert sum(out["vapor_composition"]) == pytest.approx(1.0, abs=1e-6)
    assert out["vapor_composition"][0] > 0.4


def test_saft_residual_enthalpy_is_negative_for_a_liquid() -> None:
    out = call_tool(
        "saft_residual_enthalpy",
        {
            "components": ["propane", "n-butane"],
            "x": [0.5, 0.5],
            "temperature": 300.0,
            "pressure": 20e5,
            "phase": "liquid",
        },
    )
    # Attractive interactions dominate in the liquid: departure enthalpy < 0.
    assert out["residual_enthalpy_j_mol"] < 0.0
    assert 0.0 < out["compressibility_factor"] < 0.2
