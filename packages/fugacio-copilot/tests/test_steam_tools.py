"""Copilot tools backed by the reference Helmholtz EOS (steam tables et al.)."""

import pytest

from fugacio.copilot import call_tool, tool_schemas


def test_steam_tools_are_registered() -> None:
    names = {s["name"] for s in tool_schemas()}
    assert names >= {
        "steam_state",
        "reference_fluid_state",
        "reference_saturation",
        "steam_utility_requirements",
        "steam_turbine",
    }


def test_steam_state_by_temperature() -> None:
    # Superheated steam at 0.1 MPa / 150 C: v = 1.9364 m^3/kg in the tables.
    result = call_tool("steam_state", {"pressure": 1e5, "temperature": 423.15})
    assert result["density_kg_m3"] == pytest.approx(1.0 / 1.9364, rel=1e-3)
    assert result["two_phase"] is False
    assert result["quality"] is None
    assert result["viscosity_pa_s"] == pytest.approx(1.4e-5, rel=0.1)


def test_steam_state_by_quality_and_enthalpy_round_trip() -> None:
    wet = call_tool("steam_state", {"pressure": 10e5, "quality": 0.5})
    assert wet["two_phase"] is True
    assert wet["temperature_k"] == pytest.approx(453.03, abs=0.02)
    back = call_tool("steam_state", {"pressure": 10e5, "enthalpy_kj_kg": wet["enthalpy_kj_kg"]})
    assert back["quality"] == pytest.approx(0.5, abs=1e-6)
    assert back["cp_kj_kg_k"] is None  # undefined in the dome


def test_steam_state_rejects_over_specification() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        call_tool("steam_state", {"pressure": 1e5, "temperature": 400.0, "quality": 0.5})


def test_reference_fluid_state_for_co2() -> None:
    result = call_tool(
        "reference_fluid_state", {"fluid": "co2", "temperature": 310.0, "pressure": 100e5}
    )
    assert result["fluid"] == "carbon dioxide"
    assert "Span" in result["equation_of_state"]
    # Supercritical CO2 near the critical density.
    assert 600.0 < result["density_kg_m3"] < 800.0
    assert result["two_phase"] is False


def test_reference_saturation_for_propane() -> None:
    result = call_tool("reference_saturation", {"fluid": "propane", "temperature": 300.0})
    assert result["pressure_pa"] == pytest.approx(9.9782e5, rel=1e-3)
    assert result["heat_of_vaporization_kj_kg"] == pytest.approx(330.0, rel=0.02)
    assert result["liquid_density_kg_m3"] > result["vapor_density_kg_m3"]
    with pytest.raises(ValueError, match="exactly one"):
        call_tool("reference_saturation", {"fluid": "propane"})


def test_steam_utility_requirements_heating_and_cooling() -> None:
    heating = call_tool("steam_utility_requirements", {"duty": 1e6, "utility": "lp_steam"})
    assert heating["steam_mass_flow_kg_s"] == pytest.approx(1e6 / 2108e3, rel=0.02)
    assert heating["annual_cost_usd"] > 0.0
    cooling = call_tool("steam_utility_requirements", {"duty": -1e6, "utility": "cooling_water"})
    assert cooling["water_mass_flow_kg_s"] == pytest.approx(15.9, rel=0.02)
    with pytest.raises(ValueError, match="unknown utility"):
        call_tool("steam_utility_requirements", {"duty": 1e6, "utility": "pixie_dust"})


def test_steam_turbine_tool_extracts_power() -> None:
    base = {
        "mass_flow_kg_s": 10.0,
        "inlet_pressure_pa": 40e5,
        "inlet_temperature_k": 723.15,
        "outlet_pressure_pa": 1e5,
    }
    real = call_tool("steam_turbine", {**base, "isentropic_efficiency": 0.75})
    assert real["power_w"] == pytest.approx(10.0 * 0.75 * 812.7e3, rel=2e-2)
    # At 75 % efficiency the residual enthalpy superheats the exhaust
    # (h_out ~ 2721 kJ/kg > h_g(1 bar) ~ 2675 kJ/kg): single-phase outlet.
    assert real["outlet_two_phase"] is False
    assert real["outlet_quality"] is None
    assert real["outlet_enthalpy_kj_kg"] > real["isentropic_outlet_enthalpy_kj_kg"]

    ideal = call_tool("steam_turbine", {**base, "isentropic_efficiency": 1.0})
    assert ideal["outlet_two_phase"] is True
    assert ideal["outlet_quality"] == pytest.approx(0.930, abs=0.01)
    assert ideal["power_w"] > real["power_w"]
