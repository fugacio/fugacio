"""Copilot tools backed by the property packages, the MESH column, and the exchanger."""

import pytest

from fugacio.copilot import call_tool, tool_schemas


def test_new_flowsheet_tools_are_registered() -> None:
    names = {s["name"] for s in tool_schemas()}
    assert names >= {"two_sided_heat_exchanger", "absorber"}
    by_name = {s["name"]: s for s in tool_schemas()}
    for tool in ("flash_drum", "heat_exchanger", "rigorous_distillation", "absorber"):
        assert "method" in by_name[tool]["parameters"]["properties"]


def test_flash_drum_accepts_a_method() -> None:
    base = {
        "components": ["ethanol", "water"],
        "z": [0.3, 0.7],
        "flow": 10.0,
        "temperature": 360.0,
        "pressure": 1.013e5,
    }
    ideal = call_tool("flash_drum", base)
    nrtl = call_tool("flash_drum", {**base, "method": "nrtl"})
    assert 0.0 < nrtl["vapor_fraction"] < 1.0
    # NRTL knows about the ethanol/water non-ideality; PR alone gives a different split.
    assert nrtl["vapor"]["composition"][0] != pytest.approx(
        ideal["vapor"]["composition"][0], abs=1e-3
    )
    with pytest.raises(ValueError, match="unknown thermodynamic method"):
        call_tool("flash_drum", {**base, "method": "voodoo"})


def test_rigorous_distillation_reports_profiles_and_converges() -> None:
    result = call_tool(
        "rigorous_distillation",
        {
            "components": ["benzene", "toluene"],
            "z": [0.5, 0.5],
            "feed_flow": 100.0,
            "feed_temperature": 365.0,
            "pressure": 1.013e5,
            "n_stages": 12,
            "feed_stage": 6,
            "reflux": 2.5,
            "distillate_rate": 50.0,
            "method": "srk",
        },
    )
    assert result["converged"]
    assert len(result["stage_temperatures_k"]) == 12
    assert result["stage_temperatures_k"][0] < result["stage_temperatures_k"][-1]
    assert result["distillate"]["composition"][0] > 0.95
    assert result["bottoms"]["composition"][1] > 0.95
    assert result["condenser_duty_w"] < 0.0 < result["reboiler_duty_w"]


def test_optimize_column_reflux_imposes_purity_spec() -> None:
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
            "target_purity": 0.9,
        },
    )
    assert result["converged"]
    assert result["achieved_purity"] == pytest.approx(0.9, abs=1e-6)
    assert result["reflux_ratio"] > 0.0


def test_two_sided_heat_exchanger_closes_energy_balance() -> None:
    result = call_tool(
        "two_sided_heat_exchanger",
        {
            "hot_components": ["n-hexane"],
            "hot_z": [1.0],
            "hot_flow": 20.0,
            "hot_temperature": 380.0,
            "hot_pressure": 5e5,
            "cold_components": ["water"],
            "cold_z": [1.0],
            "cold_flow": 100.0,
            "cold_temperature": 300.0,
            "cold_pressure": 3e5,
            "t_hot_out": 330.0,
            "cold_method": "iapws",
        },
    )
    assert result["duty_w"] > 0.0
    assert result["hot_out"]["temperature_k"] == pytest.approx(330.0, abs=1e-6)
    assert 300.0 < result["cold_out"]["temperature_k"] < 380.0
    assert result["min_approach_k"] > 0.0
    assert result["ua_w_per_k"] is not None and result["ua_w_per_k"] > 0.0
    assert len(result["hot_curve_k"]) == len(result["cold_curve_k"])


def test_absorber_tool_recovers_heavy_component() -> None:
    result = call_tool(
        "absorber",
        {
            "components": ["methane", "propane", "n-decane"],
            "gas_z": [0.9, 0.1, 0.0],
            "gas_flow": 100.0,
            "gas_temperature": 300.0,
            "solvent_z": [0.0, 0.0, 1.0],
            "solvent_flow": 60.0,
            "solvent_temperature": 300.0,
            "pressure": 20e5,
            "n_stages": 6,
        },
    )
    assert result["converged"]
    assert result["fraction_absorbed"][1] > 0.5  # propane captured
    assert result["fraction_absorbed"][0] < 0.2  # methane mostly passes through
    assert result["treated_gas"]["flow_mol_s"] + result["rich_solvent"][
        "flow_mol_s"
    ] == pytest.approx(160.0, rel=1e-6)
