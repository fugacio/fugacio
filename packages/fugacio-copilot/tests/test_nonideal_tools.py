"""Copilot non-ideal tools: activity coeffs, diagrams, azeotrope, LLE, screening, fitting."""

import jax.numpy as jnp
import pytest

from fugacio.copilot import call_tool, tool_schemas
from fugacio.sim import nrtl_model_for


def test_new_nonideal_tools_are_registered() -> None:
    names = {s["name"] for s in tool_schemas()}
    assert names >= {
        "activity_coefficients",
        "vle_diagram",
        "find_azeotrope",
        "liquid_liquid_split",
        "three_phase_flash",
        "residue_curve_map",
        "solvent_screening",
        "fit_activity_parameters",
    }


def test_residue_curve_map_tool_ternary() -> None:
    res = call_tool(
        "residue_curve_map",
        {
            "components": ["n-pentane", "n-hexane", "n-heptane"],
            "pressure": 101325.0,
            "method": "nrtl",
            "starts": [[0.34, 0.33, 0.33]],
            "steps": 120,
            "t_min": 250.0,
            "t_max": 460.0,
        },
    )
    assert len(res["curves"]) == 1
    curve = res["curves"][0]
    assert len(curve["x"]) == len(curve["temperature_k"])
    for row in curve["x"]:
        assert sum(row) == pytest.approx(1.0, abs=1e-6)
    # The curve runs from the light node (n-pentane) to the heavy node (n-heptane).
    assert max(row[2] for row in curve["x"]) > 0.85
    assert max(row[0] for row in curve["x"]) > 0.85


def test_activity_coefficients_ethanol_water_positive_deviation() -> None:
    res = call_tool(
        "activity_coefficients",
        {
            "components": ["ethanol", "water"],
            "x": [0.3, 0.7],
            "temperature": 343.15,
            "method": "dortmund",
        },
    )
    assert all(g > 1.0 for g in res["activity_coefficients"])
    assert len(res["ln_gamma"]) == 2


def test_vle_diagram_pxy_shapes_and_enrichment() -> None:
    res = call_tool(
        "vle_diagram",
        {
            "components": ["ethanol", "water"],
            "method": "nrtl",
            "kind": "Pxy",
            "temperature": 351.0,
            "points": 11,
        },
    )
    assert res["kind"] == "Pxy"
    assert len(res["x1"]) == len(res["y1"]) == len(res["pressure_pa"]) == 11
    assert res["y1"][0] > res["x1"][0]


def test_find_azeotrope_ethanol_water() -> None:
    res = call_tool(
        "find_azeotrope",
        {"components": ["ethanol", "water"], "method": "nrtl", "temperature": 351.0},
    )
    assert res["exists"]
    assert 0.0 < res["x_azeotrope"][0] < 1.0


def test_liquid_liquid_split_detects_gap() -> None:
    res = call_tool(
        "liquid_liquid_split",
        {
            "components": ["water", "benzene"],
            "z": [0.5, 0.5],
            "temperature": 330.0,
            "method": "unifac",
        },
    )
    assert res["splits_into_two_liquids"]
    # The two conjugate liquids are genuinely different.
    diff = abs(res["phase_I"]["composition"][0] - res["phase_II"]["composition"][0])
    assert diff > 0.2


def test_liquid_liquid_split_reports_miscible_as_stable() -> None:
    res = call_tool(
        "liquid_liquid_split",
        {
            "components": ["ethanol", "water"],
            "z": [0.5, 0.5],
            "temperature": 330.0,
            "method": "unifac",
        },
    )
    assert not res["splits_into_two_liquids"]


def test_three_phase_flash_well_formed() -> None:
    res = call_tool(
        "three_phase_flash",
        {
            "components": ["water", "benzene", "ethanol"],
            "z": [0.47, 0.47, 0.06],
            "temperature": 340.0,
            "pressure": 101325.0,
            "method": "unifac",
        },
    )
    fracs = res["vapor"]["fraction"] + res["liquid_I"]["fraction"] + res["liquid_II"]["fraction"]
    assert fracs == pytest.approx(1.0, abs=1e-4)
    assert len(res["vapor"]["composition"]) == 3


def test_solvent_screening_ranks_ascending() -> None:
    res = call_tool(
        "solvent_screening",
        {
            "solute": "acetone",
            "solvents": ["water", "benzene", "n-hexane"],
            "temperature": 298.15,
            "method": "unifac",
        },
    )
    gammas = [row["gamma_inf_solute"] for row in res["ranking"]]
    assert gammas == sorted(gammas)
    assert all(g > 0.0 for g in gammas)


def test_fit_activity_parameters_recovers_low_residual() -> None:
    # Generate synthetic bubble-pressure data from the curated NRTL, then refit.
    model = nrtl_model_for(["ethanol", "water"])
    x1 = [0.2, 0.4, 0.6, 0.8]
    t = 343.15
    p_data = [float(model.bubble_pressure(t, jnp.array([x, 1.0 - x]))[0]) for x in x1]
    res = call_tool(
        "fit_activity_parameters",
        {
            "components": ["ethanol", "water"],
            "x1": x1,
            "temperature": [t] * len(x1),
            "bubble_pressure": p_data,
            "alpha": 0.3,
        },
    )
    assert res["model"] == "nrtl"
    # A good fit exists (data come from an NRTL), so the residual should be small.
    assert res["rmse_pa"] < 0.02 * (sum(p_data) / len(p_data))
