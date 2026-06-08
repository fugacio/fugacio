"""Copilot reaction tools: heat of reaction, equilibrium, reactors, reactive flash, kinetics fit."""

import math

import pytest

from fugacio.copilot import call_tool, tool_schemas
from fugacio.thermo.constants import R

NH3 = ["nitrogen", "hydrogen", "ammonia"]
NH3_EQ = "nitrogen + 3 hydrogen = 2 ammonia"


def test_reaction_tools_are_registered() -> None:
    names = {s["name"] for s in tool_schemas()}
    assert names >= {
        "heat_of_reaction",
        "reaction_equilibrium",
        "reactor",
        "reactive_flash",
        "fit_kinetics",
    }


def test_heat_of_reaction_ammonia_is_exothermic() -> None:
    res = call_tool("heat_of_reaction", {"equation": NH3_EQ, "components": NH3})
    assert res["exothermic"]
    assert res["delta_h_rxn_j_mol"] < 0.0
    # Ammonia synthesis is spontaneous at 298 K (negative Delta G, K > 1).
    assert res["delta_g_rxn_j_mol"] < 0.0
    assert res["k"] > 1.0
    assert res["ln_k"] == pytest.approx(math.log(res["k"]), rel=1e-6)


def test_reaction_equilibrium_ammonia_feasible() -> None:
    res = call_tool(
        "reaction_equilibrium",
        {
            "components": NH3,
            "equations": [NH3_EQ],
            "n": [1.0, 3.0, 0.0],
            "temperature": 700.0,
            "pressure": 100e5,
        },
    )
    assert sum(res["mole_fractions"]) == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < res["extent"][0] < 1.0
    assert all(m >= 0.0 for m in res["moles"])


def test_reaction_equilibrium_pressure_raises_conversion() -> None:
    def ammonia_fraction(pressure: float) -> float:
        res = call_tool(
            "reaction_equilibrium",
            {
                "components": NH3,
                "equations": [NH3_EQ],
                "n": [1.0, 3.0, 0.0],
                "temperature": 700.0,
                "pressure": pressure,
            },
        )
        return res["mole_fractions"][2]

    assert ammonia_fraction(300e5) > ammonia_fraction(10e5)


def test_reactor_equilibrium_isothermal_removes_heat() -> None:
    res = call_tool(
        "reactor",
        {
            "components": NH3,
            "equation": NH3_EQ,
            "n": [1.0, 3.0, 0.0],
            "temperature": 700.0,
            "pressure": 100e5,
            "mode": "equilibrium",
        },
    )
    # Exothermic reaction held isothermal -> negative duty (heat removed).
    assert res["duty_w"] < 0.0
    assert res["conversion"][0] > 0.0  # nitrogen is consumed
    assert res["outlet_temperature_k"] == pytest.approx(700.0)


def test_reactor_stoichiometric_sets_conversion() -> None:
    res = call_tool(
        "reactor",
        {
            "components": NH3,
            "equation": NH3_EQ,
            "n": [1.0, 3.0, 0.0],
            "temperature": 700.0,
            "pressure": 50e5,
            "mode": "stoichiometric",
            "conversion": 0.5,
        },
    )
    assert res["conversion"][0] == pytest.approx(0.5, abs=1e-6)


def test_reactor_cstr_isomerization() -> None:
    res = call_tool(
        "reactor",
        {
            "components": ["n-butane", "isobutane"],
            "equation": "n-butane = isobutane",
            "n": [10.0, 0.0],
            "temperature": 350.0,
            "pressure": 5e5,
            "mode": "cstr",
            "a": 2.0e3,
            "ea": 30e3,
            "orders": [1.0, 0.0],
            "volume": 2.0,
        },
    )
    assert 0.0 < res["conversion"][0] < 1.0
    assert res["extent"][0] > 0.0


def test_reactor_cstr_requires_kinetics() -> None:
    with pytest.raises(ValueError, match="needs power-law"):
        call_tool(
            "reactor",
            {
                "components": ["n-butane", "isobutane"],
                "equation": "n-butane = isobutane",
                "n": [10.0, 0.0],
                "temperature": 350.0,
                "pressure": 5e5,
                "mode": "cstr",
            },
        )


def test_reactive_flash_esterification() -> None:
    res = call_tool(
        "reactive_flash",
        {
            "components": ["acetic acid", "ethanol", "ethyl acetate", "water"],
            "equation": "acetic acid + ethanol = ethyl acetate + water",
            "n": [1.0, 1.0, 1e-3, 1e-3],
            "temperature": 355.0,
            "pressure": 101325.0,
            "method": "nrtl",
        },
    )
    assert res["extent"][0] > 0.1
    assert 0.0 <= res["vapor_fraction"] <= 1.0
    total = res["vapor"]["flow_mol_s"] + res["liquid"]["flow_mol_s"]
    assert total == pytest.approx(2.002, rel=1e-6)  # equimolar reaction conserves moles


def test_fit_kinetics_recovers_arrhenius() -> None:
    a_true, ea_true = 1.0e7, 60e3
    temps = [300.0, 350.0, 400.0, 450.0, 500.0]
    k = [a_true * math.exp(-ea_true / (R * t)) for t in temps]
    res = call_tool("fit_kinetics", {"temperature": temps, "rate_constant": k})
    assert res["activation_energy_j_mol"] == pytest.approx(ea_true, rel=1e-6)
    assert res["pre_exponential_a"] == pytest.approx(a_true, rel=1e-4)
    assert res["r_squared"] == pytest.approx(1.0, abs=1e-9)
