"""Physical-property and diffusivity copilot tools."""

import pytest

from fugacio.copilot import call_tool, tool_schemas


def test_property_tools_are_registered() -> None:
    names = {s["name"] for s in tool_schemas()}
    assert names >= {"physical_properties", "binary_diffusivity"}


def test_physical_properties_pure_water() -> None:
    result = call_tool(
        "physical_properties",
        {"components": ["water"], "x": [1.0], "temperature": 298.15},
    )
    assert result["liquid_density_kg_m3"] == pytest.approx(997.0, rel=0.025)
    assert result["liquid_viscosity_pa_s"] == pytest.approx(8.9e-4, rel=0.05)
    assert result["surface_tension_n_m"] == pytest.approx(0.072, rel=0.03)
    assert result["heat_of_vaporization_j_mol"][0] == pytest.approx(43990.0, rel=0.02)
    assert result["liquid_heat_capacity_j_mol_k"][0] == pytest.approx(75.3, rel=0.35)


def test_physical_properties_mixture_brackets_pure_values() -> None:
    mix = call_tool(
        "physical_properties",
        {
            "components": ["benzene", "toluene"],
            "x": [0.5, 0.5],
            "temperature": 298.15,
        },
    )
    pure = [
        call_tool(
            "physical_properties",
            {"components": ["benzene", "toluene"], "x": x, "temperature": 298.15},
        )
        for x in ([1.0, 0.0], [0.0, 1.0])
    ]
    for key in ("liquid_density_kg_m3", "liquid_viscosity_pa_s", "surface_tension_n_m"):
        lo = min(p[key] for p in pure)
        hi = max(p[key] for p in pure)
        assert lo <= mix[key] <= hi


def test_vapor_density_tracks_pressure() -> None:
    low = call_tool(
        "physical_properties",
        {"components": ["nitrogen"], "x": [1.0], "temperature": 300.0, "pressure": 1.0e5},
    )
    high = call_tool(
        "physical_properties",
        {"components": ["nitrogen"], "x": [1.0], "temperature": 300.0, "pressure": 2.0e5},
    )
    assert high["vapor_density_kg_m3"] == pytest.approx(2.0 * low["vapor_density_kg_m3"], rel=0.02)


def test_binary_diffusivity_tool() -> None:
    result = call_tool(
        "binary_diffusivity",
        {"solute": "ethanol", "solvent": "water", "temperature": 298.15},
    )
    # Wilke-Chang for ethanol in water: experimental 1.24e-9 m^2/s (10-20% method).
    assert result["liquid_diffusivity_m2_s"] == pytest.approx(1.24e-9, rel=0.25)
    # Gas-phase Fuller value at 1 atm is of order 1e-5 m^2/s.
    assert 5.0e-6 < result["gas_diffusivity_m2_s"] < 5.0e-5


def test_component_properties_include_new_fields() -> None:
    props = call_tool("component_properties", {"component": "water"})
    assert props["dipole_moment_debye"] == pytest.approx(1.85, abs=0.1)
    assert props["critical_volume_m3_mol"] == pytest.approx(5.6e-5, rel=0.05)
