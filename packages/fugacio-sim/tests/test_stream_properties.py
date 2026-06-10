"""Stream-level volumetric/transport properties and the stream-aware sizing helper."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import (
    Stream,
    column_diameter_for,
    liquid_density,
    liquid_thermal_conductivity,
    liquid_viscosity,
    liquid_volumetric_flow,
    mass_flow,
    surface_tension,
    vapor_density,
    vapor_thermal_conductivity,
    vapor_viscosity,
    vapor_volumetric_flow,
)
from fugacio.thermo.constants import R


def _water_liquid() -> Stream:
    return Stream.from_fractions(("water",), jnp.array([1.0]), 10.0, 298.15, 1.0e5)


def _air_vapor() -> Stream:
    return Stream.from_fractions(
        ("nitrogen", "oxygen"), jnp.array([0.79, 0.21]), 100.0, 300.0, 1.0e5
    )


def _bt_mix(t: float = 360.0, p: float = 1.0e5) -> Stream:
    return Stream.from_fractions(("benzene", "toluene"), jnp.array([0.5, 0.5]), 50.0, t, p)


def test_liquid_density_water() -> None:
    assert float(liquid_density(_water_liquid())) == pytest.approx(997.0, rel=0.025)


def test_vapor_density_near_ideal() -> None:
    s = _air_vapor()
    mw = 0.79 * 28.014 + 0.21 * 31.998
    rho_ideal = float(s.p) * mw * 1e-3 / (R * float(s.t))
    assert float(vapor_density(s)) == pytest.approx(rho_ideal, rel=0.01)


def test_volumetric_flows_consistent_with_mass_flow() -> None:
    liq = _water_liquid()
    assert float(liquid_volumetric_flow(liq)) == pytest.approx(
        float(mass_flow(liq)) / float(liquid_density(liq)), rel=1e-12
    )
    vap = _air_vapor()
    assert float(vapor_volumetric_flow(vap)) == pytest.approx(
        float(mass_flow(vap)) / float(vapor_density(vap)), rel=1e-12
    )


def test_transport_spot_values() -> None:
    assert float(liquid_viscosity(_water_liquid())) == pytest.approx(8.9e-4, rel=0.05)
    assert float(vapor_viscosity(_air_vapor())) == pytest.approx(1.85e-5, rel=0.05)
    assert float(liquid_thermal_conductivity(_water_liquid())) == pytest.approx(0.607, rel=0.05)
    assert float(vapor_thermal_conductivity(_air_vapor())) == pytest.approx(0.026, rel=0.07)
    assert float(surface_tension(_water_liquid())) == pytest.approx(0.072, rel=0.03)


def test_column_diameter_for_is_sane_and_differentiable() -> None:
    # A benzene/toluene column vapour at ~1 atm: expect a diameter of order a metre,
    # decreasing as pressure (hence vapour density) rises.
    def diameter(p: float) -> jnp.ndarray:
        return column_diameter_for(_bt_mix(p=p))

    d = float(diameter(1.0e5))
    assert 0.3 < d < 5.0
    assert float(diameter(2.0e5)) < d
    g = float(jax.grad(lambda p: diameter(p))(1.0e5))
    assert g < 0.0


def test_properties_differentiable_in_state() -> None:
    s = _bt_mix()

    def rho_of_t(t: jnp.ndarray) -> jnp.ndarray:
        return liquid_density(Stream(n=s.n, t=t, p=s.p, components=s.components))

    g = float(jax.grad(rho_of_t)(360.0))
    assert g < 0.0  # liquid expands on heating

    def mu_of_n(n: jnp.ndarray) -> jnp.ndarray:
        return liquid_viscosity(Stream(n=n, t=s.t, p=s.p, components=s.components))

    grad_n = jax.grad(mu_of_n)(s.n)
    assert bool(jnp.all(jnp.isfinite(grad_n)))
