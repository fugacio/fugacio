"""Differential tests: PC-SAFT vs Clapeyron.jl (an independent implementation).

Clapeyron.jl implements the *same* Gross-Sadowski PC-SAFT equation of state in
Julia, with its own residual-Helmholtz code, density solver, and Maxwell
construction. Agreement here therefore checks Fugacio's PC-SAFT math end to end
against a foreign implementation, not just internal consistency.

The catch is parameter provenance: Clapeyron's tabulated ``m``, ``sigma`` and
``epsilon`` may differ slightly from the values vendored in
`fugacio.thermo.saft._data`. So these checks use non-associating species, whose
Gross-Sadowski 2001 parameters are standard, and keep a modest tolerance bounded
by that provenance, not by the math.

The whole module is skipped unless ``juliacall`` (and a working Clapeyron.jl) is
importable, matching the project's other Julia oracles; it never runs in CI.
"""

import jax.numpy as jnp
import pytest

from fugacio.thermo import oracles
from fugacio.thermo.constants import R
from fugacio.thermo.saft import (
    compressibility_factor,
    molar_density,
    pressure,
    psat_saft,
    saft_parameters_for,
)

pytestmark = [pytest.mark.oracle]

needs_clapeyron = pytest.mark.skipif(
    not oracles.HAVE_CLAPEYRON, reason="juliacall (Clapeyron.jl) not installed"
)


@needs_clapeyron
@pytest.mark.parametrize("component", ["propane", "n-butane", "n-heptane", "carbon dioxide"])
def test_pure_pressure_vs_clapeyron(component: str) -> None:
    params = saft_parameters_for([component])
    x = jnp.ones(1)
    t = 350.0
    for rho in (50.0, 2000.0, 9000.0):  # vapour, intermediate, liquid-like
        ref = oracles.clapeyron_pcsaft([component], [1.0], t, rho)
        mine = float(pressure(params, rho, t, x))
        assert mine == pytest.approx(ref["pressure"], rel=2e-3)
        z_mine = float(compressibility_factor(params, rho, t, x))
        assert z_mine == pytest.approx(ref["z"], rel=2e-3)


@needs_clapeyron
def test_mixture_pressure_vs_clapeyron() -> None:
    components = ["propane", "n-butane"]
    params = saft_parameters_for(components, use_database_kij=False)
    x = jnp.array([0.4, 0.6])
    t, rho = 320.0, 7000.0
    ref = oracles.clapeyron_pcsaft(components, [0.4, 0.6], t, rho)
    assert float(pressure(params, rho, t, x)) == pytest.approx(ref["pressure"], rel=3e-3)


@needs_clapeyron
@pytest.mark.parametrize(("component", "t"), [("propane", 280.0), ("n-heptane", 360.0)])
def test_pure_saturation_pressure_vs_clapeyron(component: str, t: float) -> None:
    params = saft_parameters_for([component])
    arr_p = oracles.clapeyron_pcsaft_saturation_pressure(component, t)
    guess = float(arr_p)
    mine = float(psat_saft(params, t, guess))
    assert mine == pytest.approx(arr_p, rel=3e-3)


@needs_clapeyron
def test_liquid_density_vs_clapeyron() -> None:
    """Compressed-liquid density from a (T, P) solve against Clapeyron's pressure."""
    component = "n-pentane"
    params = saft_parameters_for([component])
    x = jnp.ones(1)
    t, p = 300.0, 50e5
    rho = float(molar_density(params, t, p, x, phase="liquid"))
    # Round trip through Clapeyron: its pressure at my density should equal P.
    ref = oracles.clapeyron_pcsaft([component], [1.0], t, rho)
    assert ref["pressure"] == pytest.approx(p, rel=3e-3)
    assert p / (rho * R * t) == pytest.approx(ref["z"], rel=3e-3)
