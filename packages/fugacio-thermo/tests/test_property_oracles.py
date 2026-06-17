"""Differential tests: correlation/volumetric/transport properties vs CoolProp.

These are *oracle* tests (marker: ``oracle``), excluded from the default suite;
run them with ``just oracles``. CoolProp evaluates reference multiparameter
Helmholtz equations of state and dedicated transport correlations (functional
forms and fits entirely independent of Fugacio's DIPPR-style correlations), so
agreement here is evidence of *accuracy*, not just faithful transcription.

Tolerances are set per property family to the expected accuracy of the
correlation class (tight for fitted forms like DIPPR-105 density, loose for
corresponding-states estimates like Rowlinson-Bondi), so a regression that
swaps a coefficient or breaks unit handling fails loudly while normal
correlation-vs-reference scatter passes.

The ``chemicals``-backed tests at the bottom are kernel-isolation checks: the
same named mixing rule evaluated with identical inputs must agree to near
machine precision.
"""

import jax.numpy as jnp
import pytest

from fugacio.thermo import oracles
from fugacio.thermo.components import get
from fugacio.thermo.constants import R
from fugacio.thermo.correlations import heat_of_vaporization, liquid_heat_capacity
from fugacio.thermo.eos import PR, molar_volume
from fugacio.thermo.reference import antoine_psat
from fugacio.thermo.transport.surface_tension import surface_tensions, winterfeld_scriven_davis
from fugacio.thermo.transport.thermal_conductivity import (
    dippr9h_mixture,
    gas_thermal_conductivities,
    liquid_thermal_conductivities,
)
from fugacio.thermo.transport.viscosity import (
    gas_viscosities,
    liquid_viscosities,
    wilke_mixture_viscosity,
)
from fugacio.thermo.volumetric import liquid_density, translated_liquid_volume_for

pytestmark = [pytest.mark.oracle]

needs_coolprop = pytest.mark.skipif(not oracles.HAVE_COOLPROP, reason="CoolProp not installed")
needs_chemicals = pytest.mark.skipif(not oracles.HAVE_CHEMICALS, reason="chemicals not installed")

# Fluids spanning the chemistry Fugacio targets (alkanes, aromatics, alcohols,
# inorganic gases, polar/associating species) that CoolProp ships reference
# equations for. Components CoolProp lacks are skipped at runtime by CAS lookup.
FLUIDS = [
    "water",
    "methanol",
    "ethanol",
    "acetone",
    "benzene",
    "toluene",
    "m-xylene",
    "o-xylene",
    "p-xylene",
    "ethylbenzene",
    "cyclohexane",
    "n-pentane",
    "n-hexane",
    "n-heptane",
    "n-octane",
    "n-nonane",
    "n-decane",
    "methane",
    "ethane",
    "propane",
    "n-butane",
    "isobutane",
    "ethylene",
    "propylene",
    "nitrogen",
    "oxygen",
    "argon",
    "carbon monoxide",
    "carbon dioxide",
    "hydrogen sulfide",
    "sulfur dioxide",
    "ammonia",
]

# Subset where corresponding-states estimates (Rowlinson-Bondi, Peneloux-PR)
# are expected to hold; strongly polar/associating fluids are excluded because
# those estimators are documented as unreliable for them.
NONPOLAR = [
    "benzene",
    "toluene",
    "cyclohexane",
    "n-pentane",
    "n-hexane",
    "n-heptane",
    "n-octane",
    "methane",
    "ethane",
    "propane",
    "n-butane",
    "isobutane",
    "nitrogen",
    "oxygen",
    "argon",
    "carbon monoxide",
]


def _skip_unless_coolprop(name: str) -> None:
    if not oracles.coolprop_supports(name):
        pytest.skip(f"CoolProp has no reference EOS for {name}")


def _sat_temperature(name: str) -> float:
    """A saturation temperature well inside both libraries' validity ranges."""
    comp = get(name)
    t_triple = oracles.coolprop_triple_temperature(name)
    return float(min(max(0.7 * comp.tc, t_triple + 5.0), 0.9 * comp.tc))


@needs_coolprop
@pytest.mark.parametrize("name", FLUIDS)
def test_saturated_liquid_density_vs_coolprop(name: str) -> None:
    _skip_unless_coolprop(name)
    t = _sat_temperature(name)
    ref = oracles.coolprop_saturation(name, t)["rho_liquid"]
    got = float(liquid_density([name], t, jnp.array([1.0])))
    assert got == pytest.approx(ref, rel=0.03)


@needs_coolprop
@pytest.mark.parametrize("name", FLUIDS)
def test_heat_of_vaporization_vs_coolprop(name: str) -> None:
    _skip_unless_coolprop(name)
    t = _sat_temperature(name)
    ref = oracles.coolprop_saturation(name, t)["hvap"]
    got = float(heat_of_vaporization([name], t)[0])
    assert got == pytest.approx(ref, rel=0.03)


@needs_coolprop
@pytest.mark.parametrize("name", FLUIDS)
def test_antoine_psat_vs_coolprop(name: str) -> None:
    """Curated/backfilled Antoine constants reproduce the reference Psat in-range."""
    _skip_unless_coolprop(name)
    comp = get(name)
    if comp.antoine is None:
        pytest.skip(f"{name} has no Antoine constants")
    t_triple = oracles.coolprop_triple_temperature(name)
    lo = max(comp.antoine.t_min, t_triple + 1.0)
    hi = min(comp.antoine.t_max, 0.95 * comp.tc)
    if hi <= lo:
        pytest.skip(f"{name}: Antoine range does not overlap CoolProp's")
    t = min(max(0.7 * comp.tc, lo), hi)
    ref = oracles.coolprop_saturation(name, t)["psat"]
    a = comp.antoine
    got = float(antoine_psat(t, a.a, a.b, a.c))
    assert got == pytest.approx(ref, rel=0.05)


# CoolProp's n-pentane viscosity (Quinones-Cisneros 2006 f-theory) is ~20% below
# the experimental consensus in the ambient liquid range (1.80e-4 Pa*s at 298 K
# vs 2.24e-4 from DIPPR/Perry, VDI-PPDS, and Viswanath-Natarajan alike), so for
# this one fluid the oracle itself is the outlier.
LIQUID_VISCOSITY_OUTLIERS = {"n-pentane"}


@needs_coolprop
@pytest.mark.parametrize("name", FLUIDS)
def test_liquid_viscosity_vs_coolprop(name: str) -> None:
    _skip_unless_coolprop(name)
    if name in LIQUID_VISCOSITY_OUTLIERS:
        pytest.skip(f"CoolProp's {name} viscosity model disagrees with the literature")
    t = _sat_temperature(name)
    ref = oracles.coolprop_saturation(name, t)
    if "mu_liquid" not in ref:
        pytest.skip(f"CoolProp has no viscosity model for {name}")
    got = float(liquid_viscosities([name], t)[0])
    assert got == pytest.approx(ref["mu_liquid"], rel=0.15)


@needs_coolprop
@pytest.mark.parametrize("name", FLUIDS)
def test_liquid_thermal_conductivity_vs_coolprop(name: str) -> None:
    _skip_unless_coolprop(name)
    t = _sat_temperature(name)
    ref = oracles.coolprop_saturation(name, t)
    if "k_liquid" not in ref:
        pytest.skip(f"CoolProp has no conductivity model for {name}")
    got = float(liquid_thermal_conductivities([name], t)[0])
    assert got == pytest.approx(ref["k_liquid"], rel=0.15)


@needs_coolprop
@pytest.mark.parametrize("name", FLUIDS)
def test_surface_tension_vs_coolprop(name: str) -> None:
    _skip_unless_coolprop(name)
    t = _sat_temperature(name)
    ref = oracles.coolprop_saturation(name, t)
    if "sigma" not in ref:
        pytest.skip(f"CoolProp has no surface-tension model for {name}")
    got = float(surface_tensions([name], t)[0])
    assert got == pytest.approx(ref["sigma"], rel=0.05)


def _gas_state(name: str) -> tuple[float, float]:
    """A dilute superheated-vapour state for gas-phase transport comparisons."""
    comp = get(name)
    return 0.9 * comp.tc, 1.0e4


# Fluids whose CoolProp transport model is itself a predictive
# extended-corresponding-states estimate (+-10-20%), not a fitted correlation.
ECS_GAS_TRANSPORT = {"ethylbenzene": 0.20}


@needs_coolprop
@pytest.mark.parametrize("name", FLUIDS)
def test_gas_viscosity_vs_coolprop(name: str) -> None:
    _skip_unless_coolprop(name)
    t, p = _gas_state(name)
    ref = oracles.coolprop_gas_state(name, t, p)
    if "mu" not in ref:
        pytest.skip(f"CoolProp has no viscosity model for {name}")
    got = float(gas_viscosities([name], t)[0])
    assert got == pytest.approx(ref["mu"], rel=ECS_GAS_TRANSPORT.get(name, 0.10))


@needs_coolprop
@pytest.mark.parametrize("name", FLUIDS)
def test_gas_thermal_conductivity_vs_coolprop(name: str) -> None:
    _skip_unless_coolprop(name)
    t, p = _gas_state(name)
    ref = oracles.coolprop_gas_state(name, t, p)
    if "k" not in ref:
        pytest.skip(f"CoolProp has no conductivity model for {name}")
    got = float(gas_thermal_conductivities([name], t)[0])
    assert got == pytest.approx(ref["k"], rel=0.15)


@needs_coolprop
@pytest.mark.parametrize("name", NONPOLAR)
def test_liquid_heat_capacity_vs_coolprop(name: str) -> None:
    """Rowlinson-Bondi liquid Cp tracks the reference for normal fluids."""
    _skip_unless_coolprop(name)
    t = _sat_temperature(name)
    if t < 200.0:
        # Rowlinson-Bondi builds on the ideal-gas Cp polynomial, which is
        # fitted above ~200 K; extrapolating it into the cryogenic range
        # tests the polynomial's tail, not the correlation.
        pytest.skip(f"{name}: cp_ig polynomial not fitted below 200 K")
    ref = oracles.coolprop_saturation(name, t)["cp_liquid"]
    got = float(liquid_heat_capacity([name], t)[0])
    assert got == pytest.approx(ref, rel=0.15)


@needs_coolprop
@pytest.mark.parametrize("name", FLUIDS)
def test_pr_vapor_compressibility_vs_coolprop(name: str) -> None:
    """PR compressibility tracks the reference EOS in the supercritical gas region."""
    _skip_unless_coolprop(name)
    comp = get(name)
    t, p = 1.2 * comp.tc, 0.5 * comp.pc
    ref = oracles.coolprop_gas_state(name, t, p)["z"]
    v = molar_volume(
        PR,
        t,
        p,
        jnp.array([1.0]),
        jnp.array([comp.tc]),
        jnp.array([comp.pc]),
        jnp.array([comp.omega]),
        phase="vapor",
    )
    got = float(p * v / (R * t))
    assert got == pytest.approx(ref, rel=0.04)


@needs_coolprop
@pytest.mark.parametrize("name", NONPOLAR)
def test_translated_liquid_volume_vs_coolprop(name: str) -> None:
    """Peneloux-translated PR liquid volume lands near the reference density."""
    _skip_unless_coolprop(name)
    comp = get(name)
    t_triple = oracles.coolprop_triple_temperature(name)
    t = float(min(max(0.6 * comp.tc, t_triple + 5.0), 0.85 * comp.tc))
    sat = oracles.coolprop_saturation(name, t)
    p = 2.0 * sat["psat"]
    v = float(translated_liquid_volume_for([name], t, p, jnp.array([1.0])))
    v_ref = comp.mw * 1.0e-3 / oracles.coolprop_gas_state(name, t, p)["rho"]
    assert v == pytest.approx(v_ref, rel=0.06)


# --- chemicals kernel-isolation checks ------------------------------------------


@needs_chemicals
def test_wilke_kernel_matches_chemicals() -> None:
    y = [0.25, 0.55, 0.20]
    mu = [1.10e-5, 1.79e-5, 2.06e-5]
    mw = [16.043, 28.014, 31.999]
    got = float(wilke_mixture_viscosity(jnp.array(y), jnp.array(mu), jnp.array(mw)))
    ref = oracles.chemicals_wilke_viscosity(y, mu, mw)
    assert got == pytest.approx(ref, rel=1e-10)


@needs_chemicals
def test_dippr9h_kernel_matches_chemicals() -> None:
    w = [0.35, 0.65]
    k = [0.59, 0.16]
    got = float(dippr9h_mixture(jnp.array(w), jnp.array(k)))
    ref = oracles.chemicals_dippr9h_conductivity(w, k)
    assert got == pytest.approx(ref, rel=1e-10)


@needs_chemicals
def test_winterfeld_scriven_davis_kernel_matches_chemicals() -> None:
    x = [0.4, 0.6]
    sigma = [0.022, 0.072]
    v_liquid = [5.87e-5, 1.81e-5]  # m^3/mol
    rhom = [1.0 / v for v in v_liquid]
    got = float(winterfeld_scriven_davis(jnp.array(x), jnp.array(sigma), jnp.array(v_liquid)))
    ref = oracles.chemicals_winterfeld_scriven_davis(x, sigma, rhom)
    assert got == pytest.approx(ref, rel=1e-10)
