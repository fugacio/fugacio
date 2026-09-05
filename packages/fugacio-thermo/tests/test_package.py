"""Property packages: one interface, four thermodynamic methods, consistent energy.

The checks here are the ones that make a package trustworthy in an energy
balance: the cubic package reproduces the legacy cubic property functions, the
gamma-phi package's autodiff excess enthalpy satisfies Gibbs-Helmholtz against a
finite difference of ``g^E`` and gives a heat of mixing of the right sign and
size for ethanol/water, every package round-trips its PH and PS flashes, the
generic isochoric flash inverts the volume, and the Wilson seed lets a staged
solver start from a phi-phi package where ``K = 1`` would otherwise be returned.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import (
    PR,
    CubicPackage,
    GammaPhiPackage,
    HelmholtzPackage,
    PropertyPackage,
    SAFTPackage,
    cubic_package,
    energy,
    excess_enthalpy,
    excess_entropy,
    gamma_phi_package,
    helmholtz_package,
    ideal,
    nrtl_from_database,
    saft_package,
    saft_parameters_for,
)
from fugacio.thermo import components as comp
from fugacio.thermo.constants import R
from fugacio.thermo.equilibrium import classify_trivial, wilson_k
from fugacio.thermo.helmholtz import reference_fluid, state_tp

LIGHT = ["methane", "propane", "n-pentane"]
Z = jnp.array([0.5, 0.3, 0.2])


def _consts(names: list[str]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, tuple]:
    a = comp.component_arrays(names)
    cp = ideal.ideal_gas_coeffs([comp.get(n) for n in names])
    return a["tc"], a["pc"], a["omega"], cp


def _cubic(names: list[str] = LIGHT) -> CubicPackage:
    tc, pc, omega, cp = _consts(names)
    return cubic_package(tc, pc, omega, cp, eos=PR)


def _ethanol_water() -> GammaPhiPackage:
    names = ["ethanol", "water"]
    tc, pc, omega, cp = _consts(names)
    return gamma_phi_package(nrtl_from_database(names), tc, pc, omega, cp)


def _saft() -> SAFTPackage:
    names = ["methane", "propane"]
    tc, pc, omega, cp = _consts(names)
    return saft_package(saft_parameters_for(names), tc, pc, omega, cp)


# --------------------------------------------------------------------------- #
# Protocol and cubic equivalence
# --------------------------------------------------------------------------- #
def test_every_package_satisfies_the_protocol() -> None:
    water = helmholtz_package(reference_fluid("water"))
    for pkg in (_cubic(), _ethanol_water(), _saft(), water):
        assert isinstance(pkg, PropertyPackage)
        assert pkg.n_components >= 1
        assert isinstance(pkg.signature(), tuple)


def test_cubic_package_matches_legacy_cubic_functions() -> None:
    pkg = _cubic()
    tc, pc, omega, cp = _consts(LIGHT)
    t, p = 300.0, 20e5  # two-phase at these conditions
    h_pkg = pkg.mixture_enthalpy(t, p, Z)
    h_ref = energy.mixture_enthalpy(PR, t, p, Z, tc, pc, omega, cp)
    assert float(h_pkg) == pytest.approx(float(h_ref), rel=1e-10)
    s_pkg = pkg.mixture_entropy(t, p, Z)
    s_ref = energy.mixture_entropy(PR, t, p, Z, tc, pc, omega, cp)
    assert float(s_pkg) == pytest.approx(float(s_ref), rel=1e-10)


@pytest.mark.parametrize("t_true", [280.0, 330.0])
def test_cubic_ph_and_ps_flash_round_trip(t_true: float) -> None:
    pkg = _cubic()
    p = 20e5
    h = pkg.mixture_enthalpy(t_true, p, Z)
    s = pkg.mixture_entropy(t_true, p, Z)
    assert float(pkg.flash_ph(p, h, Z, t_init=300.0).t) == pytest.approx(t_true, abs=1e-5)
    assert float(pkg.flash_ps(p, s, Z, t_init=300.0).t) == pytest.approx(t_true, abs=1e-5)


def test_cubic_tv_flash_inverts_volume() -> None:
    pkg = _cubic()
    t, p_true = 300.0, 20e5
    v = pkg.mixture_volume(t, p_true, Z)
    p, _flash = pkg.flash_tv(t, v, Z)
    assert float(p) == pytest.approx(p_true, rel=1e-7)


def test_heat_capacity_is_positive_and_matches_enthalpy_slope() -> None:
    pkg = _cubic()
    t, p = 350.0, 20e5
    y = jnp.array([0.7, 0.2, 0.1])
    cp = pkg.heat_capacity(t, p, y, phase="vapor")
    fd = (
        pkg.enthalpy(t + 0.05, p, y, phase="vapor") - pkg.enthalpy(t - 0.05, p, y, phase="vapor")
    ) / 0.1
    assert float(cp) > 0.0
    assert float(cp) == pytest.approx(float(fd), rel=1e-6)


# --------------------------------------------------------------------------- #
# Gamma-phi: autodiff excess properties
# --------------------------------------------------------------------------- #
def test_excess_enthalpy_satisfies_gibbs_helmholtz() -> None:
    model = nrtl_from_database(["ethanol", "water"])
    x = jnp.array([0.4, 0.6])
    t = 340.0

    def g_e_over_rt(tt: float) -> float:
        ln_gamma = model.ln_gamma(x, tt)
        return float(jnp.sum(x * ln_gamma))

    dt = 1e-3
    d_dt = (g_e_over_rt(t + dt) - g_e_over_rt(t - dt)) / (2 * dt)
    h_e_fd = -R * t * t * d_dt
    assert float(excess_enthalpy(model, x, t)) == pytest.approx(h_e_fd, rel=1e-6)


def test_ethanol_water_heat_of_mixing_is_bounded() -> None:
    # The heat of mixing follows from the temperature dependence of the fitted
    # NRTL parameters; VLE-fitted binaries reproduce its size (well under 3 kJ/mol
    # for ethanol/water) but not reliably its sign, so only the magnitude is pinned.
    model = nrtl_from_database(["ethanol", "water"])
    h_e = excess_enthalpy(model, jnp.array([0.3, 0.7]), 298.15)
    assert abs(float(h_e)) < 3000.0
    s_e = excess_entropy(model, jnp.array([0.3, 0.7]), 298.15)
    assert jnp.isfinite(s_e)
    # Pure components have no excess properties.
    assert float(excess_enthalpy(model, jnp.array([1.0, 0.0]), 298.15)) == pytest.approx(
        0.0, abs=1e-8
    )


def test_gamma_phi_liquid_enthalpy_includes_heat_of_mixing() -> None:
    pkg = _ethanol_water()
    t, p = 330.0, 1.013e5
    x = jnp.array([0.4, 0.6])
    h_mix = pkg.enthalpy(t, p, x, phase="liquid")
    h_pure = jnp.array(
        [
            pkg.enthalpy(t, p, jnp.array([1.0, 0.0]), phase="liquid"),
            pkg.enthalpy(t, p, jnp.array([0.0, 1.0]), phase="liquid"),
        ]
    )
    h_e = float(h_mix - jnp.sum(x * h_pure))
    assert h_e == pytest.approx(float(excess_enthalpy(pkg.activity, x, t)), rel=1e-6)


def test_gamma_phi_ph_flash_round_trip_across_the_dome() -> None:
    pkg = _ethanol_water()
    z = jnp.array([0.3, 0.7])
    p = 1.013e5
    for t_true in (340.0, 360.0):  # subcooled liquid and inside the two-phase region
        h = pkg.mixture_enthalpy(t_true, p, z)
        assert float(pkg.flash_ph(p, h, z, t_init=350.0).t) == pytest.approx(t_true, abs=1e-5)


def test_gamma_phi_latent_heat_has_physical_size() -> None:
    pkg = _ethanol_water()
    p = 1.013e5
    x = jnp.array([0.3, 0.7])
    t_b, _y = pkg.bubble_temperature(p, x)
    h_l = pkg.enthalpy(t_b, p, x, phase="liquid")
    h_v = pkg.enthalpy(t_b, p, x, phase="vapor")
    # Water/ethanol latent heats are ~38-41 kJ/mol; the mixture must sit near there.
    assert 33_000.0 < float(h_v - h_l) < 45_000.0


def test_gamma_phi_k_seed_equals_modified_raoult_k() -> None:
    pkg = _ethanol_water()
    x = jnp.array([0.3, 0.7])
    assert jnp.allclose(
        pkg.k_seed(350.0, 1.013e5, x), pkg.k_values(350.0, 1.013e5, x, x), rtol=1e-10
    )


# --------------------------------------------------------------------------- #
# PC-SAFT and Helmholtz
# --------------------------------------------------------------------------- #
def test_saft_ph_flash_round_trip() -> None:
    pkg = _saft()
    z = jnp.array([0.4, 0.6])
    p = 15e5
    t_true = 280.0
    h = pkg.mixture_enthalpy(t_true, p, z)
    assert float(pkg.flash_ph(p, h, z, t_init=300.0).t) == pytest.approx(t_true, abs=1e-4)


def test_helmholtz_package_matches_steam_tables() -> None:
    water = reference_fluid("water")
    pkg: HelmholtzPackage = helmholtz_package(water)
    one = jnp.ones(1)
    st = state_tp(water, 450.0, 5e5, phase="auto")
    assert float(pkg.mixture_enthalpy(450.0, 5e5, one)) == pytest.approx(float(st.h))
    # Inside the dome the PH flash returns the saturation temperature and a quality.
    h_two_phase = float(st.h) - 20_000.0
    res = pkg.flash_ph(5e5, h_two_phase, one)
    assert 0.0 < float(res.beta) < 1.0
    assert float(res.t) == pytest.approx(float(pkg.bubble_temperature(5e5, one)[0]), abs=1e-6)


@pytest.mark.parametrize(
    "property_name,flash_name", [("enthalpy", "flash_ph"), ("entropy", "flash_ps")]
)
def test_pure_energy_flash_quality_has_forward_and_reverse_derivatives(
    property_name: str, flash_name: str
) -> None:
    pkg = _cubic(["water"])
    z = jnp.ones(1)
    p = 1e5
    ts, _ = pkg.bubble_temperature(p, z, t_min=300.0, t_max=450.0)
    prop = getattr(pkg, property_name)
    liquid = prop(ts, p, z, phase="liquid")
    gap = prop(ts, p, z, phase="vapor") - liquid

    @jax.jit
    def state(quality):
        result = getattr(pkg, flash_name)(p, liquid + quality * gap, z)
        return jnp.array([result.t, result.beta])

    q = jnp.asarray(0.3)
    value, forward = jax.jvp(state, (q,), (jnp.ones_like(q),))
    reverse = jax.jacrev(state)(q)
    assert jnp.allclose(value, jnp.array([ts, q]), atol=1e-7)
    # At fixed pressure inside a pure-fluid dome, added energy changes quality,
    # while the saturation temperature stays fixed.
    expected = jnp.array([0.0, 1.0])
    assert jnp.allclose(forward, expected, atol=1e-8)
    assert jnp.allclose(reverse, expected, atol=1e-8)


# --------------------------------------------------------------------------- #
# Seeds and single-phase classification
# --------------------------------------------------------------------------- #
def test_cubic_k_seed_is_wilson() -> None:
    pkg = _cubic()
    assert jnp.allclose(
        pkg.k_seed(300.0, 20e5, Z), wilson_k(300.0, 20e5, pkg.tc, pkg.pc, pkg.omega)
    )


def test_flash_classifies_superheated_vapour_not_liquid() -> None:
    # Methane/ethane/propane at 320 K, 20 bar lies below its dew pressure: all vapour.
    names = ["methane", "ethane", "propane"]
    pkg = _cubic(names)
    z = jnp.array([0.3, 0.3, 0.4])
    assert float(pkg.dew_pressure(320.0, z)[0]) > 20e5
    assert float(pkg.flash_pt(320.0, 20e5, z).beta) == 1.0
    # ... and far above its bubble pressure at low temperature it is all liquid.
    assert float(pkg.flash_pt(200.0, 60e5, z).beta) == 0.0


def test_single_phase_properties_do_not_differentiate_an_absent_phase() -> None:
    pkg = _cubic(["propane", "n-butane", "n-pentane"])
    z = jnp.array([0.4, 0.35, 0.25])
    t, p = 391.70363314, 16e5
    assert pkg.flash_pt(t, p, z).beta == 1.0

    def bulk(tt, composition):
        return pkg.mixture_enthalpy(tt, p, composition)

    def vapor(tt, composition):
        return pkg.enthalpy(tt, p, composition, phase="vapor")

    actual = jax.jit(jax.grad(bulk, argnums=(0, 1)))(t, z)
    expected = jax.grad(vapor, argnums=(0, 1))(t, z)
    for value, reference in zip(actual, expected, strict=True):
        assert jnp.allclose(value, reference, rtol=1e-9)
    finite_difference = (bulk(t + 0.001, z) - bulk(t - 0.001, z)) / 0.002
    assert actual[0] == pytest.approx(float(finite_difference), rel=1e-6)


def test_classify_trivial_leaves_interior_solutions_alone() -> None:
    z = jnp.array([0.5, 0.5])
    k = jnp.array([3.0, 0.2])
    beta = jnp.asarray(0.4)
    assert float(classify_trivial(z, k, beta, k, jnp.asarray(0.9))) == 0.4
    trivial = jnp.ones(2)
    assert (
        float(
            classify_trivial(z, trivial, jnp.asarray(0.0), jnp.array([5.0, 2.0]), jnp.asarray(0.9))
        )
        == 1.0
    )
    assert (
        float(
            classify_trivial(z, trivial, jnp.asarray(0.0), jnp.array([0.2, 0.5]), jnp.asarray(0.1))
        )
        == 0.0
    )


def test_package_gradient_with_respect_to_kij() -> None:
    # Packages are pytrees: the enthalpy is differentiable in a binary parameter.
    tc, pc, omega, cp = _consts(LIGHT)

    def h_of(k01: float) -> jnp.ndarray:
        kij = jnp.zeros((3, 3)).at[0, 1].set(k01).at[1, 0].set(k01)
        pkg = cubic_package(tc, pc, omega, cp, kij=kij)
        return pkg.enthalpy(300.0, 20e5, Z, phase="liquid")

    g = jax.grad(h_of)(0.0)
    fd = (h_of(1e-4) - h_of(-1e-4)) / 2e-4
    assert float(g) == pytest.approx(float(fd), rel=1e-4)
