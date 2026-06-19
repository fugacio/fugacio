"""Behaviour and literature-value tests for the PC-SAFT equation of state.

These exercise the parameter bank, the combining rules, the residual Helmholtz
energy and its property derivatives, the density solver, pure saturation, and the
mixture phase-equilibrium routines against physically required behaviour and a
handful of well-known reference values. They use no external libraries; the
first-principles autodiff identities live in ``test_saft_consistency.py`` and the
Clapeyron.jl cross-checks in ``test_saft_oracles.py``.
"""

import jax.numpy as jnp
import pytest

from fugacio.thermo import component_arrays
from fugacio.thermo.saft import (
    SAFTModel,
    alpha_residual,
    compressibility_factor,
    ln_fugacity_coefficients,
    molar_density,
    pressure,
    psat_saft,
    saft_model,
    saft_parameters_for,
    segment_diameter,
    site_fractions,
)
from fugacio.thermo.saft.parameters import ANGSTROM, epsilon_ij, sigma_ij


def _wilson_guess(component: str, t: float) -> float:
    arr = component_arrays([component])
    tc, pc, omega = float(arr["tc"][0]), float(arr["pc"][0]), float(arr["omega"][0])
    return pc * float(jnp.exp(5.373 * (1.0 + omega) * (1.0 - tc / t)))


def _model(components: list[str]) -> SAFTModel:
    arr = component_arrays(components)
    params = saft_parameters_for(components)
    return saft_model(params, arr["tc"], arr["pc"], arr["omega"])


# ---------------------------------------------------------------------------
# Parameters and combining rules
# ---------------------------------------------------------------------------


def test_parameter_bank_loads_and_converts_units() -> None:
    params = saft_parameters_for(["propane"])
    assert params.n_components == 1
    assert not params.associating
    # sigma stored in metres (literature angstrom value 3.6184 A).
    assert float(params.sigma[0]) == pytest.approx(3.6184 * ANGSTROM, rel=1e-12)
    assert float(params.m[0]) == pytest.approx(2.0020, rel=1e-12)
    assert float(params.epsilon[0]) == pytest.approx(208.11, rel=1e-12)


def test_water_is_associating_with_two_sites() -> None:
    params = saft_parameters_for(["water"])
    assert params.associating
    assert float(params.n_sites_a[0]) == 1.0
    assert float(params.n_sites_b[0]) == 1.0
    assert float(params.epsilon_ab[0]) > 0.0
    assert float(params.kappa_ab[0]) > 0.0


def test_unknown_component_raises() -> None:
    with pytest.raises(KeyError):
        saft_parameters_for(["unobtainium"])


def test_database_kij_is_symmetric_and_applied() -> None:
    params = saft_parameters_for(["carbon dioxide", "propane"])
    kij = params.kij
    assert float(kij[0, 1]) == pytest.approx(0.1135, rel=1e-12)
    assert float(kij[0, 1]) == float(kij[1, 0])
    assert float(kij[0, 0]) == 0.0
    # epsilon_ij reduced below the geometric mean by (1 - kij).
    geom = (float(params.epsilon[0]) * float(params.epsilon[1])) ** 0.5
    assert float(epsilon_ij(params)[0, 1]) == pytest.approx(geom * (1.0 - 0.1135), rel=1e-12)


def test_sigma_combining_rule_is_arithmetic_mean() -> None:
    params = saft_parameters_for(["methane", "n-decane"])
    s = sigma_ij(params)
    assert float(s[0, 1]) == pytest.approx(0.5 * (float(params.sigma[0]) + float(params.sigma[1])))


def test_segment_diameter_is_below_sigma_and_shrinks_with_temperature() -> None:
    params = saft_parameters_for(["n-hexane"])
    d_cold = float(segment_diameter(params, 250.0)[0])
    d_hot = float(segment_diameter(params, 500.0)[0])
    sigma = float(params.sigma[0])
    # Chen-Kreglewski soft repulsion: d -> sigma as T -> 0 and 0.88 sigma as
    # T -> infinity, so the effective diameter shrinks as temperature rises.
    assert d_hot < d_cold < sigma
    assert d_hot > 0.88 * sigma


# ---------------------------------------------------------------------------
# Residual Helmholtz energy and ideal-gas limit
# ---------------------------------------------------------------------------


def test_residual_helmholtz_vanishes_in_the_dilute_limit() -> None:
    params = saft_parameters_for(["nitrogen"])
    x = jnp.ones(1)
    a_dilute = float(alpha_residual(params, 1e-6, 300.0, x))
    assert abs(a_dilute) < 1e-9


def test_compressibility_factor_approaches_unity_in_the_dilute_limit() -> None:
    params = saft_parameters_for(["propane"])
    x = jnp.ones(1)
    z = float(compressibility_factor(params, 1e-3, 350.0, x))
    assert z == pytest.approx(1.0, abs=1e-6)


def test_liquid_compressibility_factor_is_small_and_positive() -> None:
    params = saft_parameters_for(["n-hexane"])
    x = jnp.ones(1)
    rho = molar_density(params, 300.0, 1e5, x, phase="liquid")
    z = float(compressibility_factor(params, rho, 300.0, x))
    assert 0.0 < z < 0.05  # dense liquid well below the ideal gas


# ---------------------------------------------------------------------------
# Density solver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phase", ["liquid", "vapor"])
def test_density_solver_round_trips_pressure(phase: str) -> None:
    params = saft_parameters_for(["propane"])
    x = jnp.ones(1)
    t, p = 300.0, (5e5 if phase == "vapor" else 5e6)
    rho = molar_density(params, t, p, x, phase=phase)
    assert float(pressure(params, rho, t, x)) == pytest.approx(p, rel=1e-7)


def test_liquid_branch_is_denser_than_vapor_branch() -> None:
    params = saft_parameters_for(["propane"])
    x = jnp.ones(1)
    t, p = 280.0, 6e5  # below the propane saturation pressure: two real roots
    rho_l = float(molar_density(params, t, p, x, phase="liquid"))
    rho_v = float(molar_density(params, t, p, x, phase="vapor"))
    assert rho_l > 10.0 * rho_v


def test_stable_branch_selects_lower_gibbs_root() -> None:
    params = saft_parameters_for(["propane"])
    x = jnp.ones(1)
    # Well above saturation: the stable phase is the dense liquid.
    rho_stable = float(molar_density(params, 280.0, 20e5, x, phase="stable"))
    rho_liquid = float(molar_density(params, 280.0, 20e5, x, phase="liquid"))
    assert rho_stable == pytest.approx(rho_liquid, rel=1e-6)


def test_water_liquid_density_is_physically_reasonable() -> None:
    params = saft_parameters_for(["water"])
    x = jnp.ones(1)
    rho = float(molar_density(params, 298.15, 1e5, x, phase="liquid"))  # mol/m^3
    mass_density = rho * 18.015 / 1000.0  # kg/m^3
    assert 850.0 < mass_density < 1050.0


# ---------------------------------------------------------------------------
# Pure saturation pressure (literature values)
# ---------------------------------------------------------------------------


def test_propane_saturation_pressure_matches_experiment() -> None:
    params = saft_parameters_for(["propane"])
    psat = float(psat_saft(params, 300.0, _wilson_guess("propane", 300.0)))
    # NIST: propane Psat(300 K) ~ 9.97 bar; PC-SAFT reproduces it to ~2 %.
    assert psat == pytest.approx(9.97e5, rel=0.05)


def test_normal_boiling_point_gives_about_one_atmosphere() -> None:
    # n-heptane boils at ~371.6 K; its PC-SAFT Psat there should be ~1 atm.
    params = saft_parameters_for(["n-heptane"])
    psat = float(psat_saft(params, 371.6, _wilson_guess("n-heptane", 371.6)))
    assert psat == pytest.approx(101325.0, rel=0.06)


def test_saturation_pressure_increases_with_temperature() -> None:
    params = saft_parameters_for(["n-pentane"])
    p_lo = float(psat_saft(params, 300.0, _wilson_guess("n-pentane", 300.0)))
    p_hi = float(psat_saft(params, 340.0, _wilson_guess("n-pentane", 340.0)))
    assert p_hi > p_lo > 0.0


# ---------------------------------------------------------------------------
# Mixture fugacity and the pure-component limit
# ---------------------------------------------------------------------------


def test_mixture_fugacity_reduces_to_pure_in_the_pure_limit() -> None:
    pure = saft_parameters_for(["ethane"])
    mix = saft_parameters_for(["ethane", "n-heptane"])
    t, p = 300.0, 8e5
    ln_phi_pure = float(ln_fugacity_coefficients(pure, t, p, jnp.ones(1), phase="vapor")[0])
    x = jnp.array([1.0 - 1e-9, 1e-9])
    ln_phi_mix = float(ln_fugacity_coefficients(mix, t, p, x, phase="vapor")[0])
    assert ln_phi_mix == pytest.approx(ln_phi_pure, rel=1e-5)


# ---------------------------------------------------------------------------
# Phase equilibrium
# ---------------------------------------------------------------------------


def test_flash_conserves_mass_in_the_two_phase_region() -> None:
    model = _model(["propane", "n-butane"])
    z = jnp.array([0.5, 0.5])
    res = model.flash_pt(320.0, 8e5, z)
    assert 0.0 < float(res.beta) < 1.0
    assert float(res.x.sum()) == pytest.approx(1.0, abs=1e-9)
    assert float(res.y.sum()) == pytest.approx(1.0, abs=1e-9)
    recombined = float(res.beta) * res.y + (1.0 - float(res.beta)) * res.x
    assert jnp.allclose(recombined, z, atol=1e-9)


def test_flash_returns_single_phase_vapor_below_the_dew_pressure() -> None:
    model = _model(["ethanol", "water"])
    res = model.flash_pt(350.0, 0.5e5, jnp.array([0.5, 0.5]))  # below dew P
    assert float(res.beta) == pytest.approx(1.0, abs=1e-6)
    assert bool(jnp.all(jnp.isfinite(res.x)))
    assert bool(jnp.all(jnp.isfinite(res.y)))


def test_flash_returns_single_phase_liquid_above_the_bubble_pressure() -> None:
    model = _model(["propane", "n-butane"])
    res = model.flash_pt(320.0, 30e5, jnp.array([0.5, 0.5]))  # compressed liquid
    assert float(res.beta) == pytest.approx(0.0, abs=1e-6)


def test_bubble_and_dew_pressure_bracket_and_order() -> None:
    model = _model(["propane", "n-butane"])
    z = jnp.array([0.4, 0.6])
    p_bub, y = model.bubble_pressure(320.0, z)
    p_dew, x = model.dew_pressure(320.0, z)
    assert float(p_bub) > float(p_dew) > 0.0
    assert float(y.sum()) == pytest.approx(1.0, abs=1e-9)
    assert float(x.sum()) == pytest.approx(1.0, abs=1e-9)
    # Vapour is richer in the more volatile propane than the liquid feed.
    assert float(y[0]) > float(z[0])


def test_flash_at_bubble_pressure_has_vanishing_vapor_fraction() -> None:
    model = _model(["propane", "n-butane"])
    x = jnp.array([0.4, 0.6])
    p_bub, _ = model.bubble_pressure(320.0, x)
    res = model.flash_pt(320.0, float(p_bub), x)
    assert float(res.beta) == pytest.approx(0.0, abs=1e-4)


def test_bubble_temperature_inverts_bubble_pressure() -> None:
    model = _model(["propane", "n-butane"])
    x = jnp.array([0.4, 0.6])
    p = 8e5
    t_bub, _ = model.bubble_temperature(p, x)
    p_check, _ = model.bubble_pressure(float(t_bub), x)
    assert float(p_check) == pytest.approx(p, rel=1e-5)


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


def test_stability_flags_two_phase_feed_unstable() -> None:
    model = _model(["propane", "n-butane"])
    z = jnp.array([0.5, 0.5])
    # Inside the dome (between dew and bubble pressure): must be unstable.
    p_bub, _ = model.bubble_pressure(320.0, z)
    p_dew, _ = model.dew_pressure(320.0, z)
    p_mid = 0.5 * (float(p_bub) + float(p_dew))
    assert not bool(model.stability(320.0, p_mid, z).stable)


def test_stability_flags_compressed_liquid_stable() -> None:
    model = _model(["propane", "n-butane"])
    z = jnp.array([0.5, 0.5])
    assert bool(model.stability(320.0, 30e5, z).stable)


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------


def test_site_fractions_are_bounded_and_symmetric_for_one_alcohol() -> None:
    params = saft_parameters_for(["ethanol"])
    x = jnp.ones(1)
    rho = molar_density(params, 298.15, 1e5, x, phase="liquid")
    xa, xb = site_fractions(params, rho, 298.15, x)
    assert 0.0 < float(xa[0]) < 1.0
    # 2B scheme with equal A/B counts: bonded fractions of A and B coincide.
    assert float(xa[0]) == pytest.approx(float(xb[0]), rel=1e-9)


def test_association_lowers_the_saturation_pressure_of_water() -> None:
    """Switching association off must raise water's volatility (no H-bond network)."""
    from dataclasses import replace

    params = saft_parameters_for(["water"])
    no_assoc = replace(params, associating=False)
    t = 373.15
    p_assoc = float(psat_saft(params, t, _wilson_guess("water", t)))
    p_plain = float(psat_saft(no_assoc, t, _wilson_guess("water", t)))
    assert p_assoc < p_plain
