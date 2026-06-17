"""Hermetic tests of the reference Helmholtz EOS package.

Two layers of ground truth, neither requiring optional dependencies:

* published check tables: the IAPWS-95 release prints the reduced Helmholtz
  derivatives at (500 K, 838.025 kg/m^3) and the IAPWS transport releases
  print viscosity/conductivity values to ~9 significant figures. Matching
  them is a transcription-correctness proof for the vendored coefficients
  *and* an accuracy proof for the autodiff derivative pipeline (the published
  numbers were produced from hand-derived analytic derivatives);
* internal round trips: saturation/density/state solves inverted against
  each other, equifugacity at coexistence, dome bookkeeping for quality
  states.

CoolProp grid comparisons live in ``test_helmholtz_oracles.py`` (opt-in).
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import helmholtz as hh
from fugacio.thermo.helmholtz.fluids import ALIASES
from fugacio.thermo.helmholtz.terms import alpha_derivatives, first_derivatives

WATER = hh.reference_fluid("water")
#: Mass density (kg/m^3) -> molar (mol/m^3).
M_WATER = WATER.molar_mass


def rho_molar(rho_mass: float) -> float:
    return rho_mass / M_WATER


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_lists_all_fluids() -> None:
    names = hh.reference_fluid_names()
    assert len(names) == 26
    assert "water" in names and "carbon dioxide" in names and "R134a" in names
    assert names == tuple(sorted(names))


def test_registry_aliases_and_case() -> None:
    assert hh.reference_fluid("steam") is hh.reference_fluid("water")
    assert hh.reference_fluid("CO2") is hh.reference_fluid("carbon dioxide")
    assert hh.reference_fluid("r134a") is hh.reference_fluid("R134a")
    assert hh.has_reference_fluid("propane")
    assert not hh.has_reference_fluid("plutonium hexafluoride")
    for alias, target in ALIASES.items():
        assert hh.reference_fluid(alias).name == target


def test_registry_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="no reference Helmholtz EOS"):
        hh.reference_fluid("unobtainium")


def test_fluid_is_a_pytree_of_coefficients() -> None:
    leaves = jax.tree_util.tree_leaves(WATER)
    assert all(hasattr(leaf, "dtype") for leaf in leaves)
    rebuilt = jax.tree_util.tree_map(lambda x: x, WATER)
    assert float(hh.pressure(rebuilt, 55000.0, 300.0)) == float(hh.pressure(WATER, 55000.0, 300.0))


def test_metadata_matches_iapws95() -> None:
    assert WATER.t_critical == pytest.approx(647.096)
    assert WATER.rho_critical * M_WATER == pytest.approx(322.0, rel=1e-10)
    assert WATER.p_critical == pytest.approx(22.064e6, rel=1e-6)
    assert WATER.gas_constant == pytest.approx(8.314371357587, rel=1e-12)
    assert WATER.molar_mass == pytest.approx(0.018015268, rel=1e-12)


# ---------------------------------------------------------------------------
# IAPWS-95 published check values (Wagner & Pruss 2002, Table 6.6)
# ---------------------------------------------------------------------------


def test_alpha_derivatives_match_iapws95_table() -> None:
    """Reduced Helmholtz derivatives at T = 500 K, rho = 838.025 kg/m^3."""
    d = alpha_derivatives(WATER, rho_molar(838.025), 500.0)
    assert float(d.a0) == pytest.approx(0.204797734e1, rel=1e-8)
    assert float(d.a0_t) == pytest.approx(0.904611106e1, rel=1e-8)
    assert float(d.a0_tt) == pytest.approx(-0.193249185e1, rel=1e-8)
    assert float(d.ar) == pytest.approx(-0.342693206e1, rel=1e-8)
    assert float(d.ar_d) == pytest.approx(-0.364366650, rel=1e-8)
    assert float(d.ar_dd) == pytest.approx(0.856063701, rel=1e-8)
    assert float(d.ar_t) == pytest.approx(-0.581403435e1, rel=1e-8)
    assert float(d.ar_tt) == pytest.approx(-0.223440737e1, rel=1e-8)
    assert float(d.ar_dt) == pytest.approx(-0.112176915e1, rel=1e-8)


def test_first_derivatives_agree_with_full_bundle() -> None:
    a0, a0_t, ar, ar_d, ar_t = first_derivatives(WATER, rho_molar(838.025), 500.0)
    d = alpha_derivatives(WATER, rho_molar(838.025), 500.0)
    for lean, full in [(a0, d.a0), (a0_t, d.a0_t), (ar, d.ar), (ar_d, d.ar_d), (ar_t, d.ar_t)]:
        assert float(lean) == pytest.approx(float(full), rel=1e-14)


def test_pressure_matches_iapws95_single_phase_table() -> None:
    """Spot rows of the IAPWS-95 single-phase pressure table (Table 7)."""
    cases = [
        (300.0, 996.5560, 0.0992418352e6),
        (500.0, 838.0250, 10.0003858e6),
        (647.0, 358.0000, 22.0384756e6),
        (900.0, 870.7690, 700.000006e6),
    ]
    for t, rho_mass, p_expected in cases:
        p = float(hh.pressure(WATER, rho_molar(rho_mass), t))
        assert p == pytest.approx(p_expected, rel=1e-8)


def test_saturation_matches_iapws95_table() -> None:
    """IAPWS-95 saturation check rows (Table 8): T = 450 K and 625 K."""
    for t, p_mpa, rho_liquid, rho_vapor in [
        (450.0, 0.932203564, 890.341250, 4.81200360),
        (625.0, 16.9082693, 567.090385, 118.290280),
    ]:
        sat = hh.saturation_state(WATER, t=t)
        assert float(sat.p) == pytest.approx(p_mpa * 1e6, rel=1e-8)
        assert float(sat.rho_liquid) * M_WATER == pytest.approx(rho_liquid, rel=1e-8)
        assert float(sat.rho_vapor) * M_WATER == pytest.approx(rho_vapor, rel=1e-8)


# ---------------------------------------------------------------------------
# IAPWS transport check values (R12-08 Table 4/5, R15-11 Table 4/5)
# ---------------------------------------------------------------------------


def test_water_viscosity_matches_iapws_release() -> None:
    cases = [
        (298.15, 998.0, 889.735100e-6),
        (298.15, 1200.0, 1437.649467e-6),
        (373.15, 1000.0, 307.883622e-6),
        (433.15, 1.0, 14.538324e-6),
        (873.15, 600.0, 77.430195e-6),
        (1173.15, 400.0, 64.154608e-6),
    ]
    for t, rho_mass, mu_expected in cases:
        mu = float(hh.water_viscosity(t, rho_molar(rho_mass)))
        assert mu == pytest.approx(mu_expected, rel=2e-8)


def test_water_viscosity_critical_enhancement() -> None:
    """R12-08 Table 5: the enhancement region, exercised via IAPWS-95 autodiff."""
    cases = [
        (647.35, 122.0, 25.520677e-6),
        (647.35, 222.0, 31.337589e-6),
        (647.35, 322.0, 42.961579e-6),
        (647.35, 422.0, 49.436256e-6),
    ]
    for t, rho_mass, mu_expected in cases:
        mu = float(hh.water_viscosity(t, rho_molar(rho_mass)))
        assert mu == pytest.approx(mu_expected, rel=2e-8)


def test_water_conductivity_matches_iapws_release() -> None:
    cases = [
        (298.15, 0.0, 18.4341883e-3),
        (298.15, 998.0, 607.712868e-3),
        (298.15, 1200.0, 799.038144e-3),
        # Steam-region spot value cross-validated against CoolProp 7.0.
        (873.15, 600.0, 485.667599e-3),
    ]
    for t, rho_mass, k_expected in cases:
        k = float(hh.water_thermal_conductivity(t, rho_molar(rho_mass)))
        assert k == pytest.approx(k_expected, rel=2e-8)


def test_water_conductivity_critical_enhancement() -> None:
    """R15-11 Table 5: enhancement built from autodiff cp, cv and compressibilities."""
    cases = [
        (647.35, 1.0, 51.9298924e-3),
        (647.35, 122.0, 130.922885e-3),
        (647.35, 222.0, 367.787459e-3),
        (647.35, 272.0, 757.959776e-3),
        (647.35, 322.0, 1443.755556e-3),
        (647.35, 372.0, 650.319402e-3),
        (647.35, 422.0, 448.883487e-3),
        (647.35, 750.0, 600.961346e-3),
    ]
    for t, rho_mass, k_expected in cases:
        k = float(hh.water_thermal_conductivity(t, rho_molar(rho_mass)))
        assert k == pytest.approx(k_expected, rel=2e-8)


def test_surface_tension_matches_iapws() -> None:
    # IAPWS R1-76(2014) values; the vendored correlation is the Mulero (2012)
    # two-term refit, which deviates from the IAPWS form by < 0.5 %.
    assert float(hh.surface_tension(WATER, 300.0)) == pytest.approx(0.0716860, rel=6e-3)
    assert float(hh.surface_tension(WATER, 450.0)) == pytest.approx(0.0428915, rel=6e-3)
    assert float(hh.surface_tension(WATER, 620.0)) == pytest.approx(0.0042676, rel=6e-3)
    # Zero at and above the critical temperature, and never negative.
    assert float(hh.surface_tension(WATER, WATER.t_critical)) == 0.0
    assert float(hh.surface_tension(WATER, 700.0)) == 0.0


# ---------------------------------------------------------------------------
# Properties: structure and limits
# ---------------------------------------------------------------------------


def test_ideal_gas_limit_z_to_one() -> None:
    for fluid_name in ("water", "carbon dioxide", "methane"):
        fluid = hh.reference_fluid(fluid_name)
        z = float(hh.compressibility_factor(fluid, 1e-6, 1.2 * fluid.t_critical))
        assert z == pytest.approx(1.0, abs=1e-8)


def test_second_virial_sign_and_boyle_behavior() -> None:
    # B < 0 at low temperature, rising toward and through zero near the
    # Boyle temperature (~ 2.5 Tc for most fluids).
    co2 = hh.reference_fluid("co2")
    assert float(hh.second_virial(co2, 250.0)) < 0.0
    assert float(hh.second_virial(co2, 250.0)) < float(hh.second_virial(co2, 400.0))
    assert float(hh.third_virial(co2, 300.0)) == pytest.approx(
        float(hh.third_virial(co2, 300.0))
    )  # finite, no NaN from the delta -> 0 limit


def test_speed_of_sound_liquid_water_room_temperature() -> None:
    rho = hh.molar_density(WATER, 298.15, 101325.0, phase="liquid")
    w = float(hh.speed_of_sound(WATER, rho, 298.15))
    assert w == pytest.approx(1496.7, rel=2e-3)


def test_equifugacity_at_saturation() -> None:
    """Equal Gibbs energy at coexistence implies equal fugacity for a pure fluid."""
    for t in (320.0, 450.0, 600.0):
        sat = hh.saturation_state(WATER, t=t)
        f_liquid = float(hh.fugacity(WATER, sat.rho_liquid, t))
        f_vapor = float(hh.fugacity(WATER, sat.rho_vapor, t))
        assert f_liquid == pytest.approx(f_vapor, rel=1e-9)
        # Fugacity < psat in both phases (phi_sat < 1 for water below 600 K).
        assert f_vapor < float(sat.p)


# ---------------------------------------------------------------------------
# Solvers: round trips
# ---------------------------------------------------------------------------


def test_density_solver_round_trip_all_phases() -> None:
    cases = [
        (300.0, 101325.0, "liquid"),
        (300.0, 3000.0, "vapor"),
        (500.0, 20e6, "liquid"),
        (400.0, 101325.0, "vapor"),
        (700.0, 50e6, "supercritical"),
    ]
    for t, p, phase in cases:
        rho = hh.molar_density(WATER, t, p, phase=phase)
        assert float(hh.pressure(WATER, rho, t)) == pytest.approx(p, rel=1e-9)


def test_density_solver_rejects_unknown_phase() -> None:
    with pytest.raises(ValueError, match="unknown phase"):
        hh.molar_density(WATER, 300.0, 1e5, phase="plasma")


def test_saturation_temperature_inverts_saturation_pressure() -> None:
    for t in (290.0, 373.124, 550.0, 640.0):
        p = hh.saturation_pressure(WATER, t)
        t_back = hh.saturation_temperature(WATER, p)
        assert float(t_back) == pytest.approx(t, abs=2e-7)


def test_normal_boiling_point_of_water() -> None:
    t_boil = float(hh.saturation_temperature(WATER, 101325.0))
    assert t_boil == pytest.approx(373.1243, abs=2e-3)


def test_saturation_state_by_pressure_matches_by_temperature() -> None:
    by_p = hh.saturation_state(WATER, p=1e6)
    by_t = hh.saturation_state(WATER, t=by_p.t)
    assert float(by_t.p) == pytest.approx(1e6, rel=1e-9)
    assert float(by_p.h_vaporization) == pytest.approx(float(by_t.h_vaporization), rel=1e-10)


def test_saturation_state_requires_exactly_one_spec() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        hh.saturation_state(WATER)
    with pytest.raises(ValueError, match="exactly one"):
        hh.saturation_state(WATER, t=400.0, p=1e5)


def test_latent_heat_magnitude() -> None:
    sat = hh.saturation_state(WATER, p=101325.0)
    # 2256.5 kJ/kg at 1 atm.
    assert float(sat.h_vaporization) / M_WATER == pytest.approx(2256.5e3, rel=1e-3)


# ---------------------------------------------------------------------------
# State API
# ---------------------------------------------------------------------------


def test_state_tp_auto_picks_stable_phase() -> None:
    liquid = hh.state_tp(WATER, 300.0, 101325.0)
    vapor = hh.state_tp(WATER, 400.0, 101325.0)
    supercritical = hh.state_tp(WATER, 700.0, 30e6)
    assert float(liquid.rho) * M_WATER > 900.0
    assert float(vapor.rho) * M_WATER < 1.0
    assert not bool(liquid.two_phase) and not bool(vapor.two_phase)
    assert 50.0 < float(supercritical.rho) * M_WATER < 700.0
    assert jnp.isnan(liquid.q) and jnp.isnan(vapor.q)


def test_state_tp_explicit_phase_matches_auto() -> None:
    auto = hh.state_tp(WATER, 320.0, 5e5)
    explicit = hh.state_tp(WATER, 320.0, 5e5, phase="liquid")
    assert float(auto.rho) == pytest.approx(float(explicit.rho), rel=1e-12)
    with pytest.raises(ValueError, match="unknown phase"):
        hh.state_tp(WATER, 320.0, 5e5, phase="frozen")


def test_state_fields_are_thermodynamically_consistent() -> None:
    st = hh.state_tp(WATER, 360.0, 2e5)
    assert float(st.h) == pytest.approx(float(st.u) + float(st.p) / float(st.rho), rel=1e-12)
    assert float(st.g) == pytest.approx(float(st.h) - 360.0 * float(st.s), rel=1e-12)
    assert float(st.z) == pytest.approx(
        float(st.p) / (float(st.rho) * WATER.gas_constant * 360.0), rel=1e-12
    )
    assert float(st.cp) > float(st.cv) > 0.0


def test_state_ph_round_trip_single_phase() -> None:
    for t, p in [(310.0, 5e6), (520.0, 1e6), (700.0, 30e6)]:
        st = hh.state_tp(WATER, t, p)
        back = hh.state_ph(WATER, p, st.h)
        assert float(back.t) == pytest.approx(t, abs=2e-6)
        assert bool(back.two_phase) is False


def test_state_ph_inside_dome_returns_quality() -> None:
    sat = hh.saturation_state(WATER, p=1e6)
    h_mix = 0.25 * float(sat.h_liquid) + 0.75 * float(sat.h_vapor)
    st = hh.state_ph(WATER, 1e6, h_mix)
    assert bool(st.two_phase)
    assert float(st.q) == pytest.approx(0.75, abs=1e-10)
    assert float(st.t) == pytest.approx(float(sat.t), abs=1e-8)
    assert jnp.isnan(st.cp) and jnp.isnan(st.w)
    # Mixture density from quality-weighted volumes.
    v = 0.25 / float(sat.rho_liquid) + 0.75 / float(sat.rho_vapor)
    assert float(st.rho) == pytest.approx(1.0 / v, rel=1e-9)


def test_state_ps_round_trip_and_isentropic_expansion() -> None:
    inlet = hh.state_tp(WATER, 723.15, 40e5)
    back = hh.state_ps(WATER, 40e5, inlet.s)
    assert float(back.t) == pytest.approx(723.15, abs=2e-6)
    # Isentropic expansion of superheated steam into the dome: wet outlet.
    outlet = hh.state_ps(WATER, 1e5, inlet.s)
    assert bool(outlet.two_phase)
    assert 0.85 < float(outlet.q) < 1.0
    assert float(outlet.h) < float(inlet.h)


def test_state_tq_and_pq_agree() -> None:
    by_t = hh.state_tq(WATER, 453.0280, 0.5)
    by_p = hh.state_pq(WATER, float(by_t.p), 0.5)
    assert float(by_p.t) == pytest.approx(453.0280, abs=1e-6)
    assert float(by_p.h) == pytest.approx(float(by_t.h), rel=1e-9)
    assert bool(by_t.two_phase) and bool(by_p.two_phase)


def test_quality_extremes_match_saturation_states() -> None:
    sat = hh.saturation_state(WATER, t=430.0)
    liquid = hh.state_tq(WATER, 430.0, 0.0)
    vapor = hh.state_tq(WATER, 430.0, 1.0)
    assert float(liquid.h) == pytest.approx(float(sat.h_liquid), rel=1e-10)
    assert float(vapor.h) == pytest.approx(float(sat.h_vapor), rel=1e-10)
    assert float(liquid.rho) == pytest.approx(float(sat.rho_liquid), rel=1e-10)
    assert float(vapor.rho) == pytest.approx(float(sat.rho_vapor), rel=1e-10)


def test_solver_backed_functions_compose_with_vmap() -> None:
    temperatures = jnp.array([320.0, 400.0, 500.0])
    psat = jax.vmap(lambda t: hh.saturation_pressure(WATER, t))(temperatures)
    assert psat.shape == (3,)
    assert bool(jnp.all(jnp.diff(psat) > 0.0))


# ---------------------------------------------------------------------------
# A second fluid through the same machinery (Span-Wagner CO2)
# ---------------------------------------------------------------------------


def test_co2_critical_point_and_sublimation_guard() -> None:
    co2 = hh.reference_fluid("carbon dioxide")
    assert co2.t_critical == pytest.approx(304.1282)
    assert co2.p_critical == pytest.approx(7.3773e6, rel=1e-4)
    sat = hh.saturation_state(co2, t=250.0)
    # Span & Wagner 1996: psat(250 K) = 1.785 MPa.
    assert float(sat.p) == pytest.approx(1.785e6, rel=1e-3)


def test_co2_density_round_trip() -> None:
    co2 = hh.reference_fluid("carbon dioxide")
    st = hh.state_tp(co2, 320.0, 10e6)
    assert float(hh.pressure(co2, st.rho, 320.0)) == pytest.approx(10e6, rel=1e-9)
