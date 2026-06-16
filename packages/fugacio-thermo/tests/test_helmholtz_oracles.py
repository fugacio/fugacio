"""Differential tests: reference Helmholtz EOS vs CoolProp, fluid by fluid.

CoolProp implements the *same published equations of state* from independent
code (hand-derived analytic derivatives in C++, its own density and Maxwell
solvers), so agreement here checks coefficient transcription, the autodiff
derivative pipeline, and the solver stack end to end. Tolerances are far
tighter than for correlation-vs-correlation oracles -- two implementations of
one EOS must agree to solver precision (~1e-8), not correlation scatter.

Six workhorse fluids get dense grids (saturation curve, single-phase property
grid, flash round trips); all 26 vendored fluids get spot checks on both
sides of the dome plus the saturation line. Water additionally gets its IAPWS
transport formulations graded on a grid that crosses the critical
enhancement region.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import helmholtz as hh
from fugacio.thermo import oracles
from fugacio.thermo.helmholtz._data import FLUID_DATA

pytestmark = [pytest.mark.oracle]

needs_coolprop = pytest.mark.skipif(not oracles.HAVE_COOLPROP, reason="CoolProp not installed")

#: Fluids that get dense grids; the rest get spot checks.
CORE_FLUIDS = ["water", "carbon dioxide", "methane", "propane", "ammonia", "R134a"]
ALL_FLUIDS = sorted(FLUID_DATA)


def _propssi(key: str, *args: object) -> float:
    from CoolProp.CoolProp import PropsSI

    return float(PropsSI(key, *args))  # type: ignore[arg-type]


def _coolprop_name(fluid_name: str) -> str:
    return str(FLUID_DATA[fluid_name]["coolprop_name"])


# ---------------------------------------------------------------------------
# Saturation curves
# ---------------------------------------------------------------------------


@needs_coolprop
@pytest.mark.parametrize("fluid_name", CORE_FLUIDS)
def test_saturation_curve_vs_coolprop(fluid_name: str) -> None:
    fluid = hh.reference_fluid(fluid_name)
    cp_name = _coolprop_name(fluid_name)
    t_lo = max(fluid.t_triple * 1.01, 0.45 * fluid.t_critical)
    reduced = jnp.linspace(t_lo / fluid.t_critical, 0.995, 9)
    temperatures = reduced * fluid.t_critical

    sat = jax.vmap(lambda t: hh.saturation_state(fluid, t=t))(temperatures)
    for i, t in enumerate(temperatures):
        t_ref = float(t)
        psat_ref = _propssi("P", "T", t_ref, "Q", 0, cp_name)
        rho_l_ref = _propssi("Dmolar", "T", t_ref, "Q", 0, cp_name)
        rho_v_ref = _propssi("Dmolar", "T", t_ref, "Q", 1, cp_name)
        hvap_ref = _propssi("Hmolar", "T", t_ref, "Q", 1, cp_name) - _propssi(
            "Hmolar", "T", t_ref, "Q", 0, cp_name
        )
        assert float(sat.p[i]) == pytest.approx(psat_ref, rel=1e-8)
        assert float(sat.rho_liquid[i]) == pytest.approx(rho_l_ref, rel=1e-8)
        assert float(sat.rho_vapor[i]) == pytest.approx(rho_v_ref, rel=1e-8)
        assert float(sat.h_vaporization[i]) == pytest.approx(hvap_ref, rel=1e-7)


@needs_coolprop
@pytest.mark.parametrize("fluid_name", ALL_FLUIDS)
def test_saturation_spot_check_every_fluid(fluid_name: str) -> None:
    fluid = hh.reference_fluid(fluid_name)
    cp_name = _coolprop_name(fluid_name)
    for fraction in (0.7, 0.9):
        t = max(fraction * fluid.t_critical, fluid.t_triple * 1.02)
        sat = hh.saturation_state(fluid, t=t)
        assert float(sat.p) == pytest.approx(_propssi("P", "T", t, "Q", 0, cp_name), rel=1e-8)
        assert float(sat.rho_liquid) == pytest.approx(
            _propssi("Dmolar", "T", t, "Q", 0, cp_name), rel=1e-8
        )
        assert float(sat.rho_vapor) == pytest.approx(
            _propssi("Dmolar", "T", t, "Q", 1, cp_name), rel=1e-8
        )


@needs_coolprop
@pytest.mark.parametrize("fluid_name", CORE_FLUIDS)
def test_saturation_temperature_inversion_vs_coolprop(fluid_name: str) -> None:
    fluid = hh.reference_fluid(fluid_name)
    cp_name = _coolprop_name(fluid_name)
    for fraction in (0.05, 0.3, 0.8):
        p = fluid.p_triple + fraction * (fluid.p_critical - fluid.p_triple)
        t = float(hh.saturation_temperature(fluid, p))
        assert t == pytest.approx(_propssi("T", "P", p, "Q", 0, cp_name), abs=2e-6)


# ---------------------------------------------------------------------------
# Single-phase property grids
# ---------------------------------------------------------------------------


@needs_coolprop
@pytest.mark.parametrize("fluid_name", CORE_FLUIDS)
def test_single_phase_property_grid_vs_coolprop(fluid_name: str) -> None:
    """P, h, s, cv, cp, w, phi on liquid, vapor and supercritical (T, rho) nodes."""
    fluid = hh.reference_fluid(fluid_name)
    cp_name = _coolprop_name(fluid_name)
    nodes: list[tuple[float, float]] = []
    for fraction in (0.6, 0.8, 0.95):
        t = max(fraction * fluid.t_critical, fluid.t_triple * 1.05)
        nodes.append((t, 1.02 * _propssi("Dmolar", "T", t, "Q", 0, cp_name)))  # liquid
        nodes.append((t, 0.5 * _propssi("Dmolar", "T", t, "Q", 1, cp_name)))  # vapor
    nodes.append((1.1 * fluid.t_critical, 1.2 * fluid.rho_critical))  # supercritical
    nodes.append((1.5 * fluid.t_critical, 0.05 * fluid.rho_critical))  # dilute gas

    for t, rho in nodes:
        for key, fn, rel in [
            ("P", hh.pressure, 1e-9),
            ("Hmolar", hh.enthalpy, 1e-8),
            ("Smolar", hh.entropy, 1e-8),
            ("Cvmolar", hh.isochoric_heat_capacity, 1e-9),
            ("Cpmolar", hh.isobaric_heat_capacity, 1e-9),
            ("A", hh.speed_of_sound, 1e-9),
        ]:
            mine = float(fn(fluid, rho, t))
            ref = _propssi(key, "T", t, "Dmolar", rho, cp_name)
            assert mine == pytest.approx(ref, rel=rel, abs=1e-10), (key, t, rho)


@needs_coolprop
@pytest.mark.parametrize("fluid_name", ALL_FLUIDS)
def test_density_solver_spot_check_every_fluid(fluid_name: str) -> None:
    fluid = hh.reference_fluid(fluid_name)
    cp_name = _coolprop_name(fluid_name)
    # Compressed liquid.
    t_liquid = max(0.7 * fluid.t_critical, fluid.t_triple * 1.05)
    p_liquid = 3.0 * _propssi("P", "T", t_liquid, "Q", 0, cp_name)
    rho = float(hh.molar_density(fluid, t_liquid, p_liquid, phase="liquid"))
    assert rho == pytest.approx(_propssi("Dmolar", "T", t_liquid, "P", p_liquid, cp_name), rel=1e-9)
    # Superheated vapor.
    t_vapor = max(0.9 * fluid.t_critical, fluid.t_triple * 1.1)
    p_vapor = 0.3 * _propssi("P", "T", t_vapor, "Q", 1, cp_name)
    rho = float(hh.molar_density(fluid, t_vapor, p_vapor, phase="vapor"))
    assert rho == pytest.approx(_propssi("Dmolar", "T", t_vapor, "P", p_vapor, cp_name), rel=1e-9)
    # Supercritical.
    t_super = 1.2 * fluid.t_critical
    p_super = 1.5 * fluid.p_critical
    rho = float(hh.molar_density(fluid, t_super, p_super, phase="supercritical"))
    assert rho == pytest.approx(_propssi("Dmolar", "T", t_super, "P", p_super, cp_name), rel=1e-9)


@needs_coolprop
@pytest.mark.parametrize("fluid_name", CORE_FLUIDS)
def test_virial_coefficients_vs_coolprop(fluid_name: str) -> None:
    fluid = hh.reference_fluid(fluid_name)
    cp_name = _coolprop_name(fluid_name)
    for fraction in (0.9, 1.2, 2.0):
        t = fraction * fluid.t_critical
        b_ref = _propssi("Bvirial", "T", t, "Dmolar", 1.0, cp_name)
        assert float(hh.second_virial(fluid, t)) == pytest.approx(b_ref, rel=1e-8)
        c_ref = _propssi("Cvirial", "T", t, "Dmolar", 1.0, cp_name)
        # CoolProp evaluates C by finite differences; agreement is limited by
        # its step size (worst for the steep GaoB terms of ammonia), not by
        # this implementation's exact AD limit.
        assert float(hh.third_virial(fluid, t)) == pytest.approx(c_ref, rel=1e-3)


# ---------------------------------------------------------------------------
# State resolution round trips
# ---------------------------------------------------------------------------


@needs_coolprop
@pytest.mark.parametrize("fluid_name", ["water", "carbon dioxide", "propane"])
def test_state_ph_against_coolprop_flash(fluid_name: str) -> None:
    fluid = hh.reference_fluid(fluid_name)
    cp_name = _coolprop_name(fluid_name)
    p = 0.4 * fluid.p_critical
    t_sat = _propssi("T", "P", p, "Q", 0, cp_name)
    h_liquid = _propssi("Hmolar", "P", p, "Q", 0, cp_name)
    h_vapor = _propssi("Hmolar", "P", p, "Q", 1, cp_name)
    cases = [
        h_liquid - 0.08 * (h_vapor - h_liquid),  # subcooled (above the triple line)
        h_liquid + 0.37 * (h_vapor - h_liquid),  # two-phase
        h_vapor + 0.8 * (h_vapor - h_liquid),  # superheated
    ]
    for h in cases:
        state = hh.state_ph(fluid, p, h)
        t_ref = _propssi("T", "P", p, "Hmolar", h, cp_name)
        q_ref = _propssi("Q", "P", p, "Hmolar", h, cp_name)
        assert float(state.t) == pytest.approx(t_ref, abs=5e-5)
        if 0.0 <= q_ref <= 1.0:
            assert bool(state.two_phase)
            assert float(state.q) == pytest.approx(q_ref, abs=1e-7)
            assert float(state.t) == pytest.approx(t_sat, abs=1e-5)
        else:
            assert not bool(state.two_phase)


@needs_coolprop
def test_state_ps_against_coolprop_flash() -> None:
    fluid = hh.reference_fluid("water")
    inlet = hh.state_tp(fluid, 723.15, 40e5)
    outlet = hh.state_ps(fluid, 1e5, inlet.s)
    t_ref = _propssi("T", "P", 1e5, "Smolar", float(inlet.s), "Water")
    q_ref = _propssi("Q", "P", 1e5, "Smolar", float(inlet.s), "Water")
    assert float(outlet.t) == pytest.approx(t_ref, abs=5e-5)
    assert float(outlet.q) == pytest.approx(q_ref, abs=1e-7)


# ---------------------------------------------------------------------------
# Water transport and surface tension
# ---------------------------------------------------------------------------


@needs_coolprop
def test_water_transport_grid_vs_coolprop() -> None:
    """Viscosity and conductivity across liquid, vapor, supercritical and
    near-critical states (the enhancement region included)."""
    fluid = hh.reference_fluid("water")
    nodes_mass = [
        (280.0, 1000.0),
        (298.15, 997.0),
        (373.15, 958.4),
        (373.15, 0.6),
        (500.0, 838.0),
        (647.35, 222.0),
        (647.35, 322.0),
        (647.35, 422.0),
        (660.0, 322.0),
        (800.0, 100.0),
        (1000.0, 50.0),
    ]
    for t, rho_mass in nodes_mass:
        rho = rho_mass / fluid.molar_mass
        mu_ref = _propssi("V", "T", t, "Dmolar", rho, "Water")
        k_ref = _propssi("L", "T", t, "Dmolar", rho, "Water")
        assert float(hh.water_viscosity(t, rho)) == pytest.approx(mu_ref, rel=1e-9)
        assert float(hh.water_thermal_conductivity(t, rho)) == pytest.approx(k_ref, rel=1e-9)


@needs_coolprop
@pytest.mark.parametrize("fluid_name", ALL_FLUIDS)
def test_surface_tension_every_fluid_vs_coolprop(fluid_name: str) -> None:
    fluid = hh.reference_fluid(fluid_name)
    cp_name = _coolprop_name(fluid_name)
    for fraction in (0.6, 0.85):
        t = max(fraction * fluid.t_critical, fluid.t_triple * 1.02)
        try:
            sigma_ref = _propssi("I", "T", t, "Q", 0, cp_name)
        except ValueError:
            pytest.skip(f"CoolProp has no surface tension for {cp_name}")
        assert float(hh.surface_tension(fluid, t)) == pytest.approx(sigma_ref, rel=1e-10)


# ---------------------------------------------------------------------------
# Differentiability vs CoolProp finite differences
# ---------------------------------------------------------------------------


@needs_coolprop
def test_saturation_slope_vs_coolprop_finite_difference() -> None:
    """AD d(psat)/dT through the Maxwell solve vs central differences of CoolProp."""
    fluid = hh.reference_fluid("water")
    t = 500.0
    ad = float(jax.grad(lambda tt: hh.saturation_pressure(fluid, tt))(jnp.asarray(t)))
    dt = 1e-3
    fd = (
        _propssi("P", "T", t + dt, "Q", 0, "Water") - _propssi("P", "T", t - dt, "Q", 0, "Water")
    ) / (2 * dt)
    assert ad == pytest.approx(fd, rel=1e-6)
