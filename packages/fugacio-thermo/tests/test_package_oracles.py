"""Differential tests: property packages vs the ``thermo`` library and CoolProp.

Oracle tests (marker: ``oracle``), excluded from the default suite; run with
``just oracles``. Each test skips when its reference backend is missing.

* The gamma-phi package's excess enthalpy and entropy come from *automatic
  differentiation* of the activity model (Gibbs-Helmholtz). ``thermo`` implements
  the same NRTL algebra with hand-derived temperature derivatives (``HE``,
  ``SE``), so agreement on identical parameters grades the autodiff route.
* The Helmholtz package wraps the reference formulations as a one-component
  package; CoolProp's HEOS backend implements the same published equations
  independently, so the package's enthalpy, entropy, and PH/PS flashes must
  agree to solver precision.
* The cubic package's vapor pressure is compared with CoolProp's Peng-Robinson
  backend evaluated on Fugacio's own critical constants, isolating the EOS
  algebra from database differences.
"""

import contextlib

import jax.numpy as jnp
import pytest

from fugacio.thermo import ideal, oracles
from fugacio.thermo.activity.models import nrtl
from fugacio.thermo.components import component_arrays, get
from fugacio.thermo.helmholtz import reference_fluid
from fugacio.thermo.package import (
    cubic_package,
    excess_enthalpy,
    excess_entropy,
    gamma_phi_package,
    helmholtz_package,
)

pytestmark = [pytest.mark.oracle]

needs_thermo = pytest.mark.skipif(not oracles.HAVE_THERMO, reason="thermo not installed")
needs_coolprop = pytest.mark.skipif(not oracles.HAVE_COOLPROP, reason="CoolProp not installed")

# NRTL ethanol(1)/water(2): tau_ij = a_ij + b_ij / T, constant alpha.
_NRTL_A = [[0.0, -0.8], [3.5, 0.0]]
_NRTL_B = [[0.0, 670.0], [-1100.0, 0.0]]
_NRTL_ALPHA = [[0.0, 0.3], [0.3, 0.0]]

_X_GRID = [jnp.array([0.1, 0.9]), jnp.array([0.35, 0.65]), jnp.array([0.7, 0.3])]
_T_GRID = [313.15, 353.15]


def _consts(names: list[str]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, tuple]:
    a = component_arrays(names)
    cp = ideal.ideal_gas_coeffs([get(n) for n in names])
    return a["tc"], a["pc"], a["omega"], cp


def _thermo_nrtl(x: jnp.ndarray, t: float) -> object:
    from thermo.nrtl import NRTL

    return NRTL(
        xs=[float(v) for v in x],
        T=float(t),
        tau_as=_NRTL_A,
        tau_bs=_NRTL_B,
        alpha_cs=_NRTL_ALPHA,
    )


# --------------------------------------------------------------------------- #
# Gamma-phi package: autodiff excess properties vs thermo's analytic derivatives
# --------------------------------------------------------------------------- #
@needs_thermo
@pytest.mark.parametrize("x", _X_GRID)
@pytest.mark.parametrize("t", _T_GRID)
def test_nrtl_excess_enthalpy_matches_thermo(x: jnp.ndarray, t: float) -> None:
    model = nrtl(a=jnp.array(_NRTL_A), b=jnp.array(_NRTL_B), alpha=jnp.array(_NRTL_ALPHA))
    ref = _thermo_nrtl(x, t)
    h_e = float(excess_enthalpy(model, x, t))
    s_e = float(excess_entropy(model, x, t))
    assert h_e == pytest.approx(float(ref.HE()), rel=1e-6, abs=1e-6)  # type: ignore[attr-defined]
    assert s_e == pytest.approx(float(ref.SE()), rel=1e-6, abs=1e-8)  # type: ignore[attr-defined]


@needs_thermo
def test_gamma_phi_package_liquid_enthalpy_carries_thermo_heat_of_mixing() -> None:
    tc, pc, omega, cp = _consts(["ethanol", "water"])
    model = nrtl(a=jnp.array(_NRTL_A), b=jnp.array(_NRTL_B), alpha=jnp.array(_NRTL_ALPHA))
    pkg = gamma_phi_package(model, tc, pc, omega, cp)
    t, p = 323.15, 1.013e5
    x = jnp.array([0.4, 0.6])
    h_mix = pkg.enthalpy(t, p, x, phase="liquid")
    h_pure = jnp.array(
        [
            pkg.enthalpy(t, p, jnp.array([1.0, 0.0]), phase="liquid"),
            pkg.enthalpy(t, p, jnp.array([0.0, 1.0]), phase="liquid"),
        ]
    )
    heat_of_mixing = float(h_mix - x @ h_pure)
    assert heat_of_mixing == pytest.approx(float(_thermo_nrtl(x, t).HE()), rel=1e-6, abs=1e-6)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Helmholtz package vs CoolProp HEOS
# --------------------------------------------------------------------------- #
def _propssi(key: str, *args: object) -> float:
    from CoolProp.CoolProp import PropsSI

    return float(PropsSI(key, *args))  # type: ignore[arg-type]


@needs_coolprop
@pytest.mark.parametrize(
    ("t", "p"),
    [(300.0, 1.0e5), (400.0, 5.0e5), (500.0, 1.0e5), (600.0, 100.0e5), (350.0, 200.0e5)],
)
def test_helmholtz_package_single_phase_matches_coolprop(t: float, p: float) -> None:
    pkg = helmholtz_package(reference_fluid("water"))
    z = jnp.array([1.0])
    h = float(pkg.mixture_enthalpy(t, p, z))
    s = float(pkg.mixture_entropy(t, p, z))
    v = float(pkg.mixture_volume(t, p, z))
    # CoolProp's reference state differs from the formulation's; compare differences
    # against a common anchor state, which cancels the reference on both sides.
    t0, p0 = 300.0, 1.0e5
    h0 = float(pkg.mixture_enthalpy(t0, p0, z))
    s0 = float(pkg.mixture_entropy(t0, p0, z))
    dh_ref = _propssi("Hmolar", "T", t, "P", p, "Water") - _propssi(
        "Hmolar", "T", t0, "P", p0, "Water"
    )
    ds_ref = _propssi("Smolar", "T", t, "P", p, "Water") - _propssi(
        "Smolar", "T", t0, "P", p0, "Water"
    )
    assert h - h0 == pytest.approx(dh_ref, rel=1e-7, abs=1e-3)
    assert s - s0 == pytest.approx(ds_ref, rel=1e-7, abs=1e-5)
    assert v == pytest.approx(1.0 / _propssi("Dmolar", "T", t, "P", p, "Water"), rel=1e-7)


@needs_coolprop
@pytest.mark.parametrize("p", [1.0e5, 10.0e5, 50.0e5])
@pytest.mark.parametrize("quality", [0.15, 0.6, 0.95])
def test_helmholtz_package_ph_flash_matches_coolprop_dome(p: float, quality: float) -> None:
    pkg = helmholtz_package(reference_fluid("water"))
    z = jnp.array([1.0])
    # Anchor the enthalpy through the package's own saturated states so the
    # reference offset cancels, then compare the resulting T and quality.
    t_sat = _propssi("T", "P", p, "Q", 0.0, "Water")
    h_liq = _propssi("Hmolar", "P", p, "Q", 0.0, "Water")
    h_vap = _propssi("Hmolar", "P", p, "Q", 1.0, "Water")
    h_target_cp = h_liq + quality * (h_vap - h_liq)
    offset = float(pkg.mixture_enthalpy(t_sat - 5.0, p, z)) - _propssi(
        "Hmolar", "T", t_sat - 5.0, "P", p, "Water"
    )
    res = pkg.flash_ph(p, h_target_cp + offset, z)
    assert float(res.t) == pytest.approx(t_sat, rel=1e-7)
    assert float(res.beta) == pytest.approx(quality, abs=1e-6)


# --------------------------------------------------------------------------- #
# Cubic package vapor pressure vs CoolProp's PR backend on the same constants
# --------------------------------------------------------------------------- #
@needs_coolprop
@pytest.mark.parametrize("name", ["propane", "n-butane", "carbon dioxide"])
@pytest.mark.parametrize("reduced_t", [0.6, 0.75, 0.9])
def test_cubic_package_vapor_pressure_matches_coolprop_pr(name: str, reduced_t: float) -> None:
    import json

    from CoolProp import CoolProp as coolprop

    tc_a, pc_a, omega_a, cp = _consts([name])
    pkg = cubic_package(tc_a, pc_a, omega_a, cp)
    tc, pc, omega = float(tc_a[0]), float(pc_a[0]), float(omega_a[0])
    t = reduced_t * tc
    p_bub, _ = pkg.bubble_pressure(t, jnp.array([1.0]))

    # A CoolProp PR fluid defined from Fugacio's constants (not its own database).
    fluid_name = f"FugacioPR_{name.replace(' ', '_')}"
    fluid = {
        "CAS": fluid_name,
        "Tc": tc,
        "Tc_units": "K",
        "pc": pc,
        "pc_units": "Pa",
        "acentric": omega,
        "molemass": float(get(name).mw) / 1000.0,
        "molemass_units": "kg/mol",
        "aliases": [],
        "name": fluid_name,
    }
    with contextlib.suppress(ValueError):  # already registered by an earlier case
        coolprop.add_fluids_as_JSON("PR", json.dumps([fluid]))
    p_ref = _propssi("P", "T", t, "Q", 0.0, f"PR::{fluid_name}")
    assert float(p_bub) == pytest.approx(p_ref, rel=1e-6)
