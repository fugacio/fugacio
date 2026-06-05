"""Real-fluid molar properties: ideal-gas limit, Cp = dH/dT, latent heat, phase label."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import components as comp
from fugacio.thermo import ideal, properties
from fugacio.thermo.eos import PR
from fugacio.thermo.equilibrium import psat_eos


def _arrs(names: list[str]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    a = comp.component_arrays(names)
    return a["tc"], a["pc"], a["omega"]


def _cp(names: list[str]) -> tuple[jnp.ndarray, ...]:
    return ideal.ideal_gas_coeffs([comp.get(n) for n in names])


def test_real_enthalpy_reduces_to_ideal_gas_at_low_pressure() -> None:
    names = ["propane"]
    tc, pc, omega = _arrs(names)
    cp = _cp(names)
    x = jnp.array([1.0])
    h = properties.molar_enthalpy(400.0, 1e2, x, tc, pc, omega, cp, phase="vapor")
    h_ig = ideal.enthalpy_ig_mixture(400.0, x, *cp)
    assert float(h) == pytest.approx(float(h_ig), abs=5.0)


def test_molar_cp_equals_dH_dT() -> None:
    names = ["methane", "propane", "n-pentane"]
    tc, pc, omega = _arrs(names)
    cp = _cp(names)
    x = jnp.array([0.5, 0.3, 0.2])
    t, p = 330.0, 5e5
    cp_real = float(properties.molar_cp(t, p, x, tc, pc, omega, cp, phase="vapor"))
    h_of_t = lambda tt: properties.molar_enthalpy(  # noqa: E731
        tt, p, x, tc, pc, omega, cp, phase="vapor"
    )
    assert cp_real == pytest.approx(float(jax.grad(h_of_t)(jnp.asarray(t))), rel=1e-5)


def test_latent_heat_of_vaporization_is_physical() -> None:
    c = comp.get("propane")
    tc, pc, omega = _arrs(["propane"])
    cp = _cp(["propane"])
    x = jnp.array([1.0])
    t = 300.0
    p = float(psat_eos(PR, t, c.tc, c.pc, c.omega))
    h_v = properties.molar_enthalpy(t, p, x, tc, pc, omega, cp, phase="vapor")
    h_l = properties.molar_enthalpy(t, p, x, tc, pc, omega, cp, phase="liquid")
    dh_vap = float(h_v - h_l)
    # Propane latent heat near 300 K is ~14 kJ/mol; a cubic EOS lands in this band.
    assert 8e3 < dh_vap < 22e3


def test_stable_phase_labels_liquid_and_vapor() -> None:
    tc, pc, omega = _arrs(["propane"])
    x = jnp.array([1.0])
    assert properties.stable_phase(250.0, 50e5, x, tc, pc, omega) == "liquid"
    assert properties.stable_phase(400.0, 1e5, x, tc, pc, omega) == "vapor"


def test_entropy_includes_ideal_mixing() -> None:
    names = ["methane", "propane"]
    tc, pc, omega = _arrs(names)
    cp = _cp(names)
    t, p = 400.0, 1e3  # near-ideal, so residual entropy ~ 0
    x = jnp.array([0.5, 0.5])
    s_mix = float(properties.molar_entropy(t, p, x, tc, pc, omega, cp, phase="vapor"))
    s_pure = 0.0
    for i in range(2):
        xi = jnp.zeros(2).at[i].set(1.0)
        s_pure += 0.5 * float(properties.molar_entropy(t, p, xi, tc, pc, omega, cp, phase="vapor"))
    # Ideal entropy of mixing for a 50/50 split is +R ln 2 per mole.
    from fugacio.thermo.constants import R

    assert s_mix - s_pure == pytest.approx(R * jnp.log(2.0), abs=0.5)
