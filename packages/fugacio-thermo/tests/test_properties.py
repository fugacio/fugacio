"""Real-fluid molar properties from a cubic package: ideal-gas limit, Cp, latent heat."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import components as comp
from fugacio.thermo import ideal
from fugacio.thermo.constants import R
from fugacio.thermo.eos import PR
from fugacio.thermo.equilibrium import psat_eos
from fugacio.thermo.package import CubicPackage, cubic_package


def _package(names: list[str]) -> CubicPackage:
    a = comp.component_arrays(names)
    cp = ideal.ideal_gas_coeffs([comp.get(n) for n in names])
    return cubic_package(a["tc"], a["pc"], a["omega"], cp)


def test_real_enthalpy_reduces_to_ideal_gas_at_low_pressure() -> None:
    pkg = _package(["propane"])
    x = jnp.array([1.0])
    h = pkg.enthalpy(400.0, 1e2, x, phase="vapor")
    h_ig = ideal.enthalpy_ig_mixture(400.0, x, *pkg.cp)
    assert float(h) == pytest.approx(float(h_ig), abs=5.0)


def test_heat_capacity_equals_dh_dt() -> None:
    pkg = _package(["methane", "propane", "n-pentane"])
    x = jnp.array([0.5, 0.3, 0.2])
    t, p = 330.0, 5e5
    cp_real = float(pkg.heat_capacity(t, p, x, phase="vapor"))
    step = 1e-3
    h_hi = pkg.enthalpy(t + step, p, x, phase="vapor")
    h_lo = pkg.enthalpy(t - step, p, x, phase="vapor")
    assert cp_real == pytest.approx(float((h_hi - h_lo) / (2 * step)), rel=1e-5)
    assert cp_real > 0


def test_latent_heat_of_vaporization_is_physical() -> None:
    pkg = _package(["propane"])
    x = jnp.array([1.0])
    t = 300.0
    p = float(psat_eos(PR, t, pkg.tc[0], pkg.pc[0], pkg.omega[0]))
    dh_vap = float(pkg.enthalpy(t, p, x, phase="vapor") - pkg.enthalpy(t, p, x, phase="liquid"))
    # Propane latent heat near 300 K is about 14 kJ/mol; a cubic EOS lands in this band.
    assert 8e3 < dh_vap < 22e3


def test_flash_labels_compressed_liquid_and_superheated_vapor() -> None:
    pkg = _package(["propane"])
    x = jnp.array([1.0])
    assert float(pkg.flash_pt(250.0, 50e5, x).beta) == 0.0
    assert float(pkg.flash_pt(400.0, 1e5, x).beta) == 1.0


def test_entropy_includes_ideal_mixing() -> None:
    pkg = _package(["methane", "propane"])
    t, p = 400.0, 1e3  # Near-ideal, so the residual entropy is about zero.
    s_mix = float(pkg.entropy(t, p, jnp.array([0.5, 0.5]), phase="vapor"))
    s_pure = sum(
        0.5 * float(pkg.entropy(t, p, jnp.zeros(2).at[i].set(1.0), phase="vapor")) for i in range(2)
    )
    # Ideal entropy of mixing for a 50/50 split is +R ln 2 per mole.
    assert s_mix - s_pure == pytest.approx(R * jnp.log(2.0), abs=0.5)


def test_mixture_enthalpy_is_differentiable_through_the_flash() -> None:
    pkg = _package(["methane", "n-butane"])
    z = jnp.array([0.5, 0.5])
    grad = jax.grad(lambda t: pkg.mixture_enthalpy(t, 1e6, z))(250.0)
    assert jnp.isfinite(grad)
    assert float(grad) > 0
