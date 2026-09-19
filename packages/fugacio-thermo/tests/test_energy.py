"""Energy-specified flashes: PH/PS round-trips, latent heat, and exact sensitivities."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import components as comp
from fugacio.thermo import ideal
from fugacio.thermo.package import CubicPackage, cubic_package

MIX = ["methane", "propane", "n-pentane"]
Z = jnp.array([0.5, 0.3, 0.2])


def _package() -> CubicPackage:
    a = comp.component_arrays(MIX)
    cp = ideal.ideal_gas_coeffs([comp.get(n) for n in MIX])
    return cubic_package(a["tc"], a["pc"], a["omega"], cp)


PKG = _package()


def test_ph_flash_recovers_temperature() -> None:
    t_true, p = 320.0, 20e5
    h = PKG.mixture_enthalpy(t_true, p, Z)
    res = PKG.flash_ph(p, h, Z, t_init=350.0)
    assert float(res.t) == pytest.approx(t_true, abs=1e-3)


def test_ps_flash_recovers_temperature() -> None:
    t_true, p = 320.0, 20e5
    s = PKG.mixture_entropy(t_true, p, Z)
    res = PKG.flash_ps(p, s, Z, t_init=300.0)
    assert float(res.t) == pytest.approx(t_true, abs=1e-3)


def test_ph_flash_two_phase_split_matches_isothermal() -> None:
    t_true, p = 320.0, 20e5
    h = PKG.mixture_enthalpy(t_true, p, Z)
    res = PKG.flash_ph(p, h, Z, t_init=300.0)
    assert 0.0 < float(res.beta) < 1.0  # genuinely two-phase
    assert float(res.beta) == pytest.approx(float(PKG.flash_pt(t_true, p, Z).beta), abs=1e-6)
    assert float(jnp.sum(res.x)) == pytest.approx(1.0, abs=1e-6)
    assert float(jnp.sum(res.y)) == pytest.approx(1.0, abs=1e-6)


def test_dt_dh_is_reciprocal_heat_capacity() -> None:
    """In a single-phase region, dT/dH_spec = 1 / Cp (exact, via implicit diff)."""
    p = 5e5
    t0 = 360.0  # superheated vapor at 5 bar
    h0 = PKG.mixture_enthalpy(t0, p, Z)

    def t_of_h(h: jnp.ndarray) -> jnp.ndarray:
        return PKG.flash_ph(p, h, Z, t_init=t0).t

    dt_dh = float(jax.grad(t_of_h)(h0))
    cp_total = float(PKG.heat_capacity(t0, p, Z, phase="vapor"))
    assert dt_dh == pytest.approx(1.0 / cp_total, rel=1e-3)


def test_ph_gradient_matches_finite_difference() -> None:
    p = 5e5
    t0 = 360.0
    h0 = PKG.mixture_enthalpy(t0, p, Z)

    def t_of_h(h: jnp.ndarray) -> jnp.ndarray:
        return PKG.flash_ph(p, h, Z, t_init=t0).t

    ad = float(jax.grad(t_of_h)(h0))
    fd = float((t_of_h(h0 + 10.0) - t_of_h(h0 - 10.0)) / 20.0)
    assert ad == pytest.approx(fd, rel=1e-3)


def test_failed_energy_flash_is_nan_not_a_bracket_end() -> None:
    # An unreachable enthalpy (far above the bracket's hot end) fails explicitly.
    h_hot = PKG.mixture_enthalpy(1400.0, 5e5, Z)
    solved = PKG.flash_ph_with_info(5e5, h_hot + 1e7, Z, t_init=360.0)
    assert not bool(solved.report.converged)
    assert bool(jnp.isnan(PKG.flash_ph(5e5, h_hot + 1e7, Z, t_init=360.0).t))
