"""Cubic equation-of-state correctness and differentiability."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import components as comp
from fugacio.thermo import diffcheck, eos
from fugacio.thermo.consistency import fugacity_pressure_residual

ALL_EOS = [eos.VDW, eos.RK, eos.SRK, eos.PR]
EOS_IDS = [e.name for e in ALL_EOS]


def _single(name: str) -> tuple:
    c = comp.get(name)
    return jnp.array([1.0]), jnp.array([c.tc]), jnp.array([c.pc]), jnp.array([c.omega]), c


@pytest.mark.parametrize("eos_model", ALL_EOS, ids=EOS_IDS)
@pytest.mark.parametrize("phase", ["vapor", "liquid"])
def test_pressure_volume_roundtrip(eos_model: eos.CubicEOS, phase: str) -> None:
    x, tc, pc, omega, _ = _single("propane")
    t, p = 300.0, 1.0e6
    v = eos.molar_volume(eos_model, t, p, x, tc, pc, omega, phase=phase)
    p_back = eos.pressure(eos_model, t, v, x, tc, pc, omega)
    assert float(p_back) == pytest.approx(p, rel=1e-7)


@pytest.mark.parametrize("eos_model", ALL_EOS, ids=EOS_IDS)
def test_ideal_gas_limit(eos_model: eos.CubicEOS) -> None:
    x, tc, pc, omega, _ = _single("nitrogen")
    z = eos.compressibility(eos_model, 400.0, 1.0, x, tc, pc, omega, phase="vapor")
    assert float(z) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("eos_model", ALL_EOS, ids=EOS_IDS)
def test_liquid_root_below_vapor_root(eos_model: eos.CubicEOS) -> None:
    x, tc, pc, omega, _ = _single("propane")
    z_l = eos.compressibility(eos_model, 300.0, 1.0e6, x, tc, pc, omega, phase="liquid")
    z_v = eos.compressibility(eos_model, 300.0, 1.0e6, x, tc, pc, omega, phase="vapor")
    assert float(z_l) < float(z_v)


@pytest.mark.parametrize("eos_model", ALL_EOS, ids=EOS_IDS)
def test_lnphi_gradient_matches_finite_difference(eos_model: eos.CubicEOS) -> None:
    c = comp.get("propane")

    def ln_phi_of_p(p: jnp.ndarray) -> jnp.ndarray:
        value, _ = eos.ln_phi_pure(eos_model, 300.0, p, c.tc, c.pc, c.omega, phase="vapor")
        return value

    err = diffcheck.max_gradient_error(ln_phi_of_p, jnp.asarray(5.0e5), eps=10.0)
    assert float(err) < 1e-5


@pytest.mark.parametrize("eos_model", ALL_EOS, ids=EOS_IDS)
def test_fugacity_pressure_identity(eos_model: eos.CubicEOS) -> None:
    c = comp.get("propane")
    residual = fugacity_pressure_residual(
        eos_model, 300.0, 5.0e5, c.tc, c.pc, c.omega, phase="vapor"
    )
    assert float(residual) < 1e-6


def test_compress_factor_implicit_gradient() -> None:
    # dZ/dP via the custom_jvp implicit rule must match finite differences.
    x, tc, pc, omega, _ = _single("propane")

    def z_of_p(p: jnp.ndarray) -> jnp.ndarray:
        return eos.compressibility(eos.PR, 300.0, p, x, tc, pc, omega, phase="vapor")

    ad = float(jax.grad(lambda p: z_of_p(p).sum())(1.0e6))
    fd = float((z_of_p(1.0e6 + 1.0).sum() - z_of_p(1.0e6 - 1.0).sum()) / 2.0)
    assert ad == pytest.approx(fd, rel=1e-5)


def test_mixture_lnphi_reduces_to_pure() -> None:
    c = comp.get("propane")
    lp_mix, _ = eos.ln_phi_mixture(
        eos.PR,
        300.0,
        5e5,
        jnp.array([1.0]),
        jnp.array([c.tc]),
        jnp.array([c.pc]),
        jnp.array([c.omega]),
    )
    lp_pure, _ = eos.ln_phi_pure(eos.PR, 300.0, 5e5, c.tc, c.pc, c.omega)
    assert float(lp_mix[0]) == pytest.approx(float(lp_pure), rel=1e-10)
