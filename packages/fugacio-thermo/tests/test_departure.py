"""Residual (departure) functions: consistency with fugacity, Gibbs-Helmholtz, AD.

These tests need no external data: they assert the first-principles identities the
residual functions must obey -- the partial-molar Gibbs relation tying them to the
validated fugacity coefficients, the Gibbs-Helmholtz/Maxwell links between G, H, S,
and the ideal-gas limit -- turning them into graded correctness oracles.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import components as comp
from fugacio.thermo import departure as dep
from fugacio.thermo.constants import R
from fugacio.thermo.eos import PR, ln_phi_mixture


def _arrs(names: list[str]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    a = comp.component_arrays(names)
    return a["tc"], a["pc"], a["omega"]


MIX = ["methane", "propane", "n-pentane"]
Z = jnp.array([0.5, 0.3, 0.2])


def test_residual_gibbs_equals_rt_sum_x_lnphi() -> None:
    tc, pc, omega = _arrs(MIX)
    t, p = 320.0, 20e5
    for phase in ("vapor", "liquid"):
        ln_phi, _ = ln_phi_mixture(PR, t, p, Z, tc, pc, omega, phase=phase)
        g_res = dep.residual_gibbs(PR, t, p, Z, tc, pc, omega, phase=phase)
        assert float(g_res) == pytest.approx(float(R * t * jnp.sum(Z * ln_phi)), rel=1e-6)


@pytest.mark.parametrize("phase,t,p", [("vapor", 320.0, 5e5), ("liquid", 300.0, 60e5)])
def test_gibbs_helmholtz_consistency(phase: str, t: float, p: float) -> None:
    """H_res = G_res - T (dG_res/dT)_P and S_res = -(dG_res/dT)_P."""
    tc, pc, omega = _arrs(MIX)
    g_of_t = lambda tt: dep.residual_gibbs(PR, tt, p, Z, tc, pc, omega, phase=phase)  # noqa: E731
    dg_dt = float(jax.grad(g_of_t)(jnp.asarray(t)))
    h_res = float(dep.residual_enthalpy(PR, t, p, Z, tc, pc, omega, phase=phase))
    s_res = float(dep.residual_entropy(PR, t, p, Z, tc, pc, omega, phase=phase))
    assert s_res == pytest.approx(-dg_dt, rel=1e-5, abs=1e-6)
    assert h_res == pytest.approx(float(g_of_t(jnp.asarray(t))) - t * dg_dt, rel=1e-5)


def test_residuals_vanish_in_ideal_gas_limit() -> None:
    tc, pc, omega = _arrs(MIX)
    t, p = 400.0, 1.0  # 1 Pa: essentially ideal gas
    assert abs(float(dep.residual_enthalpy(PR, t, p, Z, tc, pc, omega, phase="vapor"))) < 1.0
    assert abs(float(dep.residual_entropy(PR, t, p, Z, tc, pc, omega, phase="vapor"))) < 1e-3
    assert abs(float(dep.residual_gibbs(PR, t, p, Z, tc, pc, omega, phase="vapor"))) < 1.0


def test_residual_cp_matches_finite_difference_of_enthalpy() -> None:
    tc, pc, omega = _arrs(MIX)
    t, p = 320.0, 5e5
    cp_res = float(dep.residual_cp(PR, t, p, Z, tc, pc, omega, phase="vapor"))
    h = lambda tt: dep.residual_enthalpy(PR, tt, p, Z, tc, pc, omega, phase="vapor")  # noqa: E731
    fd = float((h(t + 0.05) - h(t - 0.05)) / 0.1)
    assert cp_res == pytest.approx(fd, rel=1e-4)


def test_liquid_enthalpy_below_vapor_at_saturation() -> None:
    """The vapour root carries more enthalpy than the liquid root (latent heat > 0)."""
    tc, pc, omega = _arrs(["propane"])
    x = jnp.array([1.0])
    t, p = 300.0, 9.97e5  # near propane saturation
    h_v = float(dep.residual_enthalpy(PR, t, p, x, tc, pc, omega, phase="vapor"))
    h_l = float(dep.residual_enthalpy(PR, t, p, x, tc, pc, omega, phase="liquid"))
    assert h_v - h_l > 0.0
