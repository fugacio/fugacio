"""Energy-specified flashes: PH/PS round-trips, latent heat, and exact sensitivities."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import components as comp
from fugacio.thermo import energy, ideal
from fugacio.thermo.eos import PR


def _arrs(names: list[str]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    a = comp.component_arrays(names)
    return a["tc"], a["pc"], a["omega"]


def _cp(names: list[str]) -> tuple[jnp.ndarray, ...]:
    return ideal.ideal_gas_coeffs([comp.get(n) for n in names])


MIX = ["methane", "propane", "n-pentane"]
Z = jnp.array([0.5, 0.3, 0.2])


def test_ph_flash_recovers_temperature() -> None:
    tc, pc, omega = _arrs(MIX)
    cp = _cp(MIX)
    t_true, p = 320.0, 20e5
    h = energy.mixture_enthalpy(PR, t_true, p, Z, tc, pc, omega, cp)
    res = energy.flash_ph(PR, p, h, Z, tc, pc, omega, cp, t_init=350.0)
    assert float(res.t) == pytest.approx(t_true, abs=1e-3)


def test_ps_flash_recovers_temperature() -> None:
    tc, pc, omega = _arrs(MIX)
    cp = _cp(MIX)
    t_true, p = 320.0, 20e5
    s = energy.mixture_entropy(PR, t_true, p, Z, tc, pc, omega, cp)
    res = energy.flash_ps(PR, p, s, Z, tc, pc, omega, cp, t_init=300.0)
    assert float(res.t) == pytest.approx(t_true, abs=1e-3)


def test_ph_flash_two_phase_split_matches_isothermal() -> None:
    tc, pc, omega = _arrs(MIX)
    cp = _cp(MIX)
    t_true, p = 320.0, 20e5
    h = energy.mixture_enthalpy(PR, t_true, p, Z, tc, pc, omega, cp)
    res = energy.flash_ph(PR, p, h, Z, tc, pc, omega, cp, t_init=300.0)
    assert 0.0 < float(res.beta) < 1.0  # genuinely two-phase
    assert float(jnp.sum(res.x)) == pytest.approx(1.0, abs=1e-6)
    assert float(jnp.sum(res.y)) == pytest.approx(1.0, abs=1e-6)


def test_dT_dH_is_reciprocal_heat_capacity() -> None:
    """In a single-phase region, dT/dH_spec = 1 / Cp (exact, via implicit diff)."""
    tc, pc, omega = _arrs(MIX)
    cp = _cp(MIX)
    p = 5e5
    t0 = 360.0  # superheated vapour at 5 bar
    h0 = float(energy.mixture_enthalpy(PR, t0, p, Z, tc, pc, omega, cp))

    def t_of_h(h: jnp.ndarray) -> jnp.ndarray:
        return energy.flash_ph(PR, p, h, Z, tc, pc, omega, cp, t_init=t0).t

    dT_dH = float(jax.grad(t_of_h)(jnp.asarray(h0)))
    from fugacio.thermo.properties import molar_cp

    cp_total = float(molar_cp(t0, p, Z, tc, pc, omega, cp, phase="vapor"))
    assert dT_dH == pytest.approx(1.0 / cp_total, rel=1e-3)


def test_ph_gradient_matches_finite_difference() -> None:
    tc, pc, omega = _arrs(MIX)
    cp = _cp(MIX)
    p = 5e5
    t0 = 360.0
    h0 = float(energy.mixture_enthalpy(PR, t0, p, Z, tc, pc, omega, cp))

    def t_of_h(h: float) -> float:
        return float(energy.flash_ph(PR, p, h, Z, tc, pc, omega, cp, t_init=t0).t)

    ad = float(
        jax.grad(lambda h: energy.flash_ph(PR, p, h, Z, tc, pc, omega, cp, t_init=t0).t)(
            jnp.asarray(h0)
        )
    )
    fd = (t_of_h(h0 + 10.0) - t_of_h(h0 - 10.0)) / 20.0
    assert ad == pytest.approx(fd, rel=1e-3)
