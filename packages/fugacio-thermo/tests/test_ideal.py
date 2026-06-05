"""Ideal-gas integrals must be mutually consistent (the integrator's own oracle)."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import components as comp
from fugacio.thermo import ideal
from fugacio.thermo.constants import R

COEFFS = comp.get("water").cp_ig
A, B, C, D, E = COEFFS.a, COEFFS.b, COEFFS.c, COEFFS.d, COEFFS.e


@pytest.mark.parametrize("t", [300.0, 500.0, 800.0])
def test_dh_dt_equals_cp(t: float) -> None:
    d_h = jax.grad(lambda tt: ideal.enthalpy_ig(tt, A, B, C, D, E))(t)
    cp = ideal.cp_ig(t, A, B, C, D, E)
    assert float(d_h) == pytest.approx(float(cp), rel=1e-9)


@pytest.mark.parametrize("t", [300.0, 500.0, 800.0])
def test_ds_dt_equals_cp_over_t(t: float) -> None:
    d_s = jax.grad(lambda tt: ideal.entropy_ig(tt, 1e5, A, B, C, D, E))(t)
    cp = ideal.cp_ig(t, A, B, C, D, E)
    assert float(d_s) == pytest.approx(float(cp) / t, rel=1e-9)


def test_gibbs_is_h_minus_ts() -> None:
    t, p = 400.0, 2e5
    g = ideal.gibbs_ig(t, p, A, B, C, D, E)
    h = ideal.enthalpy_ig(t, A, B, C, D, E)
    s = ideal.entropy_ig(t, p, A, B, C, D, E)
    assert float(g) == pytest.approx(float(h - t * s), rel=1e-12)


def test_entropy_pressure_term() -> None:
    t = 350.0
    s1 = ideal.entropy_ig(t, 1e5, A, B, C, D, E)
    s2 = ideal.entropy_ig(t, 2e5, A, B, C, D, E)
    assert float(s1 - s2) == pytest.approx(R * jnp.log(2.0), rel=1e-12)


def test_reference_state_is_zero() -> None:
    assert float(ideal.enthalpy_ig(298.15, A, B, C, D, E)) == pytest.approx(0.0, abs=1e-9)


def test_ideal_gas_coeffs_stacking() -> None:
    coeffs = ideal.ideal_gas_coeffs([comp.get("methane"), comp.get("ethane")])
    assert all(arr.shape == (2,) for arr in coeffs)
    assert float(coeffs[0][0]) == pytest.approx(comp.get("methane").cp_ig.a)
