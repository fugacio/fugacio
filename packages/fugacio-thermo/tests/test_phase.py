"""Unified EquilibriumModel interface: EOS and gamma-phi behave consistently."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import components as comp
from fugacio.thermo import equilibrium as eq
from fugacio.thermo import gammaphi as gp
from fugacio.thermo.activity.models import nrtl
from fugacio.thermo.eos import PR
from fugacio.thermo.phase import eos_model, gamma_phi_model

ARR = comp.component_arrays(["ethanol", "water"])
HC = comp.component_arrays(["methane", "propane", "n-pentane"])
ALPHA = jnp.array([[0.0, 0.3], [0.3, 0.0]])


def test_eos_model_flash_matches_functional() -> None:
    model = eos_model(HC["tc"], HC["pc"], HC["omega"], eos=PR)
    z = jnp.array([0.5, 0.3, 0.2])
    r_obj = model.flash_pt(320.0, 20e5, z)
    r_fun = eq.flash_pt(PR, 320.0, 20e5, z, HC["tc"], HC["pc"], HC["omega"])
    assert float(r_obj.beta) == pytest.approx(float(r_fun.beta), abs=1e-6)


def test_gamma_phi_model_bubble_matches_functional() -> None:
    m = nrtl(a=jnp.zeros((2, 2)), b=jnp.array([[0.0, 670.0], [310.0, 0.0]]), alpha=ALPHA)
    model = gamma_phi_model(m, ARR["tc"], ARR["pc"], ARR["omega"])
    x = jnp.array([0.3, 0.7])
    p_obj, y_obj = model.bubble_pressure(350.0, x)
    p_fun, y_fun = gp.bubble_pressure_gamma(m, 350.0, x, ARR["tc"], ARR["pc"], ARR["omega"])
    assert float(p_obj) == pytest.approx(float(p_fun), rel=1e-6)
    assert float(jnp.max(jnp.abs(y_obj - y_fun))) < 1e-6


def test_eos_model_bubble_temperature_round_trip() -> None:
    arr = comp.component_arrays(["benzene", "toluene"])
    model = eos_model(arr["tc"], arr["pc"], arr["omega"], eos=PR)
    x = jnp.array([0.4, 0.6])
    # Bracket above the cold region where the cubic's liquid root degenerates.
    t_bub, _ = model.bubble_temperature(101325.0, x, t_min=280.0, t_max=420.0)
    p_at_t, _ = model.bubble_pressure(t_bub, x)
    assert float(p_at_t) == pytest.approx(101325.0, rel=1e-3)
    assert 350.0 < float(t_bub) < 380.0


def test_gradient_through_model_object() -> None:
    m = nrtl(a=jnp.zeros((2, 2)), b=jnp.array([[0.0, 670.0], [310.0, 0.0]]), alpha=ALPHA)
    model = gamma_phi_model(m, ARR["tc"], ARR["pc"], ARR["omega"])
    x = jnp.array([0.3, 0.7])

    def p_of_t(t: float) -> jax.Array:
        return model.bubble_pressure(t, x)[0]

    ad = float(jax.grad(p_of_t)(350.0))
    fd = float((p_of_t(350.5) - p_of_t(349.5)) / 1.0)
    assert ad == pytest.approx(fd, rel=5e-3)
