"""Gamma-phi VLE: Raoult limit, bubble/dew round trip, flash balance, gradients."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import components as comp
from fugacio.thermo import gammaphi as gp
from fugacio.thermo.activity.models import nrtl
from fugacio.thermo.eos import PR
from fugacio.thermo.reference import saturation_pressures

ARR = comp.component_arrays(["ethanol", "water"])
ALPHA = jnp.array([[0.0, 0.3], [0.3, 0.0]])


def _ethanol_water() -> object:
    return nrtl(a=jnp.zeros((2, 2)), b=jnp.array([[0.0, 670.0], [310.0, 0.0]]), alpha=ALPHA)


def _ideal() -> object:
    return nrtl(a=jnp.zeros((2, 2)), b=jnp.zeros((2, 2)), alpha=ALPHA)


def test_ideal_activity_recovers_raoult() -> None:
    x = jnp.array([0.4, 0.6])
    t = 350.0
    p, y = gp.bubble_pressure_gamma(_ideal(), t, x, ARR["tc"], ARR["pc"], ARR["omega"])
    psat = saturation_pressures(PR, t, ARR["tc"], ARR["pc"], ARR["omega"])
    p_raoult = float(jnp.sum(x * psat))
    assert float(p) == pytest.approx(p_raoult, rel=1e-4)
    assert float(jnp.sum(y)) == pytest.approx(1.0, abs=1e-5)


def test_bubble_dew_round_trip() -> None:
    m = _ethanol_water()
    x = jnp.array([0.3, 0.7])
    t = 350.0
    p_bub, y = gp.bubble_pressure_gamma(m, t, x, ARR["tc"], ARR["pc"], ARR["omega"])
    _, x_back = gp.dew_pressure_gamma(m, t, y, ARR["tc"], ARR["pc"], ARR["omega"])
    assert float(jnp.max(jnp.abs(x_back - x))) < 1e-3
    assert float(p_bub) > 0.0


def test_flash_material_balance() -> None:
    m = _ethanol_water()
    z = jnp.array([0.5, 0.5])
    t = 351.0
    p_bub, _ = gp.bubble_pressure_gamma(m, t, z, ARR["tc"], ARR["pc"], ARR["omega"])
    p_dew, _ = gp.dew_pressure_gamma(m, t, z, ARR["tc"], ARR["pc"], ARR["omega"])
    p = 0.5 * (float(p_bub) + float(p_dew))
    r = gp.flash_pt_gamma(m, t, p, z, ARR["tc"], ARR["pc"], ARR["omega"])
    assert 0.0 < float(r.beta) < 1.0
    balance = (1.0 - r.beta) * r.x + r.beta * r.y - z
    assert float(jnp.max(jnp.abs(balance))) < 1e-4


def test_ethanol_water_shows_azeotrope_enrichment() -> None:
    # Below the azeotrope the vapour is ethanol-enriched relative to the liquid.
    m = _ethanol_water()
    x = jnp.array([0.1, 0.9])
    _, y = gp.bubble_pressure_gamma(m, 350.0, x, ARR["tc"], ARR["pc"], ARR["omega"])
    assert float(y[0]) > float(x[0])


def test_bubble_pressure_gradient_matches_fd() -> None:
    m = _ethanol_water()
    x = jnp.array([0.3, 0.7])

    def p_of_t(t: float) -> jax.Array:
        return gp.bubble_pressure_gamma(m, t, x, ARR["tc"], ARR["pc"], ARR["omega"])[0]

    ad = float(jax.grad(p_of_t)(350.0))
    fd = float((p_of_t(350.5) - p_of_t(349.5)) / 1.0)
    assert ad == pytest.approx(fd, rel=5e-3)


def test_bubble_pressure_gradient_wrt_parameters() -> None:
    x = jnp.array([0.3, 0.7])

    def p_of_b12(b12: float) -> jax.Array:
        m = nrtl(a=jnp.zeros((2, 2)), b=jnp.array([[0.0, b12], [310.0, 0.0]]), alpha=ALPHA)
        return gp.bubble_pressure_gamma(m, 350.0, x, ARR["tc"], ARR["pc"], ARR["omega"])[0]

    ad = float(jax.grad(p_of_b12)(670.0))
    fd = float((p_of_b12(671.0) - p_of_b12(669.0)) / 2.0)
    assert ad == pytest.approx(fd, rel=5e-3)
