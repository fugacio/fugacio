"""Liquid-liquid equilibrium: isoactivity, mass balance, symmetric binodal."""

import jax.numpy as jnp
import pytest

from fugacio.thermo.activity.models import margules
from fugacio.thermo.lle import binary_binodal, flash_lle


def test_flash_lle_isoactivity_and_balance() -> None:
    model = margules(2.5, 2.5)
    z = jnp.array([0.5, 0.5])
    res = flash_lle(model, 300.0, z)
    # Two distinct liquid phases.
    assert float(jnp.max(jnp.abs(res.x_i - res.x_ii))) > 0.1
    # Isoactivity: x_i gamma_i equal across phases.
    a_i = res.x_i * jnp.exp(model.ln_gamma(res.x_i, 300.0))
    a_ii = res.x_ii * jnp.exp(model.ln_gamma(res.x_ii, 300.0))
    assert float(jnp.max(jnp.abs(a_i - a_ii))) < 1e-4
    # Overall mass balance.
    balance = (1.0 - res.psi) * res.x_i + res.psi * res.x_ii - z
    assert float(jnp.max(jnp.abs(balance))) < 1e-4


def test_symmetric_binodal_is_mirror_symmetric() -> None:
    model = margules(2.5, 2.5)
    x1_a, x1_b = binary_binodal(model, 300.0)
    # For a symmetric model the two solubilities mirror about x = 0.5.
    assert float(x1_a + x1_b) == pytest.approx(1.0, abs=1e-3)
    assert abs(float(x1_a) - float(x1_b)) > 0.2
