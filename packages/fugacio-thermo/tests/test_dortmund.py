"""Modified UNIFAC (Dortmund): limits, positive deviation, T-dependence, grads."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo.groupcontrib.dortmund import modified_unifac_activity


def test_pure_component_has_unit_activity_coefficient() -> None:
    ln_g = modified_unifac_activity(["ethanol", "water"], jnp.array([1.0, 0.0]), 350.0)
    assert float(ln_g[0]) == pytest.approx(0.0, abs=1e-5)


def test_ethanol_water_positive_deviation() -> None:
    ln_g = modified_unifac_activity(["ethanol", "water"], jnp.array([0.5, 0.5]), 343.15)
    assert bool(jnp.all(ln_g > 0.0))


def test_infinite_dilution_is_the_largest_coefficient() -> None:
    comps = ["ethanol", "water"]
    ln_g_dilute = modified_unifac_activity(comps, jnp.array([1e-3, 1.0 - 1e-3]), 343.15)
    ln_g_mid = modified_unifac_activity(comps, jnp.array([0.5, 0.5]), 343.15)
    assert float(ln_g_dilute[0]) > float(ln_g_mid[0])


def test_temperature_dependence_is_nontrivial() -> None:
    comps = ["ethanol", "water"]
    x = jnp.array([0.3, 0.7])
    low = modified_unifac_activity(comps, x, 320.0)
    high = modified_unifac_activity(comps, x, 360.0)
    assert float(jnp.max(jnp.abs(low - high))) > 1e-3


def test_activity_is_differentiable_in_temperature() -> None:
    comps = ["ethanol", "water"]
    x = jnp.array([0.3, 0.7])

    def g0(t: float) -> jax.Array:
        return modified_unifac_activity(comps, x, t)[0]

    ad = float(jax.grad(g0)(343.15))
    fd = float((g0(343.65) - g0(342.65)) / 1.0)
    assert ad == pytest.approx(fd, rel=1e-2, abs=1e-5)


def test_unknown_component_raises() -> None:
    with pytest.raises(KeyError):
        modified_unifac_activity(["ethanol", "unobtanium"], jnp.array([0.5, 0.5]), 350.0)
