"""Group-contribution methods: UNIFAC activity and Joback property estimation."""

import jax.numpy as jnp
import pytest

from fugacio.thermo.groupcontrib import joback, unifac


def test_unifac_pure_limit_is_unity() -> None:
    ln_gamma = unifac.unifac_activity(["ethanol", "water"], jnp.array([1.0 - 1e-9, 1e-9]), 298.15)
    assert float(jnp.exp(ln_gamma[0])) == pytest.approx(1.0, abs=1e-5)


def test_unifac_ethanol_water_is_positive_deviation() -> None:
    # Ethanol/water shows strong positive deviations (gamma > 1) over the range.
    gamma = jnp.exp(unifac.unifac_activity(["ethanol", "water"], jnp.array([0.2, 0.8]), 298.15))
    assert float(gamma[0]) > 1.5
    assert float(gamma[1]) > 1.0


def test_unifac_infinite_dilution_exceeds_midrange() -> None:
    def gamma0(x0: float, x1: float) -> float:
        ln_g = unifac.unifac_activity(["ethanol", "water"], jnp.array([x0, x1]), 298.15)
        return float(jnp.exp(ln_g[0]))

    g_dilute = gamma0(1e-6, 1.0 - 1e-6)
    g_mid = gamma0(0.5, 0.5)
    assert g_dilute > g_mid > 1.0


def test_unifac_hexane_benzene_mild_nonideality() -> None:
    gamma = jnp.exp(unifac.unifac_activity(["n-hexane", "benzene"], jnp.array([0.5, 0.5]), 298.15))
    assert 1.0 < float(gamma[0]) < 1.5
    assert 1.0 < float(gamma[1]) < 1.5


def test_unifac_unknown_component_raises() -> None:
    with pytest.raises(KeyError):
        unifac.unifac_activity(["ethanol", "argon"], jnp.array([0.5, 0.5]), 298.15)


def test_joback_acetone() -> None:
    c = joback.joback_estimate({"-CH3": 2, ">C=O": 1}, n_atoms=10, name="acetone", mw=58.08)
    assert c.tc == pytest.approx(508.1, rel=0.03)
    assert c.pc == pytest.approx(47.0e5, rel=0.05)
    assert c.vc is not None and c.vc * 1e6 == pytest.approx(209.0, rel=0.05)
    cp = c.cp_ig
    assert cp is not None
    from fugacio.thermo.constants import R

    cp298 = R * (cp.a + cp.b * 298.15 + cp.c * 298.15**2 + cp.e * 298.15**3)
    assert cp298 == pytest.approx(75.0, rel=0.05)


def test_joback_ethanol_critical_volume() -> None:
    c = joback.joback_estimate({"-CH3": 1, "-CH2-": 1, "-OH": 1}, n_atoms=9, name="ethanol")
    assert c.vc is not None and c.vc * 1e6 == pytest.approx(167.0, rel=0.05)
    assert c.tc == pytest.approx(514.0, rel=0.05)


def test_joback_unknown_group_raises() -> None:
    with pytest.raises(KeyError):
        joback.joback_estimate({"-CH3": 1, "-NOPE-": 1}, n_atoms=5)
