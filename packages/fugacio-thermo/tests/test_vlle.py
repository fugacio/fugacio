"""Three-phase VLLE: the 2x2 Rachford-Rice core, a real 3-phase flash, and the
heterogeneous azeotrope."""

import jax.numpy as jnp
import pytest

from fugacio.thermo import components as comp
from fugacio.thermo.activity.models import margules, nrtl
from fugacio.thermo.eos import PR
from fugacio.thermo.reference import saturation_pressures
from fugacio.thermo.vlle import _two_phase_rr, flash_vlle, heterogeneous_azeotrope

BW = comp.component_arrays(["benzene", "water"])
TERN = comp.component_arrays(["water", "benzene", "ethanol"])

# A synthetic but realistic ternary: water/benzene strongly immiscible, ethanol
# the partially-distributing entrainer -- a genuine V-L-L former near 340 K, 1 atm.
_TERN_B = jnp.array([[0.0, 1500.0, 350.0], [1500.0, 0.0, 500.0], [200.0, 130.0, 0.0]])
_TERN_ALPHA = jnp.array([[0.0, 0.2, 0.3], [0.2, 0.0, 0.3], [0.3, 0.3, 0.0]])


def _ternary() -> object:
    return nrtl(a=jnp.zeros((3, 3)), b=_TERN_B, alpha=_TERN_ALPHA)


def test_two_phase_rr_recovers_known_interior_split() -> None:
    # A non-degenerate ternary with a unique interior root beta=(0.3, 0.3):
    #   x_i=[0.6,0.3,0.1], x_ii=[0.1,0.3,0.6], y=[0.4,0.4,0.2].
    x_i = jnp.array([0.6, 0.3, 0.1])
    x_ii = jnp.array([0.1, 0.3, 0.6])
    y = jnp.array([0.4, 0.4, 0.2])
    kv = y / x_i
    kl = x_ii / x_i
    z = jnp.array([0.39, 0.33, 0.28])
    beta = _two_phase_rr(z, kv, kl)
    d = 1.0 + beta[0] * (kv - 1.0) + beta[1] * (kl - 1.0)
    assert abs(float(jnp.sum(z * (kv - 1.0) / d))) < 1e-9
    assert abs(float(jnp.sum(z * (kl - 1.0) / d))) < 1e-9
    assert bool(jnp.all(d > 0.0))
    assert float(beta[0]) == pytest.approx(0.3, abs=1e-4)
    assert float(beta[1]) == pytest.approx(0.3, abs=1e-4)


def test_flash_vlle_three_phase_split_and_balance() -> None:
    z = jnp.array([0.47, 0.47, 0.06])
    res = flash_vlle(_ternary(), 340.0, 101325.0, z, TERN["tc"], TERN["pc"], TERN["omega"])
    assert bool(res.three_phase)
    # Two genuinely distinct liquids (water-rich vs benzene-rich).
    assert float(jnp.max(jnp.abs(res.x_i - res.x_ii))) > 0.5
    # All three phase fractions strictly positive and summing to one.
    for beta in (res.beta_v, res.beta_l1, res.beta_l2):
        assert 0.0 < float(beta) < 1.0
    assert float(res.beta_v + res.beta_l1 + res.beta_l2) == pytest.approx(1.0, abs=1e-9)
    # Exact overall mass balance.
    total = res.beta_v * res.y + res.beta_l1 * res.x_i + res.beta_l2 * res.x_ii
    assert float(jnp.max(jnp.abs(total - z))) < 1e-9


def test_heterogeneous_azeotrope_satisfies_defining_condition() -> None:
    model = margules(2.6, 2.6)
    p = 101325.0
    az = heterogeneous_azeotrope(model, p, BW["tc"], BW["pc"], BW["omega"])
    assert float(jnp.max(jnp.abs(az.x_i - az.x_ii))) > 0.1
    assert float(jnp.sum(az.y)) == pytest.approx(1.0, abs=1e-6)
    # Defining condition: sum_i a_i Psat_i = P at the azeotrope temperature.
    a = az.x_i * jnp.exp(model.ln_gamma(az.x_i, az.t))
    psat = saturation_pressures(PR, az.t, BW["tc"], BW["pc"], BW["omega"])
    assert float(jnp.sum(a * psat)) == pytest.approx(p, rel=1e-4)
