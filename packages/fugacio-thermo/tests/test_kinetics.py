"""Differentiable rate laws: algebra, limiting behaviour, and gradients."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo.constants import R
from fugacio.thermo.kinetics import (
    LHHW,
    Arrhenius,
    MassActionReversible,
    PowerLaw,
    arrhenius,
    arrhenius_ref,
)


def test_arrhenius_value_and_monotonicity() -> None:
    a, ea = 1e7, 60e3
    assert float(arrhenius(400.0, a, ea)) == pytest.approx(a * jnp.exp(-ea / (R * 400.0)))
    # Rate constant rises with temperature.
    assert float(arrhenius(300.0, a, ea)) < float(arrhenius(500.0, a, ea))


def test_arrhenius_ref_matches_pre_exponential_form() -> None:
    ea, t_ref = 75e3, 350.0
    k_ref = 2.5
    a_equiv = k_ref * jnp.exp(ea / (R * t_ref))
    for t in (300.0, 350.0, 450.0):
        assert float(arrhenius_ref(t, k_ref, ea, t_ref)) == pytest.approx(
            float(arrhenius(t, a_equiv, ea)), rel=1e-10
        )


def test_arrhenius_dataclass_k() -> None:
    law = Arrhenius(a=jnp.asarray(1e6), ea=jnp.asarray(50e3))
    assert float(law.k(420.0)) == pytest.approx(float(arrhenius(420.0, 1e6, 50e3)))


def test_power_law_rate_and_zero_handling() -> None:
    law = PowerLaw(a=jnp.asarray(3.0), ea=jnp.asarray(0.0), orders=jnp.array([1.0, 2.0, 0.0]))
    c = jnp.array([2.0, 3.0, 99.0])  # third species has order 0 -> no effect
    assert float(law.rate(300.0, c)) == pytest.approx(3.0 * 2.0**1 * 3.0**2)
    # A depleted reactant (positive order) gives exactly zero rate.
    assert float(law.rate(300.0, jnp.array([0.0, 3.0, 1.0]))) == 0.0


def test_mass_action_reversible_zero_at_equilibrium() -> None:
    # A <=> B with equal forward/backward constants: rate vanishes when c_A == c_B.
    law = MassActionReversible(
        a_f=jnp.asarray(10.0),
        ea_f=jnp.asarray(40e3),
        a_r=jnp.asarray(10.0),
        ea_r=jnp.asarray(40e3),
        nu=jnp.array([-1.0, 1.0]),
    )
    assert float(law.rate(400.0, jnp.array([1.5, 1.5]))) == pytest.approx(0.0, abs=1e-12)
    # Forward-only when product is absent.
    assert float(law.rate(400.0, jnp.array([1.0, 0.0]))) > 0.0
    # Net reverse when product is in large excess.
    assert float(law.rate(400.0, jnp.array([0.1, 5.0]))) < 0.0


def test_lhhw_adsorption_inhibits_rate() -> None:
    law = LHHW(
        a=jnp.asarray(5.0),
        ea=jnp.asarray(30e3),
        orders=jnp.array([1.0, 0.0]),
        k_ads=jnp.array([0.0, 4.0]),
        ads_orders=jnp.array([0.0, 1.0]),
        sites=2.0,
    )
    low_poison = float(law.rate(450.0, jnp.array([1.0, 0.1])))
    high_poison = float(law.rate(450.0, jnp.array([1.0, 5.0])))
    assert high_poison < low_poison
    # Explicit denominator check at one point: (1 + 4*c_poison)^2.
    k = float(arrhenius(450.0, 5.0, 30e3))
    assert high_poison == pytest.approx(k * 1.0 / (1.0 + 4.0 * 5.0) ** 2)


def test_rate_gradient_wrt_activation_energy() -> None:
    law = PowerLaw(a=jnp.asarray(2.0), ea=jnp.asarray(55e3), orders=jnp.array([1.0]))
    c = jnp.array([2.0])
    t = 400.0
    r = float(law.rate(t, c))
    dr_dea = jax.grad(lambda ea: PowerLaw(law.a, ea, law.orders).rate(t, c))(jnp.asarray(55e3))
    # d r / d Ea = -r / (R T)
    assert float(dr_dea) == pytest.approx(-r / (R * t), rel=1e-6)


def test_rate_gradient_wrt_concentration() -> None:
    law = PowerLaw(a=jnp.asarray(1.0), ea=jnp.asarray(0.0), orders=jnp.array([2.0, 1.0]))
    grad = jax.grad(lambda c: law.rate(300.0, c))(jnp.array([3.0, 4.0]))
    # r = c0^2 c1 -> dr/dc0 = 2 c0 c1, dr/dc1 = c0^2
    assert float(grad[0]) == pytest.approx(2 * 3.0 * 4.0)
    assert float(grad[1]) == pytest.approx(3.0**2)


@pytest.mark.parametrize(
    ("law", "n_leaves"),
    [
        (Arrhenius(jnp.asarray(1.0), jnp.asarray(1.0)), 2),
        (PowerLaw(jnp.asarray(1.0), jnp.asarray(1.0), jnp.ones(3)), 3),
        (
            MassActionReversible(
                jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(1.0), jnp.ones(2)
            ),
            5,
        ),
        (LHHW(jnp.asarray(1.0), jnp.asarray(1.0), jnp.ones(2), jnp.ones(2), jnp.ones(2)), 5),
    ],
)
def test_rate_laws_are_pytrees(law: object, n_leaves: int) -> None:
    assert len(jax.tree_util.tree_leaves(law)) == n_leaves
