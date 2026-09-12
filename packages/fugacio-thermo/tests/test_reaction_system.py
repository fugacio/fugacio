"""Reaction references, dimensional rates, conservation, and parameter derivatives."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fugacio.thermo import Reaction, ReactionSet, ReferenceRate
from fugacio.thermo.constants import R
from fugacio.thermo.reaction_system import element_matrix

NAMES = ("n-butane", "isobutane")


def law(**kw):
    values = dict(
        k_forward=jnp.array(2.0),
        ea_forward=jnp.array(12000.0),
        forward_orders=jnp.array([1.0, 0.0]),
        k_reverse=jnp.array(0.5),
        ea_reverse=jnp.array(8000.0),
        reverse_orders=jnp.array([0.0, 1.0]),
        reference_temperature=jnp.array(300.0),
    )
    return ReferenceRate(**{**values, **kw})


def test_reference_rate_is_dimensional_and_differentiable():
    rate = law()
    inputs = jnp.array([0.8, 0.2])
    assert rate.rate(300.0, inputs) == pytest.approx(1.5)
    t = 360.0
    expected = (
        2 * np.exp(-12000 / R * (1 / t - 1 / 300)) * 0.8
        - 0.5 * np.exp(-8000 / R * (1 / t - 1 / 300)) * 0.2
    )
    assert rate.rate(t, inputs) == pytest.approx(expected)
    f = jax.jit(lambda k: replace(rate, k_forward=k).rate(t, inputs))
    assert jax.grad(f)(2.0) == pytest.approx((f(2.001) - f(1.999)) / 0.002, rel=1e-8)


def test_detailed_balance_uses_same_equilibrium_constant():
    system = ReactionSet.from_reactions(
        Reaction(NAMES, jnp.array([-1.0, 1.0])), [law(detailed_balance=True)], rate_basis="activity"
    )
    ln_k = system.ln_equilibrium_constants(350.0)[0]
    activity = jnp.array([0.1, 0.1 * jnp.exp(ln_k)])
    assert system.rate_laws[0].rate(350.0, activity, ln_k) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="fugacity"):
        ReactionSet.from_reactions(
            Reaction(NAMES, jnp.array([-1.0, 1.0])),
            [law(detailed_balance=True)],
            rate_basis="normalized_concentration",
        )
    with pytest.raises(ValueError, match="elementary"):
        ReactionSet.from_reactions(
            Reaction(NAMES, jnp.array([-1.0, 1.0])),
            [law(detailed_balance=True, forward_orders=jnp.array([2.0, 0.0]))],
            rate_basis="activity",
        )


@pytest.mark.parametrize("nu", [[-1.0, 2.0], [0.0, 0.0], [float("nan"), 1.0]])
def test_invalid_stoichiometry_rejected(nu):
    with pytest.raises(ValueError):
        ReactionSet.from_reactions(Reaction(NAMES, jnp.array(nu)))


def test_dependent_reactions_and_ambiguous_rate_basis_rejected():
    r = Reaction(NAMES, jnp.array([-1.0, 1.0]))
    with pytest.raises(ValueError, match="independent"):
        ReactionSet.from_reactions([r, r])
    with pytest.raises(ValueError, match="requires activity"):
        ReactionSet.from_reactions(r, [law()])
    with pytest.raises(ValueError, match="positive"):
        ReactionSet.from_reactions(r, reference_concentration=0)
    elements, counts = element_matrix(NAMES)
    np.testing.assert_array_equal(np.asarray(counts) @ np.array([-1, 1]), np.zeros(len(elements)))


def test_thermochemistry_is_a_pytree_parameter():
    system = ReactionSet.from_reactions(Reaction(NAMES, jnp.array([-1.0, 1.0])))
    fn = jax.jit(lambda gf: replace(system, formation_gibbs=gf).ln_equilibrium_constants(350.0)[0])
    np.testing.assert_allclose(
        jax.grad(fn)(system.formation_gibbs), -system.nu[0] / (R * 298.15), rtol=1e-10
    )
