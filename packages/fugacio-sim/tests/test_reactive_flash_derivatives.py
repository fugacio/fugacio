"""Regression for traced property and thermochemical parameters in reactive flash."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import ReactionSet, Stream, reactive_flash
from fugacio.sim.properties import resolve_package
from fugacio.thermo import Reaction, component_arrays, gamma_phi_model
from fugacio.thermo.activity.models import nrtl

NAMES = ("acetic acid", "ethanol", "ethyl acetate", "water")


def test_jitted_reactive_flash_differentiates_activity_feed_and_thermochemistry():
    constants = component_arrays(list(NAMES))
    system = ReactionSet.from_reactions(Reaction(NAMES, jnp.array([-1.0, -1.0, 1.0, 1.0])))
    alpha = 0.3 * (jnp.ones((4, 4)) - jnp.eye(4))
    direction = jnp.array(
        [[0.0, 0.2, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )

    def objective(q):
        activity = nrtl(a=q * direction, b=jnp.zeros((4, 4)), alpha=alpha)
        model = gamma_phi_model(activity, constants["tc"], constants["pc"], constants["omega"])
        rx = replace(
            system, formation_gibbs=system.formation_gibbs + q * jnp.array([0.0, 0.0, 10.0, 0.0])
        )
        feed = Stream(jnp.array([1.0 + 0.1 * q, 1.0, 1e-4, 1e-4]), 350.0, 101325.0, NAMES)
        return reactive_flash(feed, rx, 350.0 + 0.1 * q, 101325.0, model, check=False).extent[0]

    f = jax.jit(objective)
    grad = jax.jit(jax.grad(f))(0.0)
    assert jnp.isfinite(grad)
    assert grad == pytest.approx((f(0.001) - f(-0.001)) / 0.002, rel=2e-5, abs=1e-7)


def test_reactive_flash_non_equimolar_vapor_inventory_and_failure():
    names = ("ethane", "ethylene", "hydrogen")
    rx = Reaction(names, jnp.array([-1.0, 1.0, 1.0]))
    feed = Stream.from_fractions(names, [0.8, 0.1, 0.1], 10.0, 800.0, 1e5)
    pkg = resolve_package(names)
    result = reactive_flash(feed, rx, 800.0, 1e5, pkg)
    assert result.report.converged and result.phase_report.converged
    assert result.vapor.phase_known and result.liquid.phase_known
    assert result.vapor.total + result.liquid.total == pytest.approx(
        feed.total + result.extent[0], rel=1e-8
    )
    assert jnp.isfinite(result.duty)
    failed = reactive_flash(feed, rx, 800.0, 1e5, pkg, max_iter=0, check=False)
    assert not failed.report.converged
