"""Phase-changing utility train: energy closure, sensitivity, and optimization."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import Stream, enthalpy_flow, heater, minimize, package_for, pump, valve
from fugacio.sim.properties import molar_enthalpy, vapor_fraction

pytestmark = pytest.mark.plant


@pytest.mark.parametrize("method", ["pr", "iapws"])
def test_phase_changing_utility_energy_and_quality(method):
    pkg = package_for(["water"], method)
    pressure = 1e5
    z = jnp.ones(1)
    ts, _ = pkg.bubble_temperature(pressure, z, t_min=300.0, t_max=450.0)
    hl = pkg.enthalpy(ts, pressure, z, phase="liquid")
    hv = pkg.enthalpy(ts, pressure, z, phase="vapor")
    feed = Stream.from_fractions(("water",), z, 2.0, 300.0, pressure)
    q = feed.total * (0.5 * (hl + hv) - molar_enthalpy(feed, model=pkg))
    boiler = heater(feed, duty=q, model=pkg)
    assert boiler.report.converged
    assert vapor_fraction(boiler.outlet, model=pkg) == pytest.approx(0.5, abs=1e-6)
    letdown = valve(boiler.outlet, 0.5e5, model=pkg)
    assert enthalpy_flow(letdown, model=pkg) == pytest.approx(
        float(enthalpy_flow(boiler.outlet, model=pkg)), abs=1e-3
    )
    condenser = heater(letdown, t_out=300.0, model=pkg)
    circulating = pump(condenser.outlet, pressure, model=pkg)
    returned = heater(circulating.outlet, t_out=300.0, model=pkg)
    balance = boiler.duty + condenser.duty + circulating.work + returned.duty
    assert float(balance) == pytest.approx(0.0, abs=1e-2)
    assert jnp.allclose(returned.outlet.n, feed.n)
    assert returned.outlet.t == feed.t and returned.outlet.p == feed.p


def test_utility_operating_cost_optimization_and_sensitivity():
    pkg = package_for(["water"])
    z = jnp.ones(1)
    p = 1e5
    ts, _ = pkg.bubble_temperature(p, z, t_min=300.0, t_max=450.0)
    hl = pkg.enthalpy(ts, p, z, phase="liquid")
    latent = pkg.enthalpy(ts, p, z, phase="vapor") - hl
    feed = Stream.from_fractions(("water",), z, 1.0, 300.0, p)
    h0 = molar_enthalpy(feed, model=pkg)

    def cost(fraction, _=None):
        heated = heater(feed, duty=hl + fraction * latent - h0, model=pkg)
        delivered = valve(heated.outlet, 0.5e5, model=pkg)
        quality = vapor_fraction(delivered, model=pkg)
        return (quality - 0.6) ** 2 + 0.01 * fraction

    derivative = jax.grad(cost)(jnp.asarray(0.3))
    finite_difference = (cost(0.3001) - cost(0.2999)) / 0.0002
    assert float(derivative) == pytest.approx(float(finite_difference), rel=1e-4)
    optimum = minimize(cost, jnp.asarray(0.3), bounds=(0.1, 0.9), tol=1e-8)
    optimum.check()
    assert 0.1 < float(optimum.x) < 0.9
    assert float(optimum.fun) < float(cost(0.3))
    assert abs(float(jax.grad(cost)(optimum.x))) < 1e-4
