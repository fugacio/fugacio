"""Assembling dynamic units + controllers into one differentiable flowsheet ODE.

These check the three things :class:`DynamicFlowsheet` promises: that port-to-port
wiring routes outlets into downstream inlets, that a closed control loop drives its
measurement to the setpoint, and that the whole simulation is differentiable with
respect to parameters threaded through ``theta``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import Stream, pi
from fugacio.sim.dynamics import DynamicFlowsheet, DynamicResult, MixingTank, ThermalMass


def _water(n: list[float], t: float = 290.0) -> Stream:
    return Stream(
        n=jnp.asarray(n), t=jnp.asarray(t), p=jnp.asarray(101325.0), components=("water",)
    )


def test_closed_loop_temperature_control_reaches_setpoint() -> None:
    mass = ThermalMass(
        name="H", components=("water",), holdup=200.0, ua=20.0, t_ambient=290.0, t0=300.0
    )
    ctrl = pi(kc=300.0, tau_i=120.0, u_min=0.0, u_max=5.0e4)
    fs = DynamicFlowsheet()
    fs.feed("feed", _water([5.0], t=290.0))
    fs.add(mass, inputs=["feed"])
    fs.control(ctrl, measurement=("H", "temperature"), setpoint=330.0, actuator=("H", "duty"))

    res = fs.simulate(ts=jnp.linspace(0.0, 6000.0, 601))
    temperature = res.measurement("H", "temperature")
    assert isinstance(res, DynamicResult)
    assert float(temperature[0]) == pytest.approx(300.0, abs=1e-6)
    assert float(temperature[-1]) == pytest.approx(330.0, abs=0.5)
    assert "H.duty" in res.controls


def test_result_structure() -> None:
    comps = ("ethanol", "water")
    fs = DynamicFlowsheet()
    fs.feed(
        "feed",
        Stream(
            n=jnp.array([0.0, 2.0]), t=jnp.asarray(290.0), p=jnp.asarray(101325.0), components=comps
        ),
    )
    fs.add(
        MixingTank(name="M", components=comps, holdup=5.0, x0=jnp.array([1.0, 0.0])),
        inputs=["feed"],
    )
    ts = jnp.linspace(0.0, 50.0, 26)
    res = fs.simulate(ts)
    assert res.ts.shape == (26,)
    assert "M" in res.states
    assert "frac" in res.measurements["M"]


def test_units_in_series_propagate_through_ports() -> None:
    comps = ("ethanol", "water")
    feed = Stream(
        n=jnp.array([0.0, 2.0]), t=jnp.asarray(290.0), p=jnp.asarray(101325.0), components=comps
    )  # pure water feed
    fs = DynamicFlowsheet()
    fs.feed("feed", feed)
    fs.add(
        MixingTank(name="M1", components=comps, holdup=4.0, x0=jnp.array([1.0, 0.0])),
        inputs=["feed"],
    )
    fs.add(
        MixingTank(name="M2", components=comps, holdup=4.0, x0=jnp.array([1.0, 0.0])),
        inputs=["M1.0"],
    )  # second tank fed by the first tank's outlet port

    res = fs.simulate(ts=jnp.linspace(0.0, 120.0, 241))
    # Both tanks eventually flush to the pure-water inlet composition.
    assert jnp.allclose(res.measurement("M1", "frac")[-1], jnp.array([0.0, 1.0]), atol=1e-2)
    assert jnp.allclose(res.measurement("M2", "frac")[-1], jnp.array([0.0, 1.0]), atol=1e-2)
    # The downstream tank lags the upstream one (series of two first-order lags).
    half = res.ts.shape[0] // 2
    assert float(res.measurement("M2", "frac")[half, 1]) < float(
        res.measurement("M1", "frac")[half, 1]
    )


def test_open_loop_manipulation_schedule() -> None:
    fs = DynamicFlowsheet()
    fs.add(
        ThermalMass(
            name="H", components=("water",), holdup=100.0, ua=20.0, t_ambient=290.0, t0=290.0
        )
    )
    fs.manipulate("H", "duty", lambda t, theta: jnp.asarray(2000.0))
    res = fs.simulate(ts=jnp.linspace(0.0, 4000.0, 401))
    # Steady temperature = ambient + Q/UA.
    assert float(res.measurement("H", "temperature")[-1]) == pytest.approx(290.0 + 100.0, abs=0.5)


def test_flowsheet_simulation_is_differentiable_in_theta() -> None:
    fs = DynamicFlowsheet()
    fs.add(
        ThermalMass(
            name="H", components=("water",), holdup=100.0, ua=20.0, t_ambient=290.0, t0=290.0
        )
    )
    # Duty is the differentiable parameter; final temperature should rise with it.
    fs.manipulate("H", "duty", lambda t, theta: theta)
    ts = jnp.linspace(0.0, 3000.0, 301)

    def final_temperature(duty: jnp.ndarray) -> jnp.ndarray:
        return fs.simulate(ts, theta=duty).measurement("H", "temperature")[-1]

    grad = float(jax.grad(final_temperature)(jnp.asarray(1500.0)))
    assert jnp.isfinite(grad)
    assert grad > 0.0  # more duty -> hotter
