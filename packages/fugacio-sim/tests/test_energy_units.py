"""Energy-balanced unit operations: heater, valve, pump, compressor, turbine, splits.

These exercise the rigorous material+energy balances and confirm the blocks stay
differentiable with respect to their operating conditions (AD vs finite
difference), which is the property the whole flowsheet engine is built on.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import (
    Stream,
    component_separator,
    compressor,
    enthalpy_flow,
    heater,
    mass_flow,
    molar_enthalpy,
    pump,
    splitter,
    turbine,
    valve,
)

GAS = ("methane", "propane")
LIQ = ("propane", "n-pentane")


def _gas(flow: float = 100.0, t: float = 300.0, p: float = 5e5) -> Stream:
    return Stream.from_fractions(GAS, jnp.array([0.5, 0.5]), flow, t, p)


def _liquid(flow: float = 50.0, t: float = 300.0, p: float = 20e5) -> Stream:
    return Stream.from_fractions(LIQ, jnp.array([0.5, 0.5]), flow, t, p)


def test_heater_temperature_spec_sets_duty_sign() -> None:
    feed = _gas()
    hot = heater(feed, t_out=360.0)
    cold = heater(feed, t_out=260.0)
    assert float(hot.duty) > 0.0  # heating adds energy
    assert float(cold.duty) < 0.0  # cooling removes energy
    assert float(hot.outlet.t) == pytest.approx(360.0)


def test_heater_duty_spec_recovers_temperature() -> None:
    feed = _gas()
    spec = heater(feed, t_out=355.0)
    back = heater(feed, duty=spec.duty)
    assert float(back.outlet.t) == pytest.approx(355.0, abs=0.2)


def test_heater_duty_is_differentiable_in_outlet_temperature() -> None:
    feed = _gas()

    def duty(t_out: float) -> jax.Array:
        return heater(feed, t_out=t_out).duty

    g = float(jax.grad(duty)(340.0))
    fd = float((duty(341.0) - duty(339.0)) / 2.0)
    assert g == pytest.approx(fd, rel=1e-3)
    assert g > 0.0  # duty rises with target temperature (it is a heat capacity flow)


def test_valve_is_isenthalpic() -> None:
    feed = _gas(p=50e5)
    out = valve(feed, 5e5)
    assert float(out.p) == pytest.approx(5e5)
    assert float(molar_enthalpy(out)) == pytest.approx(float(molar_enthalpy(feed)), rel=1e-6)


def test_pump_raises_pressure_and_does_work() -> None:
    feed = _liquid(p=20e5)
    res = pump(feed, 60e5, efficiency=0.75)
    assert float(res.outlet.p) == pytest.approx(60e5)
    assert float(res.work) > 0.0  # work is put into the fluid
    assert float(res.outlet.t) >= float(feed.t) - 1e-6  # inefficiency warms the liquid


def test_compressor_work_and_heating() -> None:
    feed = _gas(p=5e5)
    res = compressor(feed, 20e5, efficiency=0.75)
    assert float(res.outlet.p) == pytest.approx(20e5)
    assert float(res.work) > 0.0
    assert float(res.work) > float(res.ideal_work) > 0.0  # actual exceeds reversible
    assert float(res.outlet.t) > float(feed.t)  # compression heats the gas


def test_compressor_work_is_differentiable_in_pressure() -> None:
    feed = _gas(p=5e5)

    def work(p_out: float) -> jax.Array:
        return compressor(feed, p_out, efficiency=0.75).work

    g = float(jax.grad(work)(18e5))
    fd = float((work(18e5 + 1e3) - work(18e5 - 1e3)) / 2e3)
    assert g == pytest.approx(fd, rel=1e-3)
    assert g > 0.0  # compressing to higher pressure costs more work


def test_turbine_produces_work() -> None:
    feed = _gas(t=400.0, p=30e5)
    res = turbine(feed, 5e5, efficiency=0.85)
    assert float(res.outlet.p) == pytest.approx(5e5)
    assert float(res.work) < 0.0  # the fluid delivers work to the shaft
    assert float(res.ideal_work) < float(res.work) < 0.0  # recovers only a fraction
    assert float(res.outlet.t) < float(feed.t)  # expansion cools the gas


def test_splitter_conserves_flow() -> None:
    feed = _gas(flow=100.0)
    a, b = splitter(feed, jnp.array([0.3, 0.7]))
    assert float(a.total) == pytest.approx(30.0)
    assert float(b.total) == pytest.approx(70.0)
    assert float(jnp.max(jnp.abs(a.n + b.n - feed.n))) < 1e-9
    assert float(a.t) == pytest.approx(float(feed.t))


def test_component_separator_recovers_components() -> None:
    feed = _gas(flow=100.0)
    top, bottom = component_separator(feed, jnp.array([0.95, 0.05]))
    # methane mostly overhead, propane mostly bottoms
    assert float(top.n[0]) == pytest.approx(0.95 * float(feed.n[0]))
    assert float(bottom.n[1]) == pytest.approx(0.95 * float(feed.n[1]))
    assert float(jnp.max(jnp.abs(top.n + bottom.n - feed.n))) < 1e-9


def test_enthalpy_flow_scales_with_total() -> None:
    feed = _gas(flow=10.0)
    big = _gas(flow=20.0)
    assert float(enthalpy_flow(big)) == pytest.approx(2.0 * float(enthalpy_flow(feed)), rel=1e-9)


def test_mass_flow_is_positive_and_scales() -> None:
    feed = _gas(flow=10.0)
    m = float(mass_flow(feed))
    assert m > 0.0
    assert float(mass_flow(_gas(flow=30.0))) == pytest.approx(3.0 * m, rel=1e-9)
