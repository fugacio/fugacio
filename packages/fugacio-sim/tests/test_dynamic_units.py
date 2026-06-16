"""Dynamic unit operations: balances, constitutive relations, and steady states.

Each dynamic unit is a holdup balance closed with the steady-state thermodynamics,
so the checks are conservation laws and steady states with known closed forms: a
surge tank accumulates exactly its net inflow, a constant-holdup mixer is a
composition first-order lag at the residence time, a heated thermal mass settles
at ``T_ambient + Q/UA``, an isothermal CSTR hits ``C_in/(1 + k tau)``, a gas
receiver follows the ideal-gas law, and a flash drum draws an equilibrium vapour.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from fugacio.sim import Stream, odeint, odeint_final
from fugacio.sim.dynamics import (
    DynamicCSTR,
    DynamicFlash,
    GasReceiver,
    LevelTank,
    MixingTank,
    ThermalMass,
)
from fugacio.thermo.constants import R
from fugacio.thermo.kinetics import PowerLaw, arrhenius
from fugacio.thermo.reactions import Reaction


# --------------------------------------------------------------------------- #
# Level / surge tank
# --------------------------------------------------------------------------- #
def test_level_tank_accumulates_net_inflow() -> None:
    comps = ("ethanol", "water")
    tank = LevelTank(name="T", components=comps, area=2.0, n0=jnp.array([5.0, 5.0]))
    feed = Stream(n=jnp.array([1.0, 0.5]), t=298.15, p=101325.0, components=comps)

    # No outlet (flow setpoint 0): holdup grows by exactly the inflow.
    final = odeint_final(
        lambda t, n, th: tank.evaluate(n, (feed,)).dstate,
        tank.initial_state(),
        0.0,
        10.0,
        method="rk4",
        steps=200,
    )
    assert jnp.allclose(final, jnp.array([5.0, 5.0]) + jnp.array([1.0, 0.5]) * 10.0, atol=1e-6)


def test_level_tank_constant_holdup_when_balanced() -> None:
    comps = ("ethanol", "water")
    tank = LevelTank(name="T", components=comps, n0=jnp.array([4.0, 6.0]), flow_setpoint=1.5)
    feed = Stream(n=jnp.array([0.9, 0.6]), t=298.15, p=101325.0, components=comps)  # total 1.5
    step = tank.evaluate(tank.initial_state(), (feed,))
    assert float(jnp.sum(step.dstate)) == pytest.approx(0.0, abs=1e-9)


def test_level_tank_measured_level() -> None:
    comps = ("water",)
    tank = LevelTank(name="T", components=comps, area=0.5, n0=jnp.array([100.0]))
    meas = tank.measured(tank.initial_state())
    assert set(meas) == {"level", "volume", "holdup"}
    assert float(meas["level"]) == pytest.approx(float(meas["volume"]) / 0.5, rel=1e-9)
    assert float(meas["holdup"]) == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Constant-holdup mixing tank (composition lag)
# --------------------------------------------------------------------------- #
def test_mixing_tank_relaxes_to_inlet_composition() -> None:
    comps = ("ethanol", "water")
    tank = MixingTank(name="M", components=comps, holdup=10.0, x0=jnp.array([1.0, 0.0]))
    feed = Stream(n=jnp.array([0.0, 2.0]), t=298.15, p=101325.0, components=comps)  # pure water
    ts = jnp.linspace(0.0, 60.0, 121)
    traj = odeint(
        lambda t, x, th: tank.evaluate(x, (feed,)).dstate,
        tank.initial_state(),
        ts,
        method="rk4",
        substeps=10,
    )
    # Outlet composition relaxes to pure water; fractions still sum to one.
    assert jnp.allclose(traj[-1], jnp.array([0.0, 1.0]), atol=1e-3)
    assert jnp.allclose(jnp.sum(traj, axis=1), 1.0, atol=1e-9)


def test_mixing_tank_time_constant_is_residence_time() -> None:
    comps = ("ethanol", "water")
    holdup, f_in = 10.0, 2.0
    tank = MixingTank(name="M", components=comps, holdup=holdup, x0=jnp.array([1.0, 0.0]))
    feed = Stream(n=jnp.array([0.0, f_in]), t=298.15, p=101325.0, components=comps)
    tau = holdup / f_in
    # After one residence time the step is ~63.2% complete.
    x_tau = odeint_final(
        lambda t, x, th: tank.evaluate(x, (feed,)).dstate,
        tank.initial_state(),
        0.0,
        tau,
        method="rk4",
        steps=200,
    )
    assert float(x_tau[1]) == pytest.approx(1.0 - jnp.exp(-1.0), rel=1e-3)


# --------------------------------------------------------------------------- #
# Thermal mass (temperature dynamics)
# --------------------------------------------------------------------------- #
def test_thermal_mass_settles_at_ambient_plus_q_over_ua() -> None:
    mass = ThermalMass(
        name="H",
        components=("water",),
        holdup=100.0,
        ua=20.0,
        t_ambient=298.15,
        duty_setpoint=2000.0,
        t0=298.15,
    )
    t_final = odeint_final(
        lambda t, T, th: mass.evaluate(T, ()).dstate,
        mass.initial_state(),
        0.0,
        4000.0,
        method="rk4",
        steps=400,
    )
    assert float(t_final) == pytest.approx(298.15 + 2000.0 / 20.0, abs=0.5)


def test_thermal_mass_relaxes_to_inlet_temperature() -> None:
    comps = ("water",)
    mass = ThermalMass(name="H", components=comps, holdup=50.0, ua=0.0, duty_setpoint=0.0, t0=350.0)
    feed = Stream(n=jnp.array([5.0]), t=300.0, p=101325.0, components=comps)
    t_final = odeint_final(
        lambda t, T, th: mass.evaluate(T, (feed,)).dstate,
        mass.initial_state(),
        0.0,
        400.0,
        method="rk4",
        steps=400,
    )
    assert float(t_final) == pytest.approx(300.0, abs=1.0)


# --------------------------------------------------------------------------- #
# Dynamic CSTR
# --------------------------------------------------------------------------- #
def test_isothermal_cstr_matches_analytic_conversion() -> None:
    comps = ("n-butane", "isobutane")
    rxn = Reaction.parse("n-butane -> isobutane", comps)
    a, ea, t_r = 1.0e6, 50_000.0, 350.0
    law = PowerLaw(a=a, ea=ea, orders=jnp.array([1.0, 0.0]))
    volume, q, c_in = 1.0, 0.1, 9000.0
    cstr = DynamicCSTR(
        name="R",
        components=comps,
        reactions=rxn,
        rate_laws=law,
        volume=volume,
        q=q,
        rho_cp=1.0e13,
        ua=0.0,
        jacket_t=t_r,  # huge rho_cp pins T
        c0=jnp.array([c_in, 0.0]),
        t0=t_r,
        p=3.0e5,
    )
    feed = Stream(n=jnp.array([q * c_in, 0.0]), t=t_r, p=3.0e5, components=comps)
    final = odeint_final(
        lambda t, st, th: cstr.evaluate(st, (feed,)).dstate,
        cstr.initial_state(),
        0.0,
        500.0,
        method="rk4",
        steps=500,
    )
    k = float(arrhenius(t_r, a, ea))
    tau = volume / q
    ca_expected = c_in / (1.0 + k * tau)
    assert float(final.t) == pytest.approx(t_r, abs=0.1)  # stayed isothermal
    assert float(final.c[0]) == pytest.approx(ca_expected, rel=1e-3)


def test_exothermic_cstr_heats_up() -> None:
    comps = ("n-butane", "isobutane")
    rxn = Reaction.parse("n-butane -> isobutane", comps)
    law = PowerLaw(a=1.0e8, ea=70_000.0, orders=jnp.array([1.0, 0.0]))
    cstr = DynamicCSTR(
        name="R",
        components=comps,
        reactions=rxn,
        rate_laws=law,
        volume=1.0,
        q=0.01,
        rho_cp=1.5e6,
        ua=0.0,
        jacket_t=360.0,
        c0=jnp.array([9000.0, 0.0]),
        t0=360.0,
        p=2.0e6,
    )
    feed = Stream(n=jnp.array([0.01 * 9000.0, 0.0]), t=360.0, p=2.0e6, components=comps)
    final = odeint_final(
        lambda t, st, th: cstr.evaluate(st, (feed,)).dstate,
        cstr.initial_state(),
        0.0,
        4000.0,
        method="rk4",
        steps=800,
    )
    assert float(final.t) > 360.0 + 20.0  # exothermic isomerisation self-heats


# --------------------------------------------------------------------------- #
# Gas receiver (pressure dynamics)
# --------------------------------------------------------------------------- #
def test_gas_receiver_ideal_gas_pressure() -> None:
    gas = GasReceiver(name="V", components=("methane",), volume=2.0, t=300.0, n0=jnp.array([100.0]))
    meas = gas.measured(gas.initial_state())
    assert float(meas["pressure"]) == pytest.approx(100.0 * float(R) * 300.0 / 2.0, rel=1e-9)


def test_gas_receiver_pressure_rises_with_inventory() -> None:
    comps = ("methane",)
    gas = GasReceiver(
        name="V",
        components=comps,
        volume=2.0,
        t=300.0,
        outlet="flow",
        flow_setpoint=0.0,
        n0=jnp.array([50.0]),
    )
    feed = Stream(n=jnp.array([1.0]), t=300.0, p=1.0e6, components=comps)
    ts = jnp.linspace(0.0, 100.0, 51)
    traj = odeint(
        lambda t, n, th: gas.evaluate(n, (feed,)).dstate,
        gas.initial_state(),
        ts,
        method="rk4",
        substeps=4,
    )
    # Closed outlet + steady feed -> linear inventory growth -> linear pressure rise.
    assert float(traj[-1, 0]) == pytest.approx(50.0 + 1.0 * 100.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Dynamic flash
# --------------------------------------------------------------------------- #
def test_dynamic_flash_draws_enriched_vapor() -> None:
    comps = ("propane", "n-butane")
    # 300 K / 5 bar sits inside the two-phase envelope for a 50/50 propane/n-butane mix.
    flash = DynamicFlash(
        name="F",
        components=comps,
        t=300.0,
        p=5.0e5,
        vapor_draw=0.1,
        liquid_draw=0.1,
        m0=jnp.array([5.0, 5.0]),
    )
    state = flash.initial_state()
    step = flash.evaluate(
        state, (Stream(n=jnp.array([0.1, 0.1]), t=300.0, p=5.0e5, components=comps),)
    )
    meas = flash.measured(state)
    y = step.measurements["y"]
    # Propane is the lighter (more volatile) species: vapour is enriched in it.
    assert float(y[0]) > float(meas["x"][0])


def test_dynamic_flash_measured_keys() -> None:
    comps = ("propane", "n-butane")
    flash = DynamicFlash(name="F", components=comps, m0=jnp.array([3.0, 7.0]))
    meas = flash.measured(flash.initial_state())
    assert set(meas) == {"holdup", "x"}
    assert float(meas["holdup"]) == pytest.approx(10.0)
    assert jnp.allclose(meas["x"], jnp.array([0.3, 0.7]), atol=1e-9)
