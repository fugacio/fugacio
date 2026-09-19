"""The unit contract: inventories, specifications, operating limits, and failures.

Every unit reports failure instead of returning a plausible wrong state: an
eager call raises (`ConvergenceError` for a failed solve, `ValueError` for a
violated operating limit), a flowsheet records the failing unit's report, and
a traced call returns NaN. Pure fluids keep their phase inventory between units,
and specifications that can't determine a state are rejected by name.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import (
    Flowsheet,
    Stream,
    adiabatic_flash,
    component_separator,
    compressor,
    decanter,
    flash_drum,
    heat_exchanger,
    heater,
    mix,
    package_for,
    pump,
    splitter,
    three_phase_flash,
    turbine,
    valve,
)
from fugacio.sim.acceptance import flash_drum_checked
from fugacio.thermo import flash_lle_with_info
from fugacio.thermo.acceptance import PhysicalAcceptanceError, PhysicalAcceptanceWarning
from fugacio.thermo.diagnostics import ConvergenceError

WATER = ("water",)
LIGHT = ("methane", "propane", "n-pentane")
ATM = 101325.0


def _light(flow: float = 10.0, t: float = 320.0, p: float = 20e5) -> Stream:
    return Stream.from_fractions(LIGHT, jnp.array([0.5, 0.3, 0.2]), flow, t, p)


# --------------------------------------------------------------------------- #
# Pure-fluid inventories
# --------------------------------------------------------------------------- #


def test_saturated_water_letdown_keeps_its_inventory_through_a_drum() -> None:
    pkg = package_for(WATER, "pr")
    psat = float(pkg.bubble_pressure(425.0, jnp.ones(1)).value)
    assert psat == pytest.approx(4.92e5, rel=0.01)
    feed = Stream.from_fractions(WATER, jnp.ones(1), 10.0, 425.0, psat, phase="liquid")
    letdown = valve(feed, 1e5, model=pkg)
    flashed = float(jnp.sum(letdown.vapor_n))
    assert flashed == pytest.approx(1.0175, abs=2e-3)
    # A drum at the outlet's own saturation state can't re-decide the split
    # (every quality has the same T and P): it keeps the valve's inventory.
    vapor, liquid = flash_drum(letdown, letdown.t, letdown.p, model=pkg)
    assert float(vapor.total) == pytest.approx(flashed, abs=1e-9)
    assert float(liquid.total) == pytest.approx(10.0 - flashed, abs=1e-9)
    # An adiabatic drum at the letdown pressure reaches the same split directly.
    drum = adiabatic_flash(feed, 1e5, model=pkg)
    assert float(drum.vapor.total) == pytest.approx(flashed, rel=1e-8)


def test_condenser_on_a_vapor_fraction_specification() -> None:
    # Cengel and Boles, Example 10-2: an ideal Rankine cycle with a 3 MPa, 350 C
    # turbine inlet and a 75 kPa condenser rejects 2018.6 kJ per kg of steam.
    pkg = package_for(WATER, "iapws")
    flow = 1.0 / 0.018015268  # 1 kg/s
    steam = Stream.from_fractions(WATER, jnp.ones(1), flow, 623.15, 3e6, phase="vapor")
    exhaust = turbine(steam, 75e3, efficiency=1.0, model=pkg).outlet
    quality = float(jnp.sum(exhaust.vapor_n) / exhaust.total)
    assert quality == pytest.approx(0.8861, abs=1e-3)
    condenser = heater(exhaust, vapor_fraction=0.0, model=pkg)
    assert float(condenser.duty) / 1e3 == pytest.approx(-2018.6, rel=1e-3)
    assert float(jnp.sum(condenser.outlet.vapor_n)) == 0.0
    # A temperature exactly on the saturation line leaves the quality open.
    t_sat = pkg.bubble_temperature(75e3, jnp.ones(1)).value
    with pytest.raises(ValueError, match="vapor_fraction"):
        heater(exhaust, t_out=t_sat, model=pkg)


# --------------------------------------------------------------------------- #
# Operating limits
# --------------------------------------------------------------------------- #


def test_violated_operating_limits_raise_with_the_rule() -> None:
    feed = _light()
    gas = Stream.from_fractions(("methane",), jnp.ones(1), 1.0, 300.0, 5e5)
    with pytest.raises(ValueError, match="sum to one"):
        splitter(feed, [0.5, 0.7])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        splitter(feed, [-0.2, 1.2])
    with pytest.raises(ValueError, match="recoveries"):
        component_separator(feed, [1.5, 0.5, 0.0])
    with pytest.raises(ValueError, match="liquid inlet"):
        pump(gas, 10e5)
    with pytest.raises(ValueError, match="at least the inlet pressure"):
        compressor(gas, 1e5)
    with pytest.raises(ValueError, match="at most the inlet pressure"):
        turbine(gas, 10e5)
    with pytest.raises(ValueError, match="raise pressure"):
        valve(feed, 30e5)


def test_traced_limit_violations_return_nan_instead_of_raising() -> None:
    feed = _light()

    @jax.jit
    def split(fraction):
        return splitter(feed, jnp.array([fraction, 0.7]))[0].n

    assert bool(jnp.all(jnp.isnan(split(0.5))))
    assert bool(jnp.all(jnp.isfinite(split(0.3))))


# --------------------------------------------------------------------------- #
# Flowsheets report the failing unit
# --------------------------------------------------------------------------- #


def test_infeasible_exchanger_fails_its_flowsheet_by_name() -> None:
    hot = _light(t=400.0)
    cold = _light(t=300.0)
    fs = Flowsheet()
    fs.feed("hot", hot)
    fs.feed("cold", cold)
    fs.unit(
        "exchanger",
        lambda h, c, th: heat_exchanger(h, c, t_cold_out=450.0),
        inputs=("hot", "cold"),
        outputs=("hot_out", "cold_out"),
    )
    result = fs.solve_with_info()
    assert not bool(result.converged)
    assert not bool(result.reports["unit:exchanger"].converged)
    with pytest.raises(ConvergenceError, match="exchanger"):
        fs.solve()
    # The equation-oriented exchanger solves the same balances but reports the cross.
    from fugacio.sim.eo import EOFlowsheet, HeatExchanger
    from fugacio.thermo.diagnostics import SolveStatus

    eo = EOFlowsheet()
    eo.feed("hot", hot)
    eo.feed("cold", cold)
    eo.add(HeatExchanger(inlets=("hot", "cold"), outlets=("hot_out", "cold_out"), t_cold_out=450.0))
    solution = eo.solve({}, check=False)
    assert int(solution.report.status) == int(SolveStatus.INFEASIBLE)
    with pytest.raises(ConvergenceError):
        eo.solve({})


def test_flowsheet_audit_uses_the_heat_its_units_retained() -> None:
    from fugacio.sim.acceptance import BalanceBoundary, audit_flowsheet
    from fugacio.sim.eo import EOFlowsheet, Heater

    pkg = package_for(LIGHT)
    feed = _light(t=300.0, p=1e5)
    fs = Flowsheet(model=pkg)
    fs.feed("feed", feed)
    fs.unit("heater", lambda s, th: heater(s, t_out=350.0), inputs=("feed",), outputs=("hot",))
    result = fs.solve_with_info()
    assert float(result.units["heater"].heat) > 0.0
    closed = {"heater": BalanceBoundary(("feed",), ("hot",), units=("heater",))}
    assert audit_flowsheet(result, pkg, boundaries=closed)["accepted"]
    adiabatic = {"heater": BalanceBoundary(("feed",), ("hot",))}
    assert not audit_flowsheet(result, pkg, boundaries=adiabatic)["accepted"]
    with pytest.raises(ValueError, match="no heat or work"):
        audit_flowsheet(result, pkg, boundaries={"x": BalanceBoundary((), (), units=("y",))})
    # An equation-oriented solution is audited the same way.
    eo = EOFlowsheet(model=pkg)
    eo.feed("feed", feed)
    eo.add(Heater(inlets=("feed",), outlets=("hot",), t_out=350.0))
    solution = eo.solve({})
    duty = float(result.units["heater"].heat)
    declared = {"heater": BalanceBoundary(("feed",), ("hot",), heat=duty)}
    assert audit_flowsheet(solution, pkg, boundaries=declared)["accepted"]


def _liquid_recycle(fraction: float) -> Flowsheet:
    fs = Flowsheet()
    fs.feed("fresh", _light())
    fs.unit(
        "mixer",
        lambda fresh, recycle, th: mix([fresh, recycle]),
        inputs=("fresh", "recycle"),
        outputs=("mixed",),
    )
    fs.unit(
        "drum",
        lambda mixed, th: flash_drum(mixed, 320.0, 20e5),
        inputs=("mixed",),
        outputs=("vapor", "liquid"),
    )
    fs.unit(
        "split",
        lambda liquid, th: splitter(liquid, jnp.array([th, 1.0 - th])),
        inputs=("liquid",),
        outputs=("recycle", "purge"),
    )
    return fs


def test_a_99_percent_liquid_recycle_converges_with_default_settings() -> None:
    result = _liquid_recycle(0.99).solve_with_info(jnp.asarray(0.99))
    assert bool(result.converged)
    (recycle,) = [r for name, r in result.reports.items() if name.startswith("recycle:")]
    assert int(recycle.iterations) <= 50
    fresh, vapor, purge = result["fresh"], result["vapor"], result["purge"]
    assert float(jnp.max(jnp.abs(fresh.n - vapor.n - purge.n))) < 1e-6


def test_a_loop_without_feeds_asks_for_a_tear_guess() -> None:
    fs = Flowsheet()
    fs.unit("a", lambda s, th: s, inputs=("x",), outputs=("y",))
    fs.unit("b", lambda s, th: s, inputs=("y",), outputs=("x",))
    with pytest.raises(ValueError, match=r"Flowsheet\.tear"):
        fs.solve_with_info()


# --------------------------------------------------------------------------- #
# Liquid-liquid instability is surfaced, not hidden
# --------------------------------------------------------------------------- #


BUTANOL_WATER = ("water", "1-butanol")


def test_vapor_liquid_drum_warns_about_a_second_liquid() -> None:
    pkg = package_for(BUTANOL_WATER, "nrtl")
    feed = Stream.from_fractions(BUTANOL_WATER, jnp.array([0.7, 0.3]), 10.0, 298.15, ATM)
    with pytest.warns(PhysicalAcceptanceWarning, match="two liquids"):
        flash_drum(feed, 298.15, ATM, model=pkg)
    checked = flash_drum_checked(feed, 298.15, ATM, model=pkg)
    assert not bool(checked.accepted)
    with pytest.raises(PhysicalAcceptanceError, match="two liquids"):
        checked.check()


def test_decanter_matches_the_liquid_liquid_flash() -> None:
    pkg = package_for(BUTANOL_WATER, "nrtl")
    z = jnp.array([0.7, 0.3])
    feed = Stream.from_fractions(BUTANOL_WATER, z, 10.0, 300.0, ATM)
    liquid_i, liquid_ii = decanter(feed, pkg, t=300.0)
    psi = float(flash_lle_with_info(pkg.activity, 300.0, z).value.psi)
    share = float(liquid_ii.total / feed.total)
    assert min(abs(share - psi), abs(share - (1.0 - psi))) < 1e-6
    assert float(jnp.max(jnp.abs(liquid_i.n + liquid_ii.n - feed.n))) < 1e-9


def test_three_phase_flash_of_a_two_phase_feed_names_the_right_unit() -> None:
    pkg = package_for(BUTANOL_WATER, "nrtl")
    feed = Stream.from_fractions(BUTANOL_WATER, jnp.array([0.7, 0.3]), 10.0, 300.0, ATM)
    with pytest.raises(ConvergenceError, match="decanter"):
        three_phase_flash(feed, 300.0, ATM, pkg)


# --------------------------------------------------------------------------- #
# Absent components
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["uniquac", "unifac"])
def test_an_absent_component_stays_absent_through_a_flash(method) -> None:
    names = ("ethanol", "water", "methanol")
    pkg = package_for(names, method, parameter_policy="allow_ideal")
    exact = pkg.flash_pt(355.0, ATM, jnp.array([0.5, 0.5, 0.0]))
    trace = pkg.flash_pt(355.0, ATM, jnp.array([0.5, 0.5 - 1e-12, 1e-12]))
    assert bool(jnp.all(jnp.isfinite(exact.x))) and bool(jnp.all(jnp.isfinite(exact.y)))
    assert float(exact.x[2]) == 0.0 and float(exact.y[2]) == 0.0
    assert float(exact.beta) == pytest.approx(float(trace.beta), abs=1e-6)


# --------------------------------------------------------------------------- #
# Reactors
# --------------------------------------------------------------------------- #


def test_an_overdrawn_stoichiometric_extent_is_rejected() -> None:
    from fugacio.sim import stoichiometric_reactor
    from fugacio.thermo.reactions import Reaction

    names = ("n-butane", "isobutane")
    feed = Stream(jnp.array([1.0, 0.0]), jnp.asarray(350.0), jnp.asarray(3e5), names)
    reaction = Reaction.of(names, {"n-butane": 1}, {"isobutane": 1})
    with pytest.raises(ValueError, match="more of a reactant"):
        stoichiometric_reactor(feed, reaction, extent=[2.5])


def test_adiabatic_batch_reactor_conserves_internal_energy() -> None:
    from fugacio.sim import batch_reactor
    from fugacio.thermo.constants import R
    from fugacio.thermo.ideal import enthalpy_ig
    from fugacio.thermo.kinetics import PowerLaw
    from fugacio.thermo.reactions import Reaction, reaction_arrays

    names = ("n-butane", "isobutane")
    reaction = Reaction.of(names, {"n-butane": 1}, {"isobutane": 1})
    law = PowerLaw(a=jnp.asarray(2.0e3), ea=jnp.asarray(30e3), orders=jnp.array([1.0, 0.0]))
    feed = Stream(jnp.array([10.0, 0.0]), jnp.asarray(350.0), jnp.asarray(3e5), names)
    volume = 10.0 * R * 350.0 / 3e5
    result = batch_reactor(feed, reaction, law, volume, 20.0, adiabatic=True, steps=400)
    hf, _gf, coeffs = reaction_arrays(list(names))

    def internal_energy(n, t):
        return float(jnp.sum(n * (hf + enthalpy_ig(t, *coeffs))) - R * t * jnp.sum(n))

    before = internal_energy(feed.n, 350.0)
    after = internal_energy(result.contents.n, float(result.contents.t))
    assert after == pytest.approx(before, rel=1e-8)
    assert float(result.contents.t) > 350.0  # the isomerization is exothermic
    expected_p = float(jnp.sum(result.contents.n)) * R * float(result.contents.t) / volume
    assert float(result.contents.p) == pytest.approx(expected_p, rel=1e-12)


def test_drum_warning_is_not_emitted_for_a_stable_split() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", PhysicalAcceptanceWarning)
        flash_drum(_light(), 320.0, 20e5)
