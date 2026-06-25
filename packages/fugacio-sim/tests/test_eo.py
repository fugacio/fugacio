"""Equation-oriented flowsheeting: the whole plant solved as one Newton system.

These tests pin the three properties that make the EO engine trustworthy:

1. **It agrees with the sequential-modular engine.** Every unit block is written
   as residual equations; solved simultaneously they must reproduce, to solver
   tolerance, what the sequential-modular units in `fugacio.sim.units` produce
   one at a time. This is the "EO == SM" oracle.
2. **It closes recycles without tearing** and stays differentiable end to end
   (the converged plant's gradient matches a finite difference).
3. **Design specs and optimization** ride on the same differentiable solve, with
   the nested and full-space (simultaneous) formulations agreeing.

The first solve of each distinct flowsheet pays a JIT compilation; the persistent
cache configured in ``conftest.py`` makes every later run cheap.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import (
    Stream,
    component_separator,
    compressor,
    flash_drum,
    heater,
    mix,
    pump,
    splitter,
    tear_solve,
    valve,
)
from fugacio.sim.eo import (
    ComponentSeparator,
    Compressor,
    EOFlowsheet,
    Flash,
    Heater,
    Mixer,
    Pump,
    Splitter,
    Valve,
    optimize_flowsheet_eo,
)
from fugacio.thermo.eos import PR

C = ("methane", "propane", "n-pentane")


def _gas() -> Stream:
    """A light, vapour-rich feed."""
    return Stream.from_fractions(C, jnp.array([0.80, 0.15, 0.05]), 100.0, 330.0, 40e5)


def _liquid() -> Stream:
    """A heavy, liquid feed."""
    return Stream.from_fractions(C, jnp.array([0.02, 0.20, 0.78]), 80.0, 300.0, 6e5)


def _same_stream(a: Stream, b: Stream, *, n_atol: float = 1e-4, t_atol: float = 1e-2) -> None:
    assert jnp.allclose(a.n, b.n, atol=n_atol), f"flows differ: {a.n} vs {b.n}"
    assert float(abs(a.t - b.t)) < t_atol, f"T differs: {a.t} vs {b.t}"
    assert float(abs(a.p - b.p)) < 1.0, f"P differs: {a.p} vs {b.p}"


# --------------------------------------------------------------------------- #
# 1. EO == sequential-modular oracle
# --------------------------------------------------------------------------- #
def test_eo_flash_matches_sequential_modular() -> None:
    # The feed is genuinely two-phase at the drum conditions (both products
    # present), the regime where the equifugacity residuals are well posed.
    fs = EOFlowsheet(eos=PR)
    fs.feed("feed", _fresh())
    fs.add(Flash(inlets=("feed",), outlets=("vap", "liq"), t="T", p="P"))
    sol = fs.solve({"T": 315.0, "P": 18e5})

    assert float(sol.residual_norm) < 1e-8
    vap_sm, liq_sm = flash_drum(_fresh(), 315.0, 18e5, eos=PR)
    assert float(jnp.sum(vap_sm.n)) > 1.0 and float(jnp.sum(liq_sm.n)) > 1.0  # two-phase
    _same_stream(sol["vap"], vap_sm)
    _same_stream(sol["liq"], liq_sm)
    # Material closes: vapour + liquid == feed.
    assert jnp.allclose(sol["vap"].n + sol["liq"].n, _fresh().n, atol=1e-6)


def test_eo_energy_units_match_sequential_modular() -> None:
    # A compressor -> heater -> valve chain solved simultaneously must match the
    # same chain evaluated unit by unit. Exercises the auxiliary (isentropic
    # temperature) unknown, the fixed-outlet-temperature heater, and the
    # isenthalpic valve together.
    fs = EOFlowsheet(eos=PR)
    fs.feed("feed", _gas())
    fs.add(Compressor(inlets=("feed",), outlets=("c",), p_out=80e5, efficiency=0.8))
    fs.add(Heater(inlets=("c",), outlets=("h",), t_out=360.0, dp=0.0))
    fs.add(Valve(inlets=("h",), outlets=("v",), p_out=15e5))
    sol = fs.solve()

    assert float(sol.residual_norm) < 1e-8
    c_sm = compressor(_gas(), 80e5, efficiency=0.8, eos=PR).outlet
    h_sm = heater(c_sm, t_out=360.0, dp=0.0, eos=PR).outlet
    v_sm = valve(h_sm, 15e5, eos=PR)
    _same_stream(sol["c"], c_sm)
    _same_stream(sol["h"], h_sm)
    _same_stream(sol["v"], v_sm)


def test_eo_flow_units_match_sequential_modular() -> None:
    # mix -> pump -> split -> component-separator, simultaneously vs unit by unit.
    fs = EOFlowsheet(eos=PR)
    fs.feed("a", _liquid())
    fs.feed("b", _liquid())
    fs.add(Mixer(inlets=("a", "b"), outlets=("m",)))
    fs.add(Pump(inlets=("m",), outlets=("p",), p_out=25e5, efficiency=0.75))
    fs.add(Splitter(inlets=("p",), outlets=("s1", "s2"), fractions=(0.6, 0.4)))
    fs.add(
        ComponentSeparator(inlets=("s1",), outlets=("top", "bot"), split_to_top=(0.95, 0.5, 0.05))
    )
    sol = fs.solve()

    assert float(sol.residual_norm) < 1e-8
    m_sm = mix([_liquid(), _liquid()], eos=PR)
    p_sm = pump(m_sm, 25e5, efficiency=0.75, eos=PR).outlet
    s1_sm, s2_sm = splitter(p_sm, jnp.array([0.6, 0.4]))
    top_sm, bot_sm = component_separator(s1_sm, jnp.array([0.95, 0.5, 0.05]))
    _same_stream(sol["m"], m_sm)
    _same_stream(sol["p"], p_sm)
    _same_stream(sol["s1"], s1_sm)
    _same_stream(sol["s2"], s2_sm)
    _same_stream(sol["top"], top_sm)
    _same_stream(sol["bot"], bot_sm)


# --------------------------------------------------------------------------- #
# 2. Tear-free recycle + differentiability
# --------------------------------------------------------------------------- #
def _fresh() -> Stream:
    return Stream.from_fractions(C, jnp.array([0.5, 0.3, 0.2]), 100.0, 320.0, 20e5)


def test_eo_recycle_closes_without_tearing() -> None:
    fs = EOFlowsheet(eos=PR)
    fs.feed("fresh", _fresh())
    fs.add(Mixer(inlets=("fresh", "recycle"), outlets=("mixed",), t=320.0))
    fs.add(Flash(inlets=("mixed",), outlets=("vapor", "liquid"), t="T", p="P"))
    fs.add(Splitter(inlets=("liquid",), outlets=("recycle", "purge"), fractions="r"))
    theta = {"T": 320.0, "P": 20e5, "r": jnp.array([0.5, 0.5])}
    sol = fs.solve(theta)

    assert float(sol.residual_norm) < 1e-8
    # Overall balance: fresh feed leaves as vapour product + purge (recycle cancels).
    closure = _fresh().n - (sol["vapor"].n + sol["purge"].n)
    assert float(jnp.max(jnp.abs(closure))) < 1e-6

    # Same answer as the sequential-modular tear solver.
    def _pass(recycle: Stream, th) -> Stream:
        mixed = mix([_fresh(), recycle], t=320.0)
        _v, liq = flash_drum(mixed, th["T"], th["P"], eos=PR)
        recycled, _purge = splitter(liq, jnp.array([th["r"], 1.0 - th["r"]]))
        return recycled

    guess = Stream.from_fractions(C, jnp.array([0.1, 0.3, 0.6]), 30.0, 320.0, 20e5)
    recycle_sm = tear_solve(_pass, guess, {"T": 320.0, "P": 20e5, "r": 0.5})
    assert jnp.allclose(sol["recycle"].n, recycle_sm.n, atol=1e-4)


def test_eo_recycle_gradient_matches_finite_difference() -> None:
    fs = EOFlowsheet(eos=PR)
    fs.feed("fresh", _fresh())
    fs.add(Mixer(inlets=("fresh", "recycle"), outlets=("mixed",), t=320.0))
    fs.add(Flash(inlets=("mixed",), outlets=("vapor", "liquid"), t="T", p=20e5))
    fs.add(Splitter(inlets=("liquid",), outlets=("recycle", "purge"), fractions="r"))

    def methane_recovered(temp: jax.Array) -> jax.Array:
        sol = fs.solve({"T": temp, "r": jnp.array([0.5, 0.5])})
        return sol["vapor"].n[0]

    t0 = jnp.asarray(322.0)
    ad = float(jax.grad(methane_recovered)(t0))
    h = 5e-2
    fd = float((methane_recovered(t0 + h) - methane_recovered(t0 - h)) / (2 * h))
    assert ad == pytest.approx(fd, rel=1e-3)


# --------------------------------------------------------------------------- #
# 3. Degrees of freedom
# --------------------------------------------------------------------------- #
def test_eo_degrees_of_freedom_report() -> None:
    fs = EOFlowsheet(eos=PR)
    fs.feed("feed", _gas())
    fs.add(Flash(inlets=("feed",), outlets=("vap", "liq"), t="T", p="P"))
    report = fs.degrees_of_freedom()
    # Two product streams, each n_components + 2 unknowns; the flash supplies
    # exactly that many equations, so the system is square.
    assert report.n_unknowns == 2 * (len(C) + 2)
    assert report.n_equations == report.n_unknowns
    assert report.degrees_of_freedom == 0
    assert report.per_block == {"vap": 2 * (len(C) + 2)}


def test_eo_unknown_stream_is_rejected() -> None:
    fs = EOFlowsheet(eos=PR)
    fs.feed("feed", _gas())
    fs.add(Valve(inlets=("missing",), outlets=("out",), p_out=10e5))
    with pytest.raises(ValueError, match="undefined stream"):
        fs.solve({})


def test_eo_duplicate_source_is_rejected() -> None:
    fs = EOFlowsheet(eos=PR)
    fs.feed("feed", _gas())
    fs.add(Valve(inlets=("feed",), outlets=("out",), p_out=10e5))
    fs.add(Heater(inlets=("feed",), outlets=("out",), t_out=320.0))
    with pytest.raises(ValueError, match="produced by two blocks"):
        fs.solve({})


# --------------------------------------------------------------------------- #
# 4. Design specs (an extra equation + freed variable, solved simultaneously)
# --------------------------------------------------------------------------- #
def test_eo_design_spec_drives_measurement_to_target() -> None:
    fs = EOFlowsheet(eos=PR)
    fs.feed("feed", _gas())
    fs.add(Heater(inlets=("feed",), outlets=("out",), duty="Q", dp=0.0))
    # Free the duty Q so the outlet temperature hits 360 K.
    fs.spec("Q", lambda s: s["out"].t, target=360.0, init=1.0e5)
    sol = fs.solve({})

    assert float(sol.residual_norm) < 1e-8
    assert float(sol["out"].t) == pytest.approx(360.0, abs=1e-3)
    # The freed duty matches the sequential-modular heater that achieves the same T.
    duty_sm = float(heater(_gas(), t_out=360.0, dp=0.0, eos=PR).duty)
    assert float(sol.specs["Q"]) == pytest.approx(duty_sm, rel=1e-4)


# --------------------------------------------------------------------------- #
# 5. Optimization (nested and simultaneous agree)
# --------------------------------------------------------------------------- #
def _flash_opt_flowsheet() -> EOFlowsheet:
    fs = EOFlowsheet(eos=PR)
    fs.feed("feed", _fresh())
    fs.add(Flash(inlets=("feed",), outlets=("vap", "liq"), t="T", p="P"))
    return fs


def test_optimize_flowsheet_eo_nested_hits_interior_target() -> None:
    # Choose the flash temperature so the vapour molar flow matches a target that
    # is reachable in the interior of the bracket: a smooth, interior optimum.
    vap_target = float(jnp.sum(flash_drum(_fresh(), 320.0, 20e5, eos=PR)[0].n))
    fs = _flash_opt_flowsheet()
    res = optimize_flowsheet_eo(
        fs,
        lambda s: (jnp.sum(s["vap"].n) - vap_target) ** 2,
        {"T": (312.0, 305.0, 335.0)},
        params={"P": 20e5},
    )
    assert bool(res.converged)
    assert float(res.objective) < 1e-3
    assert 305.0 < float(res.decision["T"]) < 335.0
    assert float(res.decision["T"]) == pytest.approx(320.0, abs=0.5)


def _heater_duty_flowsheet() -> EOFlowsheet:
    fs = EOFlowsheet(eos=PR)
    fs.feed("feed", _gas())
    fs.add(Heater(inlets=("feed",), outlets=("out",), duty="Q", dp=0.0))
    return fs


def test_optimize_flowsheet_eo_simultaneous_meets_target() -> None:
    # Full-space (equation-oriented) optimization: vary the heater duty *and* the
    # outlet state together, with the heater equations as equality constraints, to
    # reach a target outlet temperature. A smooth, log-free problem so the
    # augmented-Lagrangian solver can roam without hitting a phase-boundary kink.
    target = 360.0
    res = optimize_flowsheet_eo(
        _heater_duty_flowsheet(),
        lambda s: (s["out"].t - target) ** 2,
        {"Q": (5.0e4, -5.0e6, 5.0e6)},
        simultaneous=True,
    )
    assert float(res.constraint_violation) < 1e-6
    assert float(res.solution["out"].t) == pytest.approx(target, abs=0.5)


def test_optimize_flowsheet_eo_nested_and_simultaneous_agree() -> None:
    # The two formulations should converge to the same duty.
    target = 360.0
    objective = lambda s: (s["out"].t - target) ** 2  # noqa: E731
    decisions = {"Q": (5.0e4, -5.0e6, 5.0e6)}
    nested = optimize_flowsheet_eo(_heater_duty_flowsheet(), objective, decisions)
    simult = optimize_flowsheet_eo(
        _heater_duty_flowsheet(), objective, decisions, simultaneous=True
    )
    assert bool(nested.converged)
    assert float(nested.decision["Q"]) == pytest.approx(float(simult.decision["Q"]), rel=2e-2)
