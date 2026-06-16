"""Heat integration & pinch analysis: targets, curves, area/cost, and networks.

The problem table is checked against the textbook four-stream problem (whose
minimum utilities and pinch are widely published), the composite curves against
the pinch geometry (their closest approach is exactly ``dt_min``), the grand
composite against the cascade it is built from, and the area target against the
analytic ``Q/(U*LMTD)`` of a single counter-current match. The network
synthesiser is required to hit the MER utility targets while keeping ``dt_min``
on every exchanger, and the whole target stack is differentiated -- the point of
a gradient-based heat-integration model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import Stream
from fugacio.sim.integration import (
    area_target,
    capital_cost_target,
    composite_curves,
    grand_composite_curve,
    heat_stream,
    make_stream,
    minimum_utilities,
    optimal_dt_min,
    pinch_analysis,
    supertarget,
    synthesize_network,
    total_annual_cost_target,
    units_target,
)

# The classic four-stream problem (Smith, *Chemical Process Design*; Kemp, *Pinch
# Analysis*). CP in kW/K, temperatures in degrees -- the energy targets are
# scale-free, so the bare numbers stand in for SI throughout.
FOUR_STREAM = [
    make_stream(20.0, 135.0, 2.0, h=1.0, name="C1"),
    make_stream(170.0, 60.0, 3.0, h=1.0, name="H1"),
    make_stream(80.0, 140.0, 4.0, h=1.0, name="C2"),
    make_stream(150.0, 30.0, 1.5, h=1.0, name="H2"),
]


# --------------------------------------------------------------------------- #
# Energy targets (problem table algorithm)
# --------------------------------------------------------------------------- #
def test_four_stream_energy_targets() -> None:
    res = pinch_analysis(FOUR_STREAM, 10.0)
    assert float(res.hot_utility) == pytest.approx(20.0, abs=1e-6)
    assert float(res.cold_utility) == pytest.approx(60.0, abs=1e-6)
    # Pinch at interval (shifted) temperature 85 -> 90 C hot / 80 C cold.
    assert float(res.hot_pinch_temperature) == pytest.approx(90.0, abs=1e-6)
    assert float(res.cold_pinch_temperature) == pytest.approx(80.0, abs=1e-6)
    assert bool(res.has_pinch)


def test_first_law_energy_balance() -> None:
    # Q_hot,min + (heat released by hot streams) = Q_cold,min + (heat absorbed by cold).
    q_h, q_c = minimum_utilities(FOUR_STREAM, 10.0)
    hot_duty = sum(float(s.duty) for s in FOUR_STREAM if float(s.t_supply) > float(s.t_target))
    cold_duty = sum(float(s.duty) for s in FOUR_STREAM if float(s.t_supply) < float(s.t_target))
    assert float(q_h) + hot_duty == pytest.approx(float(q_c) + cold_duty, rel=1e-9)


def test_larger_dt_min_costs_more_utility() -> None:
    q_h_small, q_c_small = minimum_utilities(FOUR_STREAM, 10.0)
    q_h_big, q_c_big = minimum_utilities(FOUR_STREAM, 20.0)
    assert float(q_h_big) > float(q_h_small)
    assert float(q_c_big) > float(q_c_small)


def test_threshold_problem_has_no_pinch() -> None:
    # Large hot stream against a small cold load: only cold utility is needed.
    streams = [
        make_stream(200.0, 50.0, 10.0, name="H"),
        make_stream(40.0, 120.0, 2.0, name="C"),
    ]
    res = pinch_analysis(streams, 10.0)
    assert not bool(res.has_pinch)
    assert min(float(res.hot_utility), float(res.cold_utility)) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Composite and grand composite curves
# --------------------------------------------------------------------------- #
def test_composite_curves_touch_at_dt_min() -> None:
    cc = composite_curves(FOUR_STREAM, 10.0)
    assert float(cc.min_approach) == pytest.approx(10.0, abs=1e-6)
    # Hot composite spans the total hot duty; cold composite is offset by Q_c,min.
    assert float(cc.hot_h[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(cc.cold_h[0]) == pytest.approx(60.0, abs=1e-6)


def test_grand_composite_matches_cascade() -> None:
    gcc = grand_composite_curve(FOUR_STREAM, 10.0)
    q_h, q_c = minimum_utilities(FOUR_STREAM, 10.0)
    # Top of the GCC is the hot utility, bottom the cold utility, and it touches
    # zero at the pinch.
    assert float(gcc.net_heat_flow[0]) == pytest.approx(float(q_h), abs=1e-6)
    assert float(gcc.net_heat_flow[-1]) == pytest.approx(float(q_c), abs=1e-6)
    assert float(jnp.min(gcc.net_heat_flow)) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Area, units, and capital targets
# --------------------------------------------------------------------------- #
def test_area_target_matches_single_match_analytic() -> None:
    # One hot + one cold, equal CP, equal film coefficients: a threshold problem
    # whose area target is exactly Q/(U*LMTD) with 1/U = 1/h_hot + 1/h_cold.
    hot = make_stream(200.0, 100.0, 2.0, h=1000.0, name="H")
    cold = make_stream(80.0, 180.0, 2.0, h=1000.0, name="C")
    area = area_target([hot, cold], 20.0)
    u = 1.0 / (1.0 / 1000.0 + 1.0 / 1000.0)
    expected = 200.0 / (u * 20.0)  # Q = 200, LMTD = 20 (constant 20 K approach)
    assert float(area) == pytest.approx(expected, rel=1e-6)


def test_area_target_decreases_with_dt_min() -> None:
    a_tight = area_target(FOUR_STREAM, 5.0)
    a_loose = area_target(FOUR_STREAM, 20.0)
    assert float(a_tight) > float(a_loose) > 0.0


def test_units_target_four_stream() -> None:
    ut = units_target(FOUR_STREAM, 10.0)
    # 5 streams above the pinch (incl. hot utility), 4 below (incl. cold utility).
    assert int(ut.above_pinch) == 5
    assert int(ut.below_pinch) == 4
    assert int(ut.units) == 7


def test_capital_and_total_annual_cost_breakdown() -> None:
    target = total_annual_cost_target(FOUR_STREAM, 10.0, area_cost=(1.0e4, 800.0, 0.8))
    assert float(target.capital) > 0.0
    assert float(target.utility_cost) > 0.0
    # TAC is annualised capital plus utilities.
    assert float(target.total_annual_cost) == pytest.approx(
        float(target.annualized_capital) + float(target.utility_cost), rel=1e-9
    )
    cap = capital_cost_target(FOUR_STREAM, 10.0, area_cost=(1.0e4, 800.0, 0.8))
    assert float(cap) == pytest.approx(float(target.capital), rel=1e-9)


def test_supertarget_grid_and_optimum() -> None:
    grid = jnp.array([5.0, 10.0, 20.0, 30.0])
    st = supertarget(FOUR_STREAM, grid, area_cost=(1.0e4, 800.0, 0.8))
    assert st.total_annual_cost.shape == (4,)
    # Utilities grow monotonically with dt_min along the grid.
    assert bool(jnp.all(jnp.diff(st.hot_utility) >= -1e-6))

    opt = optimal_dt_min(FOUR_STREAM, area_cost=(1.0e4, 800.0, 0.8), bounds=(1.0, 40.0))
    assert bool(opt.converged)
    # The optimum is no worse than any grid point.
    assert float(opt.total_annual_cost) <= float(jnp.min(st.total_annual_cost)) + 1e-6


# --------------------------------------------------------------------------- #
# Stream extraction from process streams
# --------------------------------------------------------------------------- #
def test_heat_stream_from_process_stream() -> None:
    feed = Stream.from_fractions(("nitrogen",), jnp.array([1.0]), 10.0, 400.0, 1.0e5)
    hs = heat_stream(feed, 300.0, name="N2")
    # Cooling 400 -> 300 K: hot stream, positive CP, duty = CP * 100.
    assert bool(hs.is_hot)
    assert float(hs.cp) > 0.0
    assert float(hs.duty) == pytest.approx(float(hs.cp) * 100.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# Differentiability (the whole point)
# --------------------------------------------------------------------------- #
def test_energy_target_is_differentiable() -> None:
    g = jax.grad(lambda dt: pinch_analysis(FOUR_STREAM, dt).hot_utility)(10.0)
    # More approach temperature -> more hot utility.
    assert jnp.isfinite(g)
    assert float(g) > 0.0


def test_targets_differentiable_in_stream_cp() -> None:
    def hot_utility(cp_h1: jnp.ndarray) -> jnp.ndarray:
        streams = [
            make_stream(20.0, 135.0, 2.0, name="C1"),
            make_stream(170.0, 60.0, cp_h1, name="H1"),
            make_stream(80.0, 140.0, 4.0, name="C2"),
            make_stream(150.0, 30.0, 1.5, name="H2"),
        ]
        return pinch_analysis(streams, 10.0).hot_utility

    g = jax.grad(hot_utility)(jnp.asarray(3.0))
    assert jnp.isfinite(g)


def test_area_and_tac_differentiable() -> None:
    g_area = jax.grad(lambda dt: area_target(FOUR_STREAM, dt))(12.0)
    g_tac = jax.grad(
        lambda dt: total_annual_cost_target(FOUR_STREAM, dt, area_cost=(1e4, 800.0, 0.8)).area
    )(12.0)
    assert jnp.isfinite(g_area) and float(g_area) < 0.0  # area falls as dt_min grows
    assert jnp.isfinite(g_tac)


# --------------------------------------------------------------------------- #
# Network synthesis
# --------------------------------------------------------------------------- #
def test_network_synthesis_four_stream_is_feasible_mer() -> None:
    net = synthesize_network(FOUR_STREAM, 10.0)
    assert net.feasible
    assert net.achieves_mer
    assert float(net.hot_utility) == pytest.approx(20.0, abs=1e-6)
    assert float(net.cold_utility) == pytest.approx(60.0, abs=1e-6)
    assert net.min_approach >= 10.0 - 1e-6
    # Every stream's energy balance closes (checked inside verify_network).
    assert net.total_area > 0.0


def test_network_threshold_problem() -> None:
    streams = [
        make_stream(200.0, 50.0, 10.0, name="H"),
        make_stream(40.0, 120.0, 2.0, name="C"),
    ]
    net = synthesize_network(streams, 10.0)
    assert net.feasible
    assert net.achieves_mer
    # No hot utility for this threshold problem.
    assert float(net.hot_utility) == pytest.approx(0.0, abs=1e-6)


def test_network_area_at_least_target() -> None:
    # A real (non-vertical) network can only need at least the area target.
    net = synthesize_network(FOUR_STREAM, 10.0)
    target = float(area_target(FOUR_STREAM, 10.0))
    assert net.total_area >= target - 1e-6
