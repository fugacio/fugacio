"""Two-sided heat exchanger: energy closure, specifications, phase change, gradients.

The exchanger is checked for (1) an exact shared energy balance on both sides,
(2) the equivalence of its specifications (a duty found from ``UA`` reproduces
the ``UA`` a temperature spec required), (3) the minimum-approach spec landing
on the pinch, (4) a condensing steam side showing the saturation plateau in its
temperature curve, (5) the second-law cap on an infeasible request, and (6) the
implicit gradient of the ``UA``-specified duty against a finite difference.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import Stream, heat_exchanger, package_for
from fugacio.sim.properties import enthalpy_flow

C = ("methane", "ethane", "propane")


def _hot() -> Stream:
    return Stream.from_fractions(C, jnp.array([0.2, 0.3, 0.5]), 50.0, 400.0, 30e5)


def _cold() -> Stream:
    return Stream.from_fractions(C, jnp.array([0.7, 0.2, 0.1]), 80.0, 250.0, 25e5)


def test_temperature_spec_closes_both_energy_balances() -> None:
    hot, cold = _hot(), _cold()
    res = heat_exchanger(hot, cold, t_hot_out=330.0)
    assert float(res.hot_out.t) == pytest.approx(330.0, abs=1e-6)
    q_hot = enthalpy_flow(hot) - enthalpy_flow(res.hot_out)
    q_cold = enthalpy_flow(res.cold_out) - enthalpy_flow(cold)
    assert float(q_hot) == pytest.approx(float(res.duty), rel=1e-8)
    assert float(q_cold) == pytest.approx(float(res.duty), rel=1e-8)
    assert jnp.allclose(res.hot_out.n, hot.n) and jnp.allclose(res.cold_out.n, cold.n)
    assert float(res.min_approach) > 0.0
    assert float(res.ua) > 0.0 and float(res.lmtd) == pytest.approx(float(res.duty / res.ua))


def test_ua_spec_reproduces_the_duty_of_a_temperature_spec() -> None:
    hot, cold = _hot(), _cold()
    ref = heat_exchanger(hot, cold, t_hot_out=330.0)
    rated = heat_exchanger(hot, cold, ua=ref.ua)
    assert float(rated.duty) == pytest.approx(float(ref.duty), rel=1e-6)
    assert float(rated.hot_out.t) == pytest.approx(330.0, abs=1e-4)
    # An area with a coefficient is the same specification.
    sized = heat_exchanger(hot, cold, area=float(ref.ua) / 500.0, u=500.0)
    assert float(sized.duty) == pytest.approx(float(ref.duty), rel=1e-6)
    assert float(sized.area) == pytest.approx(float(ref.ua) / 500.0, rel=1e-6)


def test_min_approach_spec_lands_on_the_pinch() -> None:
    res = heat_exchanger(_hot(), _cold(), min_approach=10.0)
    assert float(res.min_approach) == pytest.approx(10.0, abs=1e-6)
    assert min(float(res.approach_hot_end), float(res.approach_cold_end)) == pytest.approx(
        10.0, abs=1e-6
    )


def test_infeasible_temperature_spec_is_capped_by_the_second_law() -> None:
    # The cold stream cannot be heated above the hot inlet; the duty saturates.
    res = heat_exchanger(_hot(), _cold(), t_cold_out=450.0)
    assert float(res.cold_out.t) <= 400.0 + 1e-6
    assert float(res.cold_out.t) > 390.0
    assert float(res.min_approach) >= -1e-6
    assert not jnp.isfinite(res.ua) or float(res.ua) > 0.0


def test_condensing_steam_side_shows_a_saturation_plateau() -> None:
    steam = Stream.from_fractions(("water",), jnp.array([1.0]), 5.0, 450.0, 5e5)
    process = Stream.from_fractions(C, jnp.array([0.2, 0.3, 0.5]), 50.0, 300.0, 30e5)
    res = heat_exchanger(
        steam, process, t_cold_out=312.0, model_hot=package_for(["water"], "iapws"), zones=6
    )
    t_sat = 424.98  # IAPWS saturation temperature at 5 bar
    plateau = jnp.abs(res.hot_curve - t_sat) < 0.05
    assert int(jnp.sum(plateau)) >= 2, res.hot_curve
    assert float(res.hot_out.t) < t_sat  # condensate is subcooled at the outlet
    assert float(res.cold_out.t) == pytest.approx(312.0, abs=1e-6)


def test_parallel_flow_has_a_smaller_lmtd_than_counter_flow() -> None:
    hot, cold = _hot(), _cold()
    counter = heat_exchanger(hot, cold, duty=2.0e5, flow="counter")
    parallel = heat_exchanger(hot, cold, duty=2.0e5, flow="parallel")
    assert float(parallel.ua) > float(counter.ua)


def test_duty_is_differentiable_in_ua() -> None:
    hot, cold = _hot(), _cold()
    ua0 = 4000.0
    grad = jax.grad(lambda ua: heat_exchanger(hot, cold, ua=ua).duty)(ua0)
    fd = (
        heat_exchanger(hot, cold, ua=ua0 * 1.001).duty
        - heat_exchanger(hot, cold, ua=ua0 * 0.999).duty
    ) / (0.002 * ua0)
    assert float(grad) == pytest.approx(float(fd), rel=1e-4)


def test_specification_validation() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        heat_exchanger(_hot(), _cold())
    with pytest.raises(ValueError, match="exactly one"):
        heat_exchanger(_hot(), _cold(), duty=1.0, ua=1.0)
    with pytest.raises(ValueError, match="coefficient u"):
        heat_exchanger(_hot(), _cold(), area=10.0)
    with pytest.raises(ValueError, match="flow"):
        heat_exchanger(_hot(), _cold(), duty=1.0, flow="cross")
