"""End-to-end flowsheet optimization against a money objective.

Ties the optimizer, the economics correlations, and the flowsheet evaluation
together: a U-shaped total-annual-cost trade-off with a known interior optimum,
a bound-constrained variant, and an inequality-constrained design.
"""

import jax.numpy as jnp
import pytest

from fugacio.sim import Stream, bare_module_cost, total_annual_cost, utility_cost
from fugacio.sim.design import optimize_flowsheet


def _simulate(theta: dict) -> dict:
    # A stand-in flowsheet: the design variable x flows straight to a stream.
    x = jnp.asarray(theta["x"])
    return {"out": Stream(jnp.array([x]), jnp.asarray(300.0), jnp.asarray(1e5), ("a",))}


def _tac(streams: dict, theta: dict) -> jnp.ndarray:
    # Classic trade-off: a bigger vessel (x = volume) costs more capital but the
    # paired cooling duty falls, so total annual cost is U-shaped in x.
    x = jnp.asarray(theta["x"])
    capex = bare_module_cost("vessel", x).bare_module
    opex = utility_cost(2.0e7 / x, "cooling_water")
    return total_annual_cost(capex, opex)


def test_optimize_flowsheet_finds_interior_minimum() -> None:
    res = optimize_flowsheet(
        _simulate,
        _tac,
        {"x": jnp.asarray(5.0)},
        ["x"],
        bounds={"x": (1.0, 100.0)},
    )
    assert bool(res.converged)
    x_star = float(res.theta["x"])
    assert 1.0 < x_star < 100.0  # interior optimum
    # The optimum beats both ends of the bracket.
    f_star = float(res.objective)
    f_lo = float(_tac(_simulate({"x": 1.0}), {"x": 1.0}))
    f_hi = float(_tac(_simulate({"x": 100.0}), {"x": 100.0}))
    assert f_star < f_lo and f_star < f_hi


def test_optimize_flowsheet_improves_over_initial_guess() -> None:
    x0 = 5.0
    f0 = float(_tac(_simulate({"x": x0}), {"x": x0}))
    res = optimize_flowsheet(
        _simulate, _tac, {"x": jnp.asarray(x0)}, ["x"], bounds={"x": (1.0, 100.0)}
    )
    assert float(res.objective) < f0


def test_optimize_flowsheet_inequality_constraint() -> None:
    # Minimize cost = x^2 subject to x >= 2 (i.e. 2 - x <= 0)  =>  x* = 2.
    def simulate(theta: dict) -> dict:
        x = jnp.asarray(theta["x"])
        return {"out": Stream(jnp.array([x]), jnp.asarray(300.0), jnp.asarray(1e5), ("a",))}

    def objective(streams: dict, theta: dict) -> jnp.ndarray:
        return jnp.asarray(theta["x"]) ** 2

    res = optimize_flowsheet(
        simulate,
        objective,
        {"x": jnp.asarray(5.0)},
        ["x"],
        ineq_constraints=lambda th: jnp.array([2.0 - jnp.asarray(th["x"])]),
    )
    assert float(res.theta["x"]) == pytest.approx(2.0, abs=1e-2)
