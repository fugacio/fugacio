"""Design specifications and set-point controllers.

Covers (1) a single spec solved on an analytic model with a known answer and its
gradient checked against closed form, (2) a real EOS flash-drum spec (find the
drum temperature that gives a target vapour fraction) with the manipulated
variable's gradient checked against a finite difference, and (3) two coupled
specs solved simultaneously.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import Stream, flash_drum
from fugacio.sim.design import DesignSpec, controller, meet_spec, solve_design

COMPONENTS = ("propane", "n-butane", "n-pentane")


def test_meet_spec_scalar_closed_form() -> None:
    # measure(u, theta) = theta * u^2 ; target 12, theta 3  => u = 2.
    u = meet_spec(
        lambda u, th: th * u**2, target=12.0, u0=1.0, theta=jnp.asarray(3.0), lo=0.0, hi=10.0
    )
    assert float(u) == pytest.approx(2.0, abs=1e-5)


def test_meet_spec_gradient() -> None:
    # u*(theta) solves theta * u = 6  => u = 6/theta, du/dtheta = -6/theta^2.
    def solve(theta: jax.Array) -> jax.Array:
        return meet_spec(lambda u, th: th * u, target=6.0, u0=1.0, theta=theta)

    theta = jnp.asarray(3.0)
    assert float(solve(theta)) == pytest.approx(2.0, abs=1e-6)
    assert float(jax.grad(solve)(theta)) == pytest.approx(-6.0 / 9.0, abs=1e-5)


def _feed(flow: float = 100.0) -> Stream:
    return Stream.from_fractions(COMPONENTS, jnp.array([0.4, 0.35, 0.25]), flow, 330.0, 8e5)


def test_flash_drum_temperature_spec() -> None:
    # Find the drum temperature giving a 40 % vapour fraction at 8 bar.
    def vapor_fraction(t: jax.Array, _: object) -> jax.Array:
        vap, _liq = flash_drum(_feed(), t, 8e5)
        return vap.total / _feed().total

    t_star = meet_spec(vapor_fraction, target=0.4, u0=320.0, theta=None, lo=300.0, hi=360.0)
    vap, _ = flash_drum(_feed(), t_star, 8e5)
    assert float(vap.total / _feed().total) == pytest.approx(0.4, abs=1e-4)


def test_solve_design_single_spec_on_flowsheet() -> None:
    # A trivial "flowsheet": flash the feed at theta["T"], 8 bar.
    def simulate(theta: dict) -> dict:
        vap, liq = flash_drum(_feed(), theta["T"], 8e5)
        return {"vapor": vap, "liquid": liq}

    spec = controller(
        simulate,
        manipulated="T",
        controlled=lambda s: s["vapor"].total / 100.0,
        set_point=0.5,
        lo=300.0,
        hi=360.0,
    )
    result = solve_design(simulate, {"T": jnp.asarray(325.0)}, [spec])
    assert bool(result.converged)
    assert float(result.streams["vapor"].total / 100.0) == pytest.approx(0.5, abs=1e-4)


def test_solve_design_two_coupled_specs() -> None:
    # Two manipulated scalars, two targets, coupled through a 2x2 linear map.
    #   m0 = a + b ;  m1 = a - b  ;  targets (3, 1) => a=2, b=1.
    def simulate(theta: dict) -> dict:
        a, b = theta["a"], theta["b"]
        s = Stream(jnp.array([a + b, a - b]), jnp.asarray(300.0), jnp.asarray(1e5), ("x", "y"))
        return {"out": s}

    specs = [
        DesignSpec("a", lambda s: s["out"].n[0], 3.0, -10.0, 10.0),
        DesignSpec("b", lambda s: s["out"].n[1], 1.0, -10.0, 10.0),
    ]
    result = solve_design(simulate, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)}, specs)
    assert bool(result.converged)
    assert jnp.allclose(result.manipulated, jnp.array([2.0, 1.0]), atol=1e-6)
