"""Differentiable recycle/tear solver and the declarative Flowsheet wrapper.

Covers (1) an analytic linear recycle where the fixed point and its parameter
gradient are known in closed form, (2) a real EOS process recycle where the
overall material balance must close and the gradient through the recycle is
checked against a finite difference, and (3) the Flowsheet builder reproducing
the same result.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import Flowsheet, Stream, flash_drum, mix, splitter, tear_solve

COMPONENTS = ("methane", "propane", "n-pentane")


def test_linear_recycle_matches_closed_form() -> None:
    # g(x) = 0.5 x + theta  =>  fixed point x* = 2 theta, dx*/dtheta = 2.
    def g(x, theta):
        return 0.5 * x + theta

    theta = jnp.array([1.0, 2.0])
    x_star = tear_solve(g, jnp.zeros(2), theta)
    assert jnp.allclose(x_star, 2.0 * theta, atol=1e-6)

    grad = jax.grad(lambda th: jnp.sum(tear_solve(g, jnp.zeros(2), th)))(theta)
    assert jnp.allclose(grad, jnp.array([2.0, 2.0]), atol=1e-5)


def test_recycle_accepts_dict_pytree_state() -> None:
    # Independent scalar recycles carried in a dict pytree.
    def g(state, theta):
        return {"a": 0.25 * state["a"] + theta, "b": 0.5 * state["b"] + 2.0 * theta}

    sol = tear_solve(g, {"a": jnp.array(0.0), "b": jnp.array(0.0)}, jnp.array(3.0))
    assert float(sol["a"]) == pytest.approx(3.0 / 0.75, rel=1e-6)  # a* = theta/(1-0.25)
    assert float(sol["b"]) == pytest.approx(2.0 * 3.0 / 0.5, rel=1e-6)  # b* = 2 theta/(1-0.5)


def _fresh() -> Stream:
    return Stream.from_fractions(COMPONENTS, jnp.array([0.5, 0.3, 0.2]), 100.0, 320.0, 20e5)


def _recycle_pass(recycle: Stream, theta) -> Stream:
    """One sequential pass: mix fresh + recycle, flash, recycle part of the liquid."""
    mixed = mix([_fresh(), recycle], t=320.0)
    _vapor, liquid = flash_drum(mixed, theta["T"], theta["P"])
    recycled, _purge = splitter(liquid, jnp.array([theta["r"], 1.0 - theta["r"]]))
    return recycled


def test_process_recycle_closes_material_balance() -> None:
    theta = {"T": jnp.asarray(320.0), "P": jnp.asarray(20e5), "r": jnp.asarray(0.5)}
    guess = Stream.from_fractions(COMPONENTS, jnp.array([0.1, 0.3, 0.6]), 30.0, 320.0, 20e5)
    recycle = tear_solve(_recycle_pass, guess, theta)

    # Self-consistency: feeding the converged recycle back reproduces it.
    assert jnp.allclose(_recycle_pass(recycle, theta).n, recycle.n, atol=1e-6)

    # Overall balance: fresh feed == vapour product + purge (recycle cancels).
    mixed = mix([_fresh(), recycle], t=320.0)
    vapor, liquid = flash_drum(mixed, theta["T"], theta["P"])
    _recycled, purge = splitter(liquid, jnp.array([theta["r"], 1.0 - theta["r"]]))
    closure = _fresh().n - (vapor.n + purge.n)
    assert float(jnp.max(jnp.abs(closure))) < 1e-5


def test_process_recycle_gradient_matches_finite_difference() -> None:
    guess = Stream.from_fractions(COMPONENTS, jnp.array([0.1, 0.3, 0.6]), 30.0, 320.0, 20e5)

    def product_flow(t_drum: float) -> jax.Array:
        theta = {"T": jnp.asarray(t_drum), "P": jnp.asarray(20e5), "r": jnp.asarray(0.5)}
        recycle = tear_solve(_recycle_pass, guess, theta)
        mixed = mix([_fresh(), recycle], t=320.0)
        vapor, _liquid = flash_drum(mixed, theta["T"], theta["P"])
        return vapor.total

    g = float(jax.grad(product_flow)(320.0))
    fd = float((product_flow(320.5) - product_flow(319.5)) / 1.0)
    assert g == pytest.approx(fd, rel=2e-3)
    assert g > 0.0  # a hotter drum makes more vapour product


def test_flowsheet_builder_reproduces_functional_recycle() -> None:
    theta = {"T": jnp.asarray(320.0), "P": jnp.asarray(20e5), "r": jnp.asarray(0.5)}
    guess = Stream.from_fractions(COMPONENTS, jnp.array([0.1, 0.3, 0.6]), 30.0, 320.0, 20e5)

    fs = Flowsheet()
    fs.feed("fresh", _fresh())
    fs.unit(
        "mixer",
        lambda fresh, rec, th: mix([fresh, rec], t=320.0),
        inputs=("fresh", "recycle"),
        outputs=("mixed",),
    )
    fs.unit(
        "drum",
        lambda mixed, th: flash_drum(mixed, th["T"], th["P"]),
        inputs=("mixed",),
        outputs=("vapor", "liquid"),
    )
    fs.unit(
        "split",
        lambda liquid, th: splitter(liquid, jnp.array([th["r"], 1.0 - th["r"]])),
        inputs=("liquid",),
        outputs=("recycle", "purge"),
    )
    fs.tear("recycle", guess)
    streams = fs.solve(theta)

    recycle = tear_solve(_recycle_pass, guess, theta)
    mixed = mix([_fresh(), recycle], t=320.0)
    vapor, _liquid = flash_drum(mixed, theta["T"], theta["P"])
    assert jnp.allclose(streams["vapor"].n, vapor.n, atol=1e-6)
    assert jnp.allclose(streams["recycle"].n, recycle.n, atol=1e-6)
