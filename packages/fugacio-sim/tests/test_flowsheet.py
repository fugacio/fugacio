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


# --------------------------------------------------------------------------- #
# Automatic partitioning and tear selection
# --------------------------------------------------------------------------- #
def _passthrough(*streams, **_):
    return streams[0]


def _graph_flowsheet() -> Flowsheet:
    """Two recycle loops around an acyclic middle, registered out of order."""
    feed = _fresh()
    fs = Flowsheet()
    fs.feed("fresh", feed)
    fs.unit("split2", lambda s, th: (s, s), inputs=("d",), outputs=("rec2", "prod"))
    fs.unit("mix2", lambda a, b, th: a, inputs=("c", "rec2"), outputs=("d",))
    fs.unit("split", lambda s, th: (s, s), inputs=("b",), outputs=("rec", "out1"))
    fs.unit("mixer", lambda a, b, th: a, inputs=("fresh", "rec"), outputs=("a",))
    fs.unit("react", _passthrough, inputs=("a",), outputs=("b",))
    fs.unit("cooler", _passthrough, inputs=("out1",), outputs=("c",))
    return fs


def test_partition_orders_blocks_and_tears_each_loop_once() -> None:
    parts = _graph_flowsheet().partition()
    assert [p.cyclic for p in parts] == [True, False, True]
    first, middle, last = parts
    assert set(first.units) == {"mixer", "react", "split"} and len(first.tears) == 1
    assert middle.units == ("cooler",)
    assert set(last.units) == {"mix2", "split2"} and len(last.tears) == 1
    # Within a torn loop the order is a valid sequential order.
    torn = first.tears[0]
    order = list(first.units)
    if torn == "rec":
        assert order == ["mixer", "react", "split"]
    elif torn == "a":
        assert order == ["react", "split", "mixer"]
    else:
        assert torn == "b" and order == ["split", "mixer", "react"]


def test_partition_honours_hand_designated_tears() -> None:
    fs = _graph_flowsheet()
    fs.tear("rec", _fresh())
    fs.tear("rec2", _fresh())
    parts = fs.partition()
    assert parts[0].tears == ("rec",) and parts[0].units == ("mixer", "react", "split")
    assert parts[2].tears == ("rec2",) and parts[2].units == ("mix2", "split2")


def test_partition_rejects_dangling_and_duplicate_streams() -> None:
    fs = Flowsheet()
    fs.feed("f", _fresh())
    fs.unit("u", _passthrough, inputs=("ghost",), outputs=("x",))
    with pytest.raises(ValueError, match="neither a feed"):
        fs.partition()
    fs2 = Flowsheet()
    fs2.feed("f", _fresh())
    fs2.unit("u1", _passthrough, inputs=("f",), outputs=("x",))
    fs2.unit("u2", _passthrough, inputs=("f",), outputs=("x",))
    with pytest.raises(ValueError, match="produced by both"):
        fs2.partition()


def test_tear_methods_agree_on_a_process_recycle() -> None:
    theta = {"T": jnp.asarray(320.0), "P": jnp.asarray(20e5), "r": jnp.asarray(0.5)}
    guess = Stream.from_fractions(COMPONENTS, jnp.array([0.1, 0.3, 0.6]), 30.0, 320.0, 20e5)
    ref = tear_solve(_recycle_pass, guess, theta, method="wegstein")
    for method in ("broyden", "newton"):
        alt = tear_solve(_recycle_pass, guess, theta, method=method)
        assert jnp.allclose(alt.n, ref.n, rtol=1e-7, atol=1e-8), method
        assert float(alt.t) == pytest.approx(float(ref.t), abs=1e-6)
    with pytest.raises(ValueError, match="unknown tear method"):
        tear_solve(_recycle_pass, guess, theta, method="magic")


def test_newton_tear_gradient_matches_wegstein() -> None:
    guess = Stream.from_fractions(COMPONENTS, jnp.array([0.1, 0.3, 0.6]), 30.0, 320.0, 20e5)

    def recycle_flow(r: float, method: str) -> jax.Array:
        theta = {"T": jnp.asarray(320.0), "P": jnp.asarray(20e5), "r": r}
        return tear_solve(_recycle_pass, guess, theta, method=method).total

    g_w = float(jax.grad(recycle_flow)(0.5, "wegstein"))
    g_n = float(jax.grad(recycle_flow)(0.5, "newton"))
    g_b = float(jax.grad(recycle_flow)(0.5, "broyden"))
    assert g_n == pytest.approx(g_w, rel=1e-6)
    assert g_b == pytest.approx(g_w, rel=1e-6)


def test_auto_partitioned_flowsheet_matches_manual_tear() -> None:
    theta = {"T": jnp.asarray(320.0), "P": jnp.asarray(20e5), "r": jnp.asarray(0.5)}
    guess = Stream.from_fractions(COMPONENTS, jnp.array([0.1, 0.3, 0.6]), 30.0, 320.0, 20e5)

    fs = Flowsheet()
    fs.feed("fresh", _fresh())
    # Registered out of order and without a tear: the flowsheet must sort it out.
    fs.unit(
        "split",
        lambda liquid, th: splitter(liquid, jnp.array([th["r"], 1.0 - th["r"]])),
        inputs=("liquid",),
        outputs=("recycle", "purge"),
    )
    fs.unit(
        "drum",
        lambda mixed, th: flash_drum(mixed, th["T"], th["P"]),
        inputs=("mixed",),
        outputs=("vapor", "liquid"),
    )
    fs.unit(
        "mixer",
        lambda fresh, rec, th: mix([fresh, rec], t=320.0),
        inputs=("fresh", "recycle"),
        outputs=("mixed",),
    )
    (block,) = fs.partition()
    assert block.cyclic and len(block.tears) == 1
    streams = fs.solve(theta, method="broyden")

    recycle = tear_solve(_recycle_pass, guess, theta)
    assert jnp.allclose(streams["recycle"].n, recycle.n, atol=1e-6)
    closure = _fresh().n - (streams["vapor"].n + streams["purge"].n)
    assert float(jnp.max(jnp.abs(closure))) < 1e-5


def test_downstream_loop_gradient_flows_through_upstream_blocks() -> None:
    # The recycle loop depends on a stream computed by an earlier block; the
    # implicit adjoint must carry the gradient through that upstream dependency.
    def purge_pentane(t_pre: float) -> jax.Array:
        fs = Flowsheet()
        fs.feed("fresh", _fresh())
        fs.unit(
            "preflash",
            lambda f, th: flash_drum(f, th["T_pre"], 20e5)[1],
            inputs=("fresh",),
            outputs=("liq0",),
        )
        fs.unit(
            "mixer",
            lambda a, rec, th: mix([a, rec], t=320.0),
            inputs=("liq0", "recycle"),
            outputs=("mixed",),
        )
        fs.unit(
            "drum",
            lambda m, th: flash_drum(m, 320.0, 20e5),
            inputs=("mixed",),
            outputs=("vapor", "liquid"),
        )
        fs.unit(
            "split",
            lambda liq, th: splitter(liq, jnp.array([0.5, 0.5])),
            inputs=("liquid",),
            outputs=("recycle", "purge"),
        )
        return fs.solve({"T_pre": t_pre})["purge"].n[2]

    g = float(jax.grad(purge_pentane)(318.0))
    fd = float((purge_pentane(318.25) - purge_pentane(317.75)) / 0.5)
    assert g == pytest.approx(fd, rel=2e-3)
