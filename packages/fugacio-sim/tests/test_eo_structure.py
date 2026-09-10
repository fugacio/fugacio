from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from fugacio.sim import Stream
from fugacio.sim.eo import EOFlowsheet, Heater


def train(count=6, mode="colored"):
    feed = Stream.from_fractions(("methane", "ethane"), jnp.array([0.8, 0.2]), 1.0, 300.0, 1e5)
    fs = EOFlowsheet(jacobian_mode=mode).feed("feed", feed)
    for i in range(count):
        fs.add(Heater(("feed" if i == 0 else f"out{i - 1}",), (f"out{i}",), t_out="t"))
    return fs


def test_structural_incidence_matches_dense_jacobian_and_reduces_directions():
    fs = train()
    report = fs.diagnose_structure()
    assert report["structurally_square_and_matched"]
    assert report["n_unknowns"] == 24
    assert report["colored_directions"] == 8
    assert not fs._plans  # Structure inspection doesn't compile or seed a solve.
    ctx = fs._context()
    names = fs._internal_names()
    tree = fs._initial_unknowns(ctx, names, {}, {"t": 350.0}, fs.feeds, None, 0)
    x, unravel = ravel_pytree(tree)
    residual = fs._residual_fn(ctx, names, {}, unravel)
    theta = {"params": {"t": 350.0}, "pkg": ctx.package, "feeds": fs.feeds}
    pattern, labels = fs._incidence(ctx)
    assert labels[:4] == ("out0:n[methane]", "out0:n[ethane]", "out0:temperature", "out0:pressure")
    np.testing.assert_allclose(
        pattern.jacobian(residual)(x, theta).matrix, jax.jacfwd(residual)(x, theta), atol=1e-12
    )


def test_eo_coloring_dense_reference_and_operating_point_cache():
    colored, dense = train(3), train(3, "dense")
    for temperature in (340.0, 360.0):
        actual = colored.solve({"t": temperature})
        expected = dense.solve({"t": temperature})
        assert actual.report.converged and expected.report.converged
        assert actual["out2"].t == pytest.approx(temperature, abs=1e-9)
        np.testing.assert_allclose(actual["out2"].n, expected["out2"].n, atol=1e-10)
    assert len(colored._plans) == 1
    derivative = jax.jit(jax.grad(lambda t: colored.solve({"t": t})["out2"].t))(345.0)
    assert derivative == pytest.approx(1.0, abs=1e-10)
    # Changing assembly strategy invalidates the compiled plan.
    colored.jacobian_mode = "dense"
    colored.solve({"t": 350.0})
    assert len(colored._plans) == 2


def test_custom_subclass_defaults_to_conservative_dependencies():
    @dataclass(frozen=True)
    class CustomHeater(Heater):
        pass

    fs = train()
    fs.blocks[0] = CustomHeater(("feed",), ("out0",), t_out="t")
    report = fs.diagnose_structure()
    assert report["blocks"][0]["dependencies"] == "all_unknowns"
    assert report["colored_directions"] == report["n_unknowns"]


def test_duplicate_auxiliaries_and_freed_variables_are_rejected():
    @dataclass(frozen=True)
    class WithAux(Heater):
        def aux_scales(self, ctx):
            return {"shared": 1.0}

    fs = train(2)
    fs.blocks = [WithAux(b.inlets, b.outlets, t_out="t") for b in fs.blocks]
    with pytest.raises(ValueError, match="multiple owners"):
        fs.diagnose_structure()
    fs = train(1)
    fs.spec("t", lambda s: s["out0"].t, 350.0, init=340.0)
    with pytest.raises(ValueError, match="already freed"):
        fs.spec("t", lambda s: s["out0"].t, 360.0, init=340.0)


def test_empty_outputs_and_invalid_dependency_contract_are_rejected():
    fs = train(1)
    fs.blocks = [Heater(("feed",), (), t_out=350.0)]
    with pytest.raises(ValueError, match="at least one outlet"):
        fs.diagnose_structure()

    @dataclass(frozen=True)
    class BadDependency(Heater):
        def residual_dependencies(self, ctx):
            return ("missing",), ()

    fs.blocks = [BadDependency(("feed",), ("out",), t_out=350.0)]
    with pytest.raises(ValueError, match="dependency declaration"):
        fs.diagnose_structure()
