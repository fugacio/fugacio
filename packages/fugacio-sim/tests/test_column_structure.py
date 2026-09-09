import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fugacio.sim import (
    ColumnFeed,
    SideDraw,
    StageDuty,
    Stream,
    distillate_rate,
    package_for,
    reflux_ratio,
    rigorous_column,
)
from fugacio.sim import distillation as dist


@pytest.mark.parametrize(
    "condenser,reboiler,subcooled,method",
    [
        ("total", "kettle", False, "pr"),
        ("total", "kettle", True, "pr"),
        ("partial", "kettle", False, "pr"),
        (None, None, False, "pr"),
        ("total", None, False, "nrtl"),
        (None, "kettle", False, "nrtl"),
    ],
)
def test_column_declared_structure_matches_dense_ad(
    monkeypatch, condenser, reboiler, subcooled, method
):
    # Capture the public adapter's physical parameters before solving. Test
    # structure away from a root so accidental zeros cannot hide couplings.
    captured = {}

    class Captured(Exception):
        pass

    def capture(theta, feeds, hints, st, *args):
        captured.update(theta=theta, structure=st)
        raise Captured

    monkeypatch.setattr(dist, "_solve", capture)
    components = ("benzene", "toluene") if method == "pr" else ("ethanol", "water")
    pkg = package_for(components, method)
    feed = Stream.from_fractions(components, jnp.array([0.3, 0.7]), 100.0, 350.0, 1.013e5)
    specs = ([reflux_ratio(2.5)] if condenser else []) + (
        [distillate_rate(40.0)] if reboiler else []
    )
    with pytest.raises(Captured):
        rigorous_column(
            [ColumnFeed(feed, 2), ColumnFeed(feed, 4)],
            5,
            p=1.013e5,
            condenser=condenser,
            reboiler=reboiler,
            specs=specs,
            side_draws=[SideDraw(3, "liquid", 0.05), SideDraw(4, "vapor", 0.04)],
            stage_duties=[StageDuty(2, 1000.0)],
            efficiency=0.85,
            model=pkg,
            reflux_temperature=340.0 if subcooled else None,
        )
    st, theta = captured["structure"], captured["theta"]
    encode, decode, residual, assemble, rows, columns, scales = dist._structured_system(st, theta)
    theta = {**theta, "_column_columns": columns, "_column_scales": scales}
    original = dist._pack(
        jnp.log(jnp.arange(10.0).reshape(5, 2) + 30),
        jnp.log(jnp.arange(10.0).reshape(5, 2) + 45),
        jnp.linspace(351.0, 375.0, 5),
        jnp.array(-2e6),
        jnp.array(3e6),
        st,
    )
    x = encode(original)
    np.testing.assert_allclose(decode(x), original, atol=1e-12)
    np.testing.assert_allclose(
        residual(x, theta), dist._residuals(original, theta, st)[rows], atol=1e-12
    )
    actual = jax.jit(lambda x, th: assemble(x, th).to_dense())(x, theta)
    expected = jax.jit(jax.jacfwd(residual))(x, theta)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_dense_and_block_columns_agree_and_side_draw_seed_is_exact():
    feed = Stream.from_fractions(
        ("benzene", "toluene"), jnp.array([0.5, 0.5]), 100.0, 365.0, 1.013e5
    )
    options = {
        "p": 1.013e5,
        "specs": [reflux_ratio(2.5), distillate_rate(45.0)],
        "side_draws": [SideDraw(4, "liquid", 0.03)],
    }
    block = rigorous_column([ColumnFeed(feed, 5)], 8, **options)
    dense = rigorous_column([ColumnFeed(feed, 5)], 8, **options, linear_solver="dense")
    assert block.report.converged and dense.report.converged
    for name in ("t", "x", "y", "liquid_flow", "vapor_flow", "condenser_duty", "reboiler_duty"):
        np.testing.assert_allclose(getattr(block, name), getattr(dense, name), rtol=1e-8, atol=1e-7)
    warm = rigorous_column(
        [ColumnFeed(feed, 5)], 8, **options, guess=block.warm_start(), max_iter=0
    )
    assert warm.report.converged and warm.report.iterations == 0
    np.testing.assert_allclose(
        block.stage_liquid[3] * 1.03, block.liquid_flow[3] * block.x[3], atol=1e-10
    )
    assert block.solver_info()["jacobian_directions"] == 17
    assert dense.solver_info()["jacobian_directions"] == 42


def test_column_strategy_validation():
    feed = Stream.from_fractions(("benzene", "toluene"), jnp.array([0.5, 0.5]), 100.0, 365.0, 1e5)
    with pytest.raises(ValueError, match="linear_solver"):
        rigorous_column(
            [ColumnFeed(feed, 4)],
            8,
            p=1e5,
            specs=[reflux_ratio(2.5), distillate_rate(50.0)],
            linear_solver="unknown",
        )


def test_column_hessian_matches_dense_and_finite_difference_of_gradient():
    feed = Stream.from_fractions(
        ("benzene", "toluene"), jnp.array([0.5, 0.5]), 100.0, 365.0, 1.013e5
    )

    def duty(reflux, solver="block"):
        result = rigorous_column(
            [ColumnFeed(feed, 5)],
            8,
            p=1.013e5,
            specs=[reflux_ratio(reflux), distillate_rate(45.0)],
            linear_solver=solver,
        )
        return result.reboiler_duty / 1e6

    gradient = jax.jit(jax.grad(duty))
    hessian = jax.jit(jax.hessian(duty))(2.5)
    dense = jax.jit(jax.hessian(lambda reflux: duty(reflux, "dense")))(2.5)
    finite_difference = (gradient(2.501) - gradient(2.499)) / 0.002
    assert jnp.isfinite(hessian)
    assert hessian == pytest.approx(dense, rel=2e-7, abs=1e-9)
    assert hessian == pytest.approx(finite_difference, rel=2e-4, abs=1e-8)
