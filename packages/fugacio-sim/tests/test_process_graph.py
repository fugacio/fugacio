"""Analytic and cross-formulation contracts for the shared process runtime."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fugacio.sim import Flowsheet, Stream
from fugacio.sim.graph import ProcessGraph, ProcessUnit, ResidualGraph, SpecifiedGraph
from fugacio.sim.numerics import SparseJacobian, clear_factor_cache, newton_iterations
from fugacio.thermo.diagnostics import SolveStatus


def matrix(values=None):
    return SparseJacobian(
        jnp.asarray([3.0, 2.0, 1.0] if values is None else values),
        (0, 1, 1),
        (0, 1, 0),
        2,
    )


def stream(flow=2.0, vapor=None):
    return Stream(
        jnp.atleast_1d(flow),
        jnp.asarray(300.0),
        jnp.asarray(1e5),
        ("methane",),
        None if vapor is None else jnp.atleast_1d(vapor),
    )


def recycle():
    flow = Flowsheet().feed("feed", stream())
    flow.unit(
        "mix",
        lambda a, b, p: Stream(a.n + b.n, a.t, a.p, a.components),
        inputs=("feed", "recycle"),
        outputs=("mixed",),
    )
    flow.unit(
        "split",
        lambda a, p: (a.scaled(p), a.scaled(1 - p)),
        inputs=("mixed",),
        outputs=("recycle", "product"),
    )
    return flow


def test_sparse_solve_checks_primal_transpose_and_multiple_right_hand_sides():
    a = matrix()
    rhs = jnp.array([[1.0, 0.0], [4.0, 2.0]])
    result = a.solve_with_info(rhs)
    assert result.report.accepted
    np.testing.assert_allclose(result.value, np.linalg.solve(np.asarray(a.to_dense()), rhs))
    np.testing.assert_allclose(a.transpose().solve(rhs), np.linalg.solve(a.to_dense().T, rhs))
    np.testing.assert_allclose(jax.jacrev(a.solve)(rhs[:, 0]), np.linalg.inv(a.to_dense()))
    assert not result.report.used_dense_fallback


def test_sparse_implicit_matrix_derivative_and_hessian_match_dense():
    def sparse(x):
        return jnp.sum(matrix(jnp.array([x, 2.0, 1.0])).solve(jnp.array([1.0, 4.0])) ** 2)

    def dense(x):
        return jnp.sum(
            jnp.linalg.solve(jnp.array([[x, 0.0], [1.0, 2.0]]), jnp.array([1.0, 4.0])) ** 2
        )

    for transform in (lambda f: f, jax.grad, jax.hessian):
        assert transform(sparse)(3.0) == pytest.approx(float(transform(dense)(3.0)), rel=1e-10)
    np.testing.assert_allclose(jax.jit(jax.grad(sparse))(3.0), jax.grad(dense)(3.0))


def test_sparse_duplicate_entries_zero_rhs_and_singular_failure():
    a = SparseJacobian(jnp.array([1.0, 2.0, 4.0]), (0, 0, 1), (0, 0, 1), 2)
    np.testing.assert_allclose(a.solve(jnp.array([6.0, 8.0])), [2.0, 2.0])
    assert a.solve_with_info(jnp.zeros(2)).report.accepted
    singular = SparseJacobian(jnp.zeros(2), (0, 1), (0, 1), 2).solve_with_info(jnp.ones(2))
    assert not singular.report.accepted
    assert jnp.isnan(singular.value).all()
    assert not singular.report.used_dense_fallback
    empty = SparseJacobian(jnp.empty(0), (), (), 2)
    np.testing.assert_array_equal(empty.to_dense(), np.zeros((2, 2)))
    np.testing.assert_array_equal(empty.matvec(jnp.ones(2)), np.zeros(2))
    assert not empty.solve_with_info(jnp.ones(2)).report.accepted


def test_sparse_factor_reuse_is_bounded_and_keyed_by_every_coefficient(monkeypatch):
    import fugacio.sim.numerics as numerics

    clear_factor_cache()
    calls = []
    original = numerics.splu

    def factor(a):
        calls.append(a.copy())
        return original(a)

    monkeypatch.setattr(numerics, "splu", factor)
    a = matrix()
    a.solve(jnp.ones(2)).block_until_ready()
    a.solve(jnp.array([2.0, 3.0])).block_until_ready()
    jax.grad(lambda b: jnp.sum(a.solve(b)))(jnp.ones(2)).block_until_ready()
    assert len(calls) == 1
    for first in range(4, 20):
        matrix(jnp.array([float(first), 2.0, 1.0])).solve(jnp.ones(2)).block_until_ready()
    assert len(numerics._FACTOR_CACHE) <= numerics._FACTOR_COUNT
    assert sum(item[1] for item in numerics._FACTOR_CACHE.values()) <= numerics._FACTOR_BYTES
    clear_factor_cache()


def test_sparse_backward_error_uses_the_coalesced_matrix_norm(monkeypatch):
    a = SparseJacobian(jnp.array([1e6, -1e6 + 1.0, 2.0]), (0, 0, 1), (0, 0, 1), 2)
    rhs = jnp.array([3.0, 4.0])
    approximate = jnp.array([3.0 + 1e-11, 2.0])
    monkeypatch.setattr(SparseJacobian, "_factor_solve", lambda *args, **kwargs: approximate)
    result = a.solve_with_info(rhs)
    dense = np.asarray(a.to_dense())
    residual = np.linalg.norm(dense @ approximate - rhs, np.inf)
    expected = residual / (
        np.linalg.norm(dense, np.inf) * np.linalg.norm(approximate, np.inf)
        + np.linalg.norm(rhs, np.inf)
    )
    assert result.report.accepted
    assert result.report.backward_error == pytest.approx(expected, rel=1e-10, abs=0)


@pytest.mark.parametrize("strategy", ["sequential", "simultaneous"])
@pytest.mark.parametrize("linear_solver", ["sparse", "dense"])
def test_graph_recycle_analytic_primal_forward_reverse_and_second_derivative(
    strategy, linear_solver
):
    flow = recycle()

    def product(p):
        return flow.solve(p, strategy=strategy, linear_solver=linear_solver)["mixed"].total

    assert product(0.2) == pytest.approx(2 / 0.8)
    assert jax.jacfwd(product)(0.2) == pytest.approx(2 / 0.8**2)
    assert jax.grad(product)(0.2) == pytest.approx(2 / 0.8**2)
    assert jax.hessian(product)(0.2) == pytest.approx(4 / 0.8**3)
    assert jax.jit(product)(0.2) == pytest.approx(2 / 0.8)


def test_feed_derivatives_and_changed_feed_values_do_not_reuse_a_stale_state():
    flow = recycle()

    def solve(n):
        current = copy.copy(flow)
        current.feeds = {"feed": stream(n)}
        return current.solve(0.2)["mixed"].total

    assert jax.grad(solve)(2.0) == pytest.approx(1.25)
    assert solve(4.0) == pytest.approx(5.0)
    assert solve(2.0) == pytest.approx(2.5)


@pytest.mark.parametrize("strategy", ["sequential", "simultaneous"])
def test_feed_only_graph_retains_its_input_derivatives(strategy):
    def solve(n):
        return Flowsheet().feed("feed", stream(n)).solve(strategy=strategy)["feed"].total

    assert solve(3.0) == 3.0
    assert jax.grad(solve)(3.0) == 1.0
    assert jax.jit(jax.jacfwd(solve))(3.0) == 1.0
    assert Flowsheet().solve(strategy=strategy) == {}


def test_primal_boundary_accepts_python_scalar_stream_coordinates():
    def solve(n):
        flow = Flowsheet().feed("feed", Stream(jnp.atleast_1d(n), 300.0, 1e5, ("methane",)))
        flow.unit(
            "unit",
            lambda s, p: Stream(s.n * 2, 300.0, 1e5, s.components),
            inputs=("feed",),
            outputs=("product",),
        )
        return flow.solve()["product"].total

    assert solve(3.0) == 6.0
    assert jax.grad(solve)(3.0) == 2.0
    assert jax.jit(jax.jacfwd(solve))(3.0) == 2.0


def test_pure_saturation_inventory_is_an_independent_connection_coordinate():
    flow = Flowsheet().feed("feed", stream(2.0, 0.5))
    flow.unit(
        "split",
        lambda a, p: (a.scaled(p), a.scaled(1 - p)),
        inputs=("feed",),
        outputs=("one", "two"),
    )
    for strategy in ("sequential", "simultaneous"):
        result = flow.solve(0.3, strategy=strategy)
        assert result["one"].vapor_n[0] == pytest.approx(0.15)
        assert result["two"].vapor_n[0] == pytest.approx(0.35)
        assert (
            jax.grad(
                lambda p, strategy=strategy: flow.solve(p, strategy=strategy)["one"].vapor_n[0]
            )(0.3)
            == 0.5
        )


def test_failed_recycle_retains_state_but_has_nonfinite_derivatives():
    flow = recycle()
    result = flow.solve_with_info(0.8, max_iter=0)
    assert not result.converged
    assert jnp.isfinite(result.streams["mixed"].total)
    derivative = jax.grad(lambda p: flow.solve_with_info(p, max_iter=0).streams["mixed"].total)(0.8)
    assert not jnp.isfinite(derivative)


def test_linear_storage_for_long_chain_and_sparse_dense_agreement():
    units = tuple(
        ProcessUnit(
            str(i), lambda a, p: a.scaled(p), ("feed" if i == 0 else str(i - 1),), (str(i),)
        )
        for i in range(100)
    )
    graph = ProcessGraph(("feed",), units).compile({str(i): stream() for i in range(100)})
    report = graph.diagnose()
    assert report["structurally_square_and_matched"]
    assert report["stored_coefficients"] < 21 * 100
    assert report["largest_unit_input"] == 4
    assert graph.size == 400
    point = graph.pack({str(i): stream() for i in range(100)})
    a, _ = graph.linearize(point, (0.5, {"feed": stream()}))
    direction = jnp.linspace(0.0, 1.0, graph.size)
    _, expected = jax.jvp(
        lambda x: graph.residual(x, (0.5, {"feed": stream()})), (point,), (direction,)
    )
    np.testing.assert_allclose(a.matvec(direction), expected, atol=1e-12)


def test_specification_border_solves_coupled_state_and_parameter_without_nested_solves():
    flow = recycle()
    seed = flow.solve(0.2)
    graph = flow.compile(seed)
    # The specified mixed flow depends on the external target; the recycle
    # fraction is an unknown. Both state and manipulated parameter differentiate.
    system = SpecifiedGraph(
        graph,
        1,
        lambda x, target: (x[0], {"feed": stream()}),
        lambda streams, bound: jnp.atleast_1d(streams["mixed"].total),
    )
    # A dynamic target belongs in the bound parameters; a local custom equation
    # graph makes that dependence explicit without adding a plant solver.
    scalar = ResidualGraph(
        2, [((0, 1), 2, lambda x, target: jnp.array([x[0] * (1 - x[1]) - 2, x[0] - target]))]
    )

    def solve(target):
        result = newton_iterations(
            scalar.residual, lambda x, p: scalar.linearize(x, p)[0], jnp.array([2.5, 0.2]), target
        )
        return scalar.attach(result.value, target, result.report.converged)

    np.testing.assert_allclose(solve(4.0), [4.0, 0.5], atol=1e-9)
    np.testing.assert_allclose(jax.jacfwd(solve)(4.0), [1.0, 0.125], atol=1e-9)
    np.testing.assert_allclose(jax.jacrev(solve)(4.0), [1.0, 0.125], atol=1e-9)
    x = jnp.concatenate((graph.pack(seed), jnp.array([0.2])))
    a, _ = system.linearize(x, 4.0)
    np.testing.assert_allclose(a.to_dense(), jax.jacfwd(system.residual)(x, 4.0))


def test_bounded_newton_retains_best_state_and_reports_impossible_specification():
    system = ResidualGraph(1, [((0,), 1, lambda x, p: x - p)])
    result = newton_iterations(
        system.residual,
        lambda x, p: system.linearize(x, p)[0],
        jnp.array([0.5]),
        2.0,
        lower=jnp.zeros(1),
        upper=jnp.ones(1),
    )
    assert not result.report.converged
    assert result.report.status == SolveStatus.STALLED
    assert result.value[0] == 1.0
    assert result.report.residual_norm == 1.0


def test_graph_rejects_ambiguous_or_missing_stream_ownership():
    with pytest.raises(ValueError, match="both a feed"):
        ProcessGraph(("a",), (ProcessUnit("one", lambda a, p: a, ("a",), ("a",)),)).edges()
    with pytest.raises(ValueError, match="undefined"):
        ProcessGraph((), (ProcessUnit("one", lambda a, p: a, ("a",), ("b",)),)).edges()


def test_retained_unit_duties_follow_the_implicit_connection_state():
    from typing import NamedTuple

    from fugacio.thermo.diagnostics import residual_report

    class Result(NamedTuple):
        outlets: tuple
        heat: object
        work: object
        report: object

    flow = recycle()

    def duty(feed, p):
        return Result((feed,), feed.total * p**2, jnp.asarray(0.0), residual_report(jnp.zeros(1)))

    flow.unit("duty", duty, inputs=("mixed",), outputs=("measured",))

    def heat(p):
        result = flow.solve_with_info(p, retain_results=True)
        return result.unit_results["duty"].heat

    expected = 2 * (0.4 / 0.8 + 0.04 / 0.8**2)
    assert jax.grad(heat)(0.2) == pytest.approx(expected)
    assert jax.jit(jax.grad(heat))(0.2) == pytest.approx(expected)


@pytest.mark.parametrize("strategy", ["sequential", "simultaneous"])
def test_unit_failure_gates_graph_sensitivities_even_with_finite_best_iterates(strategy):
    from typing import NamedTuple

    from fugacio.thermo.diagnostics import residual_report

    class Result(NamedTuple):
        outlets: tuple
        report: object

    flow = Flowsheet().feed("feed", stream())
    flow.unit(
        "failed",
        lambda feed, p: Result((feed.scaled(p),), residual_report(jnp.ones(1))),
        inputs=("feed",),
        outputs=("out",),
    )
    result = flow.solve_with_info(0.5, strategy=strategy)
    assert not result.converged
    assert result.streams["out"].total == 1.0
    derivative = jax.grad(
        lambda p: flow.solve_with_info(p, strategy=strategy).streams["out"].total
    )(0.5)
    assert not jnp.isfinite(derivative)


def test_simultaneous_derivative_checks_final_units_instead_of_seed_units():
    from typing import NamedTuple

    from fugacio.thermo.diagnostics import residual_report

    class Result(NamedTuple):
        outlets: tuple
        report: object

    flow = Flowsheet().feed("feed", stream())

    def unit(feed, recycled, p):
        output = stream(feed.total + p * recycled.total)
        report = residual_report(jnp.atleast_1d(jnp.where(recycled.total > 3, 1.0, 0.0)))
        return Result((output,), report)

    flow.unit("unit", unit, inputs=("feed", "out"), outputs=("out",))
    flow.tear("out", stream(10.0))

    def solve(p):
        return flow.solve_with_info(p, strategy="simultaneous")

    assert solve(0.2).converged
    assert jax.grad(lambda p: solve(p).streams["out"].total)(0.2) == pytest.approx(2 / 0.8**2)


def test_singular_newton_step_reports_linear_failure_without_dense_fallback():
    graph = ResidualGraph(1, [((0,), 1, lambda x, p: jnp.ones_like(x))])
    result = newton_iterations(
        graph.residual, lambda x, p: graph.linearize(x, p)[0], jnp.zeros(1), None
    )
    assert result.report.status == SolveStatus.SINGULAR
    assert result.report.iterations == 1
