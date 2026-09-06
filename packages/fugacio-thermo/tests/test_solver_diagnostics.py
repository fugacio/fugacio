"""Analytic roots and failure cases for the differentiable solver contract."""

import json

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo.diagnostics import ConvergenceError, SolveStatus, require_converged
from fugacio.thermo.energy import _implicit_temperature
from fugacio.thermo.implicit import (
    bracketed_root,
    fixed_point_with_info,
    newton_root,
    newton_system_with_info,
)


@pytest.mark.parametrize("outer", ["bracketed", "newton", "temperature"])
@pytest.mark.parametrize("inner", ["bracketed", "newton", "temperature"])
def test_nested_scalar_roots_preserve_pytree_derivatives(outer, inner):
    def root(method, residual, params, upper=30.0):
        if method == "bracketed":
            return bracketed_root(residual, params, jnp.asarray(0.01), jnp.asarray(upper))
        if method == "temperature":
            return _implicit_temperature(residual, params, 4.0, 0.01, upper, 1e-12, 100)
        return newton_root(residual, params, jnp.asarray(4.0))

    def solve(params):
        def residual(x, th):
            y = root(inner, lambda y, p: y**2 - p, x**2 + th["offset"])
            return y - th["scale"] * jnp.sum(th["weights"] ** 2)

        return root(outer, residual, params, upper=10.0)

    def exact(params):
        return jnp.sqrt((params["scale"] * jnp.sum(params["weights"] ** 2)) ** 2 - params["offset"])

    params = {
        "offset": jnp.asarray(0.7),
        "scale": jnp.asarray(2.0),
        "weights": jnp.array([0.4, 0.8, 1.1]),
    }
    direction = {
        "offset": jnp.asarray(-0.3),
        "scale": jnp.asarray(0.2),
        "weights": jnp.array([0.1, -0.2, 0.3]),
    }
    value, tangent = jax.jit(lambda p, d: jax.jvp(solve, (p,), (d,)))(params, direction)
    expected, expected_tangent = jax.jvp(exact, (params,), (direction,))
    assert value == pytest.approx(float(expected), abs=1e-10)
    assert tangent == pytest.approx(float(expected_tangent), rel=1e-9)
    for actual, wanted in zip(
        jax.tree.leaves(jax.jit(jax.grad(solve))(params)),
        jax.tree.leaves(jax.grad(exact)(params)),
        strict=True,
    ):
        assert jnp.allclose(actual, wanted, rtol=1e-9, atol=1e-10)
    # Differentiating the implicit root again must retain its parameter dependence.
    for actual, wanted in zip(
        jax.tree.leaves(jax.jit(jax.hessian(solve))(params)),
        jax.tree.leaves(jax.hessian(exact)(params)),
        strict=True,
    ):
        assert jnp.allclose(actual, wanted, rtol=1e-8, atol=1e-10)


def test_scaled_newton_jvp_vjp_hessian_and_batch():
    def solve(target):
        return newton_system_with_info(
            lambda x, th: x**2 - th,
            jnp.array([1.0]),
            target,
            scale=jnp.array([10.0]),
            residual_scale=jnp.array([100.0]),
            tol=1e-12,
            lower=jnp.array([0.0]),
        ).value[0]

    target = jnp.array([4.0])
    assert solve(target) == pytest.approx(2.0)
    assert jax.jacfwd(solve)(target)[0] == pytest.approx(0.25)
    assert jax.jacrev(solve)(target)[0] == pytest.approx(0.25)
    assert jax.hessian(solve)(target)[0, 0] == pytest.approx(-0.03125)
    assert jnp.allclose(jax.jit(jax.vmap(solve))(jnp.array([[4.0], [9.0]])), jnp.array([2.0, 3.0]))


def test_iteration_limit_is_not_success_and_has_no_valid_sensitivity():
    def solve(target):
        return newton_system_with_info(
            lambda x, th: x**2 - th, jnp.array([1.0]), target, max_iter=1
        )

    result = jax.jit(solve)(jnp.array([100.0]))
    assert not result.report.converged
    assert result.report.status == SolveStatus.MAX_ITERATIONS
    assert jnp.isfinite(result.value).all()
    assert not jnp.isfinite(jax.jacrev(lambda t: solve(t).value)(jnp.array([100.0]))).all()
    with pytest.raises(ConvergenceError, match=r"reactor.*energy"):
        require_converged(result.report, "reactor", ("energy",))
    json.dumps(result.report.to_dict(), allow_nan=False)


def test_infeasible_bounds_and_nonfinite_residual_are_reported():
    bounded = newton_system_with_info(
        lambda x, th: x - th, jnp.array([0.0]), jnp.array([2.0]), upper=jnp.array([1.0])
    )
    assert bounded.report.status == SolveStatus.STALLED
    invalid = newton_system_with_info(lambda x, _: jnp.log(x), jnp.array([-1.0]), None)
    assert invalid.report.status == SolveStatus.NONFINITE
    assert invalid.report.to_dict()["residual_norm"] is None
    json.dumps(invalid.report.to_dict(), allow_nan=False)


def test_fixed_point_checks_the_initial_solution_and_actual_residual():
    result = fixed_point_with_info(lambda x, th: 0.5 * x + th, jnp.array([2.0]), jnp.array([1.0]))
    assert result.report.converged
    assert result.report.iterations == 0
    assert jax.jacfwd(
        lambda th: fixed_point_with_info(lambda x, p: 0.5 * x + p, jnp.array([0.0]), th).value
    )(jnp.array([1.0]))[0, 0] == pytest.approx(2.0)


def test_checked_bisection_rejects_invalid_brackets_and_discontinuities():
    from fugacio.thermo.implicit import bracketed_root_with_info

    endpoint = bracketed_root_with_info(lambda x, p: x - p, 1.0, 1.0, 2.0)
    assert endpoint.value == 1.0
    assert endpoint.report.converged and endpoint.report.iterations == 0
    invalid = bracketed_root_with_info(lambda x, _: x**2 + 1, None, -1.0, 1.0)
    assert invalid.report.status == SolveStatus.INVALID_INPUT
    jump = bracketed_root_with_info(lambda x, _: jnp.where(x < 0.2, -1.0, 1.0), None, 0.0, 1.0)
    assert not jump.report.converged
    assert jump.report.residual_norm == 1.0
