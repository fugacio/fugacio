"""Analytic roots and failure cases for the differentiable solver contract."""

import json

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo.diagnostics import ConvergenceError, SolveStatus, require_converged
from fugacio.thermo.implicit import fixed_point_with_info, newton_system_with_info


@pytest.mark.parametrize("kind", ["bracketed", "newton", "temperature"])
def test_scalar_root_pytree_directions_and_hessian(kind):
    from fugacio.thermo.energy import _implicit_temperature
    from fugacio.thermo.implicit import bracketed_root, newton_root

    def solve(values, seed=1.0):
        params = {"target": values[0], "coefficient": values[1], "fixed": jnp.array([0.5, 0.5])}

        def residual(x, th):
            return th["coefficient"] * x**2 - th["target"] * jnp.sum(th["fixed"])

        if kind == "bracketed":
            return bracketed_root(residual, params, 0.1 * seed, 10.0)
        if kind == "newton":
            return newton_root(residual, params, seed)
        return _implicit_temperature(residual, params, seed, 0.1, 10.0, 1e-12, 100)

    def exact(values):
        return jnp.sqrt(values[0] / values[1])

    x = jnp.array([8.0, 2.0])
    assert solve(x) == pytest.approx(exact(x), abs=1e-10)
    assert jnp.allclose(jax.jacfwd(solve)(x), jax.jacfwd(exact)(x), atol=1e-10)
    assert jnp.allclose(jax.jacrev(solve)(x), jax.jacrev(exact)(x), atol=1e-10)
    assert jnp.allclose(jax.jit(jax.hessian(solve))(x), jax.hessian(exact)(x), atol=1e-10)
    assert jax.grad(lambda seed: solve(x, seed))(1.0) == 0.0


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
