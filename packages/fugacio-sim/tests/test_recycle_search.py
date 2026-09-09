import jax
import jax.numpy as jnp
import pytest

from fugacio.sim.flowsheet import tear_solve_with_info


def test_broyden_backtracks_nonfinite_full_step_and_preserves_implicit_derivatives():
    # At x=0, the full step 4*log(4) crosses the logarithm's domain boundary.
    # The backtracked point converges to the exact root x=theta-1.
    def solve(theta):
        return tear_solve_with_info(
            lambda x, t: x + 4 * jnp.log(t - x),
            jnp.array(0.0),
            theta,
            method="broyden",
            tol=1e-11,
        )

    result = jax.jit(solve)(4.0)
    assert result.report.converged and result.value == pytest.approx(3.0, abs=1e-10)
    assert 1 < result.report.iterations < 20
    derivative = jax.jit(jax.grad(lambda t: solve(t).value))(4.0)
    assert derivative == pytest.approx(1.0, abs=1e-10)
    assert jax.jvp(lambda t: solve(t).value, (4.0,), (2.0,))[1] == pytest.approx(2.0)
    assert jax.grad(jax.grad(lambda t: solve(t).value))(4.0) == pytest.approx(0.0, abs=1e-10)


def test_broyden_exhausted_search_and_initially_converged_point_have_honest_reports():
    failed = jax.jit(
        lambda: tear_solve_with_info(
            lambda x, t: jnp.where(x == 0, t, jnp.nan),
            jnp.array(0.0),
            jnp.array(1.0),
            method="broyden",
            max_iter=4,
        )
    )()
    assert not failed.report.converged
    assert failed.report.iterations == 1
    assert failed.value == pytest.approx(1 / 128)
    exact = tear_solve_with_info(
        lambda x, t: x + 4 * jnp.log(t - x),
        jnp.array(3.0),
        jnp.array(4.0),
        method="broyden",
        max_iter=0,
    )
    assert exact.report.converged and exact.report.iterations == 0
