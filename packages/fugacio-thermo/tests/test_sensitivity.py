import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fugacio.thermo.implicit import newton_system_with_info
from fugacio.thermo.sensitivity import derivative_strategy, linearize


@pytest.mark.parametrize("inputs,outputs", [(1, 4), (7, 2), (3, 3)])
@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_linearize_reuses_primal_and_matches_both_jacobian_orientations(
    inputs, outputs, batch_size
):
    calls = []
    matrix = jnp.arange(inputs * outputs, dtype=float).reshape(outputs, inputs) / 20

    def function(x):
        calls.append(1)
        return jnp.sin(matrix @ x), {"accepted": jnp.array(True)}

    point = jnp.linspace(0.1, 0.5, inputs)
    local = jax.block_until_ready(linearize(function, point, has_aux=True))
    expected = jax.jacfwd(lambda x: jnp.sin(matrix @ x))(point)
    for mode in ("forward", "reverse", "auto"):
        np.testing.assert_allclose(
            local.jacobian(mode=mode, batch_size=batch_size), expected, atol=1e-12
        )
    np.testing.assert_allclose(local.jvp(jnp.ones(inputs)), expected @ np.ones(inputs), atol=1e-12)
    np.testing.assert_allclose(
        local.vjp(jnp.ones(outputs)), np.ones(outputs) @ expected, atol=1e-12
    )
    assert calls == [1]
    assert local.auxiliary["accepted"]


def test_reusable_implicit_linearization_jit_reverse_and_higher_derivatives():
    def solve(x):
        result = newton_system_with_info(lambda y, t: y**2 - t, jnp.ones(4), x, tol=1e-12)
        return jnp.sum(result.value**3)[None], result.report.converged

    x = jnp.array([1.1, 1.2, 1.3, 1.4])
    local = linearize(solve, x, has_aux=True)
    assert local.auxiliary
    np.testing.assert_allclose(local.jacobian(), 1.5 * np.sqrt(x)[None], rtol=1e-10)
    np.testing.assert_allclose(jax.jit(local.vjp)(jnp.ones(1)), 1.5 * np.sqrt(x), rtol=1e-10)
    hessian = jax.jit(jax.jacfwd(lambda v: linearize(solve, v, has_aux=True).vjp(jnp.ones(1))))(x)
    np.testing.assert_allclose(hessian, np.diag(0.75 / np.sqrt(x)), rtol=1e-10, atol=1e-11)


def test_scalar_and_multidimensional_shapes():
    x = jnp.arange(6.0).reshape(2, 3) / 8

    def fn(v):
        return jnp.sum(jnp.exp(v))

    local = linearize(fn, x)
    assert local.jacobian(mode="reverse").shape == (2, 3)
    np.testing.assert_allclose(local.jacobian(), jnp.exp(x))
    scalar = linearize(lambda v: v**3, jnp.array(2.0))
    assert scalar.jacobian().shape == () and scalar.jacobian() == 12.0
    with pytest.raises(ValueError, match="tangent shape"):
        local.jvp(jnp.ones(6))
    with pytest.raises(ValueError, match="cotangent shape"):
        local.vjp(jnp.ones(1))


@pytest.mark.parametrize(
    "mode,batch", [("invalid", 1), ("auto", 0), ("auto", True), ("forward", 1.5)]
)
def test_invalid_derivative_options(mode, batch):
    with pytest.raises(ValueError):
        derivative_strategy(4, 1, mode=mode, batch_size=batch)


def test_strategy_many_variables_uses_one_reverse_direction():
    strategy = derivative_strategy(40, 1)
    assert strategy["mode"] == "reverse"
    assert strategy["directions"] == 1
    assert strategy["linearizations_per_evaluation"] == 1
