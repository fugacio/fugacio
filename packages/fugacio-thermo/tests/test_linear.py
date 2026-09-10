"""Independent checks of structured assembly, pivots, and implicit derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fugacio.thermo.implicit import implicit_solution, newton_system_with_info
from fugacio.thermo.linear import BlockLayout, BorderedBlockJacobian, dense_jacobian


def system(n=5, b=2, k=2):
    rng = np.random.default_rng(813)
    lo = rng.normal(size=(n, b, b)) * 0.2
    up = rng.normal(size=(n, b, b)) * 0.2
    lo[0], up[-1] = 0, 0
    return BorderedBlockJacobian(
        jnp.array(lo),
        jnp.array(rng.normal(size=(n, b, b)) * 0.1 + np.eye(b)[None] * 3),
        jnp.array(up),
        jnp.array(rng.normal(size=(n, b, k)) * 0.1),
        jnp.array(rng.normal(size=(k, n, b)) * 0.1),
        jnp.array(np.eye(k) * 4 + rng.normal(size=(k, k)) * 0.1),
    )


@pytest.mark.parametrize("n,b,k", [(1, 1, 0), (1, 2, 2), (2, 3, 1), (7, 2, 0), (9, 3, 2)])
def test_block_solve_transpose_and_multiple_rhs(n, b, k):
    matrix = system(n, b, k)
    dense = np.asarray(matrix.to_dense())
    rhs = jnp.arange(matrix.shape[0], dtype=float) + 1
    for a, expected in ((matrix, dense), (matrix.transpose(), dense.T)):
        np.testing.assert_allclose(a.matvec(rhs), expected @ rhs, rtol=1e-12, atol=1e-12)
        for r in (rhs, jnp.stack((rhs, rhs**2), axis=1)):
            solved = jax.jit(lambda m, v: m.solve_with_info(v))(a, r)
            assert solved.report.accepted
            assert not solved.report.used_dense_fallback
            np.testing.assert_allclose(
                solved.value, np.linalg.solve(expected, r), rtol=1e-11, atol=1e-11
            )
    rows, cols = 1 + rhs / 3, 2 + rhs / 7
    np.testing.assert_allclose(
        matrix.scaled(rows, cols).to_dense(), rows[:, None] * dense * cols, atol=1e-12
    )


def test_pivot_across_blocks_uses_checked_dense_fallback():
    matrix = BorderedBlockJacobian(
        jnp.array([[[0.0]], [[1.0]]]),
        jnp.zeros((2, 1, 1)),
        jnp.array([[[1.0]], [[0.0]]]),
        jnp.empty((2, 1, 0)),
        jnp.empty((0, 2, 1)),
        jnp.empty((0, 0)),
    )
    result = matrix.solve_with_info(jnp.array([2.0, 3.0]))
    assert result.report.accepted and result.report.used_dense_fallback
    np.testing.assert_allclose(result.value, [3.0, 2.0])
    np.testing.assert_allclose(jax.jacfwd(matrix.solve)(jnp.ones(2)), [[0.0, 1.0], [1.0, 0.0]])

    def objective(scale):
        changed = matrix._replace(lower=matrix.lower * scale, upper=matrix.upper * scale)
        return jnp.sum(changed.solve(jnp.array([2.0, 3.0])))

    # Cross-block pivoting remains necessary for every scale. Its solution is
    # [3 / scale, 2 / scale], even though the attempted local factors are NaN.
    assert jax.jit(jax.grad(objective))(2.0) == pytest.approx(-1.25)
    assert jax.jit(jax.hessian(objective))(2.0) == pytest.approx(1.25)


def test_singular_linear_system_is_rejected():
    matrix = system(2, 1, 0)._replace(
        diagonal=jnp.zeros((2, 1, 1)), lower=jnp.zeros((2, 1, 1)), upper=jnp.zeros((2, 1, 1))
    )
    result = matrix.solve_with_info(jnp.ones(2))
    assert not result.report.accepted
    assert jnp.all(jnp.isnan(result.value))


def test_checked_solve_and_adjoint_acceptance_are_invariant_to_output_units():
    matrix = system()
    direction = jnp.eye(matrix.shape[0])[0]
    solve = jax.jit(lambda a, b: a.solve_with_info(b))
    for a in (matrix, matrix.transpose()):
        expected = np.linalg.solve(np.asarray(a.to_dense()), np.asarray(direction))
        for scale in (1e-8, 1.0, 1e7, 1e12):
            result = solve(a, direction * scale)
            assert result.report.accepted
            assert not result.report.used_dense_fallback
            np.testing.assert_allclose(result.value / scale, expected, rtol=1e-10, atol=1e-12)
        zero = solve(a, jnp.zeros_like(direction))
        assert zero.report.accepted and zero.report.backward_error == 0.0

    # A large metric cotangent must remain linear in its output units too.
    _, pull = jax.vjp(matrix.solve, jnp.ones_like(direction))
    expected = np.linalg.solve(np.asarray(matrix.to_dense()).T, np.asarray(direction))
    for scale in (1e-8, 1.0, 1e7, 1e12):
        actual = jax.jit(pull)(direction * scale)[0]
        np.testing.assert_allclose(actual / scale, expected, rtol=1e-10, atol=1e-12)


def test_adjoint_allows_roundoff_in_an_exactly_zero_component():
    # The transpose's first row requires x[0] == 0. Pivoting inside the block
    # can leave roundoff there; a purely componentwise relative test reports
    # error 1 regardless of how small that otherwise accurate component is.
    matrix = BorderedBlockJacobian(
        jnp.zeros((1, 2, 2)),
        jnp.array([[[1.0, 10.0], [0.0, 3.0]]]),
        jnp.zeros((1, 2, 2)),
        jnp.empty((1, 2, 0)),
        jnp.empty((0, 1, 2)),
        jnp.empty((0, 0)),
    )
    solve = jax.jit(lambda b: matrix.transpose().solve_with_info(b))
    for scale in (1e-8, 1.0, 1e7, 1e12):
        result = solve(jnp.array([0.0, 0.1]) * scale)
        assert result.report.accepted
        np.testing.assert_allclose(result.value / scale, [0.0, 0.1 / 3], atol=1e-12)


def test_small_backward_error_cannot_hide_an_inaccurate_newton_direction():
    # The first, independent equation gives a 1e12 solution component. It
    # makes a normwise backward error look small even when cancellation in
    # elimination leaves the other equations inaccurate relative to their RHS.
    matrix = BorderedBlockJacobian(
        jnp.array([[[0.0]], [[0.0]], [[1.0]]]),
        jnp.array([[[1e-12]], [[1e-12]], [[1.0]]]),
        jnp.array([[[0.0]], [[1.0]], [[0.0]]]),
        jnp.empty((3, 1, 0)),
        jnp.empty((0, 3, 1)),
        jnp.empty((0, 0)),
    )
    rhs = jnp.array([1.0, 1.0, 0.1])
    result = jax.jit(lambda a, b: a.solve_with_info(b))(matrix, rhs)
    assert result.report.accepted and result.report.used_dense_fallback
    assert result.report.relative_residual < 1e-10
    np.testing.assert_allclose(
        result.value, np.linalg.solve(np.asarray(matrix.to_dense()), rhs), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("n,b,k", [(1, 2, 0), (2, 2, 1), (3, 1, 2), (8, 2, 2)])
def test_coloring_recovers_nonlinear_jacobian_and_its_derivative(n, b, k):
    matrix = system(n, b, k)
    layout = BlockLayout(n, b, k)
    x = jnp.linspace(0.2, 0.8, layout.size)

    def residual(x, theta):
        return matrix.matvec(jnp.sin(x)) + theta * x**2

    jac = jax.jit(lambda x: layout.linearize(residual, x, 0.3))(x)
    expected = jax.jacfwd(residual)(x, 0.3)
    np.testing.assert_allclose(jac.to_dense(), expected, atol=1e-12)
    assert layout.check(residual, x, 0.3)["accepted"]
    direction = jnp.ones_like(x)
    actual_dot = jax.jvp(
        lambda x: layout.linearize(residual, x, 0.3).to_dense(), (x,), (direction,)
    )[1]
    expected_dot = jax.jvp(lambda x: jax.jacfwd(residual)(x, 0.3), (x,), (direction,))[1]
    np.testing.assert_allclose(actual_dot, expected_dot, atol=1e-12)


def test_coloring_check_detects_an_undeclared_distant_coupling():
    layout = BlockLayout(7, 2, 1)

    def residual(x, theta):
        return x.at[0].add(theta * x[10])

    assert not layout.check(residual, jnp.ones(layout.size), 2.0)["accepted"]


def test_structured_newton_forward_reverse_hessian_batch_and_scaling():
    matrix = system(4, 2, 1)
    layout = BlockLayout(4, 2, 1)
    target = jnp.linspace(0.4, 1.2, layout.size)

    def residual(x, theta):
        return matrix.matvec(x**2) - theta * target

    def solve(theta, structured=True):
        result = newton_system_with_info(
            residual,
            jnp.ones(layout.size),
            theta,
            jacobian=(lambda x, th: layout.linearize(residual, x, th)) if structured else None,
            scale=jnp.linspace(0.5, 2.0, layout.size),
            residual_scale=jnp.linspace(1.0, 3.0, layout.size),
            lower=jnp.zeros(layout.size),
        )
        return result.value

    for transform in (lambda f: f, jax.jacfwd, jax.jacrev, lambda f: jax.jacfwd(jax.jacrev(f))):
        actual = jax.jit(transform(solve))(jnp.asarray(2.0))
        expected = transform(lambda t: solve(t, False))(jnp.asarray(2.0))
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-9)
    values = jax.jit(jax.vmap(solve))(jnp.array([1.0, 2.0, 3.0]))
    for i, theta in enumerate([1.0, 2.0, 3.0]):
        np.testing.assert_allclose(values[i], solve(theta, False), atol=1e-9)


def test_matrix_parameter_gradient_including_border_and_transpose():
    matrix = system()
    rhs = jnp.arange(matrix.shape[0], dtype=float)

    def objective(t, dense=False):
        changed = matrix._replace(
            diagonal=matrix.diagonal * t, border_rows=matrix.border_rows * t**2
        )
        value = jnp.linalg.solve(changed.to_dense(), rhs) if dense else changed.solve(rhs)
        return jnp.sum(jnp.sin(value))

    for op in (jax.grad, jax.hessian):
        assert op(objective)(1.1) == pytest.approx(op(lambda t: objective(t, True))(1.1), rel=1e-9)


def test_sequential_dense_assembly_matches_vectorized_and_higher_derivatives():
    x = jnp.linspace(0.2, 0.8, 6).reshape(2, 3)

    def residual(x, theta):
        return jnp.sin(x * theta) + 0.1 * jnp.sum(x**2)

    for transform in (lambda f: f, jax.jacfwd, jax.jacrev):
        serial = jax.jit(
            transform(lambda t: dense_jacobian(residual, x, t, vectorize=False).matrix)
        )(0.3)
        batch = transform(lambda t: dense_jacobian(residual, x, t).matrix)(0.3)
        np.testing.assert_allclose(serial, batch, rtol=1e-11, atol=1e-12)


def test_joint_implicit_linearization_reuses_residual_and_preserves_higher_derivatives():
    calls = []

    def residual(x, theta):
        calls.append(1)
        return x**2 + 0.1 * jnp.sum(x) - theta["target"]

    root = jnp.array([0.8, 1.2])
    target = root**2 + 0.1 * jnp.sum(root)

    def solve(t, strategy="sequential"):
        return implicit_solution(
            residual, root, {"target": t, "inactive": jnp.arange(3.0)}, jnp.array(True), strategy
        )

    jacobian = jax.jacfwd(solve)(target)
    assert calls == [1]
    expected = jnp.linalg.inv(jnp.diag(2 * root) + 0.1)
    np.testing.assert_allclose(jacobian, expected, atol=1e-12)
    for transform in (jax.jacrev, lambda f: jax.jacfwd(jax.jacrev(f))):
        actual = jax.jit(transform(solve))(target)
        dense = transform(lambda t: solve(t, None))(target)
        np.testing.assert_allclose(actual, dense, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("sizes", [(0, 1, 0), (1, 0, 0), (1, 1, -1), (True, 2, 0), (1, 2.5, 0)])
def test_invalid_layout(sizes):
    with pytest.raises(ValueError):
        BlockLayout(*sizes)
