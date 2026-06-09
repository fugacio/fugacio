"""Differentiable optimization core.

Covers each solver against problems with a known optimum (quadratics, Rosenbrock,
box- and equality-constrained programs, and a least-squares fit) and -- the
distinguishing feature -- checks that :func:`argmin` carries exact gradients of
the *solution* with respect to the parameters, against both closed form and a
finite difference, including the active-set case where a bound switches the
sensitivity off.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim.optimize import argmin, least_squares, minimize


def test_bfgs_quadratic() -> None:
    # min 0.5 (x-a)^T (x-a)  =>  x* = a.
    a = jnp.array([1.0, -2.0, 3.0])
    res = minimize(lambda x, _: 0.5 * jnp.sum((x - a) ** 2), jnp.zeros(3))
    assert bool(res.converged)
    assert jnp.allclose(res.x, a, atol=1e-5)


def test_bfgs_rosenbrock() -> None:
    def rosen(x: jax.Array, _: object) -> jax.Array:
        return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

    res = minimize(rosen, jnp.array([-1.2, 1.0]), max_iter=500)
    assert jnp.allclose(res.x, jnp.ones(2), atol=1e-3)


def test_newton_and_gradient_descent_agree() -> None:
    a = jnp.array([2.0, 5.0])

    def f(x: jax.Array, _: object) -> jax.Array:
        return jnp.sum((x - a) ** 2) + 0.1 * jnp.sum(x**4)

    x_newton = minimize(f, jnp.zeros(2), method="newton").x
    x_gd = minimize(f, jnp.zeros(2), method="gradient-descent", max_iter=2000).x
    assert jnp.allclose(x_newton, x_gd, atol=1e-3)


def test_spg_bound_constrained() -> None:
    # min (x-5)^2 on [0, 3]  =>  x* = 3 (upper bound active).
    res = minimize(
        lambda x, _: jnp.sum((x - 5.0) ** 2),
        jnp.array([1.0]),
        bounds=(jnp.array([0.0]), jnp.array([3.0])),
    )
    assert float(res.x[0]) == pytest.approx(3.0, abs=1e-4)


def test_auglag_equality() -> None:
    # min x^2 + y^2  s.t.  x + y = 1  =>  (0.5, 0.5).
    res = minimize(
        lambda v, _: v[0] ** 2 + v[1] ** 2,
        jnp.array([2.0, -1.0]),
        eq_constraints=lambda v, _: jnp.array([v[0] + v[1] - 1.0]),
    )
    assert bool(res.converged)
    assert jnp.allclose(res.x, jnp.array([0.5, 0.5]), atol=1e-4)
    assert float(res.constraint_violation) < 1e-5


def test_auglag_inequality() -> None:
    # min (x-2)^2  s.t.  x <= 1  =>  x* = 1.
    res = minimize(
        lambda x, _: (x[0] - 2.0) ** 2,
        jnp.array([0.0]),
        ineq_constraints=lambda x, _: jnp.array([x[0] - 1.0]),
    )
    assert float(res.x[0]) == pytest.approx(1.0, abs=1e-3)


def test_least_squares_line_fit() -> None:
    # Fit y = m x + b to exact data; recover (m, b).
    xs = jnp.array([0.0, 1.0, 2.0, 3.0])
    ys = 2.0 * xs - 1.0

    def resid(p: jax.Array, _: object) -> jax.Array:
        return p[0] * xs + p[1] - ys

    res = least_squares(resid, jnp.array([0.0, 0.0]))
    assert jnp.allclose(res.x, jnp.array([2.0, -1.0]), atol=1e-5)
    assert float(res.fun) < 1e-10


def test_argmin_unconstrained_gradient_closed_form() -> None:
    # x*(theta) = argmin (x - theta)^2 = theta  =>  d x*/d theta = 1.
    def solve(theta: jax.Array) -> jax.Array:
        return argmin(lambda x, th: jnp.sum((x - th) ** 2), jnp.zeros(2), theta)

    theta = jnp.array([3.0, -1.0])
    assert jnp.allclose(solve(theta), theta, atol=1e-5)
    jac = jax.jacobian(solve)(theta)
    assert jnp.allclose(jac, jnp.eye(2), atol=1e-4)


def test_argmin_parametric_quadratic_gradient_matches_fd() -> None:
    # min 0.5 x^T A x - (theta b)^T x  =>  x* = theta A^{-1} b, linear in theta.
    a_mat = jnp.array([[3.0, 1.0], [1.0, 2.0]])
    b = jnp.array([1.0, -1.0])

    def total(theta: float) -> jax.Array:
        x = argmin(
            lambda x, th: 0.5 * x @ a_mat @ x - th * (b @ x),
            jnp.zeros(2),
            theta,
        )
        return jnp.sum(x)

    g = float(jax.grad(total)(2.0))
    fd = float((total(2.0 + 1e-3) - total(2.0 - 1e-3)) / 2e-3)
    assert g == pytest.approx(fd, rel=1e-4)


def test_argmin_equality_constrained_gradient() -> None:
    # min x^2 + y^2 s.t. x + y = theta  =>  x*=y*=theta/2, sum = theta.
    def total(theta: jax.Array) -> jax.Array:
        v = argmin(
            lambda v, _: v[0] ** 2 + v[1] ** 2,
            jnp.array([0.0, 0.0]),
            theta,
            eq_constraints=lambda v, th: jnp.array([v[0] + v[1] - th]),
        )
        return jnp.sum(v)

    assert float(jax.grad(total)(1.0)) == pytest.approx(1.0, abs=1e-3)


def test_argmin_bound_active_set_switches_sensitivity() -> None:
    # min (x - theta)^2 s.t. x <= 1.
    #   theta < 1: x* = theta (free)   => dx*/dtheta = 1
    #   theta > 1: x* = 1     (active) => dx*/dtheta = 0
    def solve(theta: jax.Array) -> jax.Array:
        return argmin(
            lambda x, th: (x[0] - th) ** 2,
            jnp.array([0.0]),
            theta,
            bounds=(jnp.array([-10.0]), jnp.array([1.0])),
        )[0]

    assert float(jax.grad(solve)(0.5)) == pytest.approx(1.0, abs=1e-3)
    assert float(jax.grad(solve)(2.0)) == pytest.approx(0.0, abs=1e-3)


def test_argmin_is_jittable() -> None:
    f = jax.jit(lambda theta: argmin(lambda x, th: jnp.sum((x - th) ** 2), jnp.zeros(2), theta))
    assert jnp.allclose(f(jnp.array([1.0, 2.0])), jnp.array([1.0, 2.0]), atol=1e-5)
