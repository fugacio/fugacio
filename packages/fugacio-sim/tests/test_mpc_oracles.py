"""Differential tests: Fugacio's Riccati/LQR/Kalman solvers vs SciPy.

Opt-in *oracle* tests (marker: ``oracle``) that pit the in-house differentiable
Riccati recursions against SciPy's battle-tested ``solve_discrete_are`` /
``solve_continuous_are`` (and the LQR / Kalman gains built from them). SciPy ships
as a JAX dependency, so the reference is always importable; the agreement isolates
the algebra (same model, same weights) rather than any modelling choice.
"""

from __future__ import annotations

from importlib.util import find_spec

import jax.numpy as jnp
import numpy as np
import pytest

from fugacio.sim.mpc import care, dare, dlqr, kalman_gain, lqr

HAVE_SCIPY = find_spec("scipy") is not None

pytestmark = [
    pytest.mark.oracle,
    pytest.mark.skipif(not HAVE_SCIPY, reason="scipy not installed"),
]


def test_dare_matches_scipy() -> None:
    from scipy.linalg import solve_discrete_are

    a = np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.1], [0.0, 0.0, 0.95]])
    b = np.array([[0.0], [0.0], [0.1]])
    q = np.diag([3.0, 2.0, 1.0])
    r = np.array([[0.5]])
    ref = solve_discrete_are(a, b, q, r)
    got = np.asarray(dare(jnp.asarray(a), jnp.asarray(b), jnp.asarray(q), jnp.asarray(r)))
    assert np.allclose(got, ref, atol=1e-7, rtol=1e-7)


def test_dlqr_gain_matches_scipy() -> None:
    from scipy.linalg import solve_discrete_are

    a = np.array([[1.0, 0.1], [0.0, 1.0]])
    b = np.array([[0.0], [0.1]])
    q = np.diag([5.0, 1.0])
    r = np.array([[0.5]])
    x = solve_discrete_are(a, b, q, r)
    k_ref = np.linalg.solve(r + b.T @ x @ b, b.T @ x @ a)
    k_got, _ = dlqr(jnp.asarray(a), jnp.asarray(b), jnp.asarray(q), jnp.asarray(r))
    assert np.allclose(np.asarray(k_got), k_ref, atol=1e-7)


def test_care_matches_scipy() -> None:
    from scipy.linalg import solve_continuous_are

    a = np.array([[0.0, 1.0], [-2.0, -0.3]])
    b = np.array([[0.0], [1.0]])
    q = np.diag([4.0, 1.0])
    r = np.array([[0.6]])
    ref = solve_continuous_are(a, b, q, r)
    got = np.asarray(care(jnp.asarray(a), jnp.asarray(b), jnp.asarray(q), jnp.asarray(r)))
    assert np.allclose(got, ref, atol=1e-6, rtol=1e-6)


def test_continuous_lqr_gain_matches_scipy() -> None:
    from scipy.linalg import solve_continuous_are

    a = np.array([[0.0, 1.0], [-2.0, -0.3]])
    b = np.array([[0.0], [1.0]])
    q = np.diag([4.0, 1.0])
    r = np.array([[0.6]])
    x = solve_continuous_are(a, b, q, r)
    k_ref = np.linalg.solve(r, b.T @ x)
    k_got, _ = lqr(jnp.asarray(a), jnp.asarray(b), jnp.asarray(q), jnp.asarray(r))
    assert np.allclose(np.asarray(k_got), k_ref, atol=1e-6)


def test_kalman_gain_matches_scipy_dual() -> None:
    from scipy.linalg import solve_discrete_are

    a = np.array([[1.0, 0.1], [0.0, 0.95]])
    c = np.array([[1.0, 0.0]])
    qn = np.diag([1e-2, 1e-2])
    rn = np.array([[0.1]])
    # Filtering ARE is the control ARE on the dual pair (A^T, C^T).
    p_ref = solve_discrete_are(a.T, c.T, qn, rn)
    l_ref = p_ref @ c.T @ np.linalg.inv(c @ p_ref @ c.T + rn)
    l_got, p_got = kalman_gain(jnp.asarray(a), jnp.asarray(c), jnp.asarray(qn), jnp.asarray(rn))
    assert np.allclose(np.asarray(p_got), p_ref, atol=1e-7)
    assert np.allclose(np.asarray(l_got), l_ref, atol=1e-7)
