"""Differential tests: Fugacio's integrators vs SciPy's ``solve_ivp``.

Opt-in *oracle* tests (marker: ``oracle``) that pit the in-house integrators
against an independent, battle-tested ODE suite on problems with no closed form: a
Van der Pol oscillator, a consecutive-reaction kinetics chain, and a stiff linear
system. SciPy ships as a JAX dependency, so the reference is always importable; the
agreement isolates the time-stepping itself (same right-hand side, same tolerances)
rather than any modelling choice.
"""

from __future__ import annotations

from importlib.util import find_spec

import jax.numpy as jnp
import numpy as np
import pytest

from fugacio.sim import integrate, odeint

HAVE_SCIPY = find_spec("scipy") is not None

pytestmark = [
    pytest.mark.oracle,
    pytest.mark.skipif(not HAVE_SCIPY, reason="scipy not installed"),
]


def test_odeint_matches_scipy_on_van_der_pol() -> None:
    from scipy.integrate import solve_ivp

    mu = 1.0
    ts = np.linspace(0.0, 20.0, 201)
    ref = solve_ivp(
        lambda t, y: [y[1], mu * (1.0 - y[0] ** 2) * y[1] - y[0]],
        (0.0, 20.0),
        [2.0, 0.0],
        t_eval=ts,
        rtol=1e-11,
        atol=1e-13,
        method="RK45",
    )

    def rhs(t: jnp.ndarray, y: jnp.ndarray, th: None) -> jnp.ndarray:
        return jnp.array([y[1], mu * (1.0 - y[0] ** 2) * y[1] - y[0]])

    traj = odeint(rhs, jnp.array([2.0, 0.0]), jnp.asarray(ts), method="dopri5", substeps=40)
    assert np.allclose(np.asarray(traj), ref.y.T, atol=1e-5, rtol=1e-5)


def test_integrate_matches_scipy_on_consecutive_reactions() -> None:
    from scipy.integrate import solve_ivp

    # A -> B -> C with k1, k2 (classic series kinetics, no simple closed form for B,C peak).
    k1, k2 = 1.3, 0.4

    def f_np(t: float, y: np.ndarray) -> list[float]:
        a, b, _c = y
        return [-k1 * a, k1 * a - k2 * b, k2 * b]

    ref = solve_ivp(f_np, (0.0, 6.0), [1.0, 0.0, 0.0], rtol=1e-12, atol=1e-14, method="LSODA")

    def rhs(t: jnp.ndarray, y: jnp.ndarray, th: None) -> jnp.ndarray:
        a, b, _c = y[0], y[1], y[2]
        return jnp.array([-k1 * a, k1 * a - k2 * b, k2 * b])

    res = integrate(rhs, jnp.array([1.0, 0.0, 0.0]), 0.0, 6.0, rtol=1e-10, atol=1e-12)
    assert np.allclose(np.asarray(res.y), ref.y[:, -1], atol=1e-7)
    # Total moles are conserved by both.
    assert float(jnp.sum(res.y)) == pytest.approx(1.0, abs=1e-9)


def test_implicit_methods_match_scipy_on_stiff_system() -> None:
    from scipy.integrate import solve_ivp

    # Stiff linear system: eigenvalues -1000 and -1 (time-scale ratio 1000).
    mat = np.array([[-1000.0, 1.0], [0.0, -1.0]])
    ts = np.linspace(0.0, 2.0, 41)
    ref = solve_ivp(
        lambda t, y: mat @ y,
        (0.0, 2.0),
        [1.0, 1.0],
        t_eval=ts,
        rtol=1e-10,
        atol=1e-12,
        method="Radau",
    )

    mat_j = jnp.asarray(mat)

    def rhs(t: jnp.ndarray, y: jnp.ndarray, th: None) -> jnp.ndarray:
        return mat_j @ y

    traj = odeint(rhs, jnp.array([1.0, 1.0]), jnp.asarray(ts), method="trapezoidal", substeps=200)
    assert np.allclose(np.asarray(traj), ref.y.T, atol=1e-4, rtol=1e-4)
