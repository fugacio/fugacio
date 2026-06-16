"""The differentiable ODE integration core: accuracy, order, and gradients.

The integrators are checked the way a numerical-methods text would: against the
closed-form solution of linear test problems, by measuring the empirical order of
convergence under step refinement, by confirming a conserved quantity stays
conserved, and -- the point of the whole exercise -- by checking that gradients
through the solve (both the scan-based :func:`odeint` and the continuous-adjoint
:func:`integrate`) match the analytic sensitivity and finite differences.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import integrate, odeint, odeint_final
from fugacio.sim.dynamics import FIXED_METHODS, ODEResult


def _decay_final(method: str, substeps: int, k: float = 1.0, t_final: float = 2.0) -> float:
    ts = jnp.array([0.0, t_final])
    traj = odeint(
        lambda t, y, th: -th * y, jnp.asarray(1.0), ts, k, method=method, substeps=substeps
    )
    return float(traj[-1])


def _order(method: str, coarse: int, fine: int, k: float = 1.0, t_final: float = 2.0) -> float:
    exact = jnp.exp(-k * t_final)
    e_coarse = abs(_decay_final(method, coarse) - float(exact))
    e_fine = abs(_decay_final(method, fine) - float(exact))
    return float(jnp.log(e_coarse / e_fine) / jnp.log(fine / coarse))


# --------------------------------------------------------------------------- #
# Accuracy and order of convergence
# --------------------------------------------------------------------------- #
def test_all_methods_recover_exponential_decay() -> None:
    exact = float(jnp.exp(-2.0))
    for method in FIXED_METHODS:
        # First-order Euler/implicit-Euler need the loosest band; the rest are far tighter.
        rel = 2e-2 if method in ("euler", "implicit_euler") else 1e-4
        got = _decay_final(method, substeps=200)
        assert got == pytest.approx(exact, rel=rel), method


@pytest.mark.parametrize(
    ("method", "lo", "hi"),
    [
        ("euler", 0.9, 1.15),
        ("rk4", 3.7, 4.3),
        ("implicit_euler", 0.9, 1.15),
        ("trapezoidal", 1.8, 2.2),
    ],
)
def test_empirical_order_of_convergence(method: str, lo: float, hi: float) -> None:
    assert lo <= _order(method, coarse=10, fine=20) <= hi


def test_dopri5_is_high_order_accurate() -> None:
    # Fifth order: a handful of steps already beats RK4 by orders of magnitude.
    exact = float(jnp.exp(-2.0))
    err_dopri5 = abs(_decay_final("dopri5", substeps=8) - exact)
    err_rk4 = abs(_decay_final("rk4", substeps=8) - exact)
    assert err_dopri5 < 1e-6
    assert err_dopri5 < err_rk4


def test_odeint_returns_full_trajectory_on_grid() -> None:
    ts = jnp.linspace(0.0, 3.0, 31)
    traj = odeint(lambda t, y, th: -y, jnp.asarray(1.0), ts, method="rk4", substeps=10)
    assert traj.shape == (31,)
    assert float(traj[0]) == pytest.approx(1.0)
    assert jnp.allclose(traj, jnp.exp(-ts), atol=1e-6)


def test_odeint_handles_pytree_state() -> None:
    ts = jnp.linspace(0.0, 1.0, 11)

    def rhs(t: jnp.ndarray, y: dict[str, jnp.ndarray], th: None) -> dict[str, jnp.ndarray]:
        return {"a": -y["a"], "b": 2.0 * y["b"]}

    traj = odeint(
        rhs, {"a": jnp.asarray(1.0), "b": jnp.asarray(1.0)}, ts, method="rk4", substeps=10
    )
    assert float(traj["a"][-1]) == pytest.approx(float(jnp.exp(-1.0)), rel=1e-6)
    assert float(traj["b"][-1]) == pytest.approx(float(jnp.exp(2.0)), rel=1e-5)


def test_odeint_raises_on_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        odeint(lambda t, y, th: -y, jnp.asarray(1.0), jnp.array([0.0, 1.0]), method="nope")


def test_odeint_raises_on_bad_substeps() -> None:
    with pytest.raises(ValueError, match="substeps"):
        odeint(lambda t, y, th: -y, jnp.asarray(1.0), jnp.array([0.0, 1.0]), substeps=0)


# --------------------------------------------------------------------------- #
# Conservation
# --------------------------------------------------------------------------- #
def test_linear_exchange_conserves_total() -> None:
    # A <-> B first-order exchange: total A + B is invariant.
    def rhs(t: jnp.ndarray, y: jnp.ndarray, th: None) -> jnp.ndarray:
        a, b = y[0], y[1]
        return jnp.array([-0.7 * a + 0.3 * b, 0.7 * a - 0.3 * b])

    ts = jnp.linspace(0.0, 10.0, 101)
    traj = odeint(rhs, jnp.array([1.0, 0.0]), ts, method="rk4", substeps=10)
    totals = jnp.sum(traj, axis=1)
    assert jnp.allclose(totals, 1.0, atol=1e-9)


def test_undamped_oscillator_conserves_energy_with_rk4() -> None:
    def rhs(t: jnp.ndarray, y: jnp.ndarray, th: None) -> jnp.ndarray:
        return jnp.array([y[1], -y[0]])

    ts = jnp.linspace(0.0, 20.0, 51)
    traj = odeint(rhs, jnp.array([1.0, 0.0]), ts, method="rk4", substeps=40)
    energy = 0.5 * (traj[:, 0] ** 2 + traj[:, 1] ** 2)
    assert jnp.max(jnp.abs(energy - energy[0])) < 1e-5


# --------------------------------------------------------------------------- #
# Adaptive integrator
# --------------------------------------------------------------------------- #
def test_integrate_returns_oderesult_and_is_accurate() -> None:
    res = integrate(lambda t, y, th: -y, jnp.asarray(1.0), 0.0, 5.0, rtol=1e-8, atol=1e-10)
    assert isinstance(res, ODEResult)
    assert bool(res.success)
    assert float(res.y) == pytest.approx(float(jnp.exp(-5.0)), rel=1e-6)
    assert int(res.n_accepted) >= 1


def test_integrate_tighter_tolerance_reduces_error() -> None:
    exact = float(jnp.exp(-5.0))
    loose = abs(
        float(integrate(lambda t, y, th: -y, jnp.asarray(1.0), 0.0, 5.0, rtol=1e-3, atol=1e-6).y)
        - exact
    )
    tight = abs(
        float(integrate(lambda t, y, th: -y, jnp.asarray(1.0), 0.0, 5.0, rtol=1e-9, atol=1e-12).y)
        - exact
    )
    assert tight < loose


def test_integrate_solves_nonlinear_logistic() -> None:
    # Logistic dy/dt = y (1 - y); analytic y(t) = y0 e^t / (1 - y0 + y0 e^t).
    y0 = 0.1
    res = integrate(
        lambda t, y, th: y * (1.0 - y), jnp.asarray(y0), 0.0, 4.0, rtol=1e-9, atol=1e-12
    )
    exact = y0 * jnp.exp(4.0) / (1.0 - y0 + y0 * jnp.exp(4.0))
    assert float(res.y) == pytest.approx(float(exact), rel=1e-6)


# --------------------------------------------------------------------------- #
# Gradients: AD through the solve vs analytic vs finite differences
# --------------------------------------------------------------------------- #
def test_odeint_gradient_wrt_parameter_matches_analytic() -> None:
    t_final = 2.0

    def final(k: jnp.ndarray) -> jnp.ndarray:
        return odeint_final(
            lambda t, y, th: -th * y, jnp.asarray(1.0), 0.0, t_final, k, method="rk4", steps=200
        )

    k = 0.8
    grad = float(jax.grad(final)(jnp.asarray(k)))
    analytic = -t_final * float(jnp.exp(-k * t_final))
    fd = float((final(jnp.asarray(k + 1e-5)) - final(jnp.asarray(k - 1e-5))) / 2e-5)
    assert grad == pytest.approx(analytic, rel=1e-5)
    assert grad == pytest.approx(fd, rel=1e-4)


def test_odeint_gradient_wrt_initial_state() -> None:
    t_final = 1.5

    def final(y0: jnp.ndarray) -> jnp.ndarray:
        return odeint_final(lambda t, y, th: -y, y0, 0.0, t_final, method="rk4", steps=100)

    grad = float(jax.grad(final)(jnp.asarray(1.0)))
    assert grad == pytest.approx(float(jnp.exp(-t_final)), rel=1e-6)


def test_integrate_continuous_adjoint_gradient_matches_analytic() -> None:
    t_final = 3.0

    def final(k: jnp.ndarray) -> jnp.ndarray:
        return integrate(
            lambda t, y, th: -th * y, jnp.asarray(1.0), 0.0, t_final, k, rtol=1e-9, atol=1e-12
        ).y

    k = 0.5
    grad = float(jax.grad(final)(jnp.asarray(k)))
    analytic = -t_final * float(jnp.exp(-k * t_final))
    fd = float((final(jnp.asarray(k + 1e-5)) - final(jnp.asarray(k - 1e-5))) / 2e-5)
    assert grad == pytest.approx(analytic, rel=1e-5)
    assert grad == pytest.approx(fd, rel=1e-4)


def test_integrate_adjoint_matches_odeint_reverse_mode() -> None:
    # The two integrators should give the same parameter gradient on the same ODE.
    t_final = 2.0

    def via_adaptive(k: jnp.ndarray) -> jnp.ndarray:
        return integrate(
            lambda t, y, th: -th * y, jnp.asarray(1.0), 0.0, t_final, k, rtol=1e-10, atol=1e-12
        ).y

    def via_scan(k: jnp.ndarray) -> jnp.ndarray:
        return odeint_final(
            lambda t, y, th: -th * y, jnp.asarray(1.0), 0.0, t_final, k, method="dopri5", steps=200
        )

    k = jnp.asarray(0.9)
    assert float(jax.grad(via_adaptive)(k)) == pytest.approx(float(jax.grad(via_scan)(k)), rel=1e-5)
