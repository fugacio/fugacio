"""Dynamic optimization, estimation, and gradient-based controller tuning.

Each routine composes the differentiable integrator with the existing optimizers,
so the tests are end-to-end: optimal control must steer a double integrator to a
target state at minimum effort, estimation must recover a known rate constant from
its own simulated data, and the gradient PID tune must reduce the closed-loop IAE.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from fugacio.sim import iae, odeint, pi
from fugacio.sim.dynamics import estimate_dynamics, optimal_control, tune_pid


# --------------------------------------------------------------------------- #
# Optimal control
# --------------------------------------------------------------------------- #
def test_optimal_control_steers_double_integrator_to_rest() -> None:
    n = 21
    ts = jnp.linspace(0.0, 5.0, n)

    def dynamics(t: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray, theta: None) -> jnp.ndarray:
        accel = u[0] if jnp.ndim(u) else u
        return jnp.array([x[1], accel])

    def stage(t: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray, theta: None) -> jnp.ndarray:
        accel = u[0] if jnp.ndim(u) else u
        return 0.01 * accel**2

    def terminal(xf: jnp.ndarray, theta: None) -> jnp.ndarray:
        return 100.0 * (xf[0] - 1.0) ** 2 + 100.0 * xf[1] ** 2

    res = optimal_control(
        dynamics,
        jnp.array([0.0, 0.0]),
        ts,
        jnp.zeros(n - 1),
        stage,
        terminal_cost=terminal,
        bounds=(-5.0, 5.0),
        max_iter=120,
    )
    xf = res.trajectory[-1]
    assert float(xf[0]) == pytest.approx(1.0, abs=1e-2)  # reaches position target
    assert float(xf[1]) == pytest.approx(0.0, abs=1e-2)  # at rest
    assert res.u.shape == (n - 1,)


# --------------------------------------------------------------------------- #
# Parameter estimation
# --------------------------------------------------------------------------- #
def test_estimate_dynamics_recovers_rate_constant() -> None:
    k_true = 0.7
    ts = jnp.linspace(0.0, 6.0, 60)
    data = jnp.exp(-k_true * ts)  # noise-free first-order decay from y0 = 1

    def model(t: jnp.ndarray, y: jnp.ndarray, theta: dict) -> jnp.ndarray:
        return -theta["k"] * y

    res = estimate_dynamics(model, jnp.asarray(1.0), ts, data, {"k": jnp.asarray(0.2)}, max_iter=60)
    assert float(res.theta["k"]) == pytest.approx(k_true, rel=1e-3)
    assert float(res.cost) < 1e-10


def test_estimate_dynamics_with_observation_map() -> None:
    # Two-state model, only the first state is measured.
    k_true = 0.5
    ts = jnp.linspace(0.0, 8.0, 40)

    def model(t: jnp.ndarray, y: jnp.ndarray, theta: dict) -> jnp.ndarray:
        return jnp.array([-theta["k"] * y[0], theta["k"] * y[0]])

    truth = odeint(
        model, jnp.array([1.0, 0.0]), ts, {"k": jnp.asarray(k_true)}, method="rk4", substeps=4
    )
    data = truth[:, 0]
    res = estimate_dynamics(
        model,
        jnp.array([1.0, 0.0]),
        ts,
        data,
        {"k": jnp.asarray(0.2)},
        observe=lambda traj: traj[:, 0],
        max_iter=60,
    )
    assert float(res.theta["k"]) == pytest.approx(k_true, rel=1e-3)


# --------------------------------------------------------------------------- #
# Gradient-based PID tuning
# --------------------------------------------------------------------------- #
def test_tune_pid_reduces_iae() -> None:
    ts = jnp.linspace(0.0, 40.0, 201)
    kp, taup, sp = 2.0, 5.0, 1.0

    def response(gains: dict) -> jnp.ndarray:
        c = pi(kc=gains["kc"], tau_i=gains["tau_i"], u_min=-10.0, u_max=10.0)

        def rhs(t: jnp.ndarray, st: dict, th: None) -> dict:
            y, ic = st["y"], st["c"]
            u = c.output(ic, sp, y)
            return {"y": (-y + kp * u) / taup, "c": c.derivative(ic, sp, y)}

        st0 = {"y": jnp.asarray(0.0), "c": c.init_state(0.0)}
        return odeint(rhs, st0, ts, method="rk4", substeps=3)["y"]

    g0 = {"kc": jnp.asarray(0.5), "tau_i": jnp.asarray(8.0)}
    iae0 = float(iae(ts, response(g0), sp))
    res = tune_pid(
        response,
        g0,
        setpoint=sp,
        ts=ts,
        bounds=({"kc": 0.05, "tau_i": 0.5}, {"kc": 20.0, "tau_i": 50.0}),
        max_iter=60,
    )
    iae1 = float(iae(ts, response(res.x), sp))
    assert iae1 < iae0
    assert float(res.x["kc"]) > float(g0["kc"])  # tightened up the loop
