"""Advanced process control: Riccati/LQR, the QP, MPC, estimators, and tuning.

These are *self-contained* correctness tests (no external reference): the Riccati
solvers are checked by driving their own residual to zero and by the closed-loop
spectral radius / eigenvalue placement they imply; the QP against closed-form
minimizers and a finite-difference gradient; linear MPC by its exact equivalence
to the LQR (unconstrained) and by offset-free tracking and honoured constraints in
closed loop; the estimators by error reduction and the Kalman/MHE equivalence on a
linear-Gaussian model; and nonlinear MPC by regulating a pendulum. Differential
tests against SciPy's Riccati solvers live in ``test_mpc_oracles.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fugacio.sim.control.linearize import StateSpace
from fugacio.sim.mpc import (
    ClosedLoop,
    ExtendedKalmanFilter,
    GaussianState,
    KalmanFilter,
    UnscentedKalmanFilter,
    c2d,
    care,
    closed_loop_cost,
    constant_setpoint,
    dare,
    discretize,
    dlqr,
    kalman_gain,
    linear_feedback,
    linear_mpc,
    lqr,
    moving_horizon_estimate,
    nonlinear_feedback,
    nonlinear_mpc,
    quadratic_tracking,
    riccati_residual_continuous,
    riccati_residual_discrete,
    simulate_closed_loop,
    solve_qp,
    tune_mpc,
)


def _scalar(v: jnp.ndarray) -> float:
    return float(jnp.ravel(v)[0])


# --------------------------------------------------------------------------- #
# Riccati equations, LQR, Kalman gain
# --------------------------------------------------------------------------- #
def test_dare_residual_is_zero() -> None:
    a = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    b = jnp.array([[0.0], [0.1]])
    q = jnp.diag(jnp.array([2.0, 1.0]))
    r = jnp.array([[0.5]])
    x = dare(a, b, q, r)
    res = riccati_residual_discrete(a, b, q, r, x)
    assert float(jnp.max(jnp.abs(res))) < 1e-8
    assert float(jnp.max(jnp.abs(x - x.T))) < 1e-10  # symmetric


def test_dlqr_stabilizes_closed_loop() -> None:
    a = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    b = jnp.array([[0.0], [0.1]])
    k, _ = dlqr(a, b, jnp.eye(2), jnp.array([[1.0]]))
    eig = jnp.linalg.eigvals(a - b @ k)
    assert float(jnp.max(jnp.abs(eig))) < 1.0  # inside the unit disk


def test_care_residual_is_zero_and_continuous_lqr_stable() -> None:
    a = jnp.array([[0.0, 1.0], [-1.0, -0.5]])
    b = jnp.array([[0.0], [1.0]])
    q = jnp.diag(jnp.array([3.0, 1.0]))
    r = jnp.array([[0.7]])
    x = care(a, b, q, r)
    res = riccati_residual_continuous(a, b, q, r, x)
    assert float(jnp.max(jnp.abs(res))) < 1e-7
    k, _ = lqr(a, b, q, r)
    eig = jnp.linalg.eigvals(a - b @ k)
    assert float(jnp.max(jnp.real(eig))) < 0.0  # left half plane


def test_kalman_gain_solves_filter_riccati() -> None:
    a = jnp.array([[1.0, 0.1], [0.0, 0.95]])
    c = jnp.array([[1.0, 0.0]])
    qn = jnp.diag(jnp.array([1e-2, 1e-2]))
    rn = jnp.array([[0.1]])
    gain, p = kalman_gain(a, c, qn, rn)
    # P solves the *dual* DARE (A^T, C^T); reuse the discrete residual on the dual.
    res = riccati_residual_discrete(a.T, c.T, qn, rn, p)
    assert float(jnp.max(jnp.abs(res))) < 1e-8
    assert gain.shape == (2, 1)


def test_riccati_is_differentiable() -> None:
    a = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    b = jnp.array([[0.0], [0.1]])

    def cost(qscale: jnp.ndarray) -> jnp.ndarray:
        x = dare(a, b, qscale * jnp.eye(2), jnp.array([[1.0]]))
        return jnp.trace(x)

    g = jax.grad(cost)(jnp.array(2.0))
    assert bool(jnp.isfinite(g)) and float(g) > 0.0  # more state penalty -> larger cost-to-go


# --------------------------------------------------------------------------- #
# Quadratic program
# --------------------------------------------------------------------------- #
def test_qp_equality_constraint_closed_form() -> None:
    n = 4
    # min 0.5||x||^2 s.t. sum(x) = 1  ->  x = 1/n.
    sol = solve_qp(jnp.eye(n), jnp.zeros(n), a_eq=jnp.ones((1, n)), b_eq=jnp.array([1.0]))
    assert np.allclose(np.asarray(sol.x), 1.0 / n, atol=1e-6)
    assert bool(sol.converged)


def test_qp_box_clips_unconstrained_minimizer() -> None:
    # min 0.5 x^2 - 3 x on [-1, 1]  ->  x = 1 (clipped from 3).
    sol = solve_qp(jnp.array([[1.0]]), jnp.array([-3.0]), lb=-1.0, ub=1.0)
    assert _scalar(sol.x) == pytest.approx(1.0, abs=1e-6)


def test_qp_inequality_kkt() -> None:
    # min 0.5||x||^2 s.t. x >= 1 (as -x <= -1)  ->  x = 1.
    sol = solve_qp(jnp.eye(2), jnp.zeros(2), g_ineq=-jnp.eye(2), h_ineq=-jnp.ones(2))
    assert np.allclose(np.asarray(sol.x), 1.0, atol=1e-6)
    assert float(sol.primal_residual) < 1e-6


def test_qp_gradient_matches_finite_difference() -> None:
    p = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    g_ineq = jnp.array([[1.0, 1.0]])
    h_ineq = jnp.array([0.5])

    def x0_of_q(q: jnp.ndarray) -> jnp.ndarray:
        return solve_qp(p, q, g_ineq=g_ineq, h_ineq=h_ineq, lb=-2.0, ub=2.0).x[0]

    q = jnp.array([-1.0, -1.0])
    grad = jax.grad(x0_of_q)(q)
    eps = 1e-6
    fd = np.array(
        [
            (_scalar(x0_of_q(q.at[i].add(eps))) - _scalar(x0_of_q(q.at[i].add(-eps)))) / (2 * eps)
            for i in range(2)
        ]
    )
    assert np.allclose(np.asarray(grad), fd, atol=1e-4)


def test_qp_extreme_objective_stays_feasible() -> None:
    # Regression: a degenerate active set must not let the polish blow past the box.
    n = 6
    p = jnp.eye(n)
    q = jnp.full((n,), -1e3)  # unconstrained minimizer far outside the box
    # Rate-style coupling: consecutive entries differ by at most 0.4.
    rows = [(jnp.eye(n) - jnp.eye(n, k=-1)), -(jnp.eye(n) - jnp.eye(n, k=-1))]
    g_ineq = jnp.concatenate(rows, axis=0)
    h_ineq = jnp.full((2 * n,), 0.4)
    sol = solve_qp(p, q, g_ineq=g_ineq, h_ineq=h_ineq, lb=-1.0, ub=1.0)
    assert float(jnp.max(jnp.abs(sol.x))) <= 1.0 + 1e-4  # box respected
    assert float(sol.primal_residual) < 1e-3  # rate respected


# --------------------------------------------------------------------------- #
# Linear MPC
# --------------------------------------------------------------------------- #
def test_c2d_scalar_zoh() -> None:
    a, b, dt = -0.5, 2.0, 0.3
    ss = StateSpace(
        a=jnp.array([[a]]), b=jnp.array([[b]]), c=jnp.array([[1.0]]), d=jnp.zeros((1, 1))
    )
    dss = c2d(ss, dt)
    assert _scalar(dss.a) == pytest.approx(np.exp(a * dt), abs=1e-9)
    assert _scalar(dss.b) == pytest.approx((np.exp(a * dt) - 1.0) / a * b, abs=1e-9)


def test_unconstrained_mpc_reproduces_lqr() -> None:
    a = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    b = jnp.array([[0.0], [0.1]])
    c = jnp.eye(2)
    k, _ = dlqr(a, b, jnp.diag(jnp.array([5.0, 1.0])), jnp.array([[0.5]]))
    mpc = linear_mpc(
        StateSpace(a=a, b=b, c=c, d=jnp.zeros((2, 2))),
        q=jnp.array([5.0, 1.0]),
        r=0.5,
        horizon=50,
        terminal="lqr",
        disturbance="none",
    )
    for x in (jnp.array([1.0, -0.5]), jnp.array([-0.3, 0.8])):
        u = mpc.solve(x, jnp.zeros(2), jnp.zeros(1)).u
        assert _scalar(u) == pytest.approx(_scalar(-(k @ x)), abs=1e-3)


def test_mpc_offset_free_under_disturbance() -> None:
    a = jnp.array([[0.9, 0.1], [0.0, 0.85]])
    b = jnp.array([[0.0], [0.5]])
    c = jnp.array([[1.0, 0.0]])
    mpc = linear_mpc(
        StateSpace(a=a, b=b, c=c, d=jnp.zeros((1, 1))),
        q=10.0,
        r=0.1,
        horizon=15,
        control_horizon=5,
        disturbance="output",
    )
    step = jax.jit(mpc.step)
    state = mpc.init_state(jnp.zeros(2))
    x = jnp.zeros(2)
    dist = 0.3  # constant unmeasured output disturbance
    r = jnp.array([1.0])
    y = c @ x + dist
    for _ in range(80):
        u, state = step(state, c @ x + dist, r)
        x = a @ x + b @ u
        y = c @ x + dist
    assert _scalar(y) == pytest.approx(1.0, abs=1e-3)  # zero steady-state offset


def test_mpc_respects_input_box_and_rate() -> None:
    a = jnp.array([[0.9, 0.1], [0.0, 0.85]])
    b = jnp.array([[0.0], [0.5]])
    c = jnp.array([[1.0, 0.0]])
    mpc = linear_mpc(
        StateSpace(a=a, b=b, c=c, d=jnp.zeros((1, 1))),
        q=10.0,
        r=0.1,
        horizon=20,
        control_horizon=8,
        u_min=-2.0,
        u_max=2.0,
        du_max=0.5,
    )
    step = jax.jit(mpc.step)
    state = mpc.init_state(jnp.zeros(2))
    x = jnp.zeros(2)
    us = []
    for _ in range(12):
        u, state = step(state, c @ x, jnp.array([100.0]))  # unreachable setpoint
        x = a @ x + b @ u
        us.append(_scalar(u))
    assert max(abs(v) for v in us) <= 2.0 + 1e-3  # box respected despite saturation
    rates = [abs(us[i] - (us[i - 1] if i else 0.0)) for i in range(len(us))]
    assert max(rates) <= 0.5 + 1e-3  # rate respected


# --------------------------------------------------------------------------- #
# State estimation
# --------------------------------------------------------------------------- #
def test_kalman_filter_reduces_error() -> None:
    a = jnp.array([[1.0, 0.1], [0.0, 0.95]])
    c = jnp.array([[1.0, 0.0]])
    kf = KalmanFilter(a=a, b=jnp.zeros((2, 1)), c=c, q=1e-3 * jnp.eye(2), r=jnp.array([[0.05]]))
    key = jax.random.PRNGKey(0)
    x = jnp.array([1.0, 0.5])
    state = GaussianState(mean=jnp.zeros(2), cov=jnp.eye(2))
    errs0, errs = [], []
    for _ in range(60):
        key, ks = jax.random.split(key)
        x = a @ x
        y = c @ x + 0.2 * jax.random.normal(ks, (1,))
        errs0.append(float(jnp.abs(state.mean[0] - x[0])))  # prior error before update
        state = kf.step(state, jnp.zeros(1), y)
        errs.append(float(jnp.abs(state.mean[0] - x[0])))
    # The filtered estimate tracks the true state to better than the noise level.
    assert float(np.mean(errs[20:])) < 0.1


def test_kalman_covariance_converges_to_steady_state() -> None:
    a = jnp.array([[1.0, 0.1], [0.0, 0.95]])
    c = jnp.array([[1.0, 0.0]])
    kf = KalmanFilter(a=a, b=jnp.zeros((2, 1)), c=c, q=1e-2 * jnp.eye(2), r=jnp.array([[0.1]]))
    state = GaussianState(mean=jnp.zeros(2), cov=jnp.eye(2))
    for _ in range(200):
        state = kf.step(state, jnp.zeros(1), jnp.zeros(1))
    # Posterior gain implied by the converged covariance equals the steady-state gain.
    p_prior = a @ state.cov @ a.T + 1e-2 * jnp.eye(2)
    s = c @ p_prior @ c.T + jnp.array([[0.1]])
    gain_converged = jnp.linalg.solve(s.T, (p_prior @ c.T).T).T
    assert np.allclose(np.asarray(gain_converged), np.asarray(kf.steady_state_gain()), atol=1e-5)


def _pendulum_data(seed: int) -> tuple:
    dt = 0.05

    def rhs(t: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray, th: None) -> jnp.ndarray:
        return jnp.array([x[1], -9.81 * jnp.sin(x[0]) - 0.2 * x[1]])

    trans = discretize(rhs, dt, method="rk4", substeps=2)

    def f(x, u):
        return trans(x, u, None)

    def h(x):
        return x[:1]  # measure the angle only

    key = jax.random.PRNGKey(seed)
    x = jnp.array([1.0, 0.0])
    xs, ys = [], []
    for _ in range(50):
        key, ks = jax.random.split(key)
        x = f(x, jnp.zeros(1))
        xs.append(x)
        ys.append(h(x) + 0.05 * jax.random.normal(ks, (1,)))
    return f, h, jnp.stack(xs), jnp.stack(ys)


def test_ekf_and_ukf_track_nonlinear_pendulum() -> None:
    f, h, xs, ys = _pendulum_data(1)
    q = 1e-4 * jnp.eye(2)
    r = jnp.array([[0.05**2]])
    state0 = GaussianState(mean=jnp.array([0.6, 0.0]), cov=0.5 * jnp.eye(2))
    us = jnp.zeros((ys.shape[0], 1))
    for filt in (
        ExtendedKalmanFilter(f=f, h=h, q=q, r=r),
        UnscentedKalmanFilter(f=f, h=h, q=q, r=r),
    ):
        traj = filt.filter(state0, us, ys)
        err = jnp.mean(jnp.abs(traj.mean[10:] - xs[10:]))
        assert float(err) < 0.15


def test_mhe_matches_kalman_on_linear_gaussian() -> None:
    a = jnp.array([[1.0, 0.1], [0.0, 0.9]])
    b = jnp.zeros((2, 1))
    c = jnp.array([[1.0, 0.0]])
    q = 1e-2 * jnp.eye(2)
    r = jnp.array([[0.1]])
    p0 = 0.5 * jnp.eye(2)

    def f(x, u):
        return a @ x

    def h(x):
        return c @ x

    key = jax.random.PRNGKey(7)
    x = jnp.array([0.8, -0.3])
    ys = []
    for _ in range(6):
        key, ks = jax.random.split(key)
        x = a @ x
        ys.append(c @ x + 0.1 * jax.random.normal(ks, (1,)))
    ys = jnp.stack(ys)
    us = jnp.zeros((ys.shape[0] - 1, 1))
    x_prior = jnp.array([0.7, -0.2])

    mhe = moving_horizon_estimate(f, h, us, ys, x_prior, q=q, r=r, p0=p0, max_iter=200)

    # Kalman equivalent: prior is the arrival belief; update at x0, then predict/update.
    kf = KalmanFilter(a=a, b=b, c=c, q=q, r=r)
    state = kf.update(GaussianState(mean=x_prior, cov=p0), ys[0])
    for k in range(1, ys.shape[0]):
        state = kf.step(state, us[k - 1], ys[k])
    assert np.allclose(np.asarray(mhe.x), np.asarray(state.mean), atol=1e-5)


# --------------------------------------------------------------------------- #
# Nonlinear MPC
# --------------------------------------------------------------------------- #
def test_nmpc_tracks_scalar_nonlinear_system() -> None:
    def rhs(t, x, u, th):
        return jnp.array([-(x[0] ** 3) + u[0]])

    trans = discretize(rhs, 0.2, method="rk4", substeps=4)
    stage, term = quadratic_tracking(q=20.0, r=0.01)
    mpc = nonlinear_mpc(
        trans, stage_cost=stage, terminal_cost=term, horizon=8, n_input=1, u_min=-5.0, u_max=5.0
    )
    step = jax.jit(lambda x, u, w, th: mpc.step(x, u, w, th))
    x, u, warm = jnp.array([0.0]), jnp.zeros(1), jnp.zeros((8, 1))
    theta = {"r": jnp.array([1.0])}
    for _ in range(30):
        u, warm = step(x, u, warm, theta)
        x = trans(x, u, None)
    assert _scalar(x) == pytest.approx(1.0, abs=0.1)


def test_nmpc_stabilizes_pendulum_with_rate_limit() -> None:
    dt = 0.05

    def prhs(t, x, u, th):
        return jnp.array([x[1], -9.81 * jnp.sin(x[0]) - 0.3 * x[1] + u[0]])

    trans = discretize(prhs, dt, method="rk4", substeps=2)
    stage, term = quadratic_tracking(q=jnp.array([10.0, 1.0]), r=0.01)
    mpc = nonlinear_mpc(
        trans,
        stage_cost=stage,
        terminal_cost=term,
        horizon=12,
        control_horizon=6,
        n_input=1,
        u_min=-20.0,
        u_max=20.0,
        du_max=8.0,
        max_iter=30,
    )
    step = jax.jit(lambda x, u, w, th: mpc.step(x, u, w, th))
    x, u, warm = jnp.array([0.8, 0.0]), jnp.zeros(1), jnp.zeros((6, 1))
    theta = {"r": jnp.array([0.0, 0.0])}
    for _ in range(60):
        u, warm = step(x, u, warm, theta)
        x = trans(x, u, None)
    assert abs(_scalar(x[0])) < 0.05  # driven upright/down to rest


# --------------------------------------------------------------------------- #
# Closed-loop harness and tuning
# --------------------------------------------------------------------------- #
def _linear_loop_builder():
    a = jnp.array([[0.9, 0.1], [0.0, 0.85]])
    b = jnp.array([[0.0], [0.5]])
    c = jnp.array([[1.0, 0.0]])

    def plant(x, u):
        return a @ x + b @ u

    def meas(x):
        return c @ x

    def build(params: dict) -> ClosedLoop:
        mpc = linear_mpc(
            StateSpace(a=a, b=b, c=c, d=jnp.zeros((1, 1))),
            q=params["q"],
            r=params["r"],
            horizon=15,
            control_horizon=5,
            u_min=-3.0,
            u_max=3.0,
            disturbance="output",
        )
        step, c0 = linear_feedback(mpc, jnp.zeros(2))
        return simulate_closed_loop(
            plant, step, jnp.zeros(2), c0, constant_setpoint(1.0, 40), measure=meas
        )

    return build


def test_closed_loop_shapes_and_tracking() -> None:
    build = _linear_loop_builder()
    loop = jax.jit(build)({"q": 10.0, "r": 0.5})
    assert loop.states.shape == (41, 2)
    assert loop.outputs.shape == (41, 1)
    assert loop.inputs.shape == (40, 1)
    assert loop.setpoints.shape == (40, 1)
    assert _scalar(loop.outputs[-1]) == pytest.approx(1.0, abs=1e-3)


def test_tune_mpc_reduces_closed_loop_cost() -> None:
    build = _linear_loop_builder()

    def sim_from_log(lp: jnp.ndarray) -> ClosedLoop:
        return build({"q": jnp.exp(lp[0]), "r": jnp.exp(lp[1])})

    def perf(loop: ClosedLoop) -> jnp.ndarray:
        return closed_loop_cost(loop, error_weight=1.0, effort_weight=0.05)

    start = jnp.array([jnp.log(5.0), jnp.log(1.0)])
    grad = jax.grad(lambda lp: perf(sim_from_log(lp)))(start)
    assert bool(jnp.all(jnp.isfinite(grad)))
    res = tune_mpc(sim_from_log, start, performance=perf, method="bfgs", max_iter=25)
    assert float(perf(sim_from_log(res.x))) <= float(perf(sim_from_log(start))) + 1e-9


def test_nonlinear_feedback_closed_loop() -> None:
    def rhs(t, x, u, th):
        return jnp.array([-(x[0] ** 3) + u[0]])

    trans = discretize(rhs, 0.2, method="rk4", substeps=4)
    stage, term = quadratic_tracking(q=20.0, r=0.01)
    mpc = nonlinear_mpc(
        trans, stage_cost=stage, terminal_cost=term, horizon=8, n_input=1, u_min=-5.0, u_max=5.0
    )

    def plant(x, u):
        return trans(x, u, None)

    step, c0 = nonlinear_feedback(mpc)
    loop = simulate_closed_loop(plant, step, jnp.array([0.0]), c0, constant_setpoint(1.0, 25))
    assert _scalar(loop.outputs[-1]) == pytest.approx(1.0, abs=0.1)
