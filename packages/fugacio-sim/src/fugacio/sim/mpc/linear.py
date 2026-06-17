"""Linear and offset-free model predictive control.

Model predictive control replaces a fixed feedback law with an *optimization*
solved afresh every sample: predict the plant's response over a finite horizon,
choose the input sequence that minimizes a tracking-plus-effort cost subject to
the actuator and safety constraints, apply the first move, and repeat. It is the
workhorse of modern advanced process control precisely because it handles
multivariable interaction and hard constraints in one shot, which a PID loop
cannot.

This module builds the *condensed* QP (the optimizer decides the input sequence;
the states are eliminated by the prediction equations) so each step is a single
call to the differentiable `fugacio.sim.mpc.solve_qp`. Three pieces make it
production-grade rather than a textbook regulator:

* **A stabilizing terminal cost.** The terminal weight defaults to the discrete
  LQR cost-to-go (`fugacio.sim.mpc.dare`), so an *unconstrained* horizon-one
  controller reproduces the infinite-horizon LQR law exactly: the finite horizon
  inherits the LQR's nominal stability.
* **Offset-free tracking.** An augmented output-disturbance model is estimated by
  a steady-state Kalman filter and a steady-state target ``(x_ss, u_ss)`` is
  recomputed each step, so the controlled outputs reach their setpoints with *zero
  steady-state error* under unmeasured constant disturbances and plant/model
  mismatch, the property that makes MPC usable on a real plant.
* **Constraints.** Hard input magnitude and rate limits, plus optional *soft*
  output limits (slack-relaxed so the QP is always feasible), are imposed honestly
  inside the optimization.

Because the whole step is the differentiable QP, a gradient of a closed-loop
performance index flows straight through the controller's optimization (see
`fugacio.sim.mpc.tune_mpc`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import expm

from fugacio.sim.control.linearize import StateSpace
from fugacio.sim.mpc.qp import QPSettings, solve_qp
from fugacio.sim.mpc.riccati import dare, kalman_gain

ArrayLike = Array | float


def _to_floats(v: ArrayLike) -> list[float]:
    """Flatten a *static* bound (scalar or concrete array) to a list of Python floats.

    Constraint presence fixes the QP's structure, so it must be decided without a
    traced value: this reads the concrete bound (a scalar or array supplied by the
    caller, never a differentiated quantity) with pure Python so the controller can
    be *built inside* a traced function (e.g. gradient-based tuning via ``tune_mpc``).
    """
    data: Any = v.tolist() if hasattr(v, "tolist") else v  # concrete array -> Python floats
    if isinstance(data, (list, tuple)):
        return [x for item in data for x in _to_floats(item)]
    return [float(data)]


def _all_finite(v: ArrayLike) -> bool:
    """Whether *every* entry of a static bound is finite (a plain Python bool)."""
    return all(math.isfinite(x) for x in _to_floats(v))


def _any_finite(v: ArrayLike) -> bool:
    """Whether *any* entry of a static bound is finite (a plain Python bool)."""
    return any(math.isfinite(x) for x in _to_floats(v))


# --------------------------------------------------------------------------- #
# Discretization
# --------------------------------------------------------------------------- #
def c2d(ss: StateSpace, dt: ArrayLike, *, method: str = "zoh") -> StateSpace:
    """Discretize a continuous `StateSpace` at sample time ``dt``.

    Zero-order-hold exactly via the matrix exponential of the augmented block
    ``[[A, B], [0, 0]]``: ``expm(.. * dt) = [[Ad, Bd], [0, I]]``. The output map
    ``(C, D)`` is unchanged. Differentiable in ``dt`` and the model matrices.
    """
    if method != "zoh":
        raise ValueError(f"unknown discretization method {method!r}; only 'zoh' is supported")
    a = jnp.asarray(ss.a, dtype=float)
    b = jnp.asarray(ss.b, dtype=float)
    n, m = a.shape[0], b.shape[1]
    upper = jnp.concatenate([a, b], axis=1)
    lower = jnp.zeros((m, n + m))
    block = jnp.concatenate([upper, lower], axis=0)
    phi = expm(block * jnp.asarray(dt, dtype=float))
    ad = phi[:n, :n]
    bd = phi[:n, n:]
    return StateSpace(
        a=ad, b=bd, c=jnp.asarray(ss.c, dtype=float), d=jnp.asarray(ss.d, dtype=float)
    )


# --------------------------------------------------------------------------- #
# Prediction matrices (condensed)
# --------------------------------------------------------------------------- #
class _Prediction(NamedTuple):
    phi: Array  # (Np*n, n)   state propagation
    gamma: Array  # (Np*n, Nc*m) input-to-state
    g_out: Array  # (Np*p, Nc*m) input-to-output
    f_coef: Array  # (Np*p, n)   free-output coefficient (Y_free = f_coef @ x0)
    phi_n: Array  # (n, n)      terminal state propagation A^Np
    gamma_n: Array  # (n, Nc*m)  terminal input-to-state
    d_diff: Array  # (Nc*m, Nc*m) first-difference (move) operator
    e0: Array  # (Nc*m, m)   previous-input selector for the first move


def _prediction(a: Array, b: Array, c: Array, n_pred: int, n_ctrl: int) -> _Prediction:
    """Build the condensed prediction / move-difference matrices for the horizon."""
    n, m = a.shape[0], b.shape[1]
    eye_n = jnp.eye(n)
    powers = [eye_n]
    for _ in range(n_pred):
        powers.append(powers[-1] @ a)

    phi = jnp.concatenate([powers[i] for i in range(1, n_pred + 1)], axis=0)

    rows = []
    for i in range(1, n_pred + 1):
        blocks = []
        for t in range(n_pred):
            blocks.append(powers[i - 1 - t] @ b if t <= i - 1 else jnp.zeros((n, m)))
        rows.append(jnp.concatenate(blocks, axis=1))
    gamma_full = jnp.concatenate(rows, axis=0)  # (Np*n, Np*m)

    # Move blocking: input at time t is move min(t, Nc-1).
    t_hold = jnp.zeros((n_pred * m, n_ctrl * m))
    for t in range(n_pred):
        col = min(t, n_ctrl - 1)
        t_hold = t_hold.at[t * m : (t + 1) * m, col * m : (col + 1) * m].set(jnp.eye(m))
    gamma = gamma_full @ t_hold

    c_blk = jnp.kron(jnp.eye(n_pred), c)
    g_out = c_blk @ gamma
    f_coef = c_blk @ phi

    d_diff = jnp.eye(n_ctrl * m)
    if n_ctrl > 1:
        d_diff = d_diff - jnp.kron(jnp.eye(n_ctrl, k=-1), jnp.eye(m))
    e0 = jnp.concatenate([jnp.eye(m), jnp.zeros(((n_ctrl - 1) * m, m))], axis=0)

    return _Prediction(
        phi=phi,
        gamma=gamma,
        g_out=g_out,
        f_coef=f_coef,
        phi_n=powers[n_pred],
        gamma_n=gamma[-n:, :],
        d_diff=d_diff,
        e0=e0,
    )


# --------------------------------------------------------------------------- #
# Controller state and result
# --------------------------------------------------------------------------- #
class MPCState(NamedTuple):
    """Estimator + memory carried between controller steps.

    Attributes:
        x_hat: Current state estimate ``(n,)`` (the prior for this sample).
        d_hat: Output-disturbance estimate ``(p,)`` (empty when disabled).
        u_prev: Last applied input ``(m,)`` (for the rate penalty/limit).
    """

    x_hat: Array
    d_hat: Array
    u_prev: Array


class MPCResult(NamedTuple):
    """Outcome of a single `LinearMPC.solve`.

    Attributes:
        u: The first input move to apply ``(m,)``.
        u_sequence: The full optimal move sequence ``(Nc, m)``.
        x_target: The steady-state target state ``(n,)``.
        u_target: The steady-state target input ``(m,)``.
        objective: Optimal QP objective value.
        feasible: Whether the QP converged within tolerance.
    """

    u: Array
    u_sequence: Array
    x_target: Array
    u_target: Array
    objective: Array
    feasible: Array


# --------------------------------------------------------------------------- #
# The controller
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LinearMPC:
    """A condensed-QP linear MPC with offset-free tracking and constraints.

    Construct with `linear_mpc`, which discretizes (if needed), computes the
    LQR terminal cost and the disturbance-observer gain, and fills the bounds. The
    controller is discrete-time with sample period `dt`; drive a closed loop
    with `step` (estimate, optimize, advance) or query a one-shot optimal
    move with `solve`.
    """

    a: Array
    b: Array
    c: Array
    q: Array
    r: Array
    s_du: Array
    p_terminal: Array
    u_min: Array
    u_max: Array
    du_max: Array
    y_min: Array
    y_max: Array
    soft_weight: float
    obs_gain: Array
    n_pred: int
    n_ctrl: int
    dt: float
    n_dist: int
    has_rate_limit: bool
    has_output_limit: bool

    # -- dimensions ----------------------------------------------------------
    @property
    def n_state(self) -> int:
        """Number of plant states."""
        return self.a.shape[0]

    @property
    def n_input(self) -> int:
        """Number of manipulated inputs."""
        return self.b.shape[1]

    @property
    def n_output(self) -> int:
        """Number of controlled outputs."""
        return self.c.shape[0]

    # -- offset-free target --------------------------------------------------
    def target(self, r: Array, d_hat: Array) -> tuple[Array, Array]:
        """Steady-state target ``(x_ss, u_ss)`` giving ``C x_ss + d_hat = r`` at steady state.

        Solves ``[[I - A, -B], [C, 0]] [x_ss; u_ss] = [0; r - d_hat]`` (least squares
        when the actuator/output counts differ), the standard MPC target problem.
        """
        n, m = self.n_state, self.n_input
        r = jnp.atleast_1d(jnp.asarray(r, dtype=float))
        top = jnp.concatenate([jnp.eye(n) - self.a, -self.b], axis=1)
        bot = jnp.concatenate([self.c, jnp.zeros((self.n_output, m))], axis=1)
        mat = jnp.concatenate([top, bot], axis=0)
        rhs = jnp.concatenate([jnp.zeros((n,)), r - d_hat])
        sol = jnp.linalg.lstsq(mat, rhs)[0]
        return sol[:n], sol[n:]

    # -- the QP --------------------------------------------------------------
    def solve(
        self,
        x_hat: Array,
        r: Array,
        u_prev: Array,
        d_hat: Array | None = None,
        *,
        settings: QPSettings | None = None,
    ) -> MPCResult:
        """Solve the MPC QP for the current estimate; return the optimal first move.

        Args:
            x_hat: Current state estimate ``(n,)``.
            r: Output setpoint ``(p,)``.
            u_prev: Previously applied input ``(m,)`` (rate reference).
            d_hat: Output-disturbance estimate ``(p,)`` (defaults to zero).
            settings: Optional QP solver settings.
        """
        m, p = self.n_input, self.n_output
        x_hat = jnp.asarray(x_hat, dtype=float)
        u_prev = jnp.asarray(u_prev, dtype=float)
        d_hat = jnp.zeros((p,)) if d_hat is None else jnp.asarray(d_hat, dtype=float)
        x_ss, u_ss = self.target(r, d_hat)

        pred = _prediction(self.a, self.b, self.c, self.n_pred, self.n_ctrl)
        np_, nc = self.n_pred, self.n_ctrl

        # Penalize outputs y_1..y_{Np-1} as stages and x_Np with the terminal weight
        # (the last stage block is dropped so the terminal cost is not double-counted;
        # with the LQR terminal this makes the controller reproduce LQR exactly).
        q_blk = jnp.kron(jnp.eye(np_), self.q).at[-p:, -p:].set(0.0)
        r_blk = jnp.kron(jnp.eye(nc), self.r)
        s_blk = jnp.kron(jnp.eye(nc), self.s_du)

        # Output disturbance shifts the predicted output: Y = f + G U + tile(d).
        y_free = pred.f_coef @ x_hat + jnp.tile(d_hat, np_)
        r_ref = jnp.tile(jnp.atleast_1d(jnp.asarray(r, dtype=float)), np_)
        u_ss_seq = jnp.tile(u_ss, nc)

        g_mat = pred.g_out
        # Hessian / gradient of the (absolute-coordinate) tracking cost in U.
        h_u = (
            g_mat.T @ q_blk @ g_mat
            + r_blk
            + pred.d_diff.T @ s_blk @ pred.d_diff
            + pred.gamma_n.T @ self.p_terminal @ pred.gamma_n
        )
        g_u = (
            g_mat.T @ q_blk @ (y_free - r_ref)
            - r_blk @ u_ss_seq
            - pred.d_diff.T @ s_blk @ (pred.e0 @ u_prev)
            + pred.gamma_n.T @ self.p_terminal @ (pred.phi_n @ x_hat - x_ss)
        )
        h_u = 2.0 * h_u
        g_u = 2.0 * g_u

        u_lb = jnp.tile(self.u_min, nc)
        u_ub = jnp.tile(self.u_max, nc)

        # Inequalities: input-rate limits and (optional) soft output limits.
        # These are *static* (they fix the QP's row/slack structure), set at build time.
        rate_finite = self.has_rate_limit
        out_finite = self.has_output_limit

        ineq_rows: list[Array] = []
        ineq_rhs: list[Array] = []
        du_seq = jnp.tile(self.du_max, nc)
        rate_ref = pred.e0 @ u_prev
        if rate_finite:
            ineq_rows.append(pred.d_diff)
            ineq_rhs.append(rate_ref + du_seq)
            ineq_rows.append(-pred.d_diff)
            ineq_rhs.append(du_seq - rate_ref)

        n_slack = np_ * p if out_finite else 0
        if n_slack:
            y_max_seq = jnp.tile(self.y_max, np_)
            y_min_seq = jnp.tile(self.y_min, np_)
            big = jnp.where(jnp.isfinite(y_max_seq), y_max_seq, 1e12)
            small = jnp.where(jnp.isfinite(y_min_seq), y_min_seq, -1e12)
            # In z = [U; s] coordinates (s >= 0): G U - s <= y_max - f ; -G U - s <= f - y_min.
            ineq_rows = [
                jnp.concatenate([row, jnp.zeros((row.shape[0], n_slack))], axis=1)
                for row in ineq_rows
            ]
            ineq_rows.append(jnp.concatenate([g_mat, -jnp.eye(n_slack)], axis=1))
            ineq_rhs.append(big - y_free)
            ineq_rows.append(jnp.concatenate([-g_mat, -jnp.eye(n_slack)], axis=1))
            ineq_rhs.append(y_free - small)

            h_full = jnp.block(
                [
                    [h_u, jnp.zeros((nc * m, n_slack))],
                    [
                        jnp.zeros((n_slack, nc * m)),
                        2.0 * 1e-3 * self.soft_weight * jnp.eye(n_slack),
                    ],
                ]
            )
            g_full = jnp.concatenate([g_u, self.soft_weight * jnp.ones((n_slack,))])
            z_lb = jnp.concatenate([u_lb, jnp.zeros((n_slack,))])
            z_ub = jnp.concatenate([u_ub, jnp.full((n_slack,), jnp.inf)])
        else:
            h_full, g_full = h_u, g_u
            z_lb, z_ub = u_lb, u_ub

        if ineq_rows:
            g_ineq = jnp.concatenate(ineq_rows, axis=0)
            h_ineq = jnp.concatenate(ineq_rhs)
        else:
            g_ineq = None
            h_ineq = None

        qp = solve_qp(
            h_full,
            g_full,
            g_ineq=g_ineq,
            h_ineq=h_ineq,
            lb=z_lb,
            ub=z_ub,
            settings=settings or QPSettings(),
        )
        u_seq = qp.x[: nc * m].reshape(nc, m)
        return MPCResult(
            u=u_seq[0],
            u_sequence=u_seq,
            x_target=x_ss,
            u_target=u_ss,
            objective=qp.obj,
            feasible=qp.converged,
        )

    # -- closed-loop estimator + step ---------------------------------------
    def init_state(
        self,
        x0: ArrayLike,
        u0: ArrayLike | None = None,
        d0: ArrayLike | None = None,
    ) -> MPCState:
        """Initial controller state (prior estimate and last input)."""
        n, m = self.n_state, self.n_input
        x_hat = jnp.broadcast_to(jnp.asarray(x0, dtype=float), (n,))
        u_prev = jnp.zeros((m,)) if u0 is None else jnp.broadcast_to(jnp.asarray(u0, float), (m,))
        d_hat = (
            jnp.zeros((self.n_dist,))
            if d0 is None
            else jnp.broadcast_to(jnp.asarray(d0, float), (self.n_dist,))
        )
        return MPCState(x_hat=x_hat, d_hat=d_hat, u_prev=u_prev)

    def estimate(self, state: MPCState, y_meas: Array) -> MPCState:
        """Correct the augmented ``[x; d]`` estimate from a new measurement (Kalman update)."""
        y_meas = jnp.atleast_1d(jnp.asarray(y_meas, dtype=float))
        if self.n_dist == 0:
            innovation = y_meas - self.c @ state.x_hat
            x_hat = state.x_hat + self.obs_gain @ innovation
            return MPCState(x_hat=x_hat, d_hat=state.d_hat, u_prev=state.u_prev)
        aug = jnp.concatenate([state.x_hat, state.d_hat])
        y_pred = self.c @ state.x_hat + state.d_hat
        aug_new = aug + self.obs_gain @ (y_meas - y_pred)
        n = self.n_state
        return MPCState(x_hat=aug_new[:n], d_hat=aug_new[n:], u_prev=state.u_prev)

    def step(
        self,
        state: MPCState,
        y_meas: Array,
        r: Array,
        *,
        settings: QPSettings | None = None,
    ) -> tuple[Array, MPCState]:
        """One discrete controller iteration: estimate, optimize, advance the prior.

        Mirrors `fugacio.sim.control.PID.step`: returns ``(u, new_state)``
        where ``new_state`` carries the *predicted* prior estimate for the next
        sample. Drop into a closed-loop simulation (see
        `fugacio.sim.mpc.simulate_closed_loop`).
        """
        corrected = self.estimate(state, y_meas)
        res = self.solve(corrected.x_hat, r, corrected.u_prev, corrected.d_hat, settings=settings)
        u = res.u
        x_next = self.a @ corrected.x_hat + self.b @ u
        return u, MPCState(x_hat=x_next, d_hat=corrected.d_hat, u_prev=u)


def _augmented_observer_gain(
    a: Array,
    c: Array,
    n_dist: int,
    proc_noise: float,
    dist_noise: float,
    meas_noise: float,
) -> Array:
    """Steady-state Kalman gain for the output-disturbance-augmented model."""
    n = a.shape[0]
    p = c.shape[0]
    if n_dist == 0:
        qn = proc_noise * jnp.eye(n)
        rn = meas_noise * jnp.eye(p)
        gain, _ = kalman_gain(a, c, qn, rn)
        return gain
    a_aug = jnp.block([[a, jnp.zeros((n, n_dist))], [jnp.zeros((n_dist, n)), jnp.eye(n_dist)]])
    c_aug = jnp.concatenate([c, jnp.eye(p, n_dist)], axis=1)
    qn = jnp.block(
        [
            [proc_noise * jnp.eye(n), jnp.zeros((n, n_dist))],
            [jnp.zeros((n_dist, n)), dist_noise * jnp.eye(n_dist)],
        ]
    )
    rn = meas_noise * jnp.eye(p)
    gain, _ = kalman_gain(a_aug, c_aug, qn, rn)
    return gain


def linear_mpc(
    model: StateSpace,
    *,
    q: ArrayLike,
    r: ArrayLike,
    horizon: int,
    control_horizon: int | None = None,
    s_du: ArrayLike = 0.0,
    dt: ArrayLike | None = None,
    discretize: bool = False,
    u_min: ArrayLike = -jnp.inf,
    u_max: ArrayLike = jnp.inf,
    du_max: ArrayLike = jnp.inf,
    y_min: ArrayLike = -jnp.inf,
    y_max: ArrayLike = jnp.inf,
    soft_weight: float = 1e4,
    terminal: str = "lqr",
    disturbance: str = "output",
    proc_noise: float = 1e-2,
    dist_noise: float = 1.0,
    meas_noise: float = 1e-2,
) -> LinearMPC:
    """Assemble a `LinearMPC` from a state-space model and weights.

    Args:
        model: The plant `StateSpace`. Discrete by default; pass
            ``discretize=True`` with ``dt`` to ZOH-discretize a continuous model.
        q: Output tracking weight ``(p, p)`` (scalar/diagonal broadcast).
        r: Input weight ``(m, m)``.
        horizon: Prediction horizon ``Np`` (steps).
        control_horizon: Move (control) horizon ``Nc <= Np`` (defaults to ``Np``).
        s_du: Input-move (rate) weight ``(m, m)`` (default 0).
        dt: Sample time; required if ``discretize`` (and recorded on the controller).
        discretize: Whether ``model`` is continuous and must be ZOH-discretized.
        u_min: Lower input-magnitude limit (vector).
        u_max: Upper input-magnitude limit (vector).
        du_max: Per-step input-rate limit (vector).
        y_min: Lower output limit, imposed as a *soft* (slack) constraint.
        y_max: Upper output limit, imposed as a *soft* (slack) constraint.
        soft_weight: Linear penalty on output-constraint slacks.
        terminal: Terminal cost, ``"lqr"`` (DARE cost-to-go, stabilizing) or
            ``"none"`` (use ``q`` mapped to states).
        disturbance: ``"output"`` for offset-free output-disturbance tracking or
            ``"none"`` to disable the disturbance model.
        proc_noise: Process-noise weight for the observer Kalman gain.
        dist_noise: Disturbance-state noise weight for the observer Kalman gain.
        meas_noise: Measurement-noise weight for the observer Kalman gain.

    Returns:
        A ready `LinearMPC`.
    """
    if discretize:
        if dt is None:
            raise ValueError("discretize=True requires dt")
        model = c2d(model, dt)
    a = jnp.asarray(model.a, dtype=float)
    b = jnp.asarray(model.b, dtype=float)
    c = jnp.asarray(model.c, dtype=float)
    n, m, p = a.shape[0], b.shape[1], c.shape[0]
    nc = horizon if control_horizon is None else int(control_horizon)
    if nc > horizon:
        raise ValueError("control_horizon must be <= horizon")

    def _mat(v: ArrayLike, dim: int) -> Array:
        arr = jnp.asarray(v, dtype=float)
        if arr.ndim == 0:
            return arr * jnp.eye(dim)
        if arr.ndim == 1:
            return jnp.diag(arr)
        return arr

    q_mat = _mat(q, p)
    r_mat = _mat(r, m)
    s_mat = _mat(s_du, m)

    if terminal == "lqr":
        # State-space tracking weight is C^T Q C; terminal cost is its DARE cost-to-go.
        q_state = c.T @ q_mat @ c
        p_term = dare(a, b, q_state + 1e-9 * jnp.eye(n), r_mat)
    elif terminal == "none":
        p_term = c.T @ q_mat @ c
    else:
        raise ValueError(f"unknown terminal {terminal!r}; use 'lqr' or 'none'")

    n_dist = p if disturbance == "output" else 0
    if disturbance not in ("output", "none"):
        raise ValueError(f"unknown disturbance {disturbance!r}; use 'output' or 'none'")
    obs_gain = _augmented_observer_gain(a, c, n_dist, proc_noise, dist_noise, meas_noise)

    def _vec(v: ArrayLike, dim: int) -> Array:
        return jnp.broadcast_to(jnp.asarray(v, dtype=float), (dim,))

    du_vec = _vec(du_max, m)
    y_min_vec = _vec(y_min, p)
    y_max_vec = _vec(y_max, p)
    has_rate = _all_finite(du_max)
    has_output = _any_finite(y_min) or _any_finite(y_max)

    return LinearMPC(
        a=a,
        b=b,
        c=c,
        q=q_mat,
        r=r_mat,
        s_du=s_mat,
        p_terminal=p_term,
        u_min=_vec(u_min, m),
        u_max=_vec(u_max, m),
        du_max=du_vec,
        y_min=y_min_vec,
        y_max=y_max_vec,
        soft_weight=float(soft_weight),
        obs_gain=obs_gain,
        n_pred=int(horizon),
        n_ctrl=nc,
        dt=float(dt) if dt is not None else 1.0,
        n_dist=n_dist,
        has_rate_limit=has_rate,
        has_output_limit=has_output,
    )


__all__ = [
    "LinearMPC",
    "MPCResult",
    "MPCState",
    "c2d",
    "linear_mpc",
]
