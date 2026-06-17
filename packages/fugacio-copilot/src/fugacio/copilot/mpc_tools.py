"""Copilot tools for advanced process control (MPC + state estimation).

These expose :mod:`fugacio.sim.mpc` to the LLM design agent as deterministic,
JSON-in/JSON-out calculations over a *linear state-space* plant ``x+ = A x + B u``,
``y = C x``:

* ``lqr_design`` -- the infinite-horizon LQR gain and closed-loop poles for given
  state/input weights (discrete or continuous);
* ``kalman_design`` -- the steady-state Kalman filter gain, error covariance and
  estimator poles for given process/measurement noise;
* ``simulate_mpc`` -- run a constrained, offset-free linear MPC in closed loop
  against a setpoint (optionally with a constant output disturbance) and report the
  response/input trajectories and step metrics;
* ``tune_mpc_weights`` -- descend the closed-loop tracking cost on the MPC weights
  themselves, exploiting the differentiability of the controller's own QP.

Matrices are passed as nested lists; weights may be a scalar, a per-channel list
(diagonal), or a full matrix. Everything returned is plain Python (floats / lists)
so it serialises trivially for function calling.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from fugacio.sim import (
    StateSpace,
    closed_loop_cost,
    constant_setpoint,
    dlqr,
    kalman_gain,
    linear_feedback,
    linear_mpc,
    lqr,
    simulate_closed_loop,
    step_info,
    tune_mpc,
)

JsonDict = dict[str, Any]


# --------------------------------------------------------------------------- #
# Input parsing
# --------------------------------------------------------------------------- #
def _square(v: Any) -> jnp.ndarray:
    """Coerce a nested-list input to a 2-D float matrix."""
    return jnp.atleast_2d(jnp.asarray(v, dtype=float))


def _b_matrix(b: Any, n: int) -> jnp.ndarray:
    """Coerce ``B`` to ``(n, m)`` (a flat list is read as a single input column)."""
    arr = jnp.asarray(b, dtype=float)
    return arr[:, None] if arr.ndim == 1 else arr


def _c_matrix(c: Any, n: int) -> jnp.ndarray:
    """Coerce ``C`` to ``(p, n)`` (a flat list is read as a single output row)."""
    arr = jnp.asarray(c, dtype=float)
    return arr[None, :] if arr.ndim == 1 else arr


def _weight(v: Any, dim: int) -> jnp.ndarray:
    """Coerce a weight to a ``(dim, dim)`` matrix: scalar -> ``s*I``, vector -> ``diag``."""
    arr = jnp.asarray(v, dtype=float)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return jnp.diag(arr)
    return arr * jnp.eye(dim)


def _state_space(a: Any, b: Any, c: Any) -> StateSpace:
    a_m = _square(a)
    n = a_m.shape[0]
    b_m = _b_matrix(b, n)
    c_m = _c_matrix(c, n)
    return StateSpace(a=a_m, b=b_m, c=c_m, d=jnp.zeros((c_m.shape[0], b_m.shape[1])))


def _poles(matrix: jnp.ndarray, *, continuous: bool) -> tuple[list[float], bool]:
    """Eigenvalue diagnostics: magnitudes (discrete) or real parts (continuous), and stability."""
    eig = jnp.linalg.eigvals(matrix)
    if continuous:
        parts = [float(v) for v in jnp.real(eig)]
        return parts, bool(jnp.max(jnp.real(eig)) < 0.0)
    mags = [float(v) for v in jnp.abs(eig)]
    return mags, bool(jnp.max(jnp.abs(eig)) < 1.0)


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #
def _lqr_design(
    a: Any,
    b: Any,
    q: Any,
    r: Any,
    continuous: bool = False,
) -> JsonDict:
    """Infinite-horizon LQR gain ``u = -K x`` and the closed-loop poles."""
    a_m = _square(a)
    n = a_m.shape[0]
    b_m = _b_matrix(b, n)
    q_m = _weight(q, n)
    r_m = _weight(r, b_m.shape[1])
    if continuous:
        k, x = lqr(a_m, b_m, q_m, r_m)
    else:
        k, x = dlqr(a_m, b_m, q_m, r_m)
    poles, stable = _poles(a_m - b_m @ k, continuous=continuous)
    key = "pole_real_parts" if continuous else "pole_magnitudes"
    return {
        "gain": k.tolist(),
        "cost_to_go": x.tolist(),
        key: poles,
        "stable": stable,
        "continuous": bool(continuous),
    }


def _kalman_design(
    a: Any,
    c: Any,
    process_noise: Any,
    measurement_noise: Any,
) -> JsonDict:
    """Steady-state Kalman gain, prior error covariance and estimator poles."""
    a_m = _square(a)
    n = a_m.shape[0]
    c_m = _c_matrix(c, n)
    p = c_m.shape[0]
    qn = _weight(process_noise, n)
    rn = _weight(measurement_noise, p)
    gain, cov = kalman_gain(a_m, c_m, qn, rn)
    # Prior estimation-error dynamics e_{k+1} = A (I - L C) e_k.
    poles, stable = _poles(a_m @ (jnp.eye(n) - gain @ c_m), continuous=False)
    return {
        "gain": gain.tolist(),
        "error_covariance": cov.tolist(),
        "estimator_pole_magnitudes": poles,
        "stable": stable,
    }


def _thin(values: jnp.ndarray, points: int) -> list[float]:
    n = values.shape[0]
    idx = jnp.linspace(0, n - 1, min(int(points), n)).round().astype(int)
    return [float(values[i]) for i in idx]


def _build_mpc(
    model: StateSpace,
    q: Any,
    r: Any,
    horizon: int,
    control_horizon: int | None,
    s_du: float,
    constraints: JsonDict,
    discretize: bool,
    dt: float | None,
) -> Any:
    return linear_mpc(
        model,
        q=q,
        r=r,
        horizon=int(horizon),
        control_horizon=control_horizon,
        s_du=s_du,
        dt=dt,
        discretize=discretize,
        u_min=constraints.get("u_min", -jnp.inf),
        u_max=constraints.get("u_max", jnp.inf),
        du_max=constraints.get("du_max", jnp.inf),
        y_min=constraints.get("y_min", -jnp.inf),
        y_max=constraints.get("y_max", jnp.inf),
        disturbance="output",
    )


def _simulate_mpc(
    a: Any,
    b: Any,
    c: Any,
    q: Any,
    r: Any,
    setpoint: Any,
    horizon: int = 15,
    control_horizon: int | None = None,
    s_du: float = 0.0,
    n_steps: int = 60,
    continuous: bool = False,
    dt: float | None = None,
    disturbance: Any | None = None,
    u_min: float | None = None,
    u_max: float | None = None,
    du_max: float | None = None,
    points: int = 41,
) -> JsonDict:
    """Simulate a constrained, offset-free linear MPC in closed loop to a setpoint."""
    model = _state_space(a, b, c)
    n = model.a.shape[0]
    p = model.c.shape[0]
    constraints: JsonDict = {}
    for key, val in (("u_min", u_min), ("u_max", u_max), ("du_max", du_max)):
        if val is not None:
            constraints[key] = float(val)
    mpc = _build_mpc(model, q, r, horizon, control_horizon, s_du, constraints, continuous, dt)
    sp = jnp.atleast_1d(jnp.asarray(setpoint, dtype=float))
    if disturbance is None:
        dist = jnp.zeros((p,))
    else:
        dist = jnp.atleast_1d(jnp.asarray(disturbance, dtype=float))
    a_d = jnp.asarray(mpc.a, dtype=float)
    b_d = jnp.asarray(mpc.b, dtype=float)
    c_d = jnp.asarray(mpc.c, dtype=float)

    def plant(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return a_d @ x + b_d @ u

    def measure(x: jnp.ndarray) -> jnp.ndarray:
        return c_d @ x + dist

    step, ctrl0 = linear_feedback(mpc, jnp.zeros(n))
    loop = simulate_closed_loop(
        plant,
        step,
        jnp.zeros(n),
        ctrl0,
        constant_setpoint(sp, int(n_steps)),
        dt=float(mpc.dt),
        measure=measure,
    )
    t = loop.t
    outputs = loop.outputs
    metrics = []
    for j in range(p):
        info = step_info(t, outputs[:, j], float(sp[j]))
        metrics.append(
            {
                "output": j,
                "overshoot_fraction": float(info.overshoot),
                "rise_time_s": float(info.rise_time),
                "settling_time_s": float(info.settling_time),
                "steady_state_error": float(info.steady_state_error),
                "iae": float(info.iae),
            }
        )
    return {
        "time_s": _thin(t, points),
        "outputs": [_thin(outputs[:, j], points) for j in range(p)],
        "inputs": [_thin(loop.inputs[:, k], points) for k in range(model.b.shape[1])],
        "setpoint": [float(v) for v in sp],
        "final_output": [float(v) for v in outputs[-1]],
        "metrics": metrics,
    }


def _tune_mpc_weights(
    a: Any,
    b: Any,
    c: Any,
    setpoint: Any,
    q0: float = 1.0,
    r0: float = 1.0,
    horizon: int = 12,
    n_steps: int = 40,
    effort_weight: float = 0.0,
    continuous: bool = False,
    dt: float | None = None,
    u_min: float | None = None,
    u_max: float | None = None,
    max_iter: int = 20,
) -> JsonDict:
    """Tune scalar output/input MPC weights ``(q, r)`` to minimize closed-loop cost.

    Descends the tracking ISE (plus an optional input-effort term) on the log of the
    weights -- the gradient flows through the controller's own QP -- and reports the
    cost before and after.
    """
    model = _state_space(a, b, c)
    n = model.a.shape[0]
    sp = jnp.atleast_1d(jnp.asarray(setpoint, dtype=float))
    constraints: JsonDict = {}
    for key, val in (("u_min", u_min), ("u_max", u_max)):
        if val is not None:
            constraints[key] = float(val)

    def simulate(log_w: jnp.ndarray) -> Any:
        mpc = _build_mpc(
            model,
            jnp.exp(log_w[0]),
            jnp.exp(log_w[1]),
            horizon,
            None,
            0.0,
            constraints,
            continuous,
            dt,
        )
        a_d = jnp.asarray(mpc.a, dtype=float)
        b_d = jnp.asarray(mpc.b, dtype=float)
        c_d = jnp.asarray(mpc.c, dtype=float)
        step, ctrl0 = linear_feedback(mpc, jnp.zeros(n))
        return simulate_closed_loop(
            lambda x, u: a_d @ x + b_d @ u,
            step,
            jnp.zeros(n),
            ctrl0,
            constant_setpoint(sp, int(n_steps)),
            measure=lambda x: c_d @ x,
        )

    def perf(loop: Any) -> jnp.ndarray:
        return closed_loop_cost(loop, error_weight=1.0, effort_weight=float(effort_weight))

    start = jnp.array([jnp.log(float(q0)), jnp.log(float(r0))])
    cost0 = float(perf(simulate(start)))
    res = tune_mpc(simulate, start, performance=perf, method="bfgs", max_iter=int(max_iter))
    cost1 = float(perf(simulate(res.x)))
    return {
        "initial": {"q": float(q0), "r": float(r0), "cost": cost0},
        "tuned": {
            "q": float(jnp.exp(res.x[0])),
            "r": float(jnp.exp(res.x[1])),
            "cost": cost1,
        },
        "iterations": int(res.n_iter),
        "improved": bool(cost1 <= cost0 + 1e-9),
    }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def mpc_tool_specs() -> list[Any]:
    """ToolSpecs for the advanced-control layer (folded into ``default_registry``)."""
    from fugacio.copilot.tools import ToolSpec

    matrix_schema = {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    weight_schema = {
        "description": "Scalar, per-channel list (diagonal), or full matrix.",
    }
    return [
        ToolSpec(
            name="lqr_design",
            description=(
                "Infinite-horizon LQR state-feedback gain u = -K x for a linear "
                "state-space model with state weight Q and input weight R (discrete "
                "by default; set continuous=true for continuous time). Returns the "
                "gain, the Riccati cost-to-go, the closed-loop poles and stability."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "a": matrix_schema,
                    "b": matrix_schema,
                    "q": weight_schema,
                    "r": weight_schema,
                    "continuous": {"type": "boolean"},
                },
                "required": ["a", "b", "q", "r"],
            },
            run=_lqr_design,
        ),
        ToolSpec(
            name="kalman_design",
            description=(
                "Steady-state Kalman filter for x+ = A x + w, y = C x + v with "
                "process covariance and measurement covariance. Returns the update "
                "gain, the prior error covariance, and the estimator poles/stability."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "a": matrix_schema,
                    "c": matrix_schema,
                    "process_noise": weight_schema,
                    "measurement_noise": weight_schema,
                },
                "required": ["a", "c", "process_noise", "measurement_noise"],
            },
            run=_kalman_design,
        ),
        ToolSpec(
            name="simulate_mpc",
            description=(
                "Simulate a constrained, offset-free linear model predictive "
                "controller in closed loop against an output setpoint (optionally "
                "with a constant unmeasured output disturbance). Handles input "
                "magnitude/rate limits and reaches the setpoint with zero "
                "steady-state offset; returns the response/input trajectories and "
                "step metrics (overshoot, settling time, IAE)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "a": matrix_schema,
                    "b": matrix_schema,
                    "c": matrix_schema,
                    "q": weight_schema,
                    "r": weight_schema,
                    "setpoint": {"type": "array", "items": {"type": "number"}},
                    "horizon": {"type": "integer"},
                    "control_horizon": {"type": "integer"},
                    "n_steps": {"type": "integer"},
                    "continuous": {"type": "boolean"},
                    "dt": {"type": "number", "description": "Sample time (required if continuous)"},
                    "disturbance": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Constant output disturbance per output channel",
                    },
                    "u_min": {"type": "number"},
                    "u_max": {"type": "number"},
                    "du_max": {"type": "number", "description": "Per-step input rate limit"},
                    "points": {"type": "integer"},
                },
                "required": ["a", "b", "c", "q", "r", "setpoint"],
            },
            run=_simulate_mpc,
        ),
        ToolSpec(
            name="tune_mpc_weights",
            description=(
                "Tune the scalar MPC output/input weights (q, r) by gradient descent "
                "on the closed-loop tracking cost -- the gradient flows through the "
                "controller's own optimization. Returns the cost before/after and the "
                "tuned weights."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "a": matrix_schema,
                    "b": matrix_schema,
                    "c": matrix_schema,
                    "setpoint": {"type": "array", "items": {"type": "number"}},
                    "q0": {"type": "number", "description": "Initial output weight"},
                    "r0": {"type": "number", "description": "Initial input weight"},
                    "horizon": {"type": "integer"},
                    "n_steps": {"type": "integer"},
                    "effort_weight": {"type": "number"},
                    "continuous": {"type": "boolean"},
                    "dt": {"type": "number"},
                    "u_min": {"type": "number"},
                    "u_max": {"type": "number"},
                    "max_iter": {"type": "integer"},
                },
                "required": ["a", "b", "c", "setpoint"],
            },
            run=_tune_mpc_weights,
        ),
    ]


__all__ = ["mpc_tool_specs"]
