"""Closed-loop simulation and gradient-based controller tuning.

A controller is only as good as the loop it closes. This module runs a *discrete*
closed loop (plant, measurement, controller, repeat) as a single
`jax.lax.scan`, so the whole simulation is one differentiable, jit-able
function. Two payoffs follow directly:

* **One harness for every controller.** Linear MPC, nonlinear MPC and the PID of
  `fugacio.sim.control` all expose the same step protocol
  ``(controller_state, measurement, setpoint) -> (input, controller_state)``, so
  `simulate_closed_loop` drives any of them (adapters
  `linear_feedback` / `nonlinear_feedback` wrap the two MPC classes).
* **Tuning by descent, not grid search.** Because the loop is differentiable and
  the MPC step differentiates *through its own QP* (see
  `fugacio.sim.mpc.qp`), the gradient of a closed-loop performance index with
  respect to the controller weights flows straight through the optimization.
  `tune_mpc` hands that gradient to `fugacio.sim.minimize`, so the
  weights ``Q, R, ...`` are tuned the same way PID gains are in
  `fugacio.sim.dynamics.tune_pid`.

The plant is a discrete one-step map ``x+ = plant(x, u)``; wrap a continuous
right-hand side with `fugacio.sim.mpc.discretize` to obtain one. Optional
per-step process and measurement noise sequences are added so a tune can be made
robust to disturbances.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.optimize import OptimizeResult, minimize

ArrayLike = Array | float

#: A controller step ``(state, measurement, setpoint) -> (input, state)``.
ControllerStep = Callable[[Any, Array, Array], tuple[Array, Any]]
#: A discrete plant map ``(x, u) -> x+``.
Plant = Callable[[Array, Array], Array]
#: A measurement map ``x -> y``.
Measure = Callable[[Array], Array]


class ClosedLoop(NamedTuple):
    """Trajectories returned by `simulate_closed_loop`.

    With ``T`` control steps, ``n`` states, ``m`` inputs and ``p`` outputs:

    Attributes:
        t: Sample times ``(T + 1,)``.
        states: Plant states ``(T + 1, n)`` (``x_0 .. x_T``).
        outputs: Clean outputs ``measure(x)`` ``(T + 1, p)``.
        measurements: Noisy outputs fed to the controller ``(T, p)``.
        inputs: Applied inputs ``(T, m)``.
        setpoints: Output setpoints used at each step ``(T, p)``.
    """

    t: Array
    states: Array
    outputs: Array
    measurements: Array
    inputs: Array
    setpoints: Array


def constant_setpoint(r: ArrayLike, n_steps: int) -> Array:
    """A constant output setpoint trajectory ``(n_steps, p)``."""
    r = jnp.atleast_1d(jnp.asarray(r, dtype=float))
    return jnp.tile(r, (int(n_steps), 1))


def _as_setpoints(setpoints: ArrayLike) -> Array:
    sp = jnp.asarray(setpoints, dtype=float)
    return sp[:, None] if sp.ndim == 1 else sp


def simulate_closed_loop(
    plant: Plant,
    controller: ControllerStep,
    x0: ArrayLike,
    ctrl0: Any,
    setpoints: ArrayLike,
    *,
    dt: ArrayLike = 1.0,
    measure: Measure | None = None,
    process_noise: ArrayLike | None = None,
    meas_noise: ArrayLike | None = None,
) -> ClosedLoop:
    """Run a discrete closed loop and return its trajectories.

    At each step ``k`` the measurement ``y_k = measure(x_k) + v_k`` is formed, the
    controller produces ``u_k`` (advancing its own state), and the plant steps to
    ``x_{k+1} = plant(x_k, u_k) + w_k``. The whole march is a single
    `jax.lax.scan`, hence differentiable and jit-able.

    Args:
        plant: Discrete plant map ``(x, u) -> x+``.
        controller: Step ``(state, measurement, setpoint) -> (input, state)``
            (see `linear_feedback` / `nonlinear_feedback`).
        x0: Initial plant state ``(n,)``.
        ctrl0: Initial controller state.
        setpoints: Output setpoints ``(T, p)`` (or ``(T,)`` for a scalar output);
            its length sets the number of control steps ``T``.
        dt: Sample time (for the reported time axis only).
        measure: Output map ``x -> y`` (defaults to the identity, full-state
            feedback).
        process_noise: Optional additive state noise ``(T, n)``.
        meas_noise: Optional additive measurement noise ``(T, p)``.

    Returns:
        A `ClosedLoop` with the state, output, measurement, input and
        setpoint trajectories.
    """
    x0 = jnp.asarray(x0, dtype=float)
    obs = measure if measure is not None else (lambda x: x)
    sp = _as_setpoints(setpoints)
    n_steps = sp.shape[0]
    p = obs(x0).shape[0]

    w_proc = (
        jnp.zeros((n_steps, *x0.shape))
        if process_noise is None
        else jnp.broadcast_to(jnp.asarray(process_noise, dtype=float), (n_steps, *x0.shape))
    )
    w_meas = (
        jnp.zeros((n_steps, p))
        if meas_noise is None
        else jnp.broadcast_to(jnp.asarray(meas_noise, dtype=float), (n_steps, p))
    )

    def body(
        carry: tuple[Array, Any], inp: tuple[Array, Array, Array]
    ) -> tuple[tuple[Array, Any], tuple[Array, Array, Array, Array]]:
        x, cstate = carry
        r, wp, wv = inp
        y = obs(x)
        y_meas = y + wv
        u, cstate_next = controller(cstate, y_meas, r)
        u = jnp.atleast_1d(u)
        x_next = jnp.asarray(plant(x, u), dtype=float) + wp
        return (x_next, cstate_next), (x, y, y_meas, u)

    (x_final, _), (xs, ys, yms, us) = jax.lax.scan(body, (x0, ctrl0), (sp, w_proc, w_meas))

    states = jnp.concatenate([xs, x_final[None, :]], axis=0)
    outputs = jnp.concatenate([ys, obs(x_final)[None, :]], axis=0)
    t = jnp.arange(n_steps + 1, dtype=float) * jnp.asarray(dt, dtype=float)
    return ClosedLoop(
        t=t,
        states=states,
        outputs=outputs,
        measurements=yms,
        inputs=us,
        setpoints=sp,
    )


# --------------------------------------------------------------------------- #
# Performance index
# --------------------------------------------------------------------------- #
def _quad(weight: ArrayLike, v: Array) -> Array:
    """Sum of ``v_k^T W v_k`` over the leading axis (scalar / vector / matrix ``W``)."""
    w = jnp.asarray(weight, dtype=float)
    if w.ndim == 0:
        return w * jnp.sum(v * v)
    if w.ndim == 1:
        return jnp.sum(v * (w * v))
    return jnp.sum(jnp.einsum("ti,ij,tj->t", v, w, v))


def closed_loop_cost(
    loop: ClosedLoop,
    *,
    error_weight: ArrayLike = 1.0,
    effort_weight: ArrayLike = 0.0,
    move_weight: ArrayLike = 0.0,
    u_ref: ArrayLike = 0.0,
) -> Array:
    """A differentiable quadratic closed-loop performance index.

    Sums the tracking error of the *resulting* outputs against the setpoints, plus
    optional input-effort and input-move penalties:

        ``sum_k ||y_{k+1} - r_k||^2_We + ||u_k - u_ref||^2_Wu + ||u_k - u_{k-1}||^2_Wd``

    Each weight may be a scalar, a per-channel vector, or a full matrix. This is the
    natural objective for `tune_mpc`.
    """
    err = loop.outputs[1:] - loop.setpoints
    cost = _quad(error_weight, err)
    u_ref_a = jnp.asarray(u_ref, dtype=float)
    cost = cost + _quad(effort_weight, loop.inputs - u_ref_a)
    du = jnp.diff(loop.inputs, axis=0)
    cost = cost + _quad(move_weight, du)
    return cost


# --------------------------------------------------------------------------- #
# Controller adapters (MPC classes -> the closed-loop step protocol)
# --------------------------------------------------------------------------- #
def linear_feedback(
    mpc: Any,
    x0: ArrayLike,
    *,
    u0: ArrayLike | None = None,
    d0: ArrayLike | None = None,
) -> tuple[ControllerStep, Any]:
    """Adapt a `LinearMPC` to the closed-loop step protocol.

    Returns ``(step, ctrl0)`` ready for `simulate_closed_loop`; ``step`` is the
    controller's own estimate-optimize-advance iteration and ``ctrl0`` its initial
    estimator state.
    """
    return mpc.step, mpc.init_state(x0, u0, d0)


def nonlinear_feedback(
    mpc: Any,
    *,
    u0: ArrayLike | None = None,
    theta: Any = None,
) -> tuple[ControllerStep, Any]:
    """Adapt a `NonlinearMPC` to the closed-loop step protocol.

    Assumes full-state feedback (the measurement *is* the state). The setpoint passed
    by the harness is written into ``theta["r"]`` each step; any other entries of
    ``theta`` (e.g. ``"u_ss"`` or model parameters) are carried through unchanged. The
    controller state is ``(u_prev, warm_start)``.
    """
    m = mpc.n_input
    nc = mpc.n_ctrl
    u_init = jnp.zeros((m,)) if u0 is None else jnp.broadcast_to(jnp.asarray(u0, dtype=float), (m,))
    warm0 = jnp.zeros((nc, m))
    base: dict[str, Any] = {} if theta is None else dict(theta)

    def step(
        cstate: tuple[Array, Array], x_meas: Array, r: Array
    ) -> tuple[Array, tuple[Array, Array]]:
        u_prev, warm = cstate
        th = {**base, "r": r}
        u, warm_next = mpc.step(x_meas, u_prev, warm, th)
        return u, (u, warm_next)

    return step, (u_init, warm0)


# --------------------------------------------------------------------------- #
# Gradient-based weight tuning
# --------------------------------------------------------------------------- #
def tune_mpc(
    simulate: Callable[[Any], ClosedLoop],
    params0: Any,
    *,
    performance: Callable[[ClosedLoop], Array] | None = None,
    bounds: tuple[Any, Any] | None = None,
    method: str = "bfgs",
    max_iter: int = 60,
) -> OptimizeResult:
    """Tune controller parameters by descending a closed-loop performance index.

    ``simulate(params)`` must build the controller from the ``params`` pytree, run the
    closed loop, and return the `ClosedLoop`; ``performance`` scores it
    (defaults to `closed_loop_cost`, i.e. tracking ISE). The gradient flows
    through the simulation *and the MPC's own QP*, so the tune is exact first-order.

    Args:
        simulate: ``params -> ClosedLoop`` (the differentiable closed loop).
        params0: Initial parameter pytree (e.g. ``{"q": ..., "r": ...}``; tune the
            *weights*, keeping integer horizons and constraint bounds static).
        performance: Scalar score ``ClosedLoop -> ()`` to minimize.
        bounds: Optional ``(lower, upper)`` box on ``params`` (recommended: keeps
            weights positive).
        method: Unconstrained optimizer forwarded to `fugacio.sim.minimize`.
        max_iter: Maximum optimizer iterations forwarded to `fugacio.sim.minimize`.

    Returns:
        The `fugacio.sim.OptimizeResult`; ``result.x`` is the tuned params.
    """
    score = performance if performance is not None else closed_loop_cost

    def loss(params: Any, _: Any) -> Array:
        return score(simulate(params))

    return minimize(loss, params0, None, method=method, bounds=bounds, max_iter=max_iter)


__all__ = [
    "ClosedLoop",
    "ControllerStep",
    "Measure",
    "Plant",
    "closed_loop_cost",
    "constant_setpoint",
    "linear_feedback",
    "nonlinear_feedback",
    "simulate_closed_loop",
    "tune_mpc",
]
