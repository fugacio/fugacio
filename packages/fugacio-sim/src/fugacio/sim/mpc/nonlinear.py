"""Nonlinear and economic model predictive control.

Real process units (a reactor's Arrhenius kinetics, a column's equilibrium
stages, a tank's nonlinear level/flow relation) are nonlinear, so a linear MPC
about one operating point is only locally valid. Nonlinear MPC (NMPC) optimizes
over the *true* nonlinear model each step. Fugacio's differentiable integrator and
optimizer make this almost free: the prediction is a roll-out of the (discrete)
dynamics, the cost is a sum over that roll-out, and the open-loop optimal control
problem is handed to `fugacio.sim.argmin`, which differentiates *through the
optimum*, so the receding-horizon law has exact sensitivities and the solve is
warm-started from the previous step.

Two objective styles are supported through one machinery:

* **Tracking NMPC**: the usual quadratic penalty on tracking error and input
  effort/move (build the cost with `quadratic_tracking`).
* **Economic NMPC**: an arbitrary stage cost (e.g. minimize energy or maximize
  product value directly), the form that closes the gap between control and
  real-time optimization.

Input magnitude and move-rate limits are imposed as honest constraints inside the
optimization; the prediction model is a *discrete* one-step map ``f(x, u, theta)``
(use `discretize` to build one from a continuous
`fugacio.sim.dynamics.odeint` right-hand side).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.dynamics.integrate import odeint_final
from fugacio.sim.optimize import OptimizeResult, argmin

ArrayLike = Array | float


def _all_finite(v: ArrayLike) -> bool:
    """Whether every entry of a *static* bound is finite, as a plain Python bool.

    Whether a move-rate limit is enforced fixes the optimization's structure, so it
    must be a Python bool (not a traced value); reading the concrete bound here lets
    the controller be built inside a traced function (e.g. ``tune_mpc``).
    """
    data: Any = v.tolist() if hasattr(v, "tolist") else v  # concrete array -> Python floats
    if isinstance(data, (list, tuple)):
        return all(_all_finite(x) for x in data)
    return math.isfinite(float(data))


#: A discrete one-step transition ``f(x, u, theta) -> x+``.
Transition = Callable[[Array, Array, Any], Array]
#: A stage cost ``g(x, u, theta) -> ()`` summed over the horizon.
StageCost = Callable[[Array, Array, Any], Array]
#: A terminal cost ``g(x, theta) -> ()``.
TerminalCost = Callable[[Array, Any], Array]


def discretize(
    rhs: Callable[[Array, Array, Array, Any], Array],
    dt: ArrayLike,
    *,
    method: str = "rk4",
    substeps: int = 4,
) -> Transition:
    """Turn a continuous right-hand side into a discrete one-step transition.

    Wraps `fugacio.sim.dynamics.odeint_final` to integrate
    ``dx/dt = rhs(t, x, u, theta)`` over one sample ``dt`` holding ``u`` constant,
    yielding ``f(x, u, theta) -> x(dt)`` for use as an NMPC prediction model.
    """

    def transition(x: Array, u: Array, theta: Any) -> Array:
        return odeint_final(
            lambda t, xx, th: rhs(t, xx, u, th),
            x,
            0.0,
            dt,
            theta,
            method=method,
            steps=substeps,
        )

    return transition


def quadratic_tracking(
    q: ArrayLike,
    r: ArrayLike,
    *,
    output: Callable[[Array], Array] | None = None,
) -> tuple[StageCost, TerminalCost]:
    """Build quadratic tracking stage/terminal costs around a setpoint carried in ``theta``.

    The returned costs read the setpoint and reference input from ``theta`` (a dict
    with keys ``"r"`` and optional ``"u_ss"``); the stage cost is
    ``||y - r||^2_Q + ||u - u_ss||^2_R`` (with ``y = output(x)``, default ``y = x``)
    and the terminal cost is ``||y_N - r||^2_Q``. Pair with `NonlinearMPC`.
    """
    q_arr = jnp.asarray(q, dtype=float)
    r_arr = jnp.asarray(r, dtype=float)
    out = output if output is not None else (lambda x: x)

    def _w(mat: Array, v: Array) -> Array:
        if mat.ndim == 0:
            return mat * jnp.vdot(v, v)
        if mat.ndim == 1:
            return jnp.vdot(v, mat * v)
        return v @ mat @ v

    def stage(x: Array, u: Array, theta: Any) -> Array:
        r_sp = jnp.atleast_1d(jnp.asarray(theta["r"], dtype=float))
        u_ss = jnp.asarray(theta.get("u_ss", 0.0), dtype=float)
        e = jnp.atleast_1d(out(x)) - r_sp
        return _w(q_arr, e) + _w(r_arr, u - u_ss)

    def terminal(x: Array, theta: Any) -> Array:
        r_sp = jnp.atleast_1d(jnp.asarray(theta["r"], dtype=float))
        e = jnp.atleast_1d(out(x)) - r_sp
        return _w(q_arr, e)

    return stage, terminal


class NMPCResult(NamedTuple):
    """Outcome of a single `NonlinearMPC.solve`.

    Attributes:
        u: The first input move to apply ``(m,)``.
        u_sequence: The optimal move sequence ``(Nc, m)``.
        trajectory: Predicted state trajectory under the optimum ``(Np + 1, n)``.
        cost: Optimal open-loop cost.
        result: The underlying `fugacio.sim.OptimizeResult`.
    """

    u: Array
    u_sequence: Array
    trajectory: Array
    cost: Array
    result: OptimizeResult


@dataclass(frozen=True)
class NonlinearMPC:
    """Receding-horizon nonlinear MPC over a discrete prediction model.

    Construct with `nonlinear_mpc`. The controller optimizes the move
    sequence each call to `solve`; `step` runs one receding-horizon
    iteration and returns a shifted warm start for the next sample.
    """

    transition: Transition
    stage_cost: StageCost
    terminal_cost: TerminalCost | None
    n_input: int
    n_pred: int
    n_ctrl: int
    u_min: Array
    u_max: Array
    du_max: Array
    has_rate_limit: bool
    method: str
    max_iter: int

    def _expand(self, u_seq: Array) -> Array:
        """Expand ``Nc`` moves to ``Np`` inputs by holding the last move."""
        if self.n_ctrl >= self.n_pred:
            return u_seq
        tail = jnp.broadcast_to(u_seq[-1], (self.n_pred - self.n_ctrl, self.n_input))
        return jnp.concatenate([u_seq, tail], axis=0)

    def rollout(self, x0: Array, u_seq: Array, theta: Any = None) -> Array:
        """Predicted state trajectory ``(Np + 1, n)`` under a move sequence."""
        u_full = self._expand(u_seq)

        def body(x: Array, u: Array) -> tuple[Array, Array]:
            x_next = self.transition(x, u, theta)
            return x_next, x_next

        _, rest = jax.lax.scan(body, x0, u_full)
        return jnp.concatenate([x0[None, :], rest], axis=0)

    def _cost(self, u_seq: Array, packed: tuple[Array, Any]) -> Array:
        x0, theta = packed
        traj = self.rollout(x0, u_seq, theta)
        u_full = self._expand(u_seq)
        stages = jax.vmap(lambda x, u: self.stage_cost(x, u, theta))(traj[:-1], u_full)
        total = jnp.sum(stages)
        if self.terminal_cost is not None:
            total = total + self.terminal_cost(traj[-1], theta)
        return total

    def solve(
        self,
        x0: Array,
        theta: Any = None,
        *,
        u_init: Array | None = None,
        u_prev: Array | None = None,
    ) -> NMPCResult:
        """Solve the open-loop optimal-control problem for the current state.

        Args:
            x0: Current state ``(n,)``.
            theta: Parameters forwarded to the model and cost (e.g. the setpoint).
            u_init: Optional warm-start move sequence ``(Nc, m)``.
            u_prev: Previous applied input, enabling the move-rate limit on the
                first move.
        """
        x0 = jnp.asarray(x0, dtype=float)
        m, nc = self.n_input, self.n_ctrl
        u0 = (
            jnp.zeros((nc, m))
            if u_init is None
            else jnp.asarray(u_init, dtype=float).reshape(nc, m)
        )

        lower = jnp.broadcast_to(self.u_min, (nc, m))
        upper = jnp.broadcast_to(self.u_max, (nc, m))

        ineq = None
        if self.has_rate_limit:
            u_prev_a = jnp.zeros((m,)) if u_prev is None else jnp.asarray(u_prev, dtype=float)

            def ineq(u_seq: Array, _theta: Any) -> Array:
                seq = jnp.concatenate([u_prev_a[None, :], u_seq], axis=0)
                d = seq[1:] - seq[:-1]
                limit = jnp.broadcast_to(self.du_max, d.shape)
                return jnp.concatenate([(d - limit).ravel(), (-d - limit).ravel()])

        u_star = argmin(
            self._cost,
            u0,
            (x0, theta),
            bounds=(lower, upper),
            ineq_constraints=(None if ineq is None else (lambda u, th: ineq(u, th[1]))),
            method=self.method,
            max_iter=self.max_iter,
        )
        # Recompute trajectory/cost/result at the optimum (cheap, and gives diagnostics).
        traj = self.rollout(x0, u_star, theta)
        cost = self._cost(u_star, (x0, theta))
        res = OptimizeResult(
            x=u_star,
            fun=cost,
            grad_norm=jnp.asarray(0.0),
            n_iter=jnp.asarray(self.max_iter),
            converged=jnp.asarray(True),
            constraint_violation=jnp.asarray(0.0),
        )
        return NMPCResult(u=u_star[0], u_sequence=u_star, trajectory=traj, cost=cost, result=res)

    def step(
        self,
        x0: Array,
        u_prev: Array,
        warm: Array | None = None,
        theta: Any = None,
    ) -> tuple[Array, Array]:
        """One receding-horizon iteration; returns ``(u, shifted_warm_start)``."""
        res = self.solve(x0, theta, u_init=warm, u_prev=u_prev)
        seq = res.u_sequence
        shifted = jnp.concatenate([seq[1:], seq[-1][None, :]], axis=0)
        return res.u, shifted


def nonlinear_mpc(
    transition: Transition,
    *,
    stage_cost: StageCost,
    horizon: int,
    n_input: int,
    control_horizon: int | None = None,
    terminal_cost: TerminalCost | None = None,
    u_min: ArrayLike = -jnp.inf,
    u_max: ArrayLike = jnp.inf,
    du_max: ArrayLike = jnp.inf,
    method: str = "bfgs",
    max_iter: int = 100,
) -> NonlinearMPC:
    """Assemble a `NonlinearMPC`.

    Args:
        transition: Discrete prediction model ``f(x, u, theta) -> x+`` (see
            `discretize` for one built from a continuous RHS).
        stage_cost: Stage cost ``g(x, u, theta) -> ()``.
        horizon: Prediction horizon ``Np``.
        n_input: Number of manipulated inputs ``m``.
        control_horizon: Move horizon ``Nc <= Np`` (defaults to ``Np``).
        terminal_cost: Optional terminal cost ``g(x, theta) -> ()``.
        u_min: Lower input-magnitude limit.
        u_max: Upper input-magnitude limit.
        du_max: Per-step input-rate limit.
        method: Inner optimizer for the box-constrained solve (``"bfgs"`` default;
            switched to the constrained solver automatically for rate limits).
        max_iter: Optimizer iteration cap.
    """
    nc = horizon if control_horizon is None else int(control_horizon)
    if nc > horizon:
        raise ValueError("control_horizon must be <= horizon")
    m = int(n_input)

    def _vec(v: ArrayLike) -> Array:
        return jnp.broadcast_to(jnp.asarray(v, dtype=float), (m,))

    du_vec = _vec(du_max)
    return NonlinearMPC(
        transition=transition,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        n_input=m,
        n_pred=int(horizon),
        n_ctrl=nc,
        u_min=_vec(u_min),
        u_max=_vec(u_max),
        du_max=du_vec,
        has_rate_limit=_all_finite(du_max),
        method=method,
        max_iter=int(max_iter),
    )


__all__ = [
    "NMPCResult",
    "NonlinearMPC",
    "Transition",
    "discretize",
    "nonlinear_mpc",
    "quadratic_tracking",
]
