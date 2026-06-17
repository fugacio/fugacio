"""Dynamic optimization, estimation, and controller tuning.

Because the integrator is end-to-end differentiable, the three classic
"optimization over a dynamic model" problems collapse to gradients through a
simulation composed with the existing optimizers in `fugacio.sim.optimize`:

* `optimal_control`: choose a (piecewise-constant) input trajectory to
  minimize a running-plus-terminal cost, e.g. minimum-energy or minimum-time-like
  transitions; the control is the decision variable and the simulation supplies
  exact gradients of the cost with respect to it;
* `estimate_dynamics`: fit model parameters (and optionally the initial
  state) to time-series measurements by Levenberg-Marquardt least squares through
  the integrator (dynamic data reconciliation / parameter estimation);
* `tune_pid`: descend a closed-loop performance index (IAE / ISE / ITAE)
  directly on the controller gains, exploiting the fact that the gains are a
  differentiable pytree and the whole closed loop is differentiable.

All three return the corresponding `fugacio.sim.OptimizeResult`-style
outcome and re-simulate at the solution for convenience.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.control.metrics import iae, ise, itae
from fugacio.sim.dynamics.integrate import odeint
from fugacio.sim.optimize import OptimizeResult, least_squares, minimize

ArrayLike = Array | float


def _piecewise_constant(u: Array, ts: Array, t: Array) -> Array:
    """Value of a piecewise-constant control ``u`` (one value per ``ts`` interval) at ``t``."""
    idx = jnp.clip(jnp.searchsorted(ts, t, side="right") - 1, 0, u.shape[0] - 1)
    return u[idx]


# --------------------------------------------------------------------------- #
# Optimal control
# --------------------------------------------------------------------------- #
class OptimalControlResult(NamedTuple):
    """Outcome of `optimal_control`.

    Attributes:
        u: Optimal piecewise-constant control (one value, or vector, per interval).
        cost: Optimal total cost (running integral plus terminal).
        trajectory: State trajectory under the optimal control (leading time axis).
        result: The underlying `fugacio.sim.OptimizeResult`.
    """

    u: Any
    cost: Array
    trajectory: Any
    result: OptimizeResult


def optimal_control(
    dynamics: Callable[[Array, Any, Any, Any], Any],
    y0: Any,
    ts: Array,
    u_init: Array,
    stage_cost: Callable[[Array, Any, Any, Any], Array],
    *,
    terminal_cost: Callable[[Any, Any], Array] | None = None,
    theta: Any = None,
    bounds: tuple[Any, Any] | None = None,
    method: str = "bfgs",
    integrator: str = "rk4",
    substeps: int = 2,
    max_iter: int = 100,
) -> OptimalControlResult:
    """Find the input trajectory minimizing a running-plus-terminal cost.

    The control is piecewise constant on the ``ts`` grid: ``u_init`` has one entry
    (scalar or vector) per interval. The augmented state ``(y, J)`` integrates the
    running cost ``J' = stage_cost(t, y, u, theta)`` alongside the dynamics
    ``y' = dynamics(t, y, u, theta)``, and the total ``J(t_f) + terminal_cost(y_f)``
    is minimized over the control with gradients straight through the simulation.

    Args:
        dynamics: ``dynamics(t, y, u, theta) -> dy`` (note the explicit input ``u``).
        y0: Initial state pytree.
        ts: Output/decision grid (length ``N``); the control has ``N - 1`` intervals.
        u_init: Initial control guess, shape ``(N - 1,)`` or ``(N - 1, k)``.
        stage_cost: Running cost ``stage_cost(t, y, u, theta) -> ()``.
        terminal_cost: Optional terminal cost ``terminal_cost(y_f, theta) -> ()``.
        theta: Optional fixed parameter pytree.
        bounds: Optional ``(lower, upper)`` box on the control.
        method: Unconstrained optimizer used over the control (e.g. ``"bfgs"``).
        integrator: ODE integration scheme for the augmented state (e.g. ``"rk4"``).
        substeps: Integration substeps taken between consecutive ``ts`` points.
        max_iter: Maximum number of optimizer iterations.

    Returns:
        An `OptimalControlResult`.
    """
    ts = jnp.asarray(ts, dtype=float)
    u_init = jnp.asarray(u_init, dtype=float)

    def aug_rhs(t: Array, aug: tuple[Any, Array], u: Array) -> tuple[Any, Array]:
        y, _j = aug
        u_t = _piecewise_constant(u, ts, t)
        dy = dynamics(t, y, u_t, theta)
        dj = stage_cost(t, y, u_t, theta)
        return dy, dj

    def total_cost(u: Array, _: Any) -> Array:
        traj = odeint(aug_rhs, (y0, jnp.zeros(())), ts, u, method=integrator, substeps=substeps)
        y_traj, j_traj = traj
        y_final = _tree_last(y_traj)
        running = j_traj[-1]
        terminal = terminal_cost(y_final, theta) if terminal_cost is not None else jnp.asarray(0.0)
        return running + terminal

    res = minimize(total_cost, u_init, None, method=method, bounds=bounds, max_iter=max_iter)
    traj = odeint(
        lambda t, y, u: dynamics(t, y, _piecewise_constant(u, ts, t), theta),
        y0,
        ts,
        res.x,
        method=integrator,
        substeps=substeps,
    )
    return OptimalControlResult(u=res.x, cost=res.fun, trajectory=traj, result=res)


def _tree_last(traj: Any) -> Any:
    return jax.tree_util.tree_map(lambda leaf: leaf[-1], traj)


# --------------------------------------------------------------------------- #
# Parameter estimation from time-series
# --------------------------------------------------------------------------- #
class DynamicEstimateResult(NamedTuple):
    """Outcome of `estimate_dynamics`.

    Attributes:
        theta: Estimated parameter pytree.
        trajectory: Model trajectory at the estimate (leading time axis).
        cost: Half-sum-of-squares residual at the estimate.
        result: The underlying `fugacio.sim.OptimizeResult`.
    """

    theta: Any
    trajectory: Any
    cost: Array
    result: OptimizeResult


def estimate_dynamics(
    dynamics: Callable[[Array, Any, Any], Any],
    y0: Any,
    ts: Array,
    data: Array,
    theta0: Any,
    *,
    observe: Callable[[Any], Array] | None = None,
    weights: Array | None = None,
    integrator: str = "rk4",
    substeps: int = 2,
    max_iter: int = 100,
) -> DynamicEstimateResult:
    """Fit parameters ``theta`` so the simulated trajectory matches ``data``.

    Minimizes ``sum (w (observe(traj) - data))^2`` by Levenberg-Marquardt, with the
    trajectory produced by integrating ``dynamics(t, y, theta)`` from ``y0`` over
    ``ts``. ``observe`` maps the trajectory pytree to the measured quantity
    (defaults to the trajectory itself); ``weights`` optionally scales residuals.

    Returns:
        A `DynamicEstimateResult`.
    """
    ts = jnp.asarray(ts, dtype=float)
    data = jnp.asarray(data, dtype=float)
    obs = observe if observe is not None else (lambda traj: traj)
    w = jnp.asarray(weights) if weights is not None else jnp.asarray(1.0)

    def residual(theta: Any, _: Any) -> Array:
        traj = odeint(dynamics, y0, ts, theta, method=integrator, substeps=substeps)
        pred = obs(traj)
        return jnp.ravel(w * (pred - data))

    sol = least_squares(residual, theta0, None, max_iter=max_iter)
    traj = odeint(dynamics, y0, ts, sol.x, method=integrator, substeps=substeps)
    return DynamicEstimateResult(theta=sol.x, trajectory=traj, cost=sol.fun, result=sol)


# --------------------------------------------------------------------------- #
# Gradient-based PID tuning
# --------------------------------------------------------------------------- #
_OBJECTIVES: dict[str, Callable[[Array, Array, ArrayLike], Array]] = {
    "iae": iae,
    "ise": ise,
    "itae": itae,
}


def tune_pid(
    response: Callable[[Any], Array],
    gains0: Any,
    setpoint: ArrayLike,
    ts: Array,
    *,
    objective: str = "iae",
    bounds: tuple[Any, Any] | None = None,
    method: str = "bfgs",
    max_iter: int = 100,
) -> OptimizeResult:
    """Tune controller gains by minimizing a closed-loop error integral.

    ``response(gains)`` must build the closed loop from the gains pytree, simulate
    it, and return the controlled-variable trajectory sampled on ``ts``. This
    function then minimizes the chosen error integral (``"iae"``, ``"ise"`` or
    ``"itae"``) against ``setpoint`` over the gains: gradients flow through the
    whole simulated loop, so the tune is exact first-order, not a grid search.

    Args:
        response: ``response(gains) -> pv_trajectory`` (length ``len(ts)``).
        gains0: Initial gains pytree (e.g. a dict ``{"kc": ..., "tau_i": ...}``).
        setpoint: Target value for the controlled variable.
        ts: Time grid matching ``response``'s output.
        objective: Error integral to minimize.
        bounds: Optional ``(lower, upper)`` box on the gains (recommended: keeps
            gains positive).
        method: Unconstrained optimizer used over the gains (e.g. ``"bfgs"``).
        max_iter: Maximum number of optimizer iterations.

    Returns:
        The `fugacio.sim.OptimizeResult`; ``result.x`` is the tuned gains.
    """
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective must be one of {tuple(_OBJECTIVES)}, got {objective!r}")
    metric = _OBJECTIVES[objective]
    ts = jnp.asarray(ts, dtype=float)

    def loss(gains: Any, _: Any) -> Array:
        pv = response(gains)
        return metric(ts, pv, setpoint)

    return minimize(loss, gains0, None, method=method, bounds=bounds, max_iter=max_iter)


__all__ = [
    "DynamicEstimateResult",
    "OptimalControlResult",
    "estimate_dynamics",
    "optimal_control",
    "tune_pid",
]
