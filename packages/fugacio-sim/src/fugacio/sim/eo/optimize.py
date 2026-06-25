"""Optimization over an equation-oriented flowsheet.

An EO flowsheet is differentiable end to end, so a design-optimization problem
("minimize the cost / maximize the yield over these operating variables") is a
smooth program the gradient-based optimizers in `fugacio.sim.optimize` solve
directly. Two equivalent formulations are offered:

* **Nested** (``simultaneous=False``, the robust default): the decision variables
  parameterize the flowsheet, each objective evaluation solves the flowsheet to
  convergence (one Newton solve), and the optimizer descends the resulting
  reduced objective. The flowsheet solve is differentiable, so the reduced
  gradient is exact.
* **Simultaneous / full-space** (``simultaneous=True``): the decision variables
  *and* the flowsheet state are optimized together, with the flowsheet equations
  imposed as equality constraints (an augmented-Lagrangian solve). No inner loop
  converges the flowsheet; feasibility and optimality are reached at once, the
  classic equation-oriented optimization paradigm.

Both return the optimal decision and the converged flowsheet at that optimum.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from fugacio.sim.eo.flowsheet import EOFlowsheet, EOSolution
from fugacio.sim.optimize import minimize
from fugacio.sim.stream import Stream

#: An objective read off the solved streams (minimized).
Objective = Callable[[Mapping[str, Stream]], Array]

#: A decision variable: ``(init, lower, upper)``.
Decision = tuple[float, float, float]


@dataclass(frozen=True)
class EOOptResult:
    """Outcome of an equation-oriented flowsheet optimization.

    Attributes:
        decision: Optimal value of each decision variable (keyed by name).
        objective: Objective value at the optimum.
        solution: The converged `EOSolution` at the optimum.
        converged: Whether the optimizer met its tolerances.
        constraint_violation: Max flowsheet-equation violation (0 for the nested
            formulation; the feasibility residual for the simultaneous one).
    """

    decision: dict[str, Array]
    objective: Array
    solution: EOSolution
    converged: Array
    constraint_violation: Array


def optimize_flowsheet_eo(
    flowsheet: EOFlowsheet,
    objective: Objective,
    decisions: Mapping[str, Decision],
    *,
    params: Mapping[str, Any] | None = None,
    simultaneous: bool = False,
    tol: float = 1e-6,
    max_iter: int = 200,
    solve_kwargs: Mapping[str, Any] | None = None,
) -> EOOptResult:
    """Optimize an EO flowsheet over named decision variables.

    Args:
        flowsheet: The `EOFlowsheet` to optimize.
        objective: Scalar objective ``objective(streams) -> ()`` to minimize,
            read off the solved streams.
        decisions: Map of parameter key -> ``(init, lower, upper)``. Each key is
            read by some block; the optimizer varies it within its bounds.
        params: Fixed parameters merged under the decisions for every solve.
        simultaneous: Use the full-space formulation (flowsheet equations as
            equality constraints) instead of the nested one.
        tol: Optimizer tolerance.
        max_iter: Optimizer outer-iteration cap.
        solve_kwargs: Extra keyword arguments forwarded to `EOFlowsheet.solve`
            (e.g. ``sweeps``, ``tol``, ``max_iter``) in the nested formulation.

    Returns:
        An `EOOptResult` with the optimal decision and the converged flowsheet.
    """
    base = dict(params or {})
    names = list(decisions)
    lower = {k: decisions[k][1] for k in names}
    upper = {k: decisions[k][2] for k in names}
    d0 = {k: jnp.asarray(decisions[k][0], dtype=float) for k in names}
    skw = dict(solve_kwargs or {})

    if simultaneous:
        return _optimize_simultaneous(
            flowsheet, objective, base, d0, lower, upper, tol, max_iter, skw
        )
    return _optimize_nested(flowsheet, objective, base, d0, lower, upper, tol, max_iter, skw)


def _optimize_nested(
    flowsheet: EOFlowsheet,
    objective: Objective,
    base: dict[str, Any],
    d0: dict[str, Array],
    lower: dict[str, float],
    upper: dict[str, float],
    tol: float,
    max_iter: int,
    solve_kwargs: dict[str, Any],
) -> EOOptResult:
    """Nested formulation: solve the flowsheet inside each objective evaluation."""
    # One concrete solve at the initial decision warms the flowsheet's seed cache,
    # so every differentiated inner solve warm-starts from a real solution instead
    # of rebuilding the (eager, slow) sequential-modular seed each iteration.
    flowsheet.solve({**base, **d0}, check_dof=False, **solve_kwargs)

    def reduced(d: dict[str, Array], _theta: Any) -> Array:
        sol = flowsheet.solve({**base, **d}, check_dof=False, **solve_kwargs)
        return objective(sol.streams)

    res = minimize(reduced, d0, None, bounds=(lower, upper), tol=tol, max_iter=max_iter)
    final = flowsheet.solve({**base, **res.x}, check_dof=False, **solve_kwargs)
    return EOOptResult(
        decision=dict(res.x),
        objective=res.fun,
        solution=final,
        converged=res.converged,
        constraint_violation=jnp.asarray(0.0),
    )


def _decision_scales(
    flowsheet: EOFlowsheet,
    ctx: Any,
    internal: Any,
    aux_scales: Mapping[str, float],
    base: Mapping[str, Any],
    d0: Mapping[str, Array],
    u0: Any,
) -> dict[str, float]:
    """Per-decision scales that put the constraint-Jacobian columns at O(1).

    The flowsheet unknowns are already non-dimensional, but a decision variable
    enters in its natural units and frequently couples to the *scaled* residuals
    only weakly: a heater duty, for instance, appears in the energy balance
    divided by the (large) enthalpy-flow scale, so ``d residual / d duty`` is
    tiny. The single-step augmented-Lagrangian inner solver shares one step size
    across all variables, so such a decision is effectively frozen. Scaling each
    decision by the inverse magnitude of its residual-Jacobian column restores a
    common footing; a decision that does not touch the residuals at the seed
    falls back to its own magnitude.
    """
    feeds = flowsheet.feeds

    def cons_of_d(dvals: dict[str, Array]) -> Array:
        streams = flowsheet._assemble_streams(feeds, ctx, internal, u0)
        return flowsheet._assemble_residual(streams, u0, {**base, **dvals}, aux_scales, ctx)

    jac = jax.jacfwd(cons_of_d)(dict(d0))
    scales: dict[str, float] = {}
    for k in d0:
        col = float(jnp.max(jnp.abs(jac[k])))
        scales[k] = (1.0 / col) if col > 1e-12 else max(abs(float(d0[k])), 1.0)
    return scales


def _optimize_simultaneous(
    flowsheet: EOFlowsheet,
    objective: Objective,
    base: dict[str, Any],
    d0: dict[str, Array],
    lower: dict[str, float],
    upper: dict[str, float],
    tol: float,
    max_iter: int,
    solve_kwargs: dict[str, Any],
) -> EOOptResult:
    """Full-space formulation: flowsheet equations imposed as equality constraints."""
    ctx = flowsheet._context()
    internal = flowsheet._internal_names()
    aux_scales = flowsheet._aux_scales(ctx)
    sweeps = int(solve_kwargs.get("sweeps", 6))
    u0 = flowsheet._initial_unknowns(
        ctx, internal, aux_scales, {**base, **d0}, flowsheet.feeds, None, sweeps
    )
    x0, unravel = ravel_pytree(u0)

    # Optimize the decisions in a non-dimensional space ``q`` so the full-space
    # solver can actually move them (see ``_decision_scales``).
    d_scale = _decision_scales(flowsheet, ctx, internal, aux_scales, base, d0, u0)
    q0 = {k: d0[k] / d_scale[k] for k in d0}

    decision0 = {"u": x0, "q": q0}
    inf = jnp.full_like(x0, jnp.inf)
    lo = {"u": -inf, "q": {k: lower[k] / d_scale[k] for k in d0}}
    hi = {"u": inf, "q": {k: upper[k] / d_scale[k] for k in d0}}

    def _decisions(dec: dict[str, Any]) -> dict[str, Array]:
        return {k: dec["q"][k] * d_scale[k] for k in d0}

    def _streams(dec: dict[str, Any]) -> tuple[dict[str, Stream], dict[str, Any], dict[str, Any]]:
        u = unravel(dec["u"])
        params = {**base, **_decisions(dec)}
        streams = flowsheet._assemble_streams(flowsheet.feeds, ctx, internal, u)
        return streams, u, params

    def obj(dec: dict[str, Any], _theta: Any) -> Array:
        streams, _u, _params = _streams(dec)
        return objective(streams)

    def cons(dec: dict[str, Any], _theta: Any) -> Array:
        streams, u, params = _streams(dec)
        return flowsheet._assemble_residual(streams, u, params, aux_scales, ctx)

    res = minimize(
        obj,
        decision0,
        None,
        bounds=(lo, hi),
        eq_constraints=cons,
        tol=tol,
        max_iter=max_iter,
    )
    d_star = {k: res.x["q"][k] * d_scale[k] for k in d0}
    final = flowsheet.solve({**base, **d_star}, check_dof=False, **solve_kwargs)
    return EOOptResult(
        decision=d_star,
        objective=res.fun,
        solution=final,
        converged=res.converged,
        constraint_violation=res.constraint_violation,
    )
