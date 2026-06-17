"""Design specifications and set-point controllers for flowsheets.

A *design spec* is the everyday inverse problem of process design: instead of
"given the reflux ratio, what purity do I get?", you ask "what reflux ratio hits
99.5 % purity?". You nominate a **manipulated** variable (a degree of freedom:
a duty, a reflux, a split fraction, a feed temperature) and a **controlled**
variable (a calculated result: a purity, a recovery, a temperature) with a
**target**, and the solver adjusts the manipulated variable until the controlled
variable meets its target. Several specs are solved simultaneously, so coupled
targets (a column's distillate *and* bottoms purity, set by reflux *and* reboiler
duty) converge together.

Because the whole engine is differentiable, a met spec is itself differentiable:
the adjusted manipulated variable (and everything computed from it) carries
exact gradients with respect to the *unmanipulated* parameters (feed, prices,
model parameters). The spec solve reuses the implicit-function-theorem root
finders in `fugacio.thermo.implicit`, so those gradients cost a single
linear solve regardless of how many iterations the spec took, and they compose
with the recycle gradients from `fugacio.sim.flowsheet.tear_solve`.

This is the steady-state, set-point form of control: a controller here drives a
controlled variable to its set point at steady state. Dynamic controllers (PID
and friends) belong to the dynamic-simulation layer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import Array

from fugacio.sim.optimize import minimize
from fugacio.sim.stream import Stream
from fugacio.thermo.implicit import bracketed_root, newton_system

ArrayLike = Array | float

#: A flowsheet evaluation: maps a parameter mapping to the named output streams.
Simulate = Callable[[Mapping[str, Any]], dict[str, Stream]]
#: A controlled-variable measurement read off the solved streams.
Measure = Callable[[dict[str, Stream]], Array]


@dataclass(frozen=True)
class DesignSpec:
    """One design specification: adjust ``manipulated`` until ``measure`` hits ``target``.

    Attributes:
        manipulated: Key in the parameter mapping ``theta`` to adjust (the degree
            of freedom). Its current value seeds the search.
        measure: Reads the controlled variable from the solved streams, e.g.
            ``lambda s: s["distillate"].z[0]`` for a top mole fraction.
        target: Desired value of the controlled variable.
        lo: Lower bound on the manipulated variable (used by the bracketing
            solver and to keep the search physical).
        hi: Upper bound on the manipulated variable.
        name: Optional label for reporting.
    """

    manipulated: str
    measure: Measure
    target: float
    lo: float
    hi: float
    name: str = ""


class SpecResult(NamedTuple):
    """Outcome of a design-spec solve.

    Attributes:
        theta: The parameter mapping with the manipulated variables set to their
            converged values (differentiable with respect to the unmanipulated
            entries of the input ``theta``).
        streams: The solved flowsheet streams at the converged spec.
        manipulated: The converged manipulated values, aligned with the specs.
        residual: Controlled-variable errors ``measure - target`` at the solution.
        converged: Whether every spec met its target within tolerance.
    """

    theta: dict[str, Any]
    streams: dict[str, Stream]
    manipulated: Array
    residual: Array
    converged: Array


def meet_spec(
    measure: Callable[[Array, Any], Array],
    target: ArrayLike,
    u0: ArrayLike,
    theta: Any = None,
    *,
    lo: ArrayLike | None = None,
    hi: ArrayLike | None = None,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Array:
    """Find the manipulated value ``u`` such that ``measure(u, theta) == target``.

    The low-level, single-variable spec solver. When a bracket ``[lo, hi]`` is
    given it uses the robust bisection root finder (safe across the kinks an EOS
    flash produces); otherwise a damped Newton iteration. The returned ``u`` is
    differentiable with respect to ``theta`` by implicit differentiation.

    Args:
        measure: Controlled variable ``measure(u, theta) -> ()``.
        target: Desired value.
        u0: Initial guess for the manipulated variable.
        theta: Differentiable parameter pytree forwarded to ``measure``.
        lo: Lower bracket bound; when both ``lo`` and ``hi`` are given, bisection is used.
        hi: Upper bracket bound; when both ``lo`` and ``hi`` are given, bisection is used.
        tol: Convergence tolerance.
        max_iter: Iteration cap.

    Returns:
        The manipulated value meeting the spec; differentiable in ``theta``.
    """
    tgt = jnp.asarray(target)

    def residual(u: Array, th: Any) -> Array:
        return measure(u, th) - tgt

    if lo is not None and hi is not None:
        return bracketed_root(residual, theta, jnp.asarray(lo), jnp.asarray(hi), tol, max_iter)
    # Newton on a 1-vector keeps the same implicit-diff machinery as multi-spec.
    root = newton_system(
        lambda u, th: jnp.atleast_1d(residual(u[0], th)),
        jnp.atleast_1d(jnp.asarray(u0, dtype=float)),
        theta,
        tol,
        max_iter,
    )
    return root[0]


def solve_design(
    simulate: Simulate,
    theta: Mapping[str, Any],
    specs: Sequence[DesignSpec],
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> SpecResult:
    """Adjust the manipulated variables so every design spec meets its target.

    Solves the (generally coupled) system "set each manipulated variable so its
    controlled variable equals its target" with a single Newton iteration over
    the stacked specs, re-running ``simulate`` (recycles and all) at each step.
    The converged manipulated values (and the streams computed from them)
    are differentiable with respect to the unmanipulated entries of ``theta``.

    Args:
        simulate: Runs the flowsheet for a parameter mapping and returns the named
            output streams, e.g. ``flowsheet.solve`` wrapped to take a mapping, or
            any ``lambda th: {...}``.
        theta: Base parameter mapping. The ``manipulated`` keys named by the specs
            are overwritten by the solver; their values in ``theta`` seed it.
        specs: The design specs to satisfy simultaneously.
        tol: Convergence tolerance on the controlled-variable residuals.
        max_iter: Newton iteration cap.

    Returns:
        A `SpecResult`.
    """
    if not specs:
        streams = simulate(theta)
        return SpecResult(dict(theta), streams, jnp.zeros((0,)), jnp.zeros((0,)), jnp.asarray(True))

    keys = [s.manipulated for s in specs]
    u0 = jnp.array([float(jnp.asarray(theta[k])) for k in keys])
    targets = jnp.array([s.target for s in specs])
    measures = tuple(s.measure for s in specs)

    def _inject(th: Mapping[str, Any], u: Array) -> dict[str, Any]:
        merged = dict(th)
        for i, k in enumerate(keys):
            merged[k] = u[i]
        return merged

    def residual(u: Array, th: Mapping[str, Any]) -> Array:
        streams = simulate(_inject(th, u))
        measured = jnp.array([m(streams) for m in measures])
        return measured - targets

    u_star = newton_system(residual, u0, theta, tol, max_iter)
    theta_star = _inject(theta, u_star)
    streams = simulate(theta_star)
    res = jnp.array([m(streams) for m in measures]) - targets
    converged = jnp.max(jnp.abs(res)) <= jnp.sqrt(jnp.asarray(tol))
    return SpecResult(theta_star, streams, u_star, res, converged)


def controller(
    simulate: Simulate,
    *,
    manipulated: str,
    controlled: Measure,
    set_point: float,
    lo: float,
    hi: float,
    name: str = "",
) -> DesignSpec:
    """Convenience constructor for a single set-point controller as a `DesignSpec`.

    Reads as control language: drive ``controlled`` to ``set_point`` by moving
    ``manipulated`` within ``[lo, hi]``. Combine several with `solve_design`.
    """
    return DesignSpec(
        manipulated=manipulated,
        measure=controlled,
        target=set_point,
        lo=lo,
        hi=hi,
        name=name or f"{manipulated}->{set_point}",
    )


class FlowsheetOptResult(NamedTuple):
    """Outcome of a flowsheet optimization.

    Attributes:
        theta: The parameter mapping with the optimized design variables.
        streams: The solved flowsheet at the optimum.
        objective: Objective value at the optimum.
        converged: Whether the optimizer met its tolerances.
        n_iter: Iterations taken.
    """

    theta: dict[str, Any]
    streams: dict[str, Stream]
    objective: Array
    converged: Array
    n_iter: Array


def optimize_flowsheet(
    simulate: Simulate,
    objective: Callable[[dict[str, Stream], Mapping[str, Any]], Array],
    theta0: Mapping[str, Any],
    design_vars: Sequence[str],
    *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    ineq_constraints: Callable[[Mapping[str, Any]], Array] | None = None,
    method: str = "bfgs",
    tol: float = 1e-6,
    max_iter: int = 200,
) -> FlowsheetOptResult:
    """Optimize selected design variables of a flowsheet against a cost objective.

    Minimizes ``objective(simulate(theta), theta)`` over the ``design_vars`` subset
    of ``theta``, holding the rest fixed. The flowsheet (recycles and all) is
    re-solved at every objective evaluation, and gradients flow through the
    converged flowsheet by implicit differentiation. With a money objective from
    `fugacio.sim.economics` this is end-to-end design optimization.

    Args:
        simulate: Runs the flowsheet for a parameter mapping; returns named streams.
        objective: Scalar cost ``objective(streams, theta) -> ()`` to minimize.
        theta0: Base parameter mapping (seeds the design variables).
        design_vars: Keys of ``theta`` that the optimizer is free to move.
        bounds: Optional ``{var: (lo, hi)}`` box bounds on the design variables.
        ineq_constraints: Optional ``g(theta) -> (k,)`` enforced ``<= 0`` (e.g. a
            minimum-purity constraint as ``target - purity``).
        method: Unconstrained inner method (ignored when bounds/constraints apply).
        tol: Optimality tolerance.
        max_iter: Iteration cap.

    Returns:
        A `FlowsheetOptResult`.
    """
    x0 = {k: jnp.asarray(theta0[k], dtype=float) for k in design_vars}

    def merge(x: Mapping[str, Any]) -> dict[str, Any]:
        return {**dict(theta0), **dict(x)}

    def f(x: Mapping[str, Any], _: Any) -> Array:
        th = merge(x)
        return objective(simulate(th), th)

    box: tuple[Any, Any] | None = None
    if bounds is not None:
        lower = {k: bounds[k][0] if k in bounds else -jnp.inf for k in design_vars}
        upper = {k: bounds[k][1] if k in bounds else jnp.inf for k in design_vars}
        box = (lower, upper)

    ineq = None
    if ineq_constraints is not None:
        ineq = lambda x, _: ineq_constraints(merge(x))  # noqa: E731

    res = minimize(
        f,
        x0,
        None,
        method=method,
        bounds=box,
        ineq_constraints=ineq,
        tol=tol,
        max_iter=max_iter,
    )
    theta_star = merge(res.x)
    streams = simulate(theta_star)
    return FlowsheetOptResult(
        theta=theta_star,
        streams=streams,
        objective=res.fun,
        converged=res.converged,
        n_iter=res.n_iter,
    )
