"""Reproducible parameter sweeps, checked sensitivities, and constrained design studies."""

from __future__ import annotations

import gc
import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from fugacio.sim.cases.quantities import (
    TEMPERATURE,
    display_value,
    number,
    object_fields,
    quantity,
    unit_for,
)
from fugacio.sim.cases.results import CaseRun, compare_runs, sealed
from fugacio.sim.cases.runtime import CaseRunner
from fugacio.sim.cases.schema import sequence
from fugacio.sim.cases.workspace import CaseWorkspace
from fugacio.thermo.sensitivity import derivative_strategy, linearize

if TYPE_CHECKING:
    from fugacio.sim.cases.profiling import PerformanceRecorder


@dataclass(frozen=True)
class StudyResult:
    """A study manifest and the complete audited runs it references.

    ``save`` writes runs first, then the manifest, so a published manifest never
    references an unwritten run. Failed points remain first-class study records.
    """

    artifact: dict[str, Any]
    runs: tuple[CaseRun, ...]

    @property
    def study_id(self) -> str:
        """Content identity of the study manifest."""
        return self.artifact["artifact_id"]

    @property
    def accepted(self) -> bool:
        """Whether the study's explicit acceptance criteria passed."""
        return self.artifact["accepted"]

    def save(self, workspace: CaseWorkspace) -> str:
        """Persist all dependent runs and then the manifest."""
        for run in self.runs:
            workspace.save_run(run)
        return workspace.save_artifact(self.artifact)


def _names(runner: CaseRunner, names: list[str], *, metrics: bool = False) -> tuple[str, ...]:
    sequence(names, "metrics" if metrics else "parameters", minimum=1, maximum=100)
    known = runner.document["metrics"] if metrics else runner.parameters
    if not all(isinstance(k, str) and k in known for k in names) or len(set(names)) != len(names):
        raise ValueError("study names must be unique declared metrics or parameters")
    if not metrics and set(names) & set(runner.manipulated):
        raise ValueError("design-spec manipulated parameters cannot also be study variables")
    return tuple(names)


def _overrides(runner: CaseRunner, values: dict[str, Any]) -> dict[str, Any]:
    return {
        k: {
            "value": float(display_value(v, runner.parameters[k].unit)),
            "unit": runner.parameters[k].unit,
        }
        for k, v in values.items()
    }


def sweep(
    runner: CaseRunner,
    grid: dict[str, list[Any]],
    *,
    workspace: CaseWorkspace | None = None,
    max_points: int = 100,
) -> StudyResult:
    """Run a deterministic Cartesian sweep of explicit parameter quantities.

    Invalid point inputs and failed physical solves are retained, never silently
    omitted from averages or presented as successful operating points.
    """
    if not isinstance(grid, dict) or not grid:
        raise ValueError("sweep needs a nonempty parameter grid")
    names = _names(runner, list(grid))
    axes = [sequence(grid[k], "grid." + k, minimum=1, maximum=1000) for k in names]
    if (
        isinstance(max_points, bool)
        or not isinstance(max_points, int)
        or not 1 <= max_points <= 1000
    ):
        raise ValueError("max_points must be from one to 1000")
    if math.prod(map(len, axes)) > max_points:
        raise ValueError("Cartesian grid exceeds max_points")
    # Reject malformed units/references up front, while retaining valid but
    # physically impossible combinations as individual failed points below.
    for name, axis in zip(names, axes, strict=True):
        p = runner.parameters[name]
        for raw in axis:
            quantity(raw, p.dimension, "grid." + name, difference=p.difference)
    points, runs = [], []
    for row in itertools.product(*axes):
        overrides = dict(zip(names, row, strict=True))
        try:
            run = runner.run(overrides)
            runs.append(run)
            if workspace is not None:
                workspace.save_run(run)
            points.append(
                {
                    "parameters": overrides,
                    "accepted": run.accepted,
                    "run_id": run.run_id,
                    "metrics": run.to_dict()["metrics"],
                }
            )
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            points.append(
                {
                    "parameters": overrides,
                    "accepted": False,
                    "run_id": None,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
    result = StudyResult(
        sealed(
            "sweep",
            {
                "case_id": runner.case.case_id,
                "grid": grid,
                "points": points,
                "accepted": all(p["accepted"] for p in points),
                "accepted_count": sum(p["accepted"] for p in points),
                "failed_count": sum(not p["accepted"] for p in points),
            },
        ),
        tuple(runs),
    )
    if workspace is not None:
        result.save(workspace)
    return result


def _phase_signature(run: CaseRun) -> dict[str, str]:
    result = {}
    for k, s in run.to_dict()["streams"].items():
        beta = s.get("vapor_fraction")
        result[k] = (
            "empty"
            if s["flow_mol_s"] == 0
            else "unknown"
            if beta is None
            else "liquid"
            if beta < 1e-7
            else "vapor"
            if beta > 1 - 1e-7
            else "two_phase"
        )
    return result


def _failed_baseline(
    kind: str,
    runner: CaseRunner,
    baseline: CaseRun,
    request: dict[str, Any],
    workspace: CaseWorkspace | None,
) -> StudyResult:
    result = StudyResult(
        sealed(
            kind,
            {
                "case_id": runner.case.case_id,
                "baseline_id": baseline.run_id,
                "accepted": False,
                "reason": "The baseline failed its independent acceptance checks.",
                "request": request,
                "scope": "No derivative or optimization claim follows from this failed baseline.",
            },
        ),
        (baseline,),
    )
    if workspace is not None:
        result.save(workspace)
    return result


def _directional_jacobian(
    function: Callable[..., Any],
    *,
    mode: str = "auto",
    batch_size: int = 1,
    recorder: PerformanceRecorder | None = None,
) -> Callable[..., Any]:
    """Reuse one converged linearization for every direction at a design point."""
    count = 0

    def evaluate(x: Any) -> tuple[np.ndarray, np.ndarray, bool]:
        nonlocal count
        index, count = count, count + 1
        if recorder is None:
            local = linearize(function, jnp.asarray(x), has_aux=True)
            matrix = np.asarray(local.jacobian(mode=mode, batch_size=batch_size))
        else:
            local = recorder.measure(
                f"linearization[{index}]",
                lambda: linearize(function, jnp.asarray(x), has_aux=True),
            )
            matrix = np.asarray(
                recorder.measure(
                    f"jacobian[{index}]", lambda: local.jacobian(mode=mode, batch_size=batch_size)
                )
            )
        value = np.asarray(local.value)
        accepted = bool(local.auxiliary) and bool(np.all(np.isfinite(matrix)))
        return value, matrix, accepted and bool(np.all(np.isfinite(value)))

    return evaluate


def sensitivities(
    runner: CaseRunner,
    parameters: list[str],
    metrics: list[str],
    *,
    overrides: dict[str, Any] | None = None,
    relative_step: float = 1e-4,
    relative_tolerance: float = 2e-3,
    release_caches: bool = False,
    derivative_mode: str = "auto",
    derivative_batch_size: int = 1,
    workspace: CaseWorkspace | None = None,
    recorder: PerformanceRecorder | None = None,
) -> StudyResult:
    """Compare implicit JAX derivatives with centered differences of accepted runs.

    Both differences stay within declared bounds. Phase-regime changes,
    nonfinite derivatives, failed perturbed runs, and boundary points are
    reported as unverified, with no extrapolated gradient claim. This checks a
    local derivative; it doesn't establish uncertainty or a global response.
    An optional ``recorder`` observes derivative phases without changing the
    study's identity or acceptance criteria.
    """
    names = _names(runner, parameters)
    outputs = _names(runner, metrics, metrics=True)
    strategy = derivative_strategy(
        len(names), len(outputs), mode=derivative_mode, batch_size=derivative_batch_size
    )
    if not isinstance(release_caches, bool):
        raise ValueError("release_caches must be a boolean")
    if (
        not 0 < number(relative_step, "relative_step") < 0.1
        or number(relative_tolerance, "relative_tolerance") <= 0
    ):
        raise ValueError("invalid finite-difference settings")
    baseline = runner.run(overrides)
    if workspace is not None:
        workspace.save_run(baseline)
    if not baseline.accepted:
        return _failed_baseline(
            "sensitivities",
            runner,
            baseline,
            {
                "parameters": parameters,
                "metrics": metrics,
                "relative_step": relative_step,
                "relative_tolerance": relative_tolerance,
                "release_caches": release_caches,
                "derivative_mode": derivative_mode,
                "derivative_batch_size": derivative_batch_size,
            },
            workspace,
        )
    initialization = runner.initialization(baseline)
    values = runner.parameter_values(overrides)
    x = jnp.asarray([values[k] for k in names])
    if release_caches:
        jax.clear_caches()
        gc.collect()

    def vector(v: Any) -> Any:
        e = runner.evaluate(
            {**values, **dict(zip(names, v, strict=True))}, initialization=initialization
        )
        return jnp.asarray([e.metrics[k] for k in outputs]), jnp.all(
            jnp.asarray([r.converged for r in e.reports.values()])
        )

    _, derivatives, derivative_ok = _directional_jacobian(
        vector, mode=derivative_mode, batch_size=derivative_batch_size, recorder=recorder
    )(x)
    if release_caches:
        jax.clear_caches()
        gc.collect()
    rows, runs = [], [baseline]
    for col, name in enumerate(names):
        p = runner.parameters[name]
        center = float(values[name])
        scale = max(
            abs(center),
            float(p.upper - p.lower) if p.lower is not None and p.upper is not None else 1,
            1,
        )
        step = relative_step * scale
        if p.lower is not None:
            step = min(step, (center - p.lower) / 2)
        if p.upper is not None:
            step = min(step, (p.upper - center) / 2)
        if step <= max(abs(center), 1) * 1e-12:
            rows.append(
                {
                    "parameter": name,
                    "accepted": False,
                    "reason": "No centered finite-difference interval inside parameter bounds.",
                }
            )
            continue
        pair = []
        error = None
        for sign in (-1, 1):
            try:
                run = runner.run(_overrides(runner, {**values, name: center + sign * step}))
                pair.append(run)
                runs.append(run)
                if workspace is not None:
                    workspace.save_run(run)
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                error = str(exc)
                break
        phase_ok = (
            len(pair) == 2
            and all(_phase_signature(r) == _phase_signature(baseline) for r in pair)
            and "unknown" not in _phase_signature(baseline).values()
        )
        accepted = len(pair) == 2 and all(r.accepted for r in pair) and phase_ok and derivative_ok
        comparisons = {}
        for row, metric in enumerate(outputs):
            ad = float(derivatives[row, col])
            fd = None
            if len(pair) == 2:
                lo, hi = (r.to_dict()["metrics"][metric]["value_si"] for r in pair)
                if lo is not None and hi is not None:
                    fd = (hi - lo) / (2 * step)
            baseline_value = baseline.to_dict()["metrics"][metric]["value_si"]
            floor = max(abs(baseline_value), 1) / scale * 1e-6
            disagreement = (
                abs(ad - fd) / max(abs(ad), abs(fd), floor)
                if fd is not None and math.isfinite(ad)
                else None
            )
            valid = accepted and disagreement is not None and disagreement <= relative_tolerance
            metric_unit = runner.document["metrics"][metric]["unit"]
            conversion = unit_for(p.unit).factor / unit_for(metric_unit).factor
            comparisons[metric] = {
                "accepted": valid,
                "autodiff_si": ad,
                "finite_difference_si": fd,
                "relative_error": disagreement,
                "autodiff_display": ad * conversion,
                "derivative_unit": f"({metric_unit})/({p.unit} increment)",
            }
        rows.append(
            {
                "parameter": name,
                "accepted": accepted and all(c["accepted"] for c in comparisons.values()),
                "step_si": step,
                "phase_regime_unchanged": phase_ok,
                "perturbed_run_ids": [r.run_id for r in pair],
                "error": error,
                "metrics": comparisons,
            }
        )
    result = StudyResult(
        sealed(
            "sensitivities",
            {
                "case_id": runner.case.case_id,
                "baseline_id": baseline.run_id,
                "initialization": "accepted_baseline; finite-difference runs start cold",
                "accepted": all(r["accepted"] for r in rows),
                "relative_step": relative_step,
                "relative_tolerance": relative_tolerance,
                "derivative_numerically_accepted": derivative_ok,
                "derivatives": strategy,
                "release_caches": release_caches,
                "parameters": list(names),
                "metrics": list(outputs),
                "results": rows,
                "scope": (
                    "Local implicit derivatives checked against centered differences; "
                    "no global smoothness or uncertainty claim."
                ),
            },
        ),
        tuple(runs),
    )
    if workspace is not None:
        result.save(workspace)
    return result


def _constraints(runner: CaseRunner, constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for i, raw in enumerate(sequence(constraints, "constraints", maximum=100)):
        path = f"constraints[{i}]"
        d = object_fields(
            raw,
            path,
            allowed={"metric", "lower", "upper", "equal", "tolerance"},
            required={"metric", "tolerance"},
        )
        _names(runner, [d["metric"]], metrics=True)
        dim = unit_for(runner.document["metrics"][d["metric"]]["unit"]).dimension
        sides = {k for k in ("lower", "upper", "equal") if k in d}
        if not sides or ("equal" in sides and len(sides) != 1):
            raise ValueError("constraint needs lower/upper bounds or one equality")
        tol = quantity(d["tolerance"], dim, path + ".tolerance", difference=dim == TEMPERATURE)
        if tol <= 0:
            raise ValueError("constraint tolerance must be positive")
        bounds = {
            k: quantity(
                d[k],
                dim,
                path + "." + k,
                difference=runner.document["metrics"][d["metric"]]["unit"].startswith("delta_"),
            )
            for k in sides
        }
        if bounds.get("lower", -math.inf) > bounds.get("upper", math.inf):
            raise ValueError("constraint lower exceeds upper")
        result.extend(
            {"metric": d["metric"], "side": k, "target": v, "tolerance": tol}
            for k, v in bounds.items()
        )
    return result


def optimize(
    runner: CaseRunner,
    variables: list[str],
    objective: str,
    *,
    sense: str = "min",
    constraints: list[dict[str, Any]] | None = None,
    overrides: dict[str, Any] | None = None,
    max_iterations: int = 100,
    release_caches: bool = False,
    derivative_mode: str = "auto",
    derivative_batch_size: int = 1,
    workspace: CaseWorkspace | None = None,
    recorder: PerformanceRecorder | None = None,
) -> StudyResult:
    """Solve a bounded local constrained design using SLSQP and exact JAX derivatives.

    Variables use their declared bounds. Trial points get numerical checks;
    baseline and final points additionally receive independent physical audits.
    Success requires optimizer termination, feasibility, finite derivatives, and
    accepted final physics. A failed final candidate is retained, never promoted.
    Screening economics and local optimization don't imply a global optimum.
    An optional ``recorder`` measures each local linearization and Jacobian
    application without adding timing fields to the study artifact.
    """
    from scipy.optimize import minimize

    names = _names(runner, variables)
    if not isinstance(release_caches, bool):
        raise ValueError("release_caches must be a boolean")
    _names(runner, [objective], metrics=True)
    if sense not in ("min", "max"):
        raise ValueError("sense must be min or max")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or not 1 <= max_iterations <= 1000
    ):
        raise ValueError("max_iterations must be from one to 1000")
    for k in names:
        p = runner.parameters[k]
        if p.lower is None or p.upper is None or p.lower >= p.upper:
            raise ValueError("every optimization variable needs finite distinct bounds")
    limits = _constraints(runner, constraints or [])
    strategy = derivative_strategy(
        len(names), 1 + len(limits), mode=derivative_mode, batch_size=derivative_batch_size
    )
    baseline = runner.run(overrides)
    if workspace is not None:
        workspace.save_run(baseline)
    if not baseline.accepted:
        return _failed_baseline(
            "optimization",
            runner,
            baseline,
            {
                "variables": variables,
                "objective": objective,
                "sense": sense,
                "constraints": constraints or [],
                "max_iterations": max_iterations,
                "release_caches": release_caches,
                "derivative_mode": derivative_mode,
                "derivative_batch_size": derivative_batch_size,
            },
            workspace,
        )
    initialization = runner.initialization(baseline)
    values = runner.parameter_values(overrides)
    lower = jnp.asarray([runner.parameters[k].lower for k in names])
    span = jnp.asarray([runner.parameters[k].upper for k in names]) - lower
    x0 = np.asarray((jnp.asarray([values[k] for k in names]) - lower) / span)
    sign = 1 if sense == "min" else -1

    def metric_scale(name: str, value: float) -> float:
        # Annual costs are stored as USD/s, often much smaller than one. An
        # arbitrary 1 SI-unit floor can make a profitable move look stationary.
        return max(abs(value), unit_for(runner.document["metrics"][name]["unit"]).factor, 1e-12)

    scales = jnp.asarray(
        [
            metric_scale(objective, baseline.to_dict()["metrics"][objective]["value_si"]),
            *(metric_scale(c["metric"], c["target"]) for c in limits),
        ]
    )
    history: list[dict[str, Any]] = []
    if release_caches:
        jax.clear_caches()
        gc.collect()

    def evaluate(x: Any) -> tuple[Any, Any]:
        e = runner.evaluate(
            {**values, **dict(zip(names, lower + span * x, strict=True))},
            initialization=initialization,
        )
        vector = (
            jnp.asarray([sign * e.metrics[objective], *(e.metrics[c["metric"]] for c in limits)])
            / scales
        )
        ok = jnp.all(jnp.asarray([r.converged for r in e.reports.values()])) & jnp.all(
            jnp.isfinite(vector)
        )
        return vector, ok

    # Host SLSQP keeps the optimizer out of the nested unit-solver trace. The
    # differentiated objective still uses the kernels' implicit Jacobians.
    jacobian = _directional_jacobian(
        evaluate, mode=derivative_mode, batch_size=derivative_batch_size, recorder=recorder
    )
    cached_x: np.ndarray | None = None
    cached: tuple[np.ndarray, np.ndarray] | None = None

    def point(x: Any) -> tuple[np.ndarray, np.ndarray]:
        nonlocal cached_x, cached
        x = np.asarray(x)
        if cached_x is not None and np.array_equal(x, cached_x) and cached is not None:
            return cached
        valid, reason = False, None
        try:
            runner.validate_values({**values, **dict(zip(names, lower + span * x, strict=True))})
            value, jac, ok = jacobian(jnp.asarray(x))
            v, j = np.asarray(value), np.asarray(jac)
            valid = bool(ok) and bool(np.all(np.isfinite(v))) and bool(np.all(np.isfinite(j)))
            if not valid:
                reason = "Numerical solve or derivative failed."
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            reason = str(exc)
        finally:
            if release_caches:
                # The local map is gone and its results are host arrays. Drop
                # unreachable Python cycles, but retain compiled unit kernels:
                # clearing JAX's caches here forces expensive recompilation at
                # the next point and can increase its compiler memory peak.
                gc.collect()
        history.append(
            {
                "normalized_variables": x.tolist(),
                "numerically_accepted": bool(valid),
                "reason": reason,
            }
        )
        if not valid:
            # A finite rejection barrier points back toward the accepted seed.
            # It is never eligible as a successful design or a reported metric.
            distance = x - x0
            v = np.full(1 + len(limits), 1e6 + float(distance @ distance))
            j = np.tile(2 * distance, (len(v), 1))
        cached_x, cached = x.copy(), (v, j)
        return cached

    scipy_constraints = []
    for i, c in enumerate(limits, 1):
        target = c["target"] / float(scales[i])
        orientation = -1 if c["side"] == "upper" else 1

        def fun(x: Any, index: int = i, goal: float = target, direction: int = orientation) -> Any:
            return direction * (point(x)[0][index] - goal)

        def jac(x: Any, index: int = i, direction: int = orientation) -> Any:
            return direction * point(x)[1][index]

        scipy_constraints.append(
            {"type": "eq" if c["side"] == "equal" else "ineq", "fun": fun, "jac": jac}
        )
    solved = minimize(
        lambda x: point(x)[0][0],
        x0,
        jac=lambda x: point(x)[1][0],
        bounds=[(0.0, 1.0)] * len(names),
        constraints=scipy_constraints,
        method="SLSQP",
        options={"maxiter": max_iterations, "ftol": 1e-10},
    )
    point(solved.x)
    final_numerical = history[-1]["numerically_accepted"]
    final_values = {**values, **dict(zip(names, lower + span * solved.x, strict=True))}
    if release_caches:
        jax.clear_caches()
        gc.collect()
    try:
        candidate = runner.run(_overrides(runner, final_values))
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        result = StudyResult(
            sealed(
                "optimization",
                {
                    "case_id": runner.case.case_id,
                    "accepted": False,
                    "baseline_id": baseline.run_id,
                    "candidate_id": None,
                    "derivatives": strategy,
                    "variables": list(names),
                    "objective": objective,
                    "sense": sense,
                    "release_caches": release_caches,
                    "constraints": constraints or [],
                    "history": history,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "optimizer": {"success": bool(solved.success), "message": str(solved.message)},
                },
            ),
            (baseline,),
        )
        if workspace is not None:
            result.save(workspace)
        return result
    feasibility = []
    for c in limits:
        value = candidate.to_dict()["metrics"][c["metric"]]["value_si"]
        violation = (
            math.inf
            if value is None
            else abs(value - c["target"])
            if c["side"] == "equal"
            else max(c["target"] - value, 0)
            if c["side"] == "lower"
            else max(value - c["target"], 0)
        )
        feasibility.append(
            {**c, "violation_si": violation, "accepted": violation <= c["tolerance"]}
        )
    final_derivatives = np.asarray(solved.jac)
    # Re-solving the final candidate cold makes a seed-dependent branch change
    # visible. Numerical trial metrics cannot silently replace the audited run.
    trial_metrics = point(solved.x)[0] * np.asarray(scales)
    trial_metrics[0] *= sign
    initialization_checks = []
    for i, metric in enumerate([objective, *(c["metric"] for c in limits)]):
        cold = candidate.to_dict()["metrics"][metric]["value_si"]
        warm = float(trial_metrics[i])
        tolerance = 1e-6 * float(scales[i])
        initialization_checks.append(
            {
                "metric": metric,
                "trial_si": warm,
                "cold_start_si": cold,
                "tolerance_si": tolerance,
                "accepted": cold is not None and abs(warm - cold) <= tolerance,
            }
        )
    valid = (
        bool(solved.success)
        and final_numerical
        and candidate.accepted
        and all(c["accepted"] for c in feasibility)
        and bool(np.all(np.isfinite(final_derivatives)))
        and all(c["accepted"] for c in initialization_checks)
    )
    result = StudyResult(
        sealed(
            "optimization",
            {
                "case_id": runner.case.case_id,
                "accepted": valid,
                "baseline_id": baseline.run_id,
                "candidate_id": candidate.run_id,
                "derivatives": strategy,
                "initialization": "accepted_baseline; final candidate starts cold",
                "initialization_checks": initialization_checks,
                "variables": list(names),
                "objective": objective,
                "sense": sense,
                "release_caches": release_caches,
                "constraints": constraints or [],
                "optimizer": {
                    "method": "SLSQP",
                    "success": bool(solved.success),
                    "status": int(solved.status),
                    "message": str(solved.message),
                    "iterations": int(solved.nit),
                    "max_iterations": max_iterations,
                },
                "feasibility": feasibility,
                "history": history,
                "comparison": compare_runs(baseline, candidate),
                "scope": (
                    "Local constrained optimum candidate; only final and baseline "
                    "points receive full physical audits. No global optimality claim."
                ),
            },
        ),
        (baseline, candidate),
    )
    if workspace is not None:
        result.save(workspace)
    return result
