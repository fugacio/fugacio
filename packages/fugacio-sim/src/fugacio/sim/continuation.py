"""Adaptive operating-condition continuation with accepted-state warm starts.

Continuation is a host orchestration algorithm. Each point can use a compiled,
implicitly differentiable solver. Failed trials never replace the accepted seed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from fugacio.thermo.diagnostics import ConvergenceError, SolveReport, SolveStatus, require_converged


class ContinuationStep(NamedTuple):
    """One attempted point on the path, including rejected trials."""

    progress: float
    accepted: bool
    report: SolveReport


@dataclass(frozen=True)
class ContinuationResult:
    """Last accepted solution and the complete attempted path.

    ``progress == 1`` and a converged report are both required for success.
    A failed path's ``value`` belongs to its last accepted operating point.
    """

    value: Any
    progress: float
    report: SolveReport
    steps: tuple[ContinuationStep, ...]

    def check(self) -> None:
        """Raise if the requested endpoint wasn't reached."""
        require_converged(self.report, f"continuation at progress {self.progress:g}")


def continuation_solve(
    solve: Callable[[Any, Any], tuple[Any, SolveReport]],
    start: Any,
    target: Any,
    *,
    guess: Any = None,
    initial_step: float = 0.25,
    min_step: float = 1e-4,
    max_steps: int = 100,
    check: bool = True,
) -> ContinuationResult:
    """Move between parameter pytrees, reducing the step after a failed solve.

    ``solve(parameters, previous_value)`` returns a value and its report. It
    should return failed reports or raise ``ConvergenceError`` for numerical
    failures. Other exceptions propagate, since programming errors aren't
    convergence failures. The final solve supplies the endpoint's own implicit
    derivative; no derivative is defined through the adaptive path selection.

    Raises:
        ValueError: If path controls or parameter tree structures are invalid.
        ConvergenceError: If ``check`` is true and the endpoint can't be reached.
    """
    if not 0 < min_step <= initial_step <= 1 or max_steps < 1:
        raise ValueError("require 0 < min_step <= initial_step <= 1 and max_steps >= 1")
    # Some jaxlib wheels don't expose PyTreeDef's comparison types to mypy.
    start_structure: object = jax.tree_util.tree_structure(start)
    target_structure: object = jax.tree_util.tree_structure(target)
    if start_structure != target_structure:
        raise ValueError("start and target must have matching parameter trees")
    if any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves((start, target))):
        raise ValueError("continuation runs on the host; differentiate the endpoint solve")
    steps: list[ContinuationStep] = []
    value, progress, step = guess, 0.0, initial_step
    trial_progress = 0.0
    report: SolveReport
    for _ in range(max_steps):
        params = jax.tree_util.tree_map(
            lambda a, b, progress=trial_progress: (
                jnp.asarray(a) + progress * (jnp.asarray(b) - jnp.asarray(a))
            ),
            start,
            target,
        )
        try:
            candidate, report = solve(params, value)
        except ConvergenceError as exc:
            candidate, report = value, exc.report
        accepted = bool(report.converged)
        steps.append(ContinuationStep(trial_progress, accepted, report))
        if accepted:
            value, progress = candidate, trial_progress
            if progress == 1.0:
                break
            step = min(step * (1.5 if progress > 0 else 1.0), 1.0 - progress)
        else:
            if trial_progress == 0.0:
                break
            step *= 0.5
            if step < min_step:
                break
        trial_progress = min(progress + step, 1.0)
    if progress < 1.0 and bool(report.converged):
        report = report._replace(status=jnp.asarray(SolveStatus.MAX_ITERATIONS))
    result = ContinuationResult(value, progress, report, tuple(steps))
    if check:
        result.check()
    return result
