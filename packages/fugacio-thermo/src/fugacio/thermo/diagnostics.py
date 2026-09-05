"""Numerical solve reports shared by thermodynamics and process simulation.

Reports contain only arrays and are JAX pytrees. Host applications can raise a
descriptive exception with :func:`require_converged`; compiled applications
inspect ``report.converged`` and retain the report alongside their results.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class SolveStatus(IntEnum):
    """Stable, machine-readable termination codes."""

    CONVERGED = 0
    MAX_ITERATIONS = 1
    NONFINITE = 2
    SINGULAR = 3
    STALLED = 4
    INVALID_INPUT = 5
    INFEASIBLE = 6


class SolveReport(NamedTuple):
    """Termination information for a nonlinear or linear calculation.

    Attributes:
        status: A :class:`SolveStatus` code.
        iterations: Number of attempted iterations.
        residual_norm: Maximum absolute scaled residual.
        step_norm: Maximum absolute scaled last step.
        worst_equation: Index of the largest scaled residual, or -1 if absent.
    """

    status: Array
    iterations: Array
    residual_norm: Array
    step_norm: Array
    worst_equation: Array

    @property
    def converged(self) -> Array:
        """Whether the residual passed its tolerance and is finite."""
        return (self.status == SolveStatus.CONVERGED) & jnp.isfinite(self.residual_norm)

    def to_dict(self) -> dict[str, Any]:
        """Return a strict-JSON-compatible report for a concrete calculation."""
        import math

        status = SolveStatus(int(self.status))
        residual = float(self.residual_norm)
        step = float(self.step_norm)
        return {
            "converged": bool(self.converged),
            "status": status.name.lower(),
            "iterations": int(self.iterations),
            "residual_norm": residual if math.isfinite(residual) else None,
            "step_norm": step if math.isfinite(step) else None,
            "worst_equation": int(self.worst_equation),
        }


class SolveResult(NamedTuple):
    """A best available solution and its independently checked solve report."""

    value: Array
    report: SolveReport


class ConvergenceError(RuntimeError):
    """A failed calculation, with its machine-readable report attached."""

    def __init__(
        self, report: SolveReport, context: str = "calculation", labels: tuple[str, ...] = ()
    ) -> None:
        self.report = report
        self.context = context
        index = int(report.worst_equation)
        equation = labels[index] if 0 <= index < len(labels) else str(index)
        status = SolveStatus(int(report.status)).name.lower()
        super().__init__(
            f"{context} failed: {status} after {int(report.iterations)} iterations; "
            f"scaled residual {float(report.residual_norm):.6g}, equation {equation}"
        )


def require_converged(
    report: SolveReport, context: str = "calculation", labels: tuple[str, ...] = ()
) -> None:
    """Raise for a concrete failed report; leave traced reports to compiled callers.

    This function performs no callbacks or side effects inside JAX transforms.
    Differentiable solver values use nonfinite derivatives on failed solves;
    compiled callers should also inspect the returned report explicitly.

    Raises:
        ConvergenceError: If a concrete report indicates failure.
    """
    if isinstance(report.status, jax.core.Tracer):
        return
    if not bool(report.converged):
        raise ConvergenceError(report, context, labels)


def residual_report(
    residual: Array,
    tol: float = 1e-8,
    *,
    iterations: Array | int = 0,
    step_norm: Array | float = 0.0,
    failure: SolveStatus = SolveStatus.MAX_ITERATIONS,
) -> SolveReport:
    """Grade a scaled residual independently of an iteration's stopping rule."""
    flat = jnp.ravel(jnp.asarray(residual, dtype=float))
    norm = jnp.max(jnp.abs(flat), initial=0.0)
    finite = jnp.all(jnp.isfinite(flat))
    status = jnp.where(
        finite, jnp.where(norm <= tol, SolveStatus.CONVERGED, failure), SolveStatus.NONFINITE
    )
    worst = jnp.argmax(jnp.abs(flat)) if flat.size else jnp.asarray(-1)
    return jax.lax.stop_gradient(
        SolveReport(status, jnp.asarray(iterations), norm, jnp.asarray(step_norm), worst)
    )
