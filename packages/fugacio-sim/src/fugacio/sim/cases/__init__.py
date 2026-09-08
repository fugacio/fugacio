"""Portable process cases, independently audited runs, and reproducible studies."""

from fugacio.sim.cases.quantities import CaseValidationError
from fugacio.sim.cases.registry import registry_schema
from fugacio.sim.cases.results import CaseAcceptanceError, CaseRun, compare_runs
from fugacio.sim.cases.runtime import CaseEvaluation, CaseInitialization, CaseRunner, SolverOptions
from fugacio.sim.cases.schema import ProcessCase
from fugacio.sim.cases.studies import StudyResult, optimize, sensitivities, sweep
from fugacio.sim.cases.workspace import CaseWorkspace

__all__ = [
    "CaseAcceptanceError",
    "CaseEvaluation",
    "CaseInitialization",
    "CaseRun",
    "CaseRunner",
    "CaseValidationError",
    "CaseWorkspace",
    "ProcessCase",
    "SolverOptions",
    "StudyResult",
    "compare_runs",
    "optimize",
    "registry_schema",
    "sensitivities",
    "sweep",
]
