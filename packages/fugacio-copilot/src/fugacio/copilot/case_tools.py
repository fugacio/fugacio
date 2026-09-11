"""Case-building tools and accountable design submission over trusted run artifacts."""

from __future__ import annotations

from typing import Any

from fugacio.sim.cases import CaseRunner, CaseWorkspace, ProcessCase, SolverOptions, compare_runs
from fugacio.sim.cases.examples import EXAMPLES, example_case
from fugacio.sim.cases.profiling import profile
from fugacio.sim.cases.quantities import UNITS
from fugacio.sim.cases.registry import registry_schema
from fugacio.sim.cases.studies import optimize, sensitivities, sweep


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def case_format() -> dict[str, Any]:
    """Return units, registry capabilities, and a complete editable example."""
    return {
        "schema_version": 1,
        "units": list(UNITS),
        "unit_types": registry_schema(),
        "example": example_case("heater").to_dict(),
        "examples": list(EXAMPLES),
        "reaction_example": example_case("reactive-recycle").to_dict(),
        "rules": [
            "Named reaction_sets declare phase and activity or normalized_concentration inputs. "
            "Kinetic coefficients use mol/(m3 s), activation energies J/mol, "
            "and explicit reference K.",
            "Extent metrics require a reaction name; generation metrics require a component. "
            "Column reaction_volumes declare one reacting-phase volume in m3 per stage.",
            "Kinetic assumptions aren't empirical qualification. Use the retained phase, balance, "
            "integration, and solver reports when assessing a reactive design.",
            "Dimensional values require {value, unit}; parameter references use {parameter: name}.",
            (
                "Feeds and unit outlets have one producer and at most one "
                "consumer. Use a splitter for branches."
            ),
            (
                "Metrics use dimension-checked arithmetic over streams, retained "
                "unit results, parameters, or plant totals."
            ),
            (
                "A case revision is immutable. Updating it creates a new case_id "
                "and makes prior runs stale for submission."
            ),
            (
                "Run acceptance and empirical qualification are separate. Unknown "
                "validation ranges remain unknown."
            ),
        ],
    }


def _validate(case: dict[str, Any]) -> dict[str, Any]:
    parsed = ProcessCase.from_dict(case)
    runner = CaseRunner(parsed)
    return {
        "valid": True,
        "case_id": parsed.case_id,
        "case": parsed.to_dict(),
        "parameter_evidence": runner.package.evidence.to_dict(),
        "qualification": runner.qualification,
    }


def _example(name: str) -> dict[str, Any]:
    case = example_case(name)
    return {"case_id": case.case_id, "case": case.to_dict()}


def _solve(
    case: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    backend: str = "sequential",
    column_solver: str = "block",
    eo_jacobian: str = "colored",
) -> dict[str, Any]:
    return (
        CaseRunner(
            ProcessCase.from_dict(case),
            options=SolverOptions(
                backend=backend,
                recycle_method="broyden",
                column_solver=column_solver,
                eo_jacobian=eo_jacobian,
            ),
        )
        .run(overrides)
        .to_dict()
    )


def case_tool_specs() -> list[Any]:
    """Stateless schema, validation, and audited solve tools for the general copilot."""
    from fugacio.copilot.tools import ToolSpec

    return [
        ToolSpec(
            "case_format",
            "Get the portable case vocabulary, unit registry, and complete example.",
            _schema({}, []),
            case_format,
        ),
        ToolSpec(
            "case_example",
            "Get a complete editable example, including columns, recycles, and measured evidence.",
            _schema({"name": {"type": "string", "enum": list(EXAMPLES)}}, ["name"]),
            _example,
        ),
        ToolSpec(
            "validate_case",
            "Validate a complete process case and its property-package evidence.",
            _schema({"case": {"type": "object"}}, ["case"]),
            _validate,
        ),
        ToolSpec(
            "solve_case",
            "Run a complete process case; inspect independent acceptance and qualification fields.",
            _schema(
                {
                    "case": {"type": "object"},
                    "overrides": {"type": "object"},
                    "backend": {"type": "string", "enum": ["sequential", "eo"]},
                    "column_solver": {"type": "string", "enum": ["block", "dense"]},
                    "eo_jacobian": {"type": "string", "enum": ["colored", "dense"]},
                },
                ["case"],
            ),
            _solve,
        ),
    ]


class DesignSession:
    """Case revision and trusted run state for one accountable design-agent session.

    Disk artifacts are inspectable, but only runs computed in this session are
    eligible for submission. A hash verifies integrity, not authorship. The
    explicit workspace is the only storage surface exposed to model tools.
    """

    def __init__(self, workspace: CaseWorkspace) -> None:
        self.workspace = workspace
        self.current_case_id: str | None = None
        self.trusted_runs: set[str] = set()
        self.pending: dict[str, Any] | None = None
        self._runners: dict[tuple[str, str, str, str], CaseRunner] = {}

    def create_case(self, case: dict[str, Any]) -> dict[str, Any]:
        """Validate and select a new immutable case revision."""
        parsed = ProcessCase.from_dict(case)
        runner = CaseRunner(parsed, options=SolverOptions(recycle_method="broyden"))
        identity = self.workspace.save_case(parsed)
        self._runners[(identity, "sequential", "block", "colored")] = runner
        self.current_case_id, self.pending = identity, None
        return {"case_id": identity, "case": parsed.to_dict(), "selected": True}

    def load_case(self, case_id: str) -> dict[str, Any]:
        """Select a verified saved revision; prior submission eligibility is cleared."""
        return self.create_case(self.workspace.load_case(case_id).to_dict())

    def update_case(self, base_case_id: str, case: dict[str, Any]) -> dict[str, Any]:
        """Apply a complete reviewed replacement only to the current base revision."""
        if base_case_id != self.current_case_id:
            raise ValueError("stale base_case_id; inspect the current case before updating")
        result = self.create_case(case)
        result["previous_case_id"] = base_case_id
        return result

    def set_parameters(self, case_id: str, overrides: dict[str, Any]) -> dict[str, Any]:
        """Create a new revision with dimension-checked parameter defaults."""
        if case_id != self.current_case_id:
            raise ValueError("parameter update requires the current case revision")
        case = self.workspace.load_case(case_id).with_parameters(overrides)
        return self.update_case(case_id, case.to_dict())

    def _runner(
        self, case_id: str, backend: str, column_solver: str = "block", eo_jacobian: str = "colored"
    ) -> CaseRunner:
        if case_id != self.current_case_id:
            raise ValueError("select the intended case before running or studying it")
        key = (case_id, backend, column_solver, eo_jacobian)
        if key not in self._runners:
            self._runners[key] = CaseRunner(
                self.workspace.load_case(case_id),
                options=SolverOptions(
                    backend=backend,
                    recycle_method="broyden",
                    column_solver=column_solver,
                    eo_jacobian=eo_jacobian,
                ),
            )
        return self._runners[key]

    def run_case(
        self,
        case_id: str,
        overrides: dict[str, Any] | None = None,
        backend: str = "sequential",
        column_solver: str = "block",
        eo_jacobian: str = "colored",
    ) -> dict[str, Any]:
        """Compute, audit, and save a run, preserving failed calculation evidence."""
        run = self._runner(case_id, backend, column_solver, eo_jacobian).run(overrides)
        self.workspace.save_run(run)
        self.trusted_runs.add(run.run_id)
        self.pending = None
        return run.to_dict()

    def study_case(
        self,
        case_id: str,
        kind: str,
        request: dict[str, Any],
        backend: str = "sequential",
        column_solver: str = "block",
        eo_jacobian: str = "colored",
    ) -> dict[str, Any]:
        """Run a bounded sweep, optimization, or independently checked sensitivity study."""
        functions: dict[str, Any] = {
            "sweep": sweep,
            "optimization": optimize,
            "sensitivities": sensitivities,
            "profile": profile,
        }
        if kind not in functions or {"runner", "workspace", "recorder"} & request.keys():
            raise ValueError("invalid study kind or reserved request argument")
        result = functions[kind](
            self._runner(case_id, backend, column_solver, eo_jacobian),
            workspace=self.workspace,
            **request,
        )
        self.trusted_runs.update(run.run_id for run in result.runs)
        self.pending = None
        return result.artifact

    def diagnose_case(self, case_id: str) -> dict[str, Any]:
        """Inspect topology and declared dependencies without granting run eligibility."""
        return self._runner(case_id, "sequential").diagnose_structure()

    def inspect(self, artifact_id: str) -> dict[str, Any]:
        """Inspect a verified artifact without granting it trusted execution status."""
        return {
            "artifact": self.workspace.load_artifact(artifact_id),
            "computed_in_session": artifact_id in self.trusted_runs,
            "current_case_id": self.current_case_id,
        }

    def submit_design(
        self, run_id: str, metrics: list[str], baseline_id: str | None = None
    ) -> dict[str, Any]:
        """Submit selected recorded metrics from an accepted current-case run.

        Model-authored numerical claims and prose aren't accepted as arguments.
        The report is generated deterministically from the stored artifact.
        """
        if run_id not in self.trusted_runs:
            raise ValueError(
                "submission requires a run computed in this session; rerun saved designs"
            )
        run = self.workspace.load_run(run_id)
        d = run.to_dict()
        if d["case_id"] != self.current_case_id:
            raise ValueError("run belongs to a stale or different case revision")
        run.check()
        if (
            not isinstance(metrics, list)
            or not metrics
            or not all(isinstance(k, str) and k in d["metrics"] for k in metrics)
            or len(set(metrics)) != len(metrics)
        ):
            raise ValueError("submit unique metric names recorded in this run")
        comparison = None
        if any(d["metrics"][k]["source"] != "process_expression" for k in metrics):
            raise ValueError(
                "submission metrics must measure the process; "
                "literal inputs aren't computed performance"
            )
        if baseline_id is not None:
            if baseline_id not in self.trusted_runs:
                raise ValueError("baseline must also be computed in this session")
            baseline = self.workspace.load_run(baseline_id)
            baseline.check()
            if baseline.to_dict()["case"]["name"] != d["case"]["name"]:
                raise ValueError("baseline belongs to a different named process")
            comparison = compare_runs(baseline, run)
        result = {
            "submitted": True,
            "case_id": d["case_id"],
            "run_id": run_id,
            "metrics": {k: d["metrics"][k] for k in metrics},
            "comparison": comparison,
            "qualification": d["qualification"],
            **({"reaction_evidence": d["reaction_evidence"]} if "reaction_evidence" in d else {}),
            "report": run.markdown(),
        }
        self.pending = result
        return result

    def registry(self) -> dict[str, Any]:
        """Create tools bound to this session's explicit workspace and revision state."""
        from fugacio.copilot.tools import ToolSpec

        string, obj = {"type": "string"}, {"type": "object"}
        specs = case_tool_specs()[:2]
        definitions: list[tuple[str, str, dict[str, Any], list[str], Any]] = [
            (
                "create_case",
                "Create and select a portable process case.",
                {"case": obj},
                ["case"],
                self.create_case,
            ),
            (
                "load_case",
                "Select a saved case revision.",
                {"case_id": string},
                ["case_id"],
                self.load_case,
            ),
            (
                "list_cases",
                "List saved case revisions.",
                {},
                [],
                lambda: {
                    "cases": self.workspace.list_cases(),
                    "current_case_id": self.current_case_id,
                },
            ),
            (
                "update_case",
                "Replace the current revision with a complete validated case.",
                {"base_case_id": string, "case": obj},
                ["base_case_id", "case"],
                self.update_case,
            ),
            (
                "set_case_parameters",
                "Set explicit parameter quantities, creating a new case revision.",
                {"case_id": string, "overrides": obj},
                ["case_id", "overrides"],
                self.set_parameters,
            ),
            (
                "run_case",
                "Solve, audit, and save the current case.",
                {
                    "case_id": string,
                    "overrides": obj,
                    "backend": {"type": "string", "enum": ["sequential", "eo"]},
                    "column_solver": {"type": "string", "enum": ["block", "dense"]},
                    "eo_jacobian": {"type": "string", "enum": ["colored", "dense"]},
                },
                ["case_id"],
                self.run_case,
            ),
            (
                "study_case",
                "Run a reproducible study. Request accepts derivative_mode (auto/forward/reverse) "
                "and derivative_batch_size for sensitivities and optimization.",
                {
                    "case_id": string,
                    "kind": {
                        "type": "string",
                        "enum": ["sweep", "optimization", "sensitivities", "profile"],
                    },
                    "request": obj,
                    "backend": string,
                    "column_solver": {"type": "string", "enum": ["block", "dense"]},
                    "eo_jacobian": {"type": "string", "enum": ["colored", "dense"]},
                },
                ["case_id", "kind", "request"],
                self.study_case,
            ),
            (
                "diagnose_case",
                "Inspect process topology, column sizes, and structural matching without solving.",
                {"case_id": string},
                ["case_id"],
                self.diagnose_case,
            ),
            (
                "inspect_case_artifact",
                "Inspect an immutable run or study and its evidence.",
                {"artifact_id": string},
                ["artifact_id"],
                self.inspect,
            ),
            (
                "submit_design",
                (
                    "Submit recorded metrics from an accepted current-case run; this "
                    "is the only completion tool."
                ),
                {
                    "run_id": string,
                    "metrics": {"type": "array", "items": string, "minItems": 1},
                    "baseline_id": string,
                },
                ["run_id", "metrics"],
                self.submit_design,
            ),
        ]
        specs.extend(
            ToolSpec(name, description, _schema(properties, required), function)
            for name, description, properties, required, function in definitions
        )
        return {s.name: s for s in specs}
