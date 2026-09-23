"""Command-line interface for portable cases and reproducible design studies."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from fugacio.sim.cases.examples import EXAMPLES, example_case
from fugacio.sim.cases.jsonio import read_json, write_json
from fugacio.sim.cases.profiling import profile
from fugacio.sim.cases.registry import registry_schema
from fugacio.sim.cases.results import compare_runs, render_artifact
from fugacio.sim.cases.runtime import CaseRunner, SolverOptions
from fugacio.sim.cases.schema import ProcessCase
from fugacio.sim.cases.studies import optimize, sensitivities, sweep
from fugacio.sim.cases.workspace import CaseWorkspace
from fugacio.sim.compilation import enable_compilation_cache

_CACHE_HELP = (
    "persist compiled kernels in DIR so later runs skip compilation "
    "(default: the FUGACIO_JAX_CACHE environment variable, if set)"
)


def main(argv: list[str] | None = None) -> int:
    """Run a CLI command; return 0 on success, 1 for input errors, or 2 for failed checks."""
    parser = argparse.ArgumentParser(prog="fugacio", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    example = commands.add_parser("example", help="export a built-in case as JSON")
    example.add_argument("name", choices=EXAMPLES)
    example.add_argument("output")
    commands.add_parser("registry", help="describe supported portable unit types")
    validate = commands.add_parser("validate", help="validate a case without solving")
    validate.add_argument("case")
    diagnose = commands.add_parser(
        "diagnose", help="inspect declared process structure without solving"
    )
    diagnose.add_argument("case")
    for name in ("run", "sweep", "optimize", "sensitivities", "profile"):
        cmd = commands.add_parser(name)
        cmd.add_argument("case")
        if name != "run":
            cmd.add_argument("request", help="JSON study arguments")
        cmd.add_argument("--backend", choices=("sequential", "eo"), default="sequential")
        cmd.add_argument("--column-solver", choices=("block", "dense"), default="block")
        cmd.add_argument("--plant-solver", choices=("sparse", "dense"), default="sparse")
        cmd.add_argument(
            "--recycle-method", choices=("wegstein", "broyden", "newton"), default="broyden"
        )
        cmd.add_argument("--overrides", help="JSON parameter quantities (run only)")
        cmd.add_argument("--workspace", default=".fugacio-cases")
        cmd.add_argument("--jax-cache", metavar="DIR", help=_CACHE_HELP)
        cmd.add_argument("--output", help="write the result artifact as JSON")
        if name == "run":
            cmd.add_argument("--report", help="write a Markdown engineering report")
    inspect = commands.add_parser("inspect", help="verify and inspect a run or study")
    inspect.add_argument("artifact_id")
    inspect.add_argument("--workspace", default=".fugacio-cases")
    inspect.add_argument(
        "--report", action="store_true", help="print a Markdown report instead of JSON"
    )
    compare = commands.add_parser("compare")
    compare.add_argument("baseline_id")
    compare.add_argument("candidate_id")
    compare.add_argument("--workspace", default=".fugacio-cases")
    replay = commands.add_parser("replay")
    replay.add_argument("run_id")
    replay.add_argument("--workspace", default=".fugacio-cases")
    replay.add_argument("--jax-cache", metavar="DIR", help=_CACHE_HELP)
    args = parser.parse_args(argv)
    try:
        cache = getattr(args, "jax_cache", None) or os.environ.get("FUGACIO_JAX_CACHE")
        if cache and hasattr(args, "jax_cache"):
            enable_compilation_cache(cache)
        result: Any
        if args.command == "example":
            case = example_case(args.name)
            case.save(args.output)
            result = {"case_id": case.case_id, "path": str(Path(args.output).resolve())}
        elif args.command == "registry":
            result = registry_schema()
        elif args.command == "validate":
            case = ProcessCase.load(args.case)
            CaseRunner(case)  # Validate property-package capability and inline evidence too.
            result = {"valid": True, "case_id": case.case_id, "name": case.name}
        elif args.command == "diagnose":
            result = CaseRunner(ProcessCase.load(args.case)).diagnose_structure()
        elif args.command == "inspect":
            result = CaseWorkspace(args.workspace).load_artifact(args.artifact_id)
            if args.report:
                print(render_artifact(result), end="")
                return 0
        elif args.command == "compare":
            workspace = CaseWorkspace(args.workspace)
            result = compare_runs(
                workspace.load_run(args.baseline_id), workspace.load_run(args.candidate_id)
            )
            workspace.save_artifact(result)
        elif args.command == "replay":
            result = CaseWorkspace(args.workspace).replay(args.run_id).to_dict()
        else:
            workspace = CaseWorkspace(args.workspace)
            runner = CaseRunner(
                ProcessCase.load(args.case),
                options=SolverOptions(
                    backend=args.backend,
                    recycle_method=args.recycle_method,
                    column_solver=args.column_solver,
                    plant_solver=args.plant_solver,
                ),
            )
            if args.command == "run":
                run = runner.run(read_json(args.overrides) if args.overrides else None)
                workspace.save_run(run)
                result = run.to_dict()
                if args.report:
                    Path(args.report).write_text(run.markdown(), encoding="utf-8")
            else:
                if args.overrides:
                    raise ValueError("put overrides in the study request")
                functions: dict[str, Any] = {
                    "sweep": sweep,
                    "optimize": optimize,
                    "sensitivities": sensitivities,
                    "profile": profile,
                }
                request = read_json(args.request)
                if (
                    not isinstance(request, dict)
                    or {"runner", "workspace", "recorder"} & request.keys()
                ):
                    raise ValueError("invalid study request")
                study = functions[args.command](runner, workspace=workspace, **request)
                result = study.artifact
            if args.output:
                write_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if isinstance(result, dict) and result.get("accepted") is False else 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc), "error_type": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
