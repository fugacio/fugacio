"""Run isolated process benchmarks with synchronized phases and resource limits.

Examples:
    uv run python scripts/benchmark_process.py --scenario column --stages 32
    uv run python scripts/benchmark_process.py --scenario depropanizer --study optimization
    uv run python scripts/benchmark_process.py --scenario heater-bank --variables 24

Each invocation starts a fresh worker. Cold mode creates a new persistent-cache
directory; warm mode requires an existing explicitly supplied cache directory.
The parent enforces elapsed time. A worker watchdog samples process peak RSS,
including XLA compilation, and stops work above the declared memory budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("heater", "column", "depropanizer", "ethanol-train", "heater-bank"),
        default="column",
    )
    parser.add_argument(
        "--study", choices=("profile", "sensitivities", "optimization"), default="profile"
    )
    parser.add_argument("--stages", type=int, default=16)
    parser.add_argument("--variables", type=int, default=24)
    parser.add_argument("--column-solver", choices=("block", "dense"), default="block")
    parser.add_argument("--eo-jacobian", choices=("colored", "dense"), default="colored")
    parser.add_argument("--backend", choices=("sequential", "eo"), default="sequential")
    parser.add_argument("--derivative-mode", choices=("auto", "forward", "reverse"), default="auto")
    parser.add_argument("--derivative-batch-size", type=int, default=1)
    parser.add_argument("--warm-repeats", type=int, default=2)
    parser.add_argument("--cache", choices=("cold", "warm"), default="cold")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/performance/run"))
    parser.add_argument("--max-rss-gb", type=float, default=7.0)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument(
        "--release-caches",
        action="store_true",
        help="release in-memory executables between audited study phases",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.warm_repeats <= 10 or any(
        not math.isfinite(v) or v <= 0 for v in (args.max_rss_gb, args.timeout_seconds)
    ):
        parser.error("positive resource limits and one to ten warm repeats are required")
    if not 2 <= args.stages <= 100 or not 2 <= args.variables <= 80:
        parser.error("stages must be 2..100 and variables 2..80")
    if args.derivative_batch_size < 1 or not 1 <= args.max_iterations <= 1000:
        parser.error("invalid derivative batch size or optimizer iteration cap")
    if args.scenario == "column" and args.study != "profile":
        parser.error(
            "standalone column mode profiles and checks its Jacobian; use a case for studies"
        )
    if args.release_caches and args.study == "profile":
        parser.error("--release-caches applies to sensitivities and optimization")
    return args


def _write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _watch_memory(output, budget):
    import resource

    while True:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = int(rss if sys.platform == "darwin" else rss * 1024)
        if rss > budget:
            _write(
                output / "resource-limit.json",
                {"reason": "memory_limit", "peak_rss_bytes": rss, "budget_bytes": budget},
            )
            os._exit(3)
        time.sleep(0.2)


def _cgroup_memory_events():
    # Linux can kill a worker before its RSS watchdog gets another timeslice.
    # These counters complement the worker's own high-water observations.
    try:
        lines = Path("/sys/fs/cgroup/memory.events").read_text().splitlines()
        return {key: int(value) for key, value in (line.split() for line in lines)}
    except (OSError, ValueError):
        return None


def _column(args, recorder):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from fugacio.sim import (
        ColumnFeed,
        Stream,
        distillate_rate,
        enthalpy_flow,
        reflux_ratio,
        rigorous_column,
    )
    from fugacio.thermo.sensitivity import linearize

    feed = Stream.from_fractions(
        ("benzene", "toluene"), jnp.array([0.5, 0.5]), 100.0, 365.0, 1.013e5
    )

    def kernel(x):
        return rigorous_column(
            [ColumnFeed(feed, max(1, args.stages // 2))],
            args.stages,
            p=x[1],
            specs=[reflux_ratio(x[0]), distillate_rate(50.0)],
            linear_solver=args.column_solver,
            check=False,
        )

    point = jnp.array([2.5, 1.013e5])
    result = recorder.compiled_kernel("column", kernel, (point,), repeats=args.warm_repeats)

    def metrics(x):
        column = kernel(x)
        return jnp.stack((column.reboiler_duty, column.distillate.z[0], column.t[-1]))

    def derivative(x):
        local = linearize(metrics, x)
        return local.value, local.jacobian(
            mode=args.derivative_mode, batch_size=args.derivative_batch_size
        )

    value, jacobian = recorder.compiled_kernel(
        "column_derivative", derivative, (point,), repeats=args.warm_repeats
    )
    evaluate = jax.jit(metrics)
    columns = []
    for i, step in enumerate((0.001, 10.0)):
        direction = jnp.zeros(2).at[i].set(step)
        columns.append((evaluate(point + direction) - evaluate(point - direction)) / (2 * step))
    finite_difference = jnp.stack(columns, axis=1)
    error = jnp.max(
        jnp.abs(jacobian - finite_difference)
        / jnp.maximum(jnp.maximum(jnp.abs(jacobian), jnp.abs(finite_difference)), 1e-8)
    )
    material_error = jnp.max(jnp.abs(result.distillate.n + result.bottoms.n - feed.n))
    energy_error = abs(
        float(
            enthalpy_flow(feed)
            + result.condenser_duty
            + result.reboiler_duty
            - enthalpy_flow(result.distillate)
            - enthalpy_flow(result.bottoms)
        )
    )
    accepted = (
        bool(result.report.converged)
        and float(material_error) < 1e-6
        and energy_error < 1.0
        and bool(jnp.all(jnp.isfinite(jacobian)))
        and float(error) < 0.003
    )
    return {
        "accepted": accepted,
        "solver": result.solver_info(),
        "numerical": result.report.to_dict(),
        "metrics_si": np.asarray(value).tolist(),
        "jacobian_si": np.asarray(jacobian).tolist(),
        "checks": {
            "material_error_mol_s": float(material_error),
            "energy_error_w": energy_error,
            "finite_difference_relative_error": float(error),
        },
    }


def _case(args, recorder):
    from fugacio.sim.cases import (
        CaseRunner,
        CaseWorkspace,
        ProcessCase,
        SolverOptions,
        optimize,
        profile,
        sensitivities,
    )
    from fugacio.sim.cases.examples import ethanol_train_case, example_case, heater_bank_case

    if args.scenario == "heater-bank":
        case = heater_bank_case(args.variables)
    elif args.scenario == "ethanol-train":
        case = ethanol_train_case(args.stages)
    else:
        case = example_case(args.scenario)
        if args.scenario == "depropanizer":
            document = case.to_dict()
            document["units"][-1]["settings"].update(
                n_stages=args.stages, feed_stages=[args.stages // 2]
            )
            case = ProcessCase.from_dict(document)
    runner = CaseRunner(
        case,
        options=SolverOptions(
            backend=args.backend,
            recycle_method="broyden",
            tolerance=1e-8,
            column_solver=args.column_solver,
            eo_jacobian=args.eo_jacobian,
        ),
    )
    workspace = CaseWorkspace(args.output / "workspace")
    workspace.save_case(case)
    parameters = list(case.parameters)
    metrics = (
        ["reboiler_duty", "annual_cost"]
        if "reboiler_duty" in runner.document["metrics"]
        else ["duty"]
    )
    common = {
        "workspace": workspace,
        "derivative_mode": args.derivative_mode,
        "derivative_batch_size": args.derivative_batch_size,
    }
    if args.study != "profile":
        common["release_caches"] = args.release_caches
        common["recorder"] = recorder
        original_run = runner.run
        run_count = 0

        def audited(*positional, **keywords):
            nonlocal run_count
            name = f"audited_run[{run_count}]"
            run_count += 1
            return recorder.measure(name, lambda: original_run(*positional, **keywords))

        runner.run = audited
    if args.study == "profile":
        study = profile(
            runner,
            parameters=parameters,
            metrics=metrics,
            warm_repeats=args.warm_repeats,
            recorder=recorder,
            **common,
        )
    elif args.study == "sensitivities":
        study = recorder.measure(
            "sensitivity_study",
            lambda: sensitivities(runner, parameters, metrics, relative_tolerance=0.005, **common),
        )
    else:
        study = recorder.measure(
            "optimization_study",
            lambda: optimize(
                runner, parameters, "annual_cost", max_iterations=args.max_iterations, **common
            ),
        )
    return {"accepted": study.accepted, "study_id": study.study_id, "artifact": study.artifact}


def worker(args):
    # Fail explicitly on a host without POSIX RSS accounting instead of
    # silently running a supposedly bounded worker without its watchdog.
    import resource  # noqa: F401

    threading.Thread(
        target=_watch_memory, args=(args.output, args.max_rss_gb * 1e9), daemon=True
    ).start()
    import jax

    # Configure before importing Fugacio: module-level arrays can initialize
    # JAX's backend and its one-time cache-enabled decision on older versions.
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir", str(args.cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

    from fugacio.sim.cases.profiling import PerformanceRecorder
    from fugacio.sim.cases.results import json_value

    recorder = PerformanceRecorder(args.output / "progress.json")
    result = _column(args, recorder) if args.scenario == "column" else _case(args, recorder)
    result["observations"] = recorder.snapshot("completed")
    # Failed kernels can contain NaN diagnostics. Preserve them as unavailable
    # values without losing the explicit rejection or the completed phases.
    _write(args.output / "result.json", json_value(result))
    return 0 if result["accepted"] else 2


def main():
    args = arguments()
    if args.worker:
        return worker(args)
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output isn't empty; select a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cache == "cold":
        cache_parent = args.cache_dir or args.output
        cache_parent.mkdir(parents=True, exist_ok=True)
        cache = Path(tempfile.mkdtemp(prefix="cold-cache-", dir=cache_parent)).resolve()
    else:
        if args.cache_dir is None or not args.cache_dir.is_dir():
            raise SystemExit("warm mode requires an existing --cache-dir")
        cache = args.cache_dir.resolve()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--worker",
        "--output",
        str(args.output),
        "--cache-dir",
        str(cache),
    ]
    root = Path(__file__).resolve().parents[1]
    sources = [
        *sorted(
            source
            for source in root.glob("packages/*/src/**/*")
            if source.is_file() and source.suffix in (".py", ".json", ".csv", ".tsv")
        ),
        *sorted(root.glob("packages/*/pyproject.toml")),
        Path(__file__).resolve(),
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    source_hash = hashlib.sha256()
    for source in sources:
        source_hash.update(str(source.relative_to(root)).encode() + b"\0" + source.read_bytes())
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False
    ).stdout.strip()
    started = time.perf_counter()
    timed_out = False
    memory_before = _cgroup_memory_events()
    with (
        (args.output / "worker.log").open("w", encoding="utf-8") as log,
        subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "JAX_ENABLE_X64": "true"},
        ) as process,
    ):
        try:
            code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            code = process.wait()
    memory_after = _cgroup_memory_events()
    result_path = args.output / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    limit_path = args.output / "resource-limit.json"
    limit = json.loads(limit_path.read_text()) if limit_path.exists() else None
    peak = result.get("observations", {}).get("process_peak_rss_bytes")
    if limit is None and peak is not None and peak > args.max_rss_gb * 1e9:
        # A short-lived final allocation can occur between watchdog samples.
        # The completed worker's high-water mark must still enforce the budget.
        limit = {
            "reason": "memory_limit",
            "peak_rss_bytes": peak,
            "budget_bytes": args.max_rss_gb * 1e9,
            "detected_at": "completed_worker",
        }
    if limit is None and code == -9 and memory_before is not None and memory_after is not None:
        killed = memory_after.get("oom_kill", 0) - memory_before.get("oom_kill", 0)
        if killed > 0:
            limit = {"reason": "cgroup_oom_kill", "oom_kill_delta": killed}
    report = {
        "schema_version": 1,
        "kind": "process_benchmark",
        "accepted": code == 0 and result.get("accepted") is True and limit is None,
        "scenario": args.scenario,
        "study": args.study,
        "revision": revision,
        "working_tree_dirty": bool(dirty),
        "source_sha256": source_hash.hexdigest(),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "worker"
        },
        "cache": {"mode": args.cache, "directory": str(cache), "fresh_process": True},
        "elapsed_seconds": time.perf_counter() - started,
        "exit_code": code,
        "timed_out": timed_out,
        "resource_limit": limit,
        "cgroup_memory_events": {"before": memory_before, "after": memory_after},
        "result": result,
        "scope": "Observed on this host; performance doesn't extend thermodynamic qualification.",
    }
    _write(args.output / "benchmark.json", report)
    print(
        json.dumps(
            {
                "accepted": report["accepted"],
                "report": str(args.output / "benchmark.json"),
                "elapsed_seconds": report["elapsed_seconds"],
                "resource_limit": limit,
            }
        )
    )
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
