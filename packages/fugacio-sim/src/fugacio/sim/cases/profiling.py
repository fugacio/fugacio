"""Explicit performance observations tied to audited process runs.

Timings live in a separate artifact, so profiling doesn't add nondeterministic
fields to ordinary case/run identities. Compilation and execution are separated
only where a caller explicitly lowers and compiles a JAX kernel. A modular
plant's first evaluation includes its individual units' compilation work.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from fugacio.sim.cases.jsonio import write_json
from fugacio.sim.cases.results import sealed
from fugacio.sim.cases.runtime import CaseRunner
from fugacio.sim.cases.studies import StudyResult, _names
from fugacio.sim.cases.workspace import CaseWorkspace
from fugacio.thermo.sensitivity import derivative_strategy, linearize


def peak_rss_bytes() -> int | None:
    """Return process-lifetime peak resident bytes, or None where unavailable.

    This includes compilation and native allocations. It is a high-water mark
    for this process, not an allocation counter or a per-phase memory delta.
    """
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, OSError):
        return None


class PerformanceRecorder:
    """Synchronize timed operations and optionally checkpoint phase observations.

    Checkpoints describe in-progress or failed work; they aren't accepted run
    artifacts. A killed subprocess can leave its last completed phase on disk.
    Use a fresh process and an explicitly isolated persistent cache for a cold
    benchmark. This recorder never clears process-global caches on its own.
    """

    def __init__(self, checkpoint: str | Path | None = None) -> None:
        self.phases: list[dict[str, Any]] = []
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.environment = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_x64": bool(jax.config.read("jax_enable_x64")),
            "devices": [str(device) for device in jax.devices()],
            "cpu_count": os.cpu_count(),
            "persistent_cache": str(getattr(jax.config, "jax_compilation_cache_dir", None)),
            "runtime_settings": {
                name: os.environ.get(name)
                for name in (
                    "MALLOC_ARENA_MAX",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "XLA_FLAGS",
                )
            },
        }

    def snapshot(self, status: str = "running") -> dict[str, Any]:
        """Return an unsealed observation, including any completed phases."""
        return {
            "status": status,
            "environment": self.environment,
            "phases": self.phases,
            "process_peak_rss_bytes": peak_rss_bytes(),
            "memory_scope": "Process-lifetime high-water mark; phases share retained executables.",
        }

    def _save(self, status: str) -> None:
        if self.checkpoint is not None:
            write_json(self.checkpoint, self.snapshot(status))

    def measure(
        self,
        name: str,
        operation: Callable[[], Any],
        *,
        ready: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Time one operation through device completion and retain failures."""
        record: dict[str, Any] = {"name": name, "status": "running"}
        self.phases.append(record)
        self._save("running")
        started, cpu = time.perf_counter(), time.process_time()
        try:
            value = operation()
            jax.block_until_ready(value if ready is None else ready(value))
            record["status"] = "completed"
            return value
        except BaseException as exc:
            record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            record.update(
                seconds=time.perf_counter() - started,
                cpu_seconds=time.process_time() - cpu,
                process_peak_rss_bytes=peak_rss_bytes(),
            )
            statuses = {phase["status"] for phase in self.phases}
            status = (
                "failed"
                if "failed" in statuses
                else "running"
                if "running" in statuses
                else "completed"
            )
            self._save(status)

    def compiled_kernel(
        self, name: str, function: Callable[..., Any], args: tuple[Any, ...], *, repeats: int = 2
    ) -> Any:
        """Measure tracing/lowering, compilation, first execution, and warm execution.

        Call in a fresh process to characterize cold compilation. Persistent
        cache hits can reduce the compile phase and must be recorded by the
        benchmark caller. The returned value is the final synchronized result.
        """
        _repeats(repeats)
        staged = self.measure(name + ":trace_and_lower", lambda: jax.jit(function).lower(*args))
        executable = self.measure(name + ":compile", staged.compile)
        value = self.measure(name + ":first_execution", lambda: executable(*args))
        for i in range(repeats):
            value = self.measure(name + f":warm_execution[{i}]", lambda: executable(*args))
        return value


def _repeats(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10:
        raise ValueError("warm_repeats must be an integer from one to ten")


def profile(
    runner: CaseRunner,
    *,
    overrides: dict[str, Any] | None = None,
    parameters: list[str] | None = None,
    metrics: list[str] | None = None,
    derivative_mode: str = "auto",
    derivative_batch_size: int = 1,
    warm_repeats: int = 2,
    workspace: CaseWorkspace | None = None,
    recorder: PerformanceRecorder | None = None,
) -> StudyResult:
    """Measure an audited case and optional reusable implicit metric derivatives.

    First calls may include compilation; this API makes no cold-cache claim.
    Derivative observations require an accepted baseline and finite converged
    numerical evaluations. They aren't finite-difference verification; use
    ``sensitivities`` for that independent check. Failed physics remains failed
    regardless of speed. Peak RSS includes every prior allocation in the process.
    """
    _repeats(warm_repeats)
    if (parameters is None) != (metrics is None):
        raise ValueError("profiling derivatives requires both parameters and metrics")
    names = _names(runner, parameters) if parameters is not None else ()
    outputs = _names(runner, metrics, metrics=True) if metrics is not None else ()
    strategy = derivative_strategy(
        len(names) or 1, len(outputs) or 1, mode=derivative_mode, batch_size=derivative_batch_size
    )
    recorder = recorder or PerformanceRecorder()
    baseline = recorder.measure("first_audited_run", lambda: runner.run(overrides))
    runs = [baseline]
    if workspace is not None:
        workspace.save_run(baseline)
    valid = baseline.accepted
    for i in range(warm_repeats):
        run = recorder.measure(f"warm_audited_run[{i}]", lambda: runner.run(overrides))
        runs.append(run)
        valid = valid and run.accepted
        if workspace is not None:
            workspace.save_run(run)
    derivatives: Any = None
    if baseline.accepted and names:
        values = runner.parameter_values(overrides)
        initialization = runner.initialization(baseline)
        point = jnp.asarray([values[key] for key in names])

        def function(x: Any) -> Any:
            evaluation = runner.evaluate(
                {**values, **dict(zip(names, x, strict=True))}, initialization=initialization
            )
            return jnp.asarray([evaluation.metrics[key] for key in outputs]), jnp.all(
                jnp.asarray([report.converged for report in evaluation.reports.values()])
            )

        matrices = []
        for i in range(warm_repeats + 1):
            prefix = "first" if i == 0 else f"warm[{i - 1}]"
            local = recorder.measure(
                prefix + ":linearization",
                lambda: linearize(function, point, has_aux=True),
            )
            matrix = recorder.measure(
                prefix + ":jacobian",
                partial(local.jacobian, mode=derivative_mode, batch_size=derivative_batch_size),
            )
            valid = valid and bool(local.auxiliary) and bool(jnp.all(jnp.isfinite(matrix)))
            valid = valid and bool(jnp.all(jnp.isfinite(local.value)))
            matrices.append(np.asarray(matrix))
        for matrix in matrices[1:]:
            valid = valid and bool(np.allclose(matrix, matrices[0], rtol=1e-8, atol=1e-10))
        derivatives = {
            **strategy,
            "parameters": list(names),
            "metrics": list(outputs),
            "jacobian_si": matrices[0],
            "finite_difference_verified": False,
        }
    result = StudyResult(
        sealed(
            "profile",
            {
                "case_id": runner.case.case_id,
                "baseline_id": baseline.run_id,
                "run_ids": [run.run_id for run in runs],
                "accepted": valid,
                "observations": recorder.snapshot("completed"),
                "derivatives": derivatives,
                "structure": runner.diagnose_structure(),
                "cache_scope": (
                    "Caller-managed process and persistent cache; first is not necessarily cold."
                ),
                "scope": (
                    "Performance observations and numerical repeatability; "
                    "no independent gradient or qualification claim."
                ),
            },
        ),
        tuple(runs),
    )
    if workspace is not None:
        result.save(workspace)
    return result
