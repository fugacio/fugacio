import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from fugacio.sim.cases import (
    CaseRunner,
    CaseWorkspace,
    PerformanceRecorder,
    SolverOptions,
    profile,
    sensitivities,
)
from fugacio.sim.cases.cli import main
from fugacio.sim.cases.examples import example_case, heater_bank_case
from fugacio.sim.cases.jsonio import read_json
from fugacio.sim.cases.results import CaseRun, sealed


@pytest.fixture(scope="module")
def runner():
    return CaseRunner(example_case())


def test_recorder_separates_compilation_from_synchronized_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("MALLOC_ARENA_MAX", "2")
    recorder = PerformanceRecorder(tmp_path / "progress.json")
    value = recorder.compiled_kernel("square", lambda x: x**2, (jnp.array([2.0, 3.0]),), repeats=2)
    assert value.tolist() == [4.0, 9.0]
    snapshot = read_json(tmp_path / "progress.json")
    assert [p["name"] for p in snapshot["phases"]] == [
        "square:trace_and_lower",
        "square:compile",
        "square:first_execution",
        "square:warm_execution[0]",
        "square:warm_execution[1]",
    ]
    assert all(p["status"] == "completed" and p["seconds"] >= 0 for p in snapshot["phases"])
    assert snapshot["environment"]["jax_x64"]
    assert snapshot["environment"]["runtime_settings"]["MALLOC_ARENA_MAX"] == "2"


def test_recorder_checkpoints_failed_operations(tmp_path):
    recorder = PerformanceRecorder(tmp_path / "progress.json")

    def fail():
        raise RuntimeError("intentional failure")

    with pytest.raises(RuntimeError, match="intentional"):
        recorder.measure("bad_phase", fail)
    recorded = read_json(tmp_path / "progress.json")
    assert recorded["status"] == "failed"
    assert recorded["phases"][-1]["error_type"] == "RuntimeError"


def test_nested_recorder_keeps_parent_running_until_it_finishes(tmp_path):
    checkpoint = tmp_path / "progress.json"
    recorder = PerformanceRecorder(checkpoint)

    def outer():
        value = recorder.measure("inner", lambda: 2.0)
        assert read_json(checkpoint)["status"] == "running"
        return value

    assert recorder.measure("outer", outer) == 2.0
    assert read_json(checkpoint)["status"] == "completed"


def test_profile_preserves_ordinary_run_identity_and_saves_derivative_strategy(tmp_path, runner):
    baseline = runner.run(check=True)
    workspace = CaseWorkspace(tmp_path)
    result = profile(
        runner, parameters=["temperature"], metrics=["duty"], warm_repeats=1, workspace=workspace
    )
    assert result.accepted
    assert result.runs[0].run_id == baseline.run_id == result.runs[-1].run_id
    saved = workspace.load_artifact(result.study_id)
    assert saved["derivatives"]["mode"] == "forward"
    assert saved["derivatives"]["jacobian_si"][0][0] > 0
    assert not saved["derivatives"]["finite_difference_verified"]
    assert saved["observations"]["process_peak_rss_bytes"] > 0
    assert saved["structure"]["equation_oriented"]["structurally_square_and_matched"]
    assert workspace.load_run(saved["baseline_id"]).accepted


def test_profile_failed_physics_is_retained_without_derivative_claim():
    document = example_case().to_dict()
    document["metrics"]["bad"] = {"expression": {"op": "divide", "args": [1, 0]}, "unit": "1"}
    from fugacio.sim.cases import ProcessCase

    result = profile(
        CaseRunner(ProcessCase.from_dict(document)),
        parameters=["temperature"],
        metrics=["duty"],
        warm_repeats=1,
    )
    assert not result.accepted
    assert result.artifact["derivatives"] is None
    assert len(result.runs) == 2 and all(not r.accepted for r in result.runs)


def test_legacy_run_replay_preserves_dense_numerical_choices(tmp_path, runner):
    old = runner.run().to_dict()
    old["solver"].pop("column_solver")
    old["solver"].pop("eo_jacobian")
    old.pop("artifact_id")
    legacy = CaseRun.from_dict(sealed("run", old))
    workspace = CaseWorkspace(tmp_path)
    workspace.save_run(legacy)
    replay = workspace.replay(legacy.run_id)
    assert replay.accepted
    assert replay.to_dict()["solver"]["column_solver"] == "dense"
    assert replay.to_dict()["solver"]["eo_jacobian"] == "dense"


def test_new_examples_and_structure_command_are_portable(tmp_path, capsys):
    case = heater_bank_case(12)
    assert len(case.parameters) == 12
    path = tmp_path / "bank.json"
    case.save(path)
    assert main(["diagnose", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["partitions"]) == 12
    assert result["unit_instances"] == 12 and result["unit_templates"] == 1
    assert result["equation_oriented"]["colored_directions"] == 4
    assert result["equation_oriented"]["n_unknowns"] == 48
    train = CaseRunner(example_case("ethanol-train"))
    assert train.diagnose_structure()["columns"]["column"]["stages"] == 12


def test_many_variable_reverse_study_matches_audited_finite_differences():
    runner = CaseRunner(heater_bank_case(4))
    assert runner.diagnose_structure()["unit_templates"] == 1
    recorder = PerformanceRecorder()
    result = sensitivities(runner, list(runner.parameters), ["duty"], recorder=recorder)
    assert result.accepted, result.artifact
    assert result.artifact["derivatives"]["mode"] == "reverse"
    assert result.artifact["derivatives"]["directions"] == 1
    assert all(row["metrics"]["duty"]["autodiff_si"] > 0 for row in result.artifact["results"])
    assert [phase["name"] for phase in recorder.phases] == ["linearization[0]", "jacobian[0]"]
    assert all(phase["status"] == "completed" for phase in recorder.phases)


@pytest.mark.parametrize("option,value", [("column_solver", "bad"), ("eo_jacobian", "bad")])
def test_invalid_solver_choices(option, value):
    with pytest.raises(ValueError):
        SolverOptions(**{option: value})


@pytest.mark.parametrize("limit", ["time", "memory"])
def test_benchmark_limits_retain_failed_resource_reports(tmp_path, limit):
    script = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_process.py"
    output = tmp_path / limit
    settings = ["--timeout-seconds", "0.01"] if limit == "time" else ["--max-rss-gb", "0.001"]
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--scenario",
            "heater",
            "--output",
            str(output),
            *settings,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 2, process.stderr
    report = read_json(output / "benchmark.json")
    assert not report["accepted"]
    assert report["timed_out"] == (limit == "time")
    if limit == "memory":
        assert report["resource_limit"]["reason"] == "memory_limit"
        assert report["resource_limit"]["peak_rss_bytes"] > 1e6
    assert report["cache"]["fresh_process"] and report["cache"]["mode"] == "cold"


def test_benchmark_populates_persistent_cache_and_can_reuse_it(tmp_path):
    script = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_process.py"
    prior = None
    for mode in ("cold", "warm"):
        output = tmp_path / mode
        command = [
            sys.executable,
            str(script),
            "--scenario",
            "heater",
            "--warm-repeats",
            "1",
            "--output",
            str(output),
            "--cache",
            mode,
        ]
        if prior is not None:
            command.extend(["--cache-dir", prior["cache"]["directory"]])
        process = subprocess.run(command, capture_output=True, text=True, timeout=180)
        assert process.returncode == 0, process.stdout + process.stderr
        report = read_json(output / "benchmark.json")
        assert report["accepted"] and report["cache"]["mode"] == mode
        assert any(Path(report["cache"]["directory"]).rglob("*-cache"))
        if prior is not None:
            assert report["cache"]["directory"] == prior["cache"]["directory"]
            actual = report["result"]["artifact"]["derivatives"]["jacobian_si"]
            expected = prior["result"]["artifact"]["derivatives"]["jacobian_si"]
            assert actual == expected
        prior = report


def test_benchmark_final_peak_cannot_bypass_the_sampled_watchdog(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_process.py"
    spec = importlib.util.spec_from_file_location("process_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "completed"

    class FinishedWorker:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            # Model a successful worker whose final allocation exceeded its
            # budget after the last watchdog poll but before process exit.
            (output / "result.json").write_text(
                json.dumps(
                    {
                        "accepted": True,
                        "observations": {"process_peak_rss_bytes": 1_100_000_000},
                    }
                )
            )
            return self

        def __exit__(self, *args):
            pass

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", FinishedWorker)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=""))
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(output), "--max-rss-gb", "1"])
    assert module.main() == 2
    report = read_json(output / "benchmark.json")
    assert report["exit_code"] == 0 and not report["accepted"]
    assert report["resource_limit"]["peak_rss_bytes"] == 1_100_000_000
    assert report["resource_limit"]["detected_at"] == "completed_worker"
