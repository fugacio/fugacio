from __future__ import annotations

import pytest

from fugacio.sim.cases import (
    CaseRunner,
    CaseWorkspace,
    ProcessCase,
    SolverOptions,
    optimize,
    sensitivities,
    sweep,
)
from fugacio.sim.cases.cli import main
from fugacio.sim.cases.examples import example_case
from fugacio.sim.cases.jsonio import write_json


@pytest.fixture(scope="module")
def runner():
    return CaseRunner(example_case())


def test_sweep_preserves_failed_points_and_can_be_reopened(tmp_path, runner):
    workspace = CaseWorkspace(tmp_path)
    study = sweep(
        runner,
        {"temperature": [{"value": t, "unit": "K"} for t in (340, 360, 450)]},
        workspace=workspace,
    )
    assert not study.accepted
    assert study.artifact["accepted_count"] == 2
    assert study.artifact["failed_count"] == 1
    assert len(study.runs) == 2
    assert study.artifact["points"][2]["error_type"] == "CaseValidationError"
    assert workspace.load_artifact(study.study_id) == study.artifact
    assert all(workspace.load_run(r.run_id).accepted for r in study.runs)
    with pytest.raises(ValueError, match="max_points"):
        sweep(runner, {"temperature": [{"value": 350, "unit": "K"}] * 101})


def test_interrupted_sweep_retains_completed_runs(tmp_path, runner, monkeypatch):
    workspace = CaseWorkspace(tmp_path)
    original = runner.run
    completed = []

    def interrupted(overrides=None, **kwargs):
        if completed:
            raise KeyboardInterrupt
        run = original(overrides, **kwargs)
        completed.append(run)
        return run

    monkeypatch.setattr(runner, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        sweep(
            runner,
            {"temperature": [{"value": 350, "unit": "K"}, {"value": 360, "unit": "K"}]},
            workspace=workspace,
        )
    assert workspace.load_run(completed[0].run_id) == completed[0]
    assert len(list((tmp_path / "artifacts").glob("*.json"))) == 1


@pytest.mark.parametrize("kind", ["sensitivities", "optimization"])
def test_failed_baseline_is_saved_without_a_study_claim(tmp_path, kind):
    from fugacio.sim.cases import ProcessCase

    d = example_case().to_dict()
    d["specifications"] = [
        {
            "name": "impossible",
            "parameter": "temperature",
            "metric": "product_temperature",
            "target": {"value": 500, "unit": "K"},
            "tolerance": {"value": 0.001, "unit": "delta_K"},
        }
    ]
    d["parameters"]["flow"] = {"value": 1, "unit": "mol/s", "lower": 0.5, "upper": 2}
    d["feeds"]["feed"]["flow"] = {"parameter": "flow"}
    runner = CaseRunner(ProcessCase.from_dict(d), options=SolverOptions(specification_iterations=0))
    workspace = CaseWorkspace(tmp_path)
    study = (
        sensitivities(runner, ["flow"], ["duty"], workspace=workspace)
        if kind == "sensitivities"
        else optimize(runner, ["flow"], "annual_cost", workspace=workspace)
    )
    assert not study.accepted
    assert not workspace.load_run(study.artifact["baseline_id"]).accepted
    assert workspace.load_artifact(study.study_id)["reason"].startswith("The baseline failed")


def test_sensitivity_verifies_units_and_rejects_boundary_claim(runner):
    study = sensitivities(runner, ["temperature"], ["duty"])
    assert study.accepted
    metric = study.artifact["results"][0]["metrics"]["duty"]
    assert metric["relative_error"] < 1e-5
    assert metric["autodiff_display"] == pytest.approx(metric["autodiff_si"] / 1000)
    boundary = sensitivities(
        runner, ["temperature"], ["duty"], overrides={"temperature": {"value": 310, "unit": "K"}}
    )
    assert not boundary.accepted
    assert "interval" in boundary.artifact["results"][0]["reason"]


def test_eo_study_preserves_implicit_derivatives():
    runner = CaseRunner(example_case(), options=SolverOptions(backend="eo"))
    study = sensitivities(runner, ["temperature"], ["duty"])
    assert study.accepted
    assert study.artifact["results"][0]["metrics"]["duty"]["relative_error"] < 1e-5


def test_optimization_meets_constraint_and_retains_baseline(runner):
    study = optimize(
        runner,
        ["temperature"],
        "annual_cost",
        constraints=[
            {
                "metric": "product_temperature",
                "lower": {"value": 360, "unit": "K"},
                "tolerance": {"value": 1e-5, "unit": "delta_K"},
            }
        ],
    )
    assert study.accepted
    assert len(study.runs) == 2
    assert study.runs[1].to_dict()["parameters"]["temperature"]["value_si"] == pytest.approx(
        360, abs=1e-5
    )
    assert study.artifact["feasibility"][0]["accepted"]
    assert study.artifact["comparison"]["baseline_id"] == study.runs[0].run_id
    assert all(c["accepted"] for c in study.artifact["initialization_checks"])


def test_infeasible_optimization_does_not_promote_finite_candidate(runner):
    study = optimize(
        runner,
        ["temperature"],
        "annual_cost",
        max_iterations=3,
        constraints=[
            {
                "metric": "product_temperature",
                "lower": {"value": 500, "unit": "K"},
                "tolerance": {"value": 1e-5, "unit": "delta_K"},
            }
        ],
    )
    assert not study.accepted
    assert not study.artifact["feasibility"][0]["accepted"]


def test_two_variable_design_meets_temperature_and_throughput_constraints():
    d = example_case().to_dict()
    d["parameters"]["flow"] = {"value": 1.5, "unit": "mol/s", "lower": 0.5, "upper": 2}
    d["feeds"]["feed"]["flow"] = {"parameter": "flow"}
    d["metrics"]["throughput"] = {
        "expression": {"stream": "product", "property": "flow"},
        "unit": "mol/s",
    }
    runner = CaseRunner(ProcessCase.from_dict(d))
    gradient = sensitivities(
        runner, ["flow", "temperature"], ["duty", "throughput"], release_caches=True
    )
    assert gradient.accepted, gradient.artifact
    study = optimize(
        runner,
        ["flow", "temperature"],
        "annual_cost",
        release_caches=True,
        constraints=[
            {
                "metric": "product_temperature",
                "lower": {"value": 360, "unit": "K"},
                "tolerance": {"value": 1e-5, "unit": "delta_K"},
            },
            {
                "metric": "throughput",
                "lower": {"value": 1.2, "unit": "mol/s"},
                "tolerance": {"value": 1e-7, "unit": "mol/s"},
            },
        ],
    )
    assert study.accepted, study.artifact
    assert study.artifact["release_caches"] and gradient.artifact["release_caches"]
    final = study.runs[-1].to_dict()["parameters"]
    assert final["flow"]["value_si"] == pytest.approx(1.2, abs=1e-7)
    assert final["temperature"]["value_si"] == pytest.approx(360, abs=1e-5)


def test_annual_rate_objective_is_scaled_to_reach_its_bound(runner):
    study = optimize(runner, ["temperature"], "annual_cost")
    assert study.accepted
    assert study.runs[1].to_dict()["parameters"]["temperature"]["value_si"] == pytest.approx(
        310, abs=1e-5
    )


def test_optimizer_rejects_a_seed_dependent_final_metric(runner, monkeypatch):
    original = runner.evaluate

    def changed_branch(parameters=None, **kwargs):
        e = original(parameters, **kwargs)
        if kwargs.get("initialization") is not None:
            e = e._replace(metrics={**e.metrics, "annual_cost": 2 * e.metrics["annual_cost"]})
        return e

    monkeypatch.setattr(runner, "evaluate", changed_branch)
    study = optimize(runner, ["temperature"], "annual_cost")
    assert study.artifact["optimizer"]["success"]
    assert study.runs[1].accepted
    assert not study.artifact["initialization_checks"][0]["accepted"]
    assert not study.accepted


def test_cli_complete_saved_workflow(tmp_path, capsys):
    case, output, report = tmp_path / "case.json", tmp_path / "run.json", tmp_path / "report.md"
    workspace = tmp_path / "workspace"
    assert main(["example", "heater", str(case)]) == 0
    assert main(["validate", str(case)]) == 0
    assert (
        main(
            [
                "run",
                str(case),
                "--workspace",
                str(workspace),
                "--output",
                str(output),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    from fugacio.sim.cases import CaseRun

    run = CaseRun.load(output)
    assert main(["inspect", run.run_id, "--workspace", str(workspace), "--report"]) == 0
    assert report.read_text() == run.markdown()
    request = tmp_path / "sweep.json"
    write_json(request, {"grid": {"temperature": [{"value": 600, "unit": "K"}]}})
    assert main(["sweep", str(case), str(request), "--workspace", str(workspace)]) == 2
    assert main(["inspect", "../bad", "--workspace", str(workspace)]) == 1
    assert "artifact IDs" in capsys.readouterr().err
