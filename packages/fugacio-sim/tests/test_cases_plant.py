from __future__ import annotations

import pytest

from fugacio.sim.cases import (
    CaseRunner,
    CaseWorkspace,
    ProcessCase,
    SolverOptions,
    optimize,
    sensitivities,
)
from fugacio.sim.cases.examples import example_case

pytestmark = pytest.mark.plant


def test_saved_depropanizer_meets_specs_balances_and_stability(tmp_path):
    path = tmp_path / "depropanizer.json"
    example_case("depropanizer").save(path)
    case = ProcessCase.load(path)
    workspace = CaseWorkspace(tmp_path / "workspace")
    runner = CaseRunner(case, options=SolverOptions(recycle_method="broyden", tolerance=1e-8))
    run = runner.run(check=True)
    workspace.save_run(run)
    d = workspace.load_run(run.run_id).to_dict()
    assert d["metrics"]["purity"]["value_si"] == pytest.approx(0.95, abs=1e-8)
    assert d["metrics"]["recovery"]["value_si"] == pytest.approx(0.98, abs=1e-8)
    assert d["metrics"]["recovered_heat"]["value_si"] > 5e5
    assert len(d["units"]["column"]["profiles_si"]["t"]) == 16
    assert d["units"]["column"]["quantities_si"]["condenser_duty"] < 0
    assert d["checks"]["physical"]["streams"]["distillate"]["minimum_tpd"] >= -1e-7
    assert d["checks"]["physical"]["boundaries"]["plant"]["energy_relative_error"] < 1e-7
    assert d["equipment"]["economizer"]["within_size_range"]


def test_depropanizer_study_gradient(tmp_path):
    runner = CaseRunner(
        example_case("depropanizer"),
        options=SolverOptions(recycle_method="broyden", tolerance=1e-8),
    )
    study = sensitivities(
        runner,
        ["approach"],
        ["reboiler_duty", "annual_cost"],
        relative_tolerance=0.005,
        release_caches=True,
        workspace=CaseWorkspace(tmp_path / "workspace"),
    )
    assert study.accepted, study.artifact
    assert study.artifact["results"][0]["metrics"]["reboiler_duty"]["autodiff_si"] > 0


def test_depropanizer_bounded_optimization(tmp_path):
    runner = CaseRunner(
        example_case("depropanizer"),
        options=SolverOptions(recycle_method="broyden", tolerance=1e-8),
    )
    study = optimize(
        runner,
        ["pressure", "approach"],
        "annual_cost",
        max_iterations=30,
        release_caches=True,
        workspace=CaseWorkspace(tmp_path / "workspace"),
    )
    assert study.accepted, study.artifact
    baseline, candidate = study.runs
    assert (
        candidate.to_dict()["metrics"]["annual_cost"]["value_si"]
        < baseline.to_dict()["metrics"]["annual_cost"]["value_si"]
    )
    assert candidate.to_dict()["metrics"]["purity"]["value_si"] == pytest.approx(0.95, abs=1e-8)
