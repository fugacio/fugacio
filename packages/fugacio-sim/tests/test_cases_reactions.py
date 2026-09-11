"""Portable reactive workflows, backend parity, studies, persistence, and rejection."""

import copy

import jax
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
from fugacio.sim.cases.quantities import CaseValidationError


@pytest.mark.parametrize("problem", ["reactive-recycle", "reactive-separation"])
def test_reactive_workflow_end_to_end(problem, tmp_path):
    case = example_case(problem)
    runner = CaseRunner(case)
    workspace = CaseWorkspace(tmp_path)
    run = runner.run(check=True)
    data = run.to_dict()
    assert data["checks"]["physical"]["boundaries"]["plant"]["element_relative_error"] < 1e-7
    assert data["checks"]["physical"]["boundaries"]["plant"]["energy_relative_error"] < 1e-7
    assert data["metrics"]["extent"]["value_si"] > 0
    assert data["qualification"]["status"] == "not_evaluated"
    workspace.save_run(run)
    assert workspace.load_run(run.run_id) == run
    assert workspace.replay(run.run_id).accepted
    study = sensitivities(runner, ["volume", "kinetic_rate"], ["extent"], workspace=workspace)
    assert study.accepted
    assert all(
        row["metrics"]["extent"]["relative_error"] < 2e-3 for row in study.artifact["results"]
    )
    design = optimize(
        runner, ["volume"], "product_isobutane", sense="max", max_iterations=20, workspace=workspace
    )
    assert design.accepted
    assert (
        design.runs[-1].to_dict()["metrics"]["product_isobutane"]["value_si"]
        > data["metrics"]["product_isobutane"]["value_si"]
    )
    assert workspace.load_artifact(design.study_id) == design.artifact
    # Leave memory available for the next complete solver graph.
    jax.clear_caches()


def test_reactive_recycle_eo_matches_sequential():
    case = example_case("reactive-recycle")
    seq = CaseRunner(case).run(check=True)
    runner = CaseRunner(case, options=SolverOptions(backend="eo"))
    eo = runner.run(check=True)
    assert eo.to_dict()["metrics"]["extent"]["value_si"] == pytest.approx(
        seq.to_dict()["metrics"]["extent"]["value_si"], rel=1e-7
    )
    study = sensitivities(runner, ["volume"], ["extent"])
    assert study.accepted
    jax.clear_caches()


def single_unit(kind):
    d = example_case("reactive-recycle").to_dict()
    settings = {"reaction_set": "isomerization"}
    if kind == "reactive_flash":
        settings.update(t={"parameter": "temperature"}, p={"value": 10, "unit": "bar"})
        outlets = ["product", "liquid"]
    else:
        settings["t_out"] = {"parameter": "temperature"}
        if kind != "equilibrium_reactor":
            settings["volume"] = {"parameter": "volume"}
        if kind == "pfr":
            settings["steps"] = 16
        outlets = ["product"]
    d["units"] = [
        {
            "name": "reactor",
            "kind": kind,
            "inlets": ["feed"],
            "outlets": outlets,
            "settings": settings,
        }
    ]
    return d


@pytest.mark.parametrize("kind", ["equilibrium_reactor", "pfr", "reactive_flash"])
def test_registered_reactor_variants_are_accepted(kind):
    runner = CaseRunner(ProcessCase.from_dict(single_unit(kind)))
    assert runner.run(check=True).accepted
    jax.clear_caches()


@pytest.mark.parametrize(
    "change,match",
    [
        (
            lambda d: d["reaction_sets"]["isomerization"]["reactions"][0].update(nu=[-1, 2]),
            "conserve",
        ),
        (lambda d: d["parameters"]["kinetic_rate"].update(unit="1"), "dimension"),
        (lambda d: d["parameters"]["kinetic_rate"].update(value=-1, lower=-2), "nonnegative"),
        (lambda d: d["units"][1]["settings"].update(reaction_set="absent"), "declared"),
        (
            lambda d: d["reaction_sets"]["isomerization"].update(rate_basis="concentration"),
            "choose",
        ),
        (lambda d: d["metrics"]["extent"]["expression"].pop("reaction"), "selector"),
        (
            lambda d: d["reaction_sets"]["isomerization"]["reactions"][0]["rate"].update(
                k_reverse={"value": 1, "unit": "mol/(m3 s)"}
            ),
            "determines",
        ),
    ],
)
def test_reactive_case_rejects_ambiguous_or_invalid_definitions(change, match):
    d = example_case("reactive-recycle").to_dict()
    change(d)
    with pytest.raises(CaseValidationError, match=match):
        ProcessCase.from_dict(d)


def test_reaction_volume_and_kinetic_override_bounds():
    runner = CaseRunner(example_case("reactive-separation"))
    with pytest.raises(CaseValidationError):
        runner.parameter_values({"volume": {"value": -0.1, "unit": "m3"}})
    with pytest.raises(CaseValidationError):
        runner.parameter_values({"kinetic_rate": {"value": 1, "unit": "1"}})
    d = runner.case.to_dict()
    d["units"][0]["settings"]["reaction_volumes"].pop()
    with pytest.raises(CaseValidationError):
        ProcessCase.from_dict(d)


def test_reusable_reaction_references_share_a_parameter_binding():
    d = single_unit("cstr")
    first = d["units"][0]
    first["outlets"] = ["intermediate"]
    second = copy.deepcopy(first)
    second.update(name="second", inlets=["intermediate"], outlets=["product"])
    d["units"].append(second)
    runner = CaseRunner(ProcessCase.from_dict(d))
    assert runner.diagnose_structure()["unit_templates"] == 1


@pytest.mark.parametrize("name", ["reactive-recycle", "reactive-separation"])
def test_cli_exports_the_committed_reactive_examples(name, tmp_path, capsys):
    from pathlib import Path

    from fugacio.sim.cases.cli import main

    path = tmp_path / (name + ".json")
    assert main(["example", name, str(path)]) == 0
    assert ProcessCase.load(path) == example_case(name)
    committed = Path(__file__).parents[3] / "examples" / "process-cases" / (name + ".json")
    assert ProcessCase.load(committed) == ProcessCase.load(path)
    capsys.readouterr()


def test_reaction_rate_and_concentration_units_convert_to_si():
    from fugacio.sim.cases.quantities import CONCENTRATION, REACTION_RATE, quantity

    assert quantity({"value": 3.6, "unit": "kmol/(m3 h)"}, REACTION_RATE, "rate") == pytest.approx(
        1.0
    )
    assert quantity({"value": 0.1, "unit": "mol/L"}, CONCENTRATION, "reference") == pytest.approx(
        100.0
    )
    with pytest.raises(CaseValidationError):
        quantity({"value": 1, "unit": "mol/s"}, REACTION_RATE, "rate")


def test_reactive_separation_eo_retains_reaction_profiles():
    case = example_case("reactive-separation")
    runner = CaseRunner(case, options=SolverOptions(backend="eo"))
    run = runner.run(check=True)
    data = run.to_dict()
    assert data["checks"]["physical"]["boundaries"]["plant"]["energy_relative_error"] < 1e-7
    assert len(data["units"]["column"]["profiles_si"]["reaction_rates"]) == 6
    assert data["metrics"]["extent"]["value_si"] > 0
    assert data["units"]["column"]["linear_system"]["solver"] == "block"
    jax.clear_caches()


@pytest.mark.parametrize("kind", ["reactive_flash", "equilibrium_reactor"])
def test_rate_profiles_require_an_actual_kinetic_profile(kind):
    document = single_unit(kind)
    document["reaction_sets"]["isomerization"]["reactions"][0].pop("rate")
    document["metrics"]["rate"] = {
        "expression": {
            "unit": "reactor",
            "profile": "reaction_rates",
            "stage": 1,
            "reaction": "isomerize",
        },
        "unit": "mol/(m3 s)",
    }
    with pytest.raises(CaseValidationError, match="profile"):
        ProcessCase.from_dict(document)
