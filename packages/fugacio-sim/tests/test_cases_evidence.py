from __future__ import annotations

import pytest

from fugacio.sim.cases import CaseRunner, ProcessCase
from fugacio.sim.cases.examples import example_case
from fugacio.thermo.measured_regression import MeasuredFit


def test_inline_measured_fit_is_detached_and_revalidated():
    case = example_case("measured-heater")
    raw = case.to_dict()["property_package"]["measured_fit"]
    before = dict(raw)
    fit = MeasuredFit.from_dict(raw)
    assert raw == before
    assert fit.components == ("ethanol", "water")
    runner = CaseRunner(case)
    run = runner.run(check=True)
    d = run.to_dict()
    assert d["qualification"]["status"] == "qualified_properties"
    assert d["qualification"]["training_provenance_verified"]
    assert d["qualification"]["counts"]["pressure_relative_rmse"] == 72
    assert d["metrics"]["duty"]["value_si"] == pytest.approx(2819.9205, rel=1e-6)
    assert d["parameter_evidence"]["temperature_range"] == [307.44, 335.16]
    assert "not a global" in d["qualification"]["scope"]


def test_qualification_rejects_leakage_unknown_ids_and_forged_hash():
    case = example_case("measured-heater")
    d = case.to_dict()
    declaration = d["property_package"]
    declaration["qualification"]["holdout_ids"] = declaration["measured_fit"]["training_ids"][:1]
    with pytest.raises(ValueError, match="leakage"):
        CaseRunner(ProcessCase.from_dict(d))
    declaration["qualification"]["holdout_ids"] = ["fabricated-observation"]
    with pytest.raises(ValueError, match="unknown holdout"):
        CaseRunner(ProcessCase.from_dict(d))
    d = case.to_dict()
    d["property_package"]["measured_fit"]["sources"][0][1] = "0" * 64
    with pytest.raises(ValueError, match="verified training"):
        CaseRunner(ProcessCase.from_dict(d))
    d = case.to_dict()
    d["property_package"]["measured_fit"]["temperature_range"] = [200, 1000]
    with pytest.raises(ValueError, match="bounds don't match"):
        CaseRunner(ProcessCase.from_dict(d))
