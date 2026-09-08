from __future__ import annotations

import copy
import json

import pytest

from fugacio.sim.cases import CaseValidationError, ProcessCase
from fugacio.sim.cases.examples import EXAMPLES, example_case
from fugacio.sim.cases.jsonio import loads
from fugacio.sim.cases.quantities import FLOW, TEMPERATURE, quantity


@pytest.mark.parametrize("name", EXAMPLES)
def test_examples_are_portable_and_detached(tmp_path, name):
    case = example_case(name)
    path = tmp_path / "nested" / "case.json"
    case.save(path)
    assert ProcessCase.load(path) == case
    before = case.case_id
    document = case.to_dict()
    document["name"] = "changed"
    assert case.case_id == before
    assert ProcessCase.from_dict(document).case_id != before
    assert json.loads(path.read_text())["schema_version"] == 1


@pytest.mark.parametrize(
    "raw", ['{"a":1,"a":2}', '{"value":NaN}', "[Infinity]", '{"a":', "[1e999]"]
)
def test_strict_json_rejects_ambiguous_inputs(raw):
    with pytest.raises(ValueError):
        loads(raw)


def test_atomic_writer_never_creates_an_unreadable_oversized_file(tmp_path, monkeypatch):
    from fugacio.sim.cases import jsonio

    monkeypatch.setattr(jsonio, "MAX_JSON_BYTES", 200)
    path = tmp_path / "artifact.json"
    value = {"items": list(range(30))}
    jsonio.write_json(path, value)
    assert jsonio.read_json(path) == value
    previous = path.read_bytes()
    with pytest.raises(CaseValidationError, match="exceeds"):
        jsonio.write_json(path, {"oversized": "x" * 201})
    assert path.read_bytes() == previous


def test_units_and_affine_parameter_bounds():
    assert quantity({"value": 25, "unit": "degC"}, TEMPERATURE, "t") == pytest.approx(298.15)
    assert quantity({"value": 36, "unit": "delta_degF"}, TEMPERATURE, "dt", difference=True) == 20
    assert quantity({"value": 36, "unit": "kmol/h"}, FLOW, "f") == 10
    case = example_case().to_dict()
    case["parameters"]["temperature"] = {"value": 80, "unit": "degC", "lower": 40, "upper": 120}
    parsed = ProcessCase.from_dict(case)
    assert parsed.parameters["temperature"].lower == pytest.approx(313.15)
    changed = parsed.with_parameters({"temperature": {"value": 373.15, "unit": "K"}})
    assert changed.to_dict()["parameters"]["temperature"]["value"] == pytest.approx(100)
    with pytest.raises(CaseValidationError, match="above"):
        parsed.with_parameters({"temperature": {"value": 500, "unit": "K"}})


@pytest.mark.parametrize(
    "raw",
    [
        300,
        {"value": True, "unit": "K"},
        {"value": "300", "unit": "K"},
        {"value": 300, "unit": "kelvin"},
        {"value": 300, "unit": "Pa"},
        {"value": 300, "unit": "delta_K"},
    ],
)
def test_temperature_input_errors(raw):
    with pytest.raises(CaseValidationError):
        quantity(raw, TEMPERATURE, "temperature")


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), 2),
        (("name",), "../../escape"),
        (("components",), ["methane", "methane"]),
        (("components",), ["unknown substance"]),
        (("feeds", "feed", "z"), [0.2, 0.2]),
        (("feeds", "feed", "pressure"), {"value": 0, "unit": "Pa"}),
        (("feeds", "feed", "flow"), {"value": -1, "unit": "mol/s"}),
        (("units", 0, "kind"), "python"),
        (("units", 0, "settings", "t_out"), {"parameter": "missing"}),
        (("units", 0, "settings", "unknown"), 1),
        (("units", 0, "settings", "duty"), {"value": 100, "unit": "W"}),
        (("units", 0, "inlets"), ["missing"]),
        (("metrics", "duty", "unit"), "Pa"),
        (("metrics", "duty", "expression"), "__import__('os').system('id')"),
        (("metrics", "duty", "expression"), {"stream": "feed", "property": []}),
        (("property_package", "options"), {"vapor": "ideal"}),
        (("parameters", "temperature", "unit"), "delta_K"),
    ],
)
def test_path_aware_schema_rejections(path, value):
    data = example_case().to_dict()
    parent = data
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(CaseValidationError):
        ProcessCase.from_dict(data)


def test_duplicate_producers_fanout_and_disconnected_loops():
    data = example_case().to_dict()
    unit = copy.deepcopy(data["units"][0])
    unit["name"] = "second"
    unit["outlets"] = ["another"]
    data["units"].append(unit)
    with pytest.raises(CaseValidationError, match="consumed more than once"):
        ProcessCase.from_dict(data)
    unit["inlets"] = ["product"]
    unit["outlets"] = ["feed"]
    with pytest.raises(CaseValidationError, match="duplicate producer"):
        ProcessCase.from_dict(data)
    unit["inlets"] = ["loop"]
    unit["outlets"] = ["loop"]
    with pytest.raises(CaseValidationError, match="no path from a feed"):
        ProcessCase.from_dict(data)


def test_metric_cycles_and_dimension_mismatch():
    d = example_case().to_dict()
    d["metrics"]["duty"]["expression"] = {"metric": "duty"}
    with pytest.raises(CaseValidationError, match="cyclic"):
        ProcessCase.from_dict(d)
    d["metrics"]["duty"]["expression"] = {
        "op": "add",
        "args": [{"parameter": "temperature"}, {"value": 10, "unit": "W"}],
    }
    with pytest.raises(CaseValidationError, match="matching dimensions"):
        ProcessCase.from_dict(d)


def test_column_specification_dof_and_named_components():
    d = example_case("depropanizer").to_dict()
    settings = d["units"][1]["settings"]
    settings["specs"][0]["component"] = 0
    with pytest.raises(CaseValidationError, match="component"):
        ProcessCase.from_dict(d)
    settings["specs"].pop(0)
    with pytest.raises(CaseValidationError, match="spec"):
        ProcessCase.from_dict(d)


def test_reaction_elements_and_temperature_intervals():
    d = example_case("reaction").to_dict()
    d["units"][0]["settings"]["nu"] = [-1, -1, 2, 1]
    with pytest.raises(CaseValidationError, match="conserve every element"):
        ProcessCase.from_dict(d)
    d = example_case("depropanizer").to_dict()
    d["units"][0]["settings"]["min_approach"] = {"value": 15, "unit": "degC"}
    with pytest.raises(CaseValidationError, match="difference"):
        ProcessCase.from_dict(d)


def test_cost_dependencies_and_explicit_basis():
    d = example_case("depropanizer").to_dict()
    equipment = d["economics"]["equipment"][0]
    equipment["size"] = {"plant": "annual_cost"}
    with pytest.raises(CaseValidationError, match="wrong dimension"):
        ProcessCase.from_dict(d)


def test_temperature_difference_metrics_reject_affine_display_offsets():
    d = example_case().to_dict()
    expression = {
        "op": "subtract",
        "args": [
            {"stream": "product", "property": "temperature"},
            {"stream": "feed", "property": "temperature"},
        ],
    }
    d["metrics"]["rise"] = {"expression": expression, "unit": "degC"}
    with pytest.raises(CaseValidationError, match="intervals"):
        ProcessCase.from_dict(d)
    d["metrics"]["rise"]["unit"] = "delta_degF"
    assert ProcessCase.from_dict(d)
    d = example_case().to_dict()
    d["economics"]["operating_time"] = {"value": 10000, "unit": "h"}
    with pytest.raises(CaseValidationError, match="one year"):
        ProcessCase.from_dict(d)
