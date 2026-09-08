from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim.cases import (
    CaseAcceptanceError,
    CaseRun,
    CaseRunner,
    CaseWorkspace,
    ProcessCase,
    SolverOptions,
    compare_runs,
)
from fugacio.sim.cases.examples import example_case
from fugacio.sim.cases.jsonio import write_json


@pytest.fixture(scope="module")
def heater_runner():
    return CaseRunner(example_case())


@pytest.fixture(scope="module")
def heater_run(heater_runner):
    return heater_runner.run(check=True)


def test_run_retains_transfers_and_separate_acceptance(heater_run):
    d = heater_run.to_dict()
    assert d["units"]["heater"]["heat_w"] == pytest.approx(2041.2998021092956)
    assert d["metrics"]["duty"]["value"] == pytest.approx(2.0412998021092956)
    assert d["qualification"]["status"] == "not_evaluated"
    assert d["qualification"]["accepted"] is None
    assert d["parameter_evidence"]["temperature_range"] is None
    assert set(d["checks"]["physical"]["boundaries"]) == {"heater", "plant"}
    assert d["plant_si"]["cooling"] == 0
    # 8 USD/GJ * duty * 8000 h, with no annual-rate unit confusion.
    assert d["metrics"]["annual_cost"]["value"] == pytest.approx(
        8e-9 * 2041.2998021092956 * 8000 * 3600
    )
    assert "2041.2998" in heater_run.markdown()
    assert "not_evaluated" in heater_run.markdown()


def test_backends_agree(heater_run):
    eo = CaseRunner(example_case(), options=SolverOptions(backend="eo")).run(check=True)
    assert eo.to_dict()["metrics"] == heater_run.to_dict()["metrics"]
    assert eo.to_dict()["checks"]["numerical"]["reports"]["eo"]["converged"]


def test_study_initialization_requires_accepted_matching_revision(heater_runner, heater_run):
    initialization = heater_runner.initialization(heater_run)
    evaluate = jax.jit(
        lambda t: heater_runner.evaluate({"temperature": t}, initialization=initialization).metrics[
            "duty"
        ]
    )
    assert float(evaluate(350.0)) == pytest.approx(
        heater_run.to_dict()["metrics"]["duty"]["value_si"]
    )
    assert float(jax.grad(evaluate)(350.0)) > 0
    # Explicit seeds neither change the runner defaults nor alter later runs.
    evaluate(360.0)
    assert heater_runner.run().run_id == heater_run.run_id
    changed = heater_runner.case.with_parameters({"temperature": {"value": 360, "unit": "K"}})
    with pytest.raises(ValueError, match="same case revision"):
        CaseRunner(changed).initialization(heater_run)
    failed = CaseRunner(example_case("recycle"), options=SolverOptions(max_iterations=0)).run()
    assert not failed.accepted
    with pytest.raises(ValueError, match="accepted case run"):
        heater_runner.initialization(failed)


def test_persistence_integrity_revision_and_replay(tmp_path, heater_runner, heater_run):
    workspace = CaseWorkspace(tmp_path)
    identity = workspace.save_run(heater_run)
    assert workspace.save_run(heater_run) == identity
    assert workspace.load_run(identity) == heater_run
    assert workspace.replay(identity).to_dict()["metrics"] == heater_run.to_dict()["metrics"]
    assert workspace.list_cases() == [{"case_id": heater_runner.case.case_id, "name": "heater"}]
    changed = heater_runner.run({"temperature": {"value": 360, "unit": "K"}}, check=True)
    comparison = compare_runs(heater_run, changed)
    assert comparison["metrics"]["duty"]["delta_si"] > 0
    assert comparison["parameters"]["temperature"]["candidate_si"] == 360
    assert changed.run_id != identity
    data = heater_run.to_dict()
    data["metrics"]["duty"]["value"] = 0
    with pytest.raises(ValueError, match="hash"):
        CaseRun.from_dict(data)
    path = tmp_path / "artifacts" / (identity + ".json")
    write_json(path, data)
    with pytest.raises(ValueError, match="hash"):
        workspace.load_run(identity)
    with pytest.raises(ValueError, match="64"):
        workspace.load_case("../../etc/passwd")


def test_comparison_rejects_redefined_metric(heater_run):
    d = example_case().to_dict()
    d["metrics"]["duty"]["expression"] = {
        "op": "negate",
        "args": [{"unit": "heater", "property": "heat"}],
    }
    other = CaseRunner(ProcessCase.from_dict(d)).run(check=True)
    with pytest.raises(ValueError, match="changed meaning"):
        compare_runs(heater_run, other)


def test_recycle_conserves_external_material_and_retains_cooling():
    run = CaseRunner(example_case("recycle"), options=SolverOptions(recycle_method="broyden")).run(
        check=True
    )
    d = run.to_dict()
    assert d["streams"]["product"]["flow_mol_s"] == pytest.approx(1, abs=1e-8)
    assert d["streams"]["heated"]["flow_mol_s"] == pytest.approx(1 / 0.6, abs=1e-8)
    assert d["plant_si"]["heating"] > d["plant_si"]["heat"]
    assert d["units"]["mixer"]["cooling_w"] > 0
    assert any(k.startswith("recycle:") for k in d["checks"]["numerical"]["reports"])


def test_recycle_reuses_maps_with_dynamic_feeds_and_operating_points():
    document = example_case("recycle").to_dict()
    document["parameters"]["flow"] = {"value": 1, "unit": "mol/s", "lower": 0.5, "upper": 3}
    document["feeds"]["feed"]["flow"] = {"parameter": "flow"}
    runner = CaseRunner(
        ProcessCase.from_dict(document), options=SolverOptions(recycle_method="broyden")
    )
    baseline = runner.run(check=True)
    changed = runner.run(
        {"flow": {"value": 2, "unit": "mol/s"}, "temperature": {"value": 360, "unit": "K"}},
        check=True,
    ).to_dict()
    assert changed["streams"]["product"]["flow_mol_s"] == pytest.approx(2, abs=1e-8)
    assert changed["streams"]["mixed"]["flow_mol_s"] == pytest.approx(2 / 0.6, abs=1e-8)
    assert (
        changed["units"]["heater"]["heat_w"] > 2 * baseline.to_dict()["units"]["heater"]["heat_w"]
    )
    seed = runner.initialization(baseline)

    def product_flow(flow):
        return jnp.sum(runner.evaluate({"flow": flow}, initialization=seed).streams["product"].n)

    for flow in (1.0, 2.0):
        value, derivative = jax.jvp(product_flow, (jnp.asarray(flow),), (jnp.asarray(1.0),))
        assert float(value) == pytest.approx(flow, abs=1e-8)
        assert float(derivative) == pytest.approx(1, abs=1e-8)
    assert runner.run(check=True).run_id == baseline.run_id


def test_reactor_retains_generation_and_formation_energy():
    run = CaseRunner(example_case("reaction")).run(check=True)
    d = run.to_dict()
    assert d["metrics"]["benzene_production"]["value"] == pytest.approx(1.8)
    assert d["units"]["reactor"]["component_generation_mol_s"] == pytest.approx(
        [-1.8, -1.8, 1.8, 1.8]
    )
    assert d["checks"]["physical"]["boundaries"]["plant"]["element_relative_error"] < 1e-10
    assert d["units"]["reactor"]["heat_w"] < 0


def test_design_spec_and_implicit_derivative():
    d = example_case().to_dict()
    d["parameters"]["flow"] = {"value": 1, "unit": "mol/s", "lower": 0.5, "upper": 2}
    d["feeds"]["feed"]["flow"] = {"parameter": "flow"}
    d["specifications"] = [
        {
            "name": "target_duty",
            "parameter": "temperature",
            "metric": "duty",
            "target": {"value": 2.5, "unit": "kW"},
            "tolerance": {"value": 0.001, "unit": "W"},
        }
    ]
    runner = CaseRunner(ProcessCase.from_dict(d))
    run = runner.run(check=True)
    assert run.to_dict()["metrics"]["duty"]["value_si"] == pytest.approx(2500, abs=0.001)
    assert run.to_dict()["parameters"]["temperature"]["value_si"] > 360
    derivative = jax.grad(lambda f: runner.evaluate({"flow": f}).parameters["temperature"])(
        jnp.asarray(1.0)
    )
    low = runner.run({"flow": {"value": 0.999, "unit": "mol/s"}}, check=True)
    high = runner.run({"flow": {"value": 1.001, "unit": "mol/s"}}, check=True)
    fd = (
        high.to_dict()["parameters"]["temperature"]["value_si"]
        - low.to_dict()["parameters"]["temperature"]["value_si"]
    ) / 0.002
    assert float(derivative) == pytest.approx(fd, rel=1e-5)
    compiled = jax.jit(jax.grad(lambda f: runner.evaluate({"flow": f}).parameters["temperature"]))
    assert float(compiled(jnp.asarray(1.0))) == pytest.approx(float(derivative), rel=1e-8)
    seed = runner.initialization(run)
    warm = jax.jit(
        jax.grad(
            lambda f: runner.evaluate({"flow": f}, initialization=seed).parameters["temperature"]
        )
    )
    assert float(warm(jnp.asarray(1.0))) == pytest.approx(float(derivative), rel=1e-8)


def test_infeasible_design_spec_is_saved_failure():
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
    run = CaseRunner(
        ProcessCase.from_dict(d), options=SolverOptions(specification_iterations=2)
    ).run()
    assert not run.accepted
    assert not run.to_dict()["checks"]["specifications"]["accepted"]
    with pytest.raises(CaseAcceptanceError) as error:
        run.check()
    assert error.value.run == run


def test_empty_split_product_has_no_fake_composition():
    d = example_case().to_dict()
    d["units"].append(
        {
            "name": "split",
            "kind": "splitter",
            "inlets": ["product"],
            "outlets": ["full", "empty"],
            "settings": {"fractions": [1, 0]},
        }
    )
    run = CaseRunner(ProcessCase.from_dict(d)).run(check=True)
    empty = run.to_dict()["streams"]["empty"]
    assert empty["flow_mol_s"] == 0
    assert empty["mole_fractions"] is None
    assert empty["enthalpy_flow_w"] == 0
    d["metrics"]["bad_fraction"] = {
        "expression": {"stream": "empty", "property": "mole_fraction", "component": "methane"},
        "unit": "1",
    }
    invalid = CaseRunner(ProcessCase.from_dict(d)).run()
    assert not invalid.accepted
    assert invalid.to_dict()["metrics"]["bad_fraction"]["value"] is None


@pytest.mark.parametrize(
    "kind,settings",
    [
        ("compressor", {"p_out": {"value": 2, "unit": "bar"}, "efficiency": 0.8}),
        ("turbine", {"p_out": {"value": 0.5, "unit": "bar"}, "efficiency": 0.8}),
        ("valve", {"p_out": {"value": 0.5, "unit": "bar"}}),
    ],
)
def test_mechanical_units_retain_energy(kind, settings):
    d = example_case().to_dict()
    d["units"][0].update(kind=kind, settings=settings)
    d["metrics"] = {"work": {"expression": {"unit": "heater", "property": "work"}, "unit": "W"}}
    run = CaseRunner(ProcessCase.from_dict(d)).run(check=True)
    work = run.to_dict()["metrics"]["work"]["value"]
    assert work > 0 if kind == "compressor" else work < 0 if kind == "turbine" else work == 0


def test_economics_outside_correlation_range_is_not_accepted():
    d = example_case().to_dict()
    d["economics"]["equipment"] = [
        {"name": "tank", "kind": "vessel", "size": {"value": 1e-12, "unit": "m3"}, "cepci": 800}
    ]
    run = CaseRunner(ProcessCase.from_dict(d)).run()
    assert run.to_dict()["checks"]["physical"]["accepted"]
    assert not run.to_dict()["checks"]["economics"]["accepted"]
    assert not run.accepted


def test_pump_work_and_pressure_are_retained():
    d = example_case().to_dict()
    d["components"] = ["propane", "n-butane"]
    d["feeds"]["feed"].update(
        temperature={"value": 250, "unit": "K"},
        pressure={"value": 10, "unit": "bar"},
        phase="liquid",
    )
    d["units"][0].update(kind="pump", settings={"p_out": {"value": 20, "unit": "bar"}})
    d["metrics"] = {"work": {"expression": {"unit": "heater", "property": "work"}, "unit": "W"}}
    run = CaseRunner(ProcessCase.from_dict(d)).run(check=True)
    assert run.to_dict()["metrics"]["work"]["value"] > 0
    assert run.to_dict()["streams"]["product"]["pressure_pa"] == pytest.approx(20e5)


def test_flash_phase_products_agree_between_backends():
    d = example_case().to_dict()
    d["feeds"]["feed"].update(
        temperature={"value": 180, "unit": "K"}, pressure={"value": 10, "unit": "bar"}
    )
    d["units"][0].update(
        kind="flash",
        outlets=["vapor", "liquid"],
        settings={"t": {"value": 180, "unit": "K"}, "p": {"value": 10, "unit": "bar"}},
    )
    d["metrics"] = {
        "vapor_flow": {"expression": {"stream": "vapor", "property": "flow"}, "unit": "mol/s"}
    }
    case = ProcessCase.from_dict(d)
    a = CaseRunner(case).run(check=True)
    b = CaseRunner(case, options=SolverOptions(backend="eo")).run(check=True)
    vapor = b.to_dict()["streams"]["vapor"]
    assert vapor["vapor_component_flow_mol_s"] == vapor["component_flow_mol_s"]
    assert vapor["vapor_fraction"] == 1.0
    assert b.to_dict()["metrics"]["vapor_flow"]["value"] == pytest.approx(
        a.to_dict()["metrics"]["vapor_flow"]["value"], rel=1e-8
    )
    assert b.to_dict()["streams"]["vapor"]["vapor_fraction"] == pytest.approx(1)
    assert b.to_dict()["streams"]["liquid"]["vapor_fraction"] == pytest.approx(0)


def test_pure_fluid_eo_retains_saturation_quality():
    d = example_case().to_dict()
    d["components"] = ["methane"]
    d["feeds"]["feed"].update(z=[1], temperature={"value": 105, "unit": "K"}, phase="liquid")
    d["units"][0]["settings"] = {"duty": {"value": 4000, "unit": "W"}}
    d["metrics"] = {
        "quality": {"expression": {"stream": "product", "property": "vapor_fraction"}, "unit": "1"}
    }
    case = ProcessCase.from_dict(d)
    a = CaseRunner(case).run(check=True)
    b = CaseRunner(case, options=SolverOptions(backend="eo")).run(check=True)
    quality = a.to_dict()["metrics"]["quality"]["value"]
    assert 0 < quality < 1
    assert b.to_dict()["metrics"]["quality"]["value"] == pytest.approx(quality, abs=1e-7)


def test_exchanger_retains_profiles_and_independent_heat_balance():
    d = example_case().to_dict()
    d["feeds"]["hot"] = {**d["feeds"]["feed"], "temperature": {"value": 400, "unit": "K"}}
    d["units"] = [
        {
            "name": "exchanger",
            "kind": "heat_exchanger",
            "inlets": ["hot", "feed"],
            "outlets": ["cooled", "heated"],
            "settings": {
                "min_approach": {"value": 20, "unit": "delta_K"},
                "u": {"value": 200, "unit": "W/(m2 K)"},
            },
        }
    ]
    d["metrics"] = {"duty": {"expression": {"unit": "exchanger", "property": "duty"}, "unit": "W"}}
    run = CaseRunner(ProcessCase.from_dict(d)).run(check=True)
    unit = run.to_dict()["units"]["exchanger"]
    assert unit["heat_w"] == 0
    assert unit["quantities_si"]["duty"] > 0
    assert unit["quantities_si"]["area"] > 0
    assert len(unit["profiles_si"]["hot_temperature"]) >= 2


def test_pressure_raising_valve_is_not_a_physically_accepted_design():
    d = example_case().to_dict()
    d["units"][0].update(kind="valve", settings={"p_out": {"value": 2, "unit": "bar"}})
    run = CaseRunner(ProcessCase.from_dict(d)).run()
    assert not run.accepted
    limits = run.to_dict()["checks"]["unit_operating_limits"]
    assert not limits["accepted"]
    assert "pressure" in limits["units"]["heater"]["issues"][0]
