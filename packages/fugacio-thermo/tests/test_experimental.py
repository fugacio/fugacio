import hashlib
import json
from dataclasses import replace

import pytest

from fugacio.thermo.experimental import (
    _ROOT,
    corpus_manifest,
    grouped_split,
    load_corpus,
    normalize_document,
)
from fugacio.thermo.thermoml import Column, Dataset, Uncertainty, convert_values, loads


def test_vendored_corpus_identity_units_and_coverage():
    corpus = load_corpus()
    assert len(corpus) == 447
    assert len({o.components for o in corpus}) == 20
    assert {o.kind for o in corpus} == {"vle", "excess_enthalpy", "cloud_point"}
    assert len({o.source for o in corpus}) == 7
    for obs in corpus:
        assert obs.temperature > 250
        assert obs.pressure > 1000
        assert all(obs.cas)
        assert len(obs.source_sha256) == 64
    json.dumps([o.to_dict() for o in corpus], allow_nan=False)


def test_complementary_phase_tables_join_by_conditions_not_row_order():
    manifest = corpus_manifest()
    source = manifest["sources"][0]
    raw = _ROOT.joinpath(source["file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == source["sha256"]
    data = loads(raw)
    vapor = data.datasets[1]
    reversed_table = replace(vapor, rows=vapor.rows[::-1], uncertainties=vapor.uncertainties[::-1])
    shuffled = replace(data, datasets=(data.datasets[0], reversed_table))
    a = normalize_document(data, manifest["identities"], (1, 2))
    b = normalize_document(shuffled, manifest["identities"], (1, 2))
    assert a == b
    assert all(o.dataset_ids == (1, 2) for o in a)
    assert any(o.composition("Liquid") != o.composition("Gas") for o in a)
    with pytest.raises(ValueError, match="expected one"):
        a[0].measurement("mole_fraction")


def test_ambiguous_repeated_conditions_require_explicit_exclusions():
    manifest = corpus_manifest()
    source = manifest["sources"][1]
    data = loads(_ROOT.joinpath(source["file"]).read_bytes())
    with pytest.raises(ValueError, match="duplicate experimental conditions"):
        normalize_document(data, manifest["identities"], (2, 3))
    assert (
        len(normalize_document(data, manifest["identities"], (2, 3), source["excluded_rows"])) == 72
    )


def test_constraints_and_energy_uncertainty_convert_together():
    corpus = load_corpus()
    obs = next(o for o in corpus if o.kind == "excess_enthalpy")
    assert obs.measurement("temperature").role == "constraint"
    assert obs.measurement("pressure").role == "constraint"
    he = obs.measurement("excess_enthalpy")
    assert he.unit == "J/mol"
    assert abs(he.value) > 10
    assert he.uncertainty is not None


def test_uncertainty_never_infers_coverage_factor():
    assert Uncertainty(expanded=2.0, confidence_percent=95.0).standard_value is None
    assert Uncertainty(expanded=2.0, coverage_factor=2.0).standard_value == 1.0
    col = Column(1, "property", "ePropName", "Temperature, degC")
    ds = Dataset((1,), (col,), ((20.0,),), uncertainties=((Uncertainty(standard=0.5),),))
    assert ds.values_in(col, "K") == pytest.approx((293.15,))
    assert ds.standard_uncertainties(col, "K") == (0.5,)
    with pytest.raises(ValueError, match="unsupported unit"):
        convert_values((1.0,), "kg/kg", "1")
    with pytest.raises(ValueError, match="unsupported presentation"):
        Dataset((1,), (replace(col, presentation="ln(X)"),), ((1.0,),)).validate()


def test_conflicting_identifiers_and_incomplete_rows_rejected():
    manifest = corpus_manifest()
    data = loads(_ROOT.joinpath(manifest["sources"][0]["file"]).read_bytes())
    bad = replace(data.compounds[0], cas="64-17-5")
    with pytest.raises(ValueError, match="conflicting CAS"):
        normalize_document(
            replace(data, compounds=(bad, data.compounds[1])), manifest["identities"], (1, 2)
        )
    ds = replace(data.datasets[0], rows=((float("nan"),) * len(data.datasets[0].columns),))
    with pytest.raises(ValueError, match="missing or nonfinite"):
        ds.validate()
    with pytest.raises(ValueError, match="entity"):
        loads('<!DOCTYPE x [<!ENTITY a "x">]><DataReport/>')


@pytest.mark.parametrize("by", ["source", "temperature", "dataset"])
def test_holdouts_are_explicit_and_keep_joined_rows_together(by):
    corpus = load_corpus()
    obs = corpus[0]
    key = {
        "source": obs.source,
        "temperature": format(obs.temperature, ".12g"),
        "dataset": f"{obs.source}#{obs.dataset_ids[0]}",
    }[by]
    train, test = grouped_split(corpus, by=by, holdout=(key,))
    assert {o.id for o in train}.isdisjoint({o.id for o in test})
    assert len(train) + len(test) == len(corpus)
    with pytest.raises(ValueError, match="unknown holdout"):
        grouped_split(corpus, by=by, holdout=("does-not-exist",))


def test_dataset_holdout_closes_over_partially_joined_tables():
    base = load_corpus()[0]
    observations = (
        replace(base, id="one", dataset_ids=(1, 2)),
        replace(base, id="two", dataset_ids=(2, 3)),
        replace(base, id="three", dataset_ids=(9,)),
    )
    train, test = grouped_split(observations, by="dataset", holdout=(f"{base.source}#1",))
    assert [o.id for o in train] == ["three"]
    assert {o.id for o in test} == {"one", "two"}
