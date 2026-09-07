import json

import jax.numpy as jnp
import pytest

from fugacio.thermo.experimental import load_corpus
from fugacio.thermo.measured_regression import (
    FitWeights,
    MeasuredFit,
    ValidationLimits,
    fit_measured_nrtl,
    validate_fit,
)
from fugacio.thermo.qualification import qualify_ethanol_water


@pytest.fixture(scope="module")
def qualified():
    return qualify_ethanol_water()


def test_independent_publication_holdout_and_parameter_covariance(qualified):
    fit, report = qualified
    assert fit.diagnostics.converged
    assert fit.diagnostics.jacobian_rank == 4
    assert fit.diagnostics.covariance is not None
    assert fit.diagnostics.standard_errors is not None
    assert report["accepted"]
    assert report["metrics"]["pressure_relative_rmse"] < 0.05
    assert report["metrics"]["vapor_fraction_rmse"] < 0.05
    assert report["counts"]["pressure_relative_rmse"] == 72
    assert set(fit.training_ids).isdisjoint(report["holdout_ids"])
    assert fit.sources[0][0] == "10.1016/j.fluid.2011.06.009"


def test_fit_round_trip_and_component_orientation(tmp_path, qualified):
    fit, _ = qualified
    path = tmp_path / "fit.json"
    fit.save(path)
    loaded = MeasuredFit.load(path)
    assert loaded == fit
    x = jnp.array([0.3, 0.7])
    expected = fit.model().ln_gamma(x, 330.0)
    reverse = fit.model(fit.components[::-1]).ln_gamma(x[::-1], 330.0)
    assert reverse[::-1] == pytest.approx(expected)
    assert json.loads(path.read_text())["evidence"] == "measured_fit"


def test_validation_cannot_reuse_training_rows(qualified):
    fit, _ = qualified
    train = tuple(o for o in load_corpus() if o.id in fit.training_ids)
    with pytest.raises(ValueError, match="leakage"):
        validate_fit(fit, train)


def test_multi_property_fit_uses_gibbs_helmholtz_consistently():
    observations = tuple(o for o in load_corpus() if set(o.components) == {"methanol", "nmp"})
    fit = fit_measured_nrtl(observations)
    assert {o.kind for o in observations} == {"vle", "excess_enthalpy"}
    assert fit.diagnostics.jacobian_rank == 4
    assert fit.diagnostics.converged
    assert fit.diagnostics.weighted_rmse < 5


def test_single_temperature_cannot_identify_four_vle_parameters():
    observations = tuple(
        o
        for o in load_corpus()
        if set(o.components) == {"benzene", "toluene"} and o.temperature == 333.15
    )
    fit = fit_measured_nrtl(observations)
    assert not fit.diagnostics.converged
    assert fit.diagnostics.reason == "unidentifiable_parameters"
    assert fit.diagnostics.covariance is None


def test_cloud_points_and_invalid_weights_fail_explicitly():
    cloud = tuple(o for o in load_corpus() if o.kind == "cloud_point")[:1]
    with pytest.raises(ValueError, match="cloud points"):
        fit_measured_nrtl(cloud)
    with pytest.raises(ValueError):
        FitWeights(pressure_relative=0)
    with pytest.raises(ValueError):
        ValidationLimits(vapor_fraction_rmse=float("nan"))
