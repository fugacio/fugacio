"""Equation-oriented failure, phase-regime, and warm-start acceptance cases."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import Stream, enthalpy_flow, package_for
from fugacio.sim.eo import EOFlowsheet, Flash, Heater, Splitter
from fugacio.thermo.diagnostics import ConvergenceError


@pytest.mark.parametrize("temperature", [220.0, 320.0, 650.0])
def test_eo_flash_handles_present_and_absent_phases(temperature):
    feed = Stream.from_fractions(
        ("propane", "n-butane"), jnp.array([0.5, 0.5]), 10.0, temperature, 5e5
    )
    fs = (
        EOFlowsheet()
        .feed("feed", feed)
        .add(Flash(inlets=("feed",), outlets=("vapor", "liquid"), t="T", p=5e5))
    )
    result = fs.solve({"T": temperature})
    assert result.report.converged
    assert jnp.allclose(result["vapor"].n + result["liquid"].n, feed.n, atol=1e-8)
    assert result["vapor"].phase_known and result["liquid"].phase_known
    if temperature == 220.0:
        assert result["vapor"].total == pytest.approx(0.0, abs=1e-8)
    if temperature == 650.0:
        assert result["liquid"].total == pytest.approx(0.0, abs=1e-8)


def test_eo_reports_failed_iterations_and_reuses_converged_guesses():
    feed = Stream.from_fractions(("benzene", "toluene"), jnp.array([0.4, 0.6]), 10.0, 300.0, 1e5)
    fs = (
        EOFlowsheet()
        .feed("feed", feed)
        .add(Heater(inlets=("feed",), outlets=("heated",), t_out="T"))
    )
    bad = fs.solve({"T": 450.0}, sweeps=0, max_iter=0, check=False)
    assert not bad.converged
    with pytest.raises(ConvergenceError, match="heated"):
        fs.solve({"T": 450.0}, sweeps=0, max_iter=0)
    first = fs.solve({"T": 450.0})
    second = fs.solve({"T": 450.0})
    assert first.converged and second.converged
    assert second.report.iterations == 0
    diagnostic = fs.diagnose({"T": 450.0}, guess=second.streams)
    assert diagnostic["full_rank"]
    assert diagnostic["rank"] == diagnostic["n_unknowns"]


def test_eo_pure_ph_coordinate_preserves_wet_quality_through_splitter():
    pkg = package_for(["water"])
    z = jnp.ones(1)
    pressure = 1e5
    ts, _ = pkg.bubble_temperature(pressure, z, t_min=300.0, t_max=450.0)
    hl = pkg.enthalpy(ts, pressure, z, phase="liquid")
    hv = pkg.enthalpy(ts, pressure, z, phase="vapor")
    feed = Stream.from_ph(("water",), z, 2.0, pressure, 0.75 * hl + 0.25 * hv, model=pkg)
    fs = EOFlowsheet(model=pkg).feed("feed", feed)
    fs.add(Heater(inlets=("feed",), outlets=("heated",), duty="Q"))
    fs.add(Splitter(inlets=("heated",), outlets=("a", "b"), fractions=jnp.array([0.3, 0.7])))
    duty = 0.5 * (hv - hl)
    result = fs.solve({"Q": duty})
    assert result.converged
    energy_out = enthalpy_flow(result["a"], model=pkg) + enthalpy_flow(result["b"], model=pkg)
    assert energy_out == pytest.approx(float(enthalpy_flow(feed, model=pkg) + duty), abs=1e-3)
    derivative = jax.jacfwd(lambda q: enthalpy_flow(fs.solve({"Q": q})["a"], model=pkg))(duty)
    assert derivative == pytest.approx(0.3, abs=1e-5)


def test_eo_model_values_remain_dynamic_in_cached_plan():
    feed = Stream.from_fractions(("benzene", "toluene"), jnp.array([0.4, 0.6]), 10.0, 300.0, 1e5)
    pkg = package_for(feed.components)
    fs = (
        EOFlowsheet(model=pkg)
        .feed("feed", feed)
        .add(Heater(inlets=("feed",), outlets=("hot",), duty=1e5))
    )
    first = fs.solve()
    fs.model = replace(pkg, cp=tuple(1.5 * a for a in pkg.cp))
    second = fs.solve()
    reference = (
        EOFlowsheet(model=fs.model)
        .feed("feed", feed)
        .add(Heater(inlets=("feed",), outlets=("hot",), duty=1e5))
        .solve()
    )
    assert abs(float(first["hot"].t - second["hot"].t)) > 0.1
    assert second["hot"].t == pytest.approx(float(reference["hot"].t), abs=1e-6)
