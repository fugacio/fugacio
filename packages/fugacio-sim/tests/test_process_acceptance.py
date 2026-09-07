from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import Stream, package_for
from fugacio.sim.acceptance import audit_balance, heater_checked
from fugacio.thermo.acceptance import AcceptancePolicy, flash_pt_checked
from fugacio.thermo.provenance import PackageEvidence, assess_applicability


def test_missing_activity_pairs_require_explicit_assumption():
    names = ["water", "n-heptane"]
    with pytest.raises(KeyError, match="no curated NRTL"):
        package_for(names, "nrtl")
    pkg = package_for(names, "nrtl", parameter_policy="allow_ideal")
    assert pkg.evidence.pairs[0].kind == "assumed_zero"
    assert pkg.evidence.assumptions
    assert pkg.activity.ln_gamma(jnp.array([0.5, 0.5]), 330.0) == pytest.approx([0.0, 0.0])
    assert package_for(names, "unifac").evidence.pairs[0].kind == "predictive"


def test_evidence_is_static_metadata_and_arrays_stay_differentiable():
    pkg = package_for(["ethanol", "water"], "nrtl")
    leaves, tree = jax.tree.flatten(pkg)
    assert all(isinstance(v, jax.Array) for v in leaves)
    assert jax.tree.unflatten(tree, leaves).evidence == pkg.evidence
    slope = jax.grad(lambda t: jnp.sum(pkg.activity.ln_gamma(jnp.array([0.3, 0.7]), t)))(330.0)
    assert jnp.isfinite(slope)


def test_outside_observed_bounds_is_visible_and_requires_opt_in():
    pkg = package_for(["methane", "n-butane"])
    pkg = replace(pkg, evidence=replace(pkg.evidence, temperature_range=(240.0, 245.0)))
    policy = AcceptancePolicy(check_stability=False)
    rejected = flash_pt_checked(pkg, 250.0, 1e6, jnp.array([0.5, 0.5]), policy=policy)
    assert not rejected.report.accepted
    allowed = flash_pt_checked(
        pkg, 250.0, 1e6, jnp.array([0.5, 0.5]), policy=replace(policy, allow_extrapolation=True)
    )
    assert allowed.report.accepted
    assert not allowed.report.applicability.within_temperature
    unknown = assess_applicability(PackageEvidence(), 250.0, 1e6, jnp.array([0.5, 0.5]))
    assert not unknown.temperature_known


def test_checked_heater_closes_energy_and_balance_detects_wrong_duty():
    pkg = package_for(["ethanol", "water"], "nrtl")
    feed = Stream.from_fractions(("ethanol", "water"), jnp.array([0.5, 0.5]), 10.0, 330.0, 20000.0)
    result = heater_checked(feed, model=pkg, t_out=335.0)
    assert result.accepted
    result.check()
    wrong = audit_balance((feed,), (result.value.outlet,), pkg, heat=result.value.duty + 1000.0)
    assert not wrong.converged


def test_checked_valve_propagates_explicit_extrapolation_policy():
    from fugacio.sim.acceptance import valve_checked

    pkg = package_for(["methane"])
    pkg = replace(pkg, evidence=replace(pkg.evidence, pressure_range=(9e5, 11e5)))
    feed = Stream.from_fractions(("methane",), jnp.ones(1), 1.0, 300.0, 1e6)
    result = valve_checked(
        feed,
        8e5,
        model=pkg,
        policy=AcceptancePolicy(allow_extrapolation=True, check_stability=False),
    )
    assert result.accepted
    assert not result.states[-1].applicability.within_pressure
