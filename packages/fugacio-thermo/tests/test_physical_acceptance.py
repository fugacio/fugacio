from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo.acceptance import (
    AcceptancePolicy,
    PhysicalAcceptanceError,
    accepted_value,
    assess_flash,
    flash_pt_checked,
    require_accepted,
)
from fugacio.thermo.activity.models import NRTL
from fugacio.thermo.components import component_arrays, get
from fugacio.thermo.equilibrium import FlashResult
from fugacio.thermo.ideal import ideal_gas_coeffs
from fugacio.thermo.package import cubic_package, gamma_phi_package
from fugacio.thermo.provenance import PackageEvidence, PairEvidence


def package():
    names = ["methane", "n-butane"]
    arr = component_arrays(names)
    return cubic_package(
        arr["tc"], arr["pc"], arr["omega"], ideal_gas_coeffs([get(n) for n in names])
    )


def test_checked_flash_preserves_iteration_status_and_closes_physics():
    pkg = package()
    result = flash_pt_checked(pkg, 250.0, 1e6, jnp.array([0.5, 0.5]))
    assert result.report.accepted
    assert result.report.numerical.iterations > 0
    assert result.report.equilibrium_error < 1e-8
    assert result.report.stability_checked
    assert result.report.stability_converged
    assert result.report.minimum_tpd > -1e-7
    assert result.report.to_dict()["energy_error"] is None


def test_material_closure_cannot_hide_wrong_equilibrium_or_energy():
    pkg = package()
    z = jnp.array([0.5, 0.5])
    fake = FlashResult(jnp.array(0.5), z, z, jnp.ones(2))
    report = assess_flash(
        pkg, 250.0, 1e6, z, fake, target=1e6, policy=AcceptancePolicy(check_stability=False)
    )
    assert report.material_error == 0
    assert report.equilibrium_error > 0.01
    assert report.energy_error > 0.1
    assert not report.accepted
    with pytest.raises(PhysicalAcceptanceError):
        require_accepted(report)


def test_stalled_solver_and_missing_parameters_are_rejected():
    pkg = replace(
        package(),
        evidence=PackageEvidence(pairs=(PairEvidence(("methane", "n-butane"), "missing", "none"),)),
    )
    result = flash_pt_checked(
        pkg,
        250.0,
        1e6,
        jnp.array([0.5, 0.5]),
        max_iter=0,
        policy=AcceptancePolicy(check_stability=False),
    )
    assert not result.report.numerical.converged
    assert not result.report.applicability.parameters_available
    assert not result.report.accepted


def test_gradient_guard_handles_forward_and_reverse_mode():
    assert jax.grad(lambda x: accepted_value(x * x, jnp.array(True)))(2.0) == 4.0
    assert jnp.isnan(jax.grad(lambda x: accepted_value(x * x, jnp.array(False)))(2.0))
    assert jnp.isnan(jax.jvp(lambda x: accepted_value(x * x, jnp.array(False)), (2.0,), (1.0,))[1])


def test_two_liquid_instability_rejects_a_material_balanced_single_liquid():
    names = ["ethanol", "water"]
    arr = component_arrays(names)
    activity = NRTL(
        a=jnp.array([[0.0, 4.0], [4.0, 0.0]]),
        b=jnp.zeros((2, 2)),
        alpha=jnp.array([[0.0, 0.2], [0.2, 0.0]]),
        e=jnp.zeros((2, 2)),
    )
    pkg = gamma_phi_package(
        activity, arr["tc"], arr["pc"], arr["omega"], ideal_gas_coeffs([get(n) for n in names])
    )
    z = jnp.array([0.5, 0.5])
    fake = FlashResult(jnp.array(0.0), z, z, jnp.ones(2))
    report = assess_flash(pkg, 300.0, 1e6, z, fake)
    assert report.material_error == 0
    assert report.minimum_tpd < -0.01
    assert not report.accepted


def test_checked_energy_flashes_verify_the_specified_property():
    from fugacio.thermo.acceptance import flash_ph_checked, flash_ps_checked

    pkg = package()
    z = jnp.array([0.5, 0.5])
    policy = AcceptancePolicy(check_stability=False)
    h = pkg.mixture_enthalpy(250.0, 1e6, z)
    s = pkg.mixture_entropy(250.0, 1e6, z)
    for result in (
        flash_ph_checked(pkg, 1e6, h, z, policy=policy, t_init=250.0),
        flash_ps_checked(pkg, 1e6, s, z, policy=policy, t_init=250.0),
    ):
        assert result.report.accepted
        assert result.report.energy_checked
        assert result.report.energy_error < 1e-7
        assert result.value.t == pytest.approx(250.0, abs=1e-5)


def test_physical_acceptance_is_jittable_with_zero_components():
    pkg = package()
    solve = jax.jit(lambda z: flash_pt_checked(pkg, 400.0, 1e5, z).report)
    report = solve(jnp.array([1.0, 0.0]))
    assert report.accepted
    assert report.stability_converged
