"""Regression cases for phase transfer, accelerated recycles, and optimization."""

from itertools import pairwise

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import Stream, heater, mix, splitter
from fugacio.sim.flowsheet import tear_solve, tear_solve_with_info
from fugacio.sim.models import package_for
from fugacio.sim.optimize import argmin, minimize
from fugacio.sim.properties import enthalpy_flow, molar_enthalpy, resolve_package, vapor_fraction
from fugacio.thermo.diagnostics import ConvergenceError


def test_accelerated_oscillatory_recycle_has_correct_sensitivity():
    def solve(th):
        return tear_solve(lambda x, p: -2.0 * x + p, jnp.array([0.0]), th, q_max=0.9, max_iter=20)

    target = jnp.array([1.0])
    assert solve(target)[0] == pytest.approx(1.0 / 3.0)
    assert jax.jacfwd(solve)(target)[0, 0] == pytest.approx(1.0 / 3.0)
    assert jax.jacrev(solve)(target)[0, 0] == pytest.approx(1.0 / 3.0)
    failed = tear_solve_with_info(lambda x, p: x + p, target, target, max_iter=3)
    assert not failed.report.converged
    with pytest.raises(ConvergenceError):
        tear_solve(lambda x, p: x + p, target, target, max_iter=3)


@pytest.mark.parametrize("bounded", [False, True])
def test_inactive_inequality_does_not_make_kkt_singular(bounded):
    def solution(th):
        return argmin(
            lambda x, p: jnp.sum((x - p) ** 2),
            jnp.array([0.0]),
            th,
            ineq_constraints=lambda x, _: -x - 10.0,
            bounds=(jnp.array([-20.0]), jnp.array([20.0])) if bounded else None,
        )

    target = jnp.array([1.0])
    assert solution(target)[0] == pytest.approx(1.0)
    assert jax.jacfwd(solution)(target)[0, 0] == pytest.approx(1.0)
    assert jax.jacrev(solution)(target)[0, 0] == pytest.approx(1.0)


def test_active_bound_and_nonlinear_constraint_share_one_kkt_system():
    def solution(th):
        return argmin(
            lambda x, p: jnp.sum((x - p) ** 2),
            jnp.array([1.0, 1.0]),
            th,
            bounds=(jnp.array([0.0, 0.0]), jnp.array([1.0, 10.0])),
            ineq_constraints=lambda x, _: jnp.array([x[1] ** 2 - 4.0]),
        )

    th = jnp.array([3.0, 3.0])
    assert jnp.allclose(solution(th), jnp.array([1.0, 2.0]), atol=1e-5)
    assert jnp.allclose(jax.jacrev(solution)(th), jnp.zeros((2, 2)), atol=1e-5)


def test_failed_optimization_cannot_supply_a_solution_gradient():
    result = minimize(lambda x, _: jnp.sum((x - 4.0) ** 2), jnp.array([0.0]), max_iter=0)
    assert not result.report.converged
    with pytest.raises(ConvergenceError):
        argmin(lambda x, th: jnp.sum((x - th) ** 2), jnp.array([0.0]), jnp.array([4.0]), max_iter=0)
    derivative = jax.jacrev(
        lambda th: argmin(lambda x, p: jnp.sum((x - p) ** 2), jnp.array([0.0]), th, max_iter=0)
    )(jnp.array([4.0]))
    assert not jnp.isfinite(derivative).all()


def test_named_package_rejects_reversed_component_order():
    pkg = package_for(["benzene", "toluene"])
    with pytest.raises(ValueError, match="component order"):
        resolve_package(["toluene", "benzene"], pkg)
    stream = Stream.from_fractions(
        ("toluene", "benzene"), jnp.array([0.3, 0.7]), 2.0, 350.0, 1e5, phase="liquid"
    )
    ordered = stream.reordered(("benzene", "toluene"))
    assert jnp.allclose(ordered.n, jnp.array([1.4, 0.6]))
    assert ordered.phase_known
    ordered.check()


def test_pure_cubic_partial_vaporization_survives_unit_handoffs():
    pkg = package_for(["water"])
    z = jnp.ones(1)
    p = 1e5
    t_sat, _ = pkg.bubble_temperature(p, z, t_min=300.0, t_max=450.0)
    hl = pkg.enthalpy(t_sat, p, z, phase="liquid")
    hv = pkg.enthalpy(t_sat, p, z, phase="vapor")
    target = 0.6 * hl + 0.4 * hv
    feed = Stream.from_fractions(("water",), z, 2.0, 300.0, p)
    duty = 2.0 * target - enthalpy_flow(feed, model=pkg)
    result = heater(feed, duty=duty, model=pkg)
    assert molar_enthalpy(result.outlet, model=pkg) == pytest.approx(float(target), abs=1e-5)
    assert vapor_fraction(result.outlet, model=pkg) == pytest.approx(0.4, abs=1e-6)
    a, b = splitter(result.outlet, jnp.array([0.25, 0.75]))
    combined = mix([a, b], model=pkg)
    assert enthalpy_flow(combined, model=pkg) == pytest.approx(
        float(enthalpy_flow(feed, model=pkg) + duty), abs=1e-4
    )
    assert vapor_fraction(combined, model=pkg) == pytest.approx(0.4, abs=1e-6)


def test_empty_stream_extensive_properties_and_invalid_phase_inventory():
    empty = Stream.from_fractions(("benzene", "toluene"), jnp.array([0.5, 0.5]), 0.0, 350.0, 1e5)
    assert enthalpy_flow(empty) == 0.0
    invalid = Stream(
        jnp.ones(2), jnp.array(300.0), jnp.array(1e5), empty.components, jnp.array([-1.0, 0.0])
    )
    assert not invalid.report.converged


def test_superheated_ph_stream_preserves_the_enthalpy_specification_derivative():
    components = ("propane", "n-butane", "n-pentane")
    pkg = package_for(components)
    z = jnp.array([0.4, 0.35, 0.25])
    h = pkg.enthalpy(391.70363314, 16e5, z, phase="vapor")

    @jax.jit
    def returned_enthalpy(specification):
        stream = Stream.from_ph(components, z, 1.0, 16e5, specification, model=pkg)
        return molar_enthalpy(stream, model=pkg)

    value, forward = jax.jvp(returned_enthalpy, (h,), (jnp.ones_like(h),))
    assert value == pytest.approx(float(h), abs=1e-4)
    assert forward == pytest.approx(1.0, abs=1e-8)
    assert jax.grad(returned_enthalpy)(h) == pytest.approx(1.0, abs=1e-8)


def test_single_phase_flash_drum_has_zero_temperature_sensitivity():
    from fugacio.sim import flash_drum

    components = ("propane", "n-butane", "n-pentane")
    feed = Stream.from_fractions(components, jnp.array([0.4, 0.35, 0.25]), 1.0, 392.0, 16e5)

    @jax.jit
    def phase_flows(temperature):
        vapor, liquid = flash_drum(feed, temperature, 16e5)
        return jnp.array([vapor.total, liquid.total])

    point = jnp.asarray(391.70363314)
    assert jnp.allclose(phase_flows(point), jnp.array([1.0, 0.0]))
    assert jnp.allclose(jax.jacrev(phase_flows)(point), jnp.zeros(2))
    assert jnp.allclose((phase_flows(point + 0.01) - phase_flows(point - 0.01)) / 0.02, 0.0)


def test_continuation_rejects_failed_trials_without_replacing_the_seed():
    from fugacio.sim import continuation_solve
    from fugacio.thermo.diagnostics import residual_report

    used = []

    def solve(parameters, previous):
        value = float(parameters)
        accepted = previous is None or value - previous <= 0.3
        used.append((value, previous, accepted))
        return value, residual_report(jnp.array([0.0 if accepted else 1.0]))

    result = continuation_solve(solve, 0.0, 1.0, initial_step=0.5)
    assert result.progress == result.value == 1.0
    assert result.report.converged
    assert any(not step.accepted for step in result.steps)
    for current, following in pairwise(used):
        if not current[2]:
            assert following[1] == current[1]


def test_continuation_never_labels_a_partial_path_as_success():
    from fugacio.sim import continuation_solve
    from fugacio.thermo.diagnostics import residual_report

    result = continuation_solve(
        lambda p, _: (p, residual_report(jnp.zeros(1))), 0.0, 1.0, max_steps=1, check=False
    )
    assert result.progress == 0.0
    assert not result.report.converged
    with pytest.raises(ConvergenceError):
        result.check()


def test_energy_flash_reports_reject_an_unreachable_specification():
    from fugacio.thermo import flash_ph_with_info

    pkg = package_for(["benzene", "toluene"])
    result = flash_ph_with_info(pkg, 1e5, 1e9, jnp.array([0.5, 0.5]), max_iter=2)
    assert not result.report.converged
    assert result.report.residual_norm > 0.1


def test_explicit_energy_flash_initial_guess_can_be_traced():
    pkg = package_for(["benzene", "toluene"])
    z = jnp.array([0.5, 0.5])
    target = pkg.mixture_enthalpy(400.0, 1e5, z)
    solve = jax.jit(lambda initial: pkg.flash_ph(1e5, target, z, t_init=initial).t)
    assert solve(jnp.asarray(350.0)) == pytest.approx(400.0, abs=1e-5)
    assert jax.grad(solve)(jnp.asarray(350.0)) == 0.0


def test_stoichiometric_reactor_can_share_the_flowsheet_energy_reference():
    from fugacio.sim.reactors import stoichiometric_reactor
    from fugacio.thermo.reactions import Reaction, reaction_arrays

    components = ("carbon monoxide", "water", "carbon dioxide", "hydrogen")
    pkg = package_for(components)
    feed = Stream.from_fractions(components, jnp.array([0.3, 0.4, 0.1, 0.2]), 10.0, 650.0, 1e6)
    reaction = Reaction(components, jnp.array([-1.0, -1.0, 1.0, 1.0]))
    result = stoichiometric_reactor(feed, reaction, conversion=0.3, adiabatic=True, model=pkg)
    hf, _, _ = reaction_arrays(list(components))
    before = enthalpy_flow(feed, model=pkg) + feed.n @ hf
    after = enthalpy_flow(result.outlet, model=pkg) + result.outlet.n @ hf
    assert float(after - before) == pytest.approx(0.0, abs=1e-3)
    assert result.outlet.t > feed.t
