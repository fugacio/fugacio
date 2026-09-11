"""Reactive MESH closure, zero-volume limit, and structured implicit derivatives."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fugacio.sim.cases import CaseRunner
from fugacio.sim.cases.examples import example_case
from fugacio.sim.cases.reactions import build_reaction_set
from fugacio.sim.distillation import ColumnFeed, ColumnSpec, rigorous_column
from fugacio.sim.properties import enthalpy_flow


@pytest.fixture(scope="module")
def problem():
    runner = CaseRunner(example_case("reactive-separation"))
    rx = build_reaction_set(
        runner.units[0].structure["reactions"], runner.case.components, runner.defaults
    )
    inlet = runner.feeds[0].build(runner.case.components, runner.defaults, runner.package)
    return inlet, rx, runner.package


def solve(problem, volume, *, linear_solver="block", reactive=True, rate_scale=1.0):
    inlet, rx, pkg = problem
    rx = replace(rx, rate_laws=(replace(rx.rate_laws[0], k_forward=rate_scale),))
    return rigorous_column(
        [ColumnFeed(inlet, 3)],
        6,
        p=3e5,
        model=pkg,
        specs=[ColumnSpec("reflux_ratio", 3.0), ColumnSpec("distillate_rate", 5.0)],
        reactions=rx if reactive else None,
        reaction_volumes=volume if reactive else 0.0,
        linear_solver=linear_solver,
        check=False,
    )


def test_reactive_mesh_material_elements_energy_and_stage_sources(problem):
    inlet, rx, pkg = problem
    r = solve(problem, 0.3)
    assert r.report.converged
    generation = jnp.sum(r.generation, axis=0)
    np.testing.assert_allclose(r.distillate.n + r.bottoms.n, inlet.n + generation, atol=1e-7)
    np.testing.assert_allclose(rx.atoms @ generation, 0.0, atol=1e-10)
    heat = r.condenser_duty + r.reboiler_duty
    predicted = (
        enthalpy_flow(r.distillate, model=pkg)
        + enthalpy_flow(r.bottoms, model=pkg)
        - enthalpy_flow(inlet, model=pkg)
        + generation @ rx.formation_enthalpy
    )
    assert heat == pytest.approx(predicted, rel=1e-7, abs=1e-4)
    np.testing.assert_allclose(r.reaction_heat, -r.generation @ rx.formation_enthalpy)
    assert r.generation[:, 1].sum() > 0
    assert r.reaction_volumes[0] == r.reaction_volumes[-1] == 0


def test_zero_volume_matches_nonreactive_mesh(problem):
    zero, plain = solve(problem, 0.0), solve(problem, 0.0, reactive=False)
    assert zero.report.converged and plain.report.converged
    np.testing.assert_allclose(zero.distillate.n, plain.distillate.n, rtol=1e-8)
    np.testing.assert_allclose(zero.t, plain.t, rtol=1e-8)
    np.testing.assert_array_equal(zero.generation, 0.0)


def test_structured_and_dense_reactive_derivatives_match_fd(problem):
    def objective(v, mode):
        r = solve(problem, v, linear_solver=mode, rate_scale=1.0 + 0.1 * v)
        return r.distillate.n[1]

    block = jax.jit(lambda v: objective(v, "block"))
    dense = jax.jit(lambda v: objective(v, "dense"))
    assert block(0.3) == pytest.approx(dense(0.3), rel=1e-8)
    fd = (block(0.3001) - block(0.2999)) / 0.0002
    ad = jax.jit(jax.grad(block))(0.3)
    assert ad == pytest.approx(fd, rel=2e-4)
    assert ad == pytest.approx(jax.jit(jax.grad(dense))(0.3), rel=2e-7)


def test_negative_reacting_volume_is_rejected(problem):
    result = solve(problem, -0.01)
    assert not result.report.converged


def test_non_equimolar_reactive_mesh_conserves_material_and_energy():
    from fugacio.sim import ReactionSet, ReferenceRate, Stream
    from fugacio.sim.properties import resolve_package
    from fugacio.thermo import Reaction

    # Illustrative liquid ethylene dimerization: 2 C2H4 -> C4H8.
    names = ("ethylene", "1-butene")
    rx = ReactionSet.from_reactions(
        Reaction(names, jnp.array([-2.0, 1.0])),
        [
            ReferenceRate(
                jnp.array(0.02),
                jnp.array(0.0),
                jnp.array([2.0, 0.0]),
                jnp.array(0.0),
                jnp.array(0.0),
                jnp.array([0.0, 1.0]),
                jnp.array(250.0),
            )
        ],
        phase="liquid",
        rate_basis="activity",
    )
    inlet = Stream.from_fractions(names, [0.5, 0.5], 10.0, 220.0, 5e5)
    pkg = resolve_package(names)
    result = rigorous_column(
        [ColumnFeed(inlet, 3)],
        6,
        p=5e5,
        model=pkg,
        specs=[ColumnSpec("reflux_ratio", 2.0), ColumnSpec("distillate_rate", 4.0)],
        reactions=rx,
        reaction_volumes=0.1,
        check=False,
    )
    assert result.report.converged
    generation = jnp.sum(result.generation, axis=0)
    assert jnp.sum(generation) < 0
    np.testing.assert_allclose(
        result.distillate.n + result.bottoms.n, inlet.n + generation, atol=1e-6
    )
    hout = enthalpy_flow(result.distillate, model=pkg) + enthalpy_flow(result.bottoms, model=pkg)
    heat = hout - enthalpy_flow(inlet, model=pkg) + generation @ rx.formation_enthalpy
    assert heat == pytest.approx(result.condenser_duty + result.reboiler_duty, abs=1e-3)


def test_jit_cannot_ignore_volumes_without_reactions(problem):
    inlet, _, pkg = problem

    def run(volume):
        return rigorous_column(
            [ColumnFeed(inlet, 3)],
            6,
            p=3e5,
            model=pkg,
            specs=[ColumnSpec("reflux_ratio", 3.0), ColumnSpec("distillate_rate", 5.0)],
            reaction_volumes=volume,
            check=False,
        )

    compiled = jax.jit(run)
    assert compiled(0.0).report.converged
    assert not compiled(0.1).report.converged
