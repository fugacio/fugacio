"""Common-package reactors: independent integration, balances, AD, and rejection."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from fugacio.sim import ReactionSet, ReferenceRate, Stream, reaction_reactor
from fugacio.sim.properties import resolve_package
from fugacio.thermo import Reaction

NAMES = ("n-butane", "isobutane")


def system(phase="vapor"):
    return ReactionSet.from_reactions(
        Reaction(NAMES, jnp.array([-1.0, 1.0])),
        [
            ReferenceRate(
                jnp.array(1.0),
                jnp.array(0.0),
                jnp.array([1.0, 0.0]),
                jnp.array(0.1),
                jnp.array(0.0),
                jnp.array([0.0, 1.0]),
                jnp.array(400.0),
            )
        ],
        phase=phase,
        rate_basis="normalized_concentration",
        reference_concentration=100.0,
    )


def feed(t=400.0, p=2e5):
    return Stream.from_fractions(NAMES, [0.9, 0.1], 10.0, t, p)


@pytest.mark.parametrize("kind", ["equilibrium", "cstr", "pfr"])
def test_real_gas_reactors_close_and_carry_implicit_derivatives(kind):
    rx, pkg, inlet = system(), resolve_package(NAMES), feed()

    def solve(v):
        return reaction_reactor(
            inlet, rx, model=pkg, kind=kind, volume=v, t_out=400.0, steps=16, check=False
        )

    solve = jax.jit(solve)
    result = solve(2.0)
    assert result.converged
    np.testing.assert_allclose(result.outlet.n, inlet.n + result.generation, atol=1e-7)
    assert result.element_error < 1e-8
    assert result.energy_error < 1e-8
    assert result.phase_error == 0
    assert result.outlet.phase_known

    def scalar(v):
        return solve(v).extent[0]

    ad = jax.jit(jax.grad(scalar))(2.0)
    fd = (scalar(2.001) - scalar(1.999)) / 0.002
    assert ad == pytest.approx(fd, rel=2e-5, abs=1e-8)
    if kind == "equilibrium":
        residual = rx.nu @ rx.log_activities(
            pkg, result.outlet.t, result.outlet.p, result.outlet.z
        ) - rx.ln_equilibrium_constants(result.outlet.t)
        assert abs(residual[0]) < 1e-8


def test_cstr_and_pfr_match_independent_scalar_solutions():
    rx, pkg, inlet = system(), resolve_package(NAMES), feed()
    rate = jax.jit(lambda xi: rx.rates(pkg, 400.0, 2e5, (inlet.n + xi * rx.nu[0]) / 10.0)[0])
    volume = 4.0
    expected_cstr = brentq(lambda xi: xi - volume * float(rate(xi)), 0.0, 8.0)
    expected_pfr = solve_ivp(
        lambda v, xi: [float(rate(xi[0]))], [0.0, volume], [0.0], rtol=1e-10, atol=1e-12
    ).y[0, -1]
    cstr = reaction_reactor(inlet, rx, model=pkg, kind="cstr", volume=volume)
    pfr = reaction_reactor(inlet, rx, model=pkg, kind="pfr", volume=volume, steps=16)
    assert cstr.extent[0] == pytest.approx(expected_cstr, rel=1e-8)
    assert pfr.extent[0] == pytest.approx(expected_pfr, rel=1e-7)
    assert pfr.extent[0] > cstr.extent[0]


def test_feed_temperature_kinetics_and_package_parameters_are_explicit():
    rx, pkg = system(), resolve_package(NAMES)

    # Combined directional derivative exercises every parameter path in one compilation.
    def objective(q):
        reaction = replace(rx, rate_laws=(replace(rx.rate_laws[0], k_forward=1 + 0.2 * q),))
        model = replace(pkg, kij=q * jnp.array([[0.0, 0.01], [0.01, 0.0]]))
        inlet = Stream(jnp.array([9.0 + 0.1 * q, 1.0]), 400.0 + q, 2e5, NAMES)
        return reaction_reactor(
            inlet, reaction, kind="cstr", model=model, volume=3.0, t_out=400.0 + q, check=False
        ).extent[0]

    fn = jax.jit(objective)
    assert jax.jit(jax.grad(fn))(0.0) == pytest.approx((fn(0.001) - fn(-0.001)) / 0.002, rel=1e-5)


@pytest.mark.parametrize("kind", ["cstr", "pfr"])
def test_adiabatic_energy_closure_and_pressure_profile(kind):
    r = reaction_reactor(feed(), system(), kind=kind, volume=1.0, duty=0.0, dp=1e4, steps=16)
    assert r.converged
    assert r.duty == 0
    assert r.outlet.t > 400.0
    assert r.energy_error < 1e-6
    assert r.pressure_profile[-1] == 1.9e5


def test_liquid_density_controls_concentration_rate():
    rx, inlet = system("liquid"), feed(280.0, 8e5)
    r = reaction_reactor(inlet, rx, kind="cstr", volume=0.05)
    assert r.converged and r.phase_error == 0
    assert r.extent[0] > 1.0


def test_wrong_phase_and_failed_nonlinear_solves_poison_derivatives():
    rx, inlet = system("liquid"), feed()
    fn = jax.jit(lambda v: reaction_reactor(inlet, rx, kind="cstr", volume=v, check=False))
    r = fn(1.0)
    assert not r.converged and r.phase_error == 1
    assert not jnp.isfinite(jax.grad(lambda v: fn(v).extent[0])(1.0))
    failed = reaction_reactor(inlet, system(), kind="cstr", max_iter=0, check=False)
    assert not failed.converged


def test_underresolved_pfr_is_rejected_without_clipping():
    r = reaction_reactor(feed(), system(), kind="pfr", volume=1000.0, steps=1, check=False)
    assert not r.converged
    assert r.integration_error > 1 or bool(jnp.any(r.component_profile < 0))


def test_multiple_and_non_equimolar_reactions_conserve_atoms():
    names = ("n-butane", "isobutane", "ethane", "ethylene", "hydrogen")
    nu = jnp.array([[-1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 1.0]])
    laws = [
        ReferenceRate(
            jnp.array(0.1),
            jnp.array(0.0),
            jnp.maximum(-row, 0.0),
            jnp.array(0.0),
            jnp.array(0.0),
            jnp.maximum(row, 0.0),
            jnp.array(800.0),
        )
        for row in nu
    ]
    rx = ReactionSet.from_reactions(
        [Reaction(names, row) for row in nu], laws, rate_basis="activity"
    )
    inlet = Stream.from_fractions(names, [0.3, 0.1, 0.4, 0.1, 0.1], 10.0, 800.0, 1e5)
    r = reaction_reactor(inlet, rx, kind="cstr", volume=2.0)
    assert r.converged and r.outlet.total > inlet.total
    assert r.element_error < 1e-8 and r.energy_error < 1e-8
    assert jnp.all(r.extent > 0)


def test_legacy_entry_points_accept_reusable_sets_without_duplicate_laws():
    from fugacio.sim import cstr, pfr

    inlet, rx = feed(), system()
    assert cstr(inlet, rx, volume=1.0).converged
    assert pfr(inlet, rx, volume=1.0, steps=8).converged
    with pytest.raises(ValueError, match="already supplies"):
        cstr(inlet, rx, rx.rate_laws, 1.0)
    with pytest.raises(ValueError, match="volume"):
        cstr(inlet, rx)
