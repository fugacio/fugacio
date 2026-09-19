"""Silent-failure corpus: states that once returned a confident wrong answer.

Every state here is either accepted and physically valid, or explicitly failed:
a non-converged report, a NaN value from the value-only call, and a nonfinite
derivative. None may report convergence with a wrong value. Each case records
the behavior that the corpus guards against.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import (
    component_arrays,
    cubic_package,
    flash_lle_with_info,
    flash_vlle_with_info,
    get,
    ideal_gas_coeffs,
    nrtl_from_database,
    saft_package,
    saft_parameters_for,
    uniquac_from_database,
)
from fugacio.thermo.acceptance import flash_pt_checked
from fugacio.thermo.activity.models import FloryHuggins
from fugacio.thermo.diagnostics import SolveStatus
from fugacio.thermo.eos import PR
from fugacio.thermo.equilibrium import psat_eos, psat_eos_with_info
from fugacio.thermo.groupcontrib.unifac import unifac_activity
from fugacio.thermo.saft import ln_fugacity_coefficients, molar_density

ATM = 101325.0


def _arrays(names):
    a = component_arrays(list(names))
    return a["tc"], a["pc"], a["omega"], ideal_gas_coeffs([get(c) for c in names])


def _cubic(names):
    return cubic_package(*_arrays(names))


# --------------------------------------------------------------------------- #
# Phase stability: water/n-hexane is two liquids, not one
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("z", "bound"), [([0.5, 0.5], -2.0), ([0.1, 0.9], -1.5)])
def test_water_hexane_single_liquid_is_reported_unstable(z, bound) -> None:
    # Guards against: a stability test that searched only near the feed and
    # reported this strongly immiscible liquid as stable.
    # A negative distance proves instability by itself; every trial reaching a
    # stationary point (``converged``) is needed only to certify stability.
    result = _cubic(("water", "n-hexane")).stability(300.0, ATM, jnp.array(z))
    assert not bool(result.stable)
    assert float(result.tpd) <= bound


def test_water_hexane_checked_flash_rejects_the_vapor_liquid_answer() -> None:
    pkg = _cubic(("water", "n-hexane"))
    checked = flash_pt_checked(pkg, 300.0, ATM, jnp.array([0.5, 0.5]))
    assert not bool(checked.report.accepted)
    assert float(checked.report.minimum_tpd) < -1.0
    assert any("stab" in reason for reason in checked.report.failures())


def test_water_hexane_flash_never_reports_a_stalled_iteration_as_converged() -> None:
    # Guards against: a successive-substitution flash that hit its iteration
    # cap and still returned its last iterate as the answer.
    pkg = _cubic(("water", "n-hexane"))
    z = jnp.array([0.5, 0.5])
    solved = pkg.flash_pt_with_info(340.0, ATM, z)
    value = pkg.flash_pt(340.0, ATM, z)
    if bool(solved.report.converged):
        res = solved.value
        closure = res.beta * res.y + (1.0 - res.beta) * res.x - z
        assert float(jnp.max(jnp.abs(closure))) < 1e-10
        assert bool(jnp.isfinite(value.beta))
    else:
        assert int(solved.report.status) != int(SolveStatus.CONVERGED)
        assert bool(jnp.isnan(value.beta))


# --------------------------------------------------------------------------- #
# Three-phase and liquid-liquid flashes: 1-butanol/water
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("t", [300.0, 365.0, 366.5, 380.0])
def test_butanol_water_vlle_conserves_material_or_fails_explicitly(t) -> None:
    names = ("1-butanol", "water")
    tc, pc, omega, _ = _arrays(names)
    z = jnp.array([0.3, 0.7])
    solved = flash_vlle_with_info(nrtl_from_database(list(names)), t, ATM, z, tc, pc, omega)
    res = solved.value
    if bool(solved.report.converged):
        betas = jnp.array([res.beta_v, res.beta_l1, res.beta_l2])
        assert bool(jnp.all((betas >= 0.0) & (betas <= 1.0)))
        balance = res.beta_v * res.y + res.beta_l1 * res.x_i + res.beta_l2 * res.x_ii - z
        assert float(jnp.max(jnp.abs(balance))) <= 1e-10
    else:
        assert int(solved.report.status) in (
            int(SolveStatus.INFEASIBLE),
            int(SolveStatus.MAX_ITERATIONS),
            int(SolveStatus.NONFINITE),
        )


def test_butanol_water_liquid_split_matches_the_reference_phase_ratio() -> None:
    names = ("1-butanol", "water")
    solved = flash_lle_with_info(nrtl_from_database(list(names)), 300.0, jnp.array([0.3, 0.7]))
    assert bool(solved.report.converged)
    assert float(solved.value.psi) == pytest.approx(0.2542, abs=5e-4)


# --------------------------------------------------------------------------- #
# Saturation points: domain limits are failures, not extrapolations
# --------------------------------------------------------------------------- #


def test_air_bubble_temperature_on_the_default_bracket() -> None:
    # Guards against: a bubble-temperature bracket whose low end had no bubble
    # point, so the solve returned the bracket edge.
    air = _cubic(("nitrogen", "oxygen", "argon"))
    solved = air.bubble_temperature_with_info(ATM, jnp.array([0.78, 0.21, 0.01]))
    assert bool(solved.report.converged)
    assert float(solved.value.value) == pytest.approx(78.70, abs=0.1)


def test_bubble_temperature_above_the_cricondenbar_fails_with_nonfinite_derivative() -> None:
    pkg = _cubic(("methane", "n-butane"))
    x = jnp.array([0.6, 0.4])
    solved = pkg.bubble_temperature_with_info(200e5, x)
    assert not bool(solved.report.converged)
    assert bool(jnp.isnan(pkg.bubble_temperature(200e5, x).value))
    slope = jax.grad(lambda p: pkg.bubble_temperature(p, x).value)(200e5)
    assert not bool(jnp.isfinite(slope))
    # Below the cricondenbar, the same call converges.
    assert bool(pkg.bubble_temperature_with_info(50e5, x).report.converged)


def test_bubble_pressure_above_the_mixture_critical_region_is_trivial() -> None:
    # Guards against: a bubble-pressure iteration that collapsed onto y = x and
    # reported the trivial solution as a saturation point.
    pkg = _cubic(("methane", "n-butane"))
    solved = pkg.bubble_pressure_with_info(330.0, jnp.array([0.6, 0.4]))
    assert not bool(solved.report.converged)
    assert int(solved.report.status) == int(SolveStatus.TRIVIAL)
    assert bool(jnp.isnan(pkg.bubble_pressure(330.0, jnp.array([0.6, 0.4])).value))


def test_supercritical_vapor_pressure_is_out_of_domain() -> None:
    methane = get("methane")
    solved = psat_eos_with_info(PR, 300.0, methane.tc, methane.pc, methane.omega)
    assert int(solved.report.status) == int(SolveStatus.OUT_OF_DOMAIN)
    assert bool(jnp.isnan(psat_eos(PR, 300.0, methane.tc, methane.pc, methane.omega)))


def test_scalar_and_batched_vapor_pressure_agree() -> None:
    # Guards against: a batched solve that stopped every lane when the first
    # converged, so batched and scalar answers differed.
    water = get("water")
    ts = jnp.linspace(200.0, 640.0, 23)
    batched = jax.vmap(lambda t: psat_eos(PR, t, water.tc, water.pc, water.omega))(ts)
    scalar = jnp.array([float(psat_eos(PR, float(t), water.tc, water.pc, water.omega)) for t in ts])
    assert bool(jnp.all(jnp.isfinite(batched)))
    assert float(jnp.max(jnp.abs(batched / scalar - 1.0))) < 1e-8


# --------------------------------------------------------------------------- #
# Activity models at an absent component
# --------------------------------------------------------------------------- #


def _pure_limit_error(ln_gamma) -> float:
    exact = ln_gamma(jnp.array([0.5, 0.5, 0.0]))
    near = ln_gamma(jnp.array([0.5, 0.5 - 1e-12, 1e-12]))
    assert bool(jnp.all(jnp.isfinite(exact)))
    return float(jnp.max(jnp.abs(exact - near)))


def test_uniquac_is_finite_and_continuous_at_an_absent_component() -> None:
    model = uniquac_from_database(["ethanol", "water", "acetone"])
    assert _pure_limit_error(lambda x: model.ln_gamma(x, 330.0)) < 1e-8


def test_unifac_is_finite_and_continuous_at_an_absent_component() -> None:
    names = ["ethanol", "water", "acetone"]
    assert _pure_limit_error(lambda x: unifac_activity(names, x, 330.0)) < 1e-8


def test_flory_huggins_is_finite_and_continuous_at_an_absent_component() -> None:
    model = FloryHuggins(volume=jnp.array([1.0, 3.0, 12.0]))
    assert _pure_limit_error(lambda x: model.ln_gamma(x, 300.0)) < 1e-8


# --------------------------------------------------------------------------- #
# PC-SAFT: association and missing density branches
# --------------------------------------------------------------------------- #


def _saft(names):
    tc, pc, omega, cp = _arrays(names)
    return saft_package(saft_parameters_for(list(names)), tc, pc, omega, cp)


def test_associating_pcsaft_mixture_is_finite_and_differentiable() -> None:
    pkg = _saft(("methanol", "n-hexane"))
    x = jnp.array([0.4, 0.6])
    for phase in ("liquid", "vapor"):
        assert bool(jnp.isfinite(molar_density(pkg.params, 320.0, 1e5, x, phase=phase)))

    def ln_phi(t):
        return ln_fugacity_coefficients(pkg.params, t, 1e5, x, phase="liquid")

    assert bool(jnp.all(jnp.isfinite(ln_phi(320.0))))
    slope = jax.jacfwd(ln_phi)(320.0)
    step = 1e-3
    central = (ln_phi(320.0 + step) - ln_phi(320.0 - step)) / (2 * step)
    assert float(jnp.max(jnp.abs(slope - central))) < 1e-6


def test_associating_pcsaft_flash_is_accepted_or_names_its_reason() -> None:
    pkg = _saft(("methanol", "n-hexane"))
    checked = flash_pt_checked(pkg, 320.0, 1e5, jnp.array([0.4, 0.6]))
    assert bool(jnp.all(jnp.isfinite(checked.value.x)))
    if not bool(checked.report.accepted):
        assert checked.report.failures()


def test_pcsaft_compressed_liquid_flash_uses_the_only_density_root() -> None:
    # Guards against: a vapor density solve with no vapor root on this isotherm
    # diverging to NaN, which a checked flash rightly refused to call converged.
    pkg = _saft(("propane", "n-butane"))
    solved = pkg.flash_pt_with_info(320.0, 30e5, jnp.array([0.5, 0.5]))
    assert bool(solved.report.converged)
    assert float(solved.value.beta) == 0.0
