"""Phase-equilibrium correctness: saturation, flash, envelopes, stability, gradients."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import components as comp
from fugacio.thermo import equilibrium as eq
from fugacio.thermo.consistency import equifugacity_residual
from fugacio.thermo.constants import ATM
from fugacio.thermo.eos import PR
from fugacio.thermo.reference import saturation_pressures

PR_SATURATION = ["propane", "n-butane", "n-pentane", "n-hexane", "benzene", "toluene"]


@pytest.mark.parametrize("name", PR_SATURATION)
def test_saturation_pressure_at_nbp_is_one_atm(name: str) -> None:
    c = comp.get(name)
    assert c.tb is not None
    psat = float(eq.psat_eos(PR, c.tb, c.tc, c.pc, c.omega))
    assert psat == pytest.approx(ATM, rel=0.06)


def test_saturation_pressure_dpdt_matches_finite_difference() -> None:
    c = comp.get("propane")
    f = lambda t: eq.psat_eos(PR, t, c.tc, c.pc, c.omega)  # noqa: E731
    ad = float(jax.grad(f)(300.0))
    fd = float((f(300.1) - f(299.9)) / 0.2)
    assert ad == pytest.approx(fd, rel=1e-4)


def test_batched_saturation_pressures_terminate_for_extreme_column_trials() -> None:
    # This NRTL column line-search trial hung under nested vmap on Linux JAX:
    # every temperature completed individually, but the batch never returned.
    # Keep the full precision and batch shape that triggered the stalled loop.
    temperatures = jnp.array(
        [
            350.90210698251536,
            350.914248095387,
            350.9263528772387,
            350.940243580431,
            350.9634965179927,
            351.0094085986309,
            350.9411184835624,
            350.4645363076113,
            348.7283459475676,
            340.31203957044636,
            286.9646981146145,
            170.92111042008364,
            199.4483352613907,
            311.5104022551129,
            358.4089841627552,
            370.8165379299441,
            373.7457892893207,
            374.4191750206445,
            374.57303241168177,
            374.6081372532009,
        ]
    )
    arr = comp.component_arrays(["ethanol", "water"])

    def pressures(t: jax.Array, tc: jax.Array, pc: jax.Array, omega: jax.Array) -> jax.Array:
        return saturation_pressures(PR, t, tc, pc, omega)

    individual = jax.jit(pressures)
    # Subfreezing trial states still need to terminate, but the liquid-vapor
    # EOS isn't reliable there. Compare pressures in the liquid-vapor range.
    liquid_vapor = temperatures > 273.15
    expected = jnp.stack(
        [individual(t, arr["tc"], arr["pc"], arr["omega"]) for t in temperatures[liquid_vapor]]
    )
    batch = jax.jit(jax.vmap(pressures, in_axes=(0, None, None, None)))
    actual = batch(temperatures, arr["tc"], arr["pc"], arr["omega"])
    assert bool(jnp.all(jnp.isfinite(actual) & (actual > 0.0)))
    assert actual[liquid_vapor] == pytest.approx(expected, rel=1e-6)


def test_rachford_rice_residual_is_zero() -> None:
    z = jnp.array([0.4, 0.35, 0.25])
    k = jnp.array([2.5, 1.1, 0.4])
    beta = eq.rachford_rice(z, k)
    residual = float(jnp.sum(z * (k - 1.0) / (1.0 + beta * (k - 1.0))))
    assert residual == pytest.approx(0.0, abs=1e-10)
    assert 0.0 < float(beta) < 1.0


def test_rachford_rice_single_phase_clamps() -> None:
    z = jnp.array([0.5, 0.5])
    assert float(eq.rachford_rice(z, jnp.array([0.5, 0.4]))) == 0.0  # all K < 1 -> liquid
    assert float(eq.rachford_rice(z, jnp.array([2.0, 3.0]))) == 1.0  # all K > 1 -> vapor


def test_flash_material_balance_and_equifugacity() -> None:
    arr = comp.component_arrays(["methane", "propane", "n-pentane"])
    z = jnp.array([0.5, 0.3, 0.2])
    t, p = 320.0, 20e5
    r = eq.flash_pt(PR, t, p, z, arr["tc"], arr["pc"], arr["omega"])
    assert float(jnp.sum(r.x)) == pytest.approx(1.0, abs=1e-9)
    assert float(jnp.sum(r.y)) == pytest.approx(1.0, abs=1e-9)
    mass_balance = (1.0 - r.beta) * r.x + r.beta * r.y - z
    assert float(jnp.max(jnp.abs(mass_balance))) < 1e-10
    resid = equifugacity_residual(PR, t, p, r.x, r.y, arr["tc"], arr["pc"], arr["omega"])
    assert float(resid) < 1e-9


def test_flash_gradient_through_equilibrium() -> None:
    arr = comp.component_arrays(["methane", "propane", "n-pentane"])
    z = jnp.array([0.5, 0.3, 0.2])
    beta_of_p = lambda p: eq.flash_pt(PR, 320.0, p, z, arr["tc"], arr["pc"], arr["omega"]).beta  # noqa: E731
    ad = float(jax.grad(beta_of_p)(20e5))
    fd = float((beta_of_p(20e5 + 1e3) - beta_of_p(20e5 - 1e3)) / 2e3)
    assert ad == pytest.approx(fd, rel=1e-4)


def test_bubble_dew_round_trip() -> None:
    arr = comp.component_arrays(["benzene", "toluene"])
    x = jnp.array([0.4, 0.6])
    t = 370.0
    p_bub, y = eq.bubble_pressure_eos(PR, t, x, arr["tc"], arr["pc"], arr["omega"])
    _, x_back = eq.dew_pressure_eos(PR, t, y, arr["tc"], arr["pc"], arr["omega"])
    assert float(jnp.max(jnp.abs(x_back - x))) < 1e-6
    assert float(p_bub) > 0.0


def test_stability_detects_two_phase_and_single_phase() -> None:
    arr = comp.component_arrays(["methane", "propane", "n-pentane"])
    z = jnp.array([0.5, 0.3, 0.2])
    unstable = eq.stability_analysis(PR, 320.0, 20e5, z, arr["tc"], arr["pc"], arr["omega"])
    assert not bool(unstable.stable)
    # A nearly pure liquid at low pressure should be stable.
    pure = jnp.array([1e-9, 1e-9, 1.0 - 2e-9])
    stable = eq.stability_analysis(PR, 300.0, 5e5, pure, arr["tc"], arr["pc"], arr["omega"])
    assert bool(stable.stable)
