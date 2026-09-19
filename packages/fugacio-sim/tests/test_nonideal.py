"""Non-ideal sim layer: package bridge, gamma-phi flash, decanter, VLLE, diagrams."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import (
    Stream,
    azeotrope_pressure,
    decanter,
    flash_drum,
    package_for,
    pxy_diagram,
    residue_curve,
    residue_curve_map,
    three_phase_flash,
    txy_diagram,
)
from fugacio.thermo import component_arrays, get
from fugacio.thermo.activity.models import nrtl
from fugacio.thermo.ideal import ideal_gas_coeffs
from fugacio.thermo.package import CubicPackage, GammaPhiPackage, gamma_phi_package

EW = ("ethanol", "water")
ALKANES = ("n-pentane", "n-hexane", "n-heptane")  # light -> heavy


def _activity_package(names: list[str], activity) -> GammaPhiPackage:
    arr = component_arrays(names)
    cp = ideal_gas_coeffs([get(n) for n in names])
    return gamma_phi_package(activity, arr["tc"], arr["pc"], arr["omega"], cp)


def _ternary_vlle_model() -> GammaPhiPackage:
    b = jnp.array([[0.0, 1500.0, 350.0], [1500.0, 0.0, 500.0], [200.0, 130.0, 0.0]])
    alpha = jnp.array([[0.0, 0.2, 0.3], [0.2, 0.0, 0.3], [0.3, 0.3, 0.0]])
    activity = nrtl(a=jnp.zeros((3, 3)), b=b, alpha=alpha)
    return _activity_package(["water", "benzene", "ethanol"], activity)


def test_package_bridge_constructs_expected_types() -> None:
    assert isinstance(package_for(EW, "nrtl"), GammaPhiPackage)
    assert isinstance(package_for(EW, "uniquac"), GammaPhiPackage)
    assert isinstance(package_for(EW, "dortmund"), GammaPhiPackage)
    assert isinstance(package_for(["propane", "n-butane"]), CubicPackage)


def test_gamma_phi_flash_drum_two_phase_balance() -> None:
    model = package_for(EW, "nrtl")
    z = jnp.array([0.5, 0.5])
    # A drum can't raise pressure, so the feed arrives above the drum pressure.
    feed = Stream.from_fractions(EW, z, 100.0, 363.15, 3e5)
    p_bub, _ = model.bubble_pressure(363.15, z)
    p_dew, _ = model.dew_pressure(363.15, z)
    p = 0.5 * (float(p_bub) + float(p_dew))
    vapor, liquid = flash_drum(feed, 363.15, p, model=model)
    beta = float(vapor.total / feed.total)
    assert 0.0 < beta < 1.0
    assert float(jnp.max(jnp.abs((vapor.n + liquid.n) - feed.n))) < 1e-6
    # Vapor is ethanol-enriched relative to the liquid.
    assert float(vapor.z[0]) > float(liquid.z[0])


def test_flash_drum_default_package_is_peng_robinson() -> None:
    comps = ("methane", "propane", "n-pentane")
    z = jnp.array([0.5, 0.3, 0.2])
    feed = Stream.from_fractions(comps, z, 100.0, 320.0, 20e5)
    vapor, liquid = flash_drum(feed, 320.0, 20e5, model=package_for(comps, "pr"))
    assert float(jnp.max(jnp.abs((vapor.n + liquid.n) - feed.n))) < 1e-6
    assert 0.0 < float(vapor.total / feed.total) < 1.0
    default_vapor, _ = flash_drum(feed, 320.0, 20e5)
    assert float(vapor.total) == pytest.approx(float(default_vapor.total), rel=1e-12)


def test_decanter_splits_immiscible_binary() -> None:
    gp = _strong_water_benzene()
    z = jnp.array([0.5, 0.5])
    feed = Stream.from_fractions(("water", "benzene"), z, 100.0, 330.0, 101325.0)
    liq_i, liq_ii = decanter(feed, gp, t=330.0)
    # Conjugate liquids are nearly pure and mirror-symmetric for the symmetric model.
    assert float(liq_i.z[0]) > 0.9
    assert float(liq_ii.z[1]) > 0.9
    assert float(jnp.max(jnp.abs((liq_i.n + liq_ii.n) - feed.n))) < 1e-9


def _strong_water_benzene() -> GammaPhiPackage:
    activity = nrtl(
        a=jnp.zeros((2, 2)),
        b=jnp.array([[0.0, 1500.0], [1500.0, 0.0]]),
        alpha=jnp.array([[0.0, 0.2], [0.2, 0.0]]),
    )
    return _activity_package(["water", "benzene"], activity)


def test_three_phase_flash_balance_and_positive_flows() -> None:
    gp = _ternary_vlle_model()
    z = jnp.array([0.47, 0.47, 0.06])
    feed = Stream.from_fractions(("water", "benzene", "ethanol"), z, 100.0, 340.0, 101325.0)
    vapor, liq_i, liq_ii = three_phase_flash(feed, 340.0, 101325.0, gp)
    for stream in (vapor, liq_i, liq_ii):
        assert float(stream.total) > 0.0
    total = vapor.n + liq_i.n + liq_ii.n
    assert float(jnp.max(jnp.abs(total - feed.n))) < 1e-6
    # The two liquids are a water-rich and a benzene-rich phase.
    assert float(liq_i.z[0]) > 0.8 or float(liq_ii.z[0]) > 0.8


def test_pxy_diagram_shapes_and_vapor_enrichment() -> None:
    model = package_for(EW, "nrtl")
    pxy = pxy_diagram(model, 351.0, n=11)
    assert pxy.x1.shape == (11,)
    assert pxy.y1.shape == (11,)
    assert pxy.p.shape == (11,)
    # On the ethanol-dilute side the vapour is ethanol-enriched.
    assert float(pxy.y1[0]) > float(pxy.x1[0])


def test_txy_diagram_brackets_pure_boiling_points() -> None:
    model = package_for(EW, "nrtl")
    txy = txy_diagram(model, 101325.0, n=9, t_min=300.0, t_max=420.0)
    assert txy.t.shape == (9,)
    # Bubble temperatures lie between the two pure normal boiling points (~351-373 K).
    assert float(txy.t.min()) > 345.0
    assert float(txy.t.max()) < 380.0


def test_azeotrope_pressure_found_for_ethanol_water() -> None:
    model = package_for(EW, "nrtl")
    az = azeotrope_pressure(model, 351.0)
    assert bool(az.exists)
    assert 0.0 < float(az.x1) < 1.0
    # y1 == x1 at the azeotrope (verify via the bubble relation).
    _, y = model.bubble_pressure(351.0, jnp.array([az.x1, 1.0 - az.x1]))
    assert float(y[0]) == pytest.approx(float(az.x1), abs=1e-4)


def test_database_kij_shifts_eos_bubble_pressure() -> None:
    comps = ["carbon dioxide", "benzene"]
    x = jnp.array([0.3, 0.7])
    p_ideal, _ = package_for(comps).bubble_pressure(313.15, x)
    p_kij, _ = package_for(comps, use_database_kij=True).bubble_pressure(313.15, x)
    # The curated positive k_ij weakens cross-attraction, raising the bubble pressure.
    assert float(p_kij) > 1.1 * float(p_ideal)


def test_residue_curve_marches_to_heavy_and_light_nodes() -> None:
    model = package_for(ALKANES)
    x0 = jnp.array([1 / 3, 1 / 3, 1 / 3])
    fwd = residue_curve(model, x0, 101325.0, steps=80, direction=1.0, t_min=250.0, t_max=460.0)
    bwd = residue_curve(model, x0, 101325.0, steps=80, direction=-1.0, t_min=250.0, t_max=460.0)
    # Forward enriches in the heaviest (n-heptane); backward in the lightest (n-pentane).
    assert float(fwd.x[-1, 2]) > 0.95
    assert float(bwd.x[-1, 0]) > 0.95
    # Boiling temperature rises monotonically toward the heavy node.
    assert bool(jnp.all(jnp.diff(fwd.t) >= -1e-6))
    assert float(fwd.t[-1]) > float(fwd.t[0])


def test_residue_curve_stays_on_simplex() -> None:
    model = package_for(ALKANES)
    curve = residue_curve(
        model, jnp.array([0.5, 0.3, 0.2]), 101325.0, steps=40, t_min=250.0, t_max=460.0
    )
    assert curve.x.shape == (41, 3)
    assert float(jnp.min(curve.x)) >= 0.0
    assert float(jnp.max(jnp.abs(jnp.sum(curve.x, axis=1) - 1.0))) < 1e-9


def test_residue_curve_map_returns_full_curves() -> None:
    model = package_for(ALKANES)
    starts = jnp.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2]])
    curves = residue_curve_map(model, starts, 101325.0, steps=30, t_min=250.0, t_max=460.0)
    assert len(curves) == 2
    for curve in curves:
        assert curve.x.shape == (61, 3)  # 2 * steps + 1
        assert float(jnp.max(jnp.abs(jnp.sum(curve.x, axis=1) - 1.0))) < 1e-9


def test_decanter_split_is_differentiable_in_parameters() -> None:
    z = jnp.array([0.5, 0.5])
    feed = Stream.from_fractions(("water", "benzene"), z, 100.0, 330.0, 101325.0)

    def water_purity(b12: float) -> jax.Array:
        activity = nrtl(
            a=jnp.zeros((2, 2)),
            b=jnp.array([[0.0, b12], [b12, 0.0]]),
            alpha=jnp.array([[0.0, 0.2], [0.2, 0.0]]),
        )
        liq_i, _ = decanter(feed, _activity_package(["water", "benzene"], activity), t=330.0)
        return liq_i.z[0]

    grad = float(jax.grad(water_purity)(1500.0))
    assert jnp.isfinite(grad)
