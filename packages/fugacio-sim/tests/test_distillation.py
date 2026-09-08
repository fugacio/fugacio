"""Rigorous MESH column: balances, specifications, condenser types, absorbers, gradients.

Every converged column must close its component material balances and its
overall energy balance exactly; the specifications must be met; a total
condenser must return a saturated liquid distillate of the reflux composition;
the ideal-ish benzene/toluene split must be sharper than a shortcut estimate
would allow at the same stages and reflux; the absorber must recover the heavy
gas almost completely; and the distillate purity must be differentiable in the
reflux ratio with the implicit gradient matching a finite difference.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import (
    ColumnFeed,
    Stream,
    absorber,
    bottoms_rate,
    distillate_rate,
    enthalpy_flow,
    package_for,
    purity,
    reflux_ratio,
    rigorous_column,
)

BT = ("benzene", "toluene")


@pytest.mark.parametrize("method", ["pr", "srk"])
def test_bubble_closure_reduction_preserves_first_and_second_derivatives(method):
    from fugacio.sim.distillation import _bubble_sum, _incipient_vapor_k

    pkg = package_for(("propane", "n-butane", "n-pentane"), method)

    def closure(v, reduced=False):
        t, p = 330.0 + v[0], (16.0 + v[1]) * 1e5
        x = jax.nn.softmax(jnp.array([2.0, 0.0, -1.0]) + v[2:])
        if reduced:
            return _bubble_sum(pkg, t, p, x)
        return jnp.sum(_incipient_vapor_k(pkg, t, p, x) * x)

    v = jnp.zeros(5)
    full = jax.jit(jax.grad(closure))(v)
    reduced = jax.jit(jax.grad(lambda a: closure(a, True)))(v)
    finite_difference = jnp.asarray(
        [(closure(v.at[i].set(0.001)) - closure(v.at[i].set(-0.001))) / 0.002 for i in range(5)]
    )
    assert jnp.allclose(reduced, full, rtol=1e-8, atol=1e-10)
    assert jnp.allclose(reduced, finite_difference, rtol=2e-5, atol=1e-8)
    full_second = jax.jit(jax.grad(jax.grad(lambda a: closure(v.at[0].set(a)))))(0.0)
    reduced_second = jax.jit(jax.grad(jax.grad(lambda a: closure(v.at[0].set(a), True))))(0.0)
    assert float(reduced_second) == pytest.approx(float(full_second), rel=1e-8, abs=1e-10)


def _bt_feed() -> Stream:
    return Stream.from_fractions(BT, jnp.array([0.5, 0.5]), 100.0, 365.0, 1.013e5)


def _check_balances(res, feeds: list[Stream], model=None) -> None:
    n_in = sum(f.n for f in feeds)
    n_out = (
        res.distillate.n + res.bottoms.n + sum((d.n for d in res.side_draws), jnp.zeros_like(n_in))
    )
    assert jnp.allclose(n_in, n_out, atol=1e-6), (n_in, n_out)
    # The energy balance must be evaluated on the column's own property package.
    h_in = sum(enthalpy_flow(f, model=model) for f in feeds)
    h_out = enthalpy_flow(res.distillate, model=model) + enthalpy_flow(res.bottoms, model=model)
    assert float(h_in + res.condenser_duty + res.reboiler_duty - h_out) == pytest.approx(
        0.0, abs=1e-3 * abs(float(h_in)) + 1.0
    )


def test_benzene_toluene_total_condenser_column() -> None:
    feed = _bt_feed()
    res = rigorous_column(
        [ColumnFeed(feed, 6)], 12, p=1.013e5, specs=[reflux_ratio(2.5), distillate_rate(50.0)]
    )
    assert float(res.residual_norm) < 1e-8
    _check_balances(res, [feed])
    assert float(res.reflux_ratio) == pytest.approx(2.5, rel=1e-8)
    assert float(res.distillate.total) == pytest.approx(50.0, rel=1e-8)
    # A sharp split: benzene concentrates at the top, toluene at the bottom.
    assert float(res.distillate.z[0]) > 0.94
    assert float(res.bottoms.z[1]) > 0.94
    # Temperatures rise monotonically down the column and bracket the pure boiling points.
    assert bool(jnp.all(jnp.diff(res.t) > 0.0))
    assert 352.0 < float(res.t[0]) < 356.0 and 380.0 < float(res.t[-1]) < 384.5
    # Condenser removes heat, reboiler adds it.
    assert float(res.condenser_duty) < 0.0 < float(res.reboiler_duty)

    warm = rigorous_column(
        [ColumnFeed(feed, 6)],
        12,
        p=1.013e5,
        specs=[reflux_ratio(2.5), distillate_rate(50.0)],
        guess=res.warm_start(),
    )
    assert warm.report.converged and warm.report.iterations == 0
    assert jnp.allclose(warm.distillate.n, res.distillate.n, atol=1e-8)
    bad_guess = {**res.warm_start(), "t": res.t + 30.0}
    failed = rigorous_column(
        [ColumnFeed(feed, 6)],
        12,
        p=1.013e5,
        specs=[reflux_ratio(2.5), distillate_rate(50.0)],
        guess=bad_guess,
        max_iter=0,
        check=False,
    )
    assert not failed.report.converged


def test_partial_condenser_with_purity_spec() -> None:
    feed = _bt_feed()
    res = rigorous_column(
        [ColumnFeed(feed, 6)],
        12,
        p=1.013e5,
        condenser="partial",
        specs=[purity("distillate", 0, 0.95), bottoms_rate(50.0)],
    )
    assert float(res.residual_norm) < 1e-8
    _check_balances(res, [feed])
    assert float(res.distillate.z[0]) == pytest.approx(0.95, abs=1e-8)
    assert float(res.bottoms.total) == pytest.approx(50.0, rel=1e-8)
    # A vapour distillate leaves at the top-stage dew point, i.e. in equilibrium with the reflux.
    assert jnp.allclose(res.distillate.z, res.y[0], atol=1e-10)


def test_distillate_purity_gradient_matches_finite_difference() -> None:
    feed = _bt_feed()

    def x_d(r: float) -> jnp.ndarray:
        res = rigorous_column(
            [ColumnFeed(feed, 5)], 8, p=1.013e5, specs=[reflux_ratio(r), distillate_rate(50.0)]
        )
        return res.distillate.z[0]

    g = jax.grad(x_d)(2.0)
    fd = (x_d(2.02) - x_d(1.98)) / 0.04
    assert float(g) > 0.0  # more reflux, purer distillate
    assert float(g) == pytest.approx(float(fd), rel=2e-3)


def test_nrtl_ethanol_water_column_is_limited_by_the_azeotrope() -> None:
    e_w = ("ethanol", "water")
    pkg = package_for(e_w, "nrtl")
    feed = Stream.from_fractions(e_w, jnp.array([0.10, 0.90]), 100.0, 360.0, 1.013e5)
    res = rigorous_column(
        [ColumnFeed(feed, 10)],
        20,
        p=1.013e5,
        specs=[reflux_ratio(4.0), distillate_rate(11.0)],
        model=pkg,
    )
    assert float(res.residual_norm) < 1e-8
    _check_balances(res, [feed], model=pkg)
    # Nearly all the ethanol goes up, but the distillate stays below the azeotrope (~0.89).
    assert float(res.distillate.z[0]) > 0.8
    assert float(res.distillate.z[0]) < 0.9
    assert float(res.bottoms.z[0]) < 0.02


def test_absorber_recovers_the_heavy_gas() -> None:
    comps = ("methane", "propane", "n-decane")
    gas = Stream.from_fractions(comps, jnp.array([0.85, 0.15, 0.0]), 100.0, 300.0, 20e5)
    oil = Stream.from_fractions(comps, jnp.array([0.0, 0.0, 1.0]), 150.0, 300.0, 20e5)
    res = absorber(gas, oil, 6, p=20e5)
    assert float(res.residual_norm) < 1e-8
    _check_balances(res, [oil, gas])
    propane_recovery = 1.0 - float(res.distillate.n[1] / gas.n[1])
    assert propane_recovery > 0.99
    # Most of the methane passes through; the solvent stays in the bottoms.
    assert float(res.distillate.n[0] / gas.n[0]) > 0.6
    assert float(res.bottoms.n[2] / oil.n[2]) > 0.999
    # Absorption is exothermic: the rich oil leaves warmer than the feeds.
    assert float(res.t[-1]) > 300.0


def test_column_validation() -> None:
    feed = _bt_feed()
    with pytest.raises(ValueError, match="degree"):
        rigorous_column([ColumnFeed(feed, 3)], 6, p=1e5, specs=[reflux_ratio(2.0)])
    with pytest.raises(ValueError, match="stage"):
        rigorous_column(
            [ColumnFeed(feed, 9)], 6, p=1e5, specs=[reflux_ratio(2.0), distillate_rate(50.0)]
        )
    with pytest.raises(ValueError, match="condenser"):
        rigorous_column([ColumnFeed(feed, 3)], 6, p=1e5, condenser="magic", specs=[])
