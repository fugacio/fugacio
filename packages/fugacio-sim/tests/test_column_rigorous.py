"""Rigorous Wang-Henke equilibrium-stage column: balances, equilibrium, gradients.

The column is cross-checked for internal consistency (material balance, stage
equilibrium, a monotone temperature profile) and against the shortcut method
(Fenske minimum stages below the rigorous stage count), and its product purity is
shown to be differentiable with respect to the reflux ratio.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import Stream
from fugacio.sim.column import fenske_min_stages, relative_volatility, solve_column
from fugacio.thermo import PR, component_arrays, ln_phi_mixture

COMPONENTS = ("propane", "n-butane")


def _feed() -> Stream:
    return Stream.from_fractions(COMPONENTS, jnp.array([0.5, 0.5]), 100.0, 320.0, 10e5)


@pytest.fixture(scope="module")
def base_case():
    feed = _feed()
    result = solve_column(feed, n_stages=12, feed_stage=6, reflux=2.0, distillate_rate=50.0)
    return feed, result


def test_column_closes_material_balance(base_case) -> None:
    feed, res = base_case
    closure = res.distillate.n + res.bottoms.n - feed.n
    assert float(jnp.max(jnp.abs(closure))) < 1e-4
    assert float(res.distillate.total) == pytest.approx(50.0, rel=1e-6)
    assert float(res.bottoms.total) == pytest.approx(50.0, rel=1e-6)


def test_column_temperature_profile_is_monotonic(base_case) -> None:
    _, res = base_case
    diffs = jnp.diff(res.t)
    assert float(jnp.min(diffs)) > 0.0  # temperature rises from condenser to reboiler


def test_column_enriches_light_component_overhead(base_case) -> None:
    _, res = base_case
    # propane (light key) is concentrated in the distillate, n-butane in the bottoms
    assert float(res.distillate.z[0]) > 0.9
    assert float(res.bottoms.z[1]) > 0.9
    # liquid light-fraction decreases monotonically down the column
    assert float(jnp.max(jnp.diff(res.x[:, 0]))) < 0.0


def test_column_satisfies_stage_equilibrium(base_case) -> None:
    _, res = base_case
    arr = component_arrays(list(COMPONENTS))
    tc, pc, omega = arr["tc"], arr["pc"], arr["omega"]
    p = 10e5

    def stage_k(t_j, x_j, y_j):
        ln_l, _ = ln_phi_mixture(PR, t_j, p, x_j, tc, pc, omega, phase="liquid")
        ln_v, _ = ln_phi_mixture(PR, t_j, p, y_j, tc, pc, omega, phase="vapor")
        return jnp.exp(ln_l - ln_v)

    k = jax.vmap(stage_k)(res.t, res.x, res.y)
    # bubble-point closure on every stage: sum_i K_i x_i = 1
    bubble = jnp.sum(k * res.x, axis=1) - 1.0
    assert float(jnp.max(jnp.abs(bubble))) < 1e-4
    # equilibrium y_i = K_i x_i (compositions normalised)
    y_pred = k * res.x
    y_pred = y_pred / jnp.sum(y_pred, axis=1, keepdims=True)
    assert float(jnp.max(jnp.abs(y_pred - res.y))) < 1e-4


def test_column_purity_increases_with_reflux() -> None:
    feed = _feed()
    low = solve_column(feed, n_stages=12, feed_stage=6, reflux=1.2, distillate_rate=50.0)
    high = solve_column(feed, n_stages=12, feed_stage=6, reflux=4.0, distillate_rate=50.0)
    assert float(high.distillate.z[0]) > float(low.distillate.z[0])


def test_column_stage_count_exceeds_fenske_minimum(base_case) -> None:
    feed, res = base_case
    alpha = relative_volatility(
        PR,
        float(res.t[5]),
        10e5,
        feed.z,
        component_arrays(list(COMPONENTS))["tc"],
        component_arrays(list(COMPONENTS))["pc"],
        component_arrays(list(COMPONENTS))["omega"],
        ref=1,
    )
    n_min = fenske_min_stages(res.distillate.n, res.bottoms.n, lk=0, hk=1, alpha=alpha)
    assert 0.0 < float(n_min) < 12.0  # a real column needs more than the Fenske minimum


def test_column_product_purity_is_differentiable_in_reflux() -> None:
    feed = _feed()

    def purity(reflux: float) -> jax.Array:
        res = solve_column(feed, n_stages=8, feed_stage=4, reflux=reflux, distillate_rate=50.0)
        return res.distillate.z[0]

    g = float(jax.grad(purity)(2.0))
    fd = float((purity(2.05) - purity(1.95)) / 0.1)
    assert g == pytest.approx(fd, rel=5e-2)
    assert g > 0.0  # more reflux -> purer distillate
