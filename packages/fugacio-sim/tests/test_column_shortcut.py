"""Shortcut (Fenske-Underwood-Gilliland) distillation against closed-form values."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim.column import (
    fenske_min_stages,
    gilliland_stages,
    kirkbride_feed_stage,
    relative_volatility,
    shortcut_column,
    underwood_min_reflux,
)
from fugacio.thermo import PR, component_arrays


def test_fenske_matches_hand_calculation() -> None:
    # alpha_LK/alpha_HK = 2; separation = (49/1)*(49/1) = 2401; N = ln 2401 / ln 2.
    d = jnp.array([50.0, 49.0, 1.0])
    b = jnp.array([0.0, 1.0, 49.0])
    alpha = jnp.array([4.0, 2.0, 1.0])
    n_min = float(fenske_min_stages(d, b, lk=1, hk=2, alpha=alpha))
    assert n_min == pytest.approx(jnp.log(2401.0) / jnp.log(2.0), rel=1e-6)
    assert n_min == pytest.approx(11.229, abs=1e-2)


def test_underwood_binary_matches_closed_form() -> None:
    # Binary, q = 1: theta = alpha / (1 + (alpha - 1) zA); R_min from the 2nd equation.
    alpha = jnp.array([2.5, 1.0])
    z = jnp.array([0.5, 0.5])
    x_d = jnp.array([0.98, 0.02])
    r_min, theta = underwood_min_reflux(z, x_d, alpha, q=1.0, lk=0, hk=1)
    assert float(theta) == pytest.approx(1.42857, abs=1e-4)
    assert float(r_min) == pytest.approx(1.240, abs=2e-3)


def test_underwood_residual_is_satisfied() -> None:
    alpha = jnp.array([3.0, 1.7, 1.0])
    z = jnp.array([0.4, 0.35, 0.25])
    x_d = jnp.array([0.79, 0.20, 0.01])
    _r_min, theta = underwood_min_reflux(z, x_d, alpha, q=1.0, lk=0, hk=1)
    residual = float(jnp.sum(alpha * z / (alpha - theta)))  # should equal 1 - q = 0
    assert residual == pytest.approx(0.0, abs=1e-6)
    assert float(alpha[1]) < float(theta) < float(alpha[0])  # root between the keys


def test_gilliland_endpoints_and_monotonicity() -> None:
    n_min, r_min = 11.23, 1.24
    # X = 0.5 at R = 3.48; Y = 0.75 (1 - 0.5^0.5668); N = (n_min + Y)/(1 - Y).
    y = 0.75 * (1.0 - 0.5**0.5668)
    expected = (n_min + y) / (1 - y)
    assert float(gilliland_stages(n_min, r_min, 3.48)) == pytest.approx(expected, rel=1e-4)
    # More reflux -> fewer stages, approaching N_min as R grows.
    n_low = float(gilliland_stages(n_min, r_min, 1.5))
    n_high = float(gilliland_stages(n_min, r_min, 10.0))
    assert n_low > n_high > n_min


def test_gilliland_stage_count_decreases_with_reflux() -> None:
    grad = float(jax.grad(lambda r: gilliland_stages(11.23, 1.24, r))(3.0))
    assert grad < 0.0  # dN/dR < 0


def test_kirkbride_feed_stage_within_column() -> None:
    z = jnp.array([0.4, 0.35, 0.25])
    x_d = jnp.array([0.79, 0.20, 0.01])
    x_b = jnp.array([0.01, 0.49, 0.50])
    nr = float(kirkbride_feed_stage(20.0, z, x_d, x_b, d_total=55.0, b_total=45.0, lk=0, hk=1))
    assert 0.0 < nr < 20.0


def test_relative_volatility_orders_by_volatility() -> None:
    components = ("methane", "propane", "n-pentane")
    arr = component_arrays(list(components))
    z = jnp.array([0.4, 0.3, 0.3])
    alpha = relative_volatility(PR, 300.0, 10e5, z, arr["tc"], arr["pc"], arr["omega"], ref=2)
    assert float(alpha[2]) == pytest.approx(1.0)  # reference component
    assert float(alpha[0]) > float(alpha[1]) > float(alpha[2])  # methane > propane > pentane


def test_shortcut_column_is_self_consistent_and_differentiable() -> None:
    z = jnp.array([0.5, 0.5])
    # Feed 100 mol total; sharp split of a binary with alpha = 2.5.
    d = jnp.array([49.0, 1.0])
    b = jnp.array([1.0, 49.0])
    alpha = jnp.array([2.5, 1.0])
    res = shortcut_column(z, d, b, alpha, q=1.0, lk=0, hk=1, reflux_factor=1.3)
    assert float(res.r) == pytest.approx(1.3 * float(res.r_min), rel=1e-6)
    assert float(res.n_stages) > float(res.n_min) > 0.0

    # More reflux should reduce the stage count.
    def stages(factor: float) -> jax.Array:
        return shortcut_column(z, d, b, alpha, q=1.0, lk=0, hk=1, reflux_factor=factor).n_stages

    assert float(jax.grad(stages)(1.5)) < 0.0
