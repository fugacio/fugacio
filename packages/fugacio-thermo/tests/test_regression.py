"""Parameter regression: optimizers converge and recover known parameters."""

import jax.numpy as jnp
import pytest

from fugacio.thermo.activity.models import nrtl
from fugacio.thermo.groupcontrib.unifac import unifac_activity
from fugacio.thermo.regression import (
    activity_residuals,
    gradient_descent,
    levenberg_marquardt,
    predict_nrtl_from_unifac,
    predict_uniquac_from_unifac,
    unifac_ln_gamma_grid,
)

ALPHA = jnp.array([[0.0, 0.3], [0.3, 0.0]])


def test_levenberg_marquardt_solves_linear_least_squares() -> None:
    x = jnp.linspace(0.0, 1.0, 11)
    y = 3.0 * x + 1.0

    def resid(params: jnp.ndarray) -> jnp.ndarray:
        a, b = params
        return a * x + b - y

    theta, cost = levenberg_marquardt(resid, jnp.array([0.0, 0.0]), max_iter=50)
    assert float(theta[0]) == pytest.approx(3.0, abs=1e-3)
    assert float(theta[1]) == pytest.approx(1.0, abs=1e-3)
    assert float(cost) < 1e-8


def test_gradient_descent_minimises_quadratic() -> None:
    def loss(p: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum((p - jnp.array([2.0, -1.0])) ** 2)

    theta, _ = gradient_descent(loss, jnp.array([0.0, 0.0]), learning_rate=0.2, max_iter=400)
    assert float(jnp.max(jnp.abs(theta - jnp.array([2.0, -1.0])))) < 1e-2


def test_recover_nrtl_parameter_from_activity_data() -> None:
    t = 333.15
    x_data = jnp.array([[0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2]])
    true_b = jnp.array([[0.0, 650.0], [300.0, 0.0]])
    truth = nrtl(a=jnp.zeros((2, 2)), b=true_b, alpha=ALPHA)
    g_data = jnp.stack([truth.ln_gamma(x, t) for x in x_data])
    t_vec = jnp.full((x_data.shape[0],), t)

    def build(theta: jnp.ndarray) -> object:
        b = jnp.array([[0.0, theta[0]], [theta[1], 0.0]])
        return nrtl(a=jnp.zeros((2, 2)), b=b, alpha=ALPHA)

    resid = activity_residuals(build, t_vec, x_data, g_data)
    theta, cost = levenberg_marquardt(resid, jnp.array([400.0, 500.0]), max_iter=120)
    assert float(theta[0]) == pytest.approx(650.0, abs=5.0)
    assert float(theta[1]) == pytest.approx(300.0, abs=5.0)
    assert float(cost) < 1e-6


def test_unifac_ln_gamma_grid_shape_and_positivity() -> None:
    t_grid, x_grid, g_grid = unifac_ln_gamma_grid(["ethanol", "water"], [333.15, 353.15], points=9)
    assert t_grid.shape == (18,)
    assert x_grid.shape == (18, 2)
    assert g_grid.shape == (18, 2)
    assert jnp.allclose(jnp.sum(x_grid, axis=1), 1.0)
    # Ethanol/water shows positive deviations from Raoult's law (ln gamma > 0).
    assert float(jnp.min(g_grid)) > 0.0


def test_unifac_grid_rejects_non_binary() -> None:
    with pytest.raises(ValueError, match="2 components"):
        unifac_ln_gamma_grid(["ethanol", "water", "benzene"], 333.15)


def test_predict_nrtl_from_unifac_reproduces_unifac() -> None:
    comps = ["ethanol", "water"]
    model, cost = predict_nrtl_from_unifac(comps, [333.15, 353.15], points=15, max_iter=150)
    assert float(cost) < 0.1
    # Fitted NRTL reproduces UNIFAC activity coefficients at interior compositions.
    for x1 in (0.3, 0.5, 0.7):
        x = jnp.array([x1, 1.0 - x1])
        g_fit = jnp.exp(model.ln_gamma(x, 343.15))
        g_uni = jnp.exp(unifac_activity(comps, x, 343.15))
        assert jnp.allclose(g_fit, g_uni, atol=0.12)


def test_predict_uniquac_from_unifac_defaults_rq() -> None:
    comps = ["ethanol", "water"]
    model, cost = predict_uniquac_from_unifac(comps, 343.15, points=15, max_iter=150)
    assert float(cost) < 0.1
    x = jnp.array([0.4, 0.6])
    g_fit = jnp.exp(model.ln_gamma(x, 343.15))
    g_uni = jnp.exp(unifac_activity(comps, x, 343.15))
    assert jnp.allclose(g_fit, g_uni, atol=0.12)
