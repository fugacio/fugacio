import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import margules_gamma, margules_ln_gamma


def test_infinite_dilution_limits() -> None:
    a12, a21 = 0.5, 0.8
    ln_g1_at_x0, _ = margules_ln_gamma(0.0, a12, a21)
    _, ln_g2_at_x1 = margules_ln_gamma(1.0, a12, a21)
    assert float(ln_g1_at_x0) == pytest.approx(a12, abs=1e-12)
    assert float(ln_g2_at_x1) == pytest.approx(a21, abs=1e-12)


def test_known_values() -> None:
    gamma1, gamma2 = margules_gamma(0.3, 0.5, 0.8)
    assert float(gamma1) == pytest.approx(1.39543, rel=1e-4)
    assert float(gamma2) == pytest.approx(1.03479, rel=1e-4)


def test_ideal_mixture_has_unit_activity() -> None:
    gamma1, gamma2 = margules_gamma(jnp.linspace(0.0, 1.0, 5), 0.0, 0.0)
    assert jnp.allclose(gamma1, 1.0)
    assert jnp.allclose(gamma2, 1.0)


def test_gibbs_duhem_consistency_via_autodiff() -> None:
    # At constant T, P:  x1 * d(ln g1)/dx1 + x2 * d(ln g2)/dx1 == 0.
    a12, a21 = 0.5, 0.8
    d_ln_g1 = jax.grad(lambda x: margules_ln_gamma(x, a12, a21)[0])
    d_ln_g2 = jax.grad(lambda x: margules_ln_gamma(x, a12, a21)[1])
    x1 = 0.4
    residual = x1 * d_ln_g1(x1) + (1.0 - x1) * d_ln_g2(x1)
    assert float(residual) == pytest.approx(0.0, abs=1e-6)
