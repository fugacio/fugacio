"""Activity-coefficient models: limits, known values, and Gibbs-Duhem consistency."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo import activity as act
from fugacio.thermo.consistency import partial_molar_symmetry_residual

# A representative ternary set of parameters for the local-composition models.
TAU = jnp.array([[0.0, 0.3, -0.1], [0.5, 0.0, 0.2], [0.7, -0.2, 0.0]])
ALPHA = jnp.array([[0.0, 0.3, 0.3], [0.3, 0.0, 0.3], [0.3, 0.3, 0.0]])
LAMBDA = jnp.array([[1.0, 0.5, 0.7], [0.6, 1.0, 0.8], [0.4, 0.9, 1.0]])
R_UQ = jnp.array([1.4, 2.1, 0.92])
Q_UQ = jnp.array([1.4, 1.97, 1.4])
TAU_UQ = jnp.array([[1.0, 0.7, 1.2], [0.8, 1.0, 0.9], [1.1, 0.85, 1.0]])

COMPOSITIONS = [
    jnp.array([0.2, 0.3, 0.5]),
    jnp.array([0.5, 0.25, 0.25]),
    jnp.array([0.1, 0.1, 0.8]),
]


def _ln_gamma_fns():
    return {
        "nrtl": lambda x: act.nrtl_ln_gamma(x, TAU, ALPHA),
        "wilson": lambda x: act.wilson_ln_gamma(x, LAMBDA),
        "uniquac": lambda x: act.uniquac_ln_gamma(x, R_UQ, Q_UQ, TAU_UQ),
    }


@pytest.mark.parametrize("model", list(_ln_gamma_fns()))
@pytest.mark.parametrize("x", COMPOSITIONS, ids=lambda v: "-".join(f"{float(c):.2f}" for c in v))
def test_gibbs_duhem_consistency(model: str, x: jnp.ndarray) -> None:
    fn = _ln_gamma_fns()[model]
    assert float(partial_molar_symmetry_residual(fn, x)) < 1e-9


@pytest.mark.parametrize("model", list(_ln_gamma_fns()))
def test_pure_component_limit(model: str) -> None:
    fn = _ln_gamma_fns()[model]
    x = jnp.array([1.0 - 2e-9, 1e-9, 1e-9])
    assert float(fn(x)[0]) == pytest.approx(0.0, abs=1e-6)


def test_nrtl_excess_gibbs_matches_partial_molar() -> None:
    x = jnp.array([0.25, 0.35, 0.40])

    def phi(n: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(n) * act.nrtl_excess_gibbs(n / jnp.sum(n), TAU, ALPHA)

    pm = jax.grad(phi)(x)
    ln_gamma = act.nrtl_ln_gamma(x, TAU, ALPHA)
    assert float(jnp.max(jnp.abs(pm - ln_gamma))) < 1e-9


def test_wilson_excess_gibbs_matches_partial_molar() -> None:
    x = jnp.array([0.25, 0.35, 0.40])

    def phi(n: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(n) * act.wilson_excess_gibbs(n / jnp.sum(n), LAMBDA)

    pm = jax.grad(phi)(x)
    ln_gamma = act.wilson_ln_gamma(x, LAMBDA)
    assert float(jnp.max(jnp.abs(pm - ln_gamma))) < 1e-9


def test_van_laar_infinite_dilution() -> None:
    a12, a21 = 0.7, 1.1
    ln_g1, _ = act.van_laar_ln_gamma(0.0, a12, a21)
    _, ln_g2 = act.van_laar_ln_gamma(1.0, a12, a21)
    assert float(ln_g1) == pytest.approx(a12, rel=1e-9)
    assert float(ln_g2) == pytest.approx(a21, rel=1e-9)


def test_van_laar_gibbs_duhem_binary() -> None:
    a12, a21 = 0.7, 1.1
    d_ln_g1 = jax.grad(lambda x: act.van_laar_ln_gamma(x, a12, a21)[0])
    d_ln_g2 = jax.grad(lambda x: act.van_laar_ln_gamma(x, a12, a21)[1])
    x1 = 0.35
    residual = x1 * d_ln_g1(x1) + (1.0 - x1) * d_ln_g2(x1)
    assert float(residual) == pytest.approx(0.0, abs=1e-7)


def test_nrtl_tau_builder() -> None:
    a = jnp.array([[0.0, 0.5], [0.3, 0.0]])
    b = jnp.array([[0.0, 100.0], [-50.0, 0.0]])
    tau = act.nrtl_tau(300.0, a, b)
    assert float(tau[0, 1]) == pytest.approx(0.5 + 100.0 / 300.0, rel=1e-12)
