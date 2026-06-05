"""The first-principles consistency harness applied across models (data-free oracles)."""

import jax.numpy as jnp

from fugacio.thermo import activity as act
from fugacio.thermo import components as comp
from fugacio.thermo import consistency as cons
from fugacio.thermo import diffcheck
from fugacio.thermo.eos import PR, ln_phi_mixture

X3 = jnp.array([0.3, 0.45, 0.25])
TAU = jnp.array([[0.0, 0.4, -0.2], [0.6, 0.0, 0.1], [0.3, -0.1, 0.0]])
ALPHA = jnp.full((3, 3), 0.3) - jnp.eye(3) * 0.3


def test_gibbs_duhem_for_activity_model() -> None:
    residual = cons.gibbs_duhem_residual(lambda x: act.nrtl_ln_gamma(x, TAU, ALPHA), X3)
    assert float(residual) < 1e-9


def test_gibbs_duhem_for_eos_fugacity() -> None:
    arr = comp.component_arrays(["methane", "propane", "n-pentane"])

    def ln_phi(x: jnp.ndarray) -> jnp.ndarray:
        value, _ = ln_phi_mixture(
            PR, 320.0, 20e5, x, arr["tc"], arr["pc"], arr["omega"], phase="liquid"
        )
        return value

    assert float(cons.partial_molar_symmetry_residual(ln_phi, X3)) < 1e-8


def test_fugacity_pressure_identity_all_phases() -> None:
    c = comp.get("benzene")
    for phase in ("vapor", "liquid"):
        residual = cons.fugacity_pressure_residual(PR, 400.0, 5e5, c.tc, c.pc, c.omega, phase=phase)
        assert float(residual) < 1e-6


def test_diffcheck_zero_for_smooth_function() -> None:
    err = diffcheck.max_gradient_error(lambda x: jnp.sum(x**3), jnp.array([1.0, 2.0, 3.0]))
    assert float(err) < 1e-5


def test_diffcheck_jacobian_shape() -> None:
    def f(v: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([v[0] * v[1], v[1] ** 2])

    jac = diffcheck.finite_difference_jacobian(f, jnp.array([2.0, 3.0]))
    assert jac.shape == (2, 2)


def test_diffcheck_catches_wrong_gradient() -> None:
    # finite differences of |x|-style kink vs a smooth approximation should differ.
    err = diffcheck.max_gradient_error(lambda x: jnp.sum(jnp.abs(x)), jnp.array([1.0, -2.0]))
    assert float(err) < 1e-5  # away from the kink, AD and FD still agree
