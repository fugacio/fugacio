"""First-principles consistency of PC-SAFT under autodiff.

These are data-free oracles: exact thermodynamic identities that must hold
between *independently computed* quantities. Because the whole of
:mod:`fugacio.thermo.saft` is one differentiable reduced residual Helmholtz
energy, the identities can be checked through :func:`jax.grad` itself, exactly as
for the reference Helmholtz package (``test_helmholtz_consistency.py``). Agreement
to near machine precision validates the energy derivatives, the density and
saturation solvers' implicit rules, the Wertheim site-fraction custom JVP, and
the fugacity/residual-property formulas simultaneously.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp

from fugacio.thermo.constants import R
from fugacio.thermo.saft import (
    alpha_residual,
    compressibility_factor,
    ln_fugacity_coefficients,
    molar_density,
    pressure,
    psat_saft,
    residual_properties,
    saft_parameters_for,
    site_fractions,
)

PROPANE = saft_parameters_for(["propane"])
ETHANOL = saft_parameters_for(["ethanol"])
MIX = saft_parameters_for(["propane", "n-butane"])


# ---------------------------------------------------------------------------
# Defining relations of the residual Helmholtz energy
# ---------------------------------------------------------------------------


def test_pressure_is_the_density_derivative_of_the_residual_energy() -> None:
    """P = rho R T [1 + rho (d alpha_res/d rho)], the defining relation."""
    rho, t = 8000.0, 320.0
    x = jnp.ones(1)
    da_drho = float(jax.grad(lambda r: alpha_residual(PROPANE, r, t, x))(jnp.asarray(rho)))
    p_from_a = rho * R * t * (1.0 + rho * da_drho)
    assert abs(p_from_a / float(pressure(PROPANE, rho, t, x)) - 1.0) < 1e-12


def test_compressibility_factor_matches_pressure_definition() -> None:
    rho, t = 6000.0, 330.0
    x = jnp.ones(1)
    z = float(compressibility_factor(PROPANE, rho, t, x))
    z_from_p = float(pressure(PROPANE, rho, t, x)) / (rho * R * t)
    assert abs(z / z_from_p - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Fugacity / residual-property cross-checks (independent code paths)
# ---------------------------------------------------------------------------


def test_mole_fraction_weighted_log_phi_equals_residual_gibbs() -> None:
    """sum_i x_i ln phi_i = alpha_res + (Z - 1) - ln Z.

    The left side comes from the autodiff mole-number gradient of the *total*
    residual energy (a vector); the right side from the *scalar* energy and the
    density derivative. Their agreement ties the two property routes together.
    """
    t, p = 320.0, 8e5
    x = jnp.array([0.4, 0.6])
    for phase in ("liquid", "vapor"):
        rho = molar_density(MIX, t, p, x, phase=phase)
        z = float(compressibility_factor(MIX, rho, t, x))
        a = float(alpha_residual(MIX, rho, t, x))
        ln_phi = ln_fugacity_coefficients(MIX, t, p, x, phase=phase)
        lhs = float(jnp.sum(x * ln_phi))
        rhs = float(a + (z - 1.0) - jnp.log(z))
        assert abs(lhs - rhs) < 1e-9


def test_residual_gibbs_is_enthalpy_minus_t_entropy() -> None:
    t, p = 330.0, 6e5
    x = jnp.array([0.5, 0.5])
    res = residual_properties(MIX, t, p, x, phase="vapor")
    assert abs(float(res.gibbs) - (float(res.enthalpy) - t * float(res.entropy))) < 1e-7


def test_residual_enthalpy_matches_gibbs_helmholtz_derivative() -> None:
    """H_res = -T^2 d/dT (G_res / T) at constant P, with the density solve inside."""
    t, p = 300.0, 5e5
    x = jnp.array([0.5, 0.5])

    def g_res_over_t(tt: jnp.ndarray) -> jnp.ndarray:
        return residual_properties(MIX, tt, p, x, phase="vapor").gibbs / tt

    h_from_gibbs = -(t**2) * float(jax.grad(g_res_over_t)(jnp.asarray(t)))
    h_direct = float(residual_properties(MIX, t, p, x, phase="vapor").enthalpy)
    assert abs(h_from_gibbs / h_direct - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# AD vs finite differences
# ---------------------------------------------------------------------------


def test_residual_energy_density_gradient_matches_finite_difference() -> None:
    rho, t = 5000.0, 320.0
    x = jnp.ones(1)
    ad = float(jax.grad(lambda r: alpha_residual(PROPANE, r, t, x))(jnp.asarray(rho)))
    eps = rho * 1e-6
    fd = float(
        (alpha_residual(PROPANE, rho + eps, t, x) - alpha_residual(PROPANE, rho - eps, t, x))
        / (2 * eps)
    )
    assert abs(ad / fd - 1.0) < 1e-6


def test_vapor_density_solve_gradient_matches_finite_difference() -> None:
    t, x = 350.0, jnp.ones(1)

    def rho_of_p(pp: jnp.ndarray) -> jnp.ndarray:
        return molar_density(PROPANE, t, pp, x, phase="vapor")

    p = 3e5
    ad = float(jax.grad(rho_of_p)(jnp.asarray(p)))
    eps = p * 1e-6
    fd = float((rho_of_p(jnp.asarray(p + eps)) - rho_of_p(jnp.asarray(p - eps))) / (2 * eps))
    assert abs(ad / fd - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Differentiable solvers (implicit function theorem)
# ---------------------------------------------------------------------------


def test_saturation_pressure_slope_is_positive_and_matches_finite_difference() -> None:
    """AD d(psat)/dT through the Newton solve vs central differences."""
    guess = 9.97e5

    def psat_of_t(tt: jnp.ndarray) -> jnp.ndarray:
        return psat_saft(PROPANE, tt, guess)

    t = 300.0
    ad = float(jax.grad(psat_of_t)(jnp.asarray(t)))
    dt = 1e-2
    fd = float((psat_of_t(jnp.asarray(t + dt)) - psat_of_t(jnp.asarray(t - dt))) / (2 * dt))
    assert ad > 0.0
    assert abs(ad / fd - 1.0) < 1e-5


def test_saturation_pressure_is_differentiable_in_the_dispersion_energy() -> None:
    """d(psat)/d(epsilon) by AD vs finite differences: the headline capability.

    Differentiating a *solved* saturation pressure with respect to a PC-SAFT
    model parameter, through the equifugacity Newton solve, is what powers
    `fugacio.thermo.saft.regression`. Classical property libraries cannot do it.
    """
    t, guess = 300.0, 9.97e5

    def psat_of_scale(scale: jnp.ndarray) -> jnp.ndarray:
        perturbed = replace(PROPANE, epsilon=PROPANE.epsilon * scale)
        return psat_saft(perturbed, t, guess)

    ad = float(jax.grad(psat_of_scale)(jnp.asarray(1.0)))
    eps = 1e-6
    fd = float(
        (psat_of_scale(jnp.asarray(1.0 + eps)) - psat_of_scale(jnp.asarray(1.0 - eps))) / (2 * eps)
    )
    assert abs(ad / fd - 1.0) < 1e-5


def test_flash_vapor_fraction_is_differentiable_in_pressure() -> None:
    from fugacio.thermo import component_arrays
    from fugacio.thermo.saft import saft_model

    arr = component_arrays(["propane", "n-butane"])
    model = saft_model(MIX, arr["tc"], arr["pc"], arr["omega"])
    z = jnp.array([0.5, 0.5])
    t = 320.0

    def beta_of_p(pp: jnp.ndarray) -> jnp.ndarray:
        return model.flash_pt(t, pp, z).beta

    p = 8e5
    ad = float(jax.grad(beta_of_p)(jnp.asarray(p)))
    dp = p * 1e-5
    fd = float((beta_of_p(jnp.asarray(p + dp)) - beta_of_p(jnp.asarray(p - dp))) / (2 * dp))
    assert ad < 0.0  # compressing the system condenses vapour
    assert abs(ad / fd - 1.0) < 1e-3


def test_association_site_fraction_solve_gradient_matches_finite_difference() -> None:
    """d X_A/d rho through the Wertheim fixed point (its custom JVP) vs FD."""
    t, x = 298.15, jnp.ones(1)

    def xa_of_rho(rho: jnp.ndarray) -> jnp.ndarray:
        return site_fractions(ETHANOL, rho, t, x)[0][0]

    rho = float(molar_density(ETHANOL, t, 1e5, x, phase="liquid"))
    ad = float(jax.grad(xa_of_rho)(jnp.asarray(rho)))
    eps = rho * 1e-6
    fd = float((xa_of_rho(jnp.asarray(rho + eps)) - xa_of_rho(jnp.asarray(rho - eps))) / (2 * eps))
    assert ad < 0.0  # denser fluid is more strongly associated (smaller free fraction)
    assert abs(ad / fd - 1.0) < 1e-5


def test_residual_energy_gradients_are_finite_across_the_density_plane() -> None:
    """Gradient sweep for an associating fluid, including the dilute limit."""
    rhos = jnp.array([1e-3, 1.0, 100.0, 5000.0, 20000.0])
    temps = jnp.array([280.0, 350.0, 450.0])
    x = jnp.ones(1)

    def g(rho: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(lambda r: alpha_residual(ETHANOL, r, t, x))(rho)

    grid = jax.vmap(lambda r: jax.vmap(lambda t: g(r, t))(temps))(rhos)
    assert bool(jnp.all(jnp.isfinite(grid)))
