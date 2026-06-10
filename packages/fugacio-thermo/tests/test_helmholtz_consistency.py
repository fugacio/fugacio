"""First-principles consistency of the Helmholtz package under autodiff.

These are data-free oracles: exact thermodynamic identities that must hold
between *independently computed* quantities. Because every property and every
solver in :mod:`fugacio.thermo.helmholtz` is differentiable, the identities
can be checked through :func:`jax.grad` itself -- e.g. Clausius-Clapeyron
compares the AD slope of the *solved* saturation line (a gradient through a
2x2 Newton solve, via the implicit function theorem) against latent heat over
volume change computed from plain property evaluations. Agreement to near
machine precision validates the EOS derivatives, the solver implicit rules,
and the property formulas simultaneously.
"""

import jax
import jax.numpy as jnp

from fugacio.thermo import helmholtz as hh
from fugacio.thermo.helmholtz.terms import residual_alpha

WATER = hh.reference_fluid("water")


def test_clausius_clapeyron_through_the_maxwell_solve() -> None:
    """AD d(psat)/dT along the solved line equals h_vap / (T dv)."""
    for t in (300.0, 450.0, 600.0):
        dpdt = jax.grad(lambda tt: hh.saturation_pressure(WATER, tt))(jnp.asarray(t))
        sat = hh.saturation_state(WATER, t=t)
        dv = 1.0 / sat.rho_vapor - 1.0 / sat.rho_liquid
        clapeyron = sat.h_vaporization / (t * dv)
        assert abs(float(dpdt - clapeyron) / float(clapeyron)) < 5e-9


def test_saturation_temperature_gradient_is_reciprocal_slope() -> None:
    """d(Tsat)/dP from one solve is the reciprocal of d(psat)/dT from the other."""
    p = 1e6
    dtdp = jax.grad(lambda pp: hh.saturation_temperature(WATER, pp))(jnp.asarray(p))
    t_sat = hh.saturation_temperature(WATER, p)
    dpdt = jax.grad(lambda tt: hh.saturation_pressure(WATER, tt))(t_sat)
    assert abs(float(dtdp) * float(dpdt) - 1.0) < 1e-9


def test_pressure_is_density_derivative_of_helmholtz_energy() -> None:
    """P = rho^2 d(a)/d(rho) at constant T -- the defining relation."""
    rho, t = 40000.0, 500.0
    da_drho = jax.grad(lambda r: hh.helmholtz_energy(WATER, r, t))(jnp.asarray(rho))
    p_from_a = rho**2 * float(da_drho)
    p_direct = float(hh.pressure(WATER, rho, t))
    assert abs(p_from_a / p_direct - 1.0) < 1e-12


def test_entropy_is_temperature_derivative_of_helmholtz_energy() -> None:
    """s = -d(a)/dT at constant rho."""
    rho, t = 40000.0, 500.0
    da_dt = jax.grad(lambda tt: hh.helmholtz_energy(WATER, rho, tt))(jnp.asarray(t))
    assert abs(-float(da_dt) / float(hh.entropy(WATER, rho, t)) - 1.0) < 1e-12


def test_maxwell_relation_ds_dv_equals_dp_dt() -> None:
    """(ds/dv)_T = (dP/dT)_v, evaluated as two unrelated AD paths."""
    rho, t = 35000.0, 520.0
    ds_drho = jax.grad(lambda r: hh.entropy(WATER, r, t))(jnp.asarray(rho))
    ds_dv = -(rho**2) * float(ds_drho)  # v = 1/rho
    dp_dt = float(jax.grad(lambda tt: hh.pressure(WATER, rho, tt))(jnp.asarray(t)))
    assert abs(ds_dv / dp_dt - 1.0) < 1e-12


def test_cp_minus_cv_identity() -> None:
    """cp - cv = T alpha_V^2 / (rho kappa_T), all five from independent formulas."""
    rho, t = 45000.0, 420.0
    cp = float(hh.isobaric_heat_capacity(WATER, rho, t))
    cv = float(hh.isochoric_heat_capacity(WATER, rho, t))
    alpha_v = float(hh.isobaric_expansivity(WATER, rho, t))
    kappa_t = float(hh.isothermal_compressibility(WATER, rho, t))
    assert abs((cp - cv) / (t * alpha_v**2 / (rho * kappa_t)) - 1.0) < 1e-11


def test_joule_thomson_identity() -> None:
    """mu_JT = (T alpha_V - 1) / (rho cp)."""
    rho, t = 300.0, 700.0  # superheated steam
    mu_jt = float(hh.joule_thomson(WATER, rho, t))
    alpha_v = float(hh.isobaric_expansivity(WATER, rho, t))
    cp = float(hh.isobaric_heat_capacity(WATER, rho, t))
    assert abs(mu_jt / ((t * alpha_v - 1.0) / (rho * cp)) - 1.0) < 1e-10


def test_gibbs_helmholtz_relation_through_ad() -> None:
    """h = -T^2 d(g/T)/dT at constant P, with the density solve inside the grad."""
    t, p = 360.0, 2e5

    def g_over_t(tt: jnp.ndarray) -> jnp.ndarray:
        state = hh.state_tp(WATER, tt, jnp.asarray(p))
        return state.g / tt

    h_from_gibbs = -(t**2) * float(jax.grad(g_over_t)(jnp.asarray(t)))
    h_direct = float(hh.state_tp(WATER, t, p).h)
    assert abs(h_from_gibbs / h_direct - 1.0) < 1e-9


def test_density_solve_gradient_matches_finite_difference() -> None:
    t, p = 310.0, 5e6

    def rho_of_p(pp: jnp.ndarray) -> jnp.ndarray:
        return hh.molar_density(WATER, t, pp, phase="liquid")

    ad = float(jax.grad(rho_of_p)(jnp.asarray(p)))
    eps = p * 1e-6
    fd = float((rho_of_p(jnp.asarray(p + eps)) - rho_of_p(jnp.asarray(p - eps))) / (2 * eps))
    assert abs(ad / fd - 1.0) < 1e-6
    # And it equals rho * kappa_T by definition of isothermal compressibility.
    rho = float(rho_of_p(jnp.asarray(p)))
    kappa_t = float(hh.isothermal_compressibility(WATER, rho, t))
    assert abs(ad / (rho * kappa_t) - 1.0) < 1e-9


def test_latent_heat_gradient_vanishes_at_critical_point() -> None:
    """h_vap shrinks monotonically toward Tc -- AD slope must be negative."""
    dhvap_dt = jax.grad(lambda tt: hh.saturation_state(WATER, t=tt).h_vaporization)(
        jnp.asarray(550.0)
    )
    assert float(dhvap_dt) < 0.0


def test_gradient_with_respect_to_eos_coefficients() -> None:
    """The EOS itself is differentiable: d(psat)/d(n_k) matches finite differences.

    This is the capability classical property libraries cannot offer -- the
    saturation pressure differentiated with respect to a *published
    correlation coefficient*, through the Maxwell construction, enabling
    sensitivity studies and EOS refitting by gradient descent.
    """
    from dataclasses import replace

    t = 450.0

    def psat_of_coeff(scale: jnp.ndarray) -> jnp.ndarray:
        perturbed = replace(WATER, power_n=WATER.power_n * scale)
        return hh.saturation_pressure(perturbed, jnp.asarray(t))

    ad = float(jax.grad(psat_of_coeff)(jnp.asarray(1.0)))
    eps = 1e-7
    fd = float(
        (psat_of_coeff(jnp.asarray(1.0 + eps)) - psat_of_coeff(jnp.asarray(1.0 - eps))) / (2 * eps)
    )
    assert abs(ad / fd - 1.0) < 1e-5


def test_virial_expansion_consistency() -> None:
    """Z -> 1 + B rho + C rho^2 in the low-density limit of the full EOS."""
    t = 550.0
    b = float(hh.second_virial(WATER, t))
    c = float(hh.third_virial(WATER, t))
    rho = 5.0  # mol/m^3: dilute enough for the quadratic truncation
    z = float(hh.compressibility_factor(WATER, rho, t))
    z_virial = 1.0 + b * rho + c * rho**2
    assert abs(z - z_virial) < 1e-9


def test_residual_alpha_gradients_have_no_nan_anywhere_relevant() -> None:
    """Gradient sweep over the (delta, tau) plane including the critical point."""
    deltas = jnp.array([1e-8, 0.3, 1.0, 1.7, 3.0])
    taus = jnp.array([0.6, 0.9, 1.0, 1.5, 2.5])

    def g(d: jnp.ndarray, tt: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(lambda x: residual_alpha(WATER, x[0], x[1]))(jnp.stack([d, tt]))

    grid = jax.vmap(lambda d: jax.vmap(lambda tt: g(d, tt))(taus))(deltas)
    assert bool(jnp.all(jnp.isfinite(grid)))


def test_transport_properties_are_differentiable() -> None:
    """d(mu)/dT of liquid water is negative and matches finite differences."""
    rho = 55000.0

    def mu(t: jnp.ndarray) -> jnp.ndarray:
        return hh.water_viscosity(t, jnp.asarray(rho))

    ad = float(jax.grad(mu)(jnp.asarray(300.0)))
    fd = float((mu(jnp.asarray(300.05)) - mu(jnp.asarray(299.95))) / 0.1)
    assert ad < 0.0
    assert abs(ad / fd - 1.0) < 1e-4
