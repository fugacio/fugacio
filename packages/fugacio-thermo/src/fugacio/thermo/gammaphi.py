"""Gamma-phi vapour-liquid equilibrium (activity-coefficient liquid model).

The cubic-EOS flash in `fugacio.thermo.equilibrium` describes both phases
with one equation of state (the *phi-phi* approach). For the low-pressure, polar,
strongly non-ideal mixtures that dominate real separations (ethanol/water and
other azeotropes, alcohol/ketone/water systems), a cubic EOS with zero binary
interaction parameters is simply the wrong tool. The standard answer is the
*gamma-phi* approach: model the liquid with an activity-coefficient model and the
vapour with an equation of state (or as an ideal gas at low pressure).

Equilibrium equates the component fugacities

    x_i gamma_i(x, T) f_i^{0,L}(T, P) = y_i phi_i^V(y, T, P) P

so the K-values are

    K_i = y_i / x_i = gamma_i f_i^{0,L} / (phi_i^V P).

With an ideal vapour (``phi^V = 1``) and the plain saturation reference
(``f^{0,L} = Psat``), this collapses to modified Raoult's law
``K_i = gamma_i Psat_i / P``, enough to reproduce azeotropes that the
zero-``kij`` cubic cannot. The richer reference (saturation fugacity coefficient +
Poynting, see `fugacio.thermo.reference`) and an EOS vapour are available via
keyword flags.

Every routine here, K-values, the four saturation calculations (bubble/dew at
fixed ``T`` or ``P``), and the isothermal flash, is a fixed point or a
bracketed root solved by the implicit-diff primitives in
`fugacio.thermo.implicit`, so each output is differentiable end-to-end with
respect to ``T``, ``P``, composition, *and* the activity-model parameters. That
last point is what turns parameter regression (`fugacio.thermo.regression`)
into plain gradient descent.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.activity.models import ActivityModel
from fugacio.thermo.eos import PR, CubicEOS, ln_phi_mixture
from fugacio.thermo.equilibrium import FlashResult, rachford_rice
from fugacio.thermo.implicit import bracketed_root, fixed_point
from fugacio.thermo.reference import liquid_reference_fugacity, saturation_pressures

ArrayLike = Array | float


def _ln_phi_vapor(
    eos: CubicEOS,
    t: ArrayLike,
    p: ArrayLike,
    y: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    kij: Array | None,
    vapor: str,
) -> Array:
    """Log vapour fugacity coefficients, or zeros for an ideal-gas vapour."""
    if vapor == "ideal":
        return jnp.zeros_like(jnp.asarray(y))
    if vapor == "eos":
        ln_phi, _ = ln_phi_mixture(eos, t, p, y, tc, pc, omega, phase="vapor", kij=kij)
        return ln_phi
    raise ValueError(f"unknown vapor model {vapor!r}; use 'ideal' or 'eos'")


def gamma_phi_k_values(
    model: ActivityModel,
    t: ArrayLike,
    p: ArrayLike,
    x: Array,
    y: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
) -> Array:
    """Gamma-phi K-values ``K_i = gamma_i f_i^{0,L} / (phi_i^V P)``.

    Args:
        model: Liquid activity-coefficient model.
        t: Temperature (K).
        p: Pressure (Pa).
        x: Liquid mole fractions.
        y: Vapour mole fractions (only matters for an EOS vapour).
        tc: Component critical temperatures (K).
        pc: Component critical pressures (Pa).
        omega: Component acentric factors.
        eos: Cubic EOS for the saturation reference and (if selected) the vapour.
        kij: Optional binary interaction matrix for the vapour EOS.
        vapor: ``"ideal"`` (phi^V = 1) or ``"eos"``.
        poynting: Include the Poynting pressure correction in the reference.
        phi_saturation: Include the saturation fugacity coefficient in the reference.

    Returns:
        K-values aligned with ``x``.
    """
    f_ref, _ = liquid_reference_fugacity(
        eos, t, p, tc, pc, omega, poynting=poynting, phi_saturation=phi_saturation
    )
    ln_gamma = model.ln_gamma(x, t)
    ln_phi_v = _ln_phi_vapor(eos, t, p, y, tc, pc, omega, kij, vapor)
    return jnp.exp(ln_gamma) * f_ref / (jnp.exp(ln_phi_v) * jnp.asarray(p))


def bubble_pressure_gamma(
    model: ActivityModel,
    t: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> tuple[Array, Array]:
    """Bubble-point pressure and incipient vapour composition at fixed ``T``, ``x``.

    Solved as a coupled fixed point in ``(ln P, y)``: K-values give an unnormalised
    vapour ``y* = K x`` whose sum scales the pressure until it is one. Returns
    ``(P, y)``, differentiable in ``T``, ``x`` and the model parameters.
    """
    x = jnp.asarray(x)
    psat = saturation_pressures(eos, t, tc, pc, omega)
    gamma0 = jnp.exp(model.ln_gamma(x, t))
    p0 = jnp.sum(x * gamma0 * psat)
    y0 = x * gamma0 * psat / p0
    state0 = jnp.concatenate([jnp.log(p0)[None], y0])
    theta = (model, jnp.asarray(t, dtype=float), x, tc, pc, omega)

    def g(state: Array, theta: Any) -> Array:
        model_, t_, x_, tc_, pc_, omega_ = theta
        p = jnp.exp(state[0])
        y = state[1:]
        k = gamma_phi_k_values(
            model_,
            t_,
            p,
            x_,
            y,
            tc_,
            pc_,
            omega_,
            eos=eos,
            kij=kij,
            vapor=vapor,
            poynting=poynting,
            phi_saturation=phi_saturation,
        )
        y_unnorm = k * x_
        s = jnp.sum(y_unnorm)
        return jnp.concatenate([(state[0] + jnp.log(s))[None], y_unnorm / s])

    state = fixed_point(g, state0, theta, tol, max_iter)
    return jnp.exp(state[0]), state[1:]


def dew_pressure_gamma(
    model: ActivityModel,
    t: ArrayLike,
    y: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> tuple[Array, Array]:
    """Dew-point pressure and incipient liquid composition at fixed ``T``, ``y``.

    Coupled fixed point in ``(ln P, x)``: with ``x_i = y_i / K_i`` (and the
    activity coefficients re-evaluated at the updated ``x``), the pressure is
    scaled until the liquid sums to one. Returns ``(P, x)``.
    """
    y = jnp.asarray(y)
    psat = saturation_pressures(eos, t, tc, pc, omega)
    p0 = 1.0 / jnp.sum(y / psat)
    x0 = y * p0 / psat
    x0 = x0 / jnp.sum(x0)
    state0 = jnp.concatenate([jnp.log(p0)[None], x0])
    theta = (model, jnp.asarray(t, dtype=float), y, tc, pc, omega)

    def g(state: Array, theta: Any) -> Array:
        model_, t_, y_, tc_, pc_, omega_ = theta
        p = jnp.exp(state[0])
        x = state[1:]
        k = gamma_phi_k_values(
            model_,
            t_,
            p,
            x,
            y_,
            tc_,
            pc_,
            omega_,
            eos=eos,
            kij=kij,
            vapor=vapor,
            poynting=poynting,
            phi_saturation=phi_saturation,
        )
        x_unnorm = y_ / k
        s = jnp.sum(x_unnorm)
        return jnp.concatenate([(state[0] - jnp.log(s))[None], x_unnorm / s])

    state = fixed_point(g, state0, theta, tol, max_iter)
    return jnp.exp(state[0]), state[1:]


def _bubble_temperature_residual_factory(
    model: ActivityModel,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    eos: CubicEOS,
    kij: Array | None,
    vapor: str,
    poynting: bool,
    phi_saturation: bool,
    inner_iter: int,
) -> Any:
    """Build ``sum_i K_i x_i - 1`` as a function of ``(T, params)`` for the T-bracket."""

    def residual(t: Array, params: Any) -> Array:
        model_, p_, x_, tc_, pc_, omega_ = params
        # Inner sweep for the incipient vapour (needed only for an EOS vapour).
        y = x_

        def step(_: int, y_cur: Array) -> Array:
            k = gamma_phi_k_values(
                model_,
                t,
                p_,
                x_,
                y_cur,
                tc_,
                pc_,
                omega_,
                eos=eos,
                kij=kij,
                vapor=vapor,
                poynting=poynting,
                phi_saturation=phi_saturation,
            )
            yn = k * x_
            return yn / jnp.sum(yn)

        y = jax.lax.fori_loop(0, inner_iter, step, y)
        k = gamma_phi_k_values(
            model_,
            t,
            p_,
            x_,
            y,
            tc_,
            pc_,
            omega_,
            eos=eos,
            kij=kij,
            vapor=vapor,
            poynting=poynting,
            phi_saturation=phi_saturation,
        )
        return jnp.sum(k * x_) - 1.0

    return residual


def bubble_temperature_gamma(
    model: ActivityModel,
    p: ArrayLike,
    x: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
    t_min: float = 150.0,
    t_max: float = 700.0,
    tol: float = 1e-9,
    max_iter: int = 200,
    inner_iter: int = 0,
) -> tuple[Array, Array]:
    """Bubble-point temperature and incipient vapour at fixed ``P``, ``x``.

    The bubble temperature is the root of ``sum_i K_i(T) x_i = 1`` (the saturation
    sum is monotone in ``T``), found with the bracketed solver and differentiated
    by the implicit function theorem. Returns ``(T, y)``. For an ideal vapour the
    K-values are independent of ``y`` and ``inner_iter`` can stay zero; raise it
    for an EOS vapour so the incipient ``y`` settles before the bracket step.
    """
    x = jnp.asarray(x)
    residual = _bubble_temperature_residual_factory(
        model, x, tc, pc, omega, eos, kij, vapor, poynting, phi_saturation, inner_iter
    )
    params = (model, jnp.asarray(p, dtype=float), x, tc, pc, omega)
    t_star = bracketed_root(residual, params, jnp.asarray(t_min), jnp.asarray(t_max), tol, max_iter)
    k = gamma_phi_k_values(
        model,
        t_star,
        p,
        x,
        x,
        tc,
        pc,
        omega,
        eos=eos,
        kij=kij,
        vapor=vapor,
        poynting=poynting,
        phi_saturation=phi_saturation,
    )
    y_unnorm = k * x
    return t_star, y_unnorm / jnp.sum(y_unnorm)


def dew_temperature_gamma(
    model: ActivityModel,
    p: ArrayLike,
    y: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
    t_min: float = 150.0,
    t_max: float = 700.0,
    tol: float = 1e-9,
    max_iter: int = 200,
    inner_iter: int = 30,
) -> tuple[Array, Array]:
    """Dew-point temperature and incipient liquid at fixed ``P``, ``y``.

    Root of ``sum_i (y_i / K_i(T)) = 1`` in ``T`` with an inner sweep that settles
    the incipient liquid ``x`` (on which the activity coefficients depend) at each
    trial temperature. Returns ``(T, x)``.
    """
    y = jnp.asarray(y)

    def liquid_at(t: Array, params: Any) -> Array:
        model_, p_, y_, tc_, pc_, omega_ = params
        psat = saturation_pressures(eos, t, tc_, pc_, omega_)
        x = y_ * (1.0 / jnp.sum(y_ / psat)) / psat
        x = x / jnp.sum(x)

        def step(_: int, x_cur: Array) -> Array:
            k = gamma_phi_k_values(
                model_,
                t,
                p_,
                x_cur,
                y_,
                tc_,
                pc_,
                omega_,
                eos=eos,
                kij=kij,
                vapor=vapor,
                poynting=poynting,
                phi_saturation=phi_saturation,
            )
            xn = y_ / k
            return xn / jnp.sum(xn)

        return jax.lax.fori_loop(0, inner_iter, step, x)

    def residual(t: Array, params: Any) -> Array:
        model_, p_, y_, tc_, pc_, omega_ = params
        x = liquid_at(t, params)
        k = gamma_phi_k_values(
            model_,
            t,
            p_,
            x,
            y_,
            tc_,
            pc_,
            omega_,
            eos=eos,
            kij=kij,
            vapor=vapor,
            poynting=poynting,
            phi_saturation=phi_saturation,
        )
        return jnp.sum(y_ / k) - 1.0

    params = (model, jnp.asarray(p, dtype=float), y, tc, pc, omega)
    t_star = bracketed_root(residual, params, jnp.asarray(t_min), jnp.asarray(t_max), tol, max_iter)
    return t_star, liquid_at(t_star, params)


def flash_pt_gamma(
    model: ActivityModel,
    t: ArrayLike,
    p: ArrayLike,
    z: Array,
    tc: Array,
    pc: Array,
    omega: Array,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
    tol: float = 1e-12,
    max_iter: int = 300,
) -> FlashResult:
    """Isothermal-isobaric gamma-phi flash by accelerated successive substitution.

    Iterates the gamma-phi K-values to a fixed point in ``ln K`` with the
    Rachford-Rice material balance closing the phase split at each step. The
    converged ``beta``, ``x``, ``y`` are differentiable with respect to
    ``(T, P, z)`` and the activity-model parameters.
    """
    z = jnp.asarray(z)
    psat = saturation_pressures(eos, t, tc, pc, omega)
    gamma0 = jnp.exp(model.ln_gamma(z, t))
    k0 = gamma0 * psat / jnp.asarray(p)
    theta = (model, jnp.asarray(t, dtype=float), jnp.asarray(p, dtype=float), z, tc, pc, omega)

    def g(ln_k: Array, theta: Any) -> Array:
        model_, t_, p_, z_, tc_, pc_, omega_ = theta
        k = jnp.exp(ln_k)
        beta = rachford_rice(z_, k)
        denom = 1.0 + beta * (k - 1.0)
        x = z_ / denom
        y = k * x
        k_new = gamma_phi_k_values(
            model_,
            t_,
            p_,
            x,
            y,
            tc_,
            pc_,
            omega_,
            eos=eos,
            kij=kij,
            vapor=vapor,
            poynting=poynting,
            phi_saturation=phi_saturation,
        )
        return jnp.log(k_new)

    ln_k_star = fixed_point(g, jnp.log(k0), theta, tol, max_iter)
    k = jnp.exp(ln_k_star)
    beta = rachford_rice(z, k)
    denom = 1.0 + beta * (k - 1.0)
    x = z / denom
    y = k * x
    return FlashResult(beta=beta, x=x, y=y, k=k)
