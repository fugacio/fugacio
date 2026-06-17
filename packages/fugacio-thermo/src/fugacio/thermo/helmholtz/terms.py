"""Reduced Helmholtz energy ``alpha(delta, tau)`` and its derivatives by autodiff.

A multiparameter reference EOS is *one scalar function*: the reduced molar
Helmholtz energy

    alpha(delta, tau) = alpha0(delta, tau) + alphar(delta, tau),

with ``delta = rho/rho_reducing`` and ``tau = t_reducing/T``. Every
thermodynamic property (pressure, heat capacities, speed of sound, fugacity,
Joule-Thomson coefficient) is an algebraic combination of partial
derivatives of ``alpha`` (`fugacio.thermo.helmholtz.props`). Reference
implementations hand-derive and hand-code those derivatives term family by
term family; Fugacio instead evaluates the scalar ``alpha`` and lets
`jax.grad` produce the exact derivatives. That is the clearest possible
demonstration of a differentiable thermodynamics core: one formula, every
property, machine-precision consistency between them by construction.

Term families (the union needed by the vendored fluids):

* ideal: lead ``a1 + a2*tau + ln(delta)``, ``a*ln(tau)``, power ``n*tau^t``,
  and Planck-Einstein ``n*ln(1 - exp(-t*tau))``;
* residual: power/exponential ``n delta^d tau^t exp(-delta^l)``, Gaussian
  bells, the non-analytic critical-region terms of IAPWS-95 / Span-Wagner CO2,
  and the GaoB terms of the 2023 ammonia EOS.

The non-analytic terms are genuinely singular *at* the critical point (by
design, they reproduce critical anomalies), so their derivatives are guarded
with the "double where" trick to return finite values on the critical
isochore/isotherm instead of NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.helmholtz.fluids import HelmholtzFluid

ArrayLike = Array | float


def _safe_pow(base: Array, exponent: Array) -> Array:
    """``base**exponent`` for ``base >= 0`` with finite gradients at ``base == 0``.

    ``x**e`` with fractional ``e < 1`` has an unbounded derivative at ``x = 0``;
    evaluated naively under autodiff that produces ``inf * 0 = nan``. The double
    ``where`` evaluates the power on a base that is never zero and selects the
    true value afterwards, so both the value (0) and the gradient (0) are
    finite exactly on the singular locus (the critical isochore of the
    non-analytic terms).
    """
    positive = base > 0.0
    safe = jnp.where(positive, base, 1.0)
    return jnp.where(positive, safe**exponent, 0.0)


def _delta_power(delta: Array, exponent: Array) -> Array:
    """``delta**exponent`` whose ``exponent == 0`` entries stay NaN-free at 0.

    Reverse-mode AD multiplies the power-rule factor ``e * delta**(e-1)`` by
    the incoming cotangent; for ``e = 0`` at ``delta = 0`` that factor is the
    indeterminate ``0 * inf``, poisoning gradients (the virial limit) even
    though the term's true derivative is zero. Masking the zero exponents to 1
    inside an outer ``where`` keeps both value and gradient exact.
    """
    nonzero = exponent != 0.0
    powered = delta ** jnp.where(nonzero, exponent, 1.0)
    return jnp.where(nonzero, powered, 1.0)


def ideal_alpha(fluid: HelmholtzFluid, delta: ArrayLike, tau: ArrayLike) -> Array:
    """Ideal-gas part ``alpha0(delta, tau)`` of the reduced Helmholtz energy."""
    delta = jnp.asarray(delta)
    tau = jnp.asarray(tau)
    total = jnp.log(delta) + fluid.lead_a1 + fluid.lead_a2 * tau + fluid.log_tau * jnp.log(tau)
    if fluid.ideal_power_n.shape[0]:
        total = total + jnp.sum(fluid.ideal_power_n * tau**fluid.ideal_power_t)
    if fluid.pe_n.shape[0]:
        total = total + jnp.sum(fluid.pe_n * jnp.log1p(-jnp.exp(-fluid.pe_t * tau)))
    return total


def residual_alpha(fluid: HelmholtzFluid, delta: ArrayLike, tau: ArrayLike) -> Array:
    """Residual part ``alphar(delta, tau)`` of the reduced Helmholtz energy."""
    delta = jnp.asarray(delta)
    tau = jnp.asarray(tau)
    total = jnp.asarray(0.0)
    if fluid.power_n.shape[0]:
        # l = 0 marks a plain polynomial term; the masked exponent keeps the
        # exp factor at exactly 1 there without a separate term family. The
        # exponent itself is also masked (to 1) so the unselected delta**0
        # branch cannot leak a NaN gradient at delta = 0 (the virial limit).
        exponential = fluid.power_l > 0.0
        damping = jnp.where(exponential, delta ** jnp.where(exponential, fluid.power_l, 1.0), 0.0)
        total = total + jnp.sum(
            fluid.power_n * delta**fluid.power_d * tau**fluid.power_t * jnp.exp(-damping)
        )
    if fluid.gauss_n.shape[0]:
        bell = jnp.exp(
            -fluid.gauss_eta * (delta - fluid.gauss_epsilon) ** 2
            - fluid.gauss_beta * (tau - fluid.gauss_gamma) ** 2
        )
        total = total + jnp.sum(
            fluid.gauss_n * _delta_power(delta, fluid.gauss_d) * tau**fluid.gauss_t * bell
        )
    if fluid.na_n.shape[0]:
        delta_sq = (delta - 1.0) ** 2
        tau_sq = (tau - 1.0) ** 2
        theta = (1.0 - tau) + fluid.na_big_a * _safe_pow(delta_sq, 1.0 / (2.0 * fluid.na_beta))
        big_delta = theta**2 + fluid.na_big_b * _safe_pow(delta_sq, fluid.na_a)
        psi = jnp.exp(-fluid.na_big_c * delta_sq - fluid.na_big_d * tau_sq)
        total = total + jnp.sum(fluid.na_n * _safe_pow(big_delta, fluid.na_b) * delta * psi)
    if fluid.gaob_n.shape[0]:
        bell = jnp.exp(
            fluid.gaob_eta * (delta - fluid.gaob_epsilon) ** 2
            + 1.0 / (fluid.gaob_beta * (tau - fluid.gaob_gamma) ** 2 + fluid.gaob_b)
        )
        total = total + jnp.sum(
            fluid.gaob_n * _delta_power(delta, fluid.gaob_d) * tau**fluid.gaob_t * bell
        )
    return total


@dataclass(frozen=True)
class AlphaDerivatives:
    """The partial derivatives of ``alpha`` that every property formula needs.

    Subscripts denote reduced-variable partials: ``ar_d`` is
    ``d(alphar)/d(delta)`` at constant ``tau``, ``a0_tt`` is
    ``d^2(alpha0)/d(tau)^2``, and so on. All are produced by `jax.grad`
    of the scalar term sums: no hand-derived derivative code exists in this
    package.
    """

    a0: Array
    a0_t: Array
    a0_tt: Array
    ar: Array
    ar_d: Array
    ar_dd: Array
    ar_t: Array
    ar_tt: Array
    ar_dt: Array


jax.tree_util.register_dataclass(
    AlphaDerivatives,
    data_fields=["a0", "a0_t", "a0_tt", "ar", "ar_d", "ar_dd", "ar_t", "ar_tt", "ar_dt"],
    meta_fields=[],
)


def first_derivatives(
    fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike
) -> tuple[Array, Array, Array, Array, Array]:
    """``(a0, a0_t, ar, ar_d, ar_t)``, the bundle first-law properties need.

    One reverse-mode pass over each scalar term sum produces both reduced
    partials at once, which keeps the traced graph small enough to embed in
    iterative solvers (density, Maxwell construction).
    """
    delta = jnp.asarray(rho, dtype=float) / fluid.rho_reducing
    tau = fluid.t_reducing / jnp.asarray(t, dtype=float)
    x = jnp.stack([delta, tau])

    def a0(v: Array) -> Array:
        return ideal_alpha(fluid, v[0], v[1])

    def ar(v: Array) -> Array:
        return residual_alpha(fluid, v[0], v[1])

    a0_value, a0_grad = jax.value_and_grad(a0)(x)
    ar_value, ar_grad = jax.value_and_grad(ar)(x)
    return a0_value, a0_grad[1], ar_value, ar_grad[0], ar_grad[1]


def alpha_derivatives(fluid: HelmholtzFluid, rho: ArrayLike, t: ArrayLike) -> AlphaDerivatives:
    """All ``alpha`` partials at molar density ``rho`` (mol/m^3) and ``t`` (K).

    First derivatives come from one reverse pass over the packed
    ``(delta, tau)`` vector; second derivatives from one forward-over-reverse
    Hessian. Nothing is hand-derived.
    """
    delta = jnp.asarray(rho, dtype=float) / fluid.rho_reducing
    tau = fluid.t_reducing / jnp.asarray(t, dtype=float)
    x = jnp.stack([delta, tau])

    def a0(v: Array) -> Array:
        return ideal_alpha(fluid, v[0], v[1])

    def ar(v: Array) -> Array:
        return residual_alpha(fluid, v[0], v[1])

    a0_value, a0_grad = jax.value_and_grad(a0)(x)
    a0_hess = jax.hessian(a0)(x)
    ar_value, ar_grad = jax.value_and_grad(ar)(x)
    ar_hess = jax.hessian(ar)(x)
    return AlphaDerivatives(
        a0=a0_value,
        a0_t=a0_grad[1],
        a0_tt=a0_hess[1, 1],
        ar=ar_value,
        ar_d=ar_grad[0],
        ar_dd=ar_hess[0, 0],
        ar_t=ar_grad[1],
        ar_tt=ar_hess[1, 1],
        ar_dt=ar_hess[0, 1],
    )


__all__ = [
    "AlphaDerivatives",
    "alpha_derivatives",
    "first_derivatives",
    "ideal_alpha",
    "residual_alpha",
]
