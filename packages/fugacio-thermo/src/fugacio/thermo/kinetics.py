"""Differentiable reaction-rate laws.

A *rate law* maps the local state of a reacting mixture -- temperature ``T`` and
the concentration vector ``c`` (mol/m^3, aligned with a component list) -- to the
intensive rate of one reaction ``r`` (mol/m^3/s). The reactor models in
`fugacio.sim.reactors` multiply that rate by the stoichiometry of a
`Reaction` to get species production rates.

Every rate law here is a small frozen dataclass registered as a JAX pytree, so
its kinetic parameters (pre-exponential factors, activation energies, reaction
orders, adsorption constants) are *differentiable leaves*. That makes rate
constants fittable from data by the same machinery as the thermodynamic models
(`fugacio.thermo.regression`) and lets reactor outputs be differentiated
with respect to the kinetics.

Provided laws:

* `arrhenius` / `Arrhenius` -- the rate constant ``k(T) = A exp(-Ea/RT)``;
* `PowerLaw` -- an irreversible power-law rate ``k(T) * prod_i c_i^{m_i}``;
* `MassActionReversible` -- an elementary reversible rate
  ``k_f prod_{reactants} c^{|nu|} - k_r prod_{products} c^{nu}``;
* `LHHW` -- a Langmuir-Hinshelwood / Hougen-Watson rate with an adsorption
  inhibition term in the denominator.

Concentrations are clipped at zero before being raised to (possibly fractional)
powers, so a depleted species contributes a finite, well-behaved zero rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import T_REF, R

ArrayLike = Array | float


def arrhenius(t: ArrayLike, a: ArrayLike, ea: ArrayLike) -> Array:
    """Arrhenius rate constant ``k(T) = A exp(-Ea / R T)``.

    Args:
        t: Temperature (K).
        a: Pre-exponential factor (same units as the returned ``k``).
        ea: Activation energy (J/mol).
    """
    return jnp.asarray(a) * jnp.exp(-jnp.asarray(ea) / (R * jnp.asarray(t)))


def arrhenius_ref(t: ArrayLike, k_ref: ArrayLike, ea: ArrayLike, t_ref: float = T_REF) -> Array:
    """Reference-temperature Arrhenius form ``k(T) = k_ref exp(-Ea/R (1/T - 1/T_ref))``.

    Numerically better conditioned than `arrhenius` for regression, because
    ``k_ref`` is the rate constant at ``t_ref`` (an O(1)-scaled quantity) rather
    than the extrapolated intercept ``A``.
    """
    t = jnp.asarray(t)
    return jnp.asarray(k_ref) * jnp.exp(-jnp.asarray(ea) / R * (1.0 / t - 1.0 / t_ref))


def _pow(c: Array, orders: Array) -> Array:
    """``prod_i max(c_i, 0)^{orders_i}`` -- product of non-negative powers.

    Uses the "double where" trick so the result is differentiable even where a
    concentration is exactly zero: the power is evaluated on a base that is never
    zero, and the true value (1 for zero order, 0 for a depleted positive-order
    species) is selected afterwards. This keeps the forward value identical while
    avoiding the ``0**0`` / ``0**m`` NaN gradients that would otherwise propagate
    into reactor Jacobians for feeds with absent species.
    """
    safe = jnp.clip(c, 0.0, None)
    positive = safe > 0.0
    powered = jnp.where(positive, safe, 1.0) ** orders
    factors = jnp.where(orders == 0.0, 1.0, jnp.where(positive, powered, 0.0))
    return jnp.prod(factors)


@dataclass(frozen=True)
class Arrhenius:
    """An Arrhenius rate constant ``k(T) = a exp(-ea / R T)``.

    Attributes:
        a: Pre-exponential factor.
        ea: Activation energy (J/mol).
    """

    a: Array
    ea: Array

    def k(self, t: ArrayLike) -> Array:
        """Rate constant at temperature ``t``."""
        return arrhenius(t, self.a, self.ea)


@dataclass(frozen=True)
class PowerLaw:
    """Irreversible power-law rate ``r = k(T) * prod_i c_i^{orders_i}``.

    Attributes:
        a: Pre-exponential factor of the Arrhenius rate constant.
        ea: Activation energy (J/mol).
        orders: Reaction order in each component's concentration (aligned with
            ``c``); typically zero for species that do not appear in the rate.
    """

    a: Array
    ea: Array
    orders: Array

    def rate(self, t: ArrayLike, c: Array) -> Array:
        """Reaction rate (mol/m^3/s) at temperature ``t`` and concentrations ``c``."""
        return arrhenius(t, self.a, self.ea) * _pow(jnp.asarray(c), jnp.asarray(self.orders))


@dataclass(frozen=True)
class MassActionReversible:
    """Elementary reversible mass-action rate.

    ``r = k_f(T) prod_{reactants} c_i^{|nu_i|} - k_r(T) prod_{products} c_i^{nu_i}``

    with both rate constants in Arrhenius form. The stoichiometric vector ``nu``
    (negative reactants, positive products) sets the concentration orders, so the
    law is thermodynamically consistent: at equilibrium ``r = 0`` implies the
    concentration quotient equals ``k_f / k_r = K_c``.

    Attributes:
        a_f, ea_f: Forward pre-exponential and activation energy (J/mol).
        a_r, ea_r: Reverse pre-exponential and activation energy (J/mol).
        nu: Stoichiometric coefficients (negative reactants, positive products).
    """

    a_f: Array
    ea_f: Array
    a_r: Array
    ea_r: Array
    nu: Array

    def rate(self, t: ArrayLike, c: Array) -> Array:
        """Net reaction rate (mol/m^3/s) at temperature ``t`` and concentrations ``c``."""
        c = jnp.asarray(c)
        nu = jnp.asarray(self.nu)
        fwd_orders = jnp.where(nu < 0.0, -nu, 0.0)
        rev_orders = jnp.where(nu > 0.0, nu, 0.0)
        kf = arrhenius(t, self.a_f, self.ea_f)
        kr = arrhenius(t, self.a_r, self.ea_r)
        return kf * _pow(c, fwd_orders) - kr * _pow(c, rev_orders)


@dataclass(frozen=True)
class LHHW:
    """Langmuir-Hinshelwood / Hougen-Watson rate with adsorption inhibition.

    ``r = k(T) * prod_i c_i^{orders_i} / (1 + sum_i k_ads_i c_i^{ads_orders_i})^n``

    The numerator is a power-law driving force; the denominator captures
    competitive adsorption on a catalyst surface. ``n`` is the (static) number of
    active sites in the rate-determining step.

    Attributes:
        a, ea: Arrhenius parameters of the surface rate constant.
        orders: Concentration orders of the driving-force numerator.
        k_ads: Adsorption equilibrium constants per component (0 for inert/non-adsorbing).
        ads_orders: Concentration orders inside the adsorption sum.
        sites: Denominator exponent ``n`` (number of sites; static).
    """

    a: Array
    ea: Array
    orders: Array
    k_ads: Array
    ads_orders: Array
    sites: float = 1.0

    def rate(self, t: ArrayLike, c: Array) -> Array:
        """Reaction rate (mol/m^3/s) at temperature ``t`` and concentrations ``c``."""
        c = jnp.asarray(c)
        num = arrhenius(t, self.a, self.ea) * _pow(c, jnp.asarray(self.orders))
        safe = jnp.clip(c, 0.0, None)
        ads_orders = jnp.asarray(self.ads_orders)
        terms = jnp.where(ads_orders == 0.0, jnp.asarray(self.k_ads), self.k_ads * safe**ads_orders)
        # The constant 1 plus each adsorption term; drop the masked-to-k_ads ones
        # that correspond to truly absent species by requiring a positive k_ads.
        denom = (1.0 + jnp.sum(jnp.where(jnp.asarray(self.k_ads) > 0.0, terms, 0.0))) ** self.sites
        return num / denom


jax.tree_util.register_dataclass(Arrhenius, data_fields=["a", "ea"], meta_fields=[])
jax.tree_util.register_dataclass(PowerLaw, data_fields=["a", "ea", "orders"], meta_fields=[])
jax.tree_util.register_dataclass(
    MassActionReversible,
    data_fields=["a_f", "ea_f", "a_r", "ea_r", "nu"],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    LHHW,
    data_fields=["a", "ea", "orders", "k_ads", "ads_orders"],
    meta_fields=["sites"],
)


__all__ = [
    "LHHW",
    "Arrhenius",
    "MassActionReversible",
    "PowerLaw",
    "arrhenius",
    "arrhenius_ref",
]
