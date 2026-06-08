"""Regular-solution and Flory-Huggins activity-coefficient models.

Two classic *predictive-ish* models that need only pure-component descriptors,
not fitted binary parameters:

* **Scatchard-Hildebrand regular solution** -- non-polar mixtures whose excess
  enthalpy is driven by differences in the cohesive energy density (the
  solubility parameter ``delta_i``)::

      ln(gamma_i) = v_i (delta_i - delta_bar)**2 / (R T)

  with volume fraction ``phi_i = x_i v_i / sum_j x_j v_j`` and the
  volume-fraction-averaged solubility parameter ``delta_bar = sum_i phi_i delta_i``.
  Regular-solution theory assumes an athermal *entropy* of mixing, so the excess
  Gibbs energy is purely the enthalpic term above (all ``gamma_i >= 1``).

* **Flory-Huggins** -- the athermal size-asymmetry (combinatorial) contribution
  for mixtures of very different molecular volumes (polymer/solvent)::

      ln(gamma_i) = ln(phi_i / x_i) + 1 - phi_i / x_i

  which is negative, capturing the entropic stabilisation of mixing unequal-size
  molecules. Adding the two gives the Flory-Huggins-Hildebrand model.

Both are differentiable with respect to composition, temperature, and the
pure-component descriptors (so the descriptors themselves can be regressed).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import R

ArrayLike = Array | float


def volume_fractions(x: Array, volume: Array) -> Array:
    """Volume (segment) fractions ``phi_i = x_i v_i / sum_j x_j v_j``."""
    x = jnp.asarray(x)
    volume = jnp.asarray(volume)
    weighted = x * volume
    return weighted / jnp.sum(weighted)


def regular_solution_ln_gamma(x: Array, volume: Array, delta: Array, t: ArrayLike) -> Array:
    """Log activity coefficients from Scatchard-Hildebrand regular-solution theory.

    Args:
        x: Mole fractions, shape ``(n,)``.
        volume: Liquid molar volumes ``v_i`` (m^3/mol), shape ``(n,)``.
        delta: Solubility parameters ``delta_i`` (Pa**0.5 = J**0.5/m**1.5), shape ``(n,)``.
        t: Temperature (K).

    Returns:
        ``ln(gamma)`` of shape ``(n,)`` (all non-negative).
    """
    volume = jnp.asarray(volume)
    delta = jnp.asarray(delta)
    t = jnp.asarray(t)
    phi = volume_fractions(x, volume)
    delta_bar = jnp.sum(phi * delta)
    return volume * (delta - delta_bar) ** 2 / (R * t)


def regular_solution_gamma(x: Array, volume: Array, delta: Array, t: ArrayLike) -> Array:
    """Activity coefficients from regular-solution theory."""
    return jnp.exp(regular_solution_ln_gamma(x, volume, delta, t))


def flory_huggins_ln_gamma(x: Array, volume: Array) -> Array:
    """Athermal Flory-Huggins (size-asymmetry) log activity coefficients.

    Args:
        x: Mole fractions, shape ``(n,)``.
        volume: Molecular size descriptors ``v_i`` (molar volume or van der Waals
            volume; only ratios matter), shape ``(n,)``.

    Returns:
        ``ln(gamma)`` of shape ``(n,)`` (the combinatorial part, non-positive).
    """
    x = jnp.asarray(x)
    phi = volume_fractions(x, volume)
    return jnp.log(phi / x) + 1.0 - phi / x


def flory_huggins_gamma(x: Array, volume: Array) -> Array:
    """Activity coefficients from the athermal Flory-Huggins model."""
    return jnp.exp(flory_huggins_ln_gamma(x, volume))


def hildebrand_ln_gamma(x: Array, volume: Array, delta: Array, t: ArrayLike) -> Array:
    """Flory-Huggins-Hildebrand ``ln(gamma)``: regular-solution enthalpy + FH entropy.

    The sum of :func:`regular_solution_ln_gamma` (enthalpic) and
    :func:`flory_huggins_ln_gamma` (combinatorial/entropic), giving a model that
    handles both energetic and size-asymmetry effects with pure-component data.
    """
    return regular_solution_ln_gamma(x, volume, delta, t) + flory_huggins_ln_gamma(x, volume)
