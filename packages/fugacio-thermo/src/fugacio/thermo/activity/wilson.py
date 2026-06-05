"""Multicomponent Wilson activity-coefficient model.

Wilson's local-composition model is excellent for miscible mixtures (it cannot,
by construction, predict liquid-liquid splitting). For component ``i``::

    ln(gamma_i) = 1 - ln(sum_j x_j Lambda_ij) - sum_k [ x_k Lambda_ki / (sum_j x_j Lambda_kj) ]

with ``Lambda_ii = 1``. The interaction matrix is commonly built from molar
volumes and energy differences::

    Lambda_ij = (v_j / v_i) * exp(-(lambda_ij - lambda_ii) / (R T))

Everything is differentiable with respect to composition and the ``Lambda``
matrix (or the underlying energies/volumes via :func:`wilson_lambda`).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import R

ArrayLike = Array | float


def wilson_lambda(t: ArrayLike, volume: Array, energy: Array) -> Array:
    """Build the Wilson ``Lambda`` matrix from molar volumes and energy parameters.

    Args:
        t: Temperature (K).
        volume: Liquid molar volumes ``v_i``, shape ``(n,)`` (any consistent unit).
        energy: Energy differences ``(lambda_ij - lambda_ii)`` in J/mol, shape ``(n, n)``.

    Returns:
        ``Lambda`` matrix of shape ``(n, n)``.
    """
    t = jnp.asarray(t)
    volume = jnp.asarray(volume)
    v_ratio = volume[None, :] / volume[:, None]
    return v_ratio * jnp.exp(-jnp.asarray(energy) / (R * t))


def wilson_ln_gamma(x: Array, lam: Array) -> Array:
    """Log activity coefficients for a multicomponent Wilson mixture.

    Args:
        x: Mole fractions, shape ``(n,)``.
        lam: Wilson interaction matrix ``Lambda_ij``, shape ``(n, n)``, diagonal 1.

    Returns:
        ``ln(gamma)`` of shape ``(n,)``.
    """
    x = jnp.asarray(x)
    lam = jnp.asarray(lam)
    s = lam @ x  # S_i = sum_j x_j Lambda_ij
    return 1.0 - jnp.log(s) - lam.T @ (x / s)


def wilson_gamma(x: Array, lam: Array) -> Array:
    """Activity coefficients for a multicomponent Wilson mixture."""
    return jnp.exp(wilson_ln_gamma(x, lam))


def wilson_excess_gibbs(x: Array, lam: Array) -> Array:
    """Dimensionless excess Gibbs energy ``g^E / (R T)`` for a Wilson mixture."""
    x = jnp.asarray(x)
    return -jnp.sum(x * jnp.log(jnp.asarray(lam) @ x))
