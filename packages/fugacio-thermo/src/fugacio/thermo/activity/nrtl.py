"""Multicomponent NRTL (non-random two-liquid) activity-coefficient model.

NRTL represents strongly non-ideal and even partially miscible liquid mixtures
through binary interaction energies and a non-randomness factor. For component
``i`` in an ``n``-component mixture::

    ln(gamma_i) = (sum_j tau_ji G_ji x_j) / (sum_k G_ki x_k)
        + sum_j [ x_j G_ij / (sum_k G_kj x_k) ]
                * ( tau_ij - (sum_m x_m tau_mj G_mj) / (sum_k G_kj x_k) )

with ``G_ij = exp(-alpha_ij tau_ij)``, ``tau_ii = 0`` and ``alpha_ij = alpha_ji``.

The implementation is fully vectorised over the ``n x n`` parameter matrices and
differentiable with respect to composition *and* the ``tau`` / ``alpha``
parameters, which makes regressing NRTL parameters to VLE data a plain
gradient-descent problem.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

ArrayLike = Array | float


def nrtl_tau(t: ArrayLike, a: Array, b: Array, c: ArrayLike = 0.0) -> Array:
    """Build the ``tau`` matrix from the common temperature correlation.

    ``tau_ij = a_ij + b_ij / T + c_ij * ln(T)``. Pass arrays of shape ``(n, n)``
    for ``a`` and ``b`` (and optionally ``c``); the diagonal should be zero.
    """
    t = jnp.asarray(t)
    return a + b / t + jnp.asarray(c) * jnp.log(t)


def nrtl_g(tau: Array, alpha: Array) -> Array:
    """Compute ``G_ij = exp(-alpha_ij * tau_ij)`` from ``tau`` and ``alpha``."""
    return jnp.exp(-jnp.asarray(alpha) * jnp.asarray(tau))


def nrtl_ln_gamma(x: Array, tau: Array, alpha: Array) -> Array:
    """Log activity coefficients ``ln(gamma_i)`` for a multicomponent NRTL mixture.

    Args:
        x: Mole fractions, shape ``(n,)``.
        tau: Interaction parameter matrix ``tau_ij``, shape ``(n, n)``, diagonal 0.
        alpha: Non-randomness matrix ``alpha_ij = alpha_ji``, shape ``(n, n)``.

    Returns:
        ``ln(gamma)`` of shape ``(n,)``.
    """
    x = jnp.asarray(x)
    tau = jnp.asarray(tau)
    g = nrtl_g(tau, alpha)
    denom = x @ g  # D_j = sum_k x_k G_kj
    numer = x @ (g * tau)  # N_j = sum_k x_k G_kj tau_kj
    term1 = numer / denom  # indexed by i
    ratio = numer / denom  # N_j / D_j
    weighted = g * (tau - ratio[None, :])  # M_ij = G_ij (tau_ij - N_j/D_j)
    term2 = weighted @ (x / denom)
    return term1 + term2


def nrtl_gamma(x: Array, tau: Array, alpha: Array) -> Array:
    """Activity coefficients ``gamma_i`` for a multicomponent NRTL mixture."""
    return jnp.exp(nrtl_ln_gamma(x, tau, alpha))


def nrtl_excess_gibbs(x: Array, tau: Array, alpha: Array) -> Array:
    """Dimensionless excess Gibbs energy ``g^E / (R T)`` for an NRTL mixture."""
    x = jnp.asarray(x)
    g = nrtl_g(tau, alpha)
    denom = x @ g
    numer = x @ (g * jnp.asarray(tau))
    return jnp.sum(x * (numer / denom))
