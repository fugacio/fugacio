"""Activity-coefficient models for non-ideal liquid mixtures.

The two-parameter Margules model for a binary mixture is derived from the
dimensionless excess Gibbs energy::

    g^E / (R T) = x1 * x2 * (A21 * x1 + A12 * x2)

which yields the activity coefficients::

    ln(gamma_1) = x2**2 * (A12 + 2 * (A21 - A12) * x1)
    ln(gamma_2) = x1**2 * (A21 + 2 * (A12 - A21) * x2)

Because both activity coefficients come from a single ``g^E`` surface, the model
is thermodynamically consistent (it satisfies the Gibbs-Duhem relation) -- a
property the test-suite verifies directly via automatic differentiation.

All functions are written in ``jax.numpy`` and are differentiable with respect to
both composition and the model parameters ``A12`` / ``A21``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

ArrayLike = Array | float


def margules_excess_gibbs(x1: ArrayLike, a12: ArrayLike, a21: ArrayLike) -> Array:
    """Dimensionless excess Gibbs energy ``g^E / (R T)`` of a binary mixture.

    Args:
        x1: Mole fraction of component 1, in ``[0, 1]``.
        a12: Margules parameter ``A12`` (equals ``ln(gamma_1)`` at infinite dilution).
        a21: Margules parameter ``A21`` (equals ``ln(gamma_2)`` at infinite dilution).

    Returns:
        The dimensionless excess Gibbs energy.
    """
    x1 = jnp.asarray(x1)
    x2 = 1.0 - x1
    return x1 * x2 * (a21 * x1 + a12 * x2)


def margules_ln_gamma(x1: ArrayLike, a12: ArrayLike, a21: ArrayLike) -> tuple[Array, Array]:
    """Natural log of the activity coefficients for a binary Margules mixture.

    Args:
        x1: Mole fraction of component 1, in ``[0, 1]``.
        a12: Margules parameter ``A12``.
        a21: Margules parameter ``A21``.

    Returns:
        A tuple ``(ln_gamma1, ln_gamma2)``.
    """
    x1 = jnp.asarray(x1)
    x2 = 1.0 - x1
    ln_gamma1 = x2**2 * (a12 + 2.0 * (a21 - a12) * x1)
    ln_gamma2 = x1**2 * (a21 + 2.0 * (a12 - a21) * x2)
    return ln_gamma1, ln_gamma2


def margules_gamma(x1: ArrayLike, a12: ArrayLike, a21: ArrayLike) -> tuple[Array, Array]:
    """Activity coefficients ``(gamma_1, gamma_2)`` for a binary Margules mixture."""
    ln_gamma1, ln_gamma2 = margules_ln_gamma(x1, a12, a21)
    return jnp.exp(ln_gamma1), jnp.exp(ln_gamma2)
