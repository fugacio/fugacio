"""Van Laar activity-coefficient model for binary mixtures.

The van Laar equations describe mixtures whose excess Gibbs energy is asymmetric
in composition::

    ln(gamma_1) = A12 * (A21 * x2 / (A12 * x1 + A21 * x2))**2
    ln(gamma_2) = A21 * (A12 * x1 / (A12 * x1 + A21 * x2))**2

where ``A12`` and ``A21`` are the (temperature-dependent) infinite-dilution
activity coefficients ``ln(gamma_i^inf)``. Like Margules, the model derives from
a single excess-Gibbs surface and so satisfies the Gibbs-Duhem relation.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

ArrayLike = Array | float


def van_laar_ln_gamma(x1: ArrayLike, a12: ArrayLike, a21: ArrayLike) -> tuple[Array, Array]:
    """Natural log of the activity coefficients for a binary van Laar mixture.

    Args:
        x1: Mole fraction of component 1, in ``[0, 1]``.
        a12: Infinite-dilution value ``ln(gamma_1^inf)``.
        a21: Infinite-dilution value ``ln(gamma_2^inf)``.

    Returns:
        A tuple ``(ln_gamma1, ln_gamma2)``.
    """
    x1 = jnp.asarray(x1)
    x2 = 1.0 - x1
    denom = a12 * x1 + a21 * x2
    ln_gamma1 = a12 * (a21 * x2 / denom) ** 2
    ln_gamma2 = a21 * (a12 * x1 / denom) ** 2
    return ln_gamma1, ln_gamma2


def van_laar_gamma(x1: ArrayLike, a12: ArrayLike, a21: ArrayLike) -> tuple[Array, Array]:
    """Activity coefficients ``(gamma_1, gamma_2)`` for a binary van Laar mixture."""
    ln_gamma1, ln_gamma2 = van_laar_ln_gamma(x1, a12, a21)
    return jnp.exp(ln_gamma1), jnp.exp(ln_gamma2)
