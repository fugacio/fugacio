"""Multicomponent UNIQUAC activity-coefficient model.

UNIQUAC (universal quasi-chemical) splits ``ln(gamma)`` into a *combinatorial*
part (driven by molecular size ``r`` and surface area ``q``) and a *residual*
part (driven by interaction energies ``tau_ij = exp(-Delta u_ij / R T)``)::

    ln(gamma_i) = ln(gamma_i^comb) + ln(gamma_i^res)

with, using segment fraction ``phi_i`` and area fraction ``theta_i`` and the
lattice coordination number ``z = 10``::

    phi_i = r_i x_i / sum_j r_j x_j
    theta_i = q_i x_i / sum_j q_j x_j
    l_i = (z/2)(r_i - q_i) - (r_i - 1)
    ln(gamma_i^comb) = ln(phi_i/x_i) + (z/2) q_i ln(theta_i/phi_i)
        + l_i - (phi_i/x_i) sum_j x_j l_j
    ln(gamma_i^res) = q_i [ 1 - ln(sum_j theta_j tau_ji)
        - sum_j theta_j tau_ij / (sum_k theta_k tau_kj) ]

UNIQUAC is the theoretical parent of the group-contribution UNIFAC model
(:mod:`fugacio.thermo.groupcontrib.unifac`). All quantities are differentiable.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import R

ArrayLike = Array | float

#: Lattice coordination number used by UNIQUAC/UNIFAC.
Z_COORD = 10.0


def uniquac_tau(t: ArrayLike, du: Array) -> Array:
    """Build ``tau_ij = exp(-Delta u_ij / (R T))`` from interaction energies (J/mol)."""
    t = jnp.asarray(t)
    return jnp.exp(-jnp.asarray(du) / (R * t))


def uniquac_ln_gamma(x: Array, r: Array, q: Array, tau: Array) -> Array:
    """Log activity coefficients for a multicomponent UNIQUAC mixture.

    Args:
        x: Mole fractions, shape ``(n,)``.
        r: Volume (size) parameters ``r_i``, shape ``(n,)``.
        q: Surface-area parameters ``q_i``, shape ``(n,)``.
        tau: Interaction matrix ``tau_ij``, shape ``(n, n)``, diagonal 1.

    Returns:
        ``ln(gamma)`` of shape ``(n,)``.
    """
    x = jnp.asarray(x)
    r = jnp.asarray(r)
    q = jnp.asarray(q)
    tau = jnp.asarray(tau)

    phi = r * x / jnp.sum(r * x)
    theta = q * x / jnp.sum(q * x)
    ell = (Z_COORD / 2.0) * (r - q) - (r - 1.0)

    ln_gamma_c = (
        jnp.log(phi / x)
        + (Z_COORD / 2.0) * q * jnp.log(theta / phi)
        + ell
        - (phi / x) * jnp.sum(x * ell)
    )

    s = theta @ tau  # S_j = sum_k theta_k tau_kj
    ln_gamma_r = q * (1.0 - jnp.log(s) - tau @ (theta / s))
    return ln_gamma_c + ln_gamma_r


def uniquac_gamma(x: Array, r: Array, q: Array, tau: Array) -> Array:
    """Activity coefficients for a multicomponent UNIQUAC mixture."""
    return jnp.exp(uniquac_ln_gamma(x, r, q, tau))
