"""The PC-SAFT reduced residual Helmholtz energy ``alpha_res(rho, T, x)``.

PC-SAFT (Gross & Sadowski, *Ind. Eng. Chem. Res.* **2001**, 40, 1244) writes the
residual molar Helmholtz energy, divided by ``RT``, as a sum of physically
distinct contributions::

    alpha_res = alpha_hc + alpha_disp + alpha_assoc

* **hard chain** (``alpha_hc``): a Boublik-Mansoori-Carnahan-Starling-Leland
  hard-sphere reference plus the chain-formation term from Wertheim's first-order
  theory, built from the packing moments ``zeta_n`` and the contact radial
  distribution ``g_ij``;
* **dispersion** (``alpha_disp``): the perturbed-chain attractive term with the
  two universal-constant power series ``I_1(eta, mbar)`` and ``I_2(eta, mbar)``;
* **association** (``alpha_assoc``): Wertheim two-site (A/B) association, in
  `fugacio.thermo.saft.association`.

Exactly as for the reference multiparameter EOS
(`fugacio.thermo.helmholtz`), ``alpha_res`` is a *single scalar function* and
every PC-SAFT property is an autodiff derivative of it
(`fugacio.thermo.saft.properties`): one formula, every property, machine-precision
consistency by construction. All lengths are in metres and the number density is
``rho_N = N_A rho`` (per cubic metre), so the packing fraction ``eta = zeta_3`` and
the dispersion combinations are dimensionless.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import N_A
from fugacio.thermo.saft.association import alpha_association
from fugacio.thermo.saft.parameters import (
    SaftParameters,
    epsilon_ij,
    segment_diameter,
    sigma_ij,
)

ArrayLike = Array | float

#: PC-SAFT universal dispersion constants ``a_0i, a_1i, a_2i`` (Gross-Sadowski 2001).
_A_CONST = jnp.array(
    [
        [0.9105631445, -0.3084016918, -0.0906148351],
        [0.6361281449, 0.1860531159, 0.4527842806],
        [2.6861347891, -2.5030047259, 0.5962700728],
        [-26.547362491, 21.419793629, -1.7241829131],
        [97.759208784, -65.255885330, -4.1302112531],
        [-159.59154087, 83.318680481, 13.776631870],
        [91.297774084, -33.746922930, -8.6728470368],
    ]
)

#: PC-SAFT universal dispersion constants ``b_0i, b_1i, b_2i`` (Gross-Sadowski 2001).
_B_CONST = jnp.array(
    [
        [0.7240946941, -0.5755498075, 0.0976883116],
        [2.2382791861, 0.6995095521, -0.2557574982],
        [-4.0025849485, 3.8925673390, -9.1558561530],
        [-21.003576815, -17.215471648, 20.642075974],
        [26.855641363, 192.67226447, -38.804430052],
        [206.55133841, -161.82646165, 93.626774077],
        [-355.60235612, -165.20769346, -29.666905585],
    ]
)


def number_density(rho: ArrayLike) -> Array:
    """Molecular number density ``rho_N = N_A rho`` (1/m^3) from molar ``rho`` (mol/m^3)."""
    return N_A * jnp.asarray(rho, dtype=float)


def zeta_moments(params: SaftParameters, rho: ArrayLike, t: ArrayLike, x: Array) -> Array:
    """The four packing moments ``zeta_0..zeta_3`` (Gross-Sadowski eq. 9).

    ``zeta_n = (pi/6) rho_N sum_i x_i m_i d_i^n`` with ``rho_N`` the number
    density and ``d_i(T)`` the segment diameter. ``zeta_3`` is the packing
    fraction ``eta``.

    Returns:
        Array ``[zeta_0, zeta_1, zeta_2, zeta_3]``.
    """
    x = jnp.asarray(x, dtype=float)
    d = segment_diameter(params, t)
    rho_n = number_density(rho)
    powers = jnp.stack([d**0, d**1, d**2, d**3])  # shape (4, n)
    coeff = (jnp.pi / 6.0) * rho_n
    return coeff * jnp.sum(x * params.m * powers, axis=1)


def _g_hs_diagonal(params: SaftParameters, t: ArrayLike, zeta: Array) -> Array:
    """Contact radial distribution ``g_ii^hs`` for each component (Gross-Sadowski eq. 8)."""
    d = segment_diameter(params, t)
    z2, z3 = zeta[2], zeta[3]
    one_minus = 1.0 - z3
    dij = 0.5 * d  # d_i d_i / (d_i + d_i) = d_i / 2
    return 1.0 / one_minus + dij * 3.0 * z2 / one_minus**2 + dij**2 * 2.0 * z2**2 / one_minus**3


def alpha_hard_chain(params: SaftParameters, rho: ArrayLike, t: ArrayLike, x: Array) -> Array:
    """Hard-chain contribution ``alpha_hc`` to the reduced residual Helmholtz energy.

    The BMCSL hard-sphere reference scaled by the mean segment number, less the
    chain term ``sum_i x_i (m_i - 1) ln g_ii^hs`` (Gross-Sadowski eqs. 5-8).
    """
    x = jnp.asarray(x, dtype=float)
    zeta = zeta_moments(params, rho, t, x)
    z0, z1, z2, z3 = zeta[0], zeta[1], zeta[2], zeta[3]
    one_minus = 1.0 - z3
    a_hs = (
        3.0 * z1 * z2 / one_minus
        + z2**3 / (z3 * one_minus**2)
        + (z2**3 / z3**2 - z0) * jnp.log(one_minus)
    ) / z0
    mbar = jnp.sum(x * params.m)
    g_ii = _g_hs_diagonal(params, t, zeta)
    chain = jnp.sum(x * (params.m - 1.0) * jnp.log(g_ii))
    return mbar * a_hs - chain


def _i1_i2(mbar: Array, eta: Array) -> tuple[Array, Array]:
    """The dispersion power series ``I_1(eta, mbar)`` and ``I_2(eta, mbar)``."""
    ratio = (mbar - 1.0) / mbar
    ratio2 = ratio * (mbar - 2.0) / mbar
    a_coeff = _A_CONST[:, 0] + ratio * _A_CONST[:, 1] + ratio2 * _A_CONST[:, 2]
    b_coeff = _B_CONST[:, 0] + ratio * _B_CONST[:, 1] + ratio2 * _B_CONST[:, 2]
    powers = eta ** jnp.arange(7)
    return jnp.sum(a_coeff * powers), jnp.sum(b_coeff * powers)


def _c1(mbar: Array, eta: Array) -> Array:
    """Compressibility-derived dispersion factor ``C_1`` (Gross-Sadowski eq. A11)."""
    term1 = mbar * (8.0 * eta - 2.0 * eta**2) / (1.0 - eta) ** 4
    term2 = (
        (1.0 - mbar)
        * (20.0 * eta - 27.0 * eta**2 + 12.0 * eta**3 - 2.0 * eta**4)
        / ((1.0 - eta) * (2.0 - eta)) ** 2
    )
    return 1.0 / (1.0 + term1 + term2)


def _dispersion_sums(params: SaftParameters, t: ArrayLike, x: Array) -> tuple[Array, Array]:
    """The mixture combinations ``m^2 eps sigma^3`` and ``m^2 eps^2 sigma^3``."""
    t = jnp.asarray(t, dtype=float)
    x = jnp.asarray(x, dtype=float)
    eps = epsilon_ij(params) / t
    s3 = sigma_ij(params) ** 3
    xm = x * params.m
    outer = xm[:, None] * xm[None, :]
    m2es3 = jnp.sum(outer * eps * s3)
    m2e2s3 = jnp.sum(outer * eps**2 * s3)
    return m2es3, m2e2s3


def alpha_dispersion(params: SaftParameters, rho: ArrayLike, t: ArrayLike, x: Array) -> Array:
    """Dispersion contribution ``alpha_disp`` (Gross-Sadowski eqs. 16-19)."""
    x = jnp.asarray(x, dtype=float)
    mbar = jnp.sum(x * params.m)
    eta = zeta_moments(params, rho, t, x)[3]
    i1, i2 = _i1_i2(mbar, eta)
    c1 = _c1(mbar, eta)
    m2es3, m2e2s3 = _dispersion_sums(params, t, x)
    rho_n = number_density(rho)
    return -2.0 * jnp.pi * rho_n * i1 * m2es3 - jnp.pi * rho_n * mbar * c1 * i2 * m2e2s3


def alpha_residual(params: SaftParameters, rho: ArrayLike, t: ArrayLike, x: Array) -> Array:
    """Total PC-SAFT reduced residual Helmholtz energy ``A_res / (n R T)``.

    The sum of the hard-chain, dispersion, and (where active) association
    contributions, evaluated at molar density ``rho`` (mol/m^3), temperature
    ``t`` (K), and mole fractions ``x``.

    Args:
        params: PC-SAFT parameter set.
        rho: Molar density (mol/m^3).
        t: Temperature (K).
        x: Mole fractions, shape ``(n,)``.

    Returns:
        The dimensionless residual Helmholtz energy; every PC-SAFT property is an
        autodiff derivative of this scalar.
    """
    x = jnp.asarray(x, dtype=float)
    alpha = alpha_hard_chain(params, rho, t, x) + alpha_dispersion(params, rho, t, x)
    if params.associating:
        alpha = alpha + alpha_association(params, rho, t, x)
    return alpha


__all__ = [
    "alpha_dispersion",
    "alpha_hard_chain",
    "alpha_residual",
    "number_density",
    "zeta_moments",
]
