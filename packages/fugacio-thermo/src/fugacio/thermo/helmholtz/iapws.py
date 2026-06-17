"""IAPWS reference transport properties of water, with autodiff enhancements.

Implements the two current IAPWS formulations for ordinary water:

* viscosity: IAPWS R12-08 (Huber et al., J. Phys. Chem. Ref. Data 38, 2009);
* thermal conductivity: IAPWS R15-11 (Huber et al., J. Phys. Chem. Ref.
  Data 41, 2012).

Both are products of a dilute-gas term, a finite-density residual, and a
*critical enhancement*. The enhancement is where this implementation differs
from every transcription in open property libraries: it needs the isothermal
compressibility ``(d rho/d P)_T`` evaluated at the state *and* at the
reference temperature ``1.5 Tc``. Reference codes either require the caller
to supply those derivatives or fall back to a tabulated fit. Here they are
exact autodiff derivatives of the IAPWS-95 equation of state
(`fugacio.thermo.helmholtz`), so the *scientific* (not just industrial)
formulation is evaluated everywhere, closed over one differentiable graph.

Inputs are molar density (mol/m^3) and temperature (K), consistent with the
rest of the package; outputs are SI (Pa*s, W/m/K).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.helmholtz.fluids import HelmholtzFluid, reference_fluid
from fugacio.thermo.helmholtz.props import (
    isobaric_heat_capacity,
    isochoric_heat_capacity,
    pressure,
)

ArrayLike = Array | float

#: Reference constants shared by both formulations.
_T_STAR = 647.096  # K
_RHO_STAR = 322.0  # kg/m^3
_P_STAR = 22.064e6  # Pa
_NU_OVER_GAMMA = 0.630 / 1.239
_XI0 = 0.13e-9  # m
_GAMMA0 = 0.06
_T_R = 1.5  # reduced reference temperature of the enhancement

# IAPWS R12-08 dilute-gas denominators H_i.
_MU0_H = jnp.array([1.67752, 2.20462, 0.6366564, -0.241605])
# IAPWS R12-08 residual table: (i, j, H_ij) for (1/Tbar - 1)^i (rhobar - 1)^j.
_MU1_I = jnp.array([0, 1, 2, 3, 0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 0, 1, 0, 3, 4, 3, 5], dtype=float)
_MU1_J = jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6], dtype=float)
_MU1_H = jnp.array(
    [
        0.520094,
        0.0850895,
        -1.08374,
        -0.289555,
        0.222531,
        0.999115,
        1.88797,
        1.26613,
        0.120573,
        -0.281378,
        -0.906851,
        -0.772479,
        -0.489837,
        -0.257040,
        0.161913,
        0.257399,
        -0.0325372,
        0.0698452,
        0.00872102,
        -0.00435673,
        -0.000593264,
    ]
)
_MU_QC = 1.0 / 1.9e-9  # 1/m
_MU_QD = 1.0 / 1.1e-9  # 1/m
_MU_X = 0.068
_MU_XI_SWITCH = 0.3817016416e-9  # m

# IAPWS R15-11 dilute-gas denominators L_k.
_K0_L = jnp.array([2.443221e-3, 1.323095e-2, 6.770357e-3, -3.454586e-3, 4.096266e-4])
# IAPWS R15-11 residual table L_ij, i over (1/Tbar - 1), j over (rhobar - 1).
_K1_L = jnp.array(
    [
        [1.60397357, -0.646013523, 0.111443906, 0.102997357, -0.0504123634, 0.00609859258],
        [2.33771842, -2.78843778, 1.53616167, -0.463045512, 0.0832827019, -0.00719201245],
        [2.19650529, -4.54580785, 3.55777244, -1.40944978, 0.275418278, -0.0205938816],
        [-1.21051378, 1.60812989, -0.621178141, 0.0716373224, 0.0, 0.0],
        [-2.7203370, 4.57586331, -3.18369245, 1.1168348, -0.19268305, 0.012913842],
    ]
)
_K_LAMBDA = 177.8514
_K_QD = 1.0 / 0.4e-9  # 1/m


def _zeta(fluid: HelmholtzFluid, rho: Array, t: Array) -> Array:
    """Reduced compressibility ``(d rhobar / d Pbar)_T`` from the IAPWS-95 EOS."""

    def p_of_rho(r: Array) -> Array:
        return pressure(fluid, r, t)

    dp_drho = jax.grad(p_of_rho)(rho)  # Pa / (mol/m^3)
    return (_P_STAR / (_RHO_STAR / fluid.molar_mass)) / dp_drho


def _correlation_length(fluid: HelmholtzFluid, rho: Array, t: Array) -> Array:
    """Critical correlation length ``xi`` (m) of the enhancement terms."""
    rho_bar = rho * fluid.molar_mass / _RHO_STAR
    t_bar = t / _T_STAR
    zeta = _zeta(fluid, rho, t)
    zeta_ref = _zeta(fluid, rho, jnp.asarray(_T_R * _T_STAR))
    dchi = rho_bar * (zeta - zeta_ref * _T_R / t_bar)
    positive = dchi > 0.0
    safe = jnp.where(positive, dchi, 1.0)
    return jnp.where(positive, _XI0 * (safe / _GAMMA0) ** _NU_OVER_GAMMA, 0.0)


def _viscosity_base(fluid: HelmholtzFluid, rho: Array, t: Array) -> Array:
    """Dilute-gas times finite-density terms ``mu0 * mu1`` (in micro-Pa*s)."""
    t_bar = t / _T_STAR
    rho_bar = rho * fluid.molar_mass / _RHO_STAR
    mu0 = 100.0 * jnp.sqrt(t_bar) / jnp.sum(_MU0_H / t_bar ** jnp.arange(4.0))
    tot = jnp.sum(_MU1_H * (1.0 / t_bar - 1.0) ** _MU1_I * (rho_bar - 1.0) ** _MU1_J)
    return mu0 * jnp.exp(rho_bar * tot)


def _viscosity_enhancement(fluid: HelmholtzFluid, rho: Array, t: Array) -> Array:
    """Critical enhancement ``mu2`` (IAPWS R12-08, scientific formulation)."""
    xi = _correlation_length(fluid, rho, t)
    active = xi > 0.0
    xi_safe = jnp.where(active, xi, _XI0)

    qc_xi = _MU_QC * xi_safe
    qd_xi = _MU_QD * xi_safe
    psi_d = jnp.arccos(1.0 / jnp.sqrt(1.0 + qd_xi**2))
    w = jnp.sqrt(jnp.abs((qc_xi - 1.0) / (qc_xi + 1.0))) * jnp.tan(0.5 * psi_d)
    w_safe = jnp.clip(w, None, 1.0 - 1e-15)
    big_l = jnp.where(
        qc_xi > 1.0, jnp.log((1.0 + w_safe) / (1.0 - w_safe)), 2.0 * jnp.arctan(jnp.abs(w))
    )

    y_small = 0.2 * qc_xi * qd_xi**5 * (1.0 - qc_xi + qc_xi**2 - (765.0 / 504.0) * qd_xi**2)
    y_large = (
        jnp.sin(3.0 * psi_d) / 12.0
        - jnp.sin(2.0 * psi_d) / (4.0 * qc_xi)
        + (1.0 - 1.25 * qc_xi**2) * jnp.sin(psi_d) / qc_xi**2
        - ((1.0 - 1.5 * qc_xi**2) * psi_d - jnp.abs(qc_xi**2 - 1.0) ** 1.5 * big_l) / qc_xi**3
    )
    y = jnp.where(xi_safe <= _MU_XI_SWITCH, y_small, y_large)
    return jnp.where(active, jnp.exp(_MU_X * y), 1.0)


@jax.jit
def _water_viscosity(fluid: HelmholtzFluid, t: Array, rho: Array) -> Array:
    base = _viscosity_base(fluid, rho, t)
    return base * _viscosity_enhancement(fluid, rho, t) * 1e-6


def water_viscosity(t: ArrayLike, rho: ArrayLike) -> Array:
    """Viscosity of water (Pa*s) at ``t`` (K) and molar density ``rho`` (mol/m^3).

    IAPWS R12-08 including the critical enhancement, with the required
    ``(d rho/d P)_T`` derivatives taken from IAPWS-95 by autodiff.
    """
    return _water_viscosity(
        reference_fluid("water"), jnp.asarray(t, dtype=float), jnp.asarray(rho, dtype=float)
    )


@jax.jit
def _water_thermal_conductivity(fluid: HelmholtzFluid, t: Array, rho: Array) -> Array:
    t_bar = t / _T_STAR
    rho_bar = rho * fluid.molar_mass / _RHO_STAR

    k0 = jnp.sqrt(t_bar) / jnp.sum(_K0_L / t_bar ** jnp.arange(5.0))
    t_pows = (1.0 / t_bar - 1.0) ** jnp.arange(5.0)
    rho_pows = (rho_bar - 1.0) ** jnp.arange(6.0)
    k1 = jnp.exp(rho_bar * (t_pows @ _K1_L @ rho_pows))

    # Critical enhancement (R15-11 eq. 17-22): needs cp, cv from IAPWS-95 and
    # the full R12-08 viscosity (enhancement included, the convention of the
    # published check values), all on the same autodiff graph.
    xi = _correlation_length(fluid, rho, t)
    y = _K_QD * xi
    active = y > 1.2e-7
    y_safe = jnp.where(active, y, 1.0)
    cp = isobaric_heat_capacity(fluid, rho, t)
    cv = isochoric_heat_capacity(fluid, rho, t)
    cp_bar = jnp.clip(cp / fluid.gas_constant, 0.0, 1e13)
    kappa_inv = cv / cp
    z_y = (
        2.0
        / (jnp.pi * y_safe)
        * (
            ((1.0 - kappa_inv) * jnp.arctan(y_safe) + kappa_inv * y_safe)
            - (1.0 - jnp.exp(-1.0 / (1.0 / y_safe + y_safe**2 / (3.0 * rho_bar**2))))
        )
    )
    mu_bar = _viscosity_base(fluid, rho, t) * _viscosity_enhancement(fluid, rho, t)
    k2 = jnp.where(active, _K_LAMBDA * rho_bar * cp_bar * t_bar / mu_bar * z_y, 0.0)

    return (k0 * k1 + k2) * 1e-3


def water_thermal_conductivity(t: ArrayLike, rho: ArrayLike) -> Array:
    """Thermal conductivity of water (W/m/K) at ``t`` (K) and molar density ``rho`` (mol/m^3).

    IAPWS R15-11 including the critical enhancement; the correlation length is
    built from autodiff compressibilities of IAPWS-95 and the enhancement's
    ``cp``, ``cv`` and (non-enhanced) viscosity come from the same graph.
    """
    return _water_thermal_conductivity(
        reference_fluid("water"), jnp.asarray(t, dtype=float), jnp.asarray(rho, dtype=float)
    )


__all__ = ["water_thermal_conductivity", "water_viscosity"]
