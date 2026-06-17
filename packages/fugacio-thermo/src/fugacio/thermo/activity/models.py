"""Object-oriented activity-coefficient models with a uniform, differentiable API.

The functional kernels in this package (``nrtl_ln_gamma``, ``uniquac_ln_gamma``,
...) are deliberately stateless. This module wraps them in small, immutable model
*objects* that

* bundle a model's parameters (and their temperature dependence) in one place;
* expose a single uniform method ``model.ln_gamma(x, T)`` so the
  gamma-phi equilibrium engine, the liquid-liquid solver, and the parameter
  regressor can treat every model interchangeably; and
* are registered as JAX pytrees whose *parameters are differentiable leaves*,
  so fitting a model to data is a plain gradient problem and a model can be
  threaded through ``jax.grad``/``jax.jit`` as part of a parameter pytree.

Every model implements the `ActivityModel` protocol. Two free functions,
`gamma` and `excess_gibbs`, derive the activity coefficients and the
(exact, model-independent) excess Gibbs energy ``g^E/(RT) = sum_i x_i ln gamma_i``
from any model's ``ln_gamma``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.activity.margules import margules_ln_gamma
from fugacio.thermo.activity.nrtl import nrtl_ln_gamma
from fugacio.thermo.activity.regular_solution import (
    flory_huggins_ln_gamma,
    hildebrand_ln_gamma,
    regular_solution_ln_gamma,
)
from fugacio.thermo.activity.uniquac import uniquac_ln_gamma
from fugacio.thermo.activity.vanlaar import van_laar_ln_gamma
from fugacio.thermo.activity.wilson import wilson_lambda, wilson_ln_gamma
from fugacio.thermo.constants import R

ArrayLike = Array | float


@runtime_checkable
class ActivityModel(Protocol):
    """Structural type for a liquid activity-coefficient model.

    A model maps a liquid composition and temperature to the vector of log
    activity coefficients ``ln(gamma_i)``. That single method is all the
    equilibrium engines require.
    """

    def ln_gamma(self, x: Array, t: ArrayLike) -> Array:
        """Log activity coefficients ``ln(gamma_i)`` at composition ``x``, temperature ``t``."""
        ...


def gamma(model: ActivityModel, x: Array, t: ArrayLike) -> Array:
    """Activity coefficients ``gamma_i`` from any `ActivityModel`."""
    return jnp.exp(model.ln_gamma(x, t))


def excess_gibbs(model: ActivityModel, x: Array, t: ArrayLike) -> Array:
    """Dimensionless excess Gibbs energy ``g^E/(RT) = sum_i x_i ln(gamma_i)``.

    This identity holds for *every* activity model (the log activity coefficients
    are the partial molar excess Gibbs energies), so it gives a single, exact,
    model-independent ``g^E``, handy for plotting and for stability tests.
    """
    x = jnp.asarray(x)
    return jnp.sum(x * model.ln_gamma(x, t))


# --- Local-composition models --------------------------------------------------


@dataclass(frozen=True)
class NRTL:
    """NRTL model with the standard ``tau_ij = a_ij + b_ij/T + e_ij ln T`` law.

    Attributes:
        a: Constant part of ``tau`` (dimensionless), shape ``(n, n)``.
        b: ``1/T`` part of ``tau`` (K), shape ``(n, n)``.
        alpha: Non-randomness factors ``alpha_ij = alpha_ji``, shape ``(n, n)``.
        e: ``ln T`` part of ``tau`` (dimensionless), shape ``(n, n)``; usually zero.
    """

    a: Array
    b: Array
    alpha: Array
    e: Array

    def tau(self, t: ArrayLike) -> Array:
        """Temperature-dependent ``tau`` matrix."""
        t = jnp.asarray(t)
        return self.a + self.b / t + self.e * jnp.log(t)

    def ln_gamma(self, x: Array, t: ArrayLike) -> Array:
        """Log activity coefficients ``ln(gamma_i)``."""
        return nrtl_ln_gamma(x, self.tau(t), self.alpha)


@dataclass(frozen=True)
class UNIQUAC:
    """UNIQUAC model with ``tau_ij = exp(a_ij + b_ij/T)``.

    Attributes:
        r: Volume (size) parameters ``r_i``, shape ``(n,)``.
        q: Surface-area parameters ``q_i``, shape ``(n,)``.
        a: Constant part of ``ln(tau)`` (dimensionless), shape ``(n, n)``.
        b: ``1/T`` part of ``ln(tau)`` (K), shape ``(n, n)``.
    """

    r: Array
    q: Array
    a: Array
    b: Array

    def tau(self, t: ArrayLike) -> Array:
        """Temperature-dependent ``tau`` matrix (``exp(a + b/T)``)."""
        t = jnp.asarray(t)
        return jnp.exp(self.a + self.b / t)

    def ln_gamma(self, x: Array, t: ArrayLike) -> Array:
        """Log activity coefficients ``ln(gamma_i)``."""
        return uniquac_ln_gamma(x, self.r, self.q, self.tau(t))


@dataclass(frozen=True)
class Wilson:
    """Wilson model built from molar volumes and interaction energies.

    Attributes:
        volume: Liquid molar volumes ``v_i`` (any consistent unit), shape ``(n,)``.
        energy: Energy differences ``(lambda_ij - lambda_ii)`` (J/mol), shape ``(n, n)``.
    """

    volume: Array
    energy: Array

    def lam(self, t: ArrayLike) -> Array:
        """Temperature-dependent Wilson ``Lambda`` matrix."""
        return wilson_lambda(t, self.volume, self.energy)

    def ln_gamma(self, x: Array, t: ArrayLike) -> Array:
        """Log activity coefficients ``ln(gamma_i)`` (cannot predict an LLE split)."""
        return wilson_ln_gamma(x, self.lam(t))


# --- Binary two-parameter models ----------------------------------------------


@dataclass(frozen=True)
class Margules:
    """Two-parameter Margules binary model with ``A = a + b/T``.

    Attributes:
        a12: Constant part of ``A12``; ``b12``: its ``1/T`` part (K).
        a21: Constant part of ``A21``; ``b21``: its ``1/T`` part (K).
    """

    a12: Array
    b12: Array
    a21: Array
    b21: Array

    def ln_gamma(self, x: Array, t: ArrayLike) -> Array:
        """Log activity coefficients ``[ln(gamma_1), ln(gamma_2)]`` for the binary."""
        x = jnp.asarray(x)
        t = jnp.asarray(t)
        a12 = self.a12 + self.b12 / t
        a21 = self.a21 + self.b21 / t
        lng1, lng2 = margules_ln_gamma(x[0], a12, a21)
        return jnp.stack([lng1, lng2])


@dataclass(frozen=True)
class VanLaar:
    """Two-parameter van Laar binary model with ``A = a + b/T``.

    Attributes:
        a12: Constant part of ``A12``; ``b12``: its ``1/T`` part (K).
        a21: Constant part of ``A21``; ``b21``: its ``1/T`` part (K).
    """

    a12: Array
    b12: Array
    a21: Array
    b21: Array

    def ln_gamma(self, x: Array, t: ArrayLike) -> Array:
        """Log activity coefficients ``[ln(gamma_1), ln(gamma_2)]`` for the binary."""
        x = jnp.asarray(x)
        t = jnp.asarray(t)
        a12 = self.a12 + self.b12 / t
        a21 = self.a21 + self.b21 / t
        lng1, lng2 = van_laar_ln_gamma(x[0], a12, a21)
        return jnp.stack([lng1, lng2])


# --- Pure-component-descriptor models -----------------------------------------


@dataclass(frozen=True)
class RegularSolution:
    """Scatchard-Hildebrand regular-solution model (cohesive-energy descriptors).

    Attributes:
        volume: Liquid molar volumes ``v_i`` (m^3/mol), shape ``(n,)``.
        delta: Solubility parameters ``delta_i`` (Pa**0.5), shape ``(n,)``.
    """

    volume: Array
    delta: Array

    def ln_gamma(self, x: Array, t: ArrayLike) -> Array:
        """Log activity coefficients ``ln(gamma_i)`` (all non-negative)."""
        return regular_solution_ln_gamma(x, self.volume, self.delta, t)


@dataclass(frozen=True)
class FloryHuggins:
    """Athermal Flory-Huggins size-asymmetry model.

    Attributes:
        volume: Molecular size descriptors ``v_i`` (only ratios matter), shape ``(n,)``.
    """

    volume: Array

    def ln_gamma(self, x: Array, t: ArrayLike) -> Array:
        """Log activity coefficients ``ln(gamma_i)`` (the combinatorial part)."""
        return flory_huggins_ln_gamma(x, self.volume)


@dataclass(frozen=True)
class Hildebrand:
    """Flory-Huggins-Hildebrand model: regular-solution enthalpy + FH entropy.

    Attributes:
        volume: Liquid molar volumes ``v_i`` (m^3/mol), shape ``(n,)``.
        delta: Solubility parameters ``delta_i`` (Pa**0.5), shape ``(n,)``.
    """

    volume: Array
    delta: Array

    def ln_gamma(self, x: Array, t: ArrayLike) -> Array:
        """Log activity coefficients ``ln(gamma_i)``."""
        return hildebrand_ln_gamma(x, self.volume, self.delta, t)


for _cls, _fields in (
    (NRTL, ["a", "b", "alpha", "e"]),
    (UNIQUAC, ["r", "q", "a", "b"]),
    (Wilson, ["volume", "energy"]),
    (Margules, ["a12", "b12", "a21", "b21"]),
    (VanLaar, ["a12", "b12", "a21", "b21"]),
    (RegularSolution, ["volume", "delta"]),
    (FloryHuggins, ["volume"]),
    (Hildebrand, ["volume", "delta"]),
):
    jax.tree_util.register_dataclass(_cls, data_fields=_fields, meta_fields=[])


# --- Ergonomic constructors ----------------------------------------------------


def nrtl(a: Array, b: Array, alpha: Array, e: Array | None = None) -> NRTL:
    """Build an `NRTL` model, defaulting the ``ln T`` term ``e`` to zero."""
    a = jnp.asarray(a, dtype=float)
    return NRTL(
        a=a,
        b=jnp.asarray(b, dtype=float),
        alpha=jnp.asarray(alpha, dtype=float),
        e=jnp.zeros_like(a) if e is None else jnp.asarray(e, dtype=float),
    )


def nrtl_from_energies(dg: Array, alpha: Array) -> NRTL:
    """Build a temperature-independent `NRTL` from energies ``dg_ij`` (J/mol).

    Uses ``tau_ij = dg_ij / (R T)`` (i.e. ``a = 0``, ``b = dg / R``), the common
    "g_ij - g_jj" parameterisation.
    """
    dg = jnp.asarray(dg, dtype=float)
    zeros = jnp.zeros_like(dg)
    return NRTL(a=zeros, b=dg / R, alpha=jnp.asarray(alpha, dtype=float), e=zeros)


def uniquac(r: Array, q: Array, a: Array, b: Array) -> UNIQUAC:
    """Build a `UNIQUAC` model with ``tau_ij = exp(a_ij + b_ij/T)``."""
    return UNIQUAC(
        r=jnp.asarray(r, dtype=float),
        q=jnp.asarray(q, dtype=float),
        a=jnp.asarray(a, dtype=float),
        b=jnp.asarray(b, dtype=float),
    )


def uniquac_from_energies(r: Array, q: Array, du: Array) -> UNIQUAC:
    """Build a temperature-dependent `UNIQUAC` from energies ``du_ij`` (J/mol).

    Uses ``tau_ij = exp(-du_ij / (R T))`` (``a = 0``, ``b = -du / R``).
    """
    du = jnp.asarray(du, dtype=float)
    return UNIQUAC(
        r=jnp.asarray(r, dtype=float),
        q=jnp.asarray(q, dtype=float),
        a=jnp.zeros_like(du),
        b=-du / R,
    )


def margules(
    a12: ArrayLike, a21: ArrayLike, b12: ArrayLike = 0.0, b21: ArrayLike = 0.0
) -> Margules:
    """Build a `Margules` binary model (constant ``A`` unless ``b`` given)."""
    return Margules(
        a12=jnp.asarray(a12, dtype=float),
        b12=jnp.asarray(b12, dtype=float),
        a21=jnp.asarray(a21, dtype=float),
        b21=jnp.asarray(b21, dtype=float),
    )


def van_laar(a12: ArrayLike, a21: ArrayLike, b12: ArrayLike = 0.0, b21: ArrayLike = 0.0) -> VanLaar:
    """Build a `VanLaar` binary model (constant ``A`` unless ``b`` given)."""
    return VanLaar(
        a12=jnp.asarray(a12, dtype=float),
        b12=jnp.asarray(b12, dtype=float),
        a21=jnp.asarray(a21, dtype=float),
        b21=jnp.asarray(b21, dtype=float),
    )
