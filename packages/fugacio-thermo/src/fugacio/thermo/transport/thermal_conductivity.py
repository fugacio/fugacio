"""Thermal conductivity: dilute-gas and liquid, pure components and mixtures.

The same two-route dispatch as :mod:`fugacio.thermo.transport.viscosity`:

* **curated fits** -- DIPPR-102 (dilute gas) and DIPPR-100 (liquid) coefficients
  from :mod:`fugacio.thermo._property_data`;
* **corresponding states** -- the Chung et al. dilute-gas method
  (:func:`chung_thermal_conductivity_gas`, built on the Chung viscosity and the
  ideal-gas heat capacity) and Sato-Riedel for liquids
  (:func:`sato_riedel_thermal_conductivity`).

Mixtures use Wassiljewa's equation with the Mason-Saxena weights for gases
(:func:`wassiljewa_mixture`) and the DIPPR9H mass-fraction power law for liquids
(:func:`dippr9h_mixture`).

All conductivities are in W/m/K.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from fugacio.thermo._property_data import K_GAS_DIPPR102, K_LIQUID_DIPPR100
from fugacio.thermo.components import Component, get
from fugacio.thermo.constants import R
from fugacio.thermo.correlations import dippr100, dippr102
from fugacio.thermo.ideal import cp_ig
from fugacio.thermo.transport.viscosity import gas_viscosities

ArrayLike = Array | float


# --- Corresponding-states estimators ----------------------------------------------


def chung_thermal_conductivity_gas(
    t: ArrayLike,
    tc: ArrayLike,
    omega: ArrayLike,
    mw: ArrayLike,
    mu_gas: ArrayLike,
    cp_ideal: ArrayLike,
) -> Array:
    """Chung et al. dilute-gas thermal conductivity (W/m/K).

    ``k = 3.75 * Psi * eta * R / M`` with the internal-degrees-of-freedom factor::

        Psi = 1 + alpha * (0.215 + 0.28288*alpha - 1.061*beta + 0.26665*Z)
                      / (0.6366 + beta*Z + 1.061*alpha*beta)

    where ``alpha = Cv_ig/R - 3/2``, ``beta = 0.7862 - 0.7109*omega
    + 1.3168*omega^2`` and ``Z = 2.0 + 10.5*Tr^2`` (Chung et al. 1984; Poling et
    al., 5th ed., eq. 10-3.14). ``mu_gas`` is the dilute-gas viscosity (Pa*s),
    ``cp_ideal`` the ideal-gas heat capacity (J/mol/K), ``mw`` in g/mol.
    """
    tr = jnp.asarray(t) / jnp.asarray(tc)
    alpha = (jnp.asarray(cp_ideal) - R) / R - 1.5
    omega = jnp.asarray(omega)
    beta = 0.7862 - 0.7109 * omega + 1.3168 * omega**2
    z = 2.0 + 10.5 * tr**2
    psi = 1.0 + alpha * (
        (0.215 + 0.28288 * alpha - 1.061 * beta + 0.26665 * z)
        / (0.6366 + beta * z + 1.061 * alpha * beta)
    )
    m_kg = jnp.asarray(mw) * 1.0e-3
    return 3.75 * psi * jnp.asarray(mu_gas) * R / m_kg


def sato_riedel_thermal_conductivity(
    t: ArrayLike, tc: ArrayLike, tb: ArrayLike, mw: ArrayLike
) -> Array:
    """Sato-Riedel liquid thermal conductivity (W/m/K).

    ``k = (1.1053 / sqrt(MW)) * (3 + 20*(1-Tr)^(2/3)) / (3 + 20*(1-Tbr)^(2/3))``
    (Poling et al., 5th ed., eq. 10-9.2); a rough but robust estimate, typically
    within ~15-25% for organic liquids. ``mw`` in g/mol.
    """
    tc = jnp.asarray(tc)
    tr = jnp.clip(jnp.asarray(t) / tc, 0.0, 1.0)
    tbr = jnp.clip(jnp.asarray(tb) / tc, 0.0, 1.0)
    return (
        1.1053
        / jnp.sqrt(jnp.asarray(mw))
        * (3.0 + 20.0 * (1.0 - tr) ** (2.0 / 3.0))
        / (3.0 + 20.0 * (1.0 - tbr) ** (2.0 / 3.0))
    )


# --- Mixture combination rules ----------------------------------------------------


def wassiljewa_mixture(y: Array, k: Array, mu: Array, mw: Array) -> Array:
    """Wassiljewa gas-mixture conductivity with Mason-Saxena weights (W/m/K).

    ``k_m = sum_i y_i k_i / sum_j y_j A_ij`` where ``A_ij`` is the Wilke
    ``phi_ij`` evaluated from the dilute-gas *viscosities* (the Mason-Saxena
    recommendation with the proportionality constant at one).
    """
    y = jnp.asarray(y)
    k = jnp.asarray(k)
    mu = jnp.asarray(mu)
    mw = jnp.asarray(mw)
    ratio_mu = mu[:, None] / mu[None, :]
    ratio_mw = mw[None, :] / mw[:, None]
    a = (1.0 + jnp.sqrt(ratio_mu) * ratio_mw**0.25) ** 2 / jnp.sqrt(8.0 * (1.0 + 1.0 / ratio_mw))
    denom = a @ y
    return jnp.sum(y * k / denom)


def dippr9h_mixture(w: Array, k: Array) -> Array:
    """DIPPR9H (Li-style) liquid-mixture conductivity power law (W/m/K).

    ``k_m = (sum_i w_i * k_i^-2)^(-1/2)`` with *mass* fractions ``w`` -- the
    standard recommendation for nonaqueous liquid mixtures (Poling et al., 5th
    ed., eq. 10-12.4).
    """
    w = jnp.asarray(w)
    k = jnp.asarray(k)
    return jnp.sum(w / k**2) ** -0.5


# --- Name-based dispatchers -------------------------------------------------------


def _resolve(components: list[str] | list[Component]) -> list[Component]:
    return [get(c) if isinstance(c, str) else c for c in components]


def _chung_pure(comp: Component, t: ArrayLike, mu: Array) -> Array:
    if comp.cp_ig is None:
        raise ValueError(
            f"component {comp.name!r} lacks ideal-gas Cp needed for the Chung estimate"
        )
    cp = comp.cp_ig
    cp_id = cp_ig(t, cp.a, cp.b, cp.c, cp.d, cp.e)
    return chung_thermal_conductivity_gas(t, comp.tc, comp.omega, comp.mw, mu, cp_id)


def gas_thermal_conductivities(components: list[str] | list[Component], t: ArrayLike) -> Array:
    """Per-component dilute-gas thermal conductivities ``k_i(T)`` (W/m/K).

    Curated DIPPR-102 fit where available, Chung corresponding states otherwise.
    """
    resolved = _resolve(components)
    mu = gas_viscosities(resolved, t)
    values: list[Array] = []
    for i, comp in enumerate(resolved):
        fit = K_GAS_DIPPR102.get(comp.name)
        if fit is not None:
            c1, c2, c3, c4, _tmin, _tmax = fit
            values.append(dippr102(t, c1, c2, c3, c4))
        else:
            values.append(_chung_pure(comp, t, mu[i]))
    return jnp.stack(values)


def liquid_thermal_conductivities(components: list[str] | list[Component], t: ArrayLike) -> Array:
    """Per-component saturated-liquid thermal conductivities ``k_i(T)`` (W/m/K).

    Curated DIPPR-100 fit where available, Sato-Riedel otherwise (which needs a
    normal boiling point; components lacking ``tb`` raise).
    """
    values: list[Array] = []
    for comp in _resolve(components):
        fit = K_LIQUID_DIPPR100.get(comp.name)
        if fit is not None:
            c1, c2, c3, c4, c5, _tmin, _tmax = fit
            values.append(dippr100(t, c1, c2, c3, c4, c5))
        elif comp.tb is not None:
            values.append(sato_riedel_thermal_conductivity(t, comp.tc, comp.tb, comp.mw))
        else:
            raise ValueError(
                f"component {comp.name!r} has neither a liquid-conductivity fit nor a "
                f"boiling point for the Sato-Riedel estimate"
            )
    return jnp.stack(values)


def gas_mixture_thermal_conductivity(
    components: list[str] | list[Component], t: ArrayLike, y: Array
) -> Array:
    """Dilute-gas mixture conductivity by Wassiljewa / Mason-Saxena (W/m/K)."""
    resolved = _resolve(components)
    k = gas_thermal_conductivities(resolved, t)
    mu = gas_viscosities(resolved, t)
    mw = jnp.asarray([c.mw for c in resolved])
    return wassiljewa_mixture(jnp.asarray(y), k, mu, mw)


def liquid_mixture_thermal_conductivity(
    components: list[str] | list[Component], t: ArrayLike, x: Array
) -> Array:
    """Liquid mixture conductivity by the DIPPR9H mass-fraction rule (W/m/K)."""
    resolved = _resolve(components)
    k = liquid_thermal_conductivities(resolved, t)
    x = jnp.asarray(x)
    mw = jnp.asarray([c.mw for c in resolved])
    w = x * mw / jnp.sum(x * mw)
    return dippr9h_mixture(w, k)
