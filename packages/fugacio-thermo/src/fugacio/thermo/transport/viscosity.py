"""Viscosity: dilute-gas and liquid, pure components and mixtures.

Pure components come from two routes, dispatched per component:

* **curated fits** -- DIPPR-101 (liquid) and DIPPR-102 (dilute gas) coefficients
  transcribed/refitted from open data into
  `fugacio.thermo._property_data`;
* **corresponding states** -- the Chung et al. dilute-gas method
  (`chung_viscosity_gas`, needing only ``Tc``, ``Vc``, ``omega``, ``MW``
  and the dipole moment) and Letsou-Stiel for liquids
  (`letsou_stiel_viscosity`), used when no fit is tabulated.

Mixtures use the standard kinetic-theory combination rules: Wilke's
approximation for gases (`wilke_mixture_viscosity`) and the
Grunberg-Nissan logarithmic rule with zero interaction parameters for liquids
(`grunberg_nissan_viscosity`). Everything is `jax.numpy`, so
viscosities are differentiable in temperature and composition -- and the oracle
suite grades both routes against CoolProp's reference correlations.

All viscosities are in Pa*s.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from fugacio.thermo._property_data import MU_GAS_DIPPR102, MU_LIQUID_DIPPR101
from fugacio.thermo.components import Component, get
from fugacio.thermo.correlations import dippr101, dippr102

ArrayLike = Array | float


# --- Corresponding-states estimators ----------------------------------------------


def neufeld_collision_integral(t_star: ArrayLike) -> Array:
    """Neufeld et al. fit of the Lennard-Jones collision integral ``Omega_v``.

    ``Omega_v = A*T*^-B + C*exp(-D*T*) + E*exp(-F*T*)`` with the 1972 constants;
    accurate to ~0.06% over ``0.3 <= T* <= 100``.
    """
    t_star = jnp.asarray(t_star)
    return (
        1.16145 * t_star**-0.14874
        + 0.52487 * jnp.exp(-0.77320 * t_star)
        + 2.16178 * jnp.exp(-2.43787 * t_star)
    )


def reduced_dipole(dipole: ArrayLike, vc: ArrayLike, tc: ArrayLike) -> Array:
    """Chung's dimensionless dipole ``mu_r = 131.3 * mu / sqrt(Vc * Tc)``.

    ``dipole`` in debye, ``vc`` in m^3/mol (converted internally to cm^3/mol),
    ``tc`` in K.
    """
    vc_cm3 = jnp.asarray(vc) * 1.0e6
    return 131.3 * jnp.asarray(dipole) / jnp.sqrt(vc_cm3 * jnp.asarray(tc))


def chung_viscosity_gas(
    t: ArrayLike,
    tc: ArrayLike,
    vc: ArrayLike,
    omega: ArrayLike,
    mw: ArrayLike,
    dipole: ArrayLike = 0.0,
    kappa: ArrayLike = 0.0,
) -> Array:
    """Chung et al. dilute-gas viscosity (Pa*s).

    ``eta = 40.785 * Fc * sqrt(MW*T) / (Vc^(2/3) * Omega_v)`` in micropoise, with
    ``Fc = 1 - 0.2756*omega + 0.059035*mu_r^4 + kappa`` (Chung, Lee & Starling
    1984; Poling et al., 5th ed., eq. 9-4.10). ``mw`` in g/mol, ``vc`` in
    m^3/mol, ``dipole`` in debye; ``kappa`` is the association factor (0.076 for
    water, 0 for normal fluids).
    """
    t_star = 1.2593 * jnp.asarray(t) / jnp.asarray(tc)
    omega_v = neufeld_collision_integral(t_star)
    mu_r = reduced_dipole(dipole, vc, tc)
    fc = 1.0 - 0.2756 * jnp.asarray(omega) + 0.059035 * mu_r**4 + jnp.asarray(kappa)
    vc_cm3 = jnp.asarray(vc) * 1.0e6
    eta_micropoise = (
        40.785 * fc * jnp.sqrt(jnp.asarray(mw) * jnp.asarray(t)) / (vc_cm3 ** (2.0 / 3.0) * omega_v)
    )
    return eta_micropoise * 1.0e-7


def letsou_stiel_viscosity(
    t: ArrayLike, tc: ArrayLike, pc: ArrayLike, omega: ArrayLike, mw: ArrayLike
) -> Array:
    """Letsou-Stiel high-temperature liquid viscosity (Pa*s).

    ``eta * xi = (1.5174 - 2.135*Tr + 0.75*Tr^2)*1e-5
    + omega*(4.2552 - 7.674*Tr + 3.4*Tr^2)*1e-5`` with the inverse-viscosity
    group ``xi = 2173.424 * Tc^(1/6) / (sqrt(MW) * Pc^(2/3))`` (SI units).
    Nominal validity ``0.76 <= Tr <= 0.98``; Fugacio also uses it as the
    last-resort fallback at lower ``Tr``, clipped to stay positive.
    """
    tr = jnp.asarray(t) / jnp.asarray(tc)
    xi = (
        2173.424
        * jnp.asarray(tc) ** (1.0 / 6.0)
        / (jnp.sqrt(jnp.asarray(mw)) * jnp.asarray(pc) ** (2.0 / 3.0))
    )
    xi0 = (1.5174 - 2.135 * tr + 0.75 * tr**2) * 1.0e-5
    xi1 = (4.2552 - 7.674 * tr + 3.4 * tr**2) * 1.0e-5
    eta = (xi0 + jnp.asarray(omega) * xi1) / xi
    return jnp.maximum(eta, 1.0e-7)


# --- Mixture combination rules ----------------------------------------------------


def wilke_mixture_viscosity(y: Array, mu: Array, mw: Array) -> Array:
    """Wilke's kinetic-theory rule for the viscosity of a gas mixture (Pa*s).

    ``eta_m = sum_i y_i eta_i / sum_j y_j phi_ij`` with::

        phi_ij = [1 + (eta_i/eta_j)^(1/2) (M_j/M_i)^(1/4)]^2
                 / [8 (1 + M_i/M_j)]^(1/2)

    (Poling et al., 5th ed., eq. 9-5.13/14). Exact in the pure limits.
    """
    y = jnp.asarray(y)
    mu = jnp.asarray(mu)
    mw = jnp.asarray(mw)
    ratio_mu = mu[:, None] / mu[None, :]
    ratio_mw = mw[None, :] / mw[:, None]
    phi = (1.0 + jnp.sqrt(ratio_mu) * ratio_mw**0.25) ** 2 / jnp.sqrt(8.0 * (1.0 + 1.0 / ratio_mw))
    denom = phi @ y
    return jnp.sum(y * mu / denom)


def grunberg_nissan_viscosity(x: Array, mu: Array, g: Array | None = None) -> Array:
    """Grunberg-Nissan rule for the viscosity of a liquid mixture (Pa*s).

    ``ln eta_m = sum_i x_i ln eta_i + (1/2) sum_i sum_j x_i x_j G_ij`` -- the
    interaction matrix ``G`` defaults to zero (ideal logarithmic mixing), which
    is the standard engineering assumption when no binary data exist.
    """
    x = jnp.asarray(x)
    mu = jnp.asarray(mu)
    ln_mu = jnp.sum(x * jnp.log(mu))
    if g is not None:
        ln_mu = ln_mu + 0.5 * jnp.sum(x[:, None] * x[None, :] * jnp.asarray(g))
    return jnp.exp(ln_mu)


# --- Name-based dispatchers -------------------------------------------------------


def _resolve(components: list[str] | list[Component]) -> list[Component]:
    return [get(c) if isinstance(c, str) else c for c in components]


#: Chung association factors ``kappa`` for strongly hydrogen-bonded species
#: (Chung et al. 1988, Table I). Everything else defaults to zero.
CHUNG_KAPPA: dict[str, float] = {
    "water": 0.075908,
    "methanol": 0.215175,
    "ethanol": 0.174823,
    "1-propanol": 0.143453,
    "2-propanol": 0.143453,
    "1-butanol": 0.131671,
    "1-pentanol": 0.121555,
    "1-hexanol": 0.114230,
    "1-heptanol": 0.108674,
    "acetic acid": 0.091549,
}


def _chung_pure(comp: Component, t: ArrayLike) -> Array:
    if comp.vc is None:
        raise ValueError(f"component {comp.name!r} lacks Vc needed for the Chung estimate")
    dipole = comp.dipole if comp.dipole is not None else 0.0
    kappa = CHUNG_KAPPA.get(comp.name, 0.0)
    return chung_viscosity_gas(t, comp.tc, comp.vc, comp.omega, comp.mw, dipole, kappa)


def gas_viscosities(components: list[str] | list[Component], t: ArrayLike) -> Array:
    """Per-component dilute-gas viscosities ``eta_i(T)`` (Pa*s).

    Curated DIPPR-102 fit where available, Chung corresponding states otherwise.
    """
    values: list[Array] = []
    for comp in _resolve(components):
        fit = MU_GAS_DIPPR102.get(comp.name)
        if fit is not None:
            c1, c2, c3, c4, _tmin, _tmax = fit
            values.append(dippr102(t, c1, c2, c3, c4))
        else:
            values.append(_chung_pure(comp, t))
    return jnp.stack(values)


def liquid_viscosities(components: list[str] | list[Component], t: ArrayLike) -> Array:
    """Per-component saturated-liquid viscosities ``eta_i(T)`` (Pa*s).

    Curated DIPPR-101 fit where available, Letsou-Stiel otherwise.
    """
    values: list[Array] = []
    for comp in _resolve(components):
        fit = MU_LIQUID_DIPPR101.get(comp.name)
        if fit is not None:
            c1, c2, c3, c4, c5, _tmin, _tmax = fit
            values.append(dippr101(t, c1, c2, c3, c4, c5))
        else:
            values.append(letsou_stiel_viscosity(t, comp.tc, comp.pc, comp.omega, comp.mw))
    return jnp.stack(values)


def gas_mixture_viscosity(components: list[str] | list[Component], t: ArrayLike, y: Array) -> Array:
    """Dilute-gas mixture viscosity by Wilke's rule (Pa*s)."""
    resolved = _resolve(components)
    mu = gas_viscosities(resolved, t)
    mw = jnp.asarray([c.mw for c in resolved])
    return wilke_mixture_viscosity(jnp.asarray(y), mu, mw)


def liquid_mixture_viscosity(
    components: list[str] | list[Component], t: ArrayLike, x: Array
) -> Array:
    """Liquid mixture viscosity by the Grunberg-Nissan rule (Pa*s)."""
    mu = liquid_viscosities(components, t)
    return grunberg_nissan_viscosity(jnp.asarray(x), mu)
