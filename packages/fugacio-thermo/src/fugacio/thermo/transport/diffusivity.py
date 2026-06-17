"""Binary diffusion coefficients: Fuller (gas) and Wilke-Chang (liquid).

Both are the standard engineering estimators (Poling et al., 5th ed., ch. 11):

* `fuller_diffusivity`: gas-phase ``D_AB`` from atomic diffusion volumes
  (``D ~ T^1.75 / P``), good to ~5-10%;
* `wilke_chang_diffusivity`: infinite-dilution liquid ``D_AB`` from the
  solvent viscosity and the solute molar volume at its normal boiling point,
  good to ~10-20%.

The name-based wrappers (`gas_diffusivity`, `liquid_diffusivity`)
assemble the inputs from the component database: diffusion volumes from the
Fuller atomic contributions (with the special-molecule table for the common
gases), boiling-point volumes from the Tyn-Calus estimate, and the solvent
viscosity from `fugacio.thermo.transport.viscosity`.

All diffusivities are in m^2/s.
"""

from __future__ import annotations

import re

import jax.numpy as jnp
from jax import Array

from fugacio.thermo.components import Component, get
from fugacio.thermo.transport.viscosity import liquid_viscosities
from fugacio.thermo.volumetric import liquid_molar_volumes, tyn_calus_vb

ArrayLike = Array | float

#: Fuller atomic diffusion-volume increments (Fuller, Ensley & Giddings 1969).
FULLER_ATOMIC: dict[str, float] = {
    "C": 15.9,
    "H": 2.31,
    "O": 6.11,
    "N": 4.54,
    "F": 14.7,
    "Cl": 21.0,
    "Br": 21.9,
    "I": 29.8,
    "S": 22.9,
}

#: Fuller diffusion volumes for simple molecules, overriding the atomic sums.
FULLER_MOLECULES: dict[str, float] = {
    "helium": 2.67,
    "neon": 5.98,
    "argon": 16.2,
    "krypton": 24.5,
    "xenon": 32.7,
    "hydrogen": 6.12,
    "nitrogen": 18.5,
    "oxygen": 16.3,
    "carbon monoxide": 18.0,
    "carbon dioxide": 26.9,
    "nitrous oxide": 35.9,
    "ammonia": 20.7,
    "water": 13.1,
    "sulfur dioxide": 41.8,
    "chlorine": 38.4,
    "sulfur hexafluoride": 71.3,
}

#: Wilke-Chang solvent association factors ``phi``.
WILKE_CHANG_PHI: dict[str, float] = {
    "water": 2.6,
    "methanol": 1.9,
    "ethanol": 1.5,
}

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def diffusion_volume(component: str | Component) -> float:
    """Fuller diffusion volume of a component (dimensionless table units).

    Uses the special-molecule table when the species is listed there, otherwise
    sums the atomic increments over the molecular formula. Ring corrections are
    not applied (the formula alone does not reveal rings), which biases cyclic
    species' volumes slightly high.

    Raises:
        ValueError: if the formula contains an element without a Fuller increment.
    """
    comp = get(component) if isinstance(component, str) else component
    if comp.name in FULLER_MOLECULES:
        return FULLER_MOLECULES[comp.name]
    total = 0.0
    consumed = 0
    for symbol, number in _FORMULA_TOKEN.findall(comp.formula):
        count = int(number) if number else 1
        if symbol not in FULLER_ATOMIC:
            raise ValueError(
                f"no Fuller atomic diffusion volume for element {symbol!r} "
                f"in {comp.name!r} ({comp.formula})"
            )
        total += FULLER_ATOMIC[symbol] * count
        consumed += len(symbol) + len(number)
    if consumed != len(comp.formula):
        raise ValueError(f"cannot parse molecular formula {comp.formula!r}")
    return total


def fuller_diffusivity(
    t: ArrayLike,
    p: ArrayLike,
    mw_a: ArrayLike,
    mw_b: ArrayLike,
    v_a: ArrayLike,
    v_b: ArrayLike,
) -> Array:
    """Fuller-Schettler-Giddings binary gas diffusivity (m^2/s).

    ``D_AB = 0.00143 * T^1.75 / (P_bar * sqrt(M_AB) * (Vd_A^(1/3) + Vd_B^(1/3))^2)``
    in cm^2/s, with ``M_AB = 2 / (1/M_A + 1/M_B)`` (g/mol) and the tabulated
    diffusion volumes ``Vd``; converted to SI here (``p`` in Pa).
    """
    t = jnp.asarray(t)
    p_bar = jnp.asarray(p) * 1.0e-5
    m_ab = 2.0 / (1.0 / jnp.asarray(mw_a) + 1.0 / jnp.asarray(mw_b))
    denom = (
        p_bar
        * jnp.sqrt(m_ab)
        * (jnp.asarray(v_a) ** (1.0 / 3.0) + jnp.asarray(v_b) ** (1.0 / 3.0)) ** 2
    )
    return 0.00143 * t**1.75 / denom * 1.0e-4


def wilke_chang_diffusivity(
    t: ArrayLike,
    mu_solvent: ArrayLike,
    mw_solvent: ArrayLike,
    vb_solute: ArrayLike,
    phi: ArrayLike = 1.0,
) -> Array:
    """Wilke-Chang infinite-dilution liquid diffusivity (m^2/s).

    ``D_AB = 7.4e-8 * sqrt(phi*M_B) * T / (eta_B * V_A^0.6)`` in cm^2/s with the
    solvent viscosity ``eta_B`` in cP and the solute boiling-point molar volume
    ``V_A`` in cm^3/mol; SI in and out here (``mu_solvent`` in Pa*s,
    ``vb_solute`` in m^3/mol). ``phi`` is the solvent association factor (2.6
    water, 1.9 methanol, 1.5 ethanol, 1.0 otherwise).
    """
    mu_cp = jnp.asarray(mu_solvent) * 1.0e3
    vb_cm3 = jnp.asarray(vb_solute) * 1.0e6
    d_cm2 = (
        7.4e-8
        * jnp.sqrt(jnp.asarray(phi) * jnp.asarray(mw_solvent))
        * jnp.asarray(t)
        / (mu_cp * vb_cm3**0.6)
    )
    return d_cm2 * 1.0e-4


# --- Name-based dispatchers -------------------------------------------------------


def gas_diffusivity(
    component_a: str | Component, component_b: str | Component, t: ArrayLike, p: ArrayLike
) -> Array:
    """Binary gas-phase diffusion coefficient ``D_AB(T, P)`` by Fuller (m^2/s)."""
    a = get(component_a) if isinstance(component_a, str) else component_a
    b = get(component_b) if isinstance(component_b, str) else component_b
    return fuller_diffusivity(t, p, a.mw, b.mw, diffusion_volume(a), diffusion_volume(b))


def liquid_diffusivity(solute: str | Component, solvent: str | Component, t: ArrayLike) -> Array:
    """Infinite-dilution liquid diffusivity of ``solute`` in ``solvent`` (m^2/s).

    Wilke-Chang with the solvent viscosity from the curated correlations and the
    solute boiling-point volume from Tyn-Calus (needs the solute's ``vc``; falls
    back to the tabulated liquid molar volume at the system temperature when
    ``vc`` is missing).
    """
    sol = get(solute) if isinstance(solute, str) else solute
    sv = get(solvent) if isinstance(solvent, str) else solvent
    mu = liquid_viscosities([sv], t)[0]
    vb = tyn_calus_vb(sol.vc) if sol.vc is not None else liquid_molar_volumes([sol], t)[0]
    phi = WILKE_CHANG_PHI.get(sv.name, 1.0)
    return wilke_chang_diffusivity(t, mu, sv.mw, vb, phi)
