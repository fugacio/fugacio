"""Reaction stoichiometry and standard-state thermochemistry.

This module turns a chemical reaction into the temperature-dependent quantities
that govern chemical equilibrium:

* the standard enthalpy of reaction `delta_h_rxn` (``DH_rxn(T)``),
* the standard entropy of reaction `delta_s_rxn` (``DS_rxn(T)``),
* the standard Gibbs energy of reaction `delta_g_rxn` (``DG_rxn(T)``), and
* the thermodynamic equilibrium constant `equilibrium_constant`
  (``K(T) = exp(-DG_rxn / R T)``).

All four follow from the component standard formation properties
(`hform_ig` and ``gform_ig``, the
ideal-gas enthalpy and Gibbs energy of formation at 298.15 K) corrected to the
reaction temperature with Kirchhoff's law, integrating the ideal-gas heat
capacities (`fugacio.thermo.ideal.enthalpy_ig` / ``entropy_ig``)::

    DH_rxn(T) = sum_i nu_i [ Hf_i + integral_{T0}^{T} Cp_i dT ]
    DS_rxn(T) = DS_rxn(T0) + sum_i nu_i integral_{T0}^{T} (Cp_i / T) dT
    DG_rxn(T) = DH_rxn(T) - T DS_rxn(T)

with ``DS_rxn(T0) = (DH_rxn(T0) - DG_rxn(T0)) / T0``. Everything is written in
`jax.numpy`, so ``K(T)`` is differentiable in temperature *and* in the
underlying formation/heat-capacity parameters (handy for data regression and for
sensitivity of conversion to thermochemistry).

The standard state is the ideal gas at ``P_REF`` (1 bar), matching the tabulated
formation data; the equilibrium constant is therefore in terms of fugacities
referenced to 1 bar (see `fugacio.thermo.reaction_equilibrium`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.components import get
from fugacio.thermo.constants import P_REF, T_REF, R
from fugacio.thermo.ideal import enthalpy_ig, entropy_ig, ideal_gas_coeffs

ArrayLike = Array | float

CpCoeffs = tuple[Array, Array, Array, Array, Array]

_SIDE_SEP = re.compile(r"<=>|<->|⇌|⇄|→|->|=")
_TERM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)?\s*(.+?)\s*$")


def stoichiometry(
    components: Sequence[str],
    reactants: Mapping[str, float],
    products: Mapping[str, float],
) -> Array:
    """Build a stoichiometric vector aligned with ``components``.

    Reactant coefficients are stored negative, products positive. Coefficients are
    summed, so a species appearing on both sides nets out.

    Args:
        components: The ordered component names the vector is aligned to.
        reactants: ``{name: coefficient}`` consumed by the reaction (positive).
        products: ``{name: coefficient}`` produced by the reaction (positive).

    Returns:
        The stoichiometric coefficient array ``nu`` of shape ``(len(components),)``.
    """
    index = {name: i for i, name in enumerate(components)}
    nu = [0.0] * len(components)
    for name, coef in reactants.items():
        nu[index[name]] -= float(coef)
    for name, coef in products.items():
        nu[index[name]] += float(coef)
    return jnp.asarray(nu)


def parse_reaction(equation: str, components: Sequence[str]) -> Array:
    """Parse a reaction string such as ``"nitrogen + 3 hydrogen = 2 ammonia"``.

    Sides are separated by ``=`` (also ``->``, ``<=>``, ``⇌``); terms by ``+``.
    Each term is an optional numeric coefficient followed by a *component name*
    exactly as it appears in ``components`` (names may contain spaces). Matching is
    case-insensitive.

    Returns:
        The stoichiometric vector aligned with ``components``.

    Raises:
        ValueError: if the string has no side separator or names an unknown species.
    """
    parts = _SIDE_SEP.split(equation)
    if len(parts) != 2:
        raise ValueError(f"reaction must have exactly one side separator: {equation!r}")
    lookup = {name.lower(): name for name in components}

    def side(text: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for raw in text.split("+"):
            if not raw.strip():
                continue
            m = _TERM_RE.match(raw)
            if m is None:  # pragma: no cover - regex always matches non-empty input
                raise ValueError(f"cannot parse reaction term {raw!r}")
            coef = float(m.group(1)) if m.group(1) else 1.0
            name = m.group(2).strip().lower()
            if name not in lookup:
                raise ValueError(f"unknown species {m.group(2)!r} in reaction {equation!r}")
            out[lookup[name]] = out.get(lookup[name], 0.0) + coef
        return out

    return stoichiometry(components, side(parts[0]), side(parts[1]))


@dataclass(frozen=True)
class Reaction:
    """A chemical reaction: a stoichiometric vector over an ordered component list.

    Attributes:
        components: Ordered component names (the alignment of ``nu``).
        nu: Stoichiometric coefficients (negative reactants, positive products).
    """

    components: tuple[str, ...]
    nu: Array

    @staticmethod
    def of(
        components: Sequence[str],
        reactants: Mapping[str, float],
        products: Mapping[str, float],
    ) -> Reaction:
        """Build a reaction from reactant/product coefficient maps."""
        return Reaction(tuple(components), stoichiometry(components, reactants, products))

    @staticmethod
    def parse(equation: str, components: Sequence[str]) -> Reaction:
        """Build a reaction from an equation string (see `parse_reaction`)."""
        return Reaction(tuple(components), parse_reaction(equation, components))

    @property
    def reactants(self) -> dict[str, float]:
        """``{name: coefficient}`` for the consumed species (positive coefficients)."""
        return {n: -float(v) for n, v in zip(self.components, self.nu, strict=True) if float(v) < 0}

    @property
    def products(self) -> dict[str, float]:
        """``{name: coefficient}`` for the produced species (positive coefficients)."""
        return {n: float(v) for n, v in zip(self.components, self.nu, strict=True) if float(v) > 0}

    @property
    def delta_n(self) -> float:
        """Change in moles ``sum(nu)`` (mole change of the gas phase per extent)."""
        return float(jnp.sum(self.nu))


jax.tree_util.register_dataclass(Reaction, data_fields=["nu"], meta_fields=["components"])


class ReactionProperties(NamedTuple):
    """Standard-state reaction properties at a temperature.

    Attributes:
        delta_h: Standard enthalpy of reaction ``DH_rxn(T)`` (J/mol).
        delta_s: Standard entropy of reaction ``DS_rxn(T)`` (J/mol/K).
        delta_g: Standard Gibbs energy of reaction ``DG_rxn(T)`` (J/mol).
        ln_k: Natural log of the equilibrium constant.
        k: Equilibrium constant ``exp(-DG_rxn / R T)`` (dimensionless).
    """

    delta_h: Array
    delta_s: Array
    delta_g: Array
    ln_k: Array
    k: Array


def _require(value: float | None, name: str, field: str) -> float:
    if value is None:
        raise ValueError(f"component {name!r} has no {field}; reaction thermochemistry needs it")
    return value


def reaction_arrays(components: Sequence[str]) -> tuple[Array, Array, CpCoeffs]:
    """Standard formation data for ``components`` from the curated database.

    Returns:
        ``(hf, gf, (a, b, c, d, e))`` -- ideal-gas formation enthalpies and Gibbs
        energies (J/mol at 298.15 K) and the stacked ideal-gas ``Cp`` coefficients.

    Raises:
        ValueError: if any component lacks formation or heat-capacity data.
    """
    comps = [get(name) for name in components]
    hf = jnp.asarray(
        [_require(c.hform_ig, c.name, "enthalpy of formation (hform_ig)") for c in comps]
    )
    gf = jnp.asarray(
        [_require(c.gform_ig, c.name, "Gibbs energy of formation (gform_ig)") for c in comps]
    )
    return hf, gf, ideal_gas_coeffs(comps)


def delta_h_rxn(
    nu: Array, t: ArrayLike, hf: Array, a: Array, b: Array, c: Array, d: Array, e: Array
) -> Array:
    """Standard enthalpy of reaction ``DH_rxn(T)`` (J/mol), via Kirchhoff's law."""
    h_t = hf + enthalpy_ig(t, a, b, c, d, e)
    return jnp.sum(nu * h_t)


def delta_s_rxn(
    nu: Array,
    t: ArrayLike,
    hf: Array,
    gf: Array,
    a: Array,
    b: Array,
    c: Array,
    d: Array,
    e: Array,
) -> Array:
    """Standard entropy of reaction ``DS_rxn(T)`` (J/mol/K)."""
    ds0 = (jnp.sum(nu * hf) - jnp.sum(nu * gf)) / T_REF
    # entropy_ig at p = P_REF is exactly the temperature integral of Cp/T.
    ds_t = jnp.sum(nu * entropy_ig(t, P_REF, a, b, c, d, e))
    return ds0 + ds_t


def delta_g_rxn(
    nu: Array,
    t: ArrayLike,
    hf: Array,
    gf: Array,
    a: Array,
    b: Array,
    c: Array,
    d: Array,
    e: Array,
) -> Array:
    """Standard Gibbs energy of reaction ``DG_rxn(T) = DH_rxn - T DS_rxn`` (J/mol)."""
    dh = delta_h_rxn(nu, t, hf, a, b, c, d, e)
    ds = delta_s_rxn(nu, t, hf, gf, a, b, c, d, e)
    return dh - jnp.asarray(t) * ds


def equilibrium_constant(
    nu: Array,
    t: ArrayLike,
    hf: Array,
    gf: Array,
    a: Array,
    b: Array,
    c: Array,
    d: Array,
    e: Array,
) -> Array:
    """Thermodynamic equilibrium constant ``K(T) = exp(-DG_rxn / R T)``."""
    dg = delta_g_rxn(nu, t, hf, gf, a, b, c, d, e)
    return jnp.exp(-dg / (R * jnp.asarray(t)))


def reaction_properties(reaction: Reaction, t: ArrayLike) -> ReactionProperties:
    """All standard reaction properties at ``t`` for a `Reaction`.

    Resolves the component formation/heat-capacity data from the curated database
    and evaluates `delta_h_rxn`, `delta_s_rxn`, `delta_g_rxn`,
    and `equilibrium_constant`.
    """
    hf, gf, (a, b, c, d, e) = reaction_arrays(reaction.components)
    nu = reaction.nu
    dh = delta_h_rxn(nu, t, hf, a, b, c, d, e)
    ds = delta_s_rxn(nu, t, hf, gf, a, b, c, d, e)
    dg = dh - jnp.asarray(t) * ds
    ln_k = -dg / (R * jnp.asarray(t))
    return ReactionProperties(delta_h=dh, delta_s=ds, delta_g=dg, ln_k=ln_k, k=jnp.exp(ln_k))


def equilibrium_constant_of(reaction: Reaction, t: ArrayLike) -> Array:
    """Equilibrium constant ``K(T)`` for a `Reaction` (database-resolved)."""
    return reaction_properties(reaction, t).k


__all__ = [
    "CpCoeffs",
    "Reaction",
    "ReactionProperties",
    "delta_g_rxn",
    "delta_h_rxn",
    "delta_s_rxn",
    "equilibrium_constant",
    "equilibrium_constant_of",
    "parse_reaction",
    "reaction_arrays",
    "reaction_properties",
    "stoichiometry",
]
