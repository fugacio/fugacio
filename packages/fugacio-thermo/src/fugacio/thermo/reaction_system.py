"""Validated reaction sets with explicit kinetic and thermodynamic conventions.

The chemical standard state is an ideal gas at one bar. Package enthalpies
are sensible/residual quantities relative to that same ideal-gas datum; adding
formation enthalpies once gives the energy carried by a reacting stream.
Kinetics may consume concentrations or dimensionless fugacity activities.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from fugacio.thermo.components import get
from fugacio.thermo.constants import P_REF, R
from fugacio.thermo.kinetics import _pow, arrhenius_ref
from fugacio.thermo.package import HelmholtzPackage, PropertyPackage
from fugacio.thermo.reactions import CpCoeffs, Reaction, delta_g_rxn, reaction_arrays


def element_matrix(components: Sequence[str]) -> tuple[tuple[str, ...], list[list[float]]]:
    """Return elemental incidence, rejecting incomplete or ambiguous formulas."""
    compositions = []
    for name in components:
        formula = get(name).formula
        tokens = re.findall(r"[A-Z][a-z]?|\d+|[()]", formula)
        if not tokens or "".join(tokens) != formula:
            raise ValueError(f"unsupported component formula {formula!r}")
        stack: list[dict[str, float]] = [defaultdict(float)]
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "(":
                stack.append(defaultdict(float))
                i += 1
                continue
            if token == ")" and len(stack) > 1:
                group = stack.pop()
            elif token[0].isalpha():
                group = {token: 1.0}
            else:
                raise ValueError(f"unsupported component formula {formula!r}")
            multiplier = 1
            if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                multiplier = int(tokens[i + 1])
                i += 1
            for element, count in group.items():
                stack[-1][element] += multiplier * count
            i += 1
        if len(stack) != 1:
            raise ValueError(f"unbalanced component formula {formula!r}")
        compositions.append(stack[0])
    elements = tuple(sorted({key for item in compositions for key in item}))
    return elements, [[item.get(key, 0.0) for item in compositions] for key in elements]


@dataclass(frozen=True)
class ReferenceRate:
    """A dimensionally explicit power law with an optional reverse reaction.

    ``k_forward`` and ``k_reverse`` are rates in mol/(m^3 s) at
    ``reference_temperature`` for unit dimensionless inputs. Concentrations
    are divided by the reaction set's declared reference concentration first.
    Independent reverse coefficients are empirical kinetics and do not imply
    thermodynamic detailed balance. Use ``detailed_balance=True`` with
    fugacity activities to derive the reverse rate from thermochemistry.
    """

    k_forward: Array
    ea_forward: Array
    forward_orders: Array
    k_reverse: Array
    ea_reverse: Array
    reverse_orders: Array
    reference_temperature: Array
    detailed_balance: bool = False

    def rate(self, t: Array, inputs: Array, ln_k: Array | None = None) -> Array:
        """Evaluate the volumetric rate from dimensionless kinetic inputs."""
        kf = arrhenius_ref(t, self.k_forward, self.ea_forward, self.reference_temperature)
        if self.detailed_balance:
            if ln_k is None:
                raise ValueError("detailed balance requires a thermodynamic equilibrium constant")
            kr = kf * jnp.exp(-ln_k)
        else:
            kr = arrhenius_ref(t, self.k_reverse, self.ea_reverse, self.reference_temperature)
        return kf * _pow(inputs, self.forward_orders) - kr * _pow(inputs, self.reverse_orders)


jax.tree_util.register_dataclass(
    ReferenceRate,
    data_fields=[
        "k_forward",
        "ea_forward",
        "forward_orders",
        "k_reverse",
        "ea_reverse",
        "reverse_orders",
        "reference_temperature",
    ],
    meta_fields=["detailed_balance"],
)


@dataclass(frozen=True)
class ReactionSet:
    """A reusable, differentiable reaction set on an ordered component basis.

    Construct with :meth:`from_reactions`, which verifies element conservation
    and independent stoichiometries. ``phase`` selects the reacting liquid or
    vapor. ``rate_basis`` is ``concentration`` (mol/m^3),
    ``normalized_concentration`` (c/c_ref), or ``activity`` (f/P_REF).
    All rates are per reacting-phase volume, in mol/(m^3 s).
    Thermochemical arrays and rate coefficients are differentiable leaves.
    """

    components: tuple[str, ...]
    names: tuple[str, ...]
    nu: Array
    formation_enthalpy: Array
    formation_gibbs: Array
    cp: CpCoeffs
    atoms: Array
    rate_laws: tuple[Any, ...]
    reference_concentration: Array
    phase: str = "vapor"
    rate_basis: str = "concentration"

    @classmethod
    def from_reactions(
        cls,
        reactions: Reaction | Sequence[Reaction],
        rate_laws: Any = (),
        *,
        phase: str = "vapor",
        rate_basis: str = "concentration",
        reference_concentration: float | Array = 1.0,
        names: Sequence[str] | None = None,
    ) -> ReactionSet:
        """Validate reactions and load formation/Cp data without fitting kinetics.

        Arbitrary Python rate objects need a JAX pytree registration when their
        coefficients are differentiated. Bundled kinetic laws already have one.
        ReferenceRate accepts only dimensionless inputs.
        """
        rxns = (reactions,) if isinstance(reactions, Reaction) else tuple(reactions)
        if not rxns:
            raise ValueError("a reaction set must contain reactions")
        components = rxns[0].components
        if any(r.components != components for r in rxns):
            raise ValueError("reactions must share the same ordered components")
        if phase not in ("liquid", "vapor"):
            raise ValueError("reaction phase must be liquid or vapor")
        if rate_basis not in ("concentration", "normalized_concentration", "activity"):
            raise ValueError("unknown reaction rate basis")
        laws = tuple(rate_laws) if isinstance(rate_laws, (tuple, list)) else (rate_laws,)
        if laws and len(laws) != len(rxns):
            raise ValueError("provide one rate law for each reaction")
        labels = (
            tuple(names)
            if names is not None
            else tuple(f"reaction_{i + 1}" for i in range(len(rxns)))
        )
        if len(labels) != len(rxns) or len(set(labels)) != len(labels):
            raise ValueError("reaction names must be unique and match the reactions")
        nu = jnp.stack([jnp.asarray(r.nu, dtype=float) for r in rxns])
        if nu.shape != (len(rxns), len(components)):
            raise ValueError("stoichiometric shape doesn't match components")
        _, atom_rows = element_matrix(components)
        atoms = jnp.asarray(atom_rows)
        if not isinstance(nu, jax.core.Tracer):
            values = np.asarray(nu)
            if (
                not np.all(np.isfinite(values))
                or np.any(np.max(values, axis=1) <= 0)
                or np.any(np.min(values, axis=1) >= 0)
            ):
                raise ValueError("reactions need finite reactant and product coefficients")
            if np.max(np.abs(np.asarray(atom_rows) @ values.T)) > 1e-9:
                raise ValueError("reactions must conserve every element")
            if np.linalg.matrix_rank(values) < len(rxns):
                raise ValueError("reaction stoichiometries must be independent")
        for i, law in enumerate(laws):
            if isinstance(law, ReferenceRate):
                for field in ("forward_orders", "reverse_orders"):
                    orders = jnp.asarray(getattr(law, field))
                    if orders.shape != (len(components),):
                        raise ValueError("kinetic orders must match the component basis")
                    if not isinstance(orders, jax.core.Tracer) and (
                        not np.all(np.isfinite(orders)) or np.any(np.asarray(orders) < 0)
                    ):
                        raise ValueError("kinetic orders must be finite and nonnegative")
                for field in (
                    "k_forward",
                    "k_reverse",
                    "ea_forward",
                    "ea_reverse",
                    "reference_temperature",
                ):
                    value = jnp.asarray(getattr(law, field))
                    if value.ndim != 0:
                        raise ValueError("kinetic coefficients must be scalars")
                    if not isinstance(value, jax.core.Tracer) and (
                        not np.isfinite(value)
                        or (field.startswith("k_") and value < 0)
                        or (field == "reference_temperature" and value <= 0)
                    ):
                        raise ValueError(
                            "rates must be finite and nonnegative; reference T positive"
                        )
                if rate_basis == "concentration":
                    raise ValueError("ReferenceRate requires activity or normalized_concentration")
                if law.detailed_balance and rate_basis != "activity":
                    raise ValueError("detailed balance requires fugacity activities")
                if (
                    law.detailed_balance
                    and not isinstance(nu, jax.core.Tracer)
                    and not isinstance(law.forward_orders, jax.core.Tracer)
                    and not isinstance(law.reverse_orders, jax.core.Tracer)
                    and not (
                        np.allclose(law.forward_orders, np.maximum(-np.asarray(nu[i]), 0))
                        and np.allclose(law.reverse_orders, np.maximum(np.asarray(nu[i]), 0))
                    )
                ):
                    raise ValueError("detailed balance requires elementary stoichiometric orders")
        reference = jnp.asarray(reference_concentration, dtype=float)
        if not isinstance(reference, jax.core.Tracer) and (
            reference.ndim != 0 or not np.isfinite(reference) or reference <= 0
        ):
            raise ValueError("reference concentration must be finite and positive")
        hf, gf, cp = reaction_arrays(list(components))
        return cls(components, labels, nu, hf, gf, cp, atoms, laws, reference, phase, rate_basis)

    def check_package(self, package: PropertyPackage) -> None:
        """Reject reference-fluid datums and mismatched component ordering."""
        if isinstance(package, HelmholtzPackage):
            raise ValueError(
                "reaction formation energy is incompatible with reference-fluid datums"
            )
        if package.n_components != len(self.components):
            raise ValueError("reaction and property-package component counts differ")
        names = getattr(getattr(package, "evidence", None), "components", ())
        if names and tuple(names) != self.components:
            raise ValueError("reaction and property-package component order differs")

    def ln_equilibrium_constants(self, t: Array) -> Array:
        """Chemical equilibrium constants referenced to ideal-gas fugacity at one bar."""
        return jax.vmap(
            lambda nu: (
                -delta_g_rxn(nu, t, self.formation_enthalpy, self.formation_gibbs, *self.cp)
                / (R * t)
            )
        )(self.nu)

    def log_activities(self, package: PropertyPackage, t: Array, p: Array, z: Array) -> Array:
        """Dimensionless log fugacities for the declared reacting phase."""
        return (
            jnp.log(jnp.maximum(z, 1e-300))
            + package.ln_phi(t, p, z, phase=self.phase)
            + jnp.log(p / P_REF)
        )

    def rates(self, package: PropertyPackage, t: Array, p: Array, z: Array) -> Array:
        """Net reaction rates in mol/(m^3 s) of reacting phase."""
        if not self.rate_laws:
            raise ValueError("kinetic calculation requires rate laws")
        if self.rate_basis == "activity":
            inputs = jnp.exp(self.log_activities(package, t, p, z))
        else:
            inputs = z / package.volume(t, p, z, phase=self.phase)
            if self.rate_basis == "normalized_concentration":
                inputs = inputs / self.reference_concentration
        ln_k = (
            self.ln_equilibrium_constants(t)
            if any(
                isinstance(law, ReferenceRate) and law.detailed_balance for law in self.rate_laws
            )
            else jnp.zeros(self.nu.shape[0])
        )
        return jnp.stack(
            [
                law.rate(t, inputs, ln_k[i])
                if isinstance(law, ReferenceRate)
                else law.rate(t, inputs)
                for i, law in enumerate(self.rate_laws)
            ]
        )

    def enthalpy(self, package: PropertyPackage, t: Array, p: Array, z: Array) -> Array:
        """Molar sensible/residual plus formation enthalpy of the reacting phase."""
        return package.enthalpy(t, p, z, phase=self.phase) + z @ self.formation_enthalpy


jax.tree_util.register_dataclass(
    ReactionSet,
    data_fields=[
        "nu",
        "formation_enthalpy",
        "formation_gibbs",
        "cp",
        "atoms",
        "rate_laws",
        "reference_concentration",
    ],
    meta_fields=["components", "names", "phase", "rate_basis"],
)
