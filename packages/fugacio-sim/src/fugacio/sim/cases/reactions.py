"""Portable reaction definitions with explicit kinetic dimensions and references."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from fugacio.sim.cases.quantities import (
    CONCENTRATION,
    DIMENSIONLESS,
    MOLAR_ENERGY,
    REACTION_RATE,
    TEMPERATURE,
    CaseValidationError,
    number,
    object_fields,
)
from fugacio.sim.cases.schema import Parameter, identifier, resolve_value, sequence, value_spec
from fugacio.thermo.reaction_system import ReactionSet, ReferenceRate
from fugacio.thermo.reactions import Reaction

RATE_DIMENSIONS = {
    "k_forward": REACTION_RATE,
    "k_reverse": REACTION_RATE,
    "ea_forward": MOLAR_ENERGY,
    "ea_reverse": MOLAR_ENERGY,
    "reference_temperature": TEMPERATURE,
}
REACTOR_KINDS = ("equilibrium_reactor", "cstr", "pfr", "reactive_flash")


def parse_reaction_sets(
    raw: Any, components: tuple[str, ...], parameters: dict[str, Parameter]
) -> dict[str, dict[str, Any]]:
    """Validate every reusable definition, including currently unused sets."""
    if not isinstance(raw, dict) or len(raw) > 100:
        raise CaseValidationError("reaction_sets", "expected at most 100 named sets")
    result = {}
    for name, source in raw.items():
        path = "reaction_sets." + identifier(name, "reaction_sets")
        d = object_fields(
            source,
            path,
            allowed={"phase", "rate_basis", "reference_concentration", "reactions"},
            required={"phase", "reactions"},
        )
        phase, basis = d["phase"], d.get("rate_basis", "activity")
        if phase not in ("liquid", "vapor"):
            raise CaseValidationError(path + ".phase", "choose liquid or vapor")
        if basis not in ("activity", "normalized_concentration"):
            raise CaseValidationError(
                path + ".rate_basis", "choose activity or normalized_concentration"
            )
        reference = value_spec(
            d.get("reference_concentration", {"value": 1, "unit": "mol/m3"}),
            CONCENTRATION,
            parameters,
            path + ".reference_concentration",
        )
        reactions = []
        for i, entry in enumerate(
            sequence(d["reactions"], path + ".reactions", minimum=1, maximum=len(components))
        ):
            rp = f"{path}.reactions[{i}]"
            r = object_fields(entry, rp, allowed={"name", "nu", "rate"}, required={"name", "nu"})
            nu = [
                number(v, rp + ".nu")
                for v in sequence(
                    r["nu"], rp + ".nu", minimum=len(components), maximum=len(components)
                )
            ]
            item: dict[str, Any] = {"name": identifier(r["name"], rp + ".name"), "nu": nu}
            if "rate" in r:
                law = object_fields(
                    r["rate"],
                    rp + ".rate",
                    allowed=set(RATE_DIMENSIONS)
                    | {"forward_orders", "reverse_orders", "detailed_balance"},
                    required={"k_forward", "reference_temperature"},
                )
                compiled: dict[str, Any] = {
                    k: value_spec(v, RATE_DIMENSIONS[k], parameters, rp + ".rate." + k)
                    for k, v in law.items()
                    if k in RATE_DIMENSIONS
                }
                for k in ("k_reverse", "ea_forward", "ea_reverse"):
                    compiled.setdefault(k, 0.0)
                for field, sign in (("forward_orders", -1), ("reverse_orders", 1)):
                    orders = law.get(field, [max(sign * v, 0.0) for v in nu])
                    compiled[field] = [
                        value_spec(v, DIMENSIONLESS, parameters, rp + ".rate." + field)
                        for v in sequence(
                            orders,
                            rp + ".rate." + field,
                            minimum=len(components),
                            maximum=len(components),
                        )
                    ]
                detailed = law.get("detailed_balance", False)
                if not isinstance(detailed, bool):
                    raise CaseValidationError(rp, "detailed_balance must be a boolean")
                if detailed and ("k_reverse" in law or "ea_reverse" in law):
                    raise CaseValidationError(
                        rp, "detailed balance determines reverse coefficients"
                    )
                compiled["detailed_balance"] = detailed
                item["rate"] = compiled
            reactions.append(item)
        if any("rate" in r for r in reactions) and not all("rate" in r for r in reactions):
            raise CaseValidationError(path, "supply rates for every reaction or none")
        result[name] = {
            "phase": phase,
            "rate_basis": basis,
            "reference_concentration": reference,
            "reactions": reactions,
        }
        validate_reaction_values(
            result[name], {k: p.value for k, p in parameters.items()}, components, path
        )
    return result


def build_reaction_set(
    definition: dict[str, Any], components: tuple[str, ...], parameters: dict[str, Any]
) -> ReactionSet:
    """Bind kinetic parameters as pytree leaves, preserving JIT and implicit derivatives."""
    reactions, laws, names = [], [], []
    for item in definition["reactions"]:
        reactions.append(Reaction(components, jnp.asarray(item["nu"])))
        names.append(item["name"])
        if "rate" in item:
            values = {
                k: (v if k == "detailed_balance" else resolve_value(v, parameters))
                for k, v in item["rate"].items()
            }
            laws.append(ReferenceRate(**values))
    return ReactionSet.from_reactions(
        reactions,
        laws,
        names=names,
        phase=definition["phase"],
        rate_basis=definition["rate_basis"],
        reference_concentration=resolve_value(definition["reference_concentration"], parameters),
    )


def validate_reaction_values(
    definition: dict[str, Any], parameters: dict[str, float], components: tuple[str, ...], path: str
) -> None:
    """Reject invalid operating coefficients before compiling a case or study point."""
    from fugacio.sim.reaction_units import reaction_parameter_validity

    try:
        system = build_reaction_set(definition, components, parameters)
        if not bool(reaction_parameter_validity(system)):
            raise ValueError(
                "reaction coefficients must be finite, rates and orders nonnegative, "
                "and references positive"
            )
    except (TypeError, ValueError) as exc:
        raise CaseValidationError(path, str(exc)) from exc
