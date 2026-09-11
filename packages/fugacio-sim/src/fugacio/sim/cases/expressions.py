"""A dimension-checked expression grammar for measurements, specs, and studies.

Expressions select calculated quantities or combine them using a small set of
arithmetic operations. They never evaluate Python or traverse arbitrary object
attributes. The same grammar drives calculations and copilot report citations.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from fugacio.sim.cases.quantities import (
    ANNUAL_COST,
    AREA,
    CONDUCTANCE,
    DIMENSIONLESS,
    FLOW,
    MASS_FLOW,
    MOLAR_ENERGY,
    MONEY,
    POWER,
    PRESSURE,
    REACTION_RATE,
    TEMPERATURE,
    VOLUME,
    CaseValidationError,
    Dimension,
    integer,
    multiply_dimensions,
    number,
    object_fields,
    quantity,
    unit_for,
)
from fugacio.sim.cases.registry import UnitDefinition, component_index
from fugacio.sim.cases.schema import Parameter, identifier, sequence

STREAM_PROPERTIES: dict[str, Dimension] = {
    "temperature": TEMPERATURE,
    "pressure": PRESSURE,
    "flow": FLOW,
    "mass_flow": MASS_FLOW,
    "mole_fraction": DIMENSIONLESS,
    "component_flow": FLOW,
    "vapor_fraction": DIMENSIONLESS,
    "enthalpy_flow": POWER,
    "molar_enthalpy": MOLAR_ENERGY,
}
UNIT_PROPERTIES: dict[str, dict[str, Dimension]] = {
    "heat_exchanger": {
        "duty": POWER,
        "ua": CONDUCTANCE,
        "area": AREA,
        "lmtd": TEMPERATURE,
        "min_approach": TEMPERATURE,
        "approach_hot_end": TEMPERATURE,
        "approach_cold_end": TEMPERATURE,
    },
    "column": {
        "condenser_duty": POWER,
        "reboiler_duty": POWER,
        "reflux_ratio": DIMENSIONLESS,
        "boilup_ratio": DIMENSIONLESS,
    },
    "compressor": {"ideal_work": POWER},
    "turbine": {"ideal_work": POWER},
    "stoichiometric_reactor": {"extent": FLOW},
}
PLANT_PROPERTIES: dict[str, Dimension] = {
    "heat": POWER,
    "work": POWER,
    "heating": POWER,
    "cooling": POWER,
    "electricity": POWER,
    "recovered_power": POWER,
    "annual_cost": ANNUAL_COST,
    "annual_utility_cost": ANNUAL_COST,
    "annualized_capital": ANNUAL_COST,
    "installed_capital": MONEY,
}
_PROFILES = {
    "t": TEMPERATURE,
    "p": PRESSURE,
    "x": DIMENSIONLESS,
    "y": DIMENSIONLESS,
    "k": DIMENSIONLESS,
    "liquid_flow": FLOW,
    "vapor_flow": FLOW,
}


def expression_dimension(
    expression: Any,
    document: dict[str, Any],
    units: tuple[UnitDefinition, ...],
    parameters: dict[str, Parameter],
    path: str = "expression",
    *,
    stack: tuple[str, ...] = (),
    depth: int = 0,
) -> Dimension:
    """Validate an expression and infer its SI dimension before numerical evaluation."""
    if depth > 24:
        raise CaseValidationError(path, "expression nesting exceeds 24 levels")
    if isinstance(expression, int | float) and not isinstance(expression, bool):
        number(expression, path)
        return DIMENSIONLESS
    if not isinstance(expression, dict):
        raise CaseValidationError(
            path, "expected a quantity, measurement, or arithmetic expression"
        )
    kinds = set(expression) & {"value", "parameter", "stream", "unit", "plant", "metric", "op"}
    # A quantity's 'unit' is a symbol, not a process-unit selector.
    if "value" in kinds:
        kinds.discard("unit")
    if len(kinds) != 1:
        raise CaseValidationError(path, "expression needs exactly one selector or operation")
    kind = next(iter(kinds))
    if kind == "value":
        d = object_fields(expression, path, allowed={"value", "unit"}, required={"value", "unit"})
        u = unit_for(d["unit"], path + ".unit")
        quantity(d, u.dimension, path, difference=d["unit"].startswith("delta_"))
        return u.dimension
    if kind == "parameter":
        d = object_fields(expression, path, allowed={kind}, required={kind})
        name = d[kind]
        if not isinstance(name, str) or name not in parameters:
            raise CaseValidationError(path, f"unknown parameter {name!r}")
        return parameters[name].dimension
    if kind == "stream":
        d = object_fields(
            expression, path, allowed={kind, "property", "component"}, required={kind, "property"}
        )
        streams = set(document["feeds"]) | {s for u in units for s in u.outlets}
        if not isinstance(d[kind], str) or d[kind] not in streams:
            raise CaseValidationError(path, f"unknown stream {d[kind]!r}")
        prop = d["property"]
        if not isinstance(prop, str) or prop not in STREAM_PROPERTIES:
            raise CaseValidationError(path, f"unknown stream property {prop!r}")
        if prop in ("component_flow", "mole_fraction"):
            component_index(d.get("component"), tuple(document["components"]), path + ".component")
        elif "component" in d:
            raise CaseValidationError(
                path, "component applies only to component_flow or mole_fraction"
            )
        return STREAM_PROPERTIES[prop]
    if kind == "unit":
        d = object_fields(
            expression,
            path,
            allowed={kind, "property", "profile", "stage", "component", "reaction"},
            required={kind},
        )
        by_name = {u.name: u for u in units}
        if not isinstance(d[kind], str) or d[kind] not in by_name:
            raise CaseValidationError(path, f"unknown unit {d[kind]!r}")
        unit = by_name[d[kind]]
        if ("property" in d) == ("profile" in d):
            raise CaseValidationError(path, "choose property or profile")
        reactive = "reactions" in unit.structure
        if "reaction" in d:
            names = [r["name"] for r in unit.structure.get("reactions", {}).get("reactions", [])]
            if d["reaction"] not in names:
                raise CaseValidationError(path, "reaction must name a reaction in this unit's set")
        if "profile" in d:
            prop = d["profile"]
            profiles = dict(_PROFILES) if unit.kind == "column" else {}
            if reactive and unit.kind != "reactive_flash":
                if all("rate" in r for r in unit.structure["reactions"]["reactions"]):
                    profiles.update(reaction_rates=REACTION_RATE)
                if unit.kind == "column":
                    profiles.update(generation=FLOW, reaction_heat=POWER, reaction_volumes=VOLUME)
                elif unit.kind != "reactive_flash":
                    profiles.update(t=TEMPERATURE, p=PRESSURE, component_flow=FLOW)
            if not isinstance(prop, str) or prop not in profiles:
                raise CaseValidationError(
                    path, "profile measurements require a known equipment profile"
                )
            count = (
                unit.structure["n_stages"]
                if unit.kind == "column"
                else 2 * unit.structure["steps"] + 1
                if unit.kind == "pfr"
                else 2
            )
            integer(d.get("stage"), path + ".stage", 1, count)
            if prop in ("x", "y", "k", "generation", "component_flow"):
                component_index(
                    d.get("component"), tuple(document["components"]), path + ".component"
                )
            elif "component" in d:
                raise CaseValidationError(path, "this profile has no component axis")
            if (prop == "reaction_rates") != ("reaction" in d):
                raise CaseValidationError(
                    path, "reaction_rates requires a reaction selector; other profiles don't"
                )
            return profiles[prop]
        if reactive and d["property"] in ("extent", "generation"):
            if "stage" in d:
                raise CaseValidationError(path, "unit totals don't have stage selectors")
            if d["property"] == "extent":
                if "reaction" not in d or "component" in d:
                    raise CaseValidationError(path, "extent requires only a reaction selector")
            else:
                component_index(
                    d.get("component"), tuple(document["components"]), path + ".component"
                )
                if "reaction" in d:
                    raise CaseValidationError(path, "generation requires only a component selector")
            return FLOW
        if any(k in d for k in ("stage", "component", "reaction")):
            raise CaseValidationError(path, "this scalar property doesn't accept axis selectors")
        props = {"heat": POWER, "work": POWER, **UNIT_PROPERTIES.get(unit.kind, {})}
        if not isinstance(d["property"], str) or d["property"] not in props:
            raise CaseValidationError(path, f"{unit.kind} has no property {d['property']!r}")
        if d["property"] == "area" and "u" not in unit.settings:
            raise CaseValidationError(
                path, "exchanger area requires a specified heat-transfer coefficient u"
            )
        return props[d["property"]]
    if kind == "plant":
        d = object_fields(expression, path, allowed={kind}, required={kind})
        prop = d[kind]
        if not isinstance(prop, str) or prop not in PLANT_PROPERTIES:
            raise CaseValidationError(path, f"unknown plant property {prop!r}")
        if PLANT_PROPERTIES[prop] in (ANNUAL_COST, MONEY) and "economics" not in document:
            raise CaseValidationError(path, "cost measurements require an economics declaration")
        return PLANT_PROPERTIES[prop]
    if kind == "metric":
        d = object_fields(expression, path, allowed={kind}, required={kind})
        name = d[kind]
        if not isinstance(name, str) or name not in document.get("metrics", {}):
            raise CaseValidationError(path, f"unknown metric {name!r}")
        if name in stack:
            raise CaseValidationError(path, f"cyclic metric references: {(*stack, name)}")
        return expression_dimension(
            document["metrics"][name]["expression"],
            document,
            units,
            parameters,
            path,
            stack=(*stack, name),
            depth=depth + 1,
        )
    d = object_fields(expression, path, allowed={"op", "args"}, required={"op", "args"})
    op = d["op"]
    if op not in ("add", "subtract", "multiply", "divide", "negate", "abs", "square", "min", "max"):
        raise CaseValidationError(path, f"unknown operation {op!r}")
    arity = 1 if op in ("negate", "abs", "square") else 2
    args = sequence(d["args"], path + ".args", minimum=arity, maximum=arity)
    dims = [
        expression_dimension(
            v, document, units, parameters, f"{path}.args[{i}]", stack=stack, depth=depth + 1
        )
        for i, v in enumerate(args)
    ]
    if op in ("add", "subtract", "min", "max"):
        if dims[0] != dims[1]:
            raise CaseValidationError(path, f"{op} requires matching dimensions")
        return dims[0]
    if op in ("multiply", "divide"):
        return multiply_dimensions(dims[0], dims[1], -1 if op == "divide" else 1)
    return multiply_dimensions(dims[0], dims[0]) if op == "square" else dims[0]


def temperature_kind(
    expression: Any, document: dict[str, Any], parameters: dict[str, Parameter]
) -> str | None:
    """Track absolute temperatures versus intervals through declared arithmetic."""
    if not isinstance(expression, dict):
        return None
    if "value" in expression:
        symbol = expression["unit"]
        return (
            ("difference" if symbol.startswith("delta_") else "absolute")
            if unit_for(symbol).dimension == TEMPERATURE
            else None
        )
    if "parameter" in expression:
        p = parameters[expression["parameter"]]
        return (
            ("difference" if p.difference else "absolute") if p.dimension == TEMPERATURE else None
        )
    if "metric" in expression:
        return temperature_kind(
            document["metrics"][expression["metric"]]["expression"], document, parameters
        )
    if expression.get("property") == "temperature" or expression.get("profile") == "t":
        return "absolute"
    if expression.get("property") in (
        "lmtd",
        "approach_hot_end",
        "approach_cold_end",
        "min_approach",
    ):
        return "difference"
    if "op" not in expression:
        return None
    roles = [temperature_kind(e, document, parameters) for e in expression["args"]]
    op = expression["op"]
    if op == "subtract" and roles == ["absolute", "absolute"]:
        return "difference"
    if op == "add" and roles == ["absolute", "absolute"]:
        raise CaseValidationError(
            "metrics",
            "add an interval to an absolute temperature, not another absolute temperature",
        )
    if op == "subtract" and roles == ["difference", "absolute"]:
        raise CaseValidationError(
            "metrics", "cannot subtract an absolute temperature from an interval"
        )
    if op in ("min", "max") and roles[0] != roles[1]:
        raise CaseValidationError(
            "metrics", "temperature comparisons require matching absolute/interval roles"
        )
    if op == "square":
        return None
    return "absolute" if "absolute" in roles else "difference" if "difference" in roles else None


def output_derived(expression: Any, document: dict[str, Any]) -> bool:
    """Whether a metric depends on process outputs, rather than only declared inputs."""
    if not isinstance(expression, dict):
        return False
    if any(k in expression for k in ("stream", "plant")) or (
        "unit" in expression and "value" not in expression
    ):
        return True
    if "metric" in expression:
        return output_derived(document["metrics"][expression["metric"]]["expression"], document)
    return any(output_derived(v, document) for v in expression.get("args", []))


def validate_metrics(
    document: dict[str, Any], units: tuple[UnitDefinition, ...], parameters: dict[str, Parameter]
) -> None:
    """Validate named result measurements and their requested display units."""
    metrics = document.setdefault("metrics", {})
    if not isinstance(metrics, dict) or len(metrics) > 1000:
        raise CaseValidationError("metrics", "expected at most 1000 named measurements")
    # Validate shapes first so references cannot crash on an incomplete target.
    for name, raw in metrics.items():
        identifier(name, "metrics")
        object_fields(
            raw,
            "metrics." + name,
            allowed={"expression", "unit", "description"},
            required={"expression", "unit"},
        )
    for name, raw in metrics.items():
        path = "metrics." + name
        dim = expression_dimension(
            raw["expression"], document, units, parameters, path, stack=(name,)
        )
        if unit_for(raw["unit"], path + ".unit").dimension != dim:
            raise CaseValidationError(path, "display unit has the wrong dimension")
        role = temperature_kind(raw["expression"], document, parameters)
        if dim == TEMPERATURE and (role == "difference") != raw["unit"].startswith("delta_"):
            raise CaseValidationError(
                path,
                "temperature intervals require delta_ display units; "
                "absolute temperatures require absolute units",
            )


def validate_specifications(
    document: dict[str, Any], units: tuple[UnitDefinition, ...], parameters: dict[str, Parameter]
) -> None:
    """Validate bounded manipulated parameters and target measurement dimensions."""
    specs = document.setdefault("specifications", [])
    names, manipulated = set(), set()
    for i, raw in enumerate(sequence(specs, "specifications", maximum=100)):
        path = f"specifications[{i}]"
        d = object_fields(
            raw,
            path,
            allowed={"name", "parameter", "metric", "target", "tolerance"},
            required={"name", "parameter", "metric", "target", "tolerance"},
        )
        name = identifier(d["name"], path + ".name")
        p = parameters.get(d["parameter"]) if isinstance(d["parameter"], str) else None
        if p is None or p.lower is None or p.upper is None or p.lower >= p.upper:
            raise CaseValidationError(
                path, "a manipulated parameter needs finite, distinct lower and upper bounds"
            )
        if name in names or p.name in manipulated:
            raise CaseValidationError(
                path, "specification names and manipulated parameters must be unique"
            )
        names.add(name)
        manipulated.add(p.name)
        dim = expression_dimension({"metric": d["metric"]}, document, units, parameters, path)
        quantity(
            d["target"],
            dim,
            path + ".target",
            difference=document["metrics"][d["metric"]]["unit"].startswith("delta_"),
        )
        tol = quantity(d["tolerance"], dim, path + ".tolerance", difference=dim == TEMPERATURE)
        if tol <= 0:
            raise CaseValidationError(path + ".tolerance", "tolerance must be positive")


def evaluate_expression(
    expression: Any, evaluation: Any, document: dict[str, Any], package: Any
) -> Any:
    """Evaluate validated arithmetic over a differentiable case evaluation in SI."""
    from fugacio.sim import properties

    if isinstance(expression, int | float):
        return jnp.asarray(expression, dtype=float)
    if "value" in expression:
        u = unit_for(expression["unit"])
        return jnp.asarray(expression["value"] * u.factor + u.offset)
    if "parameter" in expression:
        return evaluation.parameters[expression["parameter"]]
    if "metric" in expression:
        return evaluate_expression(
            document["metrics"][expression["metric"]]["expression"], evaluation, document, package
        )
    if "stream" in expression:
        s = evaluation.streams[expression["stream"]]
        prop = expression["property"]
        if prop in ("mole_fraction", "component_flow"):
            index = component_index(expression["component"], s.components, "component")
            value = s.z[index] if prop == "mole_fraction" else s.n[index]
            return jnp.where(s.total > 0, value, jnp.nan) if prop == "mole_fraction" else value
        if prop == "flow":
            return s.total
        if prop in ("temperature", "pressure"):
            return s.t if prop == "temperature" else s.p
        if prop == "mass_flow":
            return properties.mass_flow(s)
        return getattr(properties, prop)(s, model=package)
    if "unit" in expression:
        unit = evaluation.units[expression["unit"]]
        value = (
            unit.quantities[expression["property"]]
            if "property" in expression
            else unit.profiles[expression["profile"]][expression["stage"] - 1]
        )
        if "reaction" in expression:
            definition = next(u for u in document["units"] if u["name"] == expression["unit"])
            names = [
                r["name"]
                for r in document["reaction_sets"][definition["settings"]["reaction_set"]][
                    "reactions"
                ]
            ]
            value = value[names.index(expression["reaction"])]
        if "component" in expression:
            value = value[
                component_index(expression["component"], tuple(document["components"]), "component")
            ]
        return value
    if "plant" in expression:
        return evaluation.plant[expression["plant"]]
    args = [evaluate_expression(a, evaluation, document, package) for a in expression["args"]]
    op = expression["op"]
    if op == "add":
        return args[0] + args[1]
    if op == "subtract":
        return args[0] - args[1]
    if op == "multiply":
        return args[0] * args[1]
    if op == "divide":
        # Undefined measurements remain invalid instead of being regularized
        # into plausible recoveries or costs.
        return args[0] / args[1]
    if op == "negate":
        return -args[0]
    if op == "abs":
        return jnp.abs(args[0])
    if op == "square":
        return args[0] ** 2
    if op == "min":
        return jnp.minimum(args[0], args[1])
    return jnp.maximum(args[0], args[1])


def metric_values(evaluation: Any, document: dict[str, Any], package: Any) -> dict[str, Any]:
    """Read every named metric in SI, preserving JAX gradients."""
    return {
        name: evaluate_expression(m["expression"], evaluation, document, package)
        for name, m in document["metrics"].items()
    }
