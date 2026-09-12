"""Registered case units sharing the existing physical implementations.

The registry is a closed, inspectable vocabulary for portable case files.
Settings compile to SI value trees; structural settings remain static while
operating references remain differentiable. Unit results retain all reported
energy transfers, equipment quantities, stage profiles, and reaction sources.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax.numpy as jnp

from fugacio.sim.cases.quantities import (
    AREA,
    CONDUCTANCE,
    DIMENSIONLESS,
    FILM,
    FLOW,
    POWER,
    PRESSURE,
    TEMPERATURE,
    VOLUME,
    CaseValidationError,
    Dimension,
    integer,
    number,
    object_fields,
)
from fugacio.sim.cases.schema import (
    FeedDefinition,
    Parameter,
    identifier,
    resolve_value,
    sequence,
    value_spec,
)
from fugacio.sim.stream import Stream
from fugacio.thermo.components import get
from fugacio.thermo.diagnostics import SolveReport, residual_report


@dataclass(frozen=True)
class ScalarSetting:
    """Dimensional and physical constraints for one scalar unit setting."""

    dimension: Dimension = DIMENSIONLESS
    default: float | None = None
    lower: float | None = None
    upper: float | None = None
    positive: bool = False
    difference: bool = False


_TEMPERATURE = ScalarSetting(TEMPERATURE, positive=True)
_PRESSURE = ScalarSetting(PRESSURE, positive=True)
_DROP = ScalarSetting(PRESSURE, default=0.0, lower=0.0)
_EFFICIENCY = ScalarSetting(default=0.75, positive=True, upper=1.0)


@dataclass(frozen=True)
class UnitType:
    """A registered physical unit's ports and scalar setting vocabulary."""

    name: str
    inputs: tuple[int, int]
    outputs: tuple[int, int]
    scalars: dict[str, ScalarSetting]
    required: tuple[str, ...] = ()
    exactly_one: tuple[str, ...] = ()
    structural: tuple[str, ...] = ()
    description: str = ""
    backends: tuple[str, ...] = ("sequential", "eo")


UNIT_TYPES: dict[str, UnitType] = {
    "mixer": UnitType(
        "mixer",
        (1, 100),
        (1, 1),
        {"t": _TEMPERATURE, "p": _PRESSURE},
        description="Adiabatic or temperature-specified mixer.",
    ),
    "splitter": UnitType(
        "splitter",
        (1, 1),
        (2, 100),
        {},
        structural=("fractions",),
        description="Conservative flow splitter.",
    ),
    "heater": UnitType(
        "heater",
        (1, 1),
        (1, 1),
        {"t_out": _TEMPERATURE, "duty": ScalarSetting(POWER), "dp": _DROP},
        exactly_one=("t_out", "duty"),
        description="Heater or cooler with a temperature or duty specification.",
    ),
    "valve": UnitType(
        "valve",
        (1, 1),
        (1, 1),
        {"p_out": _PRESSURE},
        required=("p_out",),
        description="Isenthalpic pressure letdown.",
    ),
    "pump": UnitType(
        "pump",
        (1, 1),
        (1, 1),
        {"p_out": _PRESSURE, "efficiency": _EFFICIENCY},
        required=("p_out",),
        description="Liquid pump retaining shaft power.",
    ),
    "compressor": UnitType(
        "compressor",
        (1, 1),
        (1, 1),
        {"p_out": _PRESSURE, "efficiency": _EFFICIENCY},
        required=("p_out",),
        description="Isentropic compressor with efficiency.",
    ),
    "turbine": UnitType(
        "turbine",
        (1, 1),
        (1, 1),
        {"p_out": _PRESSURE, "efficiency": ScalarSetting(default=0.85, positive=True, upper=1.0)},
        required=("p_out",),
        description="Isentropic expansion retaining recovered power.",
    ),
    "flash": UnitType(
        "flash",
        (1, 1),
        (2, 2),
        {"t": _TEMPERATURE, "p": _PRESSURE},
        required=("t", "p"),
        description="Isothermal flash with vapor and liquid products and required duty.",
    ),
    "component_separator": UnitType(
        "component_separator",
        (1, 1),
        (2, 2),
        {
            "top_t": _TEMPERATURE,
            "top_p": _PRESSURE,
            "bottom_t": _TEMPERATURE,
            "bottom_p": _PRESSURE,
        },
        structural=("split_to_top",),
        description="Specified component split; inferred heat is explicitly reported.",
    ),
    "heat_exchanger": UnitType(
        "heat_exchanger",
        (2, 2),
        (2, 2),
        {
            "duty": ScalarSetting(POWER, lower=0.0),
            "t_hot_out": _TEMPERATURE,
            "t_cold_out": _TEMPERATURE,
            "min_approach": ScalarSetting(TEMPERATURE, lower=0.0, difference=True),
            "ua": ScalarSetting(CONDUCTANCE, positive=True),
            "area": ScalarSetting(AREA, positive=True),
            "u": ScalarSetting(FILM, positive=True),
            "dp_hot": _DROP,
            "dp_cold": _DROP,
        },
        exactly_one=("duty", "t_hot_out", "t_cold_out", "min_approach", "ua", "area"),
        structural=("flow", "zones"),
        description=(
            "Two-sided exchanger, retaining duty, area, conductance, and temperature curves."
        ),
    ),
    "column": UnitType(
        "column",
        (1, 100),
        (2, 100),
        {
            "p": _PRESSURE,
            "efficiency": ScalarSetting(default=1.0, positive=True, upper=1.0),
            "reflux_temperature": _TEMPERATURE,
        },
        required=("p",),
        structural=(
            "n_stages",
            "feed_stages",
            "specs",
            "condenser",
            "reboiler",
            "side_draws",
            "stage_duties",
            "reaction_set",
            "reaction_volumes",
        ),
        description="Rigorous MESH column with complete products and stage profiles.",
    ),
    "stoichiometric_reactor": UnitType(
        "stoichiometric_reactor",
        (1, 1),
        (1, 1),
        {
            "conversion": ScalarSetting(lower=0.0, upper=1.0),
            "t_out": _TEMPERATURE,
            "duty": ScalarSetting(POWER),
            "dp": _DROP,
        },
        required=("conversion",),
        exactly_one=("t_out", "duty"),
        structural=("nu", "key"),
        description=(
            "One reaction with specified key-reactant conversion and formation-consistent energy."
        ),
    ),
}

for _kind in ("equilibrium_reactor", "cstr", "pfr"):
    UNIT_TYPES[_kind] = UnitType(
        _kind,
        (1, 1),
        (1, 1),
        {
            "t_out": _TEMPERATURE,
            "duty": ScalarSetting(POWER),
            "dp": _DROP,
            **(
                {"volume": ScalarSetting(VOLUME, positive=True)}
                if _kind != "equilibrium_reactor"
                else {}
            ),
        },
        required=("reaction_set",) + (("volume",) if _kind != "equilibrium_reactor" else ()),
        exactly_one=("t_out", "duty"),
        structural=("reaction_set",) + (("steps",) if _kind == "pfr" else ()),
        description="Homogeneous package reactor with generation and phase/energy checks.",
    )
UNIT_TYPES["reactive_flash"] = UnitType(
    "reactive_flash",
    (1, 1),
    (2, 2),
    {"t": _TEMPERATURE, "p": _PRESSURE},
    required=("t", "p", "reaction_set"),
    structural=("reaction_set",),
    description="Chemical and phase equilibrium with vapor/liquid products and heat duty.",
)


_COLUMN_SPEC_DIMS = {
    "reflux_ratio": DIMENSIONLESS,
    "boilup_ratio": DIMENSIONLESS,
    "reflux_rate": FLOW,
    "distillate_rate": FLOW,
    "bottoms_rate": FLOW,
    "condenser_duty": POWER,
    "reboiler_duty": POWER,
    "mole_fraction": DIMENSIONLESS,
    "recovery": DIMENSIONLESS,
    "component_flow": FLOW,
    "stage_temperature": TEMPERATURE,
}


@dataclass(frozen=True)
class UnitDefinition:
    """A validated, compiled unit definition over one case component basis."""

    name: str
    kind: str
    inlets: tuple[str, ...]
    outlets: tuple[str, ...]
    settings: dict[str, Any]
    structure: dict[str, Any] = field(default_factory=dict)

    @property
    def unit_type(self) -> UnitType:
        """Registry entry for this unit."""
        return UNIT_TYPES[self.kind]


def component_index(name: Any, components: tuple[str, ...], path: str) -> int:
    """Resolve component names in specifications, rejecting index ambiguity."""
    try:
        canonical = get(name).name if isinstance(name, str) else ""
    except KeyError as exc:
        raise CaseValidationError(path, str(exc)) from exc
    if canonical not in components:
        raise CaseValidationError(path, f"component {name!r} is not in the case basis")
    return components.index(canonical)


def element_matrix(components: tuple[str, ...]) -> tuple[tuple[str, ...], list[list[float]]]:
    """Read exact element counts from the component formulas, including parentheses."""
    from fugacio.thermo.reaction_system import element_matrix as shared_elements

    try:
        return shared_elements(components)
    except ValueError as exc:
        raise CaseValidationError("reaction", str(exc)) from exc


def _column_structure(
    raw: dict[str, Any],
    components: tuple[str, ...],
    params: dict[str, Parameter],
    path: str,
    n_in: int,
    n_out: int,
) -> dict[str, Any]:
    n = integer(raw.get("n_stages"), path + ".n_stages", 2, 500)
    stages = [
        integer(v, path + ".feed_stages", 1, n)
        for v in sequence(raw.get("feed_stages"), path + ".feed_stages", minimum=n_in, maximum=n_in)
    ]
    condenser = raw.get("condenser", "total")
    reboiler = raw.get("reboiler", "kettle")
    if condenser not in (None, "total", "partial") or reboiler not in (None, "kettle"):
        raise CaseValidationError(path, "unknown condenser or reboiler type")
    specs = []
    count = int(condenser is not None) + int(reboiler is not None)
    for i, spec in enumerate(
        sequence(raw.get("specs", []), path + ".specs", minimum=count, maximum=count)
    ):
        sp = f"{path}.specs[{i}]"
        d = object_fields(
            spec,
            sp,
            allowed={"kind", "value", "component", "product", "stage"},
            required={"kind", "value"},
        )
        kind = d["kind"]
        if kind not in _COLUMN_SPEC_DIMS:
            raise CaseValidationError(sp, f"unknown column specification {kind!r}")
        compiled: dict[str, Any] = {
            "kind": kind,
            "value": value_spec(d["value"], _COLUMN_SPEC_DIMS[kind], params, sp + ".value"),
        }
        if kind in ("mole_fraction", "recovery", "component_flow"):
            compiled["component"] = component_index(
                d.get("component"), components, sp + ".component"
            )
            if d.get("product") not in ("distillate", "bottoms"):
                raise CaseValidationError(sp + ".product", "choose distillate or bottoms")
            compiled["product"] = d["product"]
        elif "component" in d or "product" in d:
            raise CaseValidationError(
                sp,
                "component/product applies only to composition, recovery, or component-flow specs",
            )
        if kind == "stage_temperature":
            compiled["stage"] = integer(d.get("stage"), sp + ".stage", 1, n)
        elif "stage" in d:
            raise CaseValidationError(sp, "stage applies only to a stage_temperature specification")
        specs.append(compiled)
    signatures = [(s["kind"], s.get("component"), s.get("product"), s.get("stage")) for s in specs]
    if len(set(signatures)) != len(signatures):
        raise CaseValidationError(path + ".specs", "duplicate column specifications")
    draws = []
    for i, draw in enumerate(sequence(raw.get("side_draws", []), path + ".side_draws")):
        dp = f"{path}.side_draws[{i}]"
        d = object_fields(
            draw,
            dp,
            allowed={"stage", "phase", "fraction"},
            required={"stage", "phase", "fraction"},
        )
        if d["phase"] not in ("vapor", "liquid"):
            raise CaseValidationError(dp, "draw phase must be liquid or vapor")
        draws.append(
            {
                "stage": integer(d["stage"], dp + ".stage", 1, n),
                "phase": d["phase"],
                "fraction": value_spec(d["fraction"], DIMENSIONLESS, params, dp + ".fraction"),
            }
        )
    if n_out != 2 + len(draws):
        raise CaseValidationError(
            path, "column outlets must be distillate, bottoms, then one per side draw"
        )
    duties = []
    for i, duty in enumerate(sequence(raw.get("stage_duties", []), path + ".stage_duties")):
        dp = f"{path}.stage_duties[{i}]"
        d = object_fields(duty, dp, allowed={"stage", "duty"}, required={"stage", "duty"})
        duties.append(
            {
                "stage": integer(d["stage"], dp + ".stage", 1, n),
                "duty": value_spec(d["duty"], POWER, params, dp + ".duty"),
            }
        )
    return {
        "n_stages": n,
        "feed_stages": stages,
        "condenser": condenser,
        "reboiler": reboiler,
        "specs": specs,
        "side_draws": draws,
        "stage_duties": duties,
    }


def parse_units(
    raw: Any,
    components: tuple[str, ...],
    parameters: dict[str, Parameter],
    reaction_sets: Any = None,
) -> tuple[UnitDefinition, ...]:
    """Compile registered unit specifications with strict port and setting validation."""
    from fugacio.sim.cases.reactions import REACTOR_KINDS, parse_reaction_sets

    sets = parse_reaction_sets(
        {} if reaction_sets is None else reaction_sets, components, parameters
    )
    result = []
    for i, value in enumerate(sequence(raw, "units", minimum=1, maximum=500)):
        path = f"units[{i}]"
        d = object_fields(
            value,
            path,
            allowed={"name", "kind", "inlets", "outlets", "settings"},
            required={"name", "kind", "inlets", "outlets"},
        )
        name = identifier(d["name"], path + ".name")
        kind = d["kind"]
        if not isinstance(kind, str) or kind not in UNIT_TYPES:
            raise CaseValidationError(path + ".kind", f"choose from {sorted(UNIT_TYPES)}")
        entry = UNIT_TYPES[kind]
        ins = tuple(
            identifier(v, path + ".inlets")
            for v in sequence(
                d["inlets"], path + ".inlets", minimum=entry.inputs[0], maximum=entry.inputs[1]
            )
        )
        outs = tuple(
            identifier(v, path + ".outlets")
            for v in sequence(
                d["outlets"], path + ".outlets", minimum=entry.outputs[0], maximum=entry.outputs[1]
            )
        )
        sp = path + ".settings"
        source = object_fields(
            d.get("settings", {}),
            sp,
            allowed=set(entry.scalars) | set(entry.structural),
            required=set(entry.required),
        )
        if entry.exactly_one and sum(key in source for key in entry.exactly_one) != 1:
            raise CaseValidationError(sp, f"specify exactly one of {entry.exactly_one}")
        settings: dict[str, Any] = {}
        for key, rule in entry.scalars.items():
            if key in source:
                settings[key] = value_spec(
                    source[key],
                    rule.dimension,
                    parameters,
                    sp + "." + key,
                    difference=rule.difference,
                )
            elif rule.default is not None:
                settings[key] = rule.default
        structure: dict[str, Any] = {}
        if kind in ("splitter", "component_separator"):
            key = "fractions" if kind == "splitter" else "split_to_top"
            length = len(outs) if kind == "splitter" else len(components)
            settings[key] = [
                value_spec(v, DIMENSIONLESS, parameters, f"{sp}.{key}[{j}]")
                for j, v in enumerate(
                    sequence(source.get(key), sp + "." + key, minimum=length, maximum=length)
                )
            ]
        elif kind == "heat_exchanger":
            flow = source.get("flow", "counter")
            if flow not in ("counter", "co"):
                raise CaseValidationError(sp + ".flow", "choose counter or co")
            structure = {
                "flow": "parallel" if flow == "co" else flow,
                "zones": integer(source.get("zones", 1), sp + ".zones", 1, 100),
            }
            if "area" in settings and "u" not in settings:
                raise CaseValidationError(sp, "area requires u")
        elif kind == "column":
            structure = _column_structure(source, components, parameters, sp, len(ins), len(outs))
            if "reflux_temperature" in settings and structure["condenser"] != "total":
                raise CaseValidationError(sp, "reflux_temperature requires a total condenser")
        elif kind == "stoichiometric_reactor":
            nu = [
                number(v, sp + ".nu")
                for v in sequence(
                    source.get("nu"), sp + ".nu", minimum=len(components), maximum=len(components)
                )
            ]
            key_index = component_index(source.get("key"), components, sp + ".key")
            if nu[key_index] >= 0 or not any(v > 0 for v in nu):
                raise CaseValidationError(
                    sp, "key must be a reactant and the reaction needs products"
                )
            _, matrix = element_matrix(components)
            if any(abs(sum(a * b for a, b in zip(row, nu, strict=True))) > 1e-8 for row in matrix):
                raise CaseValidationError(sp + ".nu", "reaction must conserve every element")
            structure = {"nu": nu, "key": key_index}
        if "reaction_set" in source:
            selected = source["reaction_set"]
            if not isinstance(selected, str) or selected not in sets:
                raise CaseValidationError(sp + ".reaction_set", "reference a declared reaction set")
            structure["reactions"] = sets[selected]
            structure["components"] = components
            if kind in ("column", "cstr", "pfr") and any(
                "rate" not in r for r in sets[selected]["reactions"]
            ):
                raise CaseValidationError(
                    sp, "kinetic equipment requires a rate for every reaction"
                )
            if kind == "column":
                n = structure["n_stages"]
                settings["reaction_volumes"] = [
                    value_spec(v, VOLUME, parameters, sp + ".reaction_volumes")
                    for v in sequence(
                        source.get("reaction_volumes"),
                        sp + ".reaction_volumes",
                        minimum=n,
                        maximum=n,
                    )
                ]
            if kind == "pfr":
                structure["steps"] = integer(source.get("steps", 64), sp + ".steps", 1, 4096)
        elif kind in REACTOR_KINDS or "reaction_volumes" in source:
            raise CaseValidationError(sp, "reactive equipment requires a reaction_set")
        result.append(UnitDefinition(name, kind, ins, outs, settings, structure))
    if len({u.name for u in result}) != len(result):
        raise CaseValidationError("units", "unit names must be unique")
    return tuple(result)


def validate_topology(feeds: tuple[FeedDefinition, ...], units: tuple[UnitDefinition, ...]) -> None:
    """Require one producer and at most one consumer per material stream.

    A split is explicit equipment; connecting the same material stream twice
    would otherwise duplicate matter. Recycles are allowed and need a path from
    a feed plus at least one external product.
    """
    feed_names = {f.name for f in feeds}
    producers = {name: "feed" for name in feed_names}
    consumers: Counter[str] = Counter()
    for unit in units:
        for name in unit.outlets:
            if name in producers:
                raise CaseValidationError(
                    "units." + unit.name, f"duplicate producer for stream {name!r}"
                )
            producers[name] = unit.name
        consumers.update(unit.inlets)
    for name, count in consumers.items():
        if name not in producers:
            raise CaseValidationError("connections", f"stream {name!r} has no producer")
        if count > 1:
            raise CaseValidationError(
                "connections", f"stream {name!r} is consumed more than once; add a splitter"
            )
    if any(name not in consumers for name in feed_names):
        raise CaseValidationError("feeds", "every feed must be connected to a unit")
    if not set(producers) - set(consumers):
        raise CaseValidationError("connections", "case needs an external product")
    reachable = set(feed_names)
    for _ in units:
        for unit in units:
            if reachable.intersection(unit.inlets):
                reachable.update(unit.outlets)
    disconnected = [u.name for u in units if not reachable.intersection(u.inlets)]
    if disconnected:
        raise CaseValidationError("connections", f"units have no path from a feed: {disconnected}")


def validate_unit_values(units: tuple[UnitDefinition, ...], parameters: dict[str, float]) -> None:
    """Check resolved physical setting limits at each concrete operating point."""

    def r(value: Any) -> float:
        return parameters[value] if isinstance(value, str) else float(value)

    for unit in units:
        path = "units." + unit.name + ".settings"
        for key, rule in unit.unit_type.scalars.items():
            if key not in unit.settings:
                continue
            value = r(unit.settings[key])
            if (
                (rule.positive and value <= 0)
                or (rule.lower is not None and value < rule.lower)
                or (rule.upper is not None and value > rule.upper)
            ):
                raise CaseValidationError(
                    path + "." + key, "value is outside the physical setting limits"
                )
        for key in ("fractions", "split_to_top"):
            if key in unit.settings:
                values = [r(v) for v in unit.settings[key]]
                if any(v < 0 or v > 1 for v in values) or (
                    key == "fractions" and abs(sum(values) - 1) > 1e-10
                ):
                    raise CaseValidationError(
                        path + "." + key,
                        "fractions must be in [0, 1]; splitter fractions must sum to one",
                    )
        if "reactions" in unit.structure:
            from fugacio.sim.cases.reactions import validate_reaction_values

            validate_reaction_values(
                unit.structure["reactions"],
                parameters,
                unit.structure["components"],
                path + ".reaction_set",
            )
        if "reaction_volumes" in unit.settings:
            volumes = [r(v) for v in unit.settings["reaction_volumes"]]
            if any(v < 0 for v in volumes):
                raise CaseValidationError(path, "reaction volumes must be nonnegative")
            if (
                unit.structure["condenser"] == "total"
                and unit.structure["reactions"]["phase"] == "vapor"
                and volumes[0] > 0
            ):
                raise CaseValidationError(path, "total condensers have no reacting vapor volume")
        if unit.kind == "column":
            for spec in unit.structure["specs"]:
                value = r(spec["value"])
                kind = spec["kind"]
                if (kind in ("mole_fraction", "recovery") and not 0 < value < 1) or (
                    kind != "condenser_duty" and value <= 0
                ):
                    raise CaseValidationError(path + ".specs", f"invalid value for {kind}")
            groups: dict[tuple[int, str], float] = defaultdict(float)
            for draw in unit.structure["side_draws"]:
                value = r(draw["fraction"])
                if not 0 <= value < 1:
                    raise CaseValidationError(
                        path + ".side_draws", "draw fractions must be in [0, 1)"
                    )
                groups[draw["stage"], draw["phase"]] += value
            if any(v >= 1 for v in groups.values()):
                raise CaseValidationError(
                    path + ".side_draws", "draw fractions at a stage must sum to less than one"
                )


class UnitEvaluation(NamedTuple):
    """Differentiable unit results with explicit external energy and reaction sources."""

    outlets: tuple[Stream, ...]
    heat: Any
    work: Any
    generation: Any
    report: SolveReport
    quantities: dict[str, Any]
    profiles: dict[str, Any]
    heating: Any
    cooling: Any


def evaluate_unit(
    definition: UnitDefinition,
    inputs: tuple[Stream, ...],
    parameters: dict[str, Any],
    package: Any,
    *,
    guess: dict[str, Any] | None = None,
    column_solver: str = "block",
) -> UnitEvaluation:
    """Evaluate a registered unit using the existing public numerical kernels."""
    from fugacio.sim import distillation as dist
    from fugacio.sim import units as ops
    from fugacio.sim.cases.reactions import REACTOR_KINDS, build_reaction_set
    from fugacio.sim.heat_exchanger import heat_exchanger
    from fugacio.sim.properties import enthalpy_flow

    kind = definition.kind
    system = (
        build_reaction_set(definition.structure["reactions"], inputs[0].components, parameters)
        if "reactions" in definition.structure
        else None
    )
    kw = {k: resolve_value(v, parameters) for k, v in definition.settings.items()}
    zero = jnp.asarray(0.0)
    heat, work = zero, zero
    generation = jnp.zeros_like(inputs[0].n)
    report = residual_report(jnp.zeros(1))
    quantities: dict[str, Any] = {}
    profiles: dict[str, Any] = {}
    heating = cooling = None
    outputs: tuple[Stream, ...]
    result: Any

    def h(streams: tuple[Stream, ...]) -> Any:
        return sum((enthalpy_flow(s, model=package) for s in streams), zero)

    if kind == "mixer":
        outputs = (ops.mix(list(inputs), model=package, **kw),)
        heat = h(outputs) - h(inputs) if "t" in kw else zero
    elif kind == "splitter":
        outputs = ops.splitter(inputs[0], kw["fractions"])
    elif kind == "heater":
        result = ops.heater(inputs[0], model=package, **kw)
        outputs, heat, report = (result.outlet,), result.duty, result.report
    elif kind == "valve":
        outputs = (ops.valve(inputs[0], model=package, **kw),)
    elif kind in ("pump", "compressor", "turbine"):
        result = getattr(ops, kind)(inputs[0], model=package, **kw)
        outputs, work, report = (result.outlet,), result.work, result.report
        if hasattr(result, "ideal_work"):
            quantities["ideal_work"] = result.ideal_work
    elif kind == "flash":
        outputs = ops.flash_drum(inputs[0], model=package, **kw)
        heat = h(outputs) - h(inputs)
    elif kind == "component_separator":
        outputs = ops.component_separator(inputs[0], **kw)
        heat = h(outputs) - h(inputs)
    elif kind == "heat_exchanger":
        result = heat_exchanger(inputs[0], inputs[1], model=package, **kw, **definition.structure)
        outputs, report = (result.hot_out, result.cold_out), result.report
        quantities.update(
            {
                k: getattr(result, k)
                for k in (
                    "duty",
                    "ua",
                    "lmtd",
                    "area",
                    "approach_hot_end",
                    "approach_cold_end",
                    "min_approach",
                )
            }
        )
        profiles.update(hot_temperature=result.hot_curve, cold_temperature=result.cold_curve)
    elif kind == "column":
        st = definition.structure
        specs = [
            dist.ColumnSpec(**{**s, "value": resolve_value(s["value"], parameters)})
            for s in st["specs"]
        ]
        draws = [
            dist.SideDraw(**{**s, "fraction": resolve_value(s["fraction"], parameters)})
            for s in st["side_draws"]
        ]
        duties = [
            dist.StageDuty(**{**s, "duty": resolve_value(s["duty"], parameters)})
            for s in st["stage_duties"]
        ]
        result = dist.rigorous_column(
            [
                dist.ColumnFeed(feed, stage)
                for feed, stage in zip(inputs, st["feed_stages"], strict=True)
            ],
            st["n_stages"],
            specs=specs,
            condenser=st["condenser"],
            reboiler=st["reboiler"],
            side_draws=draws,
            stage_duties=duties,
            model=package,
            check=False,
            guess=guess,
            linear_solver=column_solver,
            reactions=system,
            **kw,
        )
        outputs = (result.distillate, result.bottoms, *result.side_draws)
        generation = jnp.sum(result.generation, axis=0)
        if system is not None:
            quantities["extent"] = jnp.sum(
                result.reaction_volumes[:, None] * result.reaction_rates, axis=0
            )
            quantities["generation"] = generation
            profiles.update(
                generation=result.generation,
                reaction_rates=result.reaction_rates,
                reaction_heat=result.reaction_heat,
                reaction_volumes=result.reaction_volumes,
            )
        heat_terms = jnp.array(
            [result.condenser_duty, result.reboiler_duty, *(s.duty for s in duties)]
        )
        heat = jnp.sum(heat_terms)
        heating, cooling = jnp.sum(jnp.maximum(heat_terms, 0)), jnp.sum(jnp.maximum(-heat_terms, 0))
        report = result.report
        quantities.update(
            {
                k: getattr(result, k)
                for k in ("condenser_duty", "reboiler_duty", "reflux_ratio", "boilup_ratio")
            }
        )
        profiles.update(
            {
                k: getattr(result, k)
                for k in (
                    "t",
                    "p",
                    "x",
                    "y",
                    "k",
                    "liquid_flow",
                    "vapor_flow",
                    "stage_liquid",
                    "stage_vapor",
                )
            }
        )
    elif kind in REACTOR_KINDS:
        assert system is not None
        from fugacio.sim.reaction_units import reaction_reactor
        from fugacio.sim.reactive import reactive_flash

        if kind == "reactive_flash":
            result = reactive_flash(inputs[0], system, model=package, check=False, **kw)
            outputs = (result.vapor, result.liquid)
        else:
            result = reaction_reactor(
                inputs[0],
                system,
                kind="equilibrium" if kind == "equilibrium_reactor" else kind,
                model=package,
                check=False,
                **kw,
                **({"steps": definition.structure["steps"]} if kind == "pfr" else {}),
            )
            outputs = (result.outlet,)
            profiles.update(
                coordinate=result.coordinate,
                component_flow=result.component_profile,
                t=result.temperature_profile,
                p=result.pressure_profile,
                reaction_rates=result.rate_profile,
            )
            quantities.update(
                material_error=result.material_error,
                element_error=result.element_error,
                energy_error=result.energy_error,
                phase_error=result.phase_error,
                integration_error=result.integration_error,
            )
        heat, generation, report = result.duty, result.generation, result.report
        quantities.update(extent=result.extent, generation=result.generation)
    elif kind == "stoichiometric_reactor":
        from fugacio.thermo.reactions import reaction_arrays

        nu = jnp.asarray(definition.structure["nu"])
        key = definition.structure["key"]
        extent = kw["conversion"] * inputs[0].n[key] / -nu[key]
        generation = extent * nu
        n = inputs[0].n + generation
        p = inputs[0].p - kw["dp"]
        hf, _, _ = reaction_arrays(list(inputs[0].components))
        if "t_out" in kw:
            output = Stream(n, kw["t_out"], p, inputs[0].components)
            heat = h((output,)) - h(inputs) + generation @ hf
        else:
            heat = kw["duty"]
            target = (h(inputs) + heat - generation @ hf) / jnp.sum(n)
            output = Stream.from_ph(
                inputs[0].components, n / jnp.sum(n), jnp.sum(n), p, target, model=package
            )
        outputs = (output,)
        quantities["extent"] = extent
    else:
        raise ValueError(f"unregistered unit {kind!r}")
    quantities.update(heat=heat, work=work)
    return UnitEvaluation(
        outputs,
        heat,
        work,
        generation,
        report,
        quantities,
        profiles,
        jnp.maximum(heat, 0) if heating is None else heating,
        jnp.maximum(-heat, 0) if cooling is None else cooling,
    )


def registry_schema() -> list[dict[str, Any]]:
    """Describe the portable unit vocabulary for CLI clients and copilot planning."""
    return [
        {
            "kind": entry.name,
            "description": entry.description,
            "inlets": list(entry.inputs),
            "outlets": list(entry.outputs),
            "backends": list(entry.backends),
            "settings": list(entry.scalars) + list(entry.structural),
            "required": list(entry.required),
            "exactly_one": list(entry.exactly_one),
            "scalar_settings": {
                name: {
                    "si_dimension": list(rule.dimension),
                    "default_si": rule.default,
                    "lower_si": rule.lower,
                    "upper_si": rule.upper,
                    "positive": rule.positive,
                    "temperature_difference": rule.difference,
                }
                for name, rule in entry.scalars.items()
            },
        }
        for entry in UNIT_TYPES.values()
    ]
