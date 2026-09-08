"""Versioned, immutable, portable process definitions.

Only registered units and a bounded expression grammar can appear in a case.
There are no Python expressions, imports, pickle objects, or executable file
references in the format. Arbitrary Python flowsheets remain a separate API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from fugacio.sim.cases.jsonio import canonical_json, digest, loads, read_json, write_json
from fugacio.sim.cases.quantities import (
    DIMENSIONLESS,
    FLOW,
    MOLAR_ENERGY,
    PRESSURE,
    TEMPERATURE,
    CaseValidationError,
    Dimension,
    integer,
    number,
    object_fields,
    quantity,
    unit_for,
)
from fugacio.thermo.components import get

SCHEMA_VERSION = 1
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")


def identifier(value: Any, path: str) -> str:
    """Validate a portable identifier usable in files, metrics, and connections."""
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise CaseValidationError(path, "use a letter followed by up to 63 letters, digits, _ or -")
    return value


def sequence(value: Any, path: str, *, minimum: int = 0, maximum: int = 1000) -> list[Any]:
    """Read a bounded array with a path-aware length error."""
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise CaseValidationError(path, f"expected an array with {minimum} to {maximum} items")
    return value


@dataclass(frozen=True)
class Parameter:
    """A scalar operating parameter with SI defaults and optional SI bounds."""

    name: str
    value: float
    unit: str
    lower: float | None = None
    upper: float | None = None

    @property
    def dimension(self) -> Dimension:
        """Physical dimension of this parameter."""
        return unit_for(self.unit).dimension

    @property
    def difference(self) -> bool:
        """Whether the parameter represents a temperature interval."""
        return self.unit.startswith("delta_")

    def check(self, value: float, path: str) -> None:
        """Reject overrides outside the declared bounds."""
        if self.lower is not None and value < self.lower:
            raise CaseValidationError(path, f"value {value:g} is below SI bound {self.lower:g}")
        if self.upper is not None and value > self.upper:
            raise CaseValidationError(path, f"value {value:g} is above SI bound {self.upper:g}")


def parse_parameters(data: Any) -> dict[str, Parameter]:
    """Parse scalar parameter declarations, converting defaults and bounds together."""
    if not isinstance(data, dict) or len(data) > 1000:
        raise CaseValidationError("parameters", "expected an object with at most 1000 parameters")
    result = {}
    for name, raw in data.items():
        path = "parameters." + identifier(name, "parameters")
        d = object_fields(
            raw,
            path,
            allowed={"value", "unit", "lower", "upper", "description"},
            required={"value", "unit"},
        )
        u = unit_for(d["unit"], path + ".unit")

        def convert(key: str, d: Any = d, path: str = path, u: Any = u) -> float:
            return number(number(d[key], path + "." + key) * u.factor + u.offset, path + "." + key)

        p = Parameter(
            name,
            convert("value"),
            d["unit"],
            convert("lower") if "lower" in d else None,
            convert("upper") if "upper" in d else None,
        )
        if p.lower is not None and p.upper is not None and p.lower > p.upper:
            raise CaseValidationError(path, "lower bound exceeds upper bound")
        p.check(p.value, path)
        result[name] = p
    return result


def value_spec(
    value: Any,
    dimension: Dimension,
    parameters: dict[str, Parameter],
    path: str,
    *,
    difference: bool = False,
) -> float | str:
    """Compile a literal or ``{"parameter": name}`` reference into an SI value spec."""
    if isinstance(value, dict) and "parameter" in value:
        d = object_fields(value, path, allowed={"parameter"}, required={"parameter"})
        name = d["parameter"]
        if not isinstance(name, str) or name not in parameters:
            raise CaseValidationError(path, f"unknown parameter {name!r}")
        p = parameters[name]
        if p.dimension != dimension:
            raise CaseValidationError(path, f"parameter {name!r} has the wrong dimension")
        if dimension == TEMPERATURE and p.difference != difference:
            raise CaseValidationError(
                path, "absolute temperatures and differences cannot be exchanged"
            )
        return name
    return quantity(value, dimension, path, difference=difference)


def resolve_value(value: Any, parameters: dict[str, Any]) -> Any:
    """Resolve an already validated SI value tree without making traced values concrete."""
    if isinstance(value, str):
        return parameters[value]
    if isinstance(value, list | tuple):
        return jnp.asarray([resolve_value(v, parameters) for v in value])
    return jnp.asarray(value, dtype=float)


@dataclass(frozen=True)
class FeedDefinition:
    """Compiled feed specifications, retaining explicit branch selection."""

    name: str
    flow: float | str
    z: tuple[float | str, ...]
    pressure: float | str
    temperature: float | str | None
    enthalpy: float | str | None
    phase: str | None

    def build(self, components: tuple[str, ...], parameters: dict[str, Any], package: Any) -> Any:
        """Construct a differentiable stream from operating parameters."""
        from fugacio.sim.stream import Stream

        z = resolve_value(self.z, parameters)
        flow = resolve_value(self.flow, parameters)
        p = resolve_value(self.pressure, parameters)
        if self.enthalpy is not None:
            return Stream.from_ph(
                components, z, flow, p, resolve_value(self.enthalpy, parameters), model=package
            )
        return Stream.from_fractions(
            components, z, flow, resolve_value(self.temperature, parameters), p, phase=self.phase
        )


def parse_feeds(
    data: Any, components: tuple[str, ...], parameters: dict[str, Parameter]
) -> tuple[FeedDefinition, ...]:
    """Validate feed bases and specifications without running thermodynamics."""
    if not isinstance(data, dict) or not 1 <= len(data) <= 1000:
        raise CaseValidationError("feeds", "expected one to 1000 named feeds")
    result = []
    for name, raw in data.items():
        path = "feeds." + identifier(name, "feeds")
        d = object_fields(
            raw,
            path,
            allowed={"flow", "z", "temperature", "pressure", "enthalpy", "phase"},
            required={"flow", "z", "pressure"},
        )
        if ("temperature" in d) == ("enthalpy" in d):
            raise CaseValidationError(path, "specify exactly one of temperature and enthalpy")
        phase = d.get("phase")
        if phase not in (None, "liquid", "vapor") or (phase is not None and "enthalpy" in d):
            raise CaseValidationError(
                path + ".phase", "phase is liquid or vapor, for PT feeds only"
            )

        def v(key: str, dim: Dimension, d: Any = d, path: str = path) -> float | str:
            return value_spec(d[key], dim, parameters, path + "." + key)

        fractions = sequence(d["z"], path + ".z", minimum=len(components), maximum=len(components))
        result.append(
            FeedDefinition(
                name,
                v("flow", FLOW),
                tuple(
                    value_spec(x, DIMENSIONLESS, parameters, f"{path}.z[{i}]")
                    for i, x in enumerate(fractions)
                ),
                v("pressure", PRESSURE),
                v("temperature", TEMPERATURE) if "temperature" in d else None,
                v("enthalpy", MOLAR_ENERGY) if "enthalpy" in d else None,
                phase,
            )
        )
    return tuple(result)


def validate_feed_values(feeds: tuple[FeedDefinition, ...], parameters: dict[str, float]) -> None:
    """Check a concrete set of feed conditions before compiling any physical calculation."""

    def r(value: float | str) -> float:
        return parameters[value] if isinstance(value, str) else value

    for feed in feeds:
        path = "feeds." + feed.name
        z = [r(v) for v in feed.z]
        if any(x < 0 or x > 1 for x in z) or abs(sum(z) - 1.0) > 1e-10:
            raise CaseValidationError(path + ".z", "nonnegative mole fractions must sum to one")
        if r(feed.flow) < 0:
            raise CaseValidationError(path + ".flow", "molar flow must be nonnegative")
        if r(feed.pressure) <= 0:
            raise CaseValidationError(path + ".pressure", "absolute pressure must be positive")
        if feed.temperature is not None and r(feed.temperature) <= 0:
            raise CaseValidationError(
                path + ".temperature", "absolute temperature must be positive"
            )
    if not any(r(f.flow) > 0 for f in feeds):
        raise CaseValidationError("feeds", "at least one feed must have positive flow")


def _package_schema(raw: Any, components: tuple[str, ...]) -> None:
    from fugacio.sim.models import METHODS

    d = object_fields(
        raw,
        "property_package",
        allowed={"method", "options", "measured_fit", "qualification"},
        required={"method"},
    )
    if d["method"] not in METHODS:
        raise CaseValidationError("property_package.method", f"choose from {METHODS}")
    opts = object_fields(
        d.get("options", {}),
        "property_package.options",
        allowed={
            "vapor",
            "poynting",
            "phi_saturation",
            "parameter_policy",
            "use_database_kij",
            "kij",
        },
    )
    method = d["method"]
    supported = {"parameter_policy"}
    if method in ("pr", "srk", "rk", "vdw", "pcsaft"):
        supported |= {"kij", "use_database_kij"}
    elif method in ("nrtl", "uniquac", "unifac", "dortmund"):
        supported |= {"kij", "vapor", "poynting", "phi_saturation"}
    if set(opts) - supported:
        raise CaseValidationError("property_package.options", f"unsupported options for {method}")
    if method == "iapws" and len(components) != 1:
        raise CaseValidationError(
            "property_package", "reference-fluid packages require one component"
        )
    if "measured_fit" in d and set(opts) - {"parameter_policy"}:
        raise CaseValidationError(
            "property_package.options", "measured fits retain their fitted model assumptions"
        )
    for key in ("poynting", "phi_saturation", "use_database_kij"):
        if key in opts and not isinstance(opts[key], bool):
            raise CaseValidationError("property_package.options." + key, "expected a boolean")
    if "vapor" in opts and opts["vapor"] not in ("ideal", "eos"):
        raise CaseValidationError("property_package.options.vapor", "choose ideal or eos")
    if opts.get("parameter_policy", "strict") not in ("strict", "allow_ideal"):
        raise CaseValidationError(
            "property_package.options.parameter_policy", "choose strict or allow_ideal"
        )
    if "kij" in opts:
        for i, row in enumerate(
            sequence(opts["kij"], "kij", minimum=len(components), maximum=len(components))
        ):
            for j, val in enumerate(
                sequence(row, f"kij[{i}]", minimum=len(components), maximum=len(components))
            ):
                number(val, f"kij[{i}][{j}]")
    if "measured_fit" in d:
        if d["method"] != "nrtl" or len(components) != 2:
            raise CaseValidationError(
                "property_package.measured_fit", "requires a binary NRTL package"
            )
        if not isinstance(d["measured_fit"], dict):
            raise CaseValidationError(
                "property_package.measured_fit", "expected an inline MeasuredFit artifact"
            )
    if "qualification" in d:
        q = object_fields(
            d["qualification"],
            "property_package.qualification",
            allowed={"holdout_ids"},
            required={"holdout_ids"},
        )
        ids = sequence(q["holdout_ids"], "property_package.qualification.holdout_ids", minimum=1)
        if (
            "measured_fit" not in d
            or not all(isinstance(i, str) for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise CaseValidationError(
                "property_package.qualification", "requires a measured fit and unique holdout IDs"
            )


@dataclass(frozen=True)
class ProcessCase:
    """An immutable validated case with a stable content hash.

    Construct with :meth:`from_dict` or :meth:`load`. ``to_dict`` returns a new
    object, so a caller cannot mutate a compiled case by editing its source map.
    """

    _json: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProcessCase:
        """Validate an entire process description, including topology and metric dimensions."""
        # Normalize Python tuples to JSON arrays and reject non-JSON objects.
        try:
            d = loads(canonical_json(value))
        except (TypeError, ValueError) as exc:
            raise CaseValidationError("case", str(exc)) from exc
        object_fields(
            d,
            "case",
            allowed={
                "schema_version",
                "name",
                "description",
                "components",
                "property_package",
                "parameters",
                "feeds",
                "units",
                "metrics",
                "specifications",
                "economics",
            },
            required={"schema_version", "name", "components", "property_package", "feeds", "units"},
        )
        if integer(d["schema_version"], "schema_version") != SCHEMA_VERSION:
            raise CaseValidationError("schema_version", f"supported version is {SCHEMA_VERSION}")
        identifier(d["name"], "name")
        if not isinstance(d.get("description", ""), str) or len(d.get("description", "")) > 20000:
            raise CaseValidationError("description", "expected text of at most 20000 characters")
        names = sequence(d["components"], "components", minimum=1, maximum=100)
        if not all(isinstance(n, str) for n in names):
            raise CaseValidationError("components", "expected component names")
        try:
            canonical = tuple(get(n).name for n in names)
        except KeyError as exc:
            raise CaseValidationError("components", str(exc)) from exc
        if len(set(canonical)) != len(canonical):
            raise CaseValidationError(
                "components", "component aliases must not duplicate a species"
            )
        d["components"] = list(canonical)
        _package_schema(d["property_package"], canonical)
        params = parse_parameters(d.setdefault("parameters", {}))
        feeds = parse_feeds(d["feeds"], canonical, params)
        values = {k: p.value for k, p in params.items()}
        validate_feed_values(feeds, values)
        from fugacio.sim.cases.registry import parse_units, validate_topology, validate_unit_values

        units = parse_units(d["units"], canonical, params)
        validate_topology(feeds, units)
        validate_unit_values(units, values)
        from fugacio.sim.cases.expressions import validate_metrics, validate_specifications

        validate_metrics(d, units, params)
        validate_specifications(d, units, params)
        from fugacio.sim.cases.costing import validate_economics

        validate_economics(d, units, params)
        return cls(canonical_json(d))

    @classmethod
    def load(cls, path: str | Path) -> ProcessCase:
        """Read and validate a portable JSON case."""
        return cls.from_dict(read_json(path))

    def save(self, path: str | Path) -> None:
        """Atomically save the case with its explicit input units."""
        write_json(path, self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON representation."""
        return loads(self._json)

    @property
    def name(self) -> str:
        """Portable case name."""
        return self.to_dict()["name"]

    @property
    def case_id(self) -> str:
        """SHA-256 identity of this exact case revision."""
        return digest(self.to_dict())

    @property
    def components(self) -> tuple[str, ...]:
        """Canonical ordered component basis."""
        return tuple(self.to_dict()["components"])

    @property
    def parameters(self) -> dict[str, Parameter]:
        """Fresh parameter declarations in SI."""
        return parse_parameters(self.to_dict()["parameters"])

    def with_parameters(self, overrides: dict[str, Any]) -> ProcessCase:
        """Create a new case revision using explicit quantities for parameter updates."""
        d = self.to_dict()
        for name, raw in overrides.items():
            if name not in self.parameters:
                raise CaseValidationError("parameters", f"unknown parameter {name!r}")
            p = self.parameters[name]
            si = quantity(raw, p.dimension, "parameters." + name, difference=p.difference)
            p.check(si, "parameters." + name)
            u = unit_for(p.unit)
            d["parameters"][name]["value"] = (si - u.offset) / u.factor
        return self.from_dict(d)
