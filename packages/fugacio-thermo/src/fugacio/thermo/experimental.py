"""Reproducible, phase-aware measurements from the vendored NIST ThermoML corpus.

Raw archive bytes and their SHA-256 hashes are retained. Complementary tables
are joined by identical independent conditions, never by row position. The
normalizer accepts binary VLE, excess enthalpy, and cloud-point temperatures;
it rejects other layouts instead of inventing a composition or uncertainty.
Cloud points aren't conjugate tie lines and aren't accepted by the VLE fitter.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import Any

from fugacio.thermo.components import DATABASE
from fugacio.thermo.thermoml import Column, ThermoMLData, Uncertainty, convert_values, loads

_ROOT = files("fugacio.thermo").joinpath("datasets", "measured")


@dataclass(frozen=True)
class Measurement:
    """One published value in SI units, preserving phase, role, and uncertainty."""

    quantity: str
    value: float
    unit: str
    phase: str | None
    component: str | None
    role: str
    uncertainty: Uncertainty | None = None


@dataclass(frozen=True)
class Observation:
    """One experimental condition with its joined measurements and source IDs.

    Compositions use ``components`` order. ``dataset_ids`` includes every table
    contributing to the condition, allowing splits that keep joined data together.
    """

    id: str
    source: str
    source_sha256: str
    dataset_ids: tuple[int, ...]
    components: tuple[str, str]
    cas: tuple[str, str]
    kind: str
    measurements: tuple[Measurement, ...]

    def measurement(self, quantity: str, phase: str | None = None) -> Measurement:
        """Select exactly one quantity, optionally within a phase."""
        found = [
            m
            for m in self.measurements
            if m.quantity == quantity and (phase is None or m.phase == phase)
        ]
        if len(found) != 1:
            raise ValueError(f"{self.id}: expected one {quantity}/{phase}, got {len(found)}")
        return found[0]

    @property
    def temperature(self) -> float:
        """Temperature in kelvin."""
        return self.measurement("temperature").value

    @property
    def pressure(self) -> float:
        """Pressure in pascals; unavailable pressure raises rather than defaulting."""
        return self.measurement("pressure").value

    def composition(self, phase: str = "Liquid") -> tuple[float, float]:
        """Binary composition, completing only the explicitly binary complement."""
        found = [m for m in self.measurements if m.quantity == "mole_fraction" and m.phase == phase]
        if not found or any(m.component not in self.components for m in found):
            raise ValueError(f"{self.id}: missing or unidentified {phase} composition")
        by_name = {m.component: m.value for m in found}
        if len(by_name) != len(found):
            raise ValueError("duplicate phase composition")
        x0 = by_name.get(self.components[0], 1 - by_name.get(self.components[1], math.nan))
        x1 = by_name.get(self.components[1], 1 - x0)
        if min(x0, x1) < 0 or max(x0, x1) > 1 or abs(x0 + x1 - 1) > 1e-8:
            raise ValueError(f"{self.id}: invalid binary mole fractions")
        return x0, x1

    def to_dict(self) -> dict[str, Any]:
        """Serializable record with explicit SI units and retained uncertainties."""
        return asdict(self)


def _quantity(col: Column) -> tuple[str, str]:
    q = col.quantity
    if q in (
        "Temperature",
        "Boiling temperature at pressure P",
        "Liquid-liquid equilibrium temperature",
    ):
        return "temperature", "K"
    if q in ("Pressure", "Vapor or sublimation pressure"):
        return "pressure", "Pa"
    if q == "Mole fraction":
        return "mole_fraction", "1"
    if q == "Excess molar enthalpy (molar enthalpy of mixing)":
        return "excess_enthalpy", "J/mol"
    raise ValueError(f"unsupported measurement {col.label!r}")


def _scaled_uncertainty(u: Uncertainty | None, source: str, target: str) -> Uncertainty | None:
    if u is None:
        return None
    zero, one = convert_values((0, 1), source, target)
    scale = abs(one - zero)
    return Uncertainty(
        None if u.standard is None else u.standard * scale,
        None if u.expanded is None else u.expanded * scale,
        u.coverage_factor,
        u.confidence_percent,
        u.assessment,
    )


def normalize_document(
    data: ThermoMLData,
    identities: dict[str, dict[str, str]],
    selected: tuple[int, ...],
    excluded_rows: dict[str, dict[str, str]] | None = None,
) -> tuple[Observation, ...]:
    """Normalize explicitly selected tables using CAS/InChIKey identity evidence.

    Unknown identifiers, conflicting identifiers, ambiguous joins, nonfinite
    cells, transformed properties, and incomplete conditions raise ValueError.
    A DOI and raw-file checksum are required for measured evidence.
    """
    if not data.doi or not data.sha256:
        raise ValueError("measured records require a DOI and raw-file checksum")
    cas_names = {c.cas: name for name, c in DATABASE.items() if c.cas}
    resolved: dict[int, tuple[str, str]] = {}
    for c in data.compounds:
        identity = identities.get(c.inchikey or "")
        cas = c.cas or (identity["cas"] if identity else None)
        if c.cas and identity and c.cas != identity["cas"]:
            raise ValueError("conflicting CAS and InChIKey identity")
        if cas in cas_names:
            resolved[c.org_num] = (cas_names[cas], str(cas))

    # Each condition has independent values followed by distinct properties.
    groups: dict[str, tuple[tuple[int, ...], list[int], list[Measurement], set[str]]] = {}
    available = {d.number for d in data.datasets}
    if len(set(selected)) != len(selected) or not set(selected) <= available:
        raise ValueError("duplicate or absent selected dataset ID")
    for ds in data.datasets:
        if ds.number not in selected:
            continue
        ds.validate()
        exclusions = (excluded_rows or {}).get(str(ds.number), {})
        if any(not reason or not 1 <= int(row) <= len(ds) for row, reason in exclusions.items()):
            raise ValueError("row exclusions need valid one-based indices and reasons")
        if len(ds.components) != 2 or any(c not in resolved for c in ds.components):
            raise ValueError("selected dataset needs two identified database components")
        components = tuple(sorted(ds.components, key=lambda n: resolved[n][1]))
        normalized = []
        for j, col in enumerate(ds.columns):
            quantity, unit = _quantity(col)
            if col.component is not None and col.component not in components:
                raise ValueError("composition references a component outside the mixture")
            column_values = ds.values_in(col, unit)
            uncertainties = (
                [row[j] for row in ds.uncertainties] if ds.uncertainties else [None] * len(ds)
            )
            normalized.append(
                [
                    Measurement(
                        quantity,
                        value,
                        unit,
                        col.phase,
                        resolved[col.component][0] if col.component else None,
                        col.role,
                        _scaled_uncertainty(u, col.unit or "1", unit),
                    )
                    for value, u in zip(column_values, uncertainties, strict=True)
                ]
            )
        for i in range(len(ds)):
            if str(i + 1) in exclusions:
                continue
            row = [column[i] for column in normalized]
            independent = [m for m in row if m.role != "property"]
            # Variables and constraints are equivalent conditions for joining.
            key_values = sorted(
                (
                    m.quantity,
                    (m.phase or "") if m.quantity == "mole_fraction" else "",
                    m.component or "",
                    m.value,
                )
                for m in independent
            )
            key = json.dumps([components, key_values], separators=(",", ":"))
            if key not in groups:
                groups[key] = (components, [], independent, set())
            _, ids, values, labels = groups[key]
            if ds.number in ids:
                raise ValueError("duplicate experimental conditions within a dataset")
            ids.append(int(ds.number))
            for col, m in zip(ds.columns, row, strict=True):
                if m.role == "property":
                    if any(
                        (v.quantity, v.phase, v.component) == (m.quantity, m.phase, m.component)
                        for v in values
                    ):
                        raise ValueError("ambiguous complementary-table join")
                    values.append(m)
                    labels.add(col.quantity)
    observations: list[Observation] = []
    for key, (components, ids, values, labels) in groups.items():
        kind = (
            "excess_enthalpy"
            if any(m.quantity == "excess_enthalpy" for m in values)
            else "cloud_point"
            if "Liquid-liquid equilibrium temperature" in labels
            else "vle"
        )
        pair = tuple(resolved[c][0] for c in components)
        pair_cas = tuple(resolved[c][1] for c in components)
        obs = Observation(
            hashlib.sha256((data.doi + key).encode()).hexdigest()[:20],
            data.doi,
            data.sha256,
            tuple(sorted(ids)),
            (pair[0], pair[1]),
            (pair_cas[0], pair_cas[1]),
            kind,
            tuple(values),
        )
        if obs.temperature <= 0:
            raise ValueError("temperature must be positive")
        if kind != "vle" or any(m.quantity == "pressure" for m in values):
            if obs.pressure <= 0:
                raise ValueError("pressure must be positive")
        else:
            raise ValueError("VLE condition lacks pressure")
        obs.composition("Liquid mixture 1" if kind == "cloud_point" else "Liquid")
        if any(m.phase == "Gas" and m.quantity == "mole_fraction" for m in values):
            obs.composition("Gas")
        observations.append(obs)
    return tuple(sorted(observations, key=lambda o: o.id))


def corpus_manifest() -> dict[str, Any]:
    """Read source citations, checksums, identity mappings, and explicit exclusions."""
    manifest = json.loads(_ROOT.joinpath("manifest.json").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported corpus schema")
    return manifest


def load_corpus() -> tuple[Observation, ...]:
    """Verify all raw checksums and reconstruct the measured corpus offline."""
    manifest = corpus_manifest()
    observations: list[Observation] = []
    for source in manifest["sources"]:
        filename = source["file"]
        if "/" in filename or "\\" in filename or filename in (".", ".."):
            raise ValueError("invalid corpus filename")
        raw = _ROOT.joinpath(filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"corpus checksum mismatch: {filename}")
        data = loads(raw)
        if data.doi != source["doi"] or source["evidence"] != "measured":
            raise ValueError("source identity or evidence mismatch")
        included = set(source["selected_datasets"])
        excluded = {int(n) for n in source["excluded_datasets"]}
        if included & excluded or included | excluded != {d.number for d in data.datasets}:
            raise ValueError("every source dataset must be included or explicitly excluded")
        observations.extend(
            normalize_document(
                data,
                manifest["identities"],
                tuple(source["selected_datasets"]),
                source.get("excluded_rows"),
            )
        )
    if len({o.id for o in observations}) != len(observations):
        raise ValueError("duplicate corpus observation IDs")
    return tuple(observations)


def grouped_split(
    observations: tuple[Observation, ...], *, by: str, holdout: tuple[str, ...]
) -> tuple[tuple[Observation, ...], tuple[Observation, ...]]:
    """Split by explicit source DOI, temperature (K), or source/dataset ID.

    Temperature keys use ``format(T, '.12g')``. Dataset keys use ``DOI#number``;
    selecting any joined table holds out the whole observation. No random row
    splitting is offered. Both partitions must be nonempty; missing keys raise.
    """
    if by not in ("source", "temperature", "dataset") or not holdout:
        raise ValueError("choose source, temperature, or dataset and nonempty holdout keys")

    def keys(o: Observation) -> set[str]:
        if by == "source":
            return {o.source}
        if by == "temperature":
            return {format(o.temperature, ".12g")}
        return {f"{o.source}#{n}" for n in o.dataset_ids}

    available = set().union(*(keys(o) for o in observations))
    selected = set(holdout)
    if not selected <= available:
        raise ValueError(f"unknown holdout keys: {sorted(selected - available)}")
    if by == "dataset":
        # Complementary tables form one experimental group. Close over shared
        # table IDs so a partially joined table can't straddle the partitions.
        while True:
            expanded = selected | set().union(
                *(keys(o) for o in observations if keys(o) & selected)
            )
            if expanded == selected:
                break
            selected = expanded
    train = tuple(o for o in observations if not keys(o) & selected)
    test = tuple(o for o in observations if keys(o) & selected)
    if not train or not test:
        raise ValueError("training and holdout partitions must both be nonempty")
    return train, test
