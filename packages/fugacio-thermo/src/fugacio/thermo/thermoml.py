"""Reader for the NIST ThermoML archive XML format.

`ThermoML <https://www.nist.gov/mml/acmd/trc/thermoml>`_ is the IUPAC/NIST XML
standard for thermophysical and thermochemical property data; the public
`ThermoML Archive
<https://www.nist.gov/mml/acmd/trc/thermoml/thermoml-archive>`_ holds tens of
thousands of experimental datasets. This module turns those files into tidy,
typed tables you can feed straight into `fugacio.thermo.regression`, so a
model can be fitted to *real measurements*, and predictions graded against them.

The parser is dependency-free (standard-library
`xml.etree.ElementTree` only):

* XML namespaces are stripped, so files declaring the ThermoML namespace (or none)
  parse identically;
* compounds, mixtures, variables, properties, and the numeric value rows are read
  by their *local* element names, matching the published schema without binding to
  a specific version;
* each `Dataset` exposes its columns as aligned numeric rows plus
  convenience accessors (`Dataset.temperature`, `Dataset.pressure`,
  `Dataset.mole_fraction`) with pressure unit conversion to pascal.

Synthetic schema-faithful datasets ship for tests and examples; see
`list_samples` / `load_sample`. The separate measured corpus and strict
normalization workflow live in `fugacio.thermo.experimental`.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import IO
from xml.etree import ElementTree as ET

__all__ = [
    "Column",
    "Compound",
    "Dataset",
    "ThermoMLData",
    "Uncertainty",
    "convert_values",
    "list_samples",
    "load_sample",
    "loads",
    "read_thermoml",
    "sample_path",
]

# Pressure unit -> factor to pascal. ThermoML labels carry the unit after a comma,
# e.g. "Vapor or sublimation pressure, kPa".
_PRESSURE_TO_PA: dict[str, float] = {
    "Pa": 1.0,
    "kPa": 1.0e3,
    "MPa": 1.0e6,
    "GPa": 1.0e9,
    "bar": 1.0e5,
    "kbar": 1.0e8,
    "atm": 101325.0,
    "mmHg": 133.32236842105263,
    "psia": 6894.757293168361,
    "psi": 6894.757293168361,
}

_CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")


def convert_values(values: tuple[float, ...], source: str, target: str) -> tuple[float, ...]:
    """Convert supported measurement units, rejecting unknown or incompatible units.

    Supported dimensions are pressure, temperature, molar energy, mass density,
    molar volume, and dimensionless composition. No unit is inferred when absent.
    """
    groups = [
        _PRESSURE_TO_PA,
        {"K": 1.0, "degC": 1.0, "C": 1.0, "°C": 1.0},
        {"J/mol": 1.0, "kJ/mol": 1000.0},
        {"kg/m3": 1.0, "kg/m^3": 1.0, "g/cm3": 1000.0, "g/cm^3": 1000.0},
        {"m3/mol": 1.0, "m^3/mol": 1.0, "cm3/mol": 1e-6, "cm^3/mol": 1e-6},
        {"1": 1.0, "mol/mol": 1.0, "%": 0.01},
    ]
    for group in groups:
        if source in group and target in group:
            offset_in = 273.15 if source in ("degC", "C", "°C") else 0.0
            offset_out = 273.15 if target in ("degC", "C", "°C") else 0.0
            return tuple(
                (v * group[source] + offset_in - offset_out) / group[target] for v in values
            )
    raise ValueError(f"unsupported unit conversion {source!r} -> {target!r}")


def _local(tag: str) -> str:
    """Strip an XML namespace prefix from a tag (``{ns}Name`` -> ``Name``)."""
    return tag.rsplit("}", 1)[-1]


def _find(elem: ET.Element, name: str) -> ET.Element | None:
    for child in elem.iter():
        if _local(child.tag) == name:
            return child
    return None


def _find_direct(elem: ET.Element, name: str) -> list[ET.Element]:
    """Direct children of ``elem`` whose local tag is ``name``."""
    return [c for c in elem if _local(c.tag) == name]


def _text(elem: ET.Element | None) -> str | None:
    if elem is None or elem.text is None:
        return None
    s = elem.text.strip()
    return s or None


def _first_text(elem: ET.Element, name: str) -> str | None:
    return _text(_find(elem, name))


def _label_unit(label: str | None) -> tuple[str, str | None]:
    """Split a ThermoML label ``"Pressure, kPa"`` into ``("Pressure", "kPa")``."""
    if label is None:
        return "", None
    if "," in label:
        head, _, tail = label.rpartition(",")
        return head.strip(), tail.strip()
    return label.strip(), None


@dataclass(frozen=True)
class Compound:
    """A chemical compound declared in a ThermoML document.

    Attributes:
        org_num: The document-local organization number used to reference this
            compound from mixtures and composition variables.
        name: Common name, if given.
        formula: Molecular formula, if given.
        cas: CAS registry number, if present.
        inchikey: Standard InChIKey, if present.
    """

    org_num: int
    name: str | None = None
    formula: str | None = None
    cas: str | None = None
    inchikey: str | None = None
    inchi: str | None = None


@dataclass(frozen=True)
class Uncertainty:
    """A reported uncertainty, in the associated column's original units.

    Expanded uncertainty is converted to standard uncertainty only when the
    source explicitly supplies a coverage factor. Confidence alone isn't a
    coverage factor. Missing uncertainty remains unknown.
    """

    standard: float | None = None
    expanded: float | None = None
    coverage_factor: float | None = None
    confidence_percent: float | None = None
    assessment: str | None = None

    @property
    def standard_value(self) -> float | None:
        """Standard uncertainty, or None when it cannot be recovered."""
        if self.standard is not None:
            return self.standard
        if self.expanded is not None and self.coverage_factor is not None:
            return self.expanded / self.coverage_factor
        return None


@dataclass(frozen=True)
class Column:
    """One variable or property column of a `Dataset` table.

    Attributes:
        number: The ``nVarNumber`` / ``nPropNumber`` within the dataset.
        role: ``"variable"`` (an independent, controlled quantity) or
            ``"property"`` (a measured quantity).
        kind: The ThermoML type element local name, e.g. ``"eTemperature"``,
            ``"ePressure"``, ``"eComponentComposition"``.
        label: Human-readable label including units, e.g. ``"Pressure, kPa"``.
        component: For composition columns, the ``org_num`` of the component the
            fraction refers to; otherwise ``None``.
    """

    number: int
    role: str
    kind: str
    label: str
    component: int | None = None
    phase: str | None = None
    presentation: str = "Direct value, X"
    method: str | None = None

    @property
    def quantity(self) -> str:
        """The label with any trailing unit removed."""
        return _label_unit(self.label)[0]

    @property
    def unit(self) -> str | None:
        """The unit parsed from the label, if any."""
        return _label_unit(self.label)[1]


@dataclass(frozen=True)
class Dataset:
    """A ``PureOrMixtureData`` block: a table of measurements for one mixture.

    The ``rows`` are aligned with ``columns``; a missing cell is ``float('nan')``.

    Attributes:
        components: ``org_num`` of each component participating, in document order.
        columns: The variable and property columns, in document order.
        rows: Numeric rows aligned with ``columns``.
        phase: The reported phase string, if any (e.g. ``"Liquid"``).
        number: The ``nPureOrMixtureDataNumber`` identifier, if present.
    """

    components: tuple[int, ...]
    columns: tuple[Column, ...]
    rows: tuple[tuple[float, ...], ...]
    phase: str | None = None
    number: int | None = None
    phases: tuple[str, ...] = ()
    uncertainties: tuple[tuple[Uncertainty | None, ...], ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def labels(self) -> tuple[str, ...]:
        """Column labels, in column order."""
        return tuple(c.label for c in self.columns)

    def __len__(self) -> int:
        return len(self.rows)

    def _index(self, col: Column) -> int:
        return self.columns.index(col)

    def values(self, col: Column) -> tuple[float, ...]:
        """All values of one column, in row order."""
        i = self._index(col)
        return tuple(row[i] for row in self.rows)

    def find_column(
        self,
        *,
        kind: str | None = None,
        quantity: str | None = None,
        component: int | None = None,
        phase: str | None = None,
        role: str | None = None,
    ) -> Column | None:
        """First column matching the given filters (any combination)."""
        for c in self.columns:
            if kind is not None and c.kind != kind:
                continue
            if quantity is not None and c.quantity.lower() != quantity.lower():
                continue
            if component is not None and c.component != component:
                continue
            if phase is not None and c.phase != phase:
                continue
            if role is not None and c.role != role:
                continue
            return c
        return None

    def select_column(self, **filters: str | int) -> Column:
        """Select exactly one column; reject missing and ambiguous measurements."""
        matches = [
            c
            for c in self.columns
            if all(getattr(c, key) == value for key, value in filters.items())
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one column for {filters}, found {len(matches)}")
        return matches[0]

    def values_in(self, col: Column, unit: str) -> tuple[float, ...]:
        """Read a direct measurement in a supported unit, without guessing its basis."""
        if col.presentation != "Direct value, X":
            raise ValueError(f"unsupported presentation {col.presentation!r}")
        source = col.unit
        if source is None and col.quantity == "Mole fraction":
            source = "1"
        if source is None:
            raise ValueError(f"missing unit for {col.label!r}")
        return convert_values(self.values(col), source, unit)

    def standard_uncertainties(self, col: Column, unit: str) -> tuple[float | None, ...]:
        """Standard uncertainties in a requested unit; unknown values remain None."""
        if not self.uncertainties:
            return (None,) * len(self)
        source = col.unit or ("1" if col.quantity == "Mole fraction" else "")
        zero, one = convert_values((0.0, 1.0), source, unit)
        scale = abs(one - zero)
        index = self._index(col)
        values = []
        for row in self.uncertainties:
            uncertainty = row[index]
            value = uncertainty.standard_value if uncertainty is not None else None
            values.append(None if value is None else value * scale)
        return tuple(values)

    def validate(self) -> None:
        """Reject incomplete, nonfinite, or unsupported measurement tables."""
        if self.issues:
            raise ValueError("; ".join(self.issues))
        if not self.rows or not self.components:
            raise ValueError("dataset needs components and measurement rows")
        if len(set(self.components)) != len(self.components):
            raise ValueError("duplicate dataset components")
        for col in self.columns:
            if col.presentation != "Direct value, X":
                raise ValueError(f"unsupported presentation {col.presentation!r}")
        for row in self.rows:
            if len(row) != len(self.columns) or not all(math.isfinite(v) for v in row):
                raise ValueError("missing or nonfinite measurement")

    def temperature(self) -> tuple[float, ...]:
        """Temperatures in kelvin (raises if the dataset has no temperature column)."""
        col = self.find_column(kind="eTemperature")
        if col is None:
            col = next(
                (
                    c
                    for c in self.columns
                    if c.quantity
                    in (
                        "Boiling temperature at pressure P",
                        "Temperature",
                        "Liquid-liquid equilibrium temperature",
                    )
                ),
                None,
            )
        if col is None:
            raise KeyError("dataset has no temperature column")
        return self.values_in(col, "K")

    def pressure(self, *, unit: str = "Pa") -> tuple[float, ...]:
        """Pressures converted to ``unit`` (default pascal).

        Accepts pressure stored either as a controlled variable (``ePressure``) or
        as a measured property (e.g. a vapour-pressure column).
        """
        col = self.find_column(kind="ePressure")
        if col is None:
            for c in self.columns:
                if "pressure" in c.quantity.lower():
                    col = c
                    break
        if col is None:
            raise KeyError("dataset has no pressure column")
        return self.values_in(col, unit)

    def mole_fraction(self, component: int, *, phase: str | None = None) -> tuple[float, ...]:
        """Mole fractions of ``component`` (by ``org_num``)."""
        matches = [
            c
            for c in self.columns
            if c.component == component
            and c.quantity == "Mole fraction"
            and (phase is None or c.phase == phase)
        ]
        if len(matches) > 1:
            raise ValueError("ambiguous composition; select a phase explicitly")
        col = matches[0] if matches else None
        if col is None:
            raise KeyError(f"no composition column for component {component}")
        return self.values_in(col, "1")

    def to_dict(self) -> dict[str, list[float]]:
        """The table as ``{label: [values...]}`` (duplicate labels get a suffix)."""
        out: dict[str, list[float]] = {}
        for c in self.columns:
            key = c.label
            n = 2
            while key in out:
                key = f"{c.label} #{n}"
                n += 1
            out[key] = list(self.values(c))
        return out


@dataclass(frozen=True)
class ThermoMLData:
    """A parsed ThermoML document: its compounds, datasets, and citation."""

    compounds: tuple[Compound, ...]
    datasets: tuple[Dataset, ...]
    citation: str | None = None
    doi: str | None = None
    authors: tuple[str, ...] = ()
    source_type: str | None = None
    sha256: str = ""

    def compound(self, org_num: int) -> Compound:
        """The compound with the given ``org_num`` (raises ``KeyError`` if absent)."""
        for c in self.compounds:
            if c.org_num == org_num:
                return c
        raise KeyError(f"no compound with org_num {org_num}")

    def component_names(self, dataset: Dataset) -> list[str]:
        """Best-effort names of a dataset's components (falls back to ``C{org_num}``)."""
        names = []
        for org in dataset.components:
            try:
                c = self.compound(org)
            except KeyError:
                names.append(f"C{org}")
                continue
            names.append(c.name or c.formula or f"C{org}")
        return names


def _parse_compound(elem: ET.Element) -> Compound | None:
    org = _first_text(elem, "nOrgNum")
    if org is None:
        return None
    name = _first_text(elem, "sCommonName") or _first_text(elem, "sIUPACName")
    formula = _first_text(elem, "sFormulaMolec")
    inchikey = _first_text(elem, "sStandardInChIKey")
    cas = _first_text(elem, "nCASRegistryNum") or _first_text(elem, "sCASName")
    if cas is None:
        # Fall back to a regex scan of all descendant text (CAS numbers are unique
        # enough to spot without binding to a specific element name).
        for sub in elem.iter():
            m = _CAS_RE.search(sub.text or "")
            if m:
                cas = m.group(0)
                break
    return Compound(
        org_num=int(org),
        name=name,
        formula=formula,
        cas=cas,
        inchikey=inchikey,
        inchi=_first_text(elem, "sStandardInChI"),
    )


def _parse_variable(elem: ET.Element) -> Column | None:
    num = _first_text(elem, "nVarNumber")
    vtype = _find(elem, "VariableType")
    if num is None or vtype is None:
        return None
    type_elem = next(iter(vtype), None)
    if type_elem is None:
        return None
    component = _first_text(elem, "nOrgNum")
    return Column(
        number=int(num),
        role="variable",
        kind=_local(type_elem.tag),
        label=(_text(type_elem) or _local(type_elem.tag)),
        component=int(component) if component is not None else None,
        phase=_first_text(elem, "eVarPhase"),
    )


def _parse_property(elem: ET.Element) -> Column | None:
    num = _first_text(elem, "nPropNumber")
    if num is None:
        return None
    name_elem = _find(elem, "ePropName")
    # A composition-style property may carry the component it refers to.
    reg_text = _text(_find(elem, "nOrgNum"))
    component = int(reg_text) if reg_text else None
    return Column(
        number=int(num),
        role="property",
        kind=(_local(name_elem.tag) if name_elem is not None else "eProperty"),
        label=(_text(name_elem) or "Property") if name_elem is not None else "Property",
        component=component,
        phase=_first_text(elem, "ePropPhase"),
        presentation=_first_text(elem, "ePresentation") or "Direct value, X",
        method=_first_text(elem, "eMethodName"),
    )


def _parse_num_values(
    elem: ET.Element, var_by_num: dict[int, int], prop_by_num: dict[int, int], width: int
) -> tuple[float, ...] | None:
    row = [float("nan")] * width
    seen = False
    assigned: set[int] = set()
    for vv in _find_direct(elem, "VariableValue"):
        n = _first_text(vv, "nVarNumber")
        val = _first_text(vv, "nVarValue")
        if n is None or val is None or int(n) not in var_by_num:
            raise ValueError("incomplete or undeclared variable value")
        index = var_by_num[int(n)]
        if index in assigned:
            raise ValueError(f"duplicate value for variable {n}")
        assigned.add(index)
        row[index] = float(val)
        seen = True
    for pv in _find_direct(elem, "PropertyValue"):
        n = _first_text(pv, "nPropNumber")
        val = _first_text(pv, "nPropValue")
        if n is None or val is None or int(n) not in prop_by_num:
            raise ValueError("incomplete or undeclared property value")
        index = prop_by_num[int(n)]
        if index in assigned:
            raise ValueError(f"duplicate value for property {n}")
        assigned.add(index)
        row[index] = float(val)
        seen = True
    return tuple(row) if seen else None


def _uncertainty(elem: ET.Element, metadata: ET.Element | None = None) -> Uncertainty | None:
    # Match an uncertainty assessment by ID before reading its confidence or
    # coverage factor. Taking the first declaration can attach another method's
    # uncertainty to the current measurement.
    assessment_tag = (
        "nCombUncertAssessNum"
        if _find(elem, "nCombUncertAssessNum") is not None
        else "nUncertAssessNum"
    )
    assessment = _first_text(elem, assessment_tag)
    if metadata is not None and assessment is not None:
        matches = [child for child in metadata if _first_text(child, assessment_tag) == assessment]
        if len(matches) > 1:
            raise ValueError("ambiguous uncertainty assessment")
        metadata = matches[0] if matches else None
    blocks = [child for child in elem if "Uncertainty" in _local(child.tag)]
    if len(blocks) > 1:
        raise ValueError("multiple uncertainty assessments per value aren't supported")

    def number(*names: str) -> float | None:
        for parent in (elem, metadata):
            if parent is None:
                continue
            for name in names:
                raw = _first_text(parent, name)
                if raw is not None:
                    value = float(raw)
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"invalid {name}: {raw}")
                    return value
        return None

    standard = number("nStdUncertValue", "nCombStdUncertValue", "nVarStdUncertValue")
    expanded = number("nExpandUncertValue", "nCombExpandUncertValue", "nVarExpandUncertValue")
    if standard is None and expanded is None:
        return None
    coverage = number("nCoverageFactor", "nCombCoverageFactor", "nUncertCoverageFactor")
    if coverage == 0:
        raise ValueError("uncertainty coverage factor must be positive")
    confidence = number("nCombUncertLevOfConfid", "nUncertLevOfConfid")
    if confidence is not None and not 0 < confidence < 100:
        raise ValueError("uncertainty confidence must be between zero and 100")
    return Uncertainty(
        standard,
        expanded,
        coverage,
        confidence,
        _first_text(elem, "nCombUncertAssessNum") or _first_text(elem, "nUncertAssessNum"),
    )


def _constraint(elem: ET.Element, number: int) -> tuple[Column, float]:
    ctype = _find(elem, "ConstraintType")
    if ctype is None or len(ctype) != 1:
        raise ValueError("unsupported constraint type")
    child = ctype[0]
    raw = _first_text(elem, "nConstraintValue")
    if raw is None:
        raise ValueError("constraint has no value")
    org = _first_text(elem, "nOrgNum")
    return Column(
        number,
        "constraint",
        _local(child.tag),
        _text(child) or "",
        int(org) if org else None,
        _first_text(elem, "eConstraintPhase"),
    ), float(raw)


def _parse_dataset(elem: ET.Element) -> Dataset | None:
    components = tuple(
        int(t)
        for comp in _find_direct(elem, "Component")
        if (t := _first_text(comp, "nOrgNum")) is not None
    )
    variables = [
        col for v in _find_direct(elem, "Variable") if (col := _parse_variable(v)) is not None
    ]
    properties = [
        col for p in _find_direct(elem, "Property") if (col := _parse_property(p)) is not None
    ]
    constraints = [_constraint(c, i + 1) for i, c in enumerate(_find_direct(elem, "Constraint"))]
    columns = tuple(variables + properties + [col for col, _ in constraints])
    if not columns:
        return None

    var_by_num = {col.number: i for i, col in enumerate(variables)}
    prop_by_num = {col.number: len(variables) + i for i, col in enumerate(properties)}
    if len(var_by_num) != len(variables) or len(prop_by_num) != len(properties):
        raise ValueError("duplicate variable or property number")
    width = len(columns)
    rows = []
    uncertainties = []
    metadata = {
        int(_first_text(p, "nPropNumber") or "0"): p for p in _find_direct(elem, "Property")
    }
    for nv in _find_direct(elem, "NumValues"):
        row = _parse_num_values(nv, var_by_num, prop_by_num, width)
        if row is None:
            continue
        values = list(row)
        unc: list[Uncertainty | None] = [None] * width
        for i, (_, value) in enumerate(constraints, start=len(variables) + len(properties)):
            values[i] = value
        for item in _find_direct(nv, "PropertyValue"):
            num = int(_first_text(item, "nPropNumber") or "0")
            if num in prop_by_num:
                unc[prop_by_num[num]] = _uncertainty(item, metadata.get(num))
        for item in _find_direct(nv, "VariableValue"):
            num = int(_first_text(item, "nVarNumber") or "0")
            if num in var_by_num:
                unc[var_by_num[num]] = _uncertainty(item)
        rows.append(tuple(values))
        uncertainties.append(tuple(unc))

    phases = tuple(
        p
        for child in _find_direct(elem, "PhaseID")
        if (p := _first_text(child, "ePhase")) is not None
    )
    phase = phases[0] if phases else None
    number = _first_text(elem, "nPureOrMixtureDataNumber")
    issues = []
    if len(variables) != len(_find_direct(elem, "Variable")):
        issues.append("unsupported or incomplete variable declaration")
    if len(properties) != len(_find_direct(elem, "Property")):
        issues.append("unsupported or incomplete property declaration")
    return Dataset(
        components=components,
        columns=columns,
        rows=tuple(rows),
        phase=phase,
        number=int(number) if number is not None else None,
        phases=phases,
        uncertainties=tuple(uncertainties),
        issues=tuple(issues),
    )


def _parse_root(root: ET.Element) -> ThermoMLData:
    compounds: list[Compound] = []
    datasets: list[Dataset] = []
    citation: str | None = None
    doi = None
    authors: tuple[str, ...] = ()
    source_type = None
    for elem in root.iter():
        name = _local(elem.tag)
        if name == "Compound":
            c = _parse_compound(elem)
            if c is not None:
                compounds.append(c)
        elif name == "PureOrMixtureData":
            d = _parse_dataset(elem)
            if d is not None:
                datasets.append(d)
        elif name == "Citation" and citation is None:
            citation = _first_text(elem, "sTitle") or _first_text(elem, "sAbstract")
            doi = _first_text(elem, "sDOI")
            authors = tuple(v for child in _find_direct(elem, "sAuthor") if (v := _text(child)))
            source_type = _first_text(elem, "eSourceType")
    if len({c.org_num for c in compounds}) != len(compounds):
        raise ValueError("duplicate compound organization number")
    return ThermoMLData(
        compounds=tuple(compounds),
        datasets=tuple(datasets),
        citation=citation,
        doi=doi,
        authors=authors,
        source_type=source_type,
    )


def loads(text: str | bytes) -> ThermoMLData:
    """Parse a ThermoML document from an in-memory string or bytes."""
    from dataclasses import replace

    raw = text.encode() if isinstance(text, str) else text
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("ThermoML entity declarations aren't supported")
    root = ET.fromstring(raw)
    return replace(_parse_root(root), sha256=hashlib.sha256(raw).hexdigest())


def read_thermoml(source: str | Path | IO[bytes] | IO[str]) -> ThermoMLData:
    """Parse a ThermoML document from a path or open file object.

    Args:
        source: A filesystem path (``str``/`Path`) or a readable
            file object containing ThermoML XML.

    Returns:
        The parsed `ThermoMLData`.
    """
    if isinstance(source, str | Path):
        return loads(Path(source).read_bytes())
    return loads(source.read())


def _samples_dir() -> Path:
    return Path(str(files("fugacio.thermo").joinpath("thermoml_samples")))


def list_samples() -> list[str]:
    """Names (without extension) of the bundled ThermoML sample datasets."""
    return sorted(p.stem for p in _samples_dir().glob("*.xml"))


def sample_path(name: str) -> Path:
    """Filesystem path of a bundled sample (with or without the ``.xml`` suffix)."""
    stem = name[:-4] if name.endswith(".xml") else name
    if Path(stem).name != stem or stem in (".", ".."):
        raise ValueError("sample name must be a plain filename")
    path = _samples_dir() / f"{stem}.xml"
    if not path.exists():
        available = ", ".join(list_samples()) or "(none)"
        raise FileNotFoundError(f"no bundled ThermoML sample {name!r}; available: {available}")
    return path


def load_sample(name: str) -> ThermoMLData:
    """Parse a bundled ThermoML sample by name (see `list_samples`)."""
    from dataclasses import replace

    return replace(read_thermoml(sample_path(name)), source_type="synthetic")
