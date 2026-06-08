"""Reader for the NIST ThermoML archive XML format.

`ThermoML <https://www.nist.gov/mml/acmd/trc/thermoml>`_ is the IUPAC/NIST XML
standard for thermophysical and thermochemical property data; the freely
redistributable `ThermoML Archive
<https://www.nist.gov/mml/acmd/trc/thermoml/thermoml-archive>`_ holds tens of
thousands of experimental datasets. This module turns those files into tidy,
typed tables you can feed straight into :mod:`fugacio.thermo.regression` -- so a
model can be fitted to *real measurements*, and predictions graded against them.

The parser is deliberately tolerant and dependency-free (standard-library
:mod:`xml.etree.ElementTree` only):

* XML namespaces are stripped, so files declaring the ThermoML namespace (or none)
  parse identically;
* compounds, mixtures, variables, properties, and the numeric value rows are read
  by their *local* element names, matching the published schema without binding to
  a specific version;
* each :class:`Dataset` exposes its columns as aligned numeric rows plus
  convenience accessors (:meth:`Dataset.temperature`, :meth:`Dataset.pressure`,
  :meth:`Dataset.mole_fraction`) with pressure unit conversion to pascal.

A couple of small, schema-faithful datasets ship with the package for tests and
examples; see :func:`list_samples` / :func:`load_sample`.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class Column:
    """One variable or property column of a :class:`Dataset` table.

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

    @property
    def labels(self) -> tuple[str, ...]:
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
        self, *, kind: str | None = None, quantity: str | None = None, component: int | None = None
    ) -> Column | None:
        """First column matching the given filters (any combination)."""
        for c in self.columns:
            if kind is not None and c.kind != kind:
                continue
            if quantity is not None and c.quantity.lower() != quantity.lower():
                continue
            if component is not None and c.component != component:
                continue
            return c
        return None

    def temperature(self) -> tuple[float, ...]:
        """Temperatures in kelvin (raises if the dataset has no temperature column)."""
        col = self.find_column(kind="eTemperature")
        if col is None:
            raise KeyError("dataset has no temperature column")
        return self.values(col)

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
        raw = self.values(col)
        from_pa = _PRESSURE_TO_PA.get(col.unit or "kPa", 1.0)
        to = _PRESSURE_TO_PA.get(unit, 1.0)
        return tuple(v * from_pa / to for v in raw)

    def mole_fraction(self, component: int) -> tuple[float, ...]:
        """Mole fractions of ``component`` (by ``org_num``)."""
        col = self.find_column(kind="eComponentComposition", component=component)
        if col is None:
            raise KeyError(f"no composition column for component {component}")
        return self.values(col)

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

    def compound(self, org_num: int) -> Compound:
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
    )


def _parse_num_values(
    elem: ET.Element, var_by_num: dict[int, int], prop_by_num: dict[int, int], width: int
) -> tuple[float, ...] | None:
    row = [float("nan")] * width
    seen = False
    for vv in _find_direct(elem, "VariableValue"):
        n = _first_text(vv, "nVarNumber")
        val = _first_text(vv, "nVarValue")
        if n is None or val is None or int(n) not in var_by_num:
            continue
        row[var_by_num[int(n)]] = float(val)
        seen = True
    for pv in _find_direct(elem, "PropertyValue"):
        n = _first_text(pv, "nPropNumber")
        val = _first_text(pv, "nPropValue")
        if n is None or val is None or int(n) not in prop_by_num:
            continue
        row[prop_by_num[int(n)]] = float(val)
        seen = True
    return tuple(row) if seen else None


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
    columns = tuple(variables + properties)
    if not columns:
        return None

    var_by_num = {col.number: i for i, col in enumerate(variables)}
    prop_by_num = {col.number: len(variables) + i for i, col in enumerate(properties)}
    width = len(columns)

    rows = tuple(
        row
        for nv in _find_direct(elem, "NumValues")
        if (row := _parse_num_values(nv, var_by_num, prop_by_num, width)) is not None
    )

    phase = _first_text(elem, "ePhase")
    number = _first_text(elem, "nPureOrMixtureDataNumber")
    return Dataset(
        components=components,
        columns=columns,
        rows=rows,
        phase=phase,
        number=int(number) if number is not None else None,
    )


def _parse_root(root: ET.Element) -> ThermoMLData:
    compounds: list[Compound] = []
    datasets: list[Dataset] = []
    citation: str | None = None
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
    return ThermoMLData(compounds=tuple(compounds), datasets=tuple(datasets), citation=citation)


def loads(text: str | bytes) -> ThermoMLData:
    """Parse a ThermoML document from an in-memory string or bytes."""
    root = ET.fromstring(text)
    return _parse_root(root)


def read_thermoml(source: str | Path | IO[bytes] | IO[str]) -> ThermoMLData:
    """Parse a ThermoML document from a path or open file object.

    Args:
        source: A filesystem path (``str``/:class:`~pathlib.Path`) or a readable
            file object containing ThermoML XML.

    Returns:
        The parsed :class:`ThermoMLData`.
    """
    if isinstance(source, str | Path):
        tree = ET.parse(Path(source))
        return _parse_root(tree.getroot())
    tree = ET.parse(source)
    return _parse_root(tree.getroot())


def _samples_dir() -> Path:
    return Path(str(files("fugacio.thermo").joinpath("thermoml_samples")))


def list_samples() -> list[str]:
    """Names (without extension) of the bundled ThermoML sample datasets."""
    return sorted(p.stem for p in _samples_dir().glob("*.xml"))


def sample_path(name: str) -> Path:
    """Filesystem path of a bundled sample (with or without the ``.xml`` suffix)."""
    stem = name[:-4] if name.endswith(".xml") else name
    path = _samples_dir() / f"{stem}.xml"
    if not path.exists():
        available = ", ".join(list_samples()) or "(none)"
        raise FileNotFoundError(f"no bundled ThermoML sample {name!r}; available: {available}")
    return path


def load_sample(name: str) -> ThermoMLData:
    """Parse a bundled ThermoML sample by name (see :func:`list_samples`)."""
    return read_thermoml(sample_path(name))
