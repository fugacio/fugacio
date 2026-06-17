"""Generate the bundled ThermoML sample datasets (beyond the two hand-written ones).

Each sample is an isothermal binary P-x (bubble-pressure) table computed from a
two-parameter NRTL model (``tau = b/T``, fixed ``alpha``) through Fugacio's own
gamma-phi bubble-point solver, then serialized in the NIST/IUPAC ThermoML
schema. The parameter values are chosen to reproduce the qualitative behaviour
of well-studied systems (infinite-dilution activity coefficients in the right
range, azeotropes where the real pair has one), and the generating parameters
are recorded in the file's citation block, so the data are *synthetic but
honest*: regression tests can recover the exact parameters, and the reader is
exercised on schema-faithful files.

Run from the repo root:

    uv run python scripts/gen_thermoml_samples.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from fugacio.thermo.activity.models import NRTL  # noqa: E402
from fugacio.thermo.components import component_arrays, get  # noqa: E402
from fugacio.thermo.gammaphi import bubble_pressure_gamma  # noqa: E402

OUT_DIR = Path("packages/fugacio-thermo/src/fugacio/thermo/thermoml_samples")

#: InChI strings keyed by component name (the component DB stores CAS/formula).
INCHI: dict[str, tuple[str, str]] = {
    "methanol": ("InChI=1S/CH4O/c1-2/h2H,1H3", "OKKJLVBELUTLKV-UHFFFAOYSA-N"),
    "water": ("InChI=1S/H2O/h1H2", "XLYOFNOQVPJJNP-UHFFFAOYSA-N"),
    "acetone": ("InChI=1S/C3H6O/c1-3(2)4/h1-2H3", "CSCPPACGZOOCGX-UHFFFAOYSA-N"),
    "chloroform": ("InChI=1S/CHCl3/c2-1(3)4/h1H", "HEDRZPFGACZZDS-UHFFFAOYSA-N"),
    "benzene": ("InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H", "UHOVQNZJYSORNB-UHFFFAOYSA-N"),
    "toluene": ("InChI=1S/C7H8/c1-7-5-3-2-4-6-7/h2-6H,1H3", "YXFVVABEGXRONW-UHFFFAOYSA-N"),
    "2-propanol": ("InChI=1S/C3H8O/c1-3(2)4/h3-4H,1-2H3", "KFZMGEQAYNKOFK-UHFFFAOYSA-N"),
}


@dataclass(frozen=True)
class Sample:
    """One synthetic isothermal binary VLE dataset."""

    stem: str
    components: tuple[str, str]
    t: float  # K
    b12: float  # NRTL tau = b/T coefficients (K)
    b21: float
    alpha: float = 0.3
    note: str = ""


SAMPLES = [
    Sample(
        stem="methanol_water_vle_338K",
        components=("methanol", "water"),
        t=338.15,
        b12=80.0,
        b21=220.0,
        note="positive deviations, no azeotrope at this temperature",
    ),
    Sample(
        stem="acetone_chloroform_vle_318K",
        components=("acetone", "chloroform"),
        t=318.15,
        b12=-150.0,
        b21=-130.0,
        note="negative deviations with a pressure-minimum azeotrope",
    ),
    Sample(
        stem="benzene_toluene_vle_363K",
        components=("benzene", "toluene"),
        t=363.15,
        b12=20.0,
        b21=15.0,
        note="nearly ideal solution",
    ),
    Sample(
        stem="isopropanol_water_vle_353K",
        components=("2-propanol", "water"),
        t=353.15,
        b12=170.0,
        b21=680.0,
        note="strong positive deviations with a pressure-maximum azeotrope",
    ),
]

X1_GRID = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def _compound_block(org_num: int, name: str) -> str:
    comp = get(name)
    inchi, inchikey = INCHI[name]
    return (
        "  <Compound>\n"
        "    <RegNum>\n"
        f"      <nOrgNum>{org_num}</nOrgNum>\n"
        "    </RegNum>\n"
        f"    <sCommonName>{name}</sCommonName>\n"
        f"    <sFormulaMolec>{comp.formula}</sFormulaMolec>\n"
        f"    <sStandardInChI>{inchi}</sStandardInChI>\n"
        f"    <sStandardInChIKey>{inchikey}</sStandardInChIKey>\n"
        f"    <nCASRegistryNum>{comp.cas}</nCASRegistryNum>\n"
        "  </Compound>\n"
    )


def _num_values_block(t: float, x1: float, p_kpa: float) -> str:
    return (
        "    <NumValues>\n"
        f"      <VariableValue><nVarNumber>1</nVarNumber><nVarValue>{t:.2f}</nVarValue>"
        "</VariableValue>\n"
        f"      <VariableValue><nVarNumber>2</nVarNumber><nVarValue>{x1:.2f}</nVarValue>"
        "</VariableValue>\n"
        f"      <PropertyValue><nPropNumber>1</nPropNumber><nPropValue>{p_kpa:.4f}</nPropValue>"
        "</PropertyValue>\n"
        "    </NumValues>\n"
    )


def _pressures(sample: Sample) -> list[float]:
    """Bubble pressures (kPa) over the composition grid from the NRTL model."""
    arr = component_arrays(list(sample.components))
    zeros = jnp.zeros((2, 2))
    model = NRTL(
        a=zeros,
        b=jnp.array([[0.0, sample.b12], [sample.b21, 0.0]]),
        alpha=jnp.array([[0.0, sample.alpha], [sample.alpha, 0.0]]),
        e=zeros,
    )
    out = []
    for x1 in X1_GRID:
        x = jnp.array([x1, 1.0 - x1])
        p, _y = bubble_pressure_gamma(model, sample.t, x, arr["tc"], arr["pc"], arr["omega"])
        out.append(float(p) / 1.0e3)
    return out


def _document(sample: Sample) -> str:
    c1, c2 = sample.components
    pressures = _pressures(sample)
    rows = "".join(
        _num_values_block(sample.t, x1, p) for x1, p in zip(X1_GRID, pressures, strict=True)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!--\n"
        "  Illustrative ThermoML sample bundled with fugacio.thermo for tests and"
        " examples.\n"
        f"  Isothermal ({sample.t:.2f} K) bubble-pressure (P-x) data for {c1}(1) +"
        f" {c2}(2).\n"
        f"  Values are synthetic, generated from a two-parameter NRTL (tau = b/T,\n"
        f"  alpha = {sample.alpha}, b12 = {sample.b12:g} K, b21 = {sample.b21:g} K)"
        " through Fugacio's\n"
        "  gamma-phi bubble-point solver, but the file follows the NIST/IUPAC"
        " ThermoML\n"
        f"  schema so it exercises the real reader ({sample.note}).\n"
        "  It is NOT a measured dataset; for real data use the NIST ThermoML"
        " Archive.\n"
        "  Regenerate with: uv run python scripts/gen_thermoml_samples.py\n"
        "-->\n"
        '<DataReport xmlns="http://www.iupac.org/namespaces/ThermoML">\n'
        "  <Version>\n"
        "    <nVersionMajor>4</nVersionMajor>\n"
        "    <nVersionMinor>0</nVersionMinor>\n"
        "  </Version>\n"
        "  <Citation>\n"
        f"    <sTitle>Synthetic isothermal VLE for {c1} + {c2} at {sample.t:.2f} K"
        " (illustrative)</sTitle>\n"
        f"    <sAbstract>Bundled example dataset for the Fugacio ThermoML reader;"
        f" generated from NRTL with alpha = {sample.alpha}, b12 = {sample.b12:g} K,"
        f" b21 = {sample.b21:g} K.</sAbstract>\n"
        "  </Citation>\n"
        + _compound_block(1, c1)
        + _compound_block(2, c2)
        + "  <PureOrMixtureData>\n"
        "    <nPureOrMixtureDataNumber>1</nPureOrMixtureDataNumber>\n"
        "    <Component>\n"
        "      <RegNum>\n"
        "        <nOrgNum>1</nOrgNum>\n"
        "      </RegNum>\n"
        "    </Component>\n"
        "    <Component>\n"
        "      <RegNum>\n"
        "        <nOrgNum>2</nOrgNum>\n"
        "      </RegNum>\n"
        "    </Component>\n"
        "    <PhaseID>\n"
        "      <ePhase>Liquid</ePhase>\n"
        "    </PhaseID>\n"
        "    <Variable>\n"
        "      <nVarNumber>1</nVarNumber>\n"
        "      <VariableID>\n"
        "        <VariableType>\n"
        "          <eTemperature>Temperature, K</eTemperature>\n"
        "        </VariableType>\n"
        "      </VariableID>\n"
        "    </Variable>\n"
        "    <Variable>\n"
        "      <nVarNumber>2</nVarNumber>\n"
        "      <VariableID>\n"
        "        <VariableType>\n"
        "          <eComponentComposition>Mole fraction</eComponentComposition>\n"
        "        </VariableType>\n"
        "        <RegNum>\n"
        "          <nOrgNum>1</nOrgNum>\n"
        "        </RegNum>\n"
        "      </VariableID>\n"
        "    </Variable>\n"
        "    <Property>\n"
        "      <nPropNumber>1</nPropNumber>\n"
        "      <Property-MethodID>\n"
        "        <PropertyGroup>\n"
        "          <PhaseTransition>\n"
        "            <ePropName>Pressure, kPa</ePropName>\n"
        "          </PhaseTransition>\n"
        "        </PropertyGroup>\n"
        "      </Property-MethodID>\n"
        "    </Property>\n" + rows + "  </PureOrMixtureData>\n"
        "</DataReport>\n"
    )


def main() -> None:
    for sample in SAMPLES:
        path = OUT_DIR / f"{sample.stem}.xml"
        path.write_text(_document(sample))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
