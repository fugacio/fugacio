"""Generate extended `_comp(...)` database entries from the open `chemicals` data.

This is a *one-shot authoring tool*, not part of the package. It pulls critical
constants, acentric factors, boiling points, molar masses, and formation
enthalpies from the open-source ``chemicals`` dataset, transcribes the Antoine
vapour-pressure constants (shifting the base from Pa to bar), and least-squares
fits the database's ideal-gas heat-capacity polynomial to the ``chemicals``
Poling correlation. It then injects the formatted entries into ``components.py``
just before the close of the ``_COMPONENTS`` tuple.

Run once with ``uv run python scripts/gen_components.py`` (requires the optional
``chemicals`` package). The emitted values are plain reference data baked into
the source tree; the dependency is not needed at runtime.
"""

from __future__ import annotations

from collections.abc import Callable

import chemicals as ch
import numpy as np
from chemicals import heat_capacity as hc
from chemicals import vapor_pressure as vp

hc._load_Cp_data()

R = 8.314462618

COMPONENTS_PY = "packages/fugacio-thermo/src/fugacio/thermo/components.py"
SENTINEL = "# --- Extended set (generated from the open `chemicals` dataset) ---"

# (canonical name, formula, CAS), grouped into the section headers we emit.
SECTIONS: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "More light gases & inorganics",
        [
            ("helium", "He", "7440-59-7"),
            ("neon", "Ne", "7440-01-9"),
            ("sulfur dioxide", "SO2", "7446-09-5"),
            ("nitrous oxide", "N2O", "10024-97-2"),
            ("nitric oxide", "NO", "10102-43-9"),
            ("carbon disulfide", "CS2", "75-15-0"),
            ("chlorine", "Cl2", "7782-50-5"),
            ("hydrogen chloride", "HCl", "7647-01-0"),
        ],
    ),
    (
        "More alkanes",
        [
            ("n-nonane", "C9H20", "111-84-2"),
            ("n-decane", "C10H22", "124-18-5"),
            ("n-undecane", "C11H24", "1120-21-4"),
            ("n-dodecane", "C12H26", "112-40-3"),
            ("isopentane", "C5H12", "78-78-4"),
            ("neopentane", "C5H12", "463-82-1"),
            ("2-methylpentane", "C6H14", "107-83-5"),
            ("isooctane", "C8H18", "540-84-1"),
        ],
    ),
    (
        "More cycloalkanes",
        [
            ("cyclopentane", "C5H10", "287-92-3"),
            ("methylcyclopentane", "C6H12", "96-37-7"),
            ("methylcyclohexane", "C7H14", "108-87-2"),
        ],
    ),
    (
        "More alkenes & alkynes",
        [
            ("1-butene", "C4H8", "106-98-9"),
            ("cis-2-butene", "C4H8", "590-18-1"),
            ("trans-2-butene", "C4H8", "624-64-6"),
            ("isobutylene", "C4H8", "115-11-7"),
            ("1-pentene", "C5H10", "109-67-1"),
            ("1-hexene", "C6H12", "592-41-6"),
            ("1,3-butadiene", "C4H6", "106-99-0"),
            ("isoprene", "C5H8", "78-79-5"),
            ("acetylene", "C2H2", "74-86-2"),
        ],
    ),
    (
        "More aromatics",
        [
            ("ethylbenzene", "C8H10", "100-41-4"),
            ("o-xylene", "C8H10", "95-47-6"),
            ("m-xylene", "C8H10", "108-38-3"),
            ("p-xylene", "C8H10", "106-42-3"),
            ("styrene", "C8H8", "100-42-5"),
            ("cumene", "C9H12", "98-82-8"),
            ("phenol", "C6H6O", "108-95-2"),
            ("aniline", "C6H7N", "62-53-3"),
            ("naphthalene", "C10H8", "91-20-3"),
            ("pyridine", "C5H5N", "110-86-1"),
        ],
    ),
    (
        "More alcohols & glycols",
        [
            ("1-propanol", "C3H8O", "71-23-8"),
            ("1-butanol", "C4H10O", "71-36-3"),
            ("2-butanol", "C4H10O", "78-92-2"),
            ("isobutanol", "C4H10O", "78-83-1"),
            ("tert-butanol", "C4H10O", "75-65-0"),
            ("1-pentanol", "C5H12O", "71-41-0"),
            ("1-hexanol", "C6H14O", "111-27-3"),
            ("ethylene glycol", "C2H6O2", "107-21-1"),
            ("propylene glycol", "C3H8O2", "57-55-6"),
            ("glycerol", "C3H8O3", "56-81-5"),
            ("cyclohexanol", "C6H12O", "108-93-0"),
        ],
    ),
    (
        "Ethers",
        [
            ("dimethyl ether", "C2H6O", "115-10-6"),
            ("diethyl ether", "C4H10O", "60-29-7"),
            ("mtbe", "C5H12O", "1634-04-4"),
            ("tetrahydrofuran", "C4H8O", "109-99-9"),
            ("1,4-dioxane", "C4H8O2", "123-91-1"),
        ],
    ),
    (
        "Ketones & aldehydes",
        [
            ("2-butanone", "C4H8O", "78-93-3"),
            ("mibk", "C6H12O", "108-10-1"),
            ("acetaldehyde", "C2H4O", "75-07-0"),
            ("formaldehyde", "CH2O", "50-00-0"),
            ("cyclohexanone", "C6H10O", "108-94-1"),
        ],
    ),
    (
        "Esters",
        [
            ("methyl acetate", "C3H6O2", "79-20-9"),
            ("ethyl acetate", "C4H8O2", "141-78-6"),
            ("n-butyl acetate", "C6H12O2", "123-86-4"),
            ("methyl formate", "C2H4O2", "107-31-3"),
            ("vinyl acetate", "C4H6O2", "108-05-4"),
        ],
    ),
    (
        "Carboxylic acids",
        [
            ("formic acid", "CH2O2", "64-18-6"),
            ("acetic acid", "C2H4O2", "64-19-7"),
            ("propionic acid", "C3H6O2", "79-09-4"),
            ("acrylic acid", "C3H4O2", "79-10-7"),
        ],
    ),
    (
        "Halogenated",
        [
            ("dichloromethane", "CH2Cl2", "75-09-2"),
            ("chloroform", "CHCl3", "67-66-3"),
            ("carbon tetrachloride", "CCl4", "56-23-5"),
            ("1,2-dichloroethane", "C2H4Cl2", "107-06-2"),
            ("chlorobenzene", "C6H5Cl", "108-90-7"),
            ("chloromethane", "CH3Cl", "74-87-3"),
            ("trichloroethylene", "C2HCl3", "79-01-6"),
            ("vinyl chloride", "C2H3Cl", "75-01-4"),
        ],
    ),
    (
        "Refrigerants",
        [
            ("r134a", "C2H2F4", "811-97-2"),
            ("r22", "CHClF2", "75-45-6"),
            ("r12", "CCl2F2", "75-71-8"),
            ("r32", "CH2F2", "75-10-5"),
            ("r152a", "C2H4F2", "75-37-6"),
        ],
    ),
    (
        "Nitrogen compounds",
        [
            ("acetonitrile", "C2H3N", "75-05-8"),
            ("nitromethane", "CH3NO2", "75-52-5"),
            ("methylamine", "CH5N", "74-89-5"),
            ("dimethylamine", "C2H7N", "124-40-3"),
            ("trimethylamine", "C3H9N", "75-50-3"),
            ("triethylamine", "C6H15N", "121-44-8"),
            ("dimethylformamide", "C3H7NO", "68-12-2"),
            ("nmp", "C5H9NO", "872-50-4"),
            ("monoethanolamine", "C2H7NO", "141-43-5"),
            ("diethanolamine", "C4H11NO2", "111-42-2"),
        ],
    ),
    (
        "Sulfur compounds & miscellaneous",
        [
            ("dimethyl sulfoxide", "C2H6OS", "67-68-5"),
            ("dimethyl sulfide", "C2H6S", "75-18-3"),
            ("ethylene oxide", "C2H4O", "75-21-8"),
            ("propylene oxide", "C3H6O", "75-56-9"),
            ("furan", "C4H4O", "110-00-9"),
            ("furfural", "C5H4O2", "98-01-1"),
        ],
    ),
]


def _try(fn: Callable[..., float | None], *a: str) -> float | None:
    try:
        return fn(*a)
    except Exception:
        return None


def fit_cp(cas: str) -> tuple[float, float, float, float, float, float] | None:
    """Least-squares fit ``Cp/R = a + bT + cT^2 + eT^3`` to the Poling correlation.

    Fits in the scaled variable ``tau = T/1000`` for numerical conditioning, then
    rescales back to SI coefficients. Returns ``None`` if the source coefficients
    are absent or non-finite (e.g. monatomic gases lacking a Poling fit).
    """
    if cas not in hc.Cp_data_Poling.index:
        return None
    row = hc.Cp_data_Poling.loc[cas]
    coeffs = []
    for i in range(5):
        v = float(row[f"a{i}"])
        coeffs.append(0.0 if not np.isfinite(v) else v)
    tmin = float(row["Tmin"])
    tmax = float(row["Tmax"])
    lo = max(tmin, 200.0)
    hi = min(tmax, 1500.0)
    if hi - lo < 100.0:
        lo, hi = tmin, max(tmax, tmin + 300.0)
    temps = np.linspace(lo, hi, 80)
    cp_over_r = np.array([hc.Poling(t, *coeffs) for t in temps]) / R
    if not np.all(np.isfinite(cp_over_r)):
        return None
    tau = temps / 1000.0
    e_s, c_s, b_s, a = (float(v) for v in np.polyfit(tau, cp_over_r, 3))
    b = b_s / 1e3
    c = c_s / 1e6
    e = e_s / 1e9
    return a, b, c, e, lo, hi


def antoine(cas: str) -> tuple[float, float, float, float, float] | None:
    """Transcribe Antoine constants (Pa -> bar via ``A -= 5``)."""
    if cas not in vp.Psat_data_AntoinePoling.index:
        return None
    row = vp.Psat_data_AntoinePoling.loc[cas]
    a = float(row["A"]) - 5.0
    return a, float(row["B"]), float(row["C"]), float(row["Tmin"]), float(row["Tmax"])


def fnum(x: float, nd: int) -> str:
    return f"{round(float(x), nd)!r}"


def emit(name: str, formula: str, cas: str) -> str | None:
    tc = _try(ch.Tc, cas)
    pc = _try(ch.Pc, cas)
    omega = _try(ch.omega, cas)
    if tc is None or pc is None or omega is None:
        print(f"  SKIP {name}: missing Tc/Pc/omega")
        return None
    mw = _try(ch.MW, cas)
    tb = _try(ch.Tb, cas)
    vc = _try(ch.Vc, cas)
    # Store the *derived* critical compressibility Zc = Pc Vc / (R Tc) so each record
    # is internally consistent: `chemicals` sources Vc and Zc independently and they
    # can disagree by a few percent. Fall back to the tabulated Zc when Vc is missing.
    zc = pc * vc / (R * tc) if vc is not None else _try(ch.Zc, cas)
    hf = _try(ch.Hfg, cas)

    lines = [
        "    _comp(",
        f'        "{name}",',
        f'        formula="{formula}",',
        f'        cas="{cas}",',
        f"        mw={fnum(mw, 3)},",
        f"        tc={fnum(tc, 2)},",
        f"        pc_bar={fnum(pc / 1e5, 4)},",
        f"        omega={fnum(omega, 4)},",
    ]
    if tb is not None:
        lines.append(f"        tb={fnum(tb, 2)},")
    if vc is not None:
        lines.append(f"        vc_cm3={fnum(vc * 1e6, 1)},")
    if zc is not None:
        lines.append(f"        zc={fnum(zc, 3)},")
    ant = antoine(cas)
    if ant is not None:
        a, b, c, tmn, tmx = ant
        lines.append(
            f"        antoine=_ant({fnum(a, 5)}, {fnum(b, 4)}, {fnum(c, 4)}, "
            f"{fnum(tmn, 1)}, {fnum(tmx, 1)}),"
        )
    cp = fit_cp(cas)
    if cp is not None:
        a, b, c, e, tmn, tmx = cp
        lines.append(
            f"        cp_ig=_cpp({a:.6g}, {b:.6g}, {c:.6g}, {e:.6g}, "
            f"{fnum(tmn, 1)}, {fnum(tmx, 1)}),"
        )
    if hf is not None:
        lines.append(f"        hform_ig={fnum(hf, 1)},")
    lines.append("    ),")
    return "\n".join(lines)


def main() -> None:
    with open(COMPONENTS_PY) as f:
        src = f.read()
    if SENTINEL in src:
        print("Sentinel already present; aborting to avoid double insertion.")
        return

    import fugacio.thermo.components as comp

    existing = set(comp.DATABASE)

    blocks: list[str] = [f"    {SENTINEL}"]
    n_added = 0
    for header, items in SECTIONS:
        section_blocks: list[str] = []
        for name, formula, cas in items:
            if name in existing:
                print(f"  skip {name}: already in database")
                continue
            block = emit(name, formula, cas)
            if block is not None:
                section_blocks.append(block)
                n_added += 1
        if section_blocks:
            dashes = "-" * max(3, 72 - len(header))
            blocks.append(f"    # --- {header} {dashes}")
            blocks.extend(section_blocks)

    generated = "\n".join(blocks) + "\n"
    anchor = "    ),\n)\n\n#: Canonical-name"
    if anchor not in src:
        raise SystemExit("anchor not found; cannot inject")
    src = src.replace(anchor, "    ),\n" + generated + ")\n\n#: Canonical-name", 1)
    with open(COMPONENTS_PY, "w") as f:
        f.write(src)
    print(f"Injected {n_added} components into {COMPONENTS_PY}")


if __name__ == "__main__":
    main()
