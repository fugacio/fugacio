"""Generate extended `_comp(...)` database entries from the open `chemicals` data.

This is a *one-shot authoring tool*, not part of the package. It pulls critical
constants, acentric factors, boiling points, molar masses, and formation
enthalpies from the open-source ``chemicals`` dataset, transcribes the Antoine
vapour-pressure constants (shifting the base from Pa to bar), and least-squares
fits the database's ideal-gas heat-capacity polynomial to the ``chemicals``
Poling correlation. It then injects the formatted entries into ``components.py``
just before the close of the ``_COMPONENTS`` tuple.

The first wave (``SECTIONS``) was authored as explicit ``(name, formula, CAS)``
triplets; the second wave (``SECTIONS2``) lists only canonical names (plus an
optional lookup query when the canonical name is not what ``chemicals`` indexes),
and resolves CAS numbers, formulas, and molar masses through
``chemicals.search_chemical`` so nothing is transcribed by hand. Each wave is
guarded by its own sentinel comment, so re-running the script after adding a
wave only injects the missing entries.

Run with ``uv run --group oracles python scripts/gen_components.py`` (requires
the optional ``chemicals`` package). The emitted values are plain reference data
baked into the source tree; the dependency is not needed at runtime.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import chemicals as ch
import numpy as np
from chemicals import heat_capacity as hc
from chemicals import vapor_pressure as vp

hc._load_Cp_data()

R = 8.314462618

COMPONENTS_PY = "packages/fugacio-thermo/src/fugacio/thermo/components.py"
SENTINEL = "# --- Extended set (generated from the open `chemicals` dataset) ---"
SENTINEL2 = "# --- Extended set 2 (generated from the open `chemicals` dataset) ---"

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


# Wave 2: canonical name, or (canonical name, chemicals lookup query). Organized
# by family; entries that cannot be resolved (or lack Tc/Pc/omega) are skipped
# with a message, so this list errs on the side of inclusion.
Entry = str | tuple[str, str]

SECTIONS2: list[tuple[str, list[Entry]]] = [
    (
        "Long n-alkanes & waxes",
        [
            "n-tridecane",
            "n-tetradecane",
            "n-pentadecane",
            "n-hexadecane",
            "n-heptadecane",
            "n-octadecane",
            "n-nonadecane",
            "n-eicosane",
            "n-docosane",
            "n-tetracosane",
            "n-octacosane",
            "n-triacontane",
            "squalane",
        ],
    ),
    (
        "Branched alkanes",
        [
            "3-methylpentane",
            "2,2-dimethylbutane",
            "2,3-dimethylbutane",
            "2-methylhexane",
            "3-methylhexane",
            "2,3-dimethylpentane",
            "2,4-dimethylpentane",
            "2-methylheptane",
            "2,5-dimethylhexane",
            "2,3,4-trimethylpentane",
        ],
    ),
    (
        "Cycloalkanes",
        [
            "cyclopropane",
            "cyclobutane",
            "cycloheptane",
            "cyclooctane",
            "ethylcyclopentane",
            "ethylcyclohexane",
            ("cis-decalin", "cis-decahydronaphthalene"),
            ("trans-decalin", "trans-decahydronaphthalene"),
        ],
    ),
    (
        "Alkenes, dienes & alkynes",
        [
            "1-heptene",
            "1-octene",
            "1-decene",
            "1-dodecene",
            "2-methyl-2-butene",
            "cyclopentene",
            "cyclohexene",
            "propyne",
            "1-butyne",
            ("alpha-methylstyrene", "alpha-methyl styrene"),
        ],
    ),
    (
        "Aromatics & polycyclics",
        [
            "1,2,3-trimethylbenzene",
            "1,2,4-trimethylbenzene",
            ("mesitylene", "1,3,5-trimethylbenzene"),
            ("durene", "1,2,4,5-tetramethylbenzene"),
            "n-propylbenzene",
            "n-butylbenzene",
            "sec-butylbenzene",
            "tert-butylbenzene",
            ("p-cymene", "4-isopropyltoluene"),
            "indane",
            "indene",
            ("tetralin", "1,2,3,4-tetrahydronaphthalene"),
            "1-methylnaphthalene",
            "2-methylnaphthalene",
            "biphenyl",
            "diphenylmethane",
            "anthracene",
            "phenanthrene",
            "fluorene",
            "acenaphthene",
            "pyrene",
        ],
    ),
    (
        "Phenols & cresols",
        [
            "o-cresol",
            "m-cresol",
            "p-cresol",
            ("2,6-xylenol", "2,6-dimethylphenol"),
            "catechol",
            "resorcinol",
            "hydroquinone",
        ],
    ),
    (
        "Alcohols",
        [
            "2-pentanol",
            "3-pentanol",
            "2-methyl-1-butanol",
            ("isoamyl alcohol", "3-methyl-1-butanol"),
            "1-heptanol",
            "1-octanol",
            "2-octanol",
            "2-ethyl-1-hexanol",
            "1-nonanol",
            "1-decanol",
            "1-dodecanol",
            "1-tetradecanol",
            "1-hexadecanol",
            "1-octadecanol",
            "benzyl alcohol",
            "allyl alcohol",
            "furfuryl alcohol",
            "cyclopentanol",
        ],
    ),
    (
        "Glycols, diols & glycol ethers",
        [
            "diethylene glycol",
            "triethylene glycol",
            "tetraethylene glycol",
            "dipropylene glycol",
            "1,3-propanediol",
            "1,4-butanediol",
            "1,3-butanediol",
            "neopentyl glycol",
            "1,6-hexanediol",
            "2-methoxyethanol",
            "2-ethoxyethanol",
            "2-butoxyethanol",
            ("1-methoxy-2-propanol", "propylene glycol methyl ether"),
        ],
    ),
    (
        "Ethers & acetals",
        [
            ("tame", "tert-amyl methyl ether"),
            ("etbe", "ethyl tert-butyl ether"),
            "diisopropyl ether",
            "di-n-butyl ether",
            "anisole",
            "phenetole",
            "diphenyl ether",
            ("methylal", "dimethoxymethane"),
            "2-methyltetrahydrofuran",
            "1,3-dioxolane",
            ("monoglyme", "1,2-dimethoxyethane"),
            ("diglyme", "diethylene glycol dimethyl ether"),
            "tetrahydropyran",
            "2,5-dimethylfuran",
        ],
    ),
    (
        "Aldehydes",
        [
            "propionaldehyde",
            ("n-butyraldehyde", "butyraldehyde"),
            "isobutyraldehyde",
            ("valeraldehyde", "pentanal"),
            "hexanal",
            "2-ethylhexanal",
            "benzaldehyde",
            "acrolein",
            "crotonaldehyde",
        ],
    ),
    (
        "Ketones & lactones",
        [
            "2-pentanone",
            "3-pentanone",
            ("methyl isopropyl ketone", "3-methyl-2-butanone"),
            ("pinacolone", "3,3-dimethyl-2-butanone"),
            "2-hexanone",
            "2-heptanone",
            "diisobutyl ketone",
            "acetophenone",
            "cyclopentanone",
            "isophorone",
            "mesityl oxide",
            "methyl vinyl ketone",
            "diacetone alcohol",
            "acetylacetone",
            "gamma-butyrolactone",
            "gamma-valerolactone",
        ],
    ),
    (
        "Carboxylic acids & anhydrides",
        [
            ("n-butyric acid", "butyric acid"),
            "isobutyric acid",
            ("valeric acid", "pentanoic acid"),
            ("caproic acid", "hexanoic acid"),
            ("caprylic acid", "octanoic acid"),
            ("capric acid", "decanoic acid"),
            "lauric acid",
            "myristic acid",
            "palmitic acid",
            "stearic acid",
            "oleic acid",
            "benzoic acid",
            "methacrylic acid",
            "acetic anhydride",
            "maleic anhydride",
            "phthalic anhydride",
        ],
    ),
    (
        "Esters & carbonates",
        [
            "n-propyl acetate",
            "isopropyl acetate",
            "isobutyl acetate",
            "n-amyl acetate",
            "isoamyl acetate",
            "methyl propionate",
            "ethyl propionate",
            "methyl butyrate",
            "ethyl butyrate",
            "ethyl formate",
            "methyl methacrylate",
            "methyl acrylate",
            "ethyl acrylate",
            "n-butyl acrylate",
            "2-ethylhexyl acrylate",
            "methyl benzoate",
            "dimethyl phthalate",
            "diethyl phthalate",
            "dibutyl phthalate",
            ("dioctyl phthalate", "bis(2-ethylhexyl) phthalate"),
            "dimethyl carbonate",
            "diethyl carbonate",
            "ethylene carbonate",
            "propylene carbonate",
            "methyl lactate",
            "ethyl lactate",
            ("methyl laurate", "methyl dodecanoate"),
            ("methyl myristate", "methyl tetradecanoate"),
            ("methyl palmitate", "methyl hexadecanoate"),
            ("methyl stearate", "methyl octadecanoate"),
            ("methyl oleate", "methyl cis-9-octadecenoate"),
        ],
    ),
    (
        "Amines & alkanolamines",
        [
            "ethylamine",
            "diethylamine",
            "n-propylamine",
            "isopropylamine",
            "n-butylamine",
            "tert-butylamine",
            "di-n-propylamine",
            "di-n-butylamine",
            "cyclohexylamine",
            "ethylenediamine",
            "diethylenetriamine",
            "hexamethylenediamine",
            "morpholine",
            "piperidine",
            "pyrrolidine",
            "piperazine",
            ("mdea", "methyldiethanolamine"),
            "triethanolamine",
            ("dipa", "diisopropanolamine"),
            ("amp", "2-amino-2-methyl-1-propanol"),
            ("diglycolamine", "2-(2-aminoethoxy)ethanol"),
            "n,n-dimethylaniline",
            "o-toluidine",
            "pyrrole",
            "quinoline",
            ("2-picoline", "2-methylpyridine"),
            ("3-picoline", "3-methylpyridine"),
            ("4-picoline", "4-methylpyridine"),
        ],
    ),
    (
        "Nitriles, nitro & amides",
        [
            "propionitrile",
            "butyronitrile",
            "benzonitrile",
            "acrylonitrile",
            "adiponitrile",
            "nitroethane",
            "1-nitropropane",
            "2-nitropropane",
            "nitrobenzene",
            "formamide",
            "n-methylformamide",
            ("dimethylacetamide", "n,n-dimethylacetamide"),
            "caprolactam",
            "2-pyrrolidone",
        ],
    ),
    (
        "Halogenated organics",
        [
            "bromomethane",
            "iodomethane",
            "chloroethane",
            "bromoethane",
            "1,1-dichloroethane",
            "1,1,1-trichloroethane",
            "1,1,2-trichloroethane",
            "tetrachloroethylene",
            ("vinylidene chloride", "1,1-dichloroethylene"),
            "cis-1,2-dichloroethylene",
            "trans-1,2-dichloroethylene",
            "allyl chloride",
            "epichlorohydrin",
            "benzyl chloride",
            "o-dichlorobenzene",
            "p-dichlorobenzene",
            "1,2,4-trichlorobenzene",
            "bromobenzene",
            "fluorobenzene",
            "hexafluorobenzene",
            "1-chlorobutane",
            ("chloroprene", "2-chloro-1,3-butadiene"),
            "phosgene",
            "1,2-dibromoethane",
            "perfluorohexane",
        ],
    ),
    (
        "Refrigerants & fluorocarbons",
        [
            ("r11", "trichlorofluoromethane"),
            ("r13", "chlorotrifluoromethane"),
            ("r14", "tetrafluoromethane"),
            ("r21", "dichlorofluoromethane"),
            ("r23", "trifluoromethane"),
            ("r41", "fluoromethane"),
            ("r113", "1,1,2-trichloro-1,2,2-trifluoroethane"),
            ("r114", "1,2-dichloro-1,1,2,2-tetrafluoroethane"),
            ("r115", "chloropentafluoroethane"),
            ("r116", "hexafluoroethane"),
            ("r123", "2,2-dichloro-1,1,1-trifluoroethane"),
            ("r124", "1-chloro-1,2,2,2-tetrafluoroethane"),
            ("r125", "pentafluoroethane"),
            ("r141b", "1,1-dichloro-1-fluoroethane"),
            ("r142b", "1-chloro-1,1-difluoroethane"),
            ("r143a", "1,1,1-trifluoroethane"),
            ("r218", "octafluoropropane"),
            ("r227ea", "1,1,1,2,3,3,3-heptafluoropropane"),
            ("r236fa", "1,1,1,3,3,3-hexafluoropropane"),
            ("r245fa", "1,1,1,3,3-pentafluoropropane"),
            ("r365mfc", "1,1,1,3,3-pentafluorobutane"),
            ("r1234yf", "2,3,3,3-tetrafluoropropene"),
            ("r1234ze", "trans-1,3,3,3-tetrafluoropropene"),
            ("rc318", "octafluorocyclobutane"),
            "sulfur hexafluoride",
            "nitrogen trifluoride",
        ],
    ),
    (
        "Siloxanes & silanes",
        [
            ("mm", "hexamethyldisiloxane"),
            ("mdm", "octamethyltrisiloxane"),
            ("md2m", "decamethyltetrasiloxane"),
            ("md3m", "dodecamethylpentasiloxane"),
            ("d3", "hexamethylcyclotrisiloxane"),
            ("d4", "octamethylcyclotetrasiloxane"),
            ("d5", "decamethylcyclopentasiloxane"),
            ("d6", "dodecamethylcyclohexasiloxane"),
            "tetramethylsilane",
        ],
    ),
    (
        "Sulfur compounds",
        [
            ("methyl mercaptan", "methanethiol"),
            ("ethyl mercaptan", "ethanethiol"),
            "dimethyl disulfide",
            "diethyl sulfide",
            "thiophene",
            "tetrahydrothiophene",
            "sulfolane",
            "carbonyl sulfide",
            "sulfur trioxide",
        ],
    ),
    (
        "Inorganic & light gases",
        [
            "krypton",
            "xenon",
            "deuterium",
            "ozone",
            "hydrogen bromide",
            "hydrogen fluoride",
            "hydrogen cyanide",
            "phosphine",
            "hydrazine",
            "hydrogen peroxide",
            "bromine",
        ],
    ),
    (
        "Terpenes & naturals",
        [
            "alpha-pinene",
            "beta-pinene",
            ("limonene", "d-limonene"),
            "menthol",
            ("eucalyptol", "1,8-cineole"),
            "alpha-terpineol",
            "linalool",
            "camphor",
        ],
    ),
    (
        "Other industrial solvents & monomers",
        [
            ("ethyl methyl carbonate", "methyl ethyl carbonate"),
            "trioxane",
            "vinyl toluene",
            "1-octanethiol",
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


def antoine(cas: str, tb: float | None) -> tuple[float, float, float, float, float] | None:
    """Transcribe Antoine constants (Pa -> bar via ``A -= 5``).

    Rows that fail to reproduce 1 atm at the normal boiling point within 3% are
    rejected (some source rows cover only a low-temperature window); the
    property-data backfill then fits a consistent row from the Wagner/Perry
    correlations instead.
    """
    if cas not in vp.Psat_data_AntoinePoling.index:
        return None
    row = vp.Psat_data_AntoinePoling.loc[cas]
    a = float(row["A"]) - 5.0
    b, c = float(row["B"]), float(row["C"])
    if tb is not None:
        p = 1.0e5 * 10.0 ** (a - b / (tb + c))
        if not math.isfinite(p) or abs(p - 101325.0) / 101325.0 > 0.03:
            return None
    return a, b, c, float(row["Tmin"]), float(row["Tmax"])


def fnum(x: float, nd: int) -> str:
    return f"{round(float(x), nd)!r}"


#: Critical temperatures overriding corrupt upstream rows (e.g. phenanthrene's
#: ``chemicals`` Tc of 0.869 K, a clear kK transcription error; value from NIST).
MANUAL_TC: dict[str, float] = {
    "phenanthrene": 869.25,
}


def emit(name: str, formula: str, cas: str) -> str | None:
    tc = MANUAL_TC.get(name, _try(ch.Tc, cas))
    pc = _try(ch.Pc, cas)
    omega = _try(ch.omega, cas)
    if tc is None or pc is None or omega is None:
        print(f"  SKIP {name}: missing Tc/Pc/omega")
        return None
    if not 2.0 < tc < 2000.0:
        print(f"  SKIP {name}: implausible Tc={tc} (corrupt source row?)")
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
    ant = antoine(cas, tb)
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


def resolve(query: str) -> tuple[str, str] | None:
    """Resolve a lookup query to ``(cas, formula)`` via ``chemicals``, or ``None``."""
    try:
        meta = ch.search_chemical(query)
    except Exception:
        return None
    cas = getattr(meta, "CASs", None)
    formula = getattr(meta, "formula", None)
    if not cas or not formula:
        return None
    return str(cas), str(formula)


def _inject(src: str, generated: str) -> str:
    anchor = "    ),\n)\n\n\ndef _with_supplements"
    if anchor not in src:
        raise SystemExit("anchor not found; cannot inject")
    return src.replace(anchor, "    ),\n" + generated + ")\n\n\ndef _with_supplements", 1)


def main() -> None:
    with open(COMPONENTS_PY) as f:
        src = f.read()

    import fugacio.thermo.components as comp

    existing_names = set(comp.DATABASE)
    existing_cas = {c.cas for c in comp.DATABASE.values() if c.cas}

    waves: list[tuple[str, list[tuple[str, list[tuple[str, str, str]]]]]] = []
    if SENTINEL not in src:
        waves.append((SENTINEL, [(h, list(items)) for h, items in SECTIONS]))
    if SENTINEL2 not in src:
        resolved_sections: list[tuple[str, list[tuple[str, str, str]]]] = []
        for header, entries in SECTIONS2:
            triples: list[tuple[str, str, str]] = []
            for entry in entries:
                name, query = entry if isinstance(entry, tuple) else (entry, entry)
                meta = resolve(query)
                if meta is None:
                    print(f"  SKIP {name}: cannot resolve {query!r}")
                    continue
                cas, formula = meta
                triples.append((name, formula, cas))
            resolved_sections.append((header, triples))
        waves.append((SENTINEL2, resolved_sections))
    if not waves:
        print("All sentinels already present; nothing to do.")
        return

    n_added = 0
    for sentinel, sections in waves:
        blocks: list[str] = [f"    {sentinel}"]
        for header, items in sections:
            section_blocks: list[str] = []
            for name, formula, cas in items:
                if name in existing_names:
                    print(f"  skip {name}: already in database")
                    continue
                if cas in existing_cas:
                    print(f"  skip {name}: CAS {cas} already in database")
                    continue
                block = emit(name, formula, cas)
                if block is not None:
                    section_blocks.append(block)
                    existing_names.add(name)
                    existing_cas.add(cas)
                    n_added += 1
            if section_blocks:
                dashes = "-" * max(3, 72 - len(header))
                blocks.append(f"    # --- {header} {dashes}")
                blocks.extend(section_blocks)
        src = _inject(src, "\n".join(blocks) + "\n")

    with open(COMPONENTS_PY, "w") as f:
        f.write(src)
    print(f"Injected {n_added} components into {COMPONENTS_PY}")


if __name__ == "__main__":
    main()
