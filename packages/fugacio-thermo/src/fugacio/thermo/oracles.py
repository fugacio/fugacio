"""Optional differential-testing oracles (CoolProp, ``chemicals``).

The README grades Fugacio against open reference codes. Those libraries are *not*
runtime dependencies -- they are imported lazily here so the engine installs lean
and the comparisons run only where the oracle is present (tests skip otherwise).
Saturation pressure is taken from the fast :mod:`chemicals` Wagner correlation
tables rather than the slow ``thermo.Chemical`` object.

Because the reference codes use high-accuracy multiparameter or different cubic
equations of state, agreement is expected to a tolerance, not to machine
precision: these checks catch gross errors and regressions rather than asserting
bit-for-bit equality.
"""

from __future__ import annotations

from fugacio.thermo.constants import R

# Map Fugacio canonical names to CoolProp fluid names.
_COOLPROP_NAMES: dict[str, str] = {
    "methane": "Methane",
    "ethane": "Ethane",
    "propane": "Propane",
    "n-butane": "n-Butane",
    "isobutane": "IsoButane",
    "n-pentane": "n-Pentane",
    "n-hexane": "n-Hexane",
    "n-heptane": "n-Heptane",
    "n-octane": "n-Octane",
    "cyclohexane": "CycloHexane",
    "benzene": "Benzene",
    "toluene": "Toluene",
    "water": "Water",
    "methanol": "Methanol",
    "ethanol": "Ethanol",
    "nitrogen": "Nitrogen",
    "oxygen": "Oxygen",
    "argon": "Argon",
    "carbon dioxide": "CarbonDioxide",
    "ammonia": "Ammonia",
    "hydrogen sulfide": "HydrogenSulfide",
}


def has_coolprop() -> bool:
    """Return ``True`` if the optional CoolProp package is importable."""
    try:
        import CoolProp  # noqa: F401
    except ImportError:
        return False
    return True


def has_thermo() -> bool:
    """Return ``True`` if the optional ``chemicals`` correlation library is importable.

    The saturation-pressure oracle is backed by :mod:`chemicals` (the data layer
    of the ``thermo`` project) rather than the heavyweight ``thermo.Chemical``
    object, whose constructor is far too slow to use as a test oracle.
    """
    try:
        import chemicals  # noqa: F401
    except ImportError:
        return False
    return True


def coolprop_name(name: str) -> str:
    """Translate a Fugacio component name to its CoolProp fluid name."""
    key = name.strip().lower()
    if key not in _COOLPROP_NAMES:
        raise KeyError(f"no CoolProp mapping for {name!r}")
    return _COOLPROP_NAMES[key]


def coolprop_psat(name: str, t: float) -> float:
    """Saturation pressure (Pa) from CoolProp's reference equation of state."""
    from CoolProp.CoolProp import PropsSI

    return float(PropsSI("P", "T", t, "Q", 0.0, coolprop_name(name)))


def coolprop_compressibility(name: str, t: float, p: float) -> float:
    """Single-phase compressibility factor ``Z = P / (rho_molar R T)`` from CoolProp."""
    from CoolProp.CoolProp import PropsSI

    rho_molar = float(PropsSI("Dmolar", "T", t, "P", p, coolprop_name(name)))
    return p / (rho_molar * R * t)


def thermo_psat(name: str, t: float) -> float:
    """Saturation pressure (Pa) from the ``chemicals`` Wagner correlation.

    Uses the McGarry Wagner vapour-pressure coefficients bundled with the
    :mod:`chemicals` package -- a fast reference independent of CoolProp's
    Helmholtz equation of state, giving a genuine second opinion. The slow
    ``thermo.Chemical`` constructor is deliberately avoided.

    Raises:
        KeyError: if the component has no Wagner coefficients in the dataset.
    """
    from chemicals import identifiers
    from chemicals import vapor_pressure as vp

    cas = identifiers.CAS_from_any(name)
    table = vp.Psat_data_WagnerMcGarry
    if cas not in table.index:
        raise KeyError(f"no Wagner vapour-pressure data for {name!r} (CAS {cas})")
    row = table.loc[cas]
    return float(
        vp.Wagner_original(t, row["Tc"], row["Pc"], row["A"], row["B"], row["C"], row["D"])
    )
