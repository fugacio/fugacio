"""Pure-component model and a curated database of open reference constants.

A :class:`Component` is an immutable bundle of the constants Fugacio needs to
evaluate equations of state, ideal-gas properties, and saturation pressures for
one chemical species. The values bundled in :data:`DATABASE` are textbook
reference data drawn from open sources:

* critical constants and acentric factors -- Poling, Prausnitz & O'Connell,
  *The Properties of Gases and Liquids* (5th ed.), Appendix A;
* ideal-gas heat capacities -- Smith, Van Ness & Abbott,
  *Introduction to Chemical Engineering Thermodynamics* (7th ed.), Table C.1,
  in the form ``Cp/R = a + b*T + c*T**2 + d/T**2``;
* Antoine vapour-pressure constants -- NIST Chemistry WebBook, in the form
  ``log10(P/bar) = a - b / (T/K + c)``.

``Component`` instances are deliberately *static* Python objects (not JAX
pytrees): the differentiable numerical kernels in :mod:`fugacio.thermo` operate
on plain arrays, which you extract with :func:`component_arrays`. That keeps
gradients flowing with respect to the physical parameters themselves (useful for
parameter estimation) without entangling autodiff with database bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import BAR


@dataclass(frozen=True)
class AntoineCoeffs:
    """Antoine vapour-pressure constants in NIST form.

    ``log10(P/bar) = a - b / (T/K + c)``, valid on ``[t_min, t_max]`` kelvin.
    """

    a: float
    b: float
    c: float
    t_min: float
    t_max: float


@dataclass(frozen=True)
class CpIdeal:
    """Ideal-gas heat-capacity correlation ``Cp/R = a + b*T + c*T**2 + d/T**2 + e*T**3``.

    The ``a, b, c, d`` form is the one tabulated by Smith, Van Ness & Abbott (at
    most three coefficients non-zero per species, ``e = 0``). The extra cubic
    term ``e`` lets the same dataclass also hold Joback-style estimates
    (``a + b*T + c*T**2 + e*T**3``), keeping a single ideal-gas integrator for
    both. ``t_min`` and ``t_max`` bound the fitted temperature range in kelvin.
    """

    a: float
    b: float
    c: float
    d: float
    e: float = 0.0
    t_min: float = 0.0
    t_max: float = float("inf")


@dataclass(frozen=True)
class Component:
    """Immutable pure-component constant record (SI units, except molar mass).

    Attributes:
        name: Canonical lower-case identifier (the database key).
        formula: Molecular formula, e.g. ``"C2H6O"``.
        cas: CAS registry number as a string, or ``None`` if unassigned here.
        mw: Molar mass in g/mol.
        tc: Critical temperature in K.
        pc: Critical pressure in Pa.
        omega: Pitzer acentric factor (dimensionless).
        tb: Normal boiling point in K, or ``None``.
        vc: Critical molar volume in m^3/mol, or ``None``.
        zc: Critical compressibility factor, or ``None``.
        antoine: :class:`AntoineCoeffs`, or ``None`` if not tabulated.
        cp_ig: :class:`CpIdeal`, or ``None`` if not tabulated.
        hform_ig: Standard ideal-gas enthalpy of formation (J/mol at 298.15 K).
        gform_ig: Standard ideal-gas Gibbs energy of formation (J/mol at 298.15 K).
    """

    name: str
    formula: str
    cas: str | None
    mw: float
    tc: float
    pc: float
    omega: float
    tb: float | None = None
    vc: float | None = None
    zc: float | None = None
    antoine: AntoineCoeffs | None = None
    cp_ig: CpIdeal | None = None
    hform_ig: float | None = None
    gform_ig: float | None = None


def _ant(a: float, b: float, c: float, t_min: float, t_max: float) -> AntoineCoeffs:
    return AntoineCoeffs(a=a, b=b, c=c, t_min=t_min, t_max=t_max)


def _cp(
    a: float,
    b: float = 0.0,
    c: float = 0.0,
    d: float = 0.0,
    t_min: float = 50.0,
    t_max: float = 1500.0,
) -> CpIdeal:
    """Build a :class:`CpIdeal` from Smith-Van-Ness-Abbott Table C.1 magnitudes.

    The table reports ``b`` scaled by ``1e3``, ``c`` by ``1e6`` and ``d`` by
    ``1e-5``; this helper rescales them to SI so callers can transcribe the
    table verbatim.
    """
    return CpIdeal(a=a, b=b * 1e-3, c=c * 1e-6, d=d * 1e5, t_min=t_min, t_max=t_max)


def _comp(
    name: str,
    *,
    formula: str,
    cas: str | None,
    mw: float,
    tc: float,
    pc_bar: float,
    omega: float,
    tb: float | None = None,
    vc_cm3: float | None = None,
    zc: float | None = None,
    antoine: AntoineCoeffs | None = None,
    cp_ig: CpIdeal | None = None,
    hform_ig: float | None = None,
    gform_ig: float | None = None,
) -> Component:
    """Construct a :class:`Component`, converting ``pc_bar`` and ``vc_cm3`` to SI."""
    return Component(
        name=name,
        formula=formula,
        cas=cas,
        mw=mw,
        tc=tc,
        pc=pc_bar * BAR,
        omega=omega,
        tb=tb,
        vc=None if vc_cm3 is None else vc_cm3 * 1e-6,
        zc=zc,
        antoine=antoine,
        cp_ig=cp_ig,
        hform_ig=hform_ig,
        gform_ig=gform_ig,
    )


# --- Curated open reference database -------------------------------------------
# Keyed by canonical lower-case name. Pc supplied in bar, Vc in cm^3/mol.

_COMPONENTS: tuple[Component, ...] = (
    # --- Light gases ---------------------------------------------------------
    _comp(
        "nitrogen",
        formula="N2",
        cas="7727-37-9",
        mw=28.014,
        tc=126.20,
        pc_bar=33.98,
        omega=0.037,
        tb=77.35,
        vc_cm3=89.2,
        zc=0.289,
        cp_ig=_cp(3.280, 0.593, 0.0, 0.040),
        hform_ig=0.0,
        gform_ig=0.0,
    ),
    _comp(
        "oxygen",
        formula="O2",
        cas="7782-44-7",
        mw=31.999,
        tc=154.58,
        pc_bar=50.43,
        omega=0.022,
        tb=90.17,
        vc_cm3=73.4,
        zc=0.288,
        cp_ig=_cp(3.639, 0.506, 0.0, -0.227),
        hform_ig=0.0,
        gform_ig=0.0,
    ),
    _comp(
        "argon",
        formula="Ar",
        cas="7440-37-1",
        mw=39.948,
        tc=150.86,
        pc_bar=48.98,
        omega=-0.002,
        tb=87.27,
        vc_cm3=74.6,
        zc=0.291,
        cp_ig=_cp(2.5),
        hform_ig=0.0,
        gform_ig=0.0,
    ),
    _comp(
        "hydrogen",
        formula="H2",
        cas="1333-74-0",
        mw=2.016,
        tc=33.19,
        pc_bar=13.13,
        omega=-0.216,
        tb=20.27,
        vc_cm3=64.1,
        zc=0.305,
        cp_ig=_cp(3.249, 0.422, 0.0, 0.083),
        hform_ig=0.0,
        gform_ig=0.0,
    ),
    _comp(
        "carbon monoxide",
        formula="CO",
        cas="630-08-0",
        mw=28.010,
        tc=132.92,
        pc_bar=34.99,
        omega=0.048,
        tb=81.66,
        vc_cm3=93.4,
        zc=0.292,
        cp_ig=_cp(3.376, 0.557, 0.0, -0.031),
        hform_ig=-110525.0,
        gform_ig=-137169.0,
    ),
    _comp(
        "carbon dioxide",
        formula="CO2",
        cas="124-38-9",
        mw=44.010,
        tc=304.21,
        pc_bar=73.83,
        omega=0.224,
        tb=194.75,
        vc_cm3=94.07,
        zc=0.274,
        cp_ig=_cp(5.457, 1.045, 0.0, -1.157),
        hform_ig=-393509.0,
        gform_ig=-394359.0,
    ),
    _comp(
        "water",
        formula="H2O",
        cas="7732-18-5",
        mw=18.015,
        tc=647.10,
        pc_bar=220.55,
        omega=0.345,
        tb=373.15,
        vc_cm3=55.95,
        zc=0.229,
        antoine=_ant(4.6543, 1435.264, -64.848, 255.9, 373.0),
        cp_ig=_cp(3.470, 1.450, 0.0, 0.121),
        hform_ig=-241818.0,
        gform_ig=-228572.0,
    ),
    _comp(
        "ammonia",
        formula="NH3",
        cas="7664-41-7",
        mw=17.031,
        tc=405.40,
        pc_bar=113.53,
        omega=0.256,
        tb=239.82,
        vc_cm3=72.5,
        zc=0.244,
        antoine=_ant(4.86886, 1113.928, -10.409, 239.6, 371.5),
        cp_ig=_cp(3.578, 3.020, 0.0, -0.186),
        hform_ig=-45940.0,
        gform_ig=-16401.0,
    ),
    _comp(
        "hydrogen sulfide",
        formula="H2S",
        cas="7783-06-4",
        mw=34.081,
        tc=373.40,
        pc_bar=89.63,
        omega=0.090,
        tb=212.80,
        vc_cm3=98.5,
        zc=0.284,
        cp_ig=_cp(3.931, 1.490, 0.0, -0.232),
        hform_ig=-20630.0,
        gform_ig=-33440.0,
    ),
    # --- Alkanes -------------------------------------------------------------
    _comp(
        "methane",
        formula="CH4",
        cas="74-82-8",
        mw=16.043,
        tc=190.56,
        pc_bar=45.99,
        omega=0.011,
        tb=111.66,
        vc_cm3=98.6,
        zc=0.286,
        cp_ig=_cp(1.702, 9.081, -2.164),
        hform_ig=-74520.0,
        gform_ig=-50460.0,
    ),
    _comp(
        "ethane",
        formula="C2H6",
        cas="74-84-0",
        mw=30.070,
        tc=305.32,
        pc_bar=48.72,
        omega=0.099,
        tb=184.55,
        vc_cm3=145.5,
        zc=0.279,
        cp_ig=_cp(1.131, 19.225, -5.561),
        hform_ig=-83820.0,
        gform_ig=-31855.0,
    ),
    _comp(
        "propane",
        formula="C3H8",
        cas="74-98-6",
        mw=44.096,
        tc=369.83,
        pc_bar=42.48,
        omega=0.152,
        tb=231.02,
        vc_cm3=200.0,
        zc=0.276,
        cp_ig=_cp(1.213, 28.785, -8.824),
        hform_ig=-104680.0,
        gform_ig=-24290.0,
    ),
    _comp(
        "n-butane",
        formula="C4H10",
        cas="106-97-8",
        mw=58.122,
        tc=425.12,
        pc_bar=37.96,
        omega=0.200,
        tb=272.66,
        vc_cm3=255.0,
        zc=0.274,
        cp_ig=_cp(1.935, 36.915, -11.402),
        hform_ig=-125790.0,
        gform_ig=-16570.0,
    ),
    _comp(
        "isobutane",
        formula="C4H10",
        cas="75-28-5",
        mw=58.122,
        tc=407.80,
        pc_bar=36.40,
        omega=0.186,
        tb=261.43,
        vc_cm3=262.7,
        zc=0.278,
        cp_ig=_cp(1.677, 37.853, -11.945),
        hform_ig=-134990.0,
        gform_ig=-20880.0,
    ),
    _comp(
        "n-pentane",
        formula="C5H12",
        cas="109-66-0",
        mw=72.149,
        tc=469.70,
        pc_bar=33.70,
        omega=0.251,
        tb=309.22,
        vc_cm3=311.0,
        zc=0.270,
        antoine=_ant(3.9892, 1070.617, -40.454, 268.8, 341.4),
        cp_ig=_cp(2.464, 45.351, -14.111),
        hform_ig=-146760.0,
        gform_ig=-8650.0,
    ),
    _comp(
        "n-hexane",
        formula="C6H14",
        cas="110-54-3",
        mw=86.175,
        tc=507.60,
        pc_bar=30.25,
        omega=0.301,
        tb=341.88,
        vc_cm3=368.0,
        zc=0.266,
        antoine=_ant(4.00266, 1171.530, -48.784, 286.2, 342.7),
        cp_ig=_cp(3.025, 53.722, -16.791),
        hform_ig=-166920.0,
        gform_ig=-150.0,
    ),
    _comp(
        "n-heptane",
        formula="C7H16",
        cas="142-82-5",
        mw=100.202,
        tc=540.20,
        pc_bar=27.40,
        omega=0.350,
        tb=371.57,
        vc_cm3=428.0,
        zc=0.261,
        antoine=_ant(4.02832, 1268.636, -56.199, 299.1, 372.4),
        cp_ig=_cp(3.570, 62.127, -19.486),
        hform_ig=-187800.0,
        gform_ig=8000.0,
    ),
    _comp(
        "n-octane",
        formula="C8H18",
        cas="111-65-9",
        mw=114.229,
        tc=568.70,
        pc_bar=24.90,
        omega=0.399,
        tb=398.82,
        vc_cm3=492.0,
        zc=0.259,
        antoine=_ant(4.05075, 1356.360, -63.515, 326.1, 399.7),
        cp_ig=_cp(4.108, 70.567, -22.208),
        hform_ig=-208750.0,
        gform_ig=16260.0,
    ),
    _comp(
        "cyclohexane",
        formula="C6H12",
        cas="110-82-7",
        mw=84.161,
        tc=553.50,
        pc_bar=40.73,
        omega=0.209,
        tb=353.93,
        vc_cm3=308.0,
        zc=0.273,
        antoine=_ant(3.96988, 1203.526, -50.287, 293.1, 354.7),
        cp_ig=_cp(-3.876, 63.249, -20.928),
        hform_ig=-123140.0,
        gform_ig=31920.0,
    ),
    # --- Alkenes -------------------------------------------------------------
    _comp(
        "ethylene",
        formula="C2H4",
        cas="74-85-1",
        mw=28.054,
        tc=282.34,
        pc_bar=50.41,
        omega=0.087,
        tb=169.42,
        vc_cm3=131.1,
        zc=0.281,
        cp_ig=_cp(1.424, 14.394, -4.392),
        hform_ig=52510.0,
        gform_ig=68460.0,
    ),
    _comp(
        "propylene",
        formula="C3H6",
        cas="115-07-1",
        mw=42.081,
        tc=364.90,
        pc_bar=46.00,
        omega=0.142,
        tb=225.46,
        vc_cm3=188.4,
        zc=0.289,
        cp_ig=_cp(1.637, 22.706, -6.915),
        hform_ig=19710.0,
        gform_ig=62205.0,
    ),
    # --- Aromatics -----------------------------------------------------------
    _comp(
        "benzene",
        formula="C6H6",
        cas="71-43-2",
        mw=78.114,
        tc=562.05,
        pc_bar=48.95,
        omega=0.210,
        tb=353.24,
        vc_cm3=256.0,
        zc=0.268,
        antoine=_ant(4.01814, 1203.835, -53.226, 333.4, 373.5),
        cp_ig=_cp(-0.206, 39.064, -13.301),
        hform_ig=82930.0,
        gform_ig=129665.0,
    ),
    _comp(
        "toluene",
        formula="C7H8",
        cas="108-88-3",
        mw=92.141,
        tc=591.75,
        pc_bar=41.08,
        omega=0.264,
        tb=383.79,
        vc_cm3=316.0,
        zc=0.264,
        antoine=_ant(4.07827, 1343.943, -53.773, 308.5, 384.7),
        cp_ig=_cp(0.290, 47.052, -15.716),
        hform_ig=50170.0,
        gform_ig=122050.0,
    ),
    # --- Alcohols / oxygenates ----------------------------------------------
    _comp(
        "methanol",
        formula="CH4O",
        cas="67-56-1",
        mw=32.042,
        tc=512.64,
        pc_bar=80.97,
        omega=0.565,
        tb=337.69,
        vc_cm3=118.0,
        zc=0.224,
        antoine=_ant(5.20409, 1581.341, -33.50, 288.0, 356.8),
        cp_ig=_cp(2.211, 12.216, -3.450),
        hform_ig=-200940.0,
        gform_ig=-162240.0,
    ),
    _comp(
        "ethanol",
        formula="C2H6O",
        cas="64-17-5",
        mw=46.069,
        tc=513.92,
        pc_bar=61.48,
        omega=0.649,
        tb=351.44,
        vc_cm3=167.0,
        zc=0.240,
        antoine=_ant(5.37229, 1670.409, -40.191, 293.0, 366.6),
        cp_ig=_cp(3.518, 20.001, -6.002),
        hform_ig=-234950.0,
        gform_ig=-167730.0,
    ),
    _comp(
        "2-propanol",
        formula="C3H8O",
        cas="67-63-0",
        mw=60.096,
        tc=508.30,
        pc_bar=47.62,
        omega=0.665,
        tb=355.41,
        vc_cm3=222.0,
        zc=0.248,
        antoine=_ant(5.24268, 1580.920, -53.540, 329.9, 362.4),
        hform_ig=-272295.0,
    ),
    _comp(
        "acetone",
        formula="C3H6O",
        cas="67-64-1",
        mw=58.080,
        tc=508.10,
        pc_bar=47.00,
        omega=0.307,
        tb=329.22,
        vc_cm3=209.0,
        zc=0.233,
        antoine=_ant(4.42448, 1312.253, -32.445, 259.2, 507.6),
        cp_ig=_cp(1.625, 25.700, -8.067),
        hform_ig=-217150.0,
        gform_ig=-152716.0,
    ),
)

#: Canonical-name -> :class:`Component` lookup for the curated database.
DATABASE: dict[str, Component] = {c.name: c for c in _COMPONENTS}


def get(name: str) -> Component:
    """Return the :class:`Component` for ``name`` (case-insensitive).

    Raises:
        KeyError: if ``name`` is not present in :data:`DATABASE`.
    """
    key = name.strip().lower()
    if key not in DATABASE:
        raise KeyError(f"unknown component {name!r}; known components: {sorted(DATABASE)}")
    return DATABASE[key]


def names() -> list[str]:
    """Return the sorted list of component names in the database."""
    return sorted(DATABASE)


def component_arrays(components: list[str] | list[Component]) -> dict[str, Array]:
    """Stack the core EOS constants of several components into JAX arrays.

    Args:
        components: A list of component names or :class:`Component` objects.

    Returns:
        A dict with keys ``"tc"``, ``"pc"``, ``"omega"`` and ``"mw"`` mapping to
        1-D arrays aligned with the input order, ready to feed the equation-of-
        state kernels.
    """
    resolved = [get(c) if isinstance(c, str) else c for c in components]
    return {
        "tc": jnp.asarray([c.tc for c in resolved]),
        "pc": jnp.asarray([c.pc for c in resolved]),
        "omega": jnp.asarray([c.omega for c in resolved]),
        "mw": jnp.asarray([c.mw for c in resolved]),
    }
