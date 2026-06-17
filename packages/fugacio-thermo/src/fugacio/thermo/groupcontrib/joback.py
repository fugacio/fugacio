"""Joback group-contribution estimation of pure-component constants.

When a compound is missing from the curated database, the Joback method estimates
its critical constants, boiling/melting points, formation properties, and
ideal-gas heat-capacity polynomial from a simple count of functional groups::

    Tb = 198.2 + sum_k N_k * tb_k
    Tc = Tb / (0.584 + 0.965 * S - S**2),     S = sum_k N_k * tc_k
    Pc = (0.113 + 0.0032 * n_atoms - sum_k N_k * pc_k)**-2   [bar]
    Vc = 17.5 + sum_k N_k * vc_k                              [cm^3/mol]
    Cp(T) = (A - 37.93) + (B + 0.210) T + (C - 3.91e-4) T**2 + (D + 2.06e-7) T**3

with ``A = sum_k N_k * a_k`` etc. The bundled `GROUPS` table is the public
Joback contribution set (subset covering hydrocarbons, alcohols, ethers, and
ketones). The result is returned as a `fugacio.thermo.components.Component`
so estimated species drop straight into the rest of the engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from fugacio.thermo.components import Component, CpIdeal
from fugacio.thermo.constants import R


@dataclass(frozen=True)
class JobackGroup:
    """A single Joback functional-group contribution row."""

    tc: float
    pc: float
    vc: float
    tb: float
    tm: float
    hform: float  # kJ/mol
    gform: float  # kJ/mol
    cpa: float
    cpb: float
    cpc: float
    cpd: float


# Raw Joback contributions in field order:
# (tc, pc, vc, tb, tm, hform[kJ/mol], gform[kJ/mol], cpa, cpb, cpc, cpd).
_RAW: dict[str, tuple[float, ...]] = {
    "-CH3": (0.0141, -0.0012, 65, 23.58, -5.10, -76.45, -43.96, 19.5, -8.08e-3, 1.53e-4, -9.67e-8),
    "-CH2-": (0.0189, 0.0, 56, 22.88, 11.27, -20.64, 8.42, -0.909, 9.5e-2, -5.44e-5, 1.19e-8),
    ">CH-": (0.0164, 0.002, 41, 21.74, 12.64, 29.89, 58.36, -23.0, 2.04e-1, -2.65e-4, 1.2e-7),
    ">C<": (0.0067, 0.0043, 27, 18.25, 46.43, 82.23, 116.02, -66.2, 4.27e-1, -6.41e-4, 3.01e-7),
    "=CH2": (0.0113, -0.0028, 56, 18.18, -4.32, -9.63, 3.77, 23.6, -3.81e-2, 1.72e-4, -1.03e-7),
    "=CH-": (0.0129, -0.0006, 46, 24.96, 8.73, 37.97, 48.53, -8.0, 1.05e-1, -9.63e-5, 3.56e-8),
    "ring-CH2-": (0.01, 0.0025, 48, 27.15, 7.75, -26.8, -3.68, -6.03, 8.54e-2, -8.0e-6, -1.8e-8),
    "ring>CH-": (0.0122, 0.0004, 38, 21.78, 19.88, 8.67, 40.99, -20.5, 1.62e-1, -1.6e-4, 6.24e-8),
    "ringACH": (0.0082, 0.0011, 41, 26.73, 8.13, 2.09, 11.3, -2.14, 5.74e-2, -1.64e-6, -1.59e-8),
    "ringAC": (0.0143, 0.0008, 32, 31.01, 37.02, 46.43, 54.05, -8.25, 1.01e-1, -1.42e-4, 6.78e-8),
    "-OH": (0.0741, 0.0112, 28, 92.88, 44.45, -208.04, -189.2, 25.7, -6.91e-2, 1.77e-4, -9.88e-8),
    "-O-": (0.0168, 0.0015, 18, 22.42, -15.18, -132.22, -105.0, 25.5, -6.32e-2, 1.11e-4, -5.48e-8),
    ">C=O": (0.038, 0.0031, 62, 76.75, 61.2, -133.22, -120.5, 6.45, 6.7e-2, -3.57e-5, 2.86e-9),
    "-COOH": (
        0.0791,
        0.0077,
        89,
        169.09,
        155.5,
        -426.72,
        -387.87,
        24.1,
        4.27e-2,
        8.04e-5,
        -6.87e-8,
    ),
}

#: Joback group contributions, keyed by a short group label.
GROUPS: dict[str, JobackGroup] = {name: JobackGroup(*row) for name, row in _RAW.items()}


def joback_estimate(
    groups: dict[str, int],
    n_atoms: int,
    *,
    name: str = "estimated",
    formula: str = "",
    mw: float = 0.0,
) -> Component:
    """Estimate pure-component constants from Joback groups and return a Component.

    Args:
        groups: Mapping of group label (see `GROUPS`) to occurrence count.
        n_atoms: Total number of atoms in the molecule (hydrogens included),
            required by the critical-pressure correlation.
        name: Name to assign to the resulting component.
        formula: Optional molecular formula.
        mw: Optional molar mass (g/mol); Joback does not estimate it.

    Returns:
        A `Component` with estimated ``tc``
        (K), ``pc`` (Pa), ``vc`` (m^3/mol), ``tb`` (K), formation properties
        (J/mol), and an ideal-gas ``cp_ig`` correlation.

    Raises:
        KeyError: if any group label is not in `GROUPS`.
    """
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        raise KeyError(f"unknown Joback groups: {unknown}")

    def total(attr: str) -> float:
        return sum(count * getattr(GROUPS[g], attr) for g, count in groups.items())

    tb = 198.2 + total("tb")
    s_tc = total("tc")
    tc = tb / (0.584 + 0.965 * s_tc - s_tc**2)
    pc_bar = (0.113 + 0.0032 * n_atoms - total("pc")) ** -2
    vc_cm3 = 17.5 + total("vc")
    hform = (68.29 + total("hform")) * 1.0e3
    gform = (53.88 + total("gform")) * 1.0e3

    # Joback's Cp(T) is a cubic in J/mol/K; store it in CpIdeal's /R basis using
    # the cubic ``e`` coefficient (so the database's d/T**2 term stays zero).
    cp = CpIdeal(
        a=(total("cpa") - 37.93) / R,
        b=(total("cpb") + 0.210) / R,
        c=(total("cpc") - 3.91e-4) / R,
        d=0.0,
        e=(total("cpd") + 2.06e-7) / R,
    )

    estimated = Component(
        name=name,
        formula=formula,
        cas=None,
        mw=mw,
        tc=tc,
        pc=pc_bar * 1.0e5,
        omega=0.0,
        tb=tb,
        vc=vc_cm3 * 1.0e-6,
        zc=None,
        antoine=None,
        cp_ig=cp,
        hform_ig=hform,
        gform_ig=gform,
    )
    return estimated
