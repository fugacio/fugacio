"""Differential-testing oracles: reference values from external libraries.

Fugacio validates its own results two ways: against first-principles identities
(see, e.g., `fugacio.thermo.departure`) and against *independent reference
implementations*. This module is the second kind: thin wrappers over the
third-party ``thermo`` / ``chemicals`` packages (and CoolProp where installed)
that return reference activity coefficients, bubble pressures, and property
values for cross-checking.

These backends are **never runtime dependencies**: imports are deferred and
guarded, importing this module always succeeds, and each helper raises a clear
`RuntimeError` if its backend is absent. The companion oracle tests are
opt-in (``pytest -m oracle``) and excluded from the default suite so the unit
tests stay fast and hermetic.

The activity-coefficient oracles deliberately reuse Fugacio's own group
assignments (`fugacio.thermo.groupcontrib.unifac.COMPONENT_GROUPS` and the
Dortmund table) when calling the reference UNIFAC, so a mismatch isolates the
*kernel* implementation rather than a difference in group splitting.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from importlib.util import find_spec
from typing import Any

import jax.numpy as jnp
from jax import Array

from fugacio.thermo.components import Component, component_arrays, get
from fugacio.thermo.constants import P_REF, T_REF, R
from fugacio.thermo.eos import PR, CubicEOS
from fugacio.thermo.groupcontrib._dortmund_data import DO_COMPONENT_GROUPS
from fugacio.thermo.groupcontrib.unifac import COMPONENT_GROUPS
from fugacio.thermo.reactions import Reaction
from fugacio.thermo.reference import saturation_pressures

ArrayLike = Array | float

#: Whether each optional reference backend is importable (no heavy import here).
HAVE_THERMO: bool = find_spec("thermo") is not None
HAVE_CHEMICALS: bool = find_spec("chemicals") is not None
HAVE_COOLPROP: bool = find_spec("CoolProp") is not None
#: Clapeyron.jl is reached through the ``juliacall`` bridge (needs a Julia install).
HAVE_CLAPEYRON: bool = find_spec("juliacall") is not None
#: Cantera is the reference for chemical-reaction equilibrium / standard-state thermo.
HAVE_CANTERA: bool = find_spec("cantera") is not None


def _require(name: str, available: bool) -> None:
    if not available:
        raise RuntimeError(
            f"oracle backend {name!r} is not installed; install the optional "
            f"reference packages to run differential tests"
        )


def cas_for(name: str) -> str:
    """CAS registry number for a component name (via ``chemicals``)."""
    _require("chemicals", HAVE_CHEMICALS)
    from chemicals import CAS_from_any

    return str(CAS_from_any(name))


def _matrix(values: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(c) for c in row] for row in values]


def thermo_nrtl_gamma(
    x: Sequence[float], t: float, b: Sequence[Sequence[float]], alpha: Sequence[Sequence[float]]
) -> list[float]:
    """Reference NRTL activity coefficients from ``thermo`` (``tau = b/T``, ``alpha`` const)."""
    _require("thermo", HAVE_THERMO)
    from thermo.nrtl import NRTL

    model = NRTL(xs=[float(v) for v in x], T=float(t), tau_bs=_matrix(b), alpha_cs=_matrix(alpha))
    return [float(g) for g in model.gammas()]


def thermo_uniquac_gamma(
    x: Sequence[float],
    t: float,
    r: Sequence[float],
    q: Sequence[float],
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
) -> list[float]:
    """Reference UNIQUAC activity coefficients from ``thermo`` (``tau = exp(a + b/T)``)."""
    _require("thermo", HAVE_THERMO)
    from thermo.uniquac import UNIQUAC

    model = UNIQUAC(
        xs=[float(v) for v in x],
        T=float(t),
        rs=[float(v) for v in r],
        qs=[float(v) for v in q],
        tau_as=_matrix(a),
        tau_bs=_matrix(b),
    )
    return [float(g) for g in model.gammas()]


def thermo_unifac_gamma(
    components: Sequence[str], x: Sequence[float], t: float, *, dortmund: bool = False
) -> list[float]:
    """Reference (modified) UNIFAC activity coefficients from ``thermo``.

    Uses Fugacio's own subgroup assignments so the comparison isolates the kernel.
    ``dortmund=True`` selects modified UNIFAC (``version=1``); otherwise classic
    UNIFAC (``version=0``).
    """
    _require("thermo", HAVE_THERMO)
    from thermo.unifac import UNIFAC

    table = DO_COMPONENT_GROUPS if dortmund else COMPONENT_GROUPS
    chemgroups = [dict(table[c]) for c in components]
    model = UNIFAC.from_subgroups(
        T=float(t),
        xs=[float(v) for v in x],
        chemgroups=chemgroups,
        version=1 if dortmund else 0,
    )
    return [float(g) for g in model.gammas()]


def thermo_wilson_gamma(
    x: Sequence[float], lam: Sequence[Sequence[float]], t: float = 300.0
) -> list[float]:
    """Reference Wilson activity coefficients from ``thermo`` for a given ``Lambda`` matrix.

    Passing the *same* ``Lambda`` matrix to both implementations isolates the Wilson
    gamma kernel from how ``Lambda`` is built (molar volumes and energies). ``thermo``
    parametrises ``Lambda_ij = exp(a_ij + b_ij/T + ...)``, so we set
    ``a_ij = ln(Lambda_ij)`` and leave the temperature coefficients at zero; the
    result is then independent of ``t``.
    """
    _require("thermo", HAVE_THERMO)
    import math

    from thermo.wilson import Wilson

    lam_f = _matrix(lam)
    a = [[math.log(v) for v in row] for row in lam_f]
    model = Wilson(T=float(t), xs=[float(v) for v in x], lambda_as=a)
    return [float(g) for g in model.gammas()]


def clapeyron_gamma(
    components: Sequence[str],
    x: Sequence[float],
    t: float,
    *,
    model: str = "UNIFAC",
    pressure: float = 101325.0,
) -> list[float]:
    """Reference activity coefficients from Clapeyron.jl via the ``juliacall`` bridge.

    Requires a working Julia with ``Clapeyron.jl`` installed and the optional
    ``juliacall`` Python package (``HAVE_CLAPEYRON``). ``model`` names a Clapeyron
    activity-model constructor (e.g. ``"UNIFAC"``, ``"NRTL"``, ``"Wilson"``) keyed by
    component *names* in Clapeyron's database; ``pressure`` is accepted only for API
    symmetry since activity models are pressure-independent.

    This oracle is intentionally never exercised by the default suite (no Julia in
    CI); its companion test skips unless ``juliacall`` is importable.
    """
    _require("juliacall (with Clapeyron.jl)", HAVE_CLAPEYRON)
    from juliacall import Main as jl

    jl.seval("using Clapeyron")
    names = jl.convert(jl.seval("Vector{String}"), [str(c) for c in components])
    constructor = getattr(jl, model)
    cmodel = constructor(names)
    z = jl.convert(jl.seval("Vector{Float64}"), [float(v) for v in x])
    gammas = jl.activity_coefficient(cmodel, float(pressure), float(t), z)
    return [float(g) for g in gammas]


def clapeyron_pcsaft(
    components: Sequence[str],
    x: Sequence[float],
    t: float,
    rho: float,
) -> dict[str, float]:
    """Reference PC-SAFT pressure and Z at ``(T, rho, x)`` from Clapeyron.jl.

    Builds Clapeyron's stock ``PCSAFT`` model for ``components`` (keyed by name in
    its database) and evaluates the pressure of one mole of the mixture at molar
    density ``rho`` (mol/m^3), i.e. total volume ``V = 1 / rho``. Returns the
    pressure (Pa) and compressibility factor.

    This is the strongest external oracle for `fugacio.thermo.saft`: Clapeyron is
    an independent Julia implementation of the same Gross-Sadowski equation of
    state. Agreement is bounded by *parameter provenance* (Clapeyron's tabulated
    ``m``, ``sigma``, ``epsilon`` may differ slightly from the values vendored
    here), not by the residual-Helmholtz math, so the companion tests compare
    non-associating species (whose Gross-Sadowski 2001 parameters are standard)
    and keep a modest tolerance.

    Requires a working Julia with ``Clapeyron.jl`` and the optional ``juliacall``
    package (``HAVE_CLAPEYRON``); the companion test skips unless importable.
    """
    _require("juliacall (with Clapeyron.jl)", HAVE_CLAPEYRON)
    from juliacall import Main as jl

    jl.seval("using Clapeyron")
    names = jl.convert(jl.seval("Vector{String}"), [str(c) for c in components])
    model = jl.PCSAFT(names)
    z = jl.convert(jl.seval("Vector{Float64}"), [float(v) for v in x])
    volume = 1.0 / float(rho)  # m^3 holding one mole of mixture
    p = float(jl.pressure(model, volume, float(t), z))
    return {"pressure": p, "z": p * volume / (R * float(t))}


def clapeyron_pcsaft_saturation_pressure(component: str, t: float) -> float:
    """Pure-component PC-SAFT saturation pressure (Pa) from Clapeyron.jl.

    Uses Clapeyron's ``saturation_pressure`` (its own Maxwell construction on the
    stock ``PCSAFT`` model), an oracle for `fugacio.thermo.saft.psat_saft`.
    """
    _require("juliacall (with Clapeyron.jl)", HAVE_CLAPEYRON)
    from juliacall import Main as jl

    jl.seval("using Clapeyron")
    names = jl.convert(jl.seval("Vector{String}"), [str(component)])
    model = jl.PCSAFT(names)
    psat = jl.saturation_pressure(model, float(t))
    return float(psat[0])


def modified_raoult_bubble_pressure(
    components: Sequence[str],
    x: Sequence[float],
    t: float,
    gamma: Sequence[float],
    *,
    eos: CubicEOS = PR,
) -> float:
    """Independent modified-Raoult bubble pressure ``P = sum_i x_i gamma_i Psat_i`` (Pa).

    Combines a *reference* set of activity coefficients ``gamma`` (e.g. from
    `thermo_nrtl_gamma`) with Fugacio's own EOS saturation pressures, giving
    a check on the gamma-phi bubble-point assembly that is independent of Fugacio's
    activity kernel.
    """
    arr = component_arrays(list(components))
    psat = saturation_pressures(eos, t, arr["tc"], arr["pc"], arr["omega"])
    x_arr = jnp.asarray([float(v) for v in x])
    gamma_arr = jnp.asarray([float(g) for g in gamma])
    return float(jnp.sum(x_arr * gamma_arr * psat))


# --- CoolProp property oracles --------------------------------------------------
# CoolProp wraps the reference multiparameter (Helmholtz-energy) equations of
# state and the associated transport/surface-tension correlations for ~120 pure
# fluids. That makes it the most independent oracle available for Fugacio's
# correlation stack: its property values come from entirely different functional
# forms fitted by different groups, so agreement is evidence about *accuracy*,
# not just transcription. Fluids are matched by CAS number, never by name.

_COOLPROP_BY_CAS: dict[str, str] | None = None


def _coolprop_index() -> dict[str, str]:
    """Map CAS registry number -> CoolProp fluid name for every shipped fluid."""
    global _COOLPROP_BY_CAS
    if _COOLPROP_BY_CAS is None:
        from CoolProp.CoolProp import get_fluid_param_string, get_global_param_string

        _COOLPROP_BY_CAS = {
            get_fluid_param_string(fluid, "CAS"): fluid
            for fluid in get_global_param_string("fluids_list").split(",")
        }
    return _COOLPROP_BY_CAS


def coolprop_fluid(name: str) -> str:
    """CoolProp fluid name for a database component, matched by CAS number.

    Raises `KeyError` if CoolProp has no reference EOS for the component,
    so tests can build their fluid lists with `coolprop_supports`.
    """
    _require("CoolProp", HAVE_COOLPROP)
    comp = get(name)
    index = _coolprop_index()
    if comp.cas is None or comp.cas not in index:
        raise KeyError(f"CoolProp has no pure-fluid EOS for {name!r} (CAS {comp.cas})")
    return index[comp.cas]


def coolprop_supports(name: str) -> bool:
    """Whether CoolProp ships a reference EOS for the named component."""
    if not HAVE_COOLPROP:
        return False
    try:
        coolprop_fluid(name)
    except KeyError:
        return False
    return True


def coolprop_triple_temperature(name: str) -> float:
    """Triple-point temperature (K) from CoolProp, for picking valid test states."""
    _require("CoolProp", HAVE_COOLPROP)
    from CoolProp.CoolProp import PropsSI

    return float(PropsSI("Ttriple", coolprop_fluid(name)))


def coolprop_saturation(name: str, t: float) -> dict[str, float]:
    """Reference saturation-state properties at ``t`` (K) from CoolProp.

    Always returns ``psat`` (Pa), ``rho_liquid``/``rho_vapor`` (kg/m^3),
    ``hvap`` (J/mol), and ``cp_liquid`` (J/mol/K). Transport keys
    (``mu_liquid``, ``mu_vapor``, ``k_liquid``, ``k_vapor``) and ``sigma`` (N/m)
    are included only when CoolProp has the corresponding model for the fluid.
    """
    _require("CoolProp", HAVE_COOLPROP)
    from CoolProp.CoolProp import PropsSI

    fluid = coolprop_fluid(name)
    t = float(t)
    out = {
        "psat": float(PropsSI("P", "T", t, "Q", 0, fluid)),
        "rho_liquid": float(PropsSI("D", "T", t, "Q", 0, fluid)),
        "rho_vapor": float(PropsSI("D", "T", t, "Q", 1, fluid)),
        "hvap": float(
            PropsSI("Hmolar", "T", t, "Q", 1, fluid) - PropsSI("Hmolar", "T", t, "Q", 0, fluid)
        ),
        "cp_liquid": float(PropsSI("Cpmolar", "T", t, "Q", 0, fluid)),
    }
    optional = {
        "mu_liquid": ("V", 0),
        "mu_vapor": ("V", 1),
        "k_liquid": ("L", 0),
        "k_vapor": ("L", 1),
        "sigma": ("I", 0),
    }
    for key, (prop, q) in optional.items():
        try:
            out[key] = float(PropsSI(prop, "T", t, "Q", q, fluid))
        except ValueError:
            continue  # no transport/tension model for this fluid
    return out


def coolprop_gas_state(name: str, t: float, p: float) -> dict[str, float]:
    """Reference single-phase gas properties at ``(t, p)`` from CoolProp.

    Returns ``z`` (compressibility factor) and ``rho`` (kg/m^3), plus ``mu``
    (Pa*s) and ``k`` (W/m/K) when CoolProp has transport models for the fluid.
    The caller is responsible for choosing a state that is actually gas.
    """
    _require("CoolProp", HAVE_COOLPROP)
    from CoolProp.CoolProp import PropsSI

    fluid = coolprop_fluid(name)
    t, p = float(t), float(p)
    out = {
        "z": float(PropsSI("Z", "T", t, "P", p, fluid)),
        "rho": float(PropsSI("D", "T", t, "P", p, fluid)),
    }
    for key, prop in (("mu", "V"), ("k", "L")):
        try:
            out[key] = float(PropsSI(prop, "T", t, "P", p, fluid))
        except ValueError:
            continue
    return out


# --- chemicals kernel-isolation oracles ------------------------------------------
# These reuse the ``chemicals`` library's implementation of the *same named
# correlation* with the *same inputs*, so a mismatch isolates Fugacio's kernel
# algebra (rather than data or model choice). They complement the CoolProp
# oracles above, which test accuracy against independent reference models.


def chemicals_wilke_viscosity(
    y: Sequence[float], mu: Sequence[float], mw: Sequence[float]
) -> float:
    """Reference Wilke gas-mixture viscosity (Pa*s) from ``chemicals``."""
    _require("chemicals", HAVE_CHEMICALS)
    from chemicals.viscosity import Wilke

    return float(Wilke([float(v) for v in y], [float(v) for v in mu], [float(v) for v in mw]))


def chemicals_dippr9h_conductivity(w: Sequence[float], k: Sequence[float]) -> float:
    """Reference DIPPR9H liquid-mixture thermal conductivity (W/m/K) from ``chemicals``.

    Note ``w`` are *mass* fractions, per the DIPPR9H definition.
    """
    _require("chemicals", HAVE_CHEMICALS)
    from chemicals.thermal_conductivity import DIPPR9H

    return float(DIPPR9H([float(v) for v in w], [float(v) for v in k]))


def chemicals_winterfeld_scriven_davis(
    x: Sequence[float], sigma: Sequence[float], rhom: Sequence[float]
) -> float:
    """Reference Winterfeld-Scriven-Davis mixture surface tension (N/m) from ``chemicals``.

    ``rhom`` are liquid molar densities (mol/m^3).
    """
    _require("chemicals", HAVE_CHEMICALS)
    from chemicals.interface import Winterfeld_Scriven_Davis

    return float(
        Winterfeld_Scriven_Davis(
            [float(v) for v in x], [float(v) for v in sigma], [float(v) for v in rhom]
        )
    )


# Cantera is an independent, widely used reference for chemical-reaction
# thermochemistry. Rather than rely on Cantera's own species database (whose
# formation data differ slightly from Fugacio's), we build an ideal-gas phase
# *from Fugacio's own* standard formation enthalpy/Gibbs and ideal-gas ``Cp``
# coefficients, transcribed exactly into a single-region NASA-9 polynomial. That
# isolates the comparison to the temperature-integration kernel and the
# equilibrium *solver*: agreement is then expected to machine precision, not just
# in the ballpark.
#
# The NASA-9 standard-state functions (per Cantera's ``Nasa9Poly1``) are, with
# coefficients ``[a0..a6, b0, b1]``::
#
#     Cp/R = a0 T^-2 + a1 T^-1 + a2 + a3 T + a4 T^2 + a5 T^3 + a6 T^4
#     H/RT = -a0 T^-2 + a1 ln(T)/T + a2 + a3 T/2 + a4 T^2/3 + a5 T^3/4
#            + a6 T^4/5 + b0/T
#     S/R  = -a0 T^-2/2 - a1 T^-1 + a2 ln(T) + a3 T + a4 T^2/2 + a5 T^3/3
#            + a6 T^4/4 + b1
#
# Fugacio's ``Cp/R = a + b T + c T^2 + d/T^2 + e T^3`` maps to
# ``a0=d, a2=a, a3=b, a4=c, a5=e`` (``a1=a6=0``). The integration constants
# ``b0, b1`` are fixed so the species enthalpy at ``T_REF`` equals its enthalpy of
# formation and its entropy equals the formation entropy ``(Hf - Gf)/T_REF``:
# elements cancel in any balanced reaction, so the reaction sums reproduce
# Fugacio's ``DH_rxn``/``DS_rxn``/``K(T)`` exactly. The phase reference pressure
# is pinned to `P_REF` (1 bar) so Cantera's
# equilibrium reaction quotient uses the same standard state as Fugacio.

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def _element_composition(formula: str) -> dict[str, int]:
    """Parse a molecular formula such as ``"C2H6O"`` into ``{element: count}``."""
    counts: dict[str, int] = {}
    consumed = 0
    for symbol, number in _FORMULA_TOKEN.findall(formula):
        counts[symbol] = counts.get(symbol, 0) + (int(number) if number else 1)
        consumed += len(symbol) + len(number)
    if consumed != len(formula):
        raise ValueError(f"cannot parse molecular formula {formula!r}")
    return counts


def _nasa9_data(comp: Component) -> list[float]:
    """Single-region NASA-9 coefficients ``[a0..a6, b0, b1]`` for a component."""
    if comp.hform_ig is None or comp.gform_ig is None or comp.cp_ig is None:
        raise RuntimeError(
            f"component {comp.name!r} lacks formation/Cp data needed for the cantera oracle"
        )
    cp = comp.cp_ig
    a0, a2, a3, a4, a5 = cp.d, cp.a, cp.b, cp.c, cp.e  # T^-2, 1, T, T^2, T^3
    t0 = T_REF
    s_form = (comp.hform_ig - comp.gform_ig) / t0
    b0 = comp.hform_ig / R - (-a0 / t0 + a2 * t0 + a3 * t0**2 / 2 + a4 * t0**3 / 3 + a5 * t0**4 / 4)
    b1 = s_form / R - (
        -a0 / (2 * t0**2) + a2 * math.log(t0) + a3 * t0 + a4 * t0**2 / 2 + a5 * t0**3 / 3
    )
    return [a0, 0.0, a2, a3, a4, a5, 0.0, b0, b1]


def _cantera_ideal_gas(components: Sequence[str]) -> Any:
    """Build a Cantera ideal-gas ``Solution`` from Fugacio's formation/Cp data."""
    import cantera as ct

    blocks: list[str] = []
    names: list[str] = []
    for i, name in enumerate(components):
        comp = get(name)
        species = f"s{i}"
        names.append(species)
        elements = ", ".join(f"{el}: {n}" for el, n in _element_composition(comp.formula).items())
        data = ", ".join(repr(float(x)) for x in _nasa9_data(comp))
        blocks.append(
            f"- name: {species}\n"
            f"  composition: {{{elements}}}\n"
            f"  thermo:\n"
            f"    model: NASA9\n"
            f"    reference-pressure: {P_REF!r}\n"
            f"    temperature-ranges: [100.0, 6000.0]\n"
            f"    data:\n"
            f"    - [{data}]\n"
        )
    yaml = (
        "phases:\n"
        "- name: gas\n"
        "  thermo: ideal-gas\n"
        f"  species: [{', '.join(names)}]\n"
        f"  state: {{T: {T_REF!r}, P: {P_REF!r}}}\n"
        "species:\n" + "".join(blocks)
    )
    return ct.Solution(yaml=yaml)


def cantera_reaction_properties(reaction: Reaction, t: float) -> dict[str, float]:
    """Reference ``DH_rxn``/``DS_rxn``/``DG_rxn``/``ln K``/``K`` from Cantera (J, K units).

    Cantera evaluates the standard-state species enthalpies and Gibbs energies
    from the NASA-9 polynomials built out of Fugacio's formation data; the
    reaction values are the stoichiometric sums. Use to cross-check
    `fugacio.thermo.reactions.reaction_properties`.
    """
    _require("cantera", HAVE_CANTERA)
    gas = _cantera_ideal_gas(reaction.components)
    gas.TP = float(t), float(P_REF)
    nu = [float(v) for v in reaction.nu]
    hrt = [float(v) for v in gas.standard_enthalpies_RT]
    grt = [float(v) for v in gas.standard_gibbs_RT]
    sum_h = sum(n * h for n, h in zip(nu, hrt, strict=True))
    sum_g = sum(n * g for n, g in zip(nu, grt, strict=True))
    delta_h = R * float(t) * sum_h
    delta_g = R * float(t) * sum_g
    ln_k = -sum_g
    return {
        "delta_h": delta_h,
        "delta_s": (delta_h - delta_g) / float(t),
        "delta_g": delta_g,
        "ln_k": ln_k,
        "k": math.exp(ln_k),
    }


def cantera_equilibrium_composition(
    components: Sequence[str], n_feed: Sequence[float], t: float, p: float
) -> list[float]:
    """Reference equilibrium mole fractions from Cantera's ``equilibrate('TP')``.

    Cantera minimises the Gibbs energy subject to element conservation from the
    feed, giving an equilibrium composition that is independent of Fugacio's
    extent-of-reaction solver. Use to cross-check
    `fugacio.thermo.reaction_equilibrium.equilibrium`.
    """
    _require("cantera", HAVE_CANTERA)
    gas = _cantera_ideal_gas(components)
    gas.TPX = float(t), float(p), [max(float(v), 0.0) for v in n_feed]
    gas.equilibrate("TP")
    return [float(v) for v in gas.X]


__all__ = [
    "HAVE_CANTERA",
    "HAVE_CHEMICALS",
    "HAVE_CLAPEYRON",
    "HAVE_COOLPROP",
    "HAVE_THERMO",
    "cantera_equilibrium_composition",
    "cantera_reaction_properties",
    "cas_for",
    "chemicals_dippr9h_conductivity",
    "chemicals_wilke_viscosity",
    "chemicals_winterfeld_scriven_davis",
    "clapeyron_gamma",
    "clapeyron_pcsaft",
    "clapeyron_pcsaft_saturation_pressure",
    "coolprop_fluid",
    "coolprop_gas_state",
    "coolprop_saturation",
    "coolprop_supports",
    "coolprop_triple_temperature",
    "modified_raoult_bubble_pressure",
    "thermo_nrtl_gamma",
    "thermo_unifac_gamma",
    "thermo_uniquac_gamma",
    "thermo_wilson_gamma",
]
