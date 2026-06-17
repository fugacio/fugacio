"""Reference-fluid registry: published Helmholtz EOS as JAX-ready dataclasses.

Each `HelmholtzFluid` packages one peer-reviewed multiparameter
equation of state (IAPWS-95 for water, Span & Wagner for CO2, Setzmann &
Wagner for methane, ...) in the reduced-Helmholtz form ``alpha(delta, tau)``
together with its saturation ancillary equations and surface-tension
correlation. The coefficient tables live in the generated
`fugacio.thermo.helmholtz._data` (see ``scripts/gen_helmholtz.py`` for
provenance); this module turns them into frozen dataclasses registered as JAX
pytrees, so every coefficient is a differentiable leaf and every downstream
property can be differentiated with respect to the model itself as well as the
state.

Fluids are looked up by Fugacio component name (``"water"``,
``"carbon dioxide"``, ...) with a few common aliases (``"steam"``, ``"co2"``,
``"r744"``); construction is cached, so repeated lookups are free.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import cache

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.helmholtz._data import FLUID_DATA


@dataclass(frozen=True)
class Ancillary:
    """One saturation ancillary equation (initial guesses for the Maxwell solve).

    Evaluates ``reducing * exp(f * sum(n_i * theta**t_i))`` with
    ``theta = 1 - T/t_reducing`` and ``f = t_reducing/T`` when ``using_tau_r``
    (else 1); the ``noexp`` family is ``reducing * (1 + sum(n_i * theta**t_i))``.

    Attributes:
        n: Coefficients of the expansion.
        t: Exponents of ``theta``.
        reducing: Reducing value (Pa for pressure, mol/m^3 for density).
        t_reducing: Reducing temperature (K).
        using_tau_r: Whether the exponential is premultiplied by ``Tc/T``.
        noexp: Whether the expansion is affine rather than exponential.
    """

    n: Array
    t: Array
    reducing: float
    t_reducing: float
    using_tau_r: bool
    noexp: bool


@dataclass(frozen=True)
class HelmholtzFluid:
    """A pure fluid described by a reference multiparameter Helmholtz EOS.

    The reduced Helmholtz energy ``alpha = a/(R T)`` is split into an ideal-gas
    part and a residual part, each a sum of standard term families evaluated at
    ``delta = rho/rho_reducing`` and ``tau = t_reducing/T``
    (`fugacio.thermo.helmholtz.terms`). Coefficient arrays are pytree
    leaves; scalar metadata (bounds, reducing constants) is static.

    Note ``gas_constant`` is the molar gas constant *the EOS was published
    with*, which may differ from CODATA in the last digits; using it is
    required to reproduce reference values exactly.
    """

    name: str
    cas: str
    bibtex_eos: str
    molar_mass: float
    gas_constant: float
    t_reducing: float
    rho_reducing: float
    t_critical: float
    p_critical: float
    rho_critical: float
    t_triple: float
    p_triple: float
    t_max: float
    p_max: float
    rho_max: float
    acentric: float
    sigma_tc: float
    # Ideal part: lead (a1 + a2*tau + ln delta), a*ln tau, power and
    # Planck-Einstein expansions.
    lead_a1: Array
    lead_a2: Array
    log_tau: Array
    ideal_power_n: Array
    ideal_power_t: Array
    pe_n: Array
    pe_t: Array
    # Residual part: power/exponential, Gaussian-bell, non-analytic critical
    # (water, CO2) and GaoB (ammonia) term families.
    power_n: Array
    power_t: Array
    power_d: Array
    power_l: Array
    gauss_n: Array
    gauss_t: Array
    gauss_d: Array
    gauss_eta: Array
    gauss_beta: Array
    gauss_gamma: Array
    gauss_epsilon: Array
    na_n: Array
    na_a: Array
    na_b: Array
    na_beta: Array
    na_big_a: Array
    na_big_b: Array
    na_big_c: Array
    na_big_d: Array
    gaob_n: Array
    gaob_t: Array
    gaob_d: Array
    gaob_eta: Array
    gaob_beta: Array
    gaob_gamma: Array
    gaob_epsilon: Array
    gaob_b: Array
    # Surface tension sum(a_i * (1 - T/sigma_tc)**e_i) (N/m).
    sigma_a: Array
    sigma_e: Array
    # Saturation ancillaries (initial guesses; the EOS itself is the truth).
    anc_psat: Ancillary
    anc_rho_liquid: Ancillary
    anc_rho_vapor: Ancillary


_META_FIELDS = (
    "name",
    "cas",
    "bibtex_eos",
    "molar_mass",
    "gas_constant",
    "t_reducing",
    "rho_reducing",
    "t_critical",
    "p_critical",
    "rho_critical",
    "t_triple",
    "p_triple",
    "t_max",
    "p_max",
    "rho_max",
    "acentric",
    "sigma_tc",
)

jax.tree_util.register_dataclass(
    Ancillary,
    data_fields=["n", "t"],
    meta_fields=["reducing", "t_reducing", "using_tau_r", "noexp"],
)
jax.tree_util.register_dataclass(
    HelmholtzFluid,
    data_fields=[f.name for f in fields(HelmholtzFluid) if f.name not in _META_FIELDS],
    meta_fields=list(_META_FIELDS),
)

#: Common alternative names accepted by `reference_fluid`.
ALIASES: dict[str, str] = {
    "steam": "water",
    "h2o": "water",
    "co2": "carbon dioxide",
    "r744": "carbon dioxide",
    "co": "carbon monoxide",
    "n2": "nitrogen",
    "o2": "oxygen",
    "h2": "hydrogen",
    "h2s": "hydrogen sulfide",
    "so2": "sulfur dioxide",
    "r717": "ammonia",
    "nh3": "ammonia",
    "ch4": "methane",
    "r290": "propane",
    "butane": "n-butane",
    "r600": "n-butane",
    "r600a": "isobutane",
    "pentane": "n-pentane",
    "hexane": "n-hexane",
    "octane": "n-octane",
    "ethene": "ethylene",
    "r1150": "ethylene",
    "propene": "propylene",
    "r1270": "propylene",
}

_CANONICAL: dict[str, str] = {name.lower(): name for name in FLUID_DATA}


def _array(values: object) -> Array:
    return jnp.asarray(values, dtype=float).reshape(-1)


def _columns(rows: tuple[tuple[float, ...], ...], width: int) -> tuple[Array, ...]:
    if not rows:
        return tuple(jnp.zeros((0,)) for _ in range(width))
    matrix = jnp.asarray(rows, dtype=float)
    return tuple(matrix[:, i] for i in range(width))


def _ancillary(raw: tuple, t_reducing: float) -> Ancillary:
    reducing, using_tau_r, noexp, coeffs = raw
    n, t = _columns(tuple(coeffs), 2)
    return Ancillary(
        n=n,
        t=t,
        reducing=float(reducing),
        t_reducing=t_reducing,
        using_tau_r=bool(using_tau_r),
        noexp=bool(noexp),
    )


@cache
def _build(name: str) -> HelmholtzFluid:
    raw = FLUID_DATA[name]
    t_reducing = float(raw["t_reducing"])
    power = _columns(raw["power"], 4)
    gauss = _columns(raw["gaussian"], 7)
    na = _columns(raw["nonanalytic"], 8)
    gaob = _columns(raw["gaob"], 8)
    sigma = _columns(raw["sigma"], 2)
    ideal_power = _columns(raw["ideal_power"], 2)
    pe = _columns(raw["planck_einstein"], 2)
    return HelmholtzFluid(
        name=name,
        cas=raw["cas"],
        bibtex_eos=raw["bibtex_eos"],
        molar_mass=float(raw["molar_mass"]),
        gas_constant=float(raw["gas_constant"]),
        t_reducing=t_reducing,
        rho_reducing=float(raw["rho_reducing"]),
        t_critical=float(raw["t_critical"]),
        p_critical=float(raw["p_critical"]),
        rho_critical=float(raw["rho_critical"]),
        t_triple=float(raw["t_triple"]),
        p_triple=float(raw["p_triple"]),
        t_max=float(raw["t_max"]),
        p_max=float(raw["p_max"]),
        rho_max=float(raw["rho_max"]),
        acentric=float(raw["acentric"]),
        sigma_tc=float(raw["sigma_tc"]),
        lead_a1=jnp.asarray(raw["lead"][0], dtype=float),
        lead_a2=jnp.asarray(raw["lead"][1], dtype=float),
        log_tau=jnp.asarray(raw["log_tau"], dtype=float),
        ideal_power_n=ideal_power[0],
        ideal_power_t=ideal_power[1],
        pe_n=pe[0],
        pe_t=pe[1],
        power_n=power[0],
        power_t=power[1],
        power_d=power[2],
        power_l=power[3],
        gauss_n=gauss[0],
        gauss_t=gauss[1],
        gauss_d=gauss[2],
        gauss_eta=gauss[3],
        gauss_beta=gauss[4],
        gauss_gamma=gauss[5],
        gauss_epsilon=gauss[6],
        na_n=na[0],
        na_a=na[1],
        na_b=na[2],
        na_beta=na[3],
        na_big_a=na[4],
        na_big_b=na[5],
        na_big_c=na[6],
        na_big_d=na[7],
        gaob_n=gaob[0],
        gaob_t=gaob[1],
        gaob_d=gaob[2],
        gaob_eta=gaob[3],
        gaob_beta=gaob[4],
        gaob_gamma=gaob[5],
        gaob_epsilon=gaob[6],
        gaob_b=gaob[7],
        sigma_a=sigma[0],
        sigma_e=sigma[1],
        anc_psat=_ancillary(raw["anc_psat"], t_reducing),
        anc_rho_liquid=_ancillary(raw["anc_rho_liquid"], t_reducing),
        anc_rho_vapor=_ancillary(raw["anc_rho_vapor"], t_reducing),
    )


def reference_fluid(name: str) -> HelmholtzFluid:
    """The reference Helmholtz EOS for a named fluid.

    Args:
        name: Fugacio component name (``"water"``, ``"carbon dioxide"``,
            ``"R134a"``, ...) or a common alias (``"steam"``, ``"co2"``); case
            insensitive.

    Returns:
        The cached `HelmholtzFluid`.

    Raises:
        KeyError: If no reference EOS is vendored for the name (see
            `reference_fluid_names`).
    """
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    canonical = _CANONICAL.get(key)
    if canonical is None:
        available = ", ".join(sorted(FLUID_DATA))
        raise KeyError(f"no reference Helmholtz EOS for {name!r}; available: {available}")
    return _build(canonical)


def reference_fluid_names() -> tuple[str, ...]:
    """Names of every fluid with a vendored reference EOS, sorted."""
    return tuple(sorted(FLUID_DATA))


def has_reference_fluid(name: str) -> bool:
    """Whether `reference_fluid` knows the named fluid."""
    key = name.strip().lower()
    return ALIASES.get(key, key) in _CANONICAL


__all__ = [
    "ALIASES",
    "Ancillary",
    "HelmholtzFluid",
    "has_reference_fluid",
    "reference_fluid",
    "reference_fluid_names",
]
