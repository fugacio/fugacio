"""Model bridge: turn component *names* + a method choice into an equilibrium model.

The thermo equilibrium models (`EOSModel` and
`GammaPhiModel`) take *array* constants (``tc``, ``pc``,
``omega``) and, for gamma-phi, an activity model. A flowsheet, however, works in
component *names*. This module resolves names to those arrays (reusing the cached
lookup in `fugacio.sim.properties`) and assembles the activity model from the
curated binary database (NRTL / UNIQUAC) or predictive group contribution
(UNIFAC / modified UNIFAC), returning a ready, differentiable
`EquilibriumModel`.

The returned object is what the gamma-phi-aware unit operations
(`fugacio.sim.separations`) and the T-x-y / P-x-y / azeotrope helpers
(`fugacio.sim.diagrams`) consume, so a flowsheet can switch from
Peng-Robinson to NRTL by swapping one constructor call, and stays end-to-end
differentiable, including with respect to the activity-model parameters.

`package_for` is the one-stop constructor for a full
`fugacio.thermo.PropertyPackage` (equilibrium *and* energy) by method name:
``"pr"``, ``"srk"``, ``"nrtl"``, ``"unifac"``, ``"pcsaft"``, ``"iapws"``, and so
on. Its result is what every energy-balanced unit operation, the rigorous
column, and the equation-oriented engine accept through their ``model``
argument; `as_package` upgrades any of the bare equilibrium models above to the
same interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
from jax import Array

from fugacio.sim.properties import _resolve, as_package, default_package, resolve_package
from fugacio.thermo import (
    PR,
    RK,
    SRK,
    VDW,
    CubicEOS,
    EOSModel,
    GammaPhiModel,
    HelmholtzPackage,
    PropertyPackage,
    SAFTModel,
    eos_model,
    gamma_phi_model,
    helmholtz_package,
    kij_from_database,
    modified_unifac_activity,
    nrtl_from_database,
    reference_fluid,
    saft_model,
    saft_parameters_for,
    unifac_activity,
    uniquac_from_database,
)

ArrayLike = Array | float

_CUBICS: dict[str, CubicEOS] = {"pr": PR, "srk": SRK, "rk": RK, "vdw": VDW}

#: Method names accepted by `package_for`.
METHODS: tuple[str, ...] = (
    "pr",
    "srk",
    "rk",
    "vdw",
    "nrtl",
    "uniquac",
    "unifac",
    "dortmund",
    "pcsaft",
    "iapws",
)


def eos_model_for(
    components: Sequence[str],
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    use_database_kij: bool = False,
) -> EOSModel:
    """Build an `EOSModel` for named ``components``.

    Pass ``use_database_kij=True`` to fill the binary interaction matrix from the
    curated ChemSep Peng-Robinson ``k_ij`` set (`fugacio.thermo.kij_from_database`);
    pairs without a curated value stay at zero. An explicit ``kij`` takes precedence.
    """
    tc, pc, omega, _, _ = _resolve(tuple(components))
    if kij is None and use_database_kij:
        kij = kij_from_database(list(components))
    return eos_model(tc, pc, omega, kij=kij, eos=eos)


def nrtl_model_for(
    components: Sequence[str],
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
    strict: bool = False,
    alpha_default: float = 0.3,
) -> GammaPhiModel:
    """Gamma-phi model with NRTL liquid from the curated binary database.

    Pairs absent from the database default to athermal interaction unless
    ``strict=True``; see `fugacio.thermo.nrtl_from_database`.
    """
    tc, pc, omega, _, _ = _resolve(tuple(components))
    activity = nrtl_from_database(list(components), strict=strict, alpha_default=alpha_default)
    return gamma_phi_model(
        activity,
        tc,
        pc,
        omega,
        kij=kij,
        eos=eos,
        vapor=vapor,
        poynting=poynting,
        phi_saturation=phi_saturation,
    )


def uniquac_model_for(
    components: Sequence[str],
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
    strict: bool = False,
) -> GammaPhiModel:
    """Gamma-phi model with UNIQUAC liquid from the curated database (with ``r``/``q``)."""
    tc, pc, omega, _, _ = _resolve(tuple(components))
    activity = uniquac_from_database(list(components), strict=strict)
    return gamma_phi_model(
        activity,
        tc,
        pc,
        omega,
        kij=kij,
        eos=eos,
        vapor=vapor,
        poynting=poynting,
        phi_saturation=phi_saturation,
    )


@dataclass(frozen=True)
class UnifacModel:
    """An `ActivityModel` adapter over predictive UNIFAC.

    Wraps `fugacio.thermo.unifac_activity` (classic, Hansen VLE parameters)
    or `fugacio.thermo.modified_unifac_activity` (Dortmund, T-dependent) so
    that group-contribution predictions present the same ``ln_gamma(x, T)`` API as
    the fitted activity models. Carries no fitted leaves: it is a pure predictor
    keyed by the (static) component names.
    """

    components: tuple[str, ...]
    dortmund: bool

    def ln_gamma(self, x: Array, t: ArrayLike) -> Array:
        """Log activity coefficients predicted by (modified) UNIFAC."""
        names = list(self.components)
        if self.dortmund:
            return modified_unifac_activity(names, x, t)
        return unifac_activity(names, x, t)


jax.tree_util.register_dataclass(
    UnifacModel, data_fields=[], meta_fields=["components", "dortmund"]
)


def saft_model_for(
    components: Sequence[str],
    *,
    kij: Array | None = None,
    use_database_kij: bool = True,
) -> SAFTModel:
    """Build a PC-SAFT `fugacio.thermo.SAFTModel` for named components.

    Resolves the component names to PC-SAFT parameters from the curated bank
    (`fugacio.thermo.saft_parameters_for`) and to the critical constants used to
    seed the flash K-values, returning a ready, differentiable
    `fugacio.thermo.EquilibriumModel`. Switching a flowsheet to PC-SAFT, the
    molecular-based route that handles associating fluids (water, alcohols), is
    then a single constructor swap from `eos_model_for` / `nrtl_model_for`.

    Args:
        components: Component names present in the PC-SAFT parameter bank.
        kij: Explicit ``(n, n)`` binary-correction matrix; takes precedence over
            the curated bank.
        use_database_kij: Fill binary corrections from the curated PC-SAFT
            ``k_ij`` set when ``kij`` is not given.

    Returns:
        A `fugacio.thermo.SAFTModel` over the named components.

    Raises:
        KeyError: If any component lacks curated PC-SAFT parameters.
    """
    tc, pc, omega, _, _ = _resolve(tuple(components))
    params = saft_parameters_for(list(components), kij=kij, use_database_kij=use_database_kij)
    return saft_model(params, tc, pc, omega)


def unifac_model_for(
    components: Sequence[str],
    *,
    dortmund: bool = False,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    vapor: str = "ideal",
    poynting: bool = False,
    phi_saturation: bool = False,
) -> GammaPhiModel:
    """Gamma-phi model with a predictive UNIFAC liquid (no fitted parameters needed).

    Set ``dortmund=True`` for modified UNIFAC (Dortmund) with temperature-dependent
    group interactions; otherwise classic UNIFAC is used.
    """
    tc, pc, omega, _, _ = _resolve(tuple(components))
    activity = UnifacModel(components=tuple(components), dortmund=dortmund)
    return gamma_phi_model(
        activity,
        tc,
        pc,
        omega,
        kij=kij,
        eos=eos,
        vapor=vapor,
        poynting=poynting,
        phi_saturation=phi_saturation,
    )


def helmholtz_package_for(component: str) -> HelmholtzPackage:
    """One-component reference-fluid package for a named pure fluid (water, CO2, ...).

    The name is resolved through `fugacio.thermo.reference_fluid`, so any of the
    26 vendored multiparameter formulations is accepted.
    """
    return helmholtz_package(reference_fluid(component))


def package_for(
    components: Sequence[str],
    method: str = "pr",
    **options: Any,
) -> PropertyPackage:
    """Build the property package for named ``components`` by ``method``.

    This is the constructor a flowsheet author reaches for. Every energy-balanced
    unit (`fugacio.sim.units`), the rigorous column
    (`fugacio.sim.distillation`), the two-sided heat exchanger, and the
    equation-oriented engine take its result through their ``model`` argument.

    Args:
        components: Component names (from the curated database).
        method: One of `METHODS`:

            * ``"pr"`` / ``"srk"`` / ``"rk"`` / ``"vdw"``: phi-phi cubic package
              (options: ``kij``, ``use_database_kij``);
            * ``"nrtl"`` / ``"uniquac"``: gamma-phi package with the curated
              binary parameters (options: ``vapor``, ``poynting``,
              ``phi_saturation``, ``strict``, ``eos``, ``kij``, and
              ``alpha_default`` for NRTL);
            * ``"unifac"`` / ``"dortmund"``: gamma-phi package with predictive
              (modified) UNIFAC (same options);
            * ``"pcsaft"``: PC-SAFT package (options: ``kij``, ``use_database_kij``);
            * ``"iapws"``: a single reference fluid (one component only).
        **options: Forwarded to the underlying model factory as listed above.

    Returns:
        A `fugacio.thermo.PropertyPackage` over ``components``.

    Raises:
        ValueError: for an unknown ``method`` or a multi-component ``"iapws"``.
    """
    key = method.lower()
    comps = list(components)
    if key in _CUBICS:
        kij = options.get("kij")
        if kij is None and options.get("use_database_kij", False):
            kij = kij_from_database(comps)
        return default_package(comps, eos=_CUBICS[key], kij=kij)
    if key == "nrtl":
        return as_package(nrtl_model_for(comps, **options), comps)
    if key == "uniquac":
        return as_package(uniquac_model_for(comps, **options), comps)
    if key in ("unifac", "dortmund"):
        return as_package(unifac_model_for(comps, dortmund=key == "dortmund", **options), comps)
    if key == "pcsaft":
        return as_package(saft_model_for(comps, **options), comps)
    if key in ("iapws", "helmholtz", "reference"):
        if len(comps) != 1:
            raise ValueError("the reference-fluid package describes exactly one pure component")
        return helmholtz_package_for(comps[0])
    raise ValueError(f"unknown thermodynamic method {method!r}; choose one of {METHODS}")


__all__ = [
    "METHODS",
    "UnifacModel",
    "as_package",
    "eos_model_for",
    "helmholtz_package_for",
    "nrtl_model_for",
    "package_for",
    "resolve_package",
    "saft_model_for",
    "unifac_model_for",
    "uniquac_model_for",
]
