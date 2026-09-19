"""Model bridge: turn component *names* + a method choice into a property package.

A flowsheet works in component *names*. `package_for` resolves those names to
the curated constants (reusing the cached lookup in `fugacio.sim.properties`),
assembles the activity model from the curated binary database (NRTL /
UNIQUAC) or predictive group contribution (UNIFAC / modified UNIFAC), or the
PC-SAFT and reference-fluid parameters, and returns a ready, differentiable
`fugacio.thermo.PropertyPackage` (equilibrium *and* energy) with explicit
parameter provenance. Its result is what every unit operation, the rigorous
column, the phase diagrams, and the equation-oriented engine accept through
their ``model`` argument, so a flowsheet switches from Peng-Robinson to NRTL by
changing one method name, and stays end-to-end differentiable, including with
respect to the thermodynamic parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import jax
from jax import Array

from fugacio.sim.properties import _resolve, default_package, resolve_package
from fugacio.thermo import (
    PR,
    RK,
    SRK,
    VDW,
    CubicEOS,
    HelmholtzPackage,
    PropertyPackage,
    gamma_phi_package,
    get,
    helmholtz_package,
    kij_from_database,
    modified_unifac_activity,
    nrtl_from_database,
    reference_fluid,
    saft_package,
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


def helmholtz_package_for(component: str) -> HelmholtzPackage:
    """One-component reference-fluid package for a named pure fluid (water, CO2, ...).

    The name is resolved through `fugacio.thermo.reference_fluid`, so any of the
    26 vendored multiparameter formulations is accepted.
    """
    return replace(
        helmholtz_package(reference_fluid(component)), component_names=(get(component).name,)
    )


_GAMMA_PHI_OPTIONS = frozenset({"eos", "kij", "vapor", "poynting", "phi_saturation"})


def _build(names: tuple[str, ...], key: str, options: dict[str, Any]) -> PropertyPackage:
    """The package for ``names`` by method ``key``, consuming its ``options``.

    Raises:
        TypeError: For an option the method doesn't accept.
        ValueError: For an unknown method or a multi-component reference fluid.
    """
    options = dict(options)

    def take(allowed: frozenset[str]) -> dict[str, Any]:
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise TypeError(f"method {key!r} doesn't accept option(s) {unknown}")
        return options

    comps = list(names)
    tc, pc, omega, _, cp = _resolve(names)
    if key in _CUBICS:
        opts = take(frozenset({"kij", "use_database_kij"}))
        kij = opts.get("kij")
        if kij is None and opts.get("use_database_kij", False):
            kij = kij_from_database(comps)
        return default_package(comps, eos=_CUBICS[key], kij=kij)
    if key in ("nrtl", "uniquac", "unifac", "dortmund"):
        extra = {"nrtl": {"strict", "alpha_default"}, "uniquac": {"strict"}}.get(key, set())
        opts = take(_GAMMA_PHI_OPTIONS | frozenset(extra))
        activity: Any
        if key == "nrtl":
            activity = nrtl_from_database(
                comps,
                strict=opts.pop("strict", False),
                alpha_default=opts.pop("alpha_default", 0.3),
            )
        elif key == "uniquac":
            activity = uniquac_from_database(comps, strict=opts.pop("strict", False))
        else:
            activity = UnifacModel(components=names, dortmund=key == "dortmund")
        return replace(
            gamma_phi_package(activity, tc, pc, omega, cp, **opts), component_names=names
        )
    if key == "pcsaft":
        opts = take(frozenset({"kij", "use_database_kij"}))
        params = saft_parameters_for(
            comps, kij=opts.get("kij"), use_database_kij=opts.get("use_database_kij", True)
        )
        return replace(saft_package(params, tc, pc, omega, cp), component_names=names)
    if key in ("iapws", "helmholtz", "reference"):
        take(frozenset())
        if len(comps) != 1:
            raise ValueError("the reference-fluid package describes exactly one pure component")
        return helmholtz_package_for(comps[0])
    raise ValueError(f"unknown thermodynamic method {key!r}; choose one of {METHODS}")


def package_for(
    components: Sequence[str],
    method: str = "pr",
    *,
    parameter_policy: str = "strict",
    measured_fit: Any = None,
    **options: Any,
) -> PropertyPackage:
    """Build the property package for named ``components`` by ``method``.

    This is the constructor a flowsheet author reaches for; every unit
    operation, the rigorous column, the two-sided heat exchanger, the phase
    diagrams, and the equation-oriented engine take its result through their
    ``model`` argument.

    Args:
        components: Component names (from the curated database).
        method: One of `METHODS`:

            * ``"pr"`` / ``"srk"`` / ``"rk"`` / ``"vdw"``: phi-phi cubic package
              (options: ``kij``, ``use_database_kij``);
            * ``"nrtl"`` / ``"uniquac"``: gamma-phi package with the curated
              binary parameters (options: ``vapor``, ``poynting``,
              ``phi_saturation``, ``eos``, ``kij``, and ``alpha_default`` for NRTL);
            * ``"unifac"`` / ``"dortmund"``: gamma-phi package with predictive
              (modified) UNIFAC (same options, no fitted parameters);
            * ``"pcsaft"``: PC-SAFT package (options: ``kij``, ``use_database_kij``);
            * ``"iapws"``: a single reference fluid (one component only).
        parameter_policy: ``"strict"`` (default) raises for missing NRTL/UNIQUAC
            interactions; ``"allow_ideal"`` explicitly permits zero interactions.
            Cubic zero-kij assumptions remain visible in the package evidence.
        measured_fit: A converged `fugacio.thermo.measured_regression.MeasuredFit`
            for binary NRTL. Its observed bounds travel with the package, and
            checked calculations reject extrapolation unless their acceptance
            policy explicitly permits it.
        **options: Method options as listed above.

    Returns:
        A `fugacio.thermo.PropertyPackage` with explicit provenance.

    Raises:
        ValueError: For an unknown method, an invalid policy, or an unusable fit.
        TypeError: For an option the method doesn't accept.
        KeyError: For missing curated parameters under the strict policy.
    """
    from fugacio.thermo.measured_regression import MeasuredFit
    from fugacio.thermo.provenance import PackageEvidence, PairEvidence, database_evidence

    if parameter_policy not in ("strict", "allow_ideal"):
        raise ValueError("parameter_policy must be strict or allow_ideal")
    names = tuple(get(c).name for c in components)
    if len(set(names)) != len(names) or not names:
        raise ValueError("package needs distinct components")
    key = method.lower()
    if "strict" in options:
        raise TypeError("use parameter_policy='allow_ideal' instead of strict=False")
    allow = parameter_policy == "allow_ideal"
    if measured_fit is not None:
        if key != "nrtl" or not isinstance(measured_fit, MeasuredFit) or len(names) != 2:
            raise ValueError("measured_fit requires a binary NRTL package")
        if not measured_fit.diagnostics.converged:
            raise ValueError("unconverged or unidentifiable measured fit")
        if options:
            raise ValueError(
                "measured fits retain the exact ideal-vapor PR reference used in fitting"
            )
        activity = measured_fit.model((names[0], names[1]))
        tc, pc, omega, _, cp = _resolve(names)
        fitted = replace(gamma_phi_package(activity, tc, pc, omega, cp), component_names=names)
        xr = measured_fit.composition_range
        if names != measured_fit.components:
            xr = (1 - xr[1], 1 - xr[0])
        evidence = PackageEvidence(
            "nrtl",
            names,
            (PairEvidence((names[0], names[1]), "measured_fit", "NIST ThermoML regression"),),
            (
                "Ideal vapor; PR saturation pressure; no Poynting or saturation-phi correction.",
                "Observed training bounds; independent qualification must be inspected separately.",
            ),
            measured_fit.sources,
            measured_fit.temperature_range,
            measured_fit.pressure_range,
            xr,
        )
        return replace(fitted, evidence=evidence)
    if key in ("nrtl", "uniquac"):
        options = {**options, "strict": not allow}
    pkg = _build(names, key, options)
    evidence = database_evidence(
        names,
        key,
        allow_ideal=allow,
        use_database_kij=options.get("use_database_kij", key == "pcsaft"),
        explicit_kij=options.get("kij") is not None,
    )
    if key in ("unifac", "dortmund") and (
        evidence.missing_components or any(pair.kind == "missing" for pair in evidence.pairs)
    ):
        raise KeyError(
            "predictive method lacks group assignments or interaction parameters; inspect evidence"
        )
    if key in ("nrtl", "uniquac", "unifac", "dortmund"):
        evidence = replace(
            evidence,
            assumptions=(
                *evidence.assumptions,
                f"Vapor model: {options.get('vapor', 'ideal')}; "
                "PR saturation reference unless eos overridden.",
                f"Poynting correction: {bool(options.get('poynting', False))}; "
                f"saturation fugacity correction: {bool(options.get('phi_saturation', False))}.",
            ),
        )
    return replace(pkg, evidence=evidence)  # type: ignore[type-var]


__all__ = [
    "METHODS",
    "UnifacModel",
    "helmholtz_package_for",
    "package_for",
    "resolve_package",
]
