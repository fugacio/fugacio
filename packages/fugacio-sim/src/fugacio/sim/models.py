"""Model bridge: turn component *names* + a method choice into an equilibrium model.

The thermo equilibrium models -- `EOSModel` and
`GammaPhiModel` -- take *array* constants (``tc``, ``pc``,
``omega``) and, for gamma-phi, an activity model. A flowsheet, however, works in
component *names*. This module resolves names to those arrays (reusing the cached
lookup in `fugacio.sim.properties`) and assembles the activity model from the
curated binary database (NRTL / UNIQUAC) or predictive group contribution
(UNIFAC / modified UNIFAC), returning a ready, differentiable
`EquilibriumModel`.

The returned object is what the gamma-phi-aware unit operations
(`fugacio.sim.separations`) and the T-x-y / P-x-y / azeotrope helpers
(`fugacio.sim.diagrams`) consume, so a flowsheet can switch from
Peng-Robinson to NRTL by swapping one constructor call -- and stays end-to-end
differentiable, including with respect to the activity-model parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
from jax import Array

from fugacio.sim.properties import _resolve
from fugacio.thermo import (
    PR,
    CubicEOS,
    EOSModel,
    GammaPhiModel,
    eos_model,
    gamma_phi_model,
    kij_from_database,
    modified_unifac_activity,
    nrtl_from_database,
    unifac_activity,
    uniquac_from_database,
)

ArrayLike = Array | float


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
    the fitted activity models. Carries no fitted leaves -- it is a pure predictor
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


__all__ = [
    "UnifacModel",
    "eos_model_for",
    "nrtl_model_for",
    "unifac_model_for",
    "uniquac_model_for",
]
