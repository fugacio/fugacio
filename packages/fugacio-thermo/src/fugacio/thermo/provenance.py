"""Hashable parameter evidence and explicit applicability assessments.

Evidence is static JAX metadata; parameter arrays remain differentiable leaves.
Curated and predictive parameters aren't automatically measured or qualified.
Unknown validation bounds are reported as unknown, never as infinite coverage.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.resources import files
from itertools import combinations
from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import Array

from fugacio.thermo.components import get
from fugacio.thermo.data import nrtl_params, pr_kij, uniquac_params


@lru_cache(maxsize=8)
def _source_digest(filename: str) -> tuple[str, str]:
    raw = files("fugacio.thermo").joinpath(filename).read_bytes()
    return "fugacio.thermo/" + filename, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PairEvidence:
    """Origin of one interaction pair, in the package's component order."""

    components: tuple[str, str]
    kind: str
    source: str


@dataclass(frozen=True)
class PackageEvidence:
    """Immutable provenance, model assumptions, and observed parameter bounds.

    Bounds describe the data used for parameter estimation, not a proof of model
    validity. A qualification ID identifies a separate validation artifact.
    """

    method: str = "custom"
    components: tuple[str, ...] = ()
    pairs: tuple[PairEvidence, ...] = ()
    assumptions: tuple[str, ...] = ()
    sources: tuple[tuple[str, str], ...] = ()
    temperature_range: tuple[float, float] | None = None
    pressure_range: tuple[float, float] | None = None
    composition_range: tuple[float, float] | None = None
    qualification_id: str | None = None
    missing_components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for bounds in (self.temperature_range, self.pressure_range, self.composition_range):
            if bounds is not None and (
                not all(math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]
            ):
                raise ValueError("invalid applicability bounds")

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible provenance, with unknown bounds represented by null."""
        return asdict(self)


class ApplicabilityReport(NamedTuple):
    """Array-valued condition checks; unknown ranges remain separately visible."""

    parameters_available: Array
    temperature_known: Array
    pressure_known: Array
    composition_known: Array
    within_temperature: Array
    within_pressure: Array
    within_composition: Array
    assumptions_present: Array

    @property
    def accepted(self) -> Array:
        """Parameters exist and no declared bound is exceeded."""
        return (
            self.parameters_available
            & self.within_temperature
            & self.within_pressure
            & self.within_composition
        )

    def to_dict(self) -> dict[str, Any]:
        """Concrete condition report, including unknown bounds and assumptions."""
        return {"accepted": bool(self.accepted), **{k: bool(v) for k, v in self._asdict().items()}}


def assess_applicability(
    evidence: PackageEvidence, t: Array | float, p: Array | float, z: Array
) -> ApplicabilityReport:
    """Check declared ranges without inferring empirical validity from convergence."""

    def inside(value: Array | float, bounds: tuple[float, float] | None) -> Array:
        v = jnp.asarray(value)
        valid = jnp.isfinite(v)
        return valid if bounds is None else valid & (v >= bounds[0]) & (v <= bounds[1])

    return ApplicabilityReport(
        jnp.asarray(
            not evidence.missing_components
            and all(pair.kind != "missing" for pair in evidence.pairs)
        ),
        jnp.asarray(evidence.temperature_range is not None),
        jnp.asarray(evidence.pressure_range is not None),
        jnp.asarray(evidence.composition_range is not None),
        inside(t, evidence.temperature_range) & (jnp.asarray(t) > 0),
        inside(p, evidence.pressure_range) & (jnp.asarray(p) > 0),
        inside(z[0], evidence.composition_range),
        jnp.asarray(bool(evidence.assumptions)),
    )


def database_evidence(
    components: tuple[str, ...],
    method: str,
    *,
    allow_ideal: bool = False,
    use_database_kij: bool = False,
    explicit_kij: bool = False,
) -> PackageEvidence:
    """Describe actual factory choices, distinguishing missing pairs from zero assumptions."""
    names = tuple(get(c).name for c in components)
    pairs = []
    assumptions = []
    missing_components: tuple[str, ...] = ()
    if method in ("unifac", "dortmund"):
        from fugacio.thermo.groupcontrib._dortmund_data import (
            DO_COMPONENT_GROUPS,
            DO_INTERACTIONS,
            DO_SUBGROUPS,
        )
        from fugacio.thermo.groupcontrib._unifac_data import (
            COMPONENT_GROUPS,
            INTERACTIONS,
            SUBGROUPS,
        )

        assignments = DO_COMPONENT_GROUPS if method == "dortmund" else COMPONENT_GROUPS
        subgroups = DO_SUBGROUPS if method == "dortmund" else SUBGROUPS
        interactions = DO_INTERACTIONS if method == "dortmund" else INTERACTIONS
        missing_components = tuple(c for c in names if c not in assignments)
    for a, b in combinations(names, 2):
        if method in ("nrtl", "uniquac"):
            lookup = nrtl_params if method == "nrtl" else uniquac_params
            found = lookup(a, b) is not None
            kind = "curated" if found else "assumed_zero" if allow_ideal else "missing"
            source = (
                "ChemSep interaction table" if found else "Explicit zero interaction assumption"
            )
            if not found and not allow_ideal:
                source = "No interaction parameters in the curated table"
        elif method in ("unifac", "dortmund"):
            kind, source = "predictive", "Vendored UNIFAC group interaction table"
            groups = {subgroups[sg][1] for name in (a, b) for sg in assignments.get(name, {})}
            missing = [
                (i, j) for i in groups for j in groups if i != j and (i, j) not in interactions
            ]
            if a in missing_components or b in missing_components or missing:
                kind, source = (
                    "missing",
                    f"Missing group assignments or main-group interactions: {missing}",
                )
        elif method in ("pr", "srk", "rk", "vdw"):
            found = use_database_kij and pr_kij(a, b) is not None
            kind = "user_supplied" if explicit_kij else "curated" if found else "assumed_zero"
            source = "Explicit kij" if explicit_kij else "ChemSep PR kij" if found else "kij = 0"
        elif method == "pcsaft":
            # Individual PC-SAFT pair attribution is supplied by the caller below.
            from fugacio.thermo.saft._data import saft_kij

            found = use_database_kij and saft_kij(a, b) is not None
            kind = "user_supplied" if explicit_kij else "curated" if found else "assumed_zero"
            source = (
                "Explicit kij" if explicit_kij else "PC-SAFT binary bank" if found else "kij = 0"
            )
        else:
            kind, source = "user_supplied", "Custom model; validation evidence unspecified"
        pairs.append(PairEvidence((a, b), kind, source))
    if any(p.kind == "assumed_zero" for p in pairs):
        assumptions.append("Some binary interactions are explicitly set to zero.")
    if method in ("nrtl", "uniquac") and allow_ideal:
        assumptions.append(
            "Missing activity-model interactions may be zero; "
            "UNIQUAC retains its combinatorial term."
        )
    if method in ("pr", "srk", "rk", "vdw"):
        assumptions.append("Cubic one-fluid mixing rule; kij = 0 where no correction was selected.")
    table = {
        "nrtl": "_binary_params.py",
        "uniquac": "_binary_params.py",
        "unifac": "groupcontrib/_unifac_data.py",
        "dortmund": "groupcontrib/_dortmund_data.py",
        "pcsaft": "saft/_data.py",
    }.get(method)
    if method in ("pr", "srk", "rk", "vdw") and use_database_kij and not explicit_kij:
        table = "_eos_params.py"
    sources = (_source_digest(table),) if table else ()
    return PackageEvidence(
        method,
        names,
        tuple(pairs),
        tuple(assumptions),
        sources=sources,
        missing_components=missing_components,
    )
