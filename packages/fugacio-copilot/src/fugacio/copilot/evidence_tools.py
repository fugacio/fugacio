"""Deterministic copilot tools for measured evidence and checked thermodynamics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from fugacio.sim import package_for
from fugacio.thermo.acceptance import AcceptancePolicy, flash_pt_checked, require_accepted
from fugacio.thermo.experimental import corpus_manifest, grouped_split, load_corpus
from fugacio.thermo.measured_regression import fit_measured_nrtl, validate_fit
from fugacio.thermo.provenance import PackageEvidence, assess_applicability, database_evidence


def measured_corpus() -> dict[str, Any]:
    """List measured systems and exact source coverage, without claiming model qualification."""
    observations = load_corpus()
    pairs = sorted({o.components for o in observations})
    return {
        "evidence": "measured",
        "observations": len(observations),
        "sources": corpus_manifest()["sources"],
        "systems": [
            {
                "components": list(pair),
                "properties": sorted({o.kind for o in observations if o.components == pair}),
                "observations": sum(o.components == pair for o in observations),
            }
            for pair in pairs
        ],
        "note": "Data availability isn't model qualification. Cloud points aren't tie lines.",
    }


def inspect_thermodynamic_evidence(
    components: list[str],
    method: str = "nrtl",
    temperature: float = 298.15,
    pressure: float = 101325.0,
) -> dict[str, Any]:
    """Inspect missing/curated/predictive interactions without fabricating a missing model."""
    from fugacio.sim.models import METHODS

    if method not in METHODS:
        raise ValueError(f"choose one of {METHODS}")
    evidence = database_evidence(tuple(components), method, use_database_kij=method == "pcsaft")
    z = jnp.ones(len(components)) / len(components)
    return {
        "parameter_evidence": evidence.to_dict(),
        "applicability": assess_applicability(evidence, temperature, pressure, z).to_dict(),
        "physical_acceptance": "not_checked",
        "empirical_qualification": "not_checked",
    }


def checked_flash(
    components: list[str],
    z: list[float],
    temperature: float,
    pressure: float,
    method: str = "pr",
    parameter_policy: str = "strict",
) -> dict[str, Any]:
    """Return a flash only after numerical, physical, and applicability criteria pass."""
    pkg = package_for(components, method, parameter_policy=parameter_policy)
    result = flash_pt_checked(pkg, temperature, pressure, jnp.asarray(z), policy=AcceptancePolicy())
    require_accepted(result.report, "copilot flash")
    r = result.value
    return {
        "vapor_fraction": float(r.beta),
        "liquid_composition": r.x.tolist(),
        "vapor_composition": r.y.tolist(),
        "physical_acceptance": result.report.to_dict(),
        "parameter_evidence": getattr(pkg, "evidence", PackageEvidence()).to_dict(),
        "empirical_qualification": "not_checked",
    }


def fit_measured_binary(components: list[str], holdout_source: str) -> dict[str, Any]:
    """Fit measured binary NRTL and validate against an explicitly held-out publication."""
    from fugacio.thermo.components import get

    names = {get(c).name for c in components}
    observations = tuple(
        o for o in load_corpus() if set(o.components) == names and o.kind != "cloud_point"
    )
    train, test = grouped_split(observations, by="source", holdout=(holdout_source,))
    fit = fit_measured_nrtl(train)
    return {"fit": fit.to_dict(), "validation": validate_fit(fit, test)}


def evidence_tool_specs() -> list[Any]:
    """Schemas for inspecting sources, checking states, and fitting measured data."""
    from fugacio.copilot.tools import ToolSpec

    components = {"type": "array", "items": {"type": "string"}, "minItems": 1}
    specs = []
    definitions: list[tuple[str, str, dict[str, Any], list[str], Callable[..., dict[str, Any]]]] = [
        (
            "measured_corpus",
            "List the offline measured corpus and citations; availability isn't qualification.",
            {},
            [],
            measured_corpus,
        ),
        (
            "inspect_thermodynamic_evidence",
            "Inspect parameter provenance and missing pairs before selecting a method.",
            {
                "components": components,
                "method": {"type": "string"},
                "temperature": {"type": "number"},
                "pressure": {"type": "number"},
            },
            ["components"],
            inspect_thermodynamic_evidence,
        ),
        (
            "checked_flash",
            "PT flash with physical acceptance and visible parameter assumptions.",
            {
                "components": components,
                "z": {"type": "array", "items": {"type": "number"}},
                "temperature": {"type": "number"},
                "pressure": {"type": "number"},
                "method": {"type": "string"},
                "parameter_policy": {"type": "string", "enum": ["strict", "allow_ideal"]},
            },
            ["components", "z", "temperature", "pressure"],
            checked_flash,
        ),
        (
            "fit_measured_binary",
            "Fit multi-temperature NRTL with a publication held out for validation.",
            {"components": components, "holdout_source": {"type": "string"}},
            ["components", "holdout_source"],
            fit_measured_binary,
        ),
    ]
    for name, description, properties, required, run in definitions:
        specs.append(
            ToolSpec(
                name=name,
                description=description,
                parameters={"type": "object", "properties": properties, "required": required},
                run=run,
            )
        )
    return specs
