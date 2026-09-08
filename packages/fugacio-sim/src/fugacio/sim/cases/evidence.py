"""Reconstruct property packages and independently evaluate declared evidence."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from fugacio.sim.cases.jsonio import digest
from fugacio.sim.models import package_for


def build_package(document: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Build the exact declared model; qualification never comes from an input flag.

    Inline fits can originate outside the bundled corpus. Their provenance is
    then explicitly unverified. Qualification against bundled observations
    requires every training ID and source hash to match that corpus.
    """
    declaration = document["property_package"]
    options = dict(declaration.get("options", {}))
    if "kij" in options:
        options["kij"] = jnp.asarray(options["kij"])
    qualification: dict[str, Any] = {
        "status": "not_evaluated",
        "accepted": None,
        "scope": "No empirical qualification is claimed for this process.",
    }
    if "measured_fit" in declaration:
        from fugacio.thermo.experimental import load_corpus
        from fugacio.thermo.measured_regression import MeasuredFit, validate_fit

        fit = MeasuredFit.from_dict(declaration["measured_fit"])
        options["measured_fit"] = fit
        corpus = {o.id: o for o in load_corpus()}
        training = [corpus[k] for k in fit.training_ids if k in corpus]
        verified = (
            len(training) == len(fit.training_ids)
            and all(o.components == fit.components for o in training)
            and set(fit.sources) == {(o.source, o.source_sha256) for o in training}
        )
        if verified:
            # Applicability bounds describe the actual training observations;
            # don't let an edited fit expand them while retaining trusted IDs.
            expected = (
                (min(o.temperature for o in training), max(o.temperature for o in training)),
                (min(o.pressure for o in training), max(o.pressure for o in training)),
                (
                    min(o.composition()[0] for o in training),
                    max(o.composition()[0] for o in training),
                ),
            )
            observed = (fit.temperature_range, fit.pressure_range, fit.composition_range)
            if observed != expected:
                raise ValueError("measured fit bounds don't match its training observations")
        qualification["training_provenance_verified"] = verified
        qualification["fit_id"] = digest(fit.to_dict())
        if "qualification" in declaration:
            if not verified:
                raise ValueError("qualification requires verified training IDs and source hashes")
            ids = declaration["qualification"]["holdout_ids"]
            unknown = set(ids) - corpus.keys()
            if unknown:
                raise ValueError(f"unknown holdout IDs: {sorted(unknown)}")
            result = validate_fit(fit, tuple(corpus[k] for k in ids))
            qualification.update(result)
            qualification["status"] = "qualified_properties" if result["accepted"] else "failed"
            qualification["qualification_id"] = digest(result)
    package = package_for(document["components"], declaration["method"], **options)
    return package, qualification
