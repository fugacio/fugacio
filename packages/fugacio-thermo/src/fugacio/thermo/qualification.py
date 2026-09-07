"""Executable measured-data qualification matrix with visible coverage gaps.

The matrix evaluates the current curated NRTL bank against every selected
measured system. Missing parameters are unsupported, and failed tolerances are
unqualified. Neither is relabeled as a passing test. A separate independent
publication holdout qualifies the reproducible ethanol-water fit.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import jax
import jax.numpy as jnp

from fugacio.thermo.data import nrtl_from_database
from fugacio.thermo.experimental import Observation, grouped_split, load_corpus
from fugacio.thermo.lle import binary_binodal
from fugacio.thermo.measured_regression import (
    DEFAULT_LIMITS,
    DEFAULT_WEIGHTS,
    MeasuredFit,
    _arrays,
    _predictions,
    fit_measured_nrtl,
    validate_fit,
)


def qualify_ethanol_water() -> tuple[MeasuredFit, dict[str, Any]]:
    """Fit the 2011 publication and hold out all unambiguous 2012 measurements."""
    observations = tuple(o for o in load_corpus() if set(o.components) == {"ethanol", "water"})
    train, test = grouped_split(observations, by="source", holdout=("10.1016/j.fluid.2012.12.014",))
    fit = fit_measured_nrtl(train)
    report = validate_fit(fit, test)
    report["id"] = "ethanol-water-publication-holdout"
    report["training_source"] = "10.1016/j.fluid.2011.06.009"
    return fit, report


def measured_matrix() -> list[dict[str, Any]]:
    """Evaluate current curated NRTL predictions for all corpus systems and properties.

    VLE limits are 5% pressure RMSE and 0.05 vapor mole-fraction RMSE; enthalpy
    limit is 150 J/mol. Cloud-point compositions are compared with the nearest
    nontrivial binodal branch, with 0.05 mole-fraction RMSE. Cloud-point tests
    don't infer unreported conjugate compositions from the measured data.
    """
    groups: dict[tuple[tuple[str, str], str], list[Observation]] = defaultdict(list)
    for obs in load_corpus():
        groups[(obs.components, obs.kind)].append(obs)
    cases = []
    for (components, kind), observations in sorted(groups.items()):
        case: dict[str, Any] = {
            "id": "/".join((*components, kind)),
            "components": list(components),
            "property": kind,
            "evidence": "measured_comparison",
            "model": "curated_nrtl",
            "observations": len(observations),
            "sources": sorted({o.source for o in observations}),
            "temperature_range_k": [
                min(o.temperature for o in observations),
                max(o.temperature for o in observations),
            ],
        }
        try:
            model = nrtl_from_database(list(components), strict=True)
        except KeyError as exc:
            cases.append({**case, "status": "unsupported", "reason": str(exc)})
            continue
        try:
            if kind == "cloud_point":
                t = jnp.array([o.temperature for o in observations])
                measured = jnp.array([o.composition("Liquid mixture 1")[0] for o in observations])
                lo, hi = jax.jit(jax.vmap(lambda ti, m=model: binary_binodal(m, ti)))(t)
                error = jnp.minimum(jnp.abs(lo - measured), jnp.abs(hi - measured))
                metric = float(jnp.sqrt(jnp.mean(error**2)))
                nontrivial = bool(jnp.all(jnp.abs(lo - hi) > 1e-5))

                def liquid_checks(
                    ti: jax.Array, a: jax.Array, b: jax.Array, activity: Any = model
                ) -> tuple[jax.Array, jax.Array]:
                    xa, xb = jnp.array([a, 1 - a]), jnp.array([b, 1 - b])
                    d = jnp.log(xa) + activity.ln_gamma(xa, ti)
                    equilibrium = jnp.max(jnp.abs(d - jnp.log(xb) - activity.ln_gamma(xb, ti)))
                    grid = jnp.linspace(1e-7, 1 - 1e-7, 101)
                    trials = jnp.stack([grid, 1 - grid], axis=1)
                    tpd = jax.vmap(
                        lambda w: jnp.sum(w * (jnp.log(w) + activity.ln_gamma(w, ti) - d))
                    )(trials)
                    return equilibrium, jnp.min(tpd)

                isoactivity, tpd = jax.vmap(liquid_checks)(t, lo, hi)
                physical = bool(jnp.all(isoactivity <= 1e-7) & jnp.all(tpd >= -1e-7))
                nontrivial = nontrivial and physical
                metrics, limits = (
                    {"cloud_composition_rmse": metric},
                    {"cloud_composition_rmse": 0.05},
                )
                extra = {
                    "nontrivial_physical_binodal": nontrivial,
                    "isoactivity_error": float(jnp.max(isoactivity))
                    if bool(jnp.all(jnp.isfinite(isoactivity)))
                    else None,
                    "liquid_stability_scope": "101-point liquid simplex grid; no global proof.",
                }
            else:
                a = _arrays(tuple(observations), DEFAULT_WEIGHTS)
                p, y, h = _predictions(model, a)
                residuals = {
                    "pressure_relative_rmse": (p - a["p"]) / a["p"],
                    "vapor_fraction_rmse": y - a["y"],
                    "excess_enthalpy_rmse": h - a["h"],
                }
                metrics = {
                    k: float(jnp.sqrt(jnp.mean(v**2))) for k, v in residuals.items() if v.size
                }
                limits = {k: getattr(DEFAULT_LIMITS, k) for k in metrics}
                extra, nontrivial = {}, True
            finite = all(bool(jnp.isfinite(v)) for v in metrics.values())
            passed = finite and nontrivial and all(v <= limits[k] for k, v in metrics.items())
            case.update(
                status="qualified" if passed else "unqualified",
                metrics={k: v if bool(jnp.isfinite(v)) else None for k, v in metrics.items()},
                limits=limits,
                **extra,
            )
        except (ValueError, FloatingPointError) as exc:
            case.update(status="unqualified", reason=str(exc))
        cases.append(case)
    return cases
