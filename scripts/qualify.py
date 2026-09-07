"""Rebuild measured qualification and the energy-balanced process demonstration offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp

from fugacio.sim import Flowsheet, Stream, package_for
from fugacio.sim.acceptance import BalanceBoundary, audit_flowsheet, heater_checked
from fugacio.thermo.experimental import corpus_manifest, load_corpus
from fugacio.thermo.qualification import measured_matrix, qualify_ethanol_water


def run(output: Path, *, matrix: bool = True) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    observations = load_corpus()
    fit, validation = qualify_ethanol_water()
    fit.save(output / "ethanol-water-fit.json")
    pkg = package_for(fit.components, "nrtl", measured_fit=fit)
    feed = Stream.from_fractions(fit.components, jnp.array([0.5, 0.5]), 10.0, 330.0, 20000.0)
    heated = heater_checked(feed, t_out=335.0, model=pkg)
    fs = Flowsheet()
    fs.feed("feed", feed)
    from fugacio.sim.units import heater

    fs.unit(
        "heater",
        lambda inlet, _: heater(inlet, t_out=335.0, model=pkg).outlet,
        inputs=("feed",),
        outputs=("product",),
    )
    result = fs.solve_with_info()
    process = audit_flowsheet(
        result,
        pkg,
        boundaries={
            "heater": BalanceBoundary(("feed",), ("product",), heat=float(heated.value.duty))
        },
    )
    process.update(
        inlet_temperature_k=330.0,
        outlet_temperature_k=float(heated.value.outlet.t),
        pressure_pa=20000.0,
        duty_w=float(heated.value.duty),
    )
    report = {
        "schema_version": 1,
        "required_cases_passed": validation["accepted"]
        and bool(heated.accepted)
        and process["accepted"],
        "corpus": {
            "conditions": len(observations),
            "systems": len({o.components for o in observations}),
            "sources": corpus_manifest()["sources"],
            "normalized_sha256": hashlib.sha256(
                json.dumps([o.to_dict() for o in observations], sort_keys=True).encode()
            ).hexdigest(),
        },
        "independent_holdout": validation,
        "process": process,
        "curated_matrix": measured_matrix() if matrix else None,
        "scope": "Measured fit, publication holdout, and process closure are separate evidence. "
        "Unqualified or unsupported matrix cases remain visible.",
    }
    baseline_path = (
        Path(__file__).resolve().parents[1] / "benchmarks" / "thermodynamic-qualification.json"
    )
    if matrix and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        current = {case["id"]: case for case in report["curated_matrix"]}
        regressions = [
            case_id
            for case_id in baseline["required_qualified"]
            if current.get(case_id, {}).get("status") != "qualified"
        ]
        report["baseline_regressions"] = regressions
        report["required_cases_passed"] = report["required_cases_passed"] and not regressions
    (output / "qualification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rows = [
        "# Thermodynamic qualification",
        "",
        f"Required cases passed: {report['required_cases_passed']}",
        "",
        "| System | Property | Curated NRTL status |",
        "| --- | --- | --- |",
    ]
    rows.extend(
        f"| {' + '.join(c['components'])} | {c['property']} | {c['status']} |"
        for c in report["curated_matrix"] or []
    )
    rows.extend(
        [
            "",
            "See qualification.json for errors, limits, citations, exclusions, and process checks.",
            "",
        ]
    )
    (output / "qualification.md").write_text("\n".join(rows))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/qualification"))
    parser.add_argument("--skip-matrix", action="store_true")
    args = parser.parse_args()
    result = run(args.output, matrix=not args.skip_matrix)
    print(
        json.dumps(
            {"required_cases_passed": result["required_cases_passed"], "output": str(args.output)},
            allow_nan=False,
        )
    )
    raise SystemExit(0 if result["required_cases_passed"] else 1)
