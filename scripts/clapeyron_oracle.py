"""Generate matched PC-SAFT cases and strictly compare the pinned Julia results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

from fugacio.thermo.saft import pressure, saft_parameters_for


def cases() -> list[dict]:
    result = []
    for names, x in [
        (["propane"], [1.0]),
        (["n-butane"], [1.0]),
        (["n-heptane"], [1.0]),
        (["propane", "n-butane"], [0.4, 0.6]),
    ]:
        params = saft_parameters_for(names, use_database_kij=False)
        for rho in (50.0, 2000.0, 7000.0):
            result.append(
                {
                    "id": "/".join(names) + f"/{rho:g}",
                    "components": names,
                    "composition": x,
                    "temperature_k": 350.0,
                    "density_mol_m3": rho,
                    "parameters": {
                        "segment": params.m.tolist(),
                        "sigma": params.sigma.tolist(),
                        "epsilon": params.epsilon.tolist(),
                    },
                }
            )
    return result


def verify(reference: dict) -> dict:
    if reference["julia"] != "1.10.10" or reference["clapeyron"] != "0.6.25":
        raise ValueError("oracle version mismatch")
    expected = cases()
    results = reference["results"]
    if len(results) != len(expected) or {r["id"] for r in results} != {c["id"] for c in expected}:
        raise ValueError("missing or duplicate oracle cases")
    by_id = {r["id"]: r for r in results}
    errors = []
    for case in expected:
        params = saft_parameters_for(case["components"], use_database_kij=False)
        value = float(
            pressure(
                params,
                case["density_mol_m3"],
                case["temperature_k"],
                jnp.array(case["composition"]),
            )
        )
        oracle = float(by_id[case["id"]]["pressure_pa"])
        relative = abs(value - oracle) / max(abs(oracle), 1.0)
        if not relative <= 2e-5:
            raise AssertionError(
                f"{case['id']}: Fugacio={value}, Clapeyron={oracle}, error={relative}"
            )
        errors.append(relative)
    return {
        "accepted": True,
        "cases": len(expected),
        "maximum_relative_error": max(errors),
        "relative_tolerance": 2e-5,
        "scope": "Matched nonassociating PC-SAFT pressure kernels.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("generate", "verify"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.operation == "generate":
        args.path.write_text(json.dumps(cases(), indent=2, allow_nan=False) + "\n")
    else:
        print(json.dumps(verify(json.loads(args.path.read_text())), allow_nan=False))
