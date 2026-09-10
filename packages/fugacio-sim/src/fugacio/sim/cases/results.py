"""Independent process acceptance and portable run artifacts."""

from __future__ import annotations

import math
import platform
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from fugacio.sim.acceptance import audit_stream
from fugacio.sim.cases.expressions import output_derived
from fugacio.sim.cases.jsonio import canonical_json, digest, loads, read_json, write_json
from fugacio.sim.cases.quantities import display_value
from fugacio.sim.properties import enthalpy_flow, vapor_fraction
from fugacio.thermo.acceptance import AcceptancePolicy


def json_value(value: Any) -> Any:
    """Detach arrays into strict JSON; unavailable nonfinite diagnostics become null."""
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sealed(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Create a content-addressed envelope with no timestamp-dependent identity."""
    body = json_value({"schema_version": 1, "kind": kind, **payload})
    return {**body, "artifact_id": digest(body)}


def verify_artifact(value: Any, kind: str | None = None) -> dict[str, Any]:
    """Verify a strict JSON envelope's content integrity, not its external authenticity."""
    data = loads(canonical_json(value))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or "artifact_id" not in data:
        raise ValueError("unsupported artifact envelope")
    identity = data.pop("artifact_id")
    if identity != digest(data) or (kind is not None and data.get("kind") != kind):
        raise ValueError("artifact content hash or kind mismatch")
    return {**data, "artifact_id": identity}


@dataclass(frozen=True)
class CaseRun:
    """Immutable audited result with its exact input case and numerical environment.

    The hash detects accidental alteration. It isn't a signature or a guarantee
    that an arbitrary external author actually ran the recorded calculation.
    """

    _json: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CaseRun:
        """Verify an artifact before making it available to reporting clients."""
        d = verify_artifact(value, "run")
        from fugacio.sim.cases.schema import ProcessCase

        case = ProcessCase.from_dict(d["case"])
        if case.case_id != d["case_id"]:
            raise ValueError("run case identity mismatch")
        required = {
            "accepted",
            "parameters",
            "requested_parameters_si",
            "solver",
            "policy",
            "environment",
            "metrics",
            "streams",
            "units",
            "checks",
            "qualification",
            "parameter_evidence",
        }
        if not required <= d.keys() or not isinstance(d["accepted"], bool):
            raise ValueError("incomplete run artifact")
        return cls(canonical_json(d))

    @classmethod
    def load(cls, path: str | Path) -> CaseRun:
        """Read a content-verified saved run."""
        return cls.from_dict(read_json(path))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON report."""
        return loads(self._json)

    @property
    def run_id(self) -> str:
        """Identity of the complete result, including numerical and evidence settings."""
        return self.to_dict()["artifact_id"]

    @property
    def accepted(self) -> bool:
        """Whether the recorded numerical, physical, metric, and cost checks passed."""
        return self.to_dict()["accepted"]

    def check(self) -> None:
        """Raise with the failed run attached, preserving its diagnostic evidence."""
        if not self.accepted:
            raise CaseAcceptanceError(self)

    def save(self, path: str | Path) -> None:
        """Atomically save the complete strict-JSON result."""
        write_json(path, self.to_dict())

    def markdown(self) -> str:
        """Render a deterministic engineering report with explicit qualification scope."""
        return render_report(self)


class CaseAcceptanceError(RuntimeError):
    """A failed process run; inspect ``run`` for all independently graded criteria."""

    def __init__(self, run: CaseRun) -> None:
        self.run = run
        failed = [
            k
            for k, v in run.to_dict()["checks"].items()
            if isinstance(v, dict) and v.get("accepted") is False
        ]
        super().__init__(f"case run {run.run_id[:12]} failed: {', '.join(failed)}")


@lru_cache(maxsize=16)
def _auditor(policy: AcceptancePolicy) -> Any:
    return jax.jit(lambda s, p: audit_stream(s, p, policy=policy))


_enthalpy = jax.jit(lambda s, p: enthalpy_flow(s, model=p))
_vapor_fraction = jax.jit(lambda s, p: vapor_fraction(s, model=p))


def build_run(
    runner: Any,
    evaluation: Any,
    requested: dict[str, Any],
    solver: dict[str, Any],
    policy: dict[str, Any],
) -> CaseRun:
    """Audit every unit boundary, the plant boundary, streams, and declared outputs."""
    from fugacio.sim import __version__ as sim_version
    from fugacio.thermo import __version__ as thermo_version

    numerical = {k: r.to_dict() for k, r in evaluation.reports.items()}
    numerical_ok = all(bool(r.converged) for r in evaluation.reports.values())
    streams, states, enthalpies = {}, {}, {}
    audit = _auditor(runner.policy)
    for name, s in evaluation.streams.items():
        structural = bool(s.report.converged)
        empty = structural and float(s.total) == 0
        if empty:
            state = {
                "accepted": True,
                "status": "empty",
                "equilibrium_checked": False,
                "scope": "Zero material inventory; no phase composition exists.",
            }
        elif structural:
            state = audit(s, runner.package).to_dict()
        else:
            state = {
                "accepted": False,
                "status": "invalid_stream",
                "structural": s.report.to_dict(),
            }
        states[name] = state
        h = 0.0 if empty else float(_enthalpy(s, runner.package)) if structural else math.nan
        enthalpies[name] = h
        streams[name] = {
            "component_flow_mol_s": s.n,
            "flow_mol_s": s.total,
            "temperature_k": s.t,
            "pressure_pa": s.p,
            "mole_fractions": s.z if not empty else None,
            "vapor_component_flow_mol_s": s.vapor_n if bool(s.phase_known) else None,
            "enthalpy_flow_w": h,
            "phase_inventory": "preserved" if bool(s.phase_known) else "resolved_by_PT_audit",
        }
        streams[name]["vapor_fraction"] = (
            float(_vapor_fraction(s, runner.package)) if structural and not empty else None
        )
    reactive = any(u.kind == "stoichiometric_reactor" for u in runner.units)
    formation = jnp.zeros(len(runner.case.components))
    atom_matrix = None
    if reactive:
        from fugacio.sim.cases.registry import element_matrix
        from fugacio.thermo.reactions import reaction_arrays

        formation, _, _ = reaction_arrays(list(runner.case.components))
        _, rows = element_matrix(runner.case.components)
        atom_matrix = jnp.asarray(rows)

    def boundary(
        inputs: tuple[str, ...], outputs: tuple[str, ...], heat: Any, work: Any, generation: Any
    ) -> dict[str, Any]:
        n_in = sum((evaluation.streams[k].n for k in inputs), jnp.zeros_like(formation))
        n_out = sum((evaluation.streams[k].n for k in outputs), jnp.zeros_like(formation))
        hin = sum(enthalpies[k] for k in inputs) + float(n_in @ formation)
        hout = sum(enthalpies[k] for k in outputs) + float(n_out @ formation)
        material = float(jnp.max(jnp.abs(n_out - n_in - generation))) / max(
            float(jnp.sum(jnp.abs(n_in))), 1
        )
        energy = abs(hout - hin - float(heat) - float(work)) / max(
            abs(hin) + abs(float(heat)) + abs(float(work)), 1e4
        )
        atoms = (
            float(jnp.max(jnp.abs(atom_matrix @ (n_out - n_in))))
            / max(float(jnp.sum(jnp.abs(n_in))), 1)
            if atom_matrix is not None
            else None
        )
        accepted = (
            math.isfinite(material + energy)
            and material <= runner.policy.material_tolerance
            and energy <= runner.policy.energy_relative_tolerance
            and (atoms is None or atoms <= runner.policy.material_tolerance)
        )
        return {
            "accepted": accepted,
            "material_relative_error": material,
            "energy_relative_error": energy,
            "element_relative_error": atoms,
            "component_generation_mol_s": generation,
            "heat_w": heat,
            "work_w": work,
            "energy_reference": "package sensible/residual plus standard formation"
            if reactive
            else "property package datum",
        }

    balances, units = {}, {}
    generation = jnp.zeros_like(formation)
    for definition in runner.units:
        u = evaluation.units[definition.name]
        generation = generation + u.generation
        balances[definition.name] = boundary(
            definition.inlets, definition.outlets, u.heat, u.work, u.generation
        )
        units[definition.name] = {
            "kind": definition.kind,
            "inlets": definition.inlets,
            "outlets": definition.outlets,
            "quantities_si": u.quantities,
            "profiles_si": u.profiles,
            "heat_w": u.heat,
            "work_w": u.work,
            "heating_w": u.heating,
            "cooling_w": u.cooling,
            "component_generation_mol_s": u.generation,
            "numerical": u.report.to_dict(),
        }
        if definition.kind == "column":
            stages = definition.structure["n_stages"]
            width = 2 * len(runner.case.components) + 1
            border = int(definition.structure["condenser"] is not None) + int(
                definition.structure["reboiler"] is not None
            )
            block = runner.options.column_solver == "block"
            units[definition.name]["linear_system"] = {
                "solver": runner.options.column_solver,
                "unknowns": stages * width + border,
                "stage_block_size": width,
                "border_size": border,
                "forward_directions": min(stages, 3) * width + border
                if block
                else stages * width + border,
                "border_reverse_directions": border if block else 0,
                "fallback_policy": "checked_dense" if block else "not_applicable",
            }
        if definition.kind in ("flash", "component_separator") or (
            definition.kind == "mixer" and "t" in definition.settings
        ):
            units[definition.name]["assumption"] = (
                "Required external heat is inferred from the specified outlet "
                "states; this is not a predicted equipment duty capacity."
            )
    balances["plant"] = boundary(
        tuple(f.name for f in runner.feeds),
        runner.products,
        evaluation.plant["heat"],
        evaluation.plant["work"],
        generation,
    )
    metrics = {
        k: {
            "value_si": v,
            "value": display_value(v, runner.document["metrics"][k]["unit"]),
            "unit": runner.document["metrics"][k]["unit"],
            "finite": bool(jnp.isfinite(v)),
            "source": "process_expression"
            if output_derived(runner.document["metrics"][k]["expression"], runner.document)
            else "declared_input",
        }
        for k, v in evaluation.metrics.items()
    }
    parameters = {
        k: {
            "value_si": v,
            "value": display_value(v, runner.parameters[k].unit),
            "unit": runner.parameters[k].unit,
            "manipulated": k in runner.manipulated,
        }
        for k, v in evaluation.parameters.items()
    }
    try:
        runner.validate_values(evaluation.parameters)
        operating: dict[str, Any] = {"accepted": True}
    except ValueError as exc:
        operating = {"accepted": False, "error": str(exc)}
    specs = {}
    for s in runner.document["specifications"]:
        from fugacio.sim.cases.quantities import TEMPERATURE, quantity, unit_for

        dim = unit_for(runner.document["metrics"][s["metric"]]["unit"]).dimension
        target = quantity(
            s["target"],
            dim,
            "target",
            difference=runner.document["metrics"][s["metric"]]["unit"].startswith("delta_"),
        )
        tolerance = quantity(s["tolerance"], dim, "tolerance", difference=dim == TEMPERATURE)
        error = abs(float(evaluation.metrics[s["metric"]]) - target)
        specs[s["name"]] = {
            "accepted": math.isfinite(error) and error <= tolerance,
            "error_si": error,
            "target_si": target,
            "tolerance_si": tolerance,
        }
    unit_limits: dict[str, Any] = {}
    for definition in runner.units:
        inlet_pressures = [float(evaluation.streams[k].p) for k in definition.inlets]
        outlet_pressures = [float(evaluation.streams[k].p) for k in definition.outlets]
        issues = []
        if (
            definition.kind in ("pump", "compressor")
            and outlet_pressures[0] < inlet_pressures[0] - 1e-4
        ):
            issues.append("Compression equipment requires outlet pressure at least inlet pressure.")
        if (
            definition.kind in ("valve", "turbine", "flash", "mixer", "column")
            and max(outlet_pressures) > min(inlet_pressures) + 1e-4
        ):
            issues.append(
                "This unit has no pressure-raising work specification; "
                "feed pressure is insufficient."
            )
        inlet = streams[definition.inlets[0]]
        if (
            definition.kind == "pump"
            and float(inlet["flow_mol_s"]) > 0
            and (inlet["vapor_fraction"] is None or inlet["vapor_fraction"] > 1e-7)
        ):
            issues.append("The liquid-pump model requires a liquid inlet.")
        unit_limits[definition.name] = {"accepted": not issues, "issues": issues}
    checks: dict[str, Any] = {
        "numerical": {"accepted": numerical_ok, "reports": numerical},
        "physical": {
            "accepted": all(s["accepted"] for s in states.values())
            and all(b["accepted"] for b in balances.values()),
            "streams": states,
            "boundaries": balances,
            "scope": (
                "Stream phase states and unit/plant component, element "
                "(reactive), and energy balances; finite-start stability search, "
                "no global proof."
            ),
        },
        "operating_bounds": operating,
        "unit_operating_limits": {
            "accepted": all(v["accepted"] for v in unit_limits.values()),
            "units": unit_limits,
        },
        "specifications": {
            "accepted": all(s["accepted"] for s in specs.values()),
            "results": specs,
        },
        "metrics": {"accepted": all(m["finite"] for m in metrics.values())},
        "economics": {
            "accepted": bool(evaluation.plant.get("cost_valid", True)),
            "evaluated": "economics" in runner.document,
            "scope": (
                "Screening correlations and explicitly supplied prices; no vendor "
                "quote or guaranteed market accuracy."
            ),
        },
    }
    payload = sealed(
        "run",
        {
            "case": runner.document,
            "case_id": runner.case.case_id,
            "accepted": all(c["accepted"] for c in checks.values()),
            "requested_parameters_si": requested,
            "parameters": parameters,
            "solver": solver,
            "policy": policy,
            "environment": {
                "fugacio_sim": sim_version,
                "fugacio_thermo": thermo_version,
                "jax": jax.__version__,
                "jax_backend": jax.default_backend(),
                "jax_x64": bool(jax.config.read("jax_enable_x64")),
                "python": platform.python_version(),
            },
            "streams": streams,
            "products": runner.products,
            "units": units,
            "metrics": metrics,
            "plant_si": evaluation.plant,
            "equipment": evaluation.equipment,
            "checks": checks,
            "parameter_evidence": runner.package.evidence.to_dict(),
            "qualification": runner.qualification,
        },
    )
    return CaseRun.from_dict(payload)


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(run: CaseRun) -> str:
    """Render only recorded values, keeping qualification and acceptance distinct."""
    d = run.to_dict()
    lines = [
        f"# {_cell(d['case']['name'])}",
        "",
        f"Run `{run.run_id}`",
        "",
        f"Process acceptance: **{'passed' if run.accepted else 'failed'}**. "
        f"Backend: `{d['solver']['backend']}`.",
        f"Column linear solver: `{d['solver'].get('column_solver', 'dense')}`. "
        f"EO Jacobian assembly: `{d['solver'].get('eo_jacobian', 'dense')}`.",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    lines += [
        f"| {_cell(k)} | {'passed' if v['accepted'] else 'failed'} |"
        for k, v in d["checks"].items()
    ]
    lines += [
        "",
        "## Metrics",
        "",
        "| Metric | Value | Unit | Source |",
        "| --- | ---: | --- | --- |",
    ]
    for k, m in d["metrics"].items():
        value = f"{m['value']:.8g}" if m["value"] is not None else "unavailable"
        lines.append(f"| {_cell(k)} | {value} | {_cell(m['unit'])} | {_cell(m['source'])} |")
    lines += [
        "",
        "## Streams",
        "",
        "| Stream | mol/s | K | Pa | Enthalpy, W |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for k, s in d["streams"].items():
        values = [
            f"{s[p]:.8g}" if s[p] is not None else "unavailable"
            for p in ("flow_mol_s", "temperature_k", "pressure_pa", "enthalpy_flow_w")
        ]
        lines.append(f"| {_cell(k)} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Transfers",
        "",
        "Heat and shaft work are positive into the fluid.",
        "",
        "| Unit | Heat, W | Work, W |",
        "| --- | ---: | ---: |",
    ]
    lines += [f"| {_cell(k)} | {u['heat_w']} | {u['work_w']} |" for k, u in d["units"].items()]
    lines += [
        "",
        "## Evidence and limits",
        "",
        f"Empirical qualification: `{d['qualification']['status']}`. {d['qualification']['scope']}",
        "",
        d["checks"]["physical"]["scope"],
        "",
        d["checks"]["economics"]["scope"],
        "",
        (
            "Unknown validation ranges remain unknown. Process acceptance "
            "doesn't qualify an entire plant or property-model family."
        ),
        "",
    ]
    lines += ["- " + _cell(a) for a in d["parameter_evidence"].get("assumptions", [])]
    lines += ["- " + _cell(u["assumption"]) for u in d["units"].values() if "assumption" in u]
    return "\n".join(lines) + "\n"


def compare_runs(baseline: CaseRun, candidate: CaseRun) -> dict[str, Any]:
    """Compare compatible case metrics in SI, recording both acceptance states."""
    from fugacio.sim.cases.quantities import unit_for

    a, b = baseline.to_dict(), candidate.to_dict()
    if a["case"]["components"] != b["case"]["components"]:
        raise ValueError("comparison requires the same ordered component basis")
    metrics = {}
    for name in sorted(a["metrics"].keys() & b["metrics"].keys()):
        x, y = a["metrics"][name], b["metrics"][name]
        if (
            a["case"]["metrics"][name]["expression"] != b["case"]["metrics"][name]["expression"]
            or unit_for(x["unit"]).dimension != unit_for(y["unit"]).dimension
        ):
            raise ValueError(f"metric {name!r} changed meaning or dimension")
        delta = y["value_si"] - x["value_si"] if x["finite"] and y["finite"] else None
        metrics[name] = {
            "baseline_si": x["value_si"],
            "candidate_si": y["value_si"],
            "delta_si": delta,
            "relative_change": delta / abs(x["value_si"])
            if delta is not None and x["value_si"] != 0
            else None,
        }
    if not metrics:
        raise ValueError("runs have no compatible named metrics")
    return sealed(
        "comparison",
        {
            "baseline_id": baseline.run_id,
            "candidate_id": candidate.run_id,
            "baseline_accepted": baseline.accepted,
            "candidate_accepted": candidate.accepted,
            "metrics": metrics,
            "parameters": {
                k: {
                    "baseline_si": a["parameters"].get(k, {}).get("value_si"),
                    "candidate_si": b["parameters"].get(k, {}).get("value_si"),
                }
                for k in sorted(a["parameters"].keys() | b["parameters"].keys())
            },
            "package_changed": a["case"]["property_package"] != b["case"]["property_package"],
            "solver_changed": a["solver"] != b["solver"],
            "policy_changed": a["policy"] != b["policy"],
        },
    )
