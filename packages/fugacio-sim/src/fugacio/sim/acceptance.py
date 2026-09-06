"""Process acceptance using the thermodynamic package's physical criteria.

Audit stream phase inventories and unit/plant boundaries after solving. This
keeps stability searches out of recycle iterations while ensuring accepted
process results carry equilibrium, material, energy, and applicability evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from fugacio.sim.properties import _composition, enthalpy_flow, resolve_package
from fugacio.sim.stream import Stream
from fugacio.thermo.acceptance import (
    DEFAULT_POLICY,
    AcceptancePolicy,
    PhysicalReport,
    accepted_value,
    assess_flash,
    flash_pt_checked,
    require_accepted,
)
from fugacio.thermo.diagnostics import SolveReport, require_converged, residual_report
from fugacio.thermo.equilibrium import FlashResult
from fugacio.thermo.package import PropertyPackage
from fugacio.thermo.provenance import PackageEvidence


def audit_stream(
    stream: Stream, model: PropertyPackage, *, policy: AcceptancePolicy = DEFAULT_POLICY
) -> PhysicalReport:
    """Verify a stream's preserved phase inventory, or resolve an unspecified PT state.

    Empty streams have no meaningful equilibrium composition and are rejected by
    this physical audit. Their structural Stream.report remains available.
    """
    pkg = resolve_package(stream.components, model)
    z = _composition(stream.n)

    def resolved(_: None) -> PhysicalReport:
        vapor_n = jnp.asarray(stream.vapor_n)
        liquid_n = stream.n - vapor_n
        beta = jnp.sum(vapor_n) / jnp.maximum(stream.total, 1e-300)
        x, y = _composition(liquid_n), _composition(vapor_n)
        state = FlashResult(beta, x, y, y / jnp.maximum(x, 1e-300))
        return assess_flash(pkg, stream.t, stream.p, z, state, policy=policy)

    def unspecified(_: None) -> PhysicalReport:
        return flash_pt_checked(pkg, stream.t, stream.p, z, policy=policy).report

    report = lax.cond(stream.phase_known, resolved, unspecified, None)
    valid = stream.report.converged & (stream.total > 0)
    return report._replace(input_valid=report.input_valid & valid, accepted=report.accepted & valid)


def audit_balance(
    inputs: tuple[Stream, ...],
    outputs: tuple[Stream, ...],
    model: PropertyPackage,
    *,
    heat: Array | float = 0.0,
    work: Array | float = 0.0,
    component_generation: Array | None = None,
    tolerance: float = 1e-7,
) -> SolveReport:
    """Check component and enthalpy flow closure at an explicit process boundary.

    Heat and shaft work are positive into the fluid (watts). Reactions must
    supply net component generation in mol/s and use a formation-consistent
    enthalpy package. All streams must share package component order.
    """
    if not inputs or not outputs:
        raise ValueError("balance requires input and output streams")
    for stream in (*inputs, *outputs):
        resolve_package(stream.components, model)
    inlet = sum((s.n for s in inputs), jnp.zeros_like(inputs[0].n))
    outlet = sum((s.n for s in outputs), jnp.zeros_like(inputs[0].n))
    generated = jnp.zeros_like(inlet) if component_generation is None else component_generation
    h_in = sum(enthalpy_flow(s, model=model) for s in inputs)
    h_out = sum(enthalpy_flow(s, model=model) for s in outputs)
    material_scale = jnp.maximum(jnp.sum(jnp.abs(inlet)), 1.0)
    energy_scale = jnp.maximum(jnp.abs(h_in) + jnp.abs(heat) + jnp.abs(work), 1e4)
    errors = jnp.concatenate(
        (
            (outlet - inlet - generated) / material_scale,
            jnp.atleast_1d((h_out - h_in - heat - work) / energy_scale),
        )
    )
    return residual_report(errors, tolerance)


class CheckedUnit(NamedTuple):
    """Unit result, endpoint thermodynamic evidence, and an independent balance."""

    value: Any
    states: tuple[PhysicalReport, ...]
    balance: SolveReport

    @property
    def accepted(self) -> Array:
        """Whether all endpoint states and the process boundary passed."""
        return self.balance.converged & jnp.all(jnp.array([r.accepted for r in self.states]))

    def check(self) -> None:
        """Raise at a host boundary before a failed result is used downstream."""
        require_converged(self.balance, "unit component/energy balance")
        for i, report in enumerate(self.states):
            require_accepted(report, f"unit endpoint {i}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize acceptance evidence independently of the unit's output type."""
        return {
            "accepted": bool(self.accepted),
            "balance": self.balance.to_dict(),
            "states": [r.to_dict() for r in self.states],
        }


def _checked_unit(
    value: Any,
    inputs: tuple[Stream, ...],
    outputs: tuple[Stream, ...],
    model: PropertyPackage,
    heat: Array | float,
    policy: AcceptancePolicy,
) -> CheckedUnit:
    states = tuple(audit_stream(s, model, policy=policy) for s in (*inputs, *outputs))
    balance = audit_balance(inputs, outputs, model, heat=heat)
    accepted = balance.converged & jnp.all(jnp.array([r.accepted for r in states]))
    guarded = jax.tree.map(lambda v: accepted_value(v, accepted), value)
    return CheckedUnit(guarded, states, balance)


def heater_checked(
    feed: Stream,
    *,
    model: PropertyPackage,
    policy: AcceptancePolicy = DEFAULT_POLICY,
    **options: Any,
) -> CheckedUnit:
    """Run a heater, then verify both states and its component/energy balance."""
    from fugacio.sim.units import heater

    options.setdefault("allow_extrapolation", policy.allow_extrapolation)

    result = heater(feed, model=model, **options)
    return _checked_unit(result, (feed,), (result.outlet,), model, result.duty, policy)


def valve_checked(
    feed: Stream,
    p_out: Array | float,
    *,
    model: PropertyPackage,
    policy: AcceptancePolicy = DEFAULT_POLICY,
    **options: Any,
) -> CheckedUnit:
    """Run an isenthalpic valve with independent endpoint and balance checks."""
    from fugacio.sim.units import valve

    options.setdefault("allow_extrapolation", policy.allow_extrapolation)

    result = valve(feed, p_out, model=model, **options)
    return _checked_unit(result, (feed,), (result,), model, 0.0, policy)


@dataclass(frozen=True)
class BalanceBoundary:
    """Named process boundary with explicit heat and shaft work into the fluid."""

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    heat: float = 0.0
    work: float = 0.0


def audit_flowsheet(
    result: Any,
    model: PropertyPackage,
    *,
    boundaries: dict[str, BalanceBoundary],
    policy: AcceptancePolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Audit a solved flowsheet's streams and explicitly declared process boundaries.

    Boundary heat/work must come from the solved unit specifications. A list of
    endpoints alone cannot establish energy conservation. Numerical failures are
    retained, and no missing boundary is advertised as checked.
    """
    if not boundaries:
        raise ValueError("flowsheet acceptance requires explicit balance boundaries")
    states = {name: audit_stream(s, model, policy=policy) for name, s in result.streams.items()}
    balances = {
        name: audit_balance(
            tuple(result.streams[k] for k in b.inputs),
            tuple(result.streams[k] for k in b.outputs),
            model,
            heat=b.heat,
            work=b.work,
        )
        for name, b in boundaries.items()
    }
    return {
        "accepted": bool(result.converged)
        and all(bool(r.accepted) for r in states.values())
        and all(bool(r.converged) for r in balances.values()),
        "numerical": {k: r.to_dict() for k, r in result.reports.items()},
        "streams": {k: r.to_dict() for k, r in states.items()},
        "boundaries": {k: r.to_dict() for k, r in balances.items()},
        "parameter_evidence": getattr(model, "evidence", PackageEvidence()).to_dict(),
    }
