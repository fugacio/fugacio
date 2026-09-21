"""Physical acceptance shared by checked flashes, process audits, and tools.

A small numerical residual alone doesn't establish a physical solution. These
checks separately grade input validity, component closure, present-phase
normalization, equifugacity, energy/entropy closure, finite-start tangent-plane
stability, and parameter applicability. Stability search success isn't a proof
of a global minimum; the search scope is always reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from fugacio.thermo.diagnostics import SolveReport, residual_report
from fugacio.thermo.energy import EnergyFlashResult
from fugacio.thermo.equilibrium import FlashResult
from fugacio.thermo.implicit import gate_tree
from fugacio.thermo.package import PropertyPackage
from fugacio.thermo.provenance import ApplicabilityReport, PackageEvidence, assess_applicability
from fugacio.thermo.stability import enrichment_starts, tpd_search


@dataclass(frozen=True)
class AcceptancePolicy:
    """Explicit physical tolerances and optional extrapolation/stability choices."""

    material_tolerance: float = 1e-8
    equilibrium_tolerance: float = 1e-7
    energy_relative_tolerance: float = 1e-7
    stability_tolerance: float = 1e-7
    stability_iterations: int = 160
    check_stability: bool = True
    allow_extrapolation: bool = False

    def __post_init__(self) -> None:
        tolerances = (
            self.material_tolerance,
            self.equilibrium_tolerance,
            self.energy_relative_tolerance,
            self.stability_tolerance,
        )
        if any(not math.isfinite(v) or v <= 0 for v in tolerances) or self.stability_iterations < 1:
            raise ValueError("acceptance tolerances and iteration count must be positive")


DEFAULT_POLICY = AcceptancePolicy()


class PhysicalReport(NamedTuple):
    """JAX-compatible independent acceptance criteria and their observed errors."""

    numerical: SolveReport
    input_valid: Array
    material_error: Array
    phase_error: Array
    equilibrium_error: Array
    energy_error: Array
    energy_checked: Array
    minimum_tpd: Array
    stability_converged: Array
    stability_checked: Array
    applicability: ApplicabilityReport
    accepted: Array

    def to_dict(self) -> dict[str, Any]:
        """Strict JSON with unchecked criteria represented explicitly."""

        def finite(v: Array) -> float | None:
            value = float(v)
            return value if math.isfinite(value) else None

        return {
            "accepted": bool(self.accepted),
            "numerical": self.numerical.to_dict(),
            "input_valid": bool(self.input_valid),
            "material_error": finite(self.material_error),
            "phase_error": finite(self.phase_error),
            "equilibrium_error": finite(self.equilibrium_error),
            "energy_error": finite(self.energy_error) if bool(self.energy_checked) else None,
            "energy_checked": bool(self.energy_checked),
            "stability_checked": bool(self.stability_checked),
            "minimum_tpd": finite(self.minimum_tpd) if bool(self.stability_checked) else None,
            "stability_converged": bool(self.stability_converged)
            if bool(self.stability_checked)
            else None,
            "stability_scope": (
                "Finite-start liquid/vapor tangent-plane search; no global-minimum proof."
            ),
            "applicability": self.applicability.to_dict(),
        }

    def failures(self, policy: AcceptancePolicy = DEFAULT_POLICY) -> list[str]:
        """Plain-language descriptions of the criteria a concrete state failed."""
        issues = []
        if not bool(self.input_valid):
            issues.append("the state specification is invalid or outside the model's domain")
        if not bool(self.numerical.converged):
            issues.append(f"the solve didn't converge ({self.numerical.to_dict()['status']})")
        if float(self.material_error) > policy.material_tolerance:
            issues.append(f"material balance error {float(self.material_error):.3g}")
        if float(self.phase_error) > policy.material_tolerance:
            issues.append(f"phase normalization error {float(self.phase_error):.3g}")
        if float(self.equilibrium_error) > policy.equilibrium_tolerance:
            issues.append(f"fugacity mismatch {float(self.equilibrium_error):.3g}")
        if (
            bool(self.energy_checked)
            and float(self.energy_error) > policy.energy_relative_tolerance
        ):
            issues.append(f"energy specification error {float(self.energy_error):.3g}")
        if bool(self.stability_checked):
            if float(self.minimum_tpd) < -policy.stability_tolerance:
                issues.append(
                    f"phase stability: a trial phase lowers the Gibbs energy (minimum "
                    f"tangent-plane distance {float(self.minimum_tpd):.3g}), so the state "
                    "splits further (for example, into two liquids)"
                )
            elif not bool(self.stability_converged):
                issues.append("phase stability couldn't be established (trials not stationary)")
        if not bool(self.applicability.accepted):
            issues.append("outside the parameters' evidence (extrapolation)")
        return issues


class CheckedFlash(NamedTuple):
    """PT flash value plus physical acceptance; failed values have invalid derivatives."""

    value: FlashResult
    report: PhysicalReport


class CheckedEnergyFlash(NamedTuple):
    """PH/PS state plus independent physical and specified-property acceptance."""

    value: EnergyFlashResult
    report: PhysicalReport


class PhysicalAcceptanceError(RuntimeError):
    """A concrete state failed a physical criterion, with its report attached."""

    def __init__(self, report: PhysicalReport, context: str = "thermodynamic state") -> None:
        self.report, self.context = report, context
        try:
            detail = "; ".join(report.failures()) or "inspect the attached report"
        except (TypeError, jax.errors.ConcretizationTypeError):
            detail = "inspect the attached report"
        super().__init__(f"{context} failed physical acceptance: {detail}")


class PhysicalAcceptanceWarning(UserWarning):
    """A converged result that an independent physical check found suspect.

    Issued by eager calls that don't raise for physical criteria (for example,
    `fugacio.sim.flash_drum` on a feed that splits into two liquids). Use the
    ``*_checked`` variants to turn the finding into an error.
    """


def require_accepted(report: PhysicalReport, context: str = "thermodynamic state") -> None:
    """Raise for rejected host states; compiled callers inspect ``report.accepted``."""
    if not isinstance(report.accepted, jax.core.Tracer) and not bool(report.accepted):
        raise PhysicalAcceptanceError(report, context)


def _normalize(x: Array) -> Array:
    return x / jnp.maximum(jnp.sum(x), 1e-300)


def equilibrium_residual(
    pkg: PropertyPackage, t: Array | float, p: Array | float, result: FlashResult, z: Array
) -> Array:
    """Present-phase log-fugacity residual, ignoring only exactly absent components."""

    def both(_: None) -> Array:
        x, y = _normalize(result.x), _normalize(result.y)
        delta = (
            jnp.log(jnp.maximum(x, 1e-300))
            + pkg.ln_phi(t, p, x, phase="liquid")
            - jnp.log(jnp.maximum(y, 1e-300))
            - pkg.ln_phi(t, p, y, phase="vapor")
        )
        return jnp.where(z > 0, delta, 0.0)

    return lax.cond((result.beta > 0) & (result.beta < 1), both, lambda _: jnp.zeros_like(z), None)


def phase_stability(
    pkg: PropertyPackage,
    t: Array | float,
    p: Array | float,
    z: Array,
    result: FlashResult,
    *,
    iterations: int = 160,
    tolerance: float = 1e-7,
) -> tuple[Array, Array]:
    """Search liquid and vapor trial phases against the returned state's tangent plane.

    Starts include the feed, the returned phases, and every component
    enrichment (`fugacio.thermo.stability.tpd_search`). Exactly absent
    components stay absent. Acceptance requires stationary trials and a
    nonnegative observed TPD; a negative TPD rejects even if a trial stalled.

    Returns:
        ``(minimum_tpd, all_trials_stationary)``.
    """
    z, result = lax.stop_gradient((jnp.asarray(z), result))
    support = z > 0
    x, y = _normalize(jnp.maximum(result.x, 0)), _normalize(jnp.maximum(result.y, 0))
    liquid = pkg.ln_phi_function(t, p, phase="liquid")
    vapor = pkg.ln_phi_function(t, p, phase="vapor")
    d = lax.cond(
        result.beta < 1,
        lambda _: jnp.log(jnp.maximum(x, 1e-300)) + liquid(x),
        lambda _: jnp.log(jnp.maximum(y, 1e-300)) + vapor(y),
        None,
    )
    starts = jnp.concatenate((enrichment_starts(z), x[None, :], y[None, :]))
    search = tpd_search((liquid, vapor), d, support, starts, iterations=iterations, tol=tolerance)
    return search.tpd, search.converged


def assess_flash(
    pkg: PropertyPackage,
    t: Array | float,
    p: Array | float,
    z: Array,
    result: FlashResult,
    *,
    numerical: SolveReport | None = None,
    target: Array | float | None = None,
    prop: str = "enthalpy",
    policy: AcceptancePolicy = DEFAULT_POLICY,
) -> PhysicalReport:
    """Independently assess a returned PT/PH/PS state, including maliciously supplied states.

    When no iteration report is supplied, numerical status grades the state
    residual and reports zero iterations. It doesn't invent an iteration log.
    An absent phase needn't be normalized; every present phase must be.
    """
    if prop not in ("enthalpy", "entropy"):
        raise ValueError("target property must be enthalpy or entropy")
    z = jnp.asarray(z)
    beta, x, y, _ = result
    valid = (
        (jnp.asarray(t) > 0)
        & (jnp.asarray(p) > 0)
        & jnp.isfinite(t)
        & jnp.isfinite(p)
        & jnp.all(jnp.isfinite(z))
        & jnp.all(z >= 0)
        & (jnp.abs(jnp.sum(z) - 1) <= policy.material_tolerance)
    )
    material = jnp.max(jnp.abs((1 - beta) * x + beta * y - z))
    phase = jnp.max(
        jnp.array(
            [
                jnp.maximum(-beta, 0) + jnp.maximum(beta - 1, 0),
                jnp.where(beta < 1, jnp.abs(jnp.sum(x) - 1) + jnp.max(jnp.maximum(-x, 0)), 0),
                jnp.where(beta > 0, jnp.abs(jnp.sum(y) - 1) + jnp.max(jnp.maximum(-y, 0)), 0),
            ]
        )
    )
    eq = jnp.max(jnp.abs(equilibrium_residual(pkg, t, p, result, z)))
    energy = jnp.asarray(0.0)
    if target is not None:
        fn = getattr(pkg, prop)

        def liquid(_: None) -> Array:
            return fn(t, p, x, phase="liquid")

        def vapor(_: None) -> Array:
            return fn(t, p, y, phase="vapor")

        value = lax.switch(
            jnp.where(beta <= 0, 0, jnp.where(beta >= 1, 2, 1)).astype(int),
            [liquid, lambda _: (1 - beta) * liquid(None) + beta * vapor(None), vapor],
            None,
        )
        energy = jnp.abs(value - target) / jnp.maximum(
            jnp.abs(target), 1e4 if prop == "enthalpy" else 10.0
        )
    if numerical is None:
        numerical = residual_report(jnp.array([material, phase, eq]), policy.equilibrium_tolerance)
    minimum, stable = (
        phase_stability(
            pkg,
            t,
            p,
            z,
            result,
            iterations=policy.stability_iterations,
            tolerance=policy.stability_tolerance,
        )
        if policy.check_stability
        else (jnp.asarray(0.0), jnp.asarray(False))
    )
    applicability = assess_applicability(getattr(pkg, "evidence", PackageEvidence()), t, p, z)
    # For example, a saturation-based liquid reference can't represent a
    # supercritical solute; the package declares its own domain.
    valid = valid & pkg.in_domain(t, p, z)
    applicable = (
        applicability.parameters_available if policy.allow_extrapolation else applicability.accepted
    )
    accepted = (
        valid
        & numerical.converged
        & (material <= policy.material_tolerance)
        & (phase <= policy.material_tolerance)
        & (eq <= policy.equilibrium_tolerance)
        & (energy <= policy.energy_relative_tolerance)
        & applicable
    )
    if policy.check_stability:
        accepted = accepted & stable & (minimum >= -policy.stability_tolerance)
    return lax.stop_gradient(
        PhysicalReport(
            numerical,
            valid,
            material,
            phase,
            eq,
            energy,
            jnp.asarray(target is not None),
            minimum,
            stable,
            jnp.asarray(policy.check_stability),
            applicability,
            accepted,
        )
    )


def flash_pt_checked(
    pkg: PropertyPackage,
    t: Array | float,
    p: Array | float,
    z: Array,
    *,
    policy: AcceptancePolicy = DEFAULT_POLICY,
    **options: Any,
) -> CheckedFlash:
    """Solve PT, assess physics, and attach derivative validity to the returned values."""
    solved = pkg.flash_pt_with_info(t, p, z, **options)
    report = assess_flash(pkg, t, p, z, solved.value, numerical=solved.report, policy=policy)
    return CheckedFlash(gate_tree(solved.value, report.accepted), report)


def _energy_checked(
    pkg: PropertyPackage,
    p: Array | float,
    target: Array | float,
    z: Array,
    prop: str,
    policy: AcceptancePolicy,
    options: dict[str, Any],
) -> CheckedEnergyFlash:
    solve = pkg.flash_ph_with_info if prop == "enthalpy" else pkg.flash_ps_with_info
    result = solve(p, target, z, **options)
    state = result.value
    pt = FlashResult(state.beta, state.x, state.y, state.k)
    report = assess_flash(
        pkg, state.t, p, z, pt, numerical=result.report, target=target, prop=prop, policy=policy
    )
    return CheckedEnergyFlash(gate_tree(state, report.accepted), report)


def flash_ph_checked(
    pkg: PropertyPackage,
    p: Array | float,
    h: Array | float,
    z: Array,
    *,
    policy: AcceptancePolicy = DEFAULT_POLICY,
    **options: Any,
) -> CheckedEnergyFlash:
    """PH flash with energy, equilibrium, stability, and applicability acceptance."""
    return _energy_checked(pkg, p, h, z, "enthalpy", policy, options)


def flash_ps_checked(
    pkg: PropertyPackage,
    p: Array | float,
    s: Array | float,
    z: Array,
    *,
    policy: AcceptancePolicy = DEFAULT_POLICY,
    **options: Any,
) -> CheckedEnergyFlash:
    """PS flash with entropy, equilibrium, stability, and applicability acceptance."""
    return _energy_checked(pkg, p, s, z, "entropy", policy, options)
