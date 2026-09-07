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
from fugacio.thermo.package import GammaPhiPackage, PropertyPackage
from fugacio.thermo.provenance import ApplicabilityReport, PackageEvidence, assess_applicability


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
        super().__init__(f"{context} failed physical acceptance; inspect the attached report")


def require_accepted(report: PhysicalReport, context: str = "thermodynamic state") -> None:
    """Raise for rejected host states; compiled callers inspect ``report.accepted``."""
    if not isinstance(report.accepted, jax.core.Tracer) and not bool(report.accepted):
        raise PhysicalAcceptanceError(report, context)


@jax.custom_jvp
def accepted_value(value: Array, accepted: Array) -> Array:
    """Retain a failed primal for diagnosis while invalidating its derivative."""
    return value


@accepted_value.defjvp
def _accepted_value_jvp(primals: Any, tangents: Any) -> tuple[Array, Array]:
    value, accepted = primals
    tangent, _ = tangents
    return value, tangent * jnp.where(accepted, 1.0, jnp.nan)


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
    """Search both liquid and vapor trial phases against the returned common tangent.

    Starts include the feed, returned phases, and every component enrichment.
    Exactly absent components stay absent. Acceptance requires stationary trials
    and nonnegative observed TPD. A negative TPD rejects even if a trial stalled.
    """
    z, result = lax.stop_gradient((z, result))
    support = z > 0
    x, y = _normalize(jnp.maximum(result.x, 0)), _normalize(jnp.maximum(result.y, 0))
    # Cache composition-independent gamma-phi references outside the trial loops.
    if isinstance(pkg, GammaPhiPackage):
        reference = pkg.ln_phi(t, p, z, phase="liquid") - pkg.activity.ln_gamma(z, t)

        def liquid(w: Array) -> Array:
            return pkg.activity.ln_gamma(w, t) + reference
    else:

        def liquid(w: Array) -> Array:
            return pkg.ln_phi(t, p, w, phase="liquid")

    def vapor(w: Array) -> Array:
        return pkg.ln_phi(t, p, w, phase="vapor")

    d = lax.cond(
        result.beta < 1,
        lambda _: jnp.log(jnp.maximum(x, 1e-300)) + liquid(x),
        lambda _: jnp.log(jnp.maximum(y, 1e-300)) + vapor(y),
        None,
    )
    enrich = 0.95 * jnp.eye(z.size) + 0.05 * z
    starts = jnp.concatenate((z[None, :], x[None, :], y[None, :], enrich))
    starts = jax.vmap(lambda w: _normalize(jnp.where(support, w, 0)))(starts)

    def trial(ln_phi: Any, w0: Array) -> tuple[Array, Array]:
        def composition(logw: Array) -> Array:
            return jax.nn.softmax(jnp.where(support, logw, -jnp.inf))

        def tpd(w: Array) -> Array:
            term = jnp.log(jnp.maximum(w, 1e-300)) + ln_phi(w) - d
            return jnp.sum(jnp.where(support, w * term, 0))

        def body(_: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
            logw, minimum = state
            w = composition(logw)
            proposed = jnp.clip(d - ln_phi(w), -690, 690)
            lognew = 0.5 * logw + 0.5 * proposed
            return lognew, jnp.minimum(minimum, tpd(w))

        logw, minimum = lax.fori_loop(
            0, iterations, body, (jnp.log(jnp.maximum(w0, 1e-300)), tpd(w0))
        )
        w = composition(logw)
        final = composition(d - ln_phi(w))
        stationary = jnp.max(jnp.abs(w - final)) <= tolerance
        return jnp.minimum(minimum, tpd(w)), stationary & jnp.all(jnp.isfinite(w))

    l_tpd, l_ok = jax.vmap(lambda w: trial(liquid, w))(starts)
    v_tpd, v_ok = jax.vmap(lambda w: trial(vapor, w))(starts)
    minimum = jnp.minimum(jnp.min(l_tpd), jnp.min(v_tpd))
    return lax.stop_gradient((minimum, jnp.all(l_ok) & jnp.all(v_ok) & jnp.isfinite(minimum)))


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
    # Saturation-based liquid references cannot represent supercritical solutes.
    if isinstance(pkg, GammaPhiPackage):
        valid = valid & jnp.all(jnp.where(z > 0, jnp.asarray(t) < pkg.tc, True))
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
    from fugacio.thermo.package import flash_pt_with_info

    solved = flash_pt_with_info(pkg, t, p, z, **options)
    report = assess_flash(pkg, t, p, z, solved.value, numerical=solved.report, policy=policy)
    value = jax.tree.map(lambda v: accepted_value(v, report.accepted), solved.value)
    return CheckedFlash(value, report)


def _energy_checked(
    pkg: PropertyPackage,
    p: Array | float,
    target: Array | float,
    z: Array,
    prop: str,
    policy: AcceptancePolicy,
    options: dict[str, Any],
) -> CheckedEnergyFlash:
    from fugacio.thermo.package import flash_ph_with_info, flash_ps_with_info

    solve = flash_ph_with_info if prop == "enthalpy" else flash_ps_with_info
    result = solve(pkg, p, target, z, **options)
    state = result.value
    pt = FlashResult(state.beta, state.x, state.y, state.k)
    report = assess_flash(
        pkg, state.t, p, z, pt, numerical=result.report, target=target, prop=prop, policy=policy
    )
    value = jax.tree.map(lambda v: accepted_value(v, report.accepted), state)
    return CheckedEnergyFlash(value, report)


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
