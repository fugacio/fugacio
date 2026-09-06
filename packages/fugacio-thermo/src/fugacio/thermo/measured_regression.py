"""Multi-temperature, multi-property NRTL regression with held-out validation.

The fitted convention is ``tau_ij = a_ij + b_ij/T`` with fixed alpha. Saturation
pressures use the same Peng-Robinson reference as GammaPhiPackage. Ideal vapor,
no Poynting correction, and no saturation fugacity correction are explicit model
assumptions. Excess enthalpy is the Gibbs-Helmholtz derivative of that same NRTL
model. Optimization runs on the host; residuals and Jacobians use JAX.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from fugacio.thermo.activity.models import NRTL
from fugacio.thermo.components import component_arrays
from fugacio.thermo.eos import PR
from fugacio.thermo.experimental import Observation
from fugacio.thermo.package import excess_enthalpy
from fugacio.thermo.reference import saturation_pressures


@dataclass(frozen=True)
class FitWeights:
    """Declared residual scales when standard measurement uncertainty is unknown.

    These are modeling weights, not claims about experimental uncertainty.
    Temperature/composition uncertainty isn't silently reinterpreted as pressure
    uncertainty. Covariance is conditional on treating measured inputs as exact.
    """

    pressure_relative: float = 0.02
    vapor_fraction: float = 0.02
    excess_enthalpy: float = 100.0

    def __post_init__(self) -> None:
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("residual scales must be finite and positive")


DEFAULT_WEIGHTS = FitWeights()


@dataclass(frozen=True)
class FitDiagnostics:
    """Optimizer status and local parameter identifiability at the final iterate."""

    converged: bool
    reason: str
    iterations: int
    weighted_rmse: float
    gradient_norm: float
    jacobian_rank: int
    condition_number: float | None
    degrees_of_freedom: int
    covariance: tuple[tuple[float, ...], ...] | None
    standard_errors: tuple[float, ...] | None
    uncertainty_note: str


@dataclass(frozen=True)
class MeasuredFit:
    """Portable NRTL fit with exact training IDs, source hashes, and validity bounds.

    ``theta`` is ``(tau12_ref, tau21_ref, b12/Tref, b21/Tref)``. Local covariance
    and standard errors refer to this centered coordinate system. A successful
    fit is still unqualified until independent holdout validation passes.
    """

    components: tuple[str, str]
    alpha: float
    reference_temperature: float
    theta: tuple[float, ...]
    training_ids: tuple[str, ...]
    sources: tuple[tuple[str, str], ...]
    temperature_range: tuple[float, float]
    pressure_range: tuple[float, float]
    composition_range: tuple[float, float]
    diagnostics: FitDiagnostics
    weights: FitWeights

    def __post_init__(self) -> None:
        if len(self.components) != 2 or len(set(self.components)) != 2:
            raise ValueError("a measured fit requires two distinct components")
        if len(self.theta) != 4 or not np.all(np.isfinite(self.theta)):
            raise ValueError("invalid fit coefficients")
        if (
            not 0 < self.alpha < 1
            or not np.isfinite(self.reference_temperature)
            or self.reference_temperature <= 0
        ):
            raise ValueError("invalid NRTL fit constants")
        if not self.training_ids or len(set(self.training_ids)) != len(self.training_ids):
            raise ValueError("training IDs must be nonempty and unique")
        if not self.sources or any(
            len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            for _, digest in self.sources
        ):
            raise ValueError("fit sources require raw-data SHA-256 hashes")
        for bounds in (self.temperature_range, self.pressure_range, self.composition_range):
            if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
                raise ValueError("invalid observed fit bounds")
        if (
            self.temperature_range[0] <= 0
            or self.pressure_range[0] <= 0
            or not 0 <= self.composition_range[0] <= self.composition_range[1] <= 1
        ):
            raise ValueError("observed fit bounds are outside physical input domains")

    def model(self, components: tuple[str, str] | None = None) -> NRTL:
        """Build a differentiable model, preserving requested component order."""
        theta = jnp.asarray(self.theta)
        if components is not None and components != self.components:
            if components != self.components[::-1]:
                raise ValueError("fit component identities don't match package")
            theta = theta[jnp.array([1, 0, 3, 2])]
        return _model(theta, self.reference_temperature, self.alpha)

    def to_dict(self) -> dict[str, Any]:
        """Versioned, strict-JSON fit artifact."""
        return {"schema_version": 1, "evidence": "measured_fit", **asdict(self)}

    def save(self, path: str | Path) -> None:
        """Save coefficients and complete evidence without replacing the sample bank."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> MeasuredFit:
        """Read and validate a versioned fit artifact."""
        data = json.loads(Path(path).read_text())
        if data.pop("schema_version") != 1 or data.pop("evidence") != "measured_fit":
            raise ValueError("unsupported measured-fit artifact")
        for key in (
            "components",
            "theta",
            "training_ids",
            "temperature_range",
            "pressure_range",
            "composition_range",
        ):
            data[key] = tuple(data[key])
        data["sources"] = tuple(tuple(v) for v in data["sources"])
        diag = data["diagnostics"]
        if diag["covariance"] is not None:
            diag["covariance"] = tuple(tuple(row) for row in diag["covariance"])
        if diag["standard_errors"] is not None:
            diag["standard_errors"] = tuple(diag["standard_errors"])
        data["diagnostics"] = FitDiagnostics(**diag)
        data["weights"] = FitWeights(**data["weights"])
        fit = cls(**data)
        if len(fit.theta) != 4 or not np.all(np.isfinite(fit.theta)):
            raise ValueError("invalid fit coefficients")
        if not 0 < fit.alpha < 1 or fit.reference_temperature <= 0:
            raise ValueError("invalid NRTL fit constants")
        return fit


def _model(theta: jax.Array, tref: float, alpha: float) -> NRTL:
    a12, a21 = theta[:2] - theta[2:]
    b12, b21 = theta[2:] * tref
    return NRTL(
        a=jnp.array([[0.0, a12], [a21, 0.0]]),
        b=jnp.array([[0.0, b12], [b21, 0.0]]),
        alpha=jnp.array([[0.0, alpha], [alpha, 0.0]]),
        e=jnp.zeros((2, 2)),
    )


def _check_observations(observations: tuple[Observation, ...]) -> tuple[str, str]:
    if not observations or len({o.components for o in observations}) != 1:
        raise ValueError("fit needs a nonempty set for one ordered binary pair")
    if len({o.id for o in observations}) != len(observations):
        raise ValueError("duplicate observations would overweight the fit")
    if any(o.kind not in ("vle", "excess_enthalpy") for o in observations):
        raise ValueError(
            "only VLE and excess enthalpy are supported; cloud points aren't tie lines"
        )
    return observations[0].components


def _arrays(observations: tuple[Observation, ...], weights: FitWeights) -> dict[str, Any]:
    components = _check_observations(observations)
    arr = component_arrays(list(components))
    vle = [o for o in observations if o.kind == "vle"]
    he = [o for o in observations if o.kind == "excess_enthalpy"]
    t = jnp.array([o.temperature for o in vle])
    if vle and bool(jnp.any(t[:, None] >= arr["tc"])):
        raise ValueError("gamma-phi saturation reference requires subcritical components")
    psat = (
        jax.jit(
            jax.vmap(lambda ti: saturation_pressures(PR, ti, arr["tc"], arr["pc"], arr["omega"]))
        )(t)
        if vle
        else jnp.empty((0, 2))
    )
    if not np.all(np.isfinite(psat)) or np.any(np.asarray(psat) <= 0):
        raise ValueError("nonfinite or nonpositive saturation reference")

    def scale(o: Observation, q: str, fallback: float, phase: str | None = None) -> float:
        m = o.measurement(q, phase)
        u = m.uncertainty.standard_value if m.uncertainty else None
        return u if m.role == "property" and u is not None and u > 0 else fallback

    y_indices = [
        i
        for i, o in enumerate(vle)
        if any(m.phase == "Gas" and m.quantity == "mole_fraction" for m in o.measurements)
    ]
    return {
        "t": t,
        "x": jnp.array([o.composition() for o in vle]).reshape((-1, 2)),
        "psat": psat,
        "p": jnp.array([o.pressure for o in vle]),
        "p_scale": jnp.array(
            [scale(o, "pressure", o.pressure * weights.pressure_relative) for o in vle]
        ),
        "yi": jnp.array(y_indices, dtype=int),
        "y": jnp.array([vle[i].composition("Gas")[0] for i in y_indices]),
        "y_scale": jnp.array(
            [scale(vle[i], "mole_fraction", weights.vapor_fraction, "Gas") for i in y_indices]
        ),
        "ht": jnp.array([o.temperature for o in he]),
        "hx": jnp.array([o.composition() for o in he]).reshape((-1, 2)),
        "h": jnp.array([o.measurement("excess_enthalpy").value for o in he]),
        "h_scale": jnp.array([scale(o, "excess_enthalpy", weights.excess_enthalpy) for o in he]),
    }


def _predictions(model: NRTL, arrays: dict[str, Any]) -> tuple[jax.Array, jax.Array, jax.Array]:
    a = arrays
    gamma = jnp.exp(jax.vmap(lambda x, t: model.ln_gamma(x, t))(a["x"], a["t"]))
    partial = a["x"] * gamma * a["psat"]
    p = jnp.sum(partial, axis=1)
    y = partial[:, 0] / p
    h = jax.vmap(lambda x, t: excess_enthalpy(model, x, t))(a["hx"], a["ht"])
    return p, y[a["yi"]], h


def fit_measured_nrtl(
    observations: tuple[Observation, ...],
    *,
    alpha: float = 0.3,
    weights: FitWeights = DEFAULT_WEIGHTS,
    max_iter: int = 150,
    initial: tuple[float, float, float, float] = (0.5, 0.5, 1.0, 1.0),
) -> MeasuredFit:
    """Fit four NRTL coefficients across temperatures and/or VLE and enthalpy.

    Uses damped Gauss-Newton with exact autodiff Jacobians. Rank deficiency is a
    failed fit even if the objective is stationary; covariance is then unknown.
    The returned status must be checked before using coefficients in a package.
    """
    components = _check_observations(observations)
    if not 0 < alpha < 1 or max_iter < 1 or not np.all(np.isfinite(initial)):
        raise ValueError("invalid optimizer configuration")
    arrays = _arrays(observations, weights)
    tref = float(np.mean([o.temperature for o in observations]))

    def residual(theta: jax.Array) -> jax.Array:
        p, y, h = _predictions(_model(theta, tref, alpha), arrays)
        return jnp.concatenate(
            (
                (p - arrays["p"]) / arrays["p_scale"],
                (y - arrays["y"]) / arrays["y_scale"],
                (h - arrays["h"]) / arrays["h_scale"],
            )
        )

    r_fn, j_fn = jax.jit(residual), jax.jit(jax.jacfwd(residual))
    theta = np.asarray(initial, dtype=float)
    damping = 1e-2
    converged, reason = False, "maximum_iterations"
    for _iteration in range(1, max_iter + 1):
        r, jac = np.asarray(r_fn(theta)), np.asarray(j_fn(theta))
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(jac)):
            reason = "nonfinite_residual_or_jacobian"
            break
        g, h = jac.T @ r, jac.T @ jac
        if np.linalg.norm(g, ord=np.inf) <= 1e-7 * max(1.0, float(np.linalg.norm(r))):
            converged, reason = True, "stationary_objective"
            break
        step = np.linalg.solve(h + damping * np.diag(np.maximum(np.diag(h), 1e-10)), -g)
        proposed = theta + step
        r_new = np.asarray(r_fn(proposed))
        if np.all(np.isfinite(r_new)) and r_new @ r_new < r @ r:
            improvement = float(r @ r - r_new @ r_new)
            theta = proposed
            damping = max(damping / 3, 1e-12)
            if improvement <= 1e-10 * max(1.0, float(r @ r)):
                converged, reason = True, "objective_tolerance"
                break
        else:
            damping *= 5
            if damping > 1e16:
                reason = "stalled"
                break
    r, jac = np.asarray(r_fn(theta)), np.asarray(j_fn(theta))
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(jac)):
        raise ValueError("fit ended at a nonfinite state")
    singular = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-8)) if singular.size else 0
    condition = float(singular[0] / singular[-1]) if rank == 4 else None
    dof = len(r) - 4
    covariance = None
    standard_errors = None
    if rank == 4 and dof > 0:
        cov = np.linalg.pinv(jac.T @ jac) * float(r @ r) / dof
        covariance = tuple(tuple(float(v) for v in row) for row in cov)
        standard_errors = tuple(float(v) for v in np.sqrt(np.maximum(np.diag(cov), 0)))
    if rank < 4:
        converged, reason = False, "unidentifiable_parameters"
    diag = FitDiagnostics(
        converged,
        reason,
        _iteration,
        float(np.sqrt(np.mean(r * r))),
        float(np.linalg.norm(jac.T @ r, ord=np.inf)),
        rank,
        condition,
        dof,
        covariance,
        standard_errors,
        "Local linearized covariance, scaled by residual variance; conditional on "
        "exact inputs and declared residual weights. Not a model-error bound.",
    )

    def bounds(values: list[float]) -> tuple[float, float]:
        return min(values), max(values)

    return MeasuredFit(
        components,
        alpha,
        tref,
        tuple(float(v) for v in theta),
        tuple(sorted(o.id for o in observations)),
        tuple(sorted({(o.source, o.source_sha256) for o in observations})),
        bounds([o.temperature for o in observations]),
        bounds([o.pressure for o in observations]),
        bounds([o.composition()[0] for o in observations]),
        diag,
        weights,
    )


@dataclass(frozen=True)
class ValidationLimits:
    """Explicit holdout RMSE acceptance limits, fixed before evaluating predictions."""

    pressure_relative_rmse: float = 0.05
    vapor_fraction_rmse: float = 0.05
    excess_enthalpy_rmse: float = 150.0

    def __post_init__(self) -> None:
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("validation limits must be finite and positive")


DEFAULT_LIMITS = ValidationLimits()


def validate_fit(
    fit: MeasuredFit,
    holdout: tuple[Observation, ...],
    *,
    limits: ValidationLimits = DEFAULT_LIMITS,
) -> dict[str, Any]:
    """Evaluate unseen observations and report physical errors by property.

    Training/holdout ID overlap raises. Passing a limit qualifies only these
    properties, systems, and conditions, never an entire model family.
    """
    if _check_observations(holdout) != fit.components:
        raise ValueError("holdout component order differs from fit")
    if set(fit.training_ids) & {o.id for o in holdout}:
        raise ValueError("training/holdout leakage")
    a = _arrays(holdout, fit.weights)
    p, y, h = _predictions(fit.model(), a)
    errors = {
        "pressure_relative_rmse": (p - a["p"]) / a["p"],
        "vapor_fraction_rmse": y - a["y"],
        "excess_enthalpy_rmse": h - a["h"],
    }
    metrics = {
        key: float(jnp.sqrt(jnp.mean(value**2))) for key, value in errors.items() if value.size
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        raise ValueError("nonfinite holdout prediction")
    return {
        "accepted": fit.diagnostics.converged
        and all(v <= getattr(limits, k) for k, v in metrics.items()),
        "metrics": metrics,
        "limits": asdict(limits),
        "counts": {key: int(value.size) for key, value in errors.items()},
        "holdout_ids": [o.id for o in holdout],
        "sources": sorted({o.source for o in holdout}),
        "outside_training_domain_counts": {
            "temperature": sum(
                not fit.temperature_range[0] <= o.temperature <= fit.temperature_range[1]
                for o in holdout
            ),
            "pressure": sum(
                not fit.pressure_range[0] <= o.pressure <= fit.pressure_range[1] for o in holdout
            ),
            "liquid_composition": sum(
                not fit.composition_range[0] <= o.composition()[0] <= fit.composition_range[1]
                for o in holdout
            ),
        },
        "temperature_range_k": [
            min(o.temperature for o in holdout),
            max(o.temperature for o in holdout),
        ],
        "scope": (
            "Observed binary properties only; not a global stability or model-family guarantee."
        ),
    }
