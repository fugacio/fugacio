"""Checked, property-package-consistent steady-state reaction calculations.

Reacting units use one declared homogeneous phase. Their energy includes
formation enthalpies exactly once. A phase change is a rejected homogeneous
reactor calculation; a reactive flash or reactive MESH column handles coupled
reaction and phase separation explicitly.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.properties import Model, molar_enthalpy, resolve_package
from fugacio.sim.stream import Stream
from fugacio.thermo.acceptance import accepted_value
from fugacio.thermo.diagnostics import SolveReport, SolveStatus, require_converged, residual_report
from fugacio.thermo.implicit import newton_system_with_info
from fugacio.thermo.package import flash_pt_with_info
from fugacio.thermo.reaction_system import ReactionSet, ReferenceRate


class ReactionResult(NamedTuple):
    """Reaction products, actual numerical reports, balances, and retained profiles.

    Extents and generation are in mol/s; duty is W, positive into the fluid.
    ``integration_error`` is a dimensionless step-doubling error ratio for
    PFRs (accepted at <= 1), zero for algebraic reactors. Profiles use the
    refined axial grid, including the inlet. Acceptance checks homogeneous
    phase compatibility in addition to balance closure, but a finite PT flash
    isn't an independent global phase-stability proof. Case runs add the
    existing stream stability audit.
    """

    outlet: Stream
    duty: Array
    extent: Array
    generation: Array
    report: SolveReport
    material_error: Array
    element_error: Array
    energy_error: Array
    phase_error: Array
    integration_error: Array
    coordinate: Array
    component_profile: Array
    temperature_profile: Array
    pressure_profile: Array
    rate_profile: Array

    @property
    def converged(self) -> Array:
        """Whether numerical, physical-domain, and integration checks passed."""
        return self.report.converged

    def check(self) -> None:
        """Raise with the retained report for a rejected concrete calculation."""
        require_converged(self.report, "reactor")


def reaction_parameter_validity(system: ReactionSet) -> Array:
    """Check finite reaction parameters and element conservation under JAX too."""
    valid = jnp.all(jnp.isfinite(system.nu)) & (
        jnp.max(jnp.abs(system.atoms @ system.nu.T)) <= 1e-9
    )
    valid &= jnp.isfinite(system.reference_concentration) & (system.reference_concentration > 0)
    for value in jax.tree_util.tree_leaves(system):
        valid &= jnp.all(jnp.isfinite(value))
    for i, law in enumerate(system.rate_laws):
        if isinstance(law, ReferenceRate):
            valid &= (law.k_forward >= 0) & (law.k_reverse >= 0) & (law.reference_temperature > 0)
            valid &= jnp.all(law.forward_orders >= 0) & jnp.all(law.reverse_orders >= 0)
            if law.detailed_balance:
                valid &= jnp.all(
                    jnp.abs(law.forward_orders - jnp.maximum(-system.nu[i], 0)) < 1e-12
                )
                valid &= jnp.all(jnp.abs(law.reverse_orders - jnp.maximum(system.nu[i], 0)) < 1e-12)
    return valid


def _enthalpy(n: Array, t: Array, p: Array, package: Any, system: ReactionSet) -> Array:
    total = jnp.sum(n)
    z = n / jnp.maximum(total, 1e-30)
    return total * system.enthalpy(package, t, p, z)


def _phase_error(n: Array, t: Array, p: Array, package: Any, system: ReactionSet) -> Array:
    z = n / jnp.maximum(jnp.sum(n), 1e-30)
    solved = flash_pt_with_info(package, t, p, z)
    return jnp.where(
        solved.report.converged,
        jnp.abs(solved.value.beta - (1.0 if system.phase == "vapor" else 0.0)),
        jnp.inf,
    )


def _complete(
    feed: Stream,
    package: Any,
    system: ReactionSet,
    n: Array,
    t: Array,
    p: Array,
    extent: Array,
    duty: Array,
    report: SolveReport,
    coordinate: Array,
    ns: Array,
    ts: Array,
    ps: Array,
    rates: Array,
    integration_error: Array,
    *,
    check: bool,
    phase_check: bool = True,
) -> ReactionResult:
    scale = jnp.maximum(feed.total, 1.0)
    generation = extent @ system.nu
    material = jnp.max(jnp.abs(n - feed.n - generation)) / scale
    elements = jnp.max(jnp.abs(system.atoms @ (n - feed.n))) / scale
    hin = feed.total * molar_enthalpy(feed, model=package) + feed.n @ system.formation_enthalpy
    energy = jnp.abs(_enthalpy(n, t, p, package, system) - hin - duty) / jnp.maximum(
        jnp.abs(hin) + jnp.abs(duty), scale * 1e4
    )
    # Detach audits before nested flashes so they never enlarge derivative graphs.
    detached = jax.lax.stop_gradient((ns, ts, ps, package, system))
    phase = (
        jnp.max(
            jax.vmap(lambda ni, ti, pi: _phase_error(ni, ti, pi, detached[3], detached[4]))(
                *detached[:3]
            )
        )
        if phase_check
        else jnp.asarray(0.0)
    )
    finite = jnp.all(jnp.isfinite(ns)) & jnp.all(jnp.isfinite(ts)) & jnp.all(jnp.isfinite(ps))
    domain = (
        finite
        & jnp.all(ns >= -1e-10)
        & jnp.all(jnp.sum(ns, axis=1) > 0)
        & jnp.all(ts > 0)
        & jnp.all(ps > 0)
    )
    domain &= feed.report.converged & reaction_parameter_validity(system)
    expected_vapor = feed.n if system.phase == "vapor" else jnp.zeros_like(feed.n)
    domain &= jnp.where(
        feed.phase_known, jnp.max(jnp.abs(feed.vapor_n - expected_vapor)) <= 1e-7 * scale, True
    )
    acceptance = jnp.array(
        [
            material / 1e-7,
            elements / 1e-7,
            energy / 1e-6,
            phase / 1e-7,
            integration_error,
            jnp.where(domain, 0.0, 2.0),
        ]
    )
    audit = residual_report(acceptance, tol=1.0, failure=SolveStatus.INFEASIBLE)
    ok = report.converged & audit.converged
    final = report._replace(
        status=jnp.where(report.converged, audit.status, report.status),
        residual_norm=jnp.maximum(
            report.residual_norm, jnp.max(jnp.array([material, elements, energy, phase]))
        ),
    )
    if check:
        require_converged(final, "reactor")
    n, t, p, extent, duty, coordinate, ns, ts, ps, rates = jax.tree_util.tree_map(
        lambda v: accepted_value(v, ok), (n, t, p, extent, duty, coordinate, ns, ts, ps, rates)
    )
    outlet = Stream(n, t, p, feed.components, n if system.phase == "vapor" else jnp.zeros_like(n))
    return ReactionResult(
        outlet,
        duty,
        extent,
        extent @ system.nu,
        final,
        material,
        elements,
        energy,
        phase,
        integration_error,
        coordinate,
        ns,
        ts,
        ps,
        rates,
    )


def reaction_reactor(
    feed: Stream,
    system: ReactionSet,
    *,
    kind: str = "equilibrium",
    model: Model = None,
    volume: float | Array = 1.0,
    t_out: float | Array | None = None,
    duty: float | Array | None = None,
    dp: float | Array = 0.0,
    steps: int = 64,
    integration_rtol: float = 1e-5,
    integration_atol: float = 1e-8,
    tol: float = 1e-10,
    max_iter: int = 100,
    check: bool = True,
) -> ReactionResult:
    """Solve a homogeneous equilibrium reactor, CSTR, or PFR with a common package.

    Args:
        feed: Inlet material and phase state.
        system: Validated reactions, kinetic laws, phase, and rate basis.
        kind: ``equilibrium``, ``cstr``, or ``pfr``.
        model: Property package; defaults to Peng-Robinson.
        volume: Reacting-phase volume (m^3), required positive for kinetic units.
        t_out: Isothermal temperature. Omission retains inlet T if duty is omitted.
        duty: Specified heat input (W); zero selects adiabatic operation.
            Mutually exclusive with t_out. PFR heat is distributed uniformly.
        dp: Nonnegative pressure loss (Pa); PFR pressure decreases linearly.
        steps: Coarse PFR mesh size; a 2*steps mesh supplies the returned solution.
        integration_rtol: Relative step-doubling and energy error scale.
        integration_atol: Absolute step-doubling scale in normalized coordinates.
        tol: Scaled nonlinear residual tolerance.
        max_iter: Newton iteration cap.
        check: Raise on a rejected concrete result; traced callers inspect report.

    Returns:
        A ReactionResult retaining failed primals and nonfinite failed derivatives.
        PFRs compare coarse/refined extents and temperatures, independently audit
        energy, and reject negative inventory without clipping the final solution.
    """
    if kind not in ("equilibrium", "cstr", "pfr"):
        raise ValueError("kind must be equilibrium, cstr, or pfr")
    if t_out is not None and duty is not None:
        raise ValueError("specify exactly one of t_out or duty")
    if system.components != feed.components:
        raise ValueError("reaction and feed component order differs")
    if kind != "equilibrium" and not system.rate_laws:
        raise ValueError("kinetic reactors require rate laws")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1 or steps > 4096:
        raise ValueError("steps must be an integer in [1, 4096]")
    if any(not math.isfinite(v) or v <= 0 for v in (integration_rtol, integration_atol)):
        raise ValueError("integration tolerances must be finite and positive")
    package = resolve_package(feed.components, model)
    system.check_package(package)
    p = jnp.asarray(feed.p - dp, dtype=float)
    vol = jnp.asarray(volume, dtype=float)
    thermal = duty is not None
    target_t = jnp.asarray(feed.t if t_out is None else t_out, dtype=float)
    heat = jnp.asarray(0.0 if duty is None else duty, dtype=float)
    scale = jnp.maximum(feed.total, 1.0)
    hin = feed.total * molar_enthalpy(feed, model=package) + feed.n @ system.formation_enthalpy
    # Every differentiated quantity, including kinetics and package coefficients,
    # travels through theta rather than a custom-derivative closure.
    theta: dict[str, Any] = {
        "package": package,
        "system": system,
        "feed": feed.n,
        "p": p,
        "p_in": feed.p,
        "volume": vol,
        "temperature": target_t,
        "heat": heat,
        "hin": hin,
        "scale": scale,
    }
    c, r = len(feed.components), system.nu.shape[0]

    if kind in ("equilibrium", "cstr"):

        def residual(u: Array, th: Any) -> Array:
            s, pkg = th["system"], th["package"]
            n = jnp.exp(u[:c]) * th["scale"]
            xi = u[c : c + r] * th["scale"]
            t = u[-1] * 100.0 if thermal else th["temperature"]
            z = n / jnp.sum(n)
            material = (n - th["feed"] - xi @ s.nu) / th["scale"]
            chemical = (
                s.nu @ s.log_activities(pkg, t, th["p"], z) - s.ln_equilibrium_constants(t)
                if kind == "equilibrium"
                else (xi - th["volume"] * s.rates(pkg, t, th["p"], z)) / th["scale"]
            )
            terms = [material, chemical]
            if thermal:
                terms.append(
                    jnp.atleast_1d(
                        (_enthalpy(n, t, th["p"], pkg, s) - th["hin"] - th["heat"])
                        / (th["scale"] * 1e4)
                    )
                )
            return jnp.concatenate(terms)

        seed = jnp.concatenate(
            [
                jnp.log(jnp.maximum(feed.n / scale, 1e-3)),
                jnp.zeros(r),
                jnp.atleast_1d(target_t / 100) if thermal else jnp.zeros(0),
            ]
        )
        lower = jnp.full_like(seed, -jnp.inf).at[:c].set(-70.0)
        upper = jnp.full_like(seed, jnp.inf).at[:c].set(20.0)
        if thermal:
            lower = lower.at[-1].set(0.5)
            upper = upper.at[-1].set(20.0)
        solved = newton_system_with_info(
            residual, jax.lax.stop_gradient(seed), theta, tol, max_iter, lower=lower, upper=upper
        )
        u = solved.value
        extent = u[c : c + r] * scale
        # Return the stoichiometric inventory exactly, rather than an independently
        # solved log-flow copy whose small residual can amplify in atom balances.
        n = feed.n + extent @ system.nu
        returned_u = u.at[:c].set(jnp.log(jnp.maximum(n / scale, 1e-300)))
        returned_report = residual_report(residual(returned_u, theta), tol)
        t = u[-1] * 100 if thermal else target_t
        heat = heat if thermal else _enthalpy(n, t, p, package, system) - hin
        rates = system.rates(package, t, p, n / jnp.sum(n)) if system.rate_laws else jnp.zeros(0)
        coordinate = jnp.array([0.0, vol if kind == "cstr" else 1.0])
        ns, ts, ps = jnp.stack([feed.n, n]), jnp.stack([feed.t, t]), jnp.stack([feed.p, p])
        report = solved.report._replace(
            status=jnp.where(solved.report.converged, returned_report.status, solved.report.status),
            residual_norm=jnp.maximum(solved.report.residual_norm, returned_report.residual_norm),
        )
        integration_error = jnp.asarray(0.0)
        inlet_rates = (
            system.rates(package, feed.t, feed.p, feed.z) if system.rate_laws else jnp.zeros(0)
        )
        rates = jnp.stack([inlet_rates, rates])
    else:

        def rhs(q: Array, state: Array, th: Any) -> Array:
            s, pkg = th["system"], th["package"]
            xi, t = state[:-1] * th["scale"], state[-1] * 100
            n = th["feed"] + xi @ s.nu
            # Trial states may leave the domain. Keep constitutive evaluations
            # finite while preserving and rejecting the actual negative inventory.
            safe = jnp.maximum(n, 1e-30)
            pressure = th["p_in"] + q * (th["p"] - th["p_in"])
            rates = s.rates(pkg, t, pressure, safe / jnp.sum(safe))
            dn = th["volume"] * (rates @ s.nu)
            if thermal:
                hn, ht, hp = jax.grad(_enthalpy, argnums=(0, 1, 2))(safe, t, pressure, pkg, s)
                dt = (th["heat"] - hn @ dn - hp * (th["p"] - th["p_in"])) / ht
            else:
                dt = jnp.asarray(0.0)
            return jnp.concatenate([th["volume"] * rates / th["scale"], jnp.atleast_1d(dt / 100)])

        def march(count: int) -> tuple[Array, Array]:
            state0 = jnp.concatenate([jnp.zeros(r), jnp.atleast_1d(target_t / 100)])
            dq = 1.0 / count

            def step(state: Array, q: Array) -> tuple[Array, tuple[Array, Array]]:
                a = rhs(q, state, theta)
                b = rhs(q + dq / 2, state + dq * a / 2, theta)
                d = rhs(q + dq / 2, state + dq * b / 2, theta)
                e = rhs(q + dq, state + dq * d, theta)
                new = state + dq * (a + 2 * b + 2 * d + e) / 6
                trials = jnp.stack(
                    [state, state + dq * a / 2, state + dq * b / 2, state + dq * d, new]
                )
                inventory = (
                    theta["feed"][None, :] + (trials[:, :-1] * theta["scale"]) @ theta["system"].nu
                )
                valid = (
                    jnp.all(jnp.isfinite(trials))
                    & jnp.all(trials[:, -1] > 0)
                    & jnp.all(inventory >= 0)
                )
                return new, (new, valid)

            _, (ys, valid) = jax.lax.scan(step, state0, jnp.arange(count) * dq)
            return jnp.concatenate([state0[None, :], ys]), jnp.all(valid)

        (coarse, coarse_valid), (refined, refined_valid) = march(steps), march(2 * steps)
        errors = jnp.abs(refined[::2] - coarse) / (
            integration_atol + integration_rtol * jnp.maximum(jnp.abs(refined[::2]), 1.0)
        )
        integration_error = jnp.where(coarse_valid & refined_valid, jnp.max(errors) / 15.0, jnp.inf)
        coordinate = jnp.linspace(0.0, 1.0, 2 * steps + 1)
        ns = feed.n[None, :] + (refined[:, :-1] * scale) @ system.nu
        ts = refined[:, -1] * 100
        ps = feed.p + coordinate * (p - feed.p)
        extent = refined[-1, :-1] * scale
        n, t = ns[-1], ts[-1]
        heat = heat if thermal else _enthalpy(n, t, p, package, system) - hin
        rates = jax.vmap(lambda ni, ti, pi: system.rates(package, ti, pi, ni / jnp.sum(ni)))(
            ns, ts, ps
        )
        report = residual_report(
            jnp.atleast_1d(jnp.where(integration_error <= 1.0, 0.0, integration_error)),
            iterations=2 * steps,
            failure=SolveStatus.MAX_ITERATIONS,
        )
        coordinate = coordinate * vol
    valid = jnp.isfinite(vol) & (vol > 0) & jnp.isfinite(jnp.asarray(dp)) & (jnp.asarray(dp) >= 0)
    report = report._replace(status=jnp.where(valid, report.status, SolveStatus.INVALID_INPUT))
    return _complete(
        feed,
        package,
        system,
        n,
        t,
        p,
        extent,
        heat,
        report,
        coordinate,
        ns,
        ts,
        ps,
        rates,
        integration_error,
        check=check,
    )
