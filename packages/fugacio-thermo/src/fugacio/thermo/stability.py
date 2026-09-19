"""Tangent-plane stability analysis: does a phase split at all?

A mixture of overall composition ``z`` is stable as a single phase only if no
trial phase ``w`` lies below the Gibbs-energy tangent plane at ``z``, i.e. if the
reduced tangent-plane distance

    tpd(w) = sum_i w_i (ln w_i + ln coeff_i(w) - d_i),   d_i = ln z_i + ln coeff_i(z)

is non-negative at every stationary point. ``coeff_i`` is whatever turns a
composition into a fugacity: the activity coefficient ``gamma_i`` of a liquid
activity model, or the fugacity coefficient ``phi_i`` of any phase branch of a
property package. Casting the test in terms of trial-phase functions lets one
search serve every model, and lets a caller test against any reference phase
(the feed's own lowest-Gibbs phase, or a returned flash phase).

`tpd_search` is that one search: a damped successive substitution in
``ln w`` from several starts on every trial branch, tracking the most negative
distance observed. Components absent from the feed stay exactly absent, so
feeds with zero entries are routine. A finite-start search can't prove a global
minimum; the result reports whether every trial reached a stationary point, and
a stability verdict is only established when it did.

Package-level tests live on every `fugacio.thermo.package.PropertyPackage`
(``stability``); `liquid_stability` tests a liquid against a second liquid for
activity models and seeds `fugacio.thermo.lle.flash_lle`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from fugacio.thermo.activity.models import ActivityModel

ArrayLike = Array | float
LnCoeffFn = Callable[[Array], Array]


class StabilityResult(NamedTuple):
    """Outcome of a tangent-plane stability search.

    Attributes:
        stable: ``True`` if no trial phase reached a distance below ``-tol``.
        tpd: The most negative reduced tangent-plane distance observed.
        trial: The final (normalized) composition of the most negative trial, a
            ready initial guess for the incipient phase of an unstable feed.
        branch: Index of the trial branch that produced ``trial`` (for package
            tests, 0 is liquid-like and 1 is vapour-like).
        converged: Whether every trial reached a finite stationary point. A
            verdict of stability is established only when this is also true.
    """

    stable: Array
    tpd: Array
    trial: Array
    branch: Array
    converged: Array


def _normalize(w: Array, support: Array) -> Array:
    w = jnp.where(support, jnp.maximum(w, 0.0), 0.0)
    return w / jnp.maximum(jnp.sum(w), 1e-300)


def tangent_plane_distance(ln_coeff_fn: LnCoeffFn, z: Array, w: Array) -> Array:
    """Reduced tangent-plane distance of trial phase ``w`` relative to feed ``z``.

    ``tpd(w) = sum_i w_i (ln w_i + ln coeff_i(w) - ln z_i - ln coeff_i(z))``; a
    negative value means the trial phase ``w`` lies below the tangent plane at
    ``z`` and the feed can lower its Gibbs energy by forming it. Components
    absent from ``z`` must be absent from ``w``; they contribute nothing.
    """
    z = jnp.asarray(z)
    w = jnp.asarray(w)
    support = z > 0
    d = jnp.log(jnp.where(support, z, 1.0)) + ln_coeff_fn(z)
    term = jnp.log(jnp.where(support, w, 1.0)) + ln_coeff_fn(w) - d
    return jnp.sum(jnp.where(support, w * term, 0.0))


def tpd_search(
    branches: Sequence[LnCoeffFn],
    d: Array,
    support: Array,
    starts: Array,
    *,
    iterations: int = 160,
    tol: float = 1e-7,
) -> StabilityResult:
    """Search trial phases on every branch against the reference potentials ``d``.

    Args:
        branches: Trial-phase functions ``w -> ln coeff(w)``, one per phase branch.
        d: Reference potentials ``ln x_i + ln coeff_i(x)`` of the tested phase.
        support: Components present in the tested system; others stay absent.
        starts: Initial trial compositions, shape ``(n_starts, n)``.
        iterations: Damped successive-substitution steps per start.
        tol: Stationarity tolerance on trial compositions and the stability
            threshold on the distance.

    Returns:
        The most negative distance observed and its trial phase. The search is
        a classification and carries no derivative.
    """
    branches = tuple(branches)
    d, support, starts = lax.stop_gradient((jnp.asarray(d), jnp.asarray(support), starts))
    d = jnp.where(support, d, 0.0)
    starts = jax.vmap(lambda w: _normalize(w, support))(jnp.asarray(starts, dtype=float))

    def composition(logw: Array) -> Array:
        return jax.nn.softmax(jnp.where(support, logw, -jnp.inf))

    def trial(fn: LnCoeffFn, w0: Array) -> tuple[Array, Array, Array]:
        def distance(w: Array) -> Array:
            term = jnp.log(jnp.maximum(w, 1e-300)) + fn(w) - d
            return jnp.sum(jnp.where(support, w * term, 0.0))

        def body(_: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
            logw, lowest = state
            w = composition(logw)
            proposed = jnp.clip(d - fn(w), -690.0, 690.0)
            return 0.5 * logw + 0.5 * proposed, jnp.minimum(lowest, distance(w))

        logw, lowest = lax.fori_loop(
            0, iterations, body, (jnp.log(jnp.maximum(w0, 1e-300)), distance(w0))
        )
        w = composition(logw)
        final = composition(d - fn(w))
        ok = (jnp.max(jnp.abs(w - final)) <= tol) & jnp.all(jnp.isfinite(w))
        value = jnp.minimum(lowest, distance(w))
        # A nonfinite trial (a branch that doesn't exist at this state) offers no
        # evidence of a split, but it does leave the verdict unestablished.
        value = jnp.where(jnp.isfinite(value), value, jnp.inf)
        return value, w, ok

    results = [jax.vmap(lambda w0, fn=fn: trial(fn, w0))(starts) for fn in branches]
    distances = jnp.stack([r[0] for r in results])
    trials = jnp.stack([r[1] for r in results])
    oks = jnp.stack([r[2] for r in results])
    flat = jnp.argmin(distances)
    branch, start = jnp.unravel_index(flat, distances.shape)
    lowest = distances[branch, start]
    return lax.stop_gradient(
        StabilityResult(
            stable=lowest >= -tol,
            tpd=lowest,
            trial=trials[branch, start],
            branch=branch,
            converged=jnp.all(oks),
        )
    )


def enrichment_starts(z: Array, *, strength: float = 0.95) -> Array:
    """Trial compositions enriched toward each pure component, plus the feed."""
    z = jnp.asarray(z, dtype=float)
    spikes = strength * jnp.eye(z.shape[0]) + (1.0 - strength) * z[None, :]
    return jnp.concatenate((z[None, :], spikes))


def feed_potentials(branches: Sequence[LnCoeffFn], z: Array, support: Array) -> Array:
    """Reference potentials of the feed's lowest-Gibbs single-phase branch."""
    z = jnp.asarray(z, dtype=float)
    ln_z = jnp.log(jnp.where(support, z, 1.0))
    values = jnp.stack([fn(z) for fn in branches])
    gibbs = jnp.sum(jnp.where(support, z * (ln_z + values), 0.0), axis=1)
    gibbs = jnp.where(jnp.isfinite(gibbs), gibbs, jnp.inf)
    return ln_z + values[jnp.argmin(gibbs)]


def liquid_stability(
    model: ActivityModel,
    t: ArrayLike,
    z: Array,
    *,
    iterations: int = 160,
    tol: float = 1e-7,
) -> StabilityResult:
    """Test a liquid of composition ``z`` for splitting into two liquids at ``T``.

    Uses the activity coefficients as the trial function and starts enriched
    toward each pure component. A negative ``tpd`` flags a miscibility gap; the
    returned ``trial`` seeds `fugacio.thermo.lle.flash_lle`.
    """
    z = jnp.asarray(z, dtype=float)
    support = z > 0

    def ln_gamma(w: Array) -> Array:
        return model.ln_gamma(w, t)

    d = jnp.log(jnp.where(support, z, 1.0)) + ln_gamma(z)
    return tpd_search((ln_gamma,), d, support, enrichment_starts(z), iterations=iterations, tol=tol)


__all__ = [
    "StabilityResult",
    "enrichment_starts",
    "feed_potentials",
    "liquid_stability",
    "tangent_plane_distance",
    "tpd_search",
]
