"""Tangent-plane stability analysis for an arbitrary fugacity model.

A mixture of overall composition ``z`` is *stable* as a single phase only if no
trial composition ``w`` lowers the Gibbs energy, i.e. if the (modified)
tangent-plane distance

    tm(w) = 1 + sum_i W_i (ln W_i + ln coeff_i(w) - d_i - 1),   d_i = ln z_i + ln coeff_i(z)

stays non-negative at every stationary point (``w = W / sum W``). Here
``coeff_i`` is whatever turns a composition into a fugacity: the activity
coefficient ``gamma_i`` for a liquid activity model, or the fugacity coefficient
``phi_i`` for an equation of state. Casting the test in terms of a generic
``ln_coeff_fn`` makes one implementation serve both worlds.

The companion of equilibrium: while :mod:`fugacio.thermo.equilibrium` answers
"given that it splits, into what?", this module answers "does it split at all?"
-- the test that decides whether a feed is one phase, needs a VLE flash, or (for
a liquid activity model with a miscibility gap) needs the liquid-liquid solver in
:mod:`fugacio.thermo.lle`. The most negative trial result also gives an excellent
*initial guess* for those splits, which those modules consume directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.activity.models import ActivityModel

ArrayLike = Array | float
LnCoeffFn = Callable[[Array], Array]


class TangentPlaneResult(NamedTuple):
    """Outcome of a tangent-plane stability search.

    Attributes:
        stable: ``True`` if the feed is single-phase stable (no negative ``tm``).
        tpd: The smallest modified tangent-plane distance found across all trials.
        split: The (normalised) trial composition that achieved ``tpd`` -- a ready
            initial guess for the incipient phase when the feed is unstable.
    """

    stable: Array
    tpd: Array
    split: Array


def tangent_plane_distance(ln_coeff_fn: LnCoeffFn, z: Array, w: Array) -> Array:
    """Gibbs tangent-plane distance of trial phase ``w`` relative to feed ``z``.

    ``tpd(w) = sum_i w_i (ln w_i + ln coeff_i(w) - ln z_i - ln coeff_i(z))``; a
    negative value means the trial phase ``w`` lies below the tangent plane at
    ``z`` and the feed can lower its Gibbs energy by forming it.
    """
    z = jnp.asarray(z)
    w = jnp.asarray(w)
    d = jnp.log(z) + ln_coeff_fn(z)
    return jnp.sum(w * (jnp.log(w) + ln_coeff_fn(w) - d))


def _run_trial(ln_coeff_fn: LnCoeffFn, d: Array, w0: Array, iters: int) -> tuple[Array, Array]:
    """Successive-substitution trial from ``w0``; returns ``(tm, w_converged)``."""

    def body(_: int, big_w: Array) -> Array:
        w = big_w / jnp.sum(big_w)
        return jnp.exp(d - ln_coeff_fn(w))

    big_w = jax.lax.fori_loop(0, iters, body, w0)
    w = big_w / jnp.sum(big_w)
    tm = 1.0 + jnp.sum(big_w * (jnp.log(big_w) + ln_coeff_fn(w) - d - 1.0))
    return tm, w


def stability_analysis_general(
    ln_coeff_fn: LnCoeffFn,
    z: Array,
    trials: Array,
    *,
    iters: int = 50,
    tol: float = 1e-8,
) -> TangentPlaneResult:
    """Michelsen stability test for a generic ``ln_coeff_fn`` and trial set.

    Args:
        ln_coeff_fn: Maps a composition to ``ln(coeff_i)`` (``ln gamma`` or ``ln phi``)
            at the fixed temperature/pressure of interest.
        z: Feed composition.
        trials: Stack of initial trial-phase compositions, shape ``(n_trials, n)``.
        iters: Successive-substitution iterations per trial.
        tol: Negative-``tm`` threshold below which the feed is declared unstable.

    Returns:
        A :class:`TangentPlaneResult`.
    """
    z = jnp.asarray(z)
    trials = jnp.asarray(trials)
    d = jnp.log(z) + ln_coeff_fn(z)
    tms, ws = jax.vmap(lambda w0: _run_trial(ln_coeff_fn, d, w0, iters))(trials)
    best = jnp.argmin(tms)
    tpd = tms[best]
    return TangentPlaneResult(stable=tpd >= -tol, tpd=tpd, split=ws[best])


def _enrichment_trials(z: Array, *, strength: float = 0.9) -> Array:
    """Trial compositions enriched toward each pure component (LLE seeds).

    For each component ``i``, builds a composition that mixes the feed with a spike
    toward pure ``i`` -- a robust starting set for detecting a miscibility gap.
    """
    z = jnp.asarray(z)
    n = z.shape[0]
    eye = jnp.eye(n)
    spikes = (1.0 - strength) * z[None, :] + strength * eye
    return spikes / jnp.sum(spikes, axis=1, keepdims=True)


def liquid_stability(
    model: ActivityModel,
    t: ArrayLike,
    z: Array,
    *,
    iters: int = 60,
    tol: float = 1e-8,
    strength: float = 0.9,
) -> TangentPlaneResult:
    """Test a liquid of composition ``z`` for splitting into two liquids at ``T``.

    Uses the activity coefficients as the ``ln_coeff_fn`` and trial phases enriched
    toward each pure component. A negative ``tpd`` flags a miscibility gap; the
    returned ``split`` seeds :func:`fugacio.thermo.lle.flash_lle`.
    """
    trials = _enrichment_trials(z, strength=strength)
    return stability_analysis_general(
        lambda w: model.ln_gamma(w, t), z, trials, iters=iters, tol=tol
    )
