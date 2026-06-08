"""Non-ideal separation units: model-driven flash, decanter, three-phase flash.

These complement the equation-of-state blocks in :mod:`fugacio.sim.units` with the
non-ideal phase behaviour the gamma-phi property system unlocks:

* :func:`flash_vle` -- a two-phase V-L flash driven by *any*
  :class:`~fugacio.thermo.EquilibriumModel` (EOS or gamma-phi), so the same drum
  works on Peng-Robinson or NRTL/UNIQUAC/UNIFAC;
* :func:`decanter` -- a liquid-liquid separator (settling tank) that splits one
  feed into two conjugate liquid products via the isoactivity LLE flash; and
* :func:`three_phase_flash` -- a vapour + two-liquid (V-L-L) separator for
  heterogeneous systems (water/organic decantation, heteroazeotropic columns).

Every product is a differentiable :class:`~fugacio.sim.stream.Stream`; flows carry
gradients with respect to the operating ``T``, ``P``, the feed, and -- through the
model object -- the thermodynamic parameters themselves.
"""

from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp
from jax import Array

from fugacio.sim.stream import Stream
from fugacio.thermo import FlashResult, GammaPhiModel, flash_lle, flash_vlle

ArrayLike = Array | float


class _VLEModel(Protocol):
    """Minimal structural type: anything offering an isothermal-isobaric flash."""

    def flash_pt(self, t: ArrayLike, p: ArrayLike, z: Array) -> FlashResult: ...


def flash_vle(feed: Stream, t: ArrayLike, p: ArrayLike, model: _VLEModel) -> tuple[Stream, Stream]:
    """Two-phase vapour-liquid flash of ``feed`` at ``(T, P)`` using ``model``.

    Works with any :class:`~fugacio.thermo.EquilibriumModel` -- an
    :class:`~fugacio.thermo.EOSModel` for the phi-phi route or a
    :class:`~fugacio.thermo.GammaPhiModel` for the activity-coefficient route --
    so the drum's thermodynamics are chosen by the model passed in.

    Returns:
        ``(vapor, liquid)`` product streams at ``(T, P)``.
    """
    res = model.flash_pt(t, p, feed.z)
    total = feed.total
    t_arr = jnp.asarray(t)
    p_arr = jnp.asarray(p)
    vapor = Stream(n=res.y * res.beta * total, t=t_arr, p=p_arr, components=feed.components)
    liquid = Stream(
        n=res.x * (1.0 - res.beta) * total, t=t_arr, p=p_arr, components=feed.components
    )
    return vapor, liquid


def decanter(
    feed: Stream,
    model: GammaPhiModel,
    *,
    t: ArrayLike | None = None,
    tol: float = 1e-12,
    max_iter: int = 400,
) -> tuple[Stream, Stream]:
    """Liquid-liquid settler: split ``feed`` into two conjugate liquid products.

    Solves the isoactivity LLE flash (:func:`fugacio.thermo.flash_lle`) with the
    model's activity description at temperature ``t`` (default: the feed
    temperature) and the feed pressure. For a feed outside any miscibility gap the
    LLE flash collapses to the trivial split and one product carries essentially
    the whole feed -- check with :func:`fugacio.thermo.liquid_stability` upstream
    if that matters.

    Returns:
        ``(liquid_I, liquid_II)`` product streams.
    """
    t_arr = feed.t if t is None else jnp.asarray(t)
    res = flash_lle(model.activity, t_arr, feed.z, tol=tol, max_iter=max_iter)
    total = feed.total
    liquid_i = Stream(
        n=res.x_i * (1.0 - res.psi) * total, t=t_arr, p=feed.p, components=feed.components
    )
    liquid_ii = Stream(n=res.x_ii * res.psi * total, t=t_arr, p=feed.p, components=feed.components)
    return liquid_i, liquid_ii


def three_phase_flash(
    feed: Stream,
    t: ArrayLike,
    p: ArrayLike,
    model: GammaPhiModel,
    *,
    tol: float = 1e-11,
    max_iter: int = 300,
) -> tuple[Stream, Stream, Stream]:
    """Vapour-liquid-liquid (V-L-L) flash of ``feed`` at ``(T, P)``.

    Drives the three-phase flash (:func:`fugacio.thermo.flash_vlle`) with the
    model's activity liquid and its EOS/ideal vapour. Use for heterogeneous
    systems -- water/organic decantation and heteroazeotropic distillation -- where
    a vapour coexists with two liquids.

    Returns:
        ``(vapor, liquid_I, liquid_II)`` product streams. When the feed is not
        genuinely three-phase one of the liquid flows collapses to (near) zero.
    """
    res = flash_vlle(
        model.activity,
        t,
        p,
        feed.z,
        model.tc,
        model.pc,
        model.omega,
        eos=model.eos,
        kij=model.kij,
        vapor=model.vapor,
        poynting=model.poynting,
        phi_saturation=model.phi_saturation,
        tol=tol,
        max_iter=max_iter,
    )
    total = feed.total
    t_arr = jnp.asarray(t)
    p_arr = jnp.asarray(p)
    vapor = Stream(n=res.y * res.beta_v * total, t=t_arr, p=p_arr, components=feed.components)
    liquid_i = Stream(n=res.x_i * res.beta_l1 * total, t=t_arr, p=p_arr, components=feed.components)
    liquid_ii = Stream(
        n=res.x_ii * res.beta_l2 * total, t=t_arr, p=p_arr, components=feed.components
    )
    return vapor, liquid_i, liquid_ii


__all__ = ["decanter", "flash_vle", "three_phase_flash"]
