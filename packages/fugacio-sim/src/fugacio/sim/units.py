"""Differentiable unit operations.

This is the thin end-to-end spike that proves the headline Fugacio claim: a
process unit built on the equation-of-state phase equilibrium in
:mod:`fugacio.thermo` is differentiable with respect to its operating
conditions. :func:`flash_drum` performs a rigorous isothermal-isobaric flash and
returns vapour and liquid product :class:`~fugacio.sim.stream.Stream` objects;
because the underlying :func:`fugacio.thermo.flash_pt` carries implicit-function
gradients, you can differentiate product flows, recoveries, or purities with
respect to the drum temperature, pressure, or feed -- the basis for
gradient-based flowsheet optimisation.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from fugacio.sim.stream import Stream
from fugacio.thermo import PR, CubicEOS, component_arrays, flash_pt

ArrayLike = Array | float


def flash_drum(
    feed: Stream,
    t: ArrayLike,
    p: ArrayLike,
    *,
    eos: CubicEOS = PR,
    kij: Array | None = None,
) -> tuple[Stream, Stream]:
    """Flash a feed stream at temperature ``t`` and pressure ``p``.

    Args:
        feed: Inlet :class:`~fugacio.sim.stream.Stream`.
        t: Drum temperature (K).
        p: Drum pressure (Pa).
        eos: Cubic equation of state to use (defaults to Peng-Robinson).
        kij: Optional binary interaction matrix.

    Returns:
        ``(vapor, liquid)`` product streams. Their flows are differentiable with
        respect to ``t``, ``p`` and the feed.
    """
    arr = component_arrays(list(feed.components))
    result = flash_pt(eos, t, p, feed.z, arr["tc"], arr["pc"], arr["omega"], kij=kij)
    total = feed.total
    t_arr = jnp.asarray(t)
    p_arr = jnp.asarray(p)
    vapor = Stream(
        n=result.y * result.beta * total,
        t=t_arr,
        p=p_arr,
        components=feed.components,
    )
    liquid = Stream(
        n=result.x * (1.0 - result.beta) * total,
        t=t_arr,
        p=p_arr,
        components=feed.components,
    )
    return vapor, liquid


def mix(streams: list[Stream], *, t: ArrayLike | None = None, p: ArrayLike | None = None) -> Stream:
    """Combine streams by an exact component material balance.

    The molar balance is rigorous (flows add). Outlet temperature defaults to the
    molar-flow-weighted average of the inlets (a pragmatic placeholder until an
    energy-balance mixer lands); pass ``t`` to override. Outlet pressure defaults
    to the lowest inlet pressure.

    Raises:
        ValueError: if the streams do not share an identical component list.
    """
    if not streams:
        raise ValueError("mix requires at least one stream")
    components = streams[0].components
    for s in streams[1:]:
        if s.components != components:
            raise ValueError("all streams must share the same component list to mix")
    n_total = jnp.sum(jnp.stack([s.n for s in streams]), axis=0)
    if t is None:
        flows = jnp.stack([s.total for s in streams])
        temps = jnp.stack([s.t for s in streams])
        t_out = jnp.sum(flows * temps) / jnp.sum(flows)
    else:
        t_out = jnp.asarray(t)
    p_out = jnp.min(jnp.stack([s.p for s in streams])) if p is None else jnp.asarray(p)
    return Stream(n=n_total, t=t_out, p=p_out, components=components)
