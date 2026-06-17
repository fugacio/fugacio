"""Process streams for heat integration: the data the pinch targets act on.

A `HeatStream` is the heat-integration view of a process stream: a duty
that must be supplied (a *cold* stream, heated from a low supply to a high target
temperature) or removed (a *hot* stream, cooled from a high supply to a low
target). Under the usual constant-heat-capacity assumption a stream is fully
described by its supply and target temperatures and its **heat-capacity
flowrate** ``CP = m * cp`` (W/K); the duty is then ``CP * |Ts - Tt|``.

The stream is a registered JAX pytree (temperatures, ``CP`` and the film
coefficient are differentiable leaves; the name is static metadata), so every
pinch target downstream (minimum utilities, the pinch temperature, the area
and capital targets) is differentiable with respect to the stream data. That is
what lets `fugacio.sim.integration.optimal_dt_min` optimise heat recovery
by gradients.

`heat_stream` builds a `HeatStream` straight from a
`Stream` and a target temperature, taking the duty
(and hence ``CP``) from the two-phase-aware stream enthalpy in
`fugacio.sim.properties`, so the heat-integration model inherits the real
thermodynamics of the flowsheet.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.stream import Stream

ArrayLike = Array | float

#: Default film heat-transfer coefficient (W/m^2/K) when none is supplied. A
#: middling value for a process fluid; override per stream for area targeting.
DEFAULT_FILM_COEFFICIENT = 500.0


@dataclass(frozen=True)
class HeatStream:
    """A stream carrying a sensible-heat duty for heat integration.

    Attributes:
        t_supply: Supply (inlet) temperature (K).
        t_target: Target (outlet) temperature (K).
        cp: Heat-capacity flowrate ``CP = m * cp`` (W/K), assumed constant
            between supply and target. Always positive.
        h: Film heat-transfer coefficient (W/m^2/K) for area targeting.
        name: Stream label (static metadata).

    A stream is **hot** (a heat source, to be cooled) when ``t_supply >
    t_target`` and **cold** (a heat sink, to be heated) otherwise.
    """

    t_supply: Array
    t_target: Array
    cp: Array
    h: Array
    name: str

    @property
    def is_hot(self) -> Array:
        """Whether this is a hot stream (supply hotter than target)."""
        return self.t_supply > self.t_target

    @property
    def duty(self) -> Array:
        """Magnitude of the stream duty ``CP * |Ts - Tt|`` (W)."""
        return self.cp * jnp.abs(self.t_supply - self.t_target)

    @property
    def t_hot(self) -> Array:
        """The higher of the supply and target temperatures (K)."""
        return jnp.maximum(self.t_supply, self.t_target)

    @property
    def t_cold(self) -> Array:
        """The lower of the supply and target temperatures (K)."""
        return jnp.minimum(self.t_supply, self.t_target)


jax.tree_util.register_dataclass(
    HeatStream,
    data_fields=["t_supply", "t_target", "cp", "h"],
    meta_fields=["name"],
)


def make_stream(
    t_supply: ArrayLike,
    t_target: ArrayLike,
    cp: ArrayLike,
    *,
    h: ArrayLike = DEFAULT_FILM_COEFFICIENT,
    name: str = "",
) -> HeatStream:
    """Build a `HeatStream` from supply/target temperatures and ``CP``.

    Args:
        t_supply: Supply temperature (K).
        t_target: Target temperature (K).
        cp: Heat-capacity flowrate ``CP`` (W/K), positive.
        h: Film heat-transfer coefficient (W/m^2/K).
        name: Optional label.

    Returns:
        A `HeatStream` with array-valued leaves.
    """
    return HeatStream(
        t_supply=jnp.asarray(t_supply, dtype=float),
        t_target=jnp.asarray(t_target, dtype=float),
        cp=jnp.asarray(cp, dtype=float),
        h=jnp.asarray(h, dtype=float),
        name=name,
    )


def heat_stream(
    stream: Stream,
    t_target: ArrayLike,
    *,
    h: ArrayLike = DEFAULT_FILM_COEFFICIENT,
    name: str = "",
) -> HeatStream:
    """Extract a `HeatStream` from a process `Stream` and a target ``T``.

    The duty is the *actual* enthalpy change of the stream between its temperature
    and ``t_target`` (computed with the two-phase-aware
    `fugacio.sim.properties.enthalpy_flow`), and the constant ``CP`` is that
    duty divided by the temperature span, so the heat-integration stream carries
    the flowsheet's real thermodynamics. Differentiable in the stream state.

    Args:
        stream: The process stream at its supply temperature.
        t_target: Target temperature to heat/cool the stream to (K).
        h: Film heat-transfer coefficient (W/m^2/K).
        name: Optional label.

    Returns:
        A `HeatStream`; hot if ``stream.t > t_target``, cold otherwise.
    """
    from fugacio.sim.properties import enthalpy_flow

    t_supply = stream.t
    t_tgt = jnp.asarray(t_target, dtype=float)
    target_stream = Stream(n=stream.n, t=t_tgt, p=stream.p, components=stream.components)
    duty = jnp.abs(enthalpy_flow(stream) - enthalpy_flow(target_stream))
    span = jnp.abs(t_supply - t_tgt)
    cp = duty / jnp.where(span > 1e-9, span, 1.0)
    return HeatStream(
        t_supply=jnp.asarray(t_supply, dtype=float),
        t_target=t_tgt,
        cp=cp,
        h=jnp.asarray(h, dtype=float),
        name=name,
    )


def stack(streams: list[HeatStream]) -> tuple[Array, Array, Array, Array]:
    """Stack a list of streams into ``(t_supply, t_target, cp, h)`` arrays of shape ``(n,)``."""
    if not streams:
        raise ValueError("need at least one stream")
    t_supply = jnp.stack([jnp.asarray(s.t_supply, dtype=float) for s in streams])
    t_target = jnp.stack([jnp.asarray(s.t_target, dtype=float) for s in streams])
    cp = jnp.stack([jnp.asarray(s.cp, dtype=float) for s in streams])
    h = jnp.stack([jnp.asarray(s.h, dtype=float) for s in streams])
    return t_supply, t_target, cp, h
