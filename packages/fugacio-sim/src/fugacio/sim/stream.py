"""Material streams: the data passed between flowsheet unit operations.

A `Stream` carries per-component molar flows together with temperature and
pressure. It is registered as a JAX pytree (the *flows*, ``T`` and ``P`` are
differentiable leaves while the component *names* are static metadata) so an
entire flowsheet built from streams remains end-to-end differentiable. You can
take a gradient of any downstream quantity with respect to a feed flow,
temperature, or pressure.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

ArrayLike = Array | float


@dataclass(frozen=True)
class Stream:
    """A process stream of fixed composition basis.

    Attributes:
        n: Per-component molar flow rates (mol/s), 1-D array aligned with ``components``.
        t: Temperature (K).
        p: Pressure (Pa).
        components: Canonical component names (static metadata).
    """

    n: Array
    t: Array
    p: Array
    components: tuple[str, ...]

    @property
    def total(self) -> Array:
        """Total molar flow rate (mol/s)."""
        return jnp.sum(self.n)

    @property
    def z(self) -> Array:
        """Mole fractions (the flow normalised to sum to one)."""
        return self.n / jnp.sum(self.n)

    @classmethod
    def from_fractions(
        cls,
        components: tuple[str, ...],
        z: Array,
        flow: ArrayLike,
        t: ArrayLike,
        p: ArrayLike,
    ) -> Stream:
        """Build a stream from mole fractions ``z`` and a total molar ``flow``."""
        z = jnp.asarray(z)
        return cls(
            n=z * jnp.asarray(flow),
            t=jnp.asarray(t),
            p=jnp.asarray(p),
            components=tuple(components),
        )


jax.tree_util.register_dataclass(
    Stream,
    data_fields=["n", "t", "p"],
    meta_fields=["components"],
)
