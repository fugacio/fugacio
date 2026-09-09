"""Reusable local derivatives of a converged numerical calculation.

One linearization evaluates the primal once and retains the residual data
needed by its JVP. Its transpose supplies VJPs without retracing the primal.
The object belongs to one operating point; construct a new one when that
point, its model parameters, or its initialization changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


def derivative_strategy(
    inputs: int, outputs: int, *, mode: str = "auto", batch_size: int = 1
) -> dict[str, Any]:
    """Validate derivative options and describe the selected direction count.

    ``auto`` uses the smaller of the input and output dimensions, preferring
    forward mode on a tie. This is a direction-count heuristic, not a promise
    that one orientation is faster for every property package. A batch size
    of one keeps tangent storage bounded for large nested process models.
    """
    if mode not in ("auto", "forward", "reverse"):
        raise ValueError("derivative mode must be auto, forward, or reverse")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("derivative batch size must be a positive integer")
    if inputs < 1 or outputs < 1:
        raise ValueError("linearization needs nonempty input and output arrays")
    selected = ("forward" if inputs <= outputs else "reverse") if mode == "auto" else mode
    return {
        "requested_mode": mode,
        "mode": selected,
        "inputs": inputs,
        "outputs": outputs,
        "directions": inputs if selected == "forward" else outputs,
        "batch_size": batch_size,
        "linearizations_per_evaluation": 1,
    }


@dataclass(frozen=True)
class Linearization:
    """A value and reusable JVP/VJP at one array-valued operating point.

    Create this object with ``linearize``. The stored callables are local
    linear maps, not nonlinear approximations valid at subsequent points.
    Auxiliary data, such as solve reports, isn't differentiated. Releasing
    the object releases its retained residuals; no global point cache exists.
    """

    point: Array
    value: Array
    auxiliary: Any
    _push: Callable[[Array], Array]

    def block_until_ready(self) -> Linearization:
        """Wait for both outputs and the residual buffers retained by the JVP.

        JAX's returned linear map is a registered partial pytree whose leaves
        include those buffers. Waiting for the primal output alone can miss
        asynchronous work that only a later derivative application consumes.
        This method also lets ``jax.block_until_ready`` synchronize this object.
        """
        jax.block_until_ready((self.point, self.value, self.auxiliary, self._push))
        return self

    def jvp(self, tangent: Array) -> Array:
        """Apply the local derivative to a tangent with the input's shape."""
        if tangent.shape != self.point.shape:
            raise ValueError("tangent shape must match the linearization point")
        return self._push(tangent)

    def vjp(self, cotangent: Array) -> Array:
        """Apply the transposed derivative to an output cotangent."""
        if cotangent.shape != self.value.shape:
            raise ValueError("cotangent shape must match the linearized value")
        return jax.linear_transpose(self._push, self.point)(cotangent)[0]

    def jacobian(self, *, mode: str = "auto", batch_size: int = 1) -> Array:
        """Build a Jacobian using bounded batches of the cheaper orientation.

        The result has shape ``value.shape + point.shape``, matching JAX's
        Jacobian convention. Forward and reverse modes use the same retained
        primal and the implicit derivative rules of the underlying solvers.
        """
        strategy = derivative_strategy(
            self.point.size, self.value.size, mode=mode, batch_size=batch_size
        )
        forward = strategy["mode"] == "forward"
        source = self.point if forward else self.value
        apply = self.jvp if forward else self.vjp
        pieces = []
        for start in range(0, source.size, batch_size):
            indices = jnp.arange(start, min(start + batch_size, source.size))
            seeds = jax.nn.one_hot(indices, source.size, dtype=source.dtype).reshape(
                (len(indices), *source.shape)
            )
            # Avoid adding a vmap level to large implicit kernels for the
            # default one-direction path. Batches remain explicitly opt-in.
            piece = apply(seeds[0])[None] if len(indices) == 1 else jax.vmap(apply)(seeds)
            pieces.append(piece.reshape(len(indices), -1))
        matrix = jnp.concatenate(pieces)
        return (matrix.T if forward else matrix).reshape((*self.value.shape, *self.point.shape))


def linearize(
    function: Callable[..., Any], point: Array, *, has_aux: bool = False
) -> Linearization:
    """Evaluate once and retain a reusable local derivative, optionally with reports.

    Inputs and differentiated outputs must be nonempty real floating arrays.
    Use an explicit vector adapter for parameter dictionaries or metric trees.
    The calculation needn't be wrapped in a single JIT; compiled unit kernels
    can retain their own compilation boundaries in a larger flowsheet.
    """
    point = jnp.asarray(point)
    if has_aux:
        value, push, auxiliary = jax.linearize(function, point, has_aux=True)
    else:
        value, push = jax.linearize(function, point)
        auxiliary = None
    value = jnp.asarray(value)
    if any(a.size == 0 or not jnp.issubdtype(a.dtype, jnp.floating) for a in (point, value)):
        raise ValueError("linearization requires nonempty real floating arrays")
    return Linearization(point, value, auxiliary, push)
