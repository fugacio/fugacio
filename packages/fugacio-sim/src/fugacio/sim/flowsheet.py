"""Sequential-modular flowsheet solving with a differentiable recycle/tear solver.

A flowsheet with a recycle is an implicit problem: the value of a torn recycle
stream must equal what the flowsheet computes for it once that guess is fed back
in. Writing the single forward pass as ``g(tear, theta) -> tear`` (mix the feed
with the recycle guess, run the units, return the recomputed recycle), the
converged flowsheet is the fixed point ``tear* = g(tear*, theta)``.

`tear_solve` finds that fixed point with a **Wegstein-accelerated**
iteration -- the workhorse of sequential-modular simulators, far more robust than
plain direct substitution on tight recycles -- and differentiates the *converged*
solution by the implicit function theorem (a hand-written ``custom_vjp``). The
forward iteration count never appears in the backward pass, so a gradient of any
product spec with respect to an operating variable costs one adjoint solve, no
matter how many recycle iterations were needed. That is what makes whole-process,
recycle-closed gradient optimisation tractable.

The tear state can be any JAX pytree (a `Stream`, a
list of them, a dict, ...); it is flattened internally. Convergence is judged on
a relative norm, so mixed-scale states (flows, temperature, pressure) all
converge to the same relative tolerance without manual scaling.

`Flowsheet` is a thin declarative wrapper: register feeds and unit
functions, mark a tear, and call `Flowsheet.solve`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from fugacio.sim.stream import Stream


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4, 5, 6, 7))
def _wegstein(
    g: Callable[[Array, Any], Array],
    x0: Array,
    theta: Any,
    q_min: float,
    q_max: float,
    tol: float,
    atol: float,
    max_iter: int,
) -> Array:
    """Solve the flat fixed point ``x = g(x, theta)`` with bounded Wegstein steps.

    Wegstein estimates, component-by-component, the secant slope ``s_i`` of the
    update map between the last two iterates and takes the step ``x_{n+1} =
    q x_n + (1 - q) g(x_n)`` with ``q = s/(s - 1)`` (``q = 0`` is plain direct
    substitution). ``q`` is clipped to ``[q_min, q_max]`` for stability. The slope
    is dimensionless, so no state scaling is needed; convergence uses a relative
    norm with floor ``atol``.
    """

    def cond(carry: tuple[Array, Array, Array, Array, Array]) -> Array:
        _, _, _, i, err = carry
        return (err > tol) & (i < max_iter)

    def body(
        carry: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        x_prev, g_prev, x, i, _ = carry
        gx = g(x, theta)
        dx = x - x_prev
        slope = jnp.where(jnp.abs(dx) > 1e-13, (gx - g_prev) / dx, 0.0)
        q = jnp.where(jnp.abs(slope - 1.0) > 1e-13, slope / (slope - 1.0), 0.0)
        q = jnp.clip(q, q_min, q_max)
        x_new = q * x + (1.0 - q) * gx
        err = jnp.max(jnp.abs(x_new - x) / (atol + jnp.abs(x_new)))
        return x, gx, x_new, i + 1, err

    g0 = g(x0, theta)
    init = (x0, g0, g0, jnp.asarray(1), jnp.asarray(jnp.inf))
    _, _, x_star, _, _ = jax.lax.while_loop(cond, body, init)
    return x_star


def _wegstein_fwd(
    g: Callable[[Array, Any], Array],
    x0: Array,
    theta: Any,
    q_min: float,
    q_max: float,
    tol: float,
    atol: float,
    max_iter: int,
) -> tuple[Array, tuple[Array, Any]]:
    x_star = _wegstein(g, x0, theta, q_min, q_max, tol, atol, max_iter)
    return x_star, (x_star, theta)


def _wegstein_bwd(
    g: Callable[[Array, Any], Array],
    q_min: float,
    q_max: float,
    tol: float,
    atol: float,
    max_iter: int,
    res: tuple[Array, Any],
    x_bar: Array,
) -> tuple[Array, Any]:
    """Implicit-function-theorem adjoint: solve ``(I - dg/dx)^T w = x_bar`` by contraction.

    The recycle map ``g`` is a contraction near the solution (that is why the
    recycle converges), so the transposed system is solved by the same fixed-point
    contraction reusing ``g``'s vector-Jacobian product, then the parameter
    gradient is ``theta_bar = (dg/dtheta)^T w``.
    """
    x_star, theta = res
    _, vjp_x = jax.vjp(lambda x: g(x, theta), x_star)

    def w_cond(carry: tuple[Array, Array, Array]) -> Array:
        w_prev, w, i = carry
        rel = jnp.max(jnp.abs(w - w_prev) / (atol + jnp.abs(w)))
        return (rel > tol) & (i < max_iter)

    def w_body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        _, w, i = carry
        return w, x_bar + vjp_x(w)[0], i + 1

    w1 = x_bar + vjp_x(x_bar)[0]
    _, w_star, _ = jax.lax.while_loop(w_cond, w_body, (x_bar, w1, jnp.asarray(1)))

    _, vjp_theta = jax.vjp(lambda th: g(x_star, th), theta)
    theta_bar = vjp_theta(w_star)[0]
    return jnp.zeros_like(x_star), theta_bar


_wegstein.defvjp(_wegstein_fwd, _wegstein_bwd)


def tear_solve(
    g: Callable[[Any, Any], Any],
    tear0: Any,
    theta: Any = None,
    *,
    q_min: float = -5.0,
    q_max: float = 0.0,
    tol: float = 1e-10,
    atol: float = 1e-12,
    max_iter: int = 200,
) -> Any:
    """Converge a recycle by solving the tear fixed point ``tear = g(tear, theta)``.

    Args:
        g: One sequential-modular pass of the flowsheet. Given a tear-stream guess
            (any pytree) and the parameter pytree ``theta``, it runs the units and
            returns the recomputed tear stream(s) in the *same* pytree structure.
        tear0: Initial guess for the torn stream(s).
        theta: Differentiable parameter pytree (operating conditions, specs, feed).
            Pass the quantities you want to differentiate through here -- gradients
            flow to ``theta`` by implicit differentiation. Closed-over constants are
            fine but are treated as non-differentiable.
        q_min: Lower bound on the Wegstein acceleration factor.
        q_max: Upper bound on the Wegstein acceleration factor. The default
            ``[-5, 0]`` accelerates without over-damping; widen ``q_max`` toward
            ``1`` to damp oscillatory recycles.
        tol: Relative tolerance for the convergence norm.
        atol: Absolute floor for the convergence norm.
        max_iter: Iteration cap for both the forward and adjoint solves.

    Returns:
        The converged tear stream(s), in the structure of ``tear0``. Differentiable
        with respect to ``theta``.
    """
    flat0, unravel = ravel_pytree(tear0)

    def g_flat(x: Array, theta: Any) -> Array:
        out = g(unravel(x), theta)
        y, _ = ravel_pytree(out)
        return y

    x_star = _wegstein(g_flat, flat0, theta, q_min, q_max, tol, atol, max_iter)
    return unravel(x_star)


UnitFn = Callable[..., Any]


@dataclass
class _Unit:
    name: str
    fn: UnitFn
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass
class Flowsheet:
    """A small declarative flowsheet: named streams produced by connected units.

    Build a flowsheet by registering feeds and units, then designate a recycle
    *tear* and call `solve`. Each unit is a plain function of its input
    streams (and the shared ``theta``) returning one or more output streams; the
    flowsheet evaluates the units in registration order, which the caller arranges
    to be a valid sequential-modular order with the recycle torn.

    Example::

        fs = Flowsheet()
        fs.feed("fresh", fresh_stream)
        fs.unit("mixer", lambda fresh, rec, th: mix([fresh, rec]),
                inputs=("fresh", "recycle"), outputs=("mixed",))
        fs.unit("drum", lambda mixed, th: flash_drum(mixed, th["T"], th["P"]),
                inputs=("mixed",), outputs=("vapor", "liquid"))
        fs.unit("split", lambda liq, th: splitter(liq, [th["r"], 1 - th["r"]]),
                inputs=("liquid",), outputs=("recycle", "purge"))
        fs.tear("recycle", recycle_guess)
        streams = fs.solve({"T": 320.0, "P": 2e6, "r": 0.6})
        product = streams["vapor"]
    """

    feeds: dict[str, Stream] = field(default_factory=dict)
    units: list[_Unit] = field(default_factory=list)
    tears: dict[str, Stream] = field(default_factory=dict)

    def feed(self, name: str, stream: Stream) -> Flowsheet:
        """Register a fresh feed stream by name. Returns ``self`` for chaining."""
        self.feeds[name] = stream
        return self

    def unit(
        self,
        name: str,
        fn: UnitFn,
        *,
        inputs: Sequence[str],
        outputs: Sequence[str],
    ) -> Flowsheet:
        """Register a unit ``fn(*input_streams, theta) -> output stream(s)``.

        ``fn`` receives the named input streams positionally followed by the shared
        ``theta`` pytree, and returns either a single `Stream` (for one
        output name) or a tuple/list of streams aligned with ``outputs``.
        """
        self.units.append(_Unit(name, fn, tuple(inputs), tuple(outputs)))
        return self

    def tear(self, name: str, guess: Stream) -> Flowsheet:
        """Designate stream ``name`` as a recycle tear with an initial ``guess``."""
        self.tears[name] = guess
        return self

    def _evaluate(self, tears: dict[str, Stream], theta: Any) -> dict[str, Stream]:
        """Run every unit once, returning the full map of named streams."""
        streams: dict[str, Stream] = {**self.feeds, **tears}
        for u in self.units:
            args = [streams[name] for name in u.inputs]
            result = u.fn(*args, theta)
            produced = result if isinstance(result, list | tuple) else (result,)
            if len(produced) != len(u.outputs):
                raise ValueError(
                    f"unit {u.name!r} produced {len(produced)} outputs, expected {len(u.outputs)}"
                )
            for out_name, out_stream in zip(u.outputs, produced, strict=True):
                streams[out_name] = out_stream
        return streams

    def solve(self, theta: Any = None, **tear_solve_kwargs: Any) -> dict[str, Stream]:
        """Solve the flowsheet (closing any recycle) and return all named streams.

        ``theta`` is the differentiable parameter pytree passed to every unit; any
        output stream is differentiable with respect to it. Extra keyword arguments
        are forwarded to `tear_solve`.
        """
        if not self.tears:
            return self._evaluate({}, theta)

        names = tuple(self.tears.keys())

        def g(tear_tuple: tuple[Stream, ...], th: Any) -> tuple[Stream, ...]:
            tears = dict(zip(names, tear_tuple, strict=True))
            streams = self._evaluate(tears, th)
            return tuple(streams[name] for name in names)

        guesses = tuple(self.tears[name] for name in names)
        converged = tear_solve(g, guesses, theta, **tear_solve_kwargs)
        return self._evaluate(dict(zip(names, converged, strict=True)), theta)
