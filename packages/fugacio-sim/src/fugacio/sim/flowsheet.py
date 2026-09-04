"""Sequential-modular flowsheet solving with a differentiable recycle/tear solver.

A flowsheet with a recycle is an implicit problem: the value of a torn recycle
stream must equal what the flowsheet computes for it once that guess is fed back
in. Writing the single forward pass as ``g(tear, theta) -> tear`` (mix the feed
with the recycle guess, run the units, return the recomputed recycle), the
converged flowsheet is the fixed point ``tear* = g(tear*, theta)``.

`tear_solve` finds that fixed point with one of three tear methods: a
**Wegstein-accelerated** direct substitution (the workhorse of sequential-modular
simulators, far more robust than plain substitution on tight recycles), a
**Broyden** quasi-Newton iteration (rank-one inverse-Jacobian updates, the
preferred method for several interacting tears), or a full **Newton** iteration
on ``g(x) - x`` with the autodiff Jacobian (quadratic convergence for stiff
recycles, at one Jacobian per step). All three differentiate the *converged*
solution by the implicit function theorem (a hand-written ``custom_vjp`` that
solves the small dense adjoint system ``(I - dg/dx)^T w = x_bar``). The forward
iteration count never appears in the backward pass, so a gradient of any product
spec with respect to an operating variable costs one adjoint solve, no matter how
many recycle iterations were needed. That is what makes whole-process,
recycle-closed gradient optimisation tractable.

The tear state can be any JAX pytree (a `Stream`, a
list of them, a dict, ...); it is flattened internally. Convergence is judged on
a relative norm, so mixed-scale states (flows, temperature, pressure) all
converge to the same relative tolerance without manual scaling.

`Flowsheet` is a declarative wrapper: register feeds and unit functions and call
`Flowsheet.solve`. The flowsheet analyses its own connectivity: units are
grouped into strongly connected components (Tarjan's algorithm), the
components are ordered topologically, and within every recycle loop a minimal
set of tear streams is chosen automatically (a greedy feedback-arc heuristic
that prefers the streams whose removal leaves the least coupling). Tears may
also be designated by hand with `Flowsheet.tear`, which additionally supplies
the starting guess; automatically chosen tears are seeded from a first pass
of the loop.
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

TEAR_METHODS: tuple[str, ...] = ("wegstein", "broyden", "newton")
"""Recycle convergence methods accepted by `tear_solve` and `Flowsheet.solve`."""


def _rel_err(step: Array, x_new: Array, atol: float) -> Array:
    return jnp.max(jnp.abs(step) / (atol + jnp.abs(x_new)))


def _wegstein_loop(
    g: Callable[[Array, Any], Array],
    x0: Array,
    theta: Any,
    q_min: float,
    q_max: float,
    tol: float,
    atol: float,
    max_iter: int,
) -> Array:
    """Bounded Wegstein iteration on ``x = g(x, theta)``.

    Wegstein estimates, component-by-component, the secant slope ``s_i`` of the
    update map between the last two iterates and takes the step ``x_{n+1} =
    q x_n + (1 - q) g(x_n)`` with ``q = s/(s - 1)`` (``q = 0`` is plain direct
    substitution). ``q`` is clipped to ``[q_min, q_max]`` for stability. The slope
    is dimensionless, so no state scaling is needed.
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
        return x, gx, x_new, i + 1, _rel_err(x_new - x, x_new, atol)

    g0 = g(x0, theta)
    init = (x0, g0, g0, jnp.asarray(1), jnp.asarray(jnp.inf))
    _, _, x_star, _, _ = jax.lax.while_loop(cond, body, init)
    return x_star


def _broyden_loop(
    g: Callable[[Array, Any], Array],
    x0: Array,
    theta: Any,
    tol: float,
    atol: float,
    max_iter: int,
) -> Array:
    """Broyden's (good) method on ``F(x) = g(x, theta) - x`` in scaled variables.

    The inverse Jacobian starts as ``-I`` (exact when the recycle barely feeds
    back on itself) and is corrected by rank-one updates from the observed
    secant pairs, so the cost per iteration is one flowsheet pass and no
    Jacobian, while the convergence is superlinear once the loop interactions
    have been learned. Variables are scaled by ``atol + |x0|`` so flows,
    temperatures, and pressures are treated even-handedly.
    """
    scale = atol + jnp.abs(x0)

    def f(s: Array) -> Array:
        x = s * scale
        return (g(x, theta) - x) / scale

    n = x0.shape[0]

    def cond(carry: tuple[Array, Array, Array, Array, Array]) -> Array:
        _, _, _, i, err = carry
        return (err > tol) & (i < max_iter)

    def body(
        carry: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        s, fs, b_inv, i, _ = carry
        ds = -b_inv @ fs
        s_new = s + ds
        fs_new = f(s_new)
        df = fs_new - fs
        b_df = b_inv @ df
        denom = ds @ b_df
        update = jnp.outer(ds - b_df, ds @ b_inv) / jnp.where(
            jnp.abs(denom) > 1e-300, denom, 1e-300
        )
        b_new = jnp.where(jnp.abs(denom) > 1e-14 * (1.0 + ds @ ds), b_inv + update, b_inv)
        return s_new, fs_new, b_new, i + 1, _rel_err(ds * scale, s_new * scale, atol)

    s0 = x0 / scale
    init = (s0, f(s0), -jnp.eye(n), jnp.asarray(0), jnp.asarray(jnp.inf))
    s_star, _, _, _, _ = jax.lax.while_loop(cond, body, init)
    return s_star * scale


def _newton_loop(
    g: Callable[[Array, Any], Array],
    x0: Array,
    theta: Any,
    tol: float,
    atol: float,
    max_iter: int,
) -> Array:
    """Damped Newton on ``F(x) = g(x, theta) - x`` with the autodiff Jacobian."""
    scale = atol + jnp.abs(x0)
    alphas = jnp.array([1.0, 0.5, 0.25, 0.1])

    def f(s: Array) -> Array:
        x = s * scale
        return (g(x, theta) - x) / scale

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        _, i, err = carry
        return (err > tol) & (i < max_iter)

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        s, i, _ = carry
        fs, jac = f(s), jax.jacobian(f)(s)
        ds = jnp.linalg.solve(jac, -fs)

        def norm_at(alpha: Array) -> Array:
            r = f(s + alpha * ds)
            return jnp.sqrt(jnp.sum(r * r))

        norms = jnp.stack([norm_at(a) for a in alphas])
        norms = jnp.where(jnp.isfinite(norms), norms, jnp.inf)
        step = alphas[jnp.argmin(norms)] * ds
        s_new = s + step
        return s_new, i + 1, _rel_err(step * scale, s_new * scale, atol)

    s_star, _, _ = jax.lax.while_loop(
        cond, body, (x0 / scale, jnp.asarray(0), jnp.asarray(jnp.inf))
    )
    return s_star * scale


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4, 5, 6, 7, 8))
def _tear_root(
    g: Callable[[Array, Any], Array],
    x0: Array,
    theta: Any,
    method: str,
    q_min: float,
    q_max: float,
    tol: float,
    atol: float,
    max_iter: int,
) -> Array:
    """Solve the flat fixed point ``x = g(x, theta)`` with the chosen tear method."""
    if method == "wegstein":
        return _wegstein_loop(g, x0, theta, q_min, q_max, tol, atol, max_iter)
    if method == "broyden":
        return _broyden_loop(g, x0, theta, tol, atol, max_iter)
    if method == "newton":
        return _newton_loop(g, x0, theta, tol, atol, max_iter)
    raise ValueError(f"unknown tear method {method!r}; choose from {TEAR_METHODS}")


def _tear_root_fwd(
    g: Callable[[Array, Any], Array],
    x0: Array,
    theta: Any,
    method: str,
    q_min: float,
    q_max: float,
    tol: float,
    atol: float,
    max_iter: int,
) -> tuple[Array, tuple[Array, Any]]:
    x_star = _tear_root(g, x0, theta, method, q_min, q_max, tol, atol, max_iter)
    return x_star, (x_star, theta)


def _tear_root_bwd(
    g: Callable[[Array, Any], Array],
    method: str,
    q_min: float,
    q_max: float,
    tol: float,
    atol: float,
    max_iter: int,
    res: tuple[Array, Any],
    x_bar: Array,
) -> tuple[Array, Any]:
    """Implicit-function-theorem adjoint: ``(I - dg/dx)^T w = x_bar``, then ``(dg/dtheta)^T w``.

    For the Wegstein method the recycle map is a contraction near the solution
    (that is why direct substitution converges), so the transposed system is
    solved by the same contraction using only vector-Jacobian products of ``g``,
    which scales to the long tear vectors of stage-by-stage column models. The
    Newton and Broyden methods also converge recycles that are *not*
    contractions, so their adjoint forms the (small, dense) Jacobian and solves
    the system directly.
    """
    x_star, theta = res
    _, vjp_x = jax.vjp(lambda x: g(x, theta), x_star)

    if method == "wegstein":

        def w_cond(carry: tuple[Array, Array, Array]) -> Array:
            w_prev, w, i = carry
            return (_rel_err(w - w_prev, w, atol) > tol) & (i < max_iter)

        def w_body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
            _, w, i = carry
            return w, x_bar + vjp_x(w)[0], i + 1

        w1 = x_bar + vjp_x(x_bar)[0]
        _, w_star, _ = jax.lax.while_loop(w_cond, w_body, (x_bar, w1, jnp.asarray(1)))
    else:
        jac = jax.jacobian(lambda x: g(x, theta))(x_star)
        n = x_star.shape[0]
        w_star = jnp.linalg.solve((jnp.eye(n) - jac).T, x_bar)

    _, vjp_theta = jax.vjp(lambda th: g(x_star, th), theta)
    theta_bar = vjp_theta(w_star)[0]
    return jnp.zeros_like(x_star), theta_bar


_tear_root.defvjp(_tear_root_fwd, _tear_root_bwd)


def tear_solve(
    g: Callable[[Any, Any], Any],
    tear0: Any,
    theta: Any = None,
    *,
    method: str = "wegstein",
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
            Pass the quantities you want to differentiate through here; gradients
            flow to ``theta`` by implicit differentiation. Closed-over constants are
            fine but are treated as non-differentiable.
        method: ``"wegstein"`` (default), ``"broyden"``, or ``"newton"``; see the
            module docstring for when each is preferable.
        q_min: Lower bound on the Wegstein acceleration factor.
        q_max: Upper bound on the Wegstein acceleration factor. The default
            ``[-5, 0]`` accelerates without over-damping; widen ``q_max`` toward
            ``1`` to damp oscillatory recycles.
        tol: Relative tolerance for the convergence norm.
        atol: Absolute floor for the convergence norm.
        max_iter: Iteration cap for the forward solve.

    Returns:
        The converged tear stream(s), in the structure of ``tear0``. Differentiable
        with respect to ``theta``.

    Raises:
        ValueError: for an unknown ``method``.
    """
    if method not in TEAR_METHODS:
        raise ValueError(f"unknown tear method {method!r}; choose from {TEAR_METHODS}")
    flat0, unravel = ravel_pytree(tear0)

    def g_flat(x: Array, theta: Any) -> Array:
        out = g(unravel(x), theta)
        y, _ = ravel_pytree(out)
        return y

    x_star = _tear_root(g_flat, flat0, theta, method, q_min, q_max, tol, atol, max_iter)
    return unravel(x_star)


UnitFn = Callable[..., Any]


@dataclass
class _Unit:
    name: str
    fn: UnitFn
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class Partition:
    """One block of the flowsheet's sequential-modular calculation order.

    Attributes:
        units: Unit names in evaluation order (a valid order once ``tears`` are
            treated as known).
        tears: Stream names torn to break this block's recycle loops (empty for an
            acyclic block).
    """

    units: tuple[str, ...]
    tears: tuple[str, ...]

    @property
    def cyclic(self) -> bool:
        """Whether the block contains a recycle."""
        return bool(self.tears)


# --------------------------------------------------------------------------- #
# Graph analysis
# --------------------------------------------------------------------------- #

Edge = tuple[int, int, str]  # (producer unit, consumer unit, stream name)


def _strongly_connected(n: int, edges: Sequence[Edge]) -> list[list[int]]:
    """Tarjan's algorithm; components come out in reverse topological order."""
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for a, b, _ in edges:
        adjacency[a].append(b)
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: list[int] = []
    counter = 0
    result: list[list[int]] = []

    def visit(v: int) -> None:
        nonlocal counter
        index[v] = low[v] = counter
        counter += 1
        stack.append(v)
        on_stack[v] = True
        for w in adjacency[v]:
            if index[w] < 0:
                visit(w)
                low[v] = min(low[v], low[w])
            elif on_stack[w]:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            result.append(comp)

    for v in range(n):
        if index[v] < 0:
            visit(v)
    return result


def _cyclic(nodes: Sequence[int], edges: Sequence[Edge]) -> bool:
    """Whether the subgraph on ``nodes`` with ``edges`` contains a cycle."""
    node_set = set(nodes)
    inner = [e for e in edges if e[0] in node_set and e[1] in node_set]
    if any(a == b for a, b, _ in inner):
        return True
    return any(len(c) > 1 for c in _strongly_connected(max(nodes) + 1, inner))


def _coupling(nodes: Sequence[int], edges: Sequence[Edge]) -> int:
    """Number of edges that still lie inside a recycle (score for tear selection)."""
    node_set = set(nodes)
    inner = [e for e in edges if e[0] in node_set and e[1] in node_set]
    comps = _strongly_connected(max(nodes) + 1, inner)
    comp_of = {v: i for i, c in enumerate(comps) for v in c}
    return sum(
        1
        for a, b, _ in inner
        if comp_of[a] == comp_of[b] and (a == b or len(comps[comp_of[a]]) > 1)
    )


def _topological(nodes: Sequence[int], edges: Sequence[Edge]) -> list[int]:
    """Kahn's algorithm on the acyclic subgraph; ties keep registration order."""
    node_set = set(nodes)
    indeg = dict.fromkeys(nodes, 0)
    succ: dict[int, list[int]] = {v: [] for v in nodes}
    for a, b, _ in edges:
        if a in node_set and b in node_set:
            indeg[b] += 1
            succ[a].append(b)
    ready = sorted(v for v in nodes if indeg[v] == 0)
    order: list[int] = []
    while ready:
        v = ready.pop(0)
        order.append(v)
        for w in succ[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                ready.append(w)
        ready.sort()
    if len(order) != len(nodes):  # pragma: no cover - guarded by the tear selection
        raise ValueError("internal error: graph still cyclic after tearing")
    return order


def _select_tears(
    nodes: Sequence[int], edges: Sequence[Edge], preferred: Sequence[str]
) -> tuple[str, ...]:
    """Greedy feedback-arc tear selection for one strongly connected block.

    User-designated streams (``preferred``) that lie inside the block are torn
    first. While the block is still cyclic, the candidate stream whose removal
    leaves the fewest edges inside recycle loops is torn, the classic "break the
    most loops per tear" heuristic. Ties go to a stream that flows *backwards*
    in registration order (units are usually registered in process order, so
    the back edge is the recycle the author has in mind), then to the stream
    carried by the most edges, then alphabetically.
    """
    node_set = set(nodes)
    inner = [e for e in edges if e[0] in node_set and e[1] in node_set]
    streams_inside = {name for _, _, name in inner}
    tears: list[str] = [s for s in preferred if s in streams_inside]

    def remaining(torn: Sequence[str]) -> list[Edge]:
        return [e for e in inner if e[2] not in torn]

    while _cyclic(nodes, remaining(tears)):
        candidates = sorted(streams_inside - set(tears))
        best: tuple[int, int, int, str] | None = None
        for s in candidates:
            after = remaining([*tears, s])
            score = _coupling(nodes, after)
            forward = 0 if any(a > b for a, b, name in inner if name == s) else 1
            multiplicity = -sum(1 for e in inner if e[2] == s)
            key = (score, forward, multiplicity, s)
            if best is None or key < best:
                best = key
        assert best is not None
        tears.append(best[3])
    return tuple(tears)


@dataclass
class Flowsheet:
    """A declarative flowsheet: named streams produced by connected units.

    Build a flowsheet by registering feeds and units, then call `solve`. Each
    unit is a plain function of its input streams (and the shared ``theta``)
    returning one or more output streams. The flowsheet works out the
    calculation order itself: `partition` groups the units into strongly
    connected blocks, orders the blocks, and chooses tear streams inside every
    recycle loop. `tear` optionally designates a tear by hand together with its
    starting guess; automatically selected tears are seeded by one pass of the
    loop with an empty recycle.

    Example::

        fs = Flowsheet()
        fs.feed("fresh", fresh_stream)
        fs.unit("mixer", lambda fresh, rec, th: mix([fresh, rec]),
                inputs=("fresh", "recycle"), outputs=("mixed",))
        fs.unit("drum", lambda mixed, th: flash_drum(mixed, th["T"], th["P"]),
                inputs=("mixed",), outputs=("vapor", "liquid"))
        fs.unit("split", lambda liq, th: splitter(liq, [th["r"], 1 - th["r"]]),
                inputs=("liquid",), outputs=("recycle", "purge"))
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
        """Designate stream ``name`` as a recycle tear with an initial ``guess``.

        Hand-designated tears are honoured by `partition` (and supplemented if
        they do not break every loop). Designating a tear is also the way to
        provide a starting guess for a stream whose component list differs from
        the feeds', which the automatic seed cannot infer.
        """
        self.tears[name] = guess
        return self

    # -- structure ---------------------------------------------------------- #
    def _edges(self) -> list[Edge]:
        producer: dict[str, int] = {}
        for i, u in enumerate(self.units):
            for out in u.outputs:
                if out in producer:
                    raise ValueError(
                        f"stream {out!r} is produced by both {self.units[producer[out]].name!r} "
                        f"and {u.name!r}"
                    )
                if out in self.feeds:
                    raise ValueError(f"stream {out!r} is both a feed and an output of {u.name!r}")
                producer[out] = i
        edges: list[Edge] = []
        for j, u in enumerate(self.units):
            for name in u.inputs:
                if name in producer:
                    edges.append((producer[name], j, name))
                elif name not in self.feeds:
                    raise ValueError(
                        f"unit {u.name!r} consumes stream {name!r}, which is neither a feed "
                        "nor produced by any unit"
                    )
        return edges

    def partition(self) -> list[Partition]:
        """Analyse the connectivity and return the calculation order.

        Units are grouped into strongly connected components (Tarjan), the
        components are ordered so every block's inputs are computed before it,
        and inside each cyclic block tear streams are selected greedily (honouring
        any hand-designated tears) and the units are ordered topologically with
        the tears treated as known.

        Raises:
            ValueError: if a stream is consumed but never produced, or produced
                twice.
        """
        edges = self._edges()
        n = len(self.units)
        comps = _strongly_connected(n, edges)
        comps.reverse()  # topological order of the condensation
        preferred = tuple(self.tears.keys())
        result: list[Partition] = []
        for comp in comps:
            nodes = sorted(comp)
            if not _cyclic(nodes, edges):
                result.append(Partition((self.units[nodes[0]].name,), ()))
                continue
            tears = _select_tears(nodes, edges, preferred)
            order = _topological(nodes, [e for e in edges if e[2] not in tears])
            result.append(Partition(tuple(self.units[i].name for i in order), tears))
        return result

    # -- evaluation --------------------------------------------------------- #
    def _run_units(
        self, names: Sequence[str], streams: dict[str, Stream], theta: Any
    ) -> dict[str, Stream]:
        by_name = {u.name: u for u in self.units}
        for name in names:
            u = by_name[name]
            args = [streams[s] for s in u.inputs]
            result = u.fn(*args, theta)
            produced = result if isinstance(result, list | tuple) else (result,)
            if len(produced) != len(u.outputs):
                raise ValueError(
                    f"unit {u.name!r} produced {len(produced)} outputs, expected {len(u.outputs)}"
                )
            for out_name, out_stream in zip(u.outputs, produced, strict=True):
                streams[out_name] = out_stream
        return streams

    def _seed_tears(
        self, block: Partition, streams: dict[str, Stream], theta: Any
    ) -> tuple[Stream, ...]:
        """Starting guesses for a block's tears: hand-given, else one empty-recycle pass."""
        guesses: dict[str, Stream] = {}
        missing = [t for t in block.tears if t not in self.tears]
        if missing:
            template = next(iter(streams.values()))
            trial = dict(streams)
            for t in missing:
                trial[t] = Stream(
                    n=jnp.zeros_like(template.n),
                    t=template.t,
                    p=template.p,
                    components=template.components,
                )
            for t in block.tears:
                if t in self.tears:
                    trial[t] = self.tears[t]
            trial = self._run_units(block.units, trial, theta)
            for t in missing:
                guesses[t] = trial[t]
        for t in block.tears:
            if t in self.tears:
                guesses[t] = self.tears[t]
        return tuple(jax.lax.stop_gradient(guesses[t]) for t in block.tears)

    def _evaluate(self, tears: dict[str, Stream], theta: Any) -> dict[str, Stream]:
        """Run every unit once in registration order (legacy single-pass evaluation)."""
        streams: dict[str, Stream] = {**self.feeds, **tears}
        return self._run_units([u.name for u in self.units], streams, theta)

    def solve(self, theta: Any = None, **tear_solve_kwargs: Any) -> dict[str, Stream]:
        """Solve the flowsheet (closing every recycle) and return all named streams.

        The blocks from `partition` are evaluated in order; each cyclic block is
        converged with `tear_solve` on its tear streams (keyword arguments such as
        ``method="broyden"`` are forwarded). ``theta`` is the differentiable
        parameter pytree passed to every unit; every output stream is
        differentiable with respect to it.
        """
        streams: dict[str, Stream] = dict(self.feeds)
        for block in self.partition():
            if not block.cyclic:
                self._run_units(block.units, streams, theta)
                continue
            guesses = self._seed_tears(block, streams, theta)
            # The upstream streams travel inside the parameter pytree (not as
            # closed-over values) so the implicit adjoint differentiates through
            # them and no tracer leaks into the custom-VJP solver.
            converged = tear_solve(
                self._block_map(block), guesses, (theta, dict(streams)), **tear_solve_kwargs
            )
            streams.update(zip(block.tears, converged, strict=True))
            self._run_units(block.units, streams, theta)
        return streams

    def _block_map(
        self, block: Partition
    ) -> Callable[[tuple[Stream, ...], tuple[Any, dict[str, Stream]]], tuple[Stream, ...]]:
        """The recycle map ``g(tears, (theta, upstream)) -> tears`` of one cyclic block."""

        def g(
            tear_tuple: tuple[Stream, ...], params: tuple[Any, dict[str, Stream]]
        ) -> tuple[Stream, ...]:
            th, base = params
            local = dict(base)
            local.update(zip(block.tears, tear_tuple, strict=True))
            self._run_units(block.units, local, th)
            return tuple(local[name] for name in block.tears)

        return g


__all__ = ["TEAR_METHODS", "Flowsheet", "Partition", "tear_solve"]
