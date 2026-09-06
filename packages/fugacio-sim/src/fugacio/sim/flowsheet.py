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
solution by the implicit function theorem (a ``custom_jvp`` whose transpose
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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from fugacio.sim.stream import Stream
from fugacio.thermo.diagnostics import (
    ConvergenceError,
    SolveReport,
    SolveResult,
    require_converged,
    residual_report,
)
from fugacio.thermo.implicit import implicit_solution, newton_system_with_info

TEAR_METHODS: tuple[str, ...] = ("wegstein", "broyden", "newton")
"""Recycle convergence methods accepted by `tear_solve` and `Flowsheet.solve`."""


class TearResult(NamedTuple):
    """Recycle state and the independently checked fixed-point residual."""

    value: Any
    report: SolveReport


def _tear_iterations(
    g: Callable[[Array, Any], Array],
    x0: Array,
    theta: Any,
    method: str,
    q_min: float,
    q_max: float,
    tol: float,
    atol: float,
    max_iter: int,
) -> SolveResult:
    """Converge a recycle while retaining iterations and a residual check."""
    scale = jnp.maximum(jnp.abs(x0), 1.0)

    def residual(y: Array, th: Any) -> Array:
        x = y * scale
        return (g(x, th) - x) / scale

    if method == "newton":
        result = newton_system_with_info(residual, x0 / scale, theta, tol, max_iter)
        return SolveResult(result.value * scale, result.report)

    def error(x: Array, r: Array) -> Array:
        return jnp.max(jnp.abs(r) / (atol + jnp.maximum(jnp.abs(x), 1.0)))

    gx0 = g(x0, theta)
    n = x0.size

    def cond(carry: tuple) -> Array:
        x, gx, _, _, _, i, _, _ = carry
        return (error(x, gx - x) > tol) & jnp.all(jnp.isfinite(gx)) & (i < max_iter)

    def body(carry: tuple) -> tuple:
        x, gx, prev_x, prev_g, inverse, i, _, _ = carry
        if method == "wegstein":
            dx = x - prev_x
            slope = (gx - prev_g) / jnp.where(jnp.abs(dx) > 1e-13, dx, 1.0)
            slope = jnp.where((i > 0) & (jnp.abs(dx) > 1e-13), slope, 0.0)
            denominator = slope - 1.0
            q = slope / jnp.where(jnp.abs(denominator) > 1e-13, denominator, 1.0)
            q = jnp.where(jnp.abs(denominator) > 1e-13, q, 0.0)
            q = jnp.clip(q, q_min, q_max)
            x_new = q * x + (1.0 - q) * gx
        else:
            r = (gx - x) / scale
            ds = -inverse @ r
            x_new = x + ds * scale
        g_new = g(x_new, theta)
        if method == "broyden":
            direction = x_new - x
            old_error = error(x, gx - x)

            def search_cond(state: tuple) -> Array:
                alpha, point, mapped = state
                score = error(point, mapped - point)
                return (alpha > 1 / 128) & (~jnp.isfinite(score) | (score > old_error))

            def search_body(state: tuple) -> tuple:
                alpha, _, _ = state
                alpha = alpha / 2
                point = x + alpha * direction
                return alpha, point, g(point, theta)

            _, x_new, g_new = jax.lax.while_loop(
                search_cond, search_body, (jnp.asarray(1.0), x_new, g_new)
            )
            ds = (x_new - x) / scale
            df = ((g_new - x_new) - (gx - x)) / scale
            bdf = inverse @ df
            denominator = ds @ bdf
            update = jnp.outer(ds - bdf, ds @ inverse) / jnp.where(
                jnp.abs(denominator) > 1e-30, denominator, 1.0
            )
            inverse = jnp.where(
                (jnp.abs(denominator) > 1e-14) & jnp.all(jnp.isfinite(update)),
                inverse + update,
                -jnp.eye(n),
            )
        step = error(x_new, x_new - x)
        return x_new, g_new, x, gx, inverse, i + 1, step, error(x_new, g_new - x_new)

    initial = (x0, gx0, x0, gx0, -jnp.eye(n), jnp.asarray(0), jnp.asarray(0.0), error(x0, gx0 - x0))
    x, gx, _, _, _, iterations, step, _ = jax.lax.while_loop(cond, body, initial)
    r = (gx - x) / (atol + jnp.maximum(jnp.abs(x), 1.0))
    return SolveResult(x, residual_report(r, tol, iterations=iterations, step_norm=step))


def tear_solve_with_info(
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
) -> TearResult:
    """Solve a recycle, reporting its actual fixed-point residual.

    ``g(tear, theta)`` and ``tear0`` must have identical pytree structures.
    Both JVPs and VJPs solve the linearized fixed-point equations independently
    of the forward acceleration. An unconverged iterate has nonfinite
    sensitivities. ``q_max`` may be raised toward one to damp oscillatory maps.

    Args:
        g: One flowsheet pass.
        tear0: Starting recycle state, as any floating-point pytree.
        theta: Differentiable operating parameters and upstream streams.
        method: Wegstein, Broyden, or Newton.
        q_min: Minimum Wegstein acceleration factor.
        q_max: Maximum Wegstein acceleration factor.
        tol: Scaled fixed-point residual tolerance.
        atol: Absolute floor added to the per-variable scaling.
        max_iter: Forward iteration cap.

    Returns:
        The best recycle state and its solve report.

    Raises:
        ValueError: For an unknown method or invalid numerical options.
    """
    if method not in TEAR_METHODS:
        raise ValueError(f"unknown tear method {method!r}; choose from {TEAR_METHODS}")
    if tol <= 0 or atol <= 0 or max_iter < 0 or q_min > q_max:
        raise ValueError("invalid tear solver tolerances, iteration cap, or acceleration bounds")
    flat0, unravel = ravel_pytree(tear0)

    def g_flat(x: Array, th: Any) -> Array:
        out = g(unravel(x), th)
        return ravel_pytree(out)[0]

    start, params = jax.lax.stop_gradient((flat0, theta))
    raw = _tear_iterations(g_flat, start, params, method, q_min, q_max, tol, atol, max_iter)
    report = jax.lax.stop_gradient(raw.report)
    value = implicit_solution(
        lambda x, th: g_flat(x, th) - x,
        jax.lax.stop_gradient(raw.value),
        theta,
        report.converged,
    )
    return TearResult(unravel(value), report)


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
    check: bool = True,
) -> Any:
    """Return a converged recycle, with implicit forward and reverse derivatives.

    Eager calls raise on failure by default. Under JAX transforms the failed
    value is nonfinite; use :func:`tear_solve_with_info` to retain diagnostics
    and the best iterate. ``check=False`` also returns that best iterate.

    Raises:
        ConvergenceError: If an eager calculation does not converge.
        ValueError: If numerical options are invalid.
    """
    result = tear_solve_with_info(
        g,
        tear0,
        theta,
        method=method,
        q_min=q_min,
        q_max=q_max,
        tol=tol,
        atol=atol,
        max_iter=max_iter,
    )
    if check:
        require_converged(result.report, "recycle")
        return jax.tree_util.tree_map(
            lambda x: jnp.where(result.report.converged, x, jnp.nan), result.value
        )
    return result.value


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


@dataclass(frozen=True)
class FlowsheetResult:
    """Named streams with per-block convergence and physical-state checks.

    Attributes:
        streams: All feeds and computed streams.
        reports: Solve reports keyed by unit or recycle-block name.
    """

    streams: dict[str, Stream]
    reports: dict[str, SolveReport]

    @property
    def converged(self) -> Array:
        """Whether every block and physical-state check succeeded."""
        return jnp.all(jnp.asarray([r.converged for r in self.reports.values()]))

    @property
    def report(self) -> SolveReport:
        """The first failed block report, or the largest converged residual."""
        reports = tuple(self.reports.values())
        if not reports:
            return residual_report(jnp.zeros(0))
        scores = jnp.array([jnp.where(r.converged, r.residual_norm, jnp.inf) for r in reports])
        index = jnp.argmax(scores)
        return jax.tree_util.tree_map(lambda *values: jnp.stack(values)[index], *reports)

    def __getitem__(self, name: str) -> Stream:
        return self.streams[name]

    def check(self) -> None:
        """Raise with the responsible block's name if any calculation failed."""
        for name, report in self.reports.items():
            require_converged(report, name)


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
        if name in self.feeds:
            raise ValueError(f"duplicate feed name {name!r}")
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
        if any(u.name == name for u in self.units):
            raise ValueError(f"duplicate unit name {name!r}")
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
            try:
                result = u.fn(*args, theta)
            except ConvergenceError as exc:
                raise ConvergenceError(exc.report, f"unit {u.name}: {exc.context}") from exc
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

    def solve_with_info(
        self,
        theta: Any = None,
        *,
        guess: Mapping[str, Stream] | None = None,
        **tear_solve_kwargs: Any,
    ) -> FlowsheetResult:
        """Solve every block and retain convergence reports alongside streams.

        Failed blocks retain their best iterates for diagnosis. Use ``result.check()``
        at an application boundary, or inspect ``result.converged`` inside JIT.
        """
        streams: dict[str, Stream] = dict(self.feeds)
        reports: dict[str, SolveReport] = {}
        for block in self.partition():
            if not block.cyclic:
                self._run_units(block.units, streams, theta)
            else:
                if guess is not None and all(name in guess for name in block.tears):
                    guesses = tuple(jax.lax.stop_gradient(guess[name]) for name in block.tears)
                else:
                    guesses = self._seed_tears(block, streams, theta)
                if guess is not None:
                    guesses = tuple(
                        jax.lax.stop_gradient(guess.get(name, seed))
                        for name, seed in zip(block.tears, guesses, strict=True)
                    )
                converged = tear_solve_with_info(
                    self._block_map(block), guesses, (theta, dict(streams)), **tear_solve_kwargs
                )
                reports["recycle:" + ",".join(block.units)] = converged.report
                streams.update(zip(block.tears, converged.value, strict=True))
                self._run_units(block.units, streams, theta)
        for name, stream in streams.items():
            reports["stream:" + name] = stream.report
        return FlowsheetResult(streams, reports)

    def solve(
        self, theta: Any = None, *, check: bool = True, **tear_solve_kwargs: Any
    ) -> dict[str, Stream]:
        """Return named streams after checked recycle convergence.

        Eager failures identify the responsible block. Compiled failures produce
        nonfinite streams; :meth:`solve_with_info` retains the reports and best
        iterates for inspection. Set ``check=False`` to return best iterates.

        Raises:
            ConvergenceError: If an eager block or stream check fails.
        """
        result = self.solve_with_info(theta, **tear_solve_kwargs)
        if not check:
            return result.streams
        result.check()
        return jax.tree_util.tree_map(
            lambda x: jnp.where(result.converged, x, jnp.nan), result.streams
        )

    def solve_path(
        self,
        start: Any,
        target: Any,
        *,
        solve_options: Mapping[str, Any] | None = None,
        **options: Any,
    ) -> Any:
        """Continue between operating conditions with accepted recycle warm starts.

        ``options`` are forwarded to :func:`continuation_solve`. Configure the
        recycle method and tolerances through ``solve_options``.
        """
        from fugacio.sim.continuation import continuation_solve

        def solve(params: Any, previous: Any) -> tuple[FlowsheetResult, SolveReport]:
            settings = dict(solve_options or {})
            if previous is not None:
                settings["guess"] = previous.streams
            result = self.solve_with_info(params, **settings)
            return result, result.report

        return continuation_solve(solve, start, target, **options)

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


jax.tree_util.register_dataclass(
    FlowsheetResult, data_fields=["streams", "reports"], meta_fields=[]
)

__all__ = [
    "TEAR_METHODS",
    "Flowsheet",
    "FlowsheetResult",
    "Partition",
    "TearResult",
    "tear_solve",
    "tear_solve_with_info",
]
