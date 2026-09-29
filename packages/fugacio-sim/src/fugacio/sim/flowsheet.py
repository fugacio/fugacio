"""Process construction, execution, and checked implicit sensitivities.

A Flowsheet builds a shared ProcessGraph from physical unit kernels and named
streams. Sequential execution orders acyclic units and converges cyclic
partitions by tearing. Simultaneous execution solves the connection equations
with locally assembled sparse Newton steps. Both strategies differentiate the
same converged graph using local unit Jacobians and a sparse implicit solve.

Concrete process iterations run on the host, preserving compiled unit kernels.
An explicit enclosing JIT stages the orchestration. Unit reports, independent
stream checks, and optional physical audits remain acceptance requirements.
The standalone tear_solve functions also support arbitrary floating pytrees;
their small fixed-point derivative uses a dense reference system.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from fugacio.sim.graph import CompiledGraph, Partition, ProcessGraph, ProcessUnit
from fugacio.sim.numerics import is_traced, newton_iterations, while_loop
from fugacio.sim.stream import Stream
from fugacio.thermo._iteration import primal_call
from fugacio.thermo.diagnostics import (
    ConvergenceError,
    SolveReport,
    SolveResult,
    require_converged,
    residual_report,
)
from fugacio.thermo.implicit import gate_tree, implicit_solution
from fugacio.thermo.linear import dense_jacobian

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
    mapped0: Array | None = None,
) -> SolveResult:
    """Converge a recycle while retaining iterations and a residual check."""
    scale = jnp.maximum(jnp.abs(x0), 1.0)

    def residual(y: Array, th: Any) -> Array:
        x = y * scale
        return (g(x, th) - x) / scale

    if method == "newton":
        result = newton_iterations(
            residual,
            lambda x, p: dense_jacobian(residual, x, p, vectorize=False),
            x0 / scale,
            theta,
            tolerance=tol,
            max_iterations=max_iter,
            scale=jnp.ones_like(x0),
        )
        return SolveResult(result.value * scale, result.report)

    def error(x: Array, r: Array) -> Array:
        return jnp.max(jnp.abs(r) / (atol + jnp.maximum(jnp.abs(x), 1.0)))

    gx0 = g(x0, theta) if mapped0 is None else mapped0
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
        if method == "broyden":
            direction = x_new - x
            old_error = error(x, gx - x)

            def search_cond(state: tuple) -> Array:
                return ~state[3]

            def search_body(state: tuple) -> tuple:
                alpha, point, _, _ = state
                mapped = g(point, theta)
                score = error(point, mapped - point)
                retry = (alpha > 1 / 128) & (~jnp.isfinite(score) | (score > old_error))
                alpha = jnp.where(retry, alpha / 2, alpha)
                following = jnp.where(retry, x + alpha * direction, point)
                return alpha, following, mapped, ~retry

            # One call site for every trial, including alpha=1. Embedding g
            # before AND inside the search duplicates a full column/exchanger
            # graph in XLA. The map in the final state always belongs to x_new.
            _, x_new, g_new, _ = while_loop(
                search_cond, search_body, (jnp.asarray(1.0), x_new, gx, jnp.asarray(False))
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
        else:
            g_new = g(x_new, theta)
        step = error(x_new, x_new - x)
        return x_new, g_new, x, gx, inverse, i + 1, step, error(x_new, g_new - x_new)

    initial = (x0, gx0, x0, gx0, -jnp.eye(n), jnp.asarray(0), jnp.asarray(0.0), error(x0, gx0 - x0))
    x, gx, _, _, _, iterations, step, _ = while_loop(cond, body, initial)
    r = (gx - x) / (atol + jnp.maximum(jnp.abs(x), 1.0))
    return SolveResult(x, residual_report(r, tol, iterations=iterations, step_norm=step))


def _make_flat_map(g: Callable[..., Any], unravel: Callable[..., Any]) -> Callable[..., Any]:
    def flat_map(x: Array, theta: Any) -> Array:
        return ravel_pytree(g(unravel(x), theta))[0]

    return flat_map


_cached_flat_map = lru_cache(maxsize=32)(_make_flat_map)


def tear_solve_with_info(
    g: Callable[[Any, Any], Any],
    tear0: Any,
    theta: Any = None,
    *,
    method: str = "broyden",
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
        method: Broyden (default), Wegstein, or Newton.
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

    # The iteration kernel stages this map for the forward solve. Leave the
    # map itself unstaged so implicit linearization can call the individual
    # unit kernels without compiling a combined column/exchanger derivative.
    # Stable callback identities let the forward kernel reuse its executable.
    try:
        hash((g, unravel))
    except TypeError:
        # Arbitrary callable objects remain supported even if they can't be
        # cache keys. JAX's ordinary ravel inverse is hashable by tree/shape.
        g_flat = _make_flat_map(g, unravel)
    else:
        g_flat = _cached_flat_map(g, unravel)

    start, params = jax.lax.stop_gradient((flat0, theta))
    # A modular host solve can reuse compiled unit kernels for this initial
    # pass. Keep it out of the iteration executable, where it would duplicate
    # the complete recycle body. Under an enclosing JIT it remains traceable.
    mapped = None if method == "newton" else g_flat(start, params)
    raw = _tear_iterations(g_flat, start, params, method, q_min, q_max, tol, atol, max_iter, mapped)
    report = jax.lax.stop_gradient(raw.report)

    def residual(x: Array, th: Any) -> Array:
        return g_flat(x, th) - x

    value = implicit_solution(
        residual,
        jax.lax.stop_gradient(raw.value),
        theta,
        report.converged,
        "sequential",
    )
    return TearResult(unravel(value), report)


def tear_solve(
    g: Callable[[Any, Any], Any],
    tear0: Any,
    theta: Any = None,
    *,
    method: str = "broyden",
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


def _unit_outputs(result: Any) -> tuple[tuple[Stream, ...], UnitRecord]:
    """Streams a unit produced, and the energy/report evidence it returned."""
    zero = jnp.asarray(0.0)
    if hasattr(result, "outlets"):
        return tuple(result.outlets), UnitRecord(
            heat=jnp.asarray(getattr(result, "heat", zero)),
            work=jnp.asarray(getattr(result, "work", zero)),
            report=getattr(result, "report", residual_report(jnp.zeros(1))),
        )
    produced = tuple(result) if isinstance(result, list | tuple) else (result,)
    return produced, UnitRecord(heat=zero, work=zero, report=residual_report(jnp.zeros(1)))


@dataclass(frozen=True)
class UnitRecord:
    """Energy and solve evidence a unit returned alongside its outlets.

    Attributes:
        heat: Heat into the fluid (W); zero if the unit reports none.
        work: Shaft work into the fluid (W); zero if the unit reports none.
        report: The unit's own solve report (a converged placeholder for units
            that return plain streams).
    """

    heat: Array
    work: Array
    report: SolveReport


@dataclass(frozen=True)
class FlowsheetResult:
    """Named streams with per-block convergence, unit evidence, and physical audits.

    Attributes:
        streams: All feeds and computed streams.
        reports: Solve reports keyed by ``recycle:``, ``unit:``, or ``stream:`` name.
        units: Heat, work, and report retained from each unit's result.
        audits: Physical acceptance of each stream (empty without a model).
        unit_results: Complete unit pytrees when ``retain_results=True``.
    """

    streams: dict[str, Stream]
    reports: dict[str, SolveReport]
    units: dict[str, UnitRecord] = field(default_factory=dict)
    audits: dict[str, Any] = field(default_factory=dict)
    unit_results: dict[str, Any] = field(default_factory=dict)

    @property
    def converged(self) -> Array:
        """Whether every recycle, unit, and stream-structure check succeeded."""
        return jnp.all(jnp.asarray([r.converged for r in self.reports.values()]))

    @property
    def accepted(self) -> Array:
        """Converged, and every non-empty audited stream passed physical acceptance.

        Without audits (no flowsheet ``model``), this equals `converged`: no
        physical acceptance is claimed that wasn't checked.
        """
        ok = [
            (self.streams[name].total <= 0) | report.accepted
            for name, report in self.audits.items()
        ]
        return self.converged & jnp.all(jnp.asarray(ok if ok else [True]))

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
        """Raise with the responsible block's name if any calculation failed.

        Raises:
            ConvergenceError: For a failed recycle, unit, or stream structure.
            PhysicalAcceptanceError: For an audited stream that failed physical
                acceptance (for example, a vapor-liquid state that is actually
                liquid-liquid unstable).
        """
        from fugacio.thermo.acceptance import require_accepted

        for name, report in self.reports.items():
            require_converged(report, name)
        for name, audit in self.audits.items():
            if float(self.streams[name].total) > 0:
                require_accepted(audit, f"stream {name}")


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
    units: list[ProcessUnit] = field(default_factory=list)
    tears: dict[str, Stream] = field(default_factory=dict)
    model: Any = None
    _maps: dict[Partition, Any] = field(default_factory=dict, repr=False, compare=False)
    _graphs: dict[Any, CompiledGraph] = field(default_factory=dict, repr=False, compare=False)

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
        self.units.append(ProcessUnit(name, fn, tuple(inputs), tuple(outputs)))
        self._maps.clear()
        self._graphs.clear()
        return self

    def tear(self, name: str, guess: Stream) -> Flowsheet:
        """Designate stream ``name`` as a recycle tear with an initial ``guess``.

        Hand-designated tears are honoured by `partition` (and supplemented if
        they do not break every loop). Designating a tear is also the way to
        provide a starting guess for a stream whose component list differs from
        the feeds', which the automatic seed cannot infer.
        """
        self.tears[name] = guess
        self._maps.clear()
        self._graphs.clear()
        return self

    # -- structure ---------------------------------------------------------- #
    @property
    def graph(self) -> ProcessGraph:
        """The canonical graph used by every execution strategy."""
        return ProcessGraph(tuple(self.feeds), tuple(self.units), tuple(self.tears))

    def partition(self) -> list[Partition]:
        """Validate connectivity and choose recycle partitions in the shared graph."""
        return self.graph.partition()

    # -- evaluation --------------------------------------------------------- #
    def _run_units(
        self,
        names: Sequence[str],
        streams: dict[str, Stream],
        theta: Any,
        records: dict[str, UnitRecord] | None = None,
    ) -> dict[str, Stream]:
        by_name = {u.name: u for u in self.units}
        for name in names:
            u = by_name[name]
            args = [streams[s] for s in u.inputs]
            try:
                result = u.fn(*args, theta)
            except ConvergenceError as exc:
                raise ConvergenceError(exc.report, f"unit {u.name}: {exc.context}") from exc
            produced, record = _unit_outputs(result)
            if len(produced) != len(u.outputs):
                raise ValueError(
                    f"unit {u.name!r} produced {len(produced)} outputs, expected {len(u.outputs)}"
                )
            for out_name, out_stream in zip(u.outputs, produced, strict=True):
                streams[out_name] = out_stream
            if records is not None:
                records[u.name] = record
        return streams

    def _seed_tears(
        self, block: Partition, streams: dict[str, Stream], theta: Any
    ) -> tuple[Stream, ...]:
        """Starting guesses for a block's tears: hand-given, else one empty-recycle pass."""
        guesses: dict[str, Stream] = {}
        missing = [t for t in block.tears if t not in self.tears]
        if missing:
            if not streams:
                raise ValueError(
                    f"recycle block {', '.join(block.units)} has no upstream stream to seed "
                    f"its tear(s) {', '.join(missing)}; supply a guess with Flowsheet.tear"
                )
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

    def _solve_primal(
        self,
        parameters: Any,
        guess: Mapping[str, Stream] | None,
        strategy: str,
        linear_solver: str,
        tear_solve_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Stream], dict[str, SolveReport], dict[str, UnitRecord]]:
        """Converge numerical states independently of the surrounding AD trace."""
        raw_theta, raw_feeds = jax.lax.stop_gradient(parameters)
        streams: dict[str, Stream] = dict(raw_feeds)
        reports: dict[str, SolveReport] = {}
        records: dict[str, UnitRecord] = {}
        for block in self.partition():
            if not block.cyclic:
                self._run_units(block.units, streams, raw_theta, records)
            else:
                if guess is not None and all(name in guess for name in block.tears):
                    guesses = tuple(jax.lax.stop_gradient(guess[name]) for name in block.tears)
                else:
                    guesses = self._seed_tears(block, streams, raw_theta)
                if guess is not None:
                    guesses = tuple(
                        jax.lax.stop_gradient(guess.get(name, seed))
                        for name, seed in zip(block.tears, guesses, strict=True)
                    )
                if strategy == "simultaneous":
                    streams.update(zip(block.tears, guesses, strict=True))
                    self._run_units(block.units, streams, raw_theta, records)
                    continue
                converged = tear_solve_with_info(
                    self._block_map(block), guesses, (raw_theta, dict(streams)), **tear_solve_kwargs
                )
                reports["recycle:" + ",".join(block.units)] = converged.report
                streams.update(zip(block.tears, converged.value, strict=True))
                self._run_units(block.units, streams, raw_theta, records)
        if self.units:
            graph = self.compile(streams, linear_solver=linear_solver)
            value = jax.lax.stop_gradient(graph.pack(streams))
            if strategy == "simultaneous":
                if guess is not None:
                    value = jax.lax.stop_gradient(graph.pack({**streams, **guess}))
                solved = newton_iterations(
                    graph.residual,
                    lambda x, p: graph.linearize(x, p)[0],
                    value,
                    (raw_theta, raw_feeds),
                    tolerance=tear_solve_kwargs.get("tol", 1e-9),
                    max_iterations=tear_solve_kwargs.get("max_iter", 100),
                )
                value = solved.value
                reports["process"] = solved.report
                streams = graph.unpack(value, raw_feeds)
        return streams, reports, records

    def solve_with_info(
        self,
        theta: Any = None,
        *,
        guess: Mapping[str, Stream] | None = None,
        audit: bool = True,
        strategy: str = "sequential",
        linear_solver: str = "sparse",
        retain_results: bool = False,
        **tear_solve_kwargs: Any,
    ) -> FlowsheetResult:
        """Solve every block and retain reports, unit evidence, and stream audits.

        Failed blocks retain their best iterates for diagnosis. Use ``result.check()``
        at an application boundary, or inspect ``result.converged`` and
        ``result.accepted`` inside JIT. With a flowsheet ``model`` and ``audit``,
        every stream is audited for physical acceptance after convergence.
        ``strategy`` selects ``sequential`` or ``simultaneous`` on the same graph.
        ``linear_solver="dense"`` explicitly selects the small reference solver;
        the default sparse CPU solver never falls back to a dense plant matrix.
        """
        if strategy not in ("sequential", "simultaneous"):
            raise ValueError("strategy must be sequential or simultaneous")
        parameters = (theta, self.feeds)

        def forward(parameters: Any, guesses: Any) -> Any:
            return self._solve_primal(
                parameters, guesses, strategy, linear_solver, tear_solve_kwargs
            )

        streams, reports, records = primal_call(forward, (parameters, guess))
        unit_results: dict[str, Any] = {}
        if self.units:
            graph = self.compile(streams, linear_solver=linear_solver)
            value = graph.pack(streams)
            valid = jnp.all(jnp.asarray([r.converged for r in reports.values()]))
            if strategy == "sequential":
                valid &= jnp.all(jnp.asarray([r.report.converged for r in records.values()]))
            valid &= jnp.all(jnp.asarray([s.report.converged for s in streams.values()]))
            attached = graph.attach(value, parameters, valid)
            streams = graph.unpack(attached, self.feeds)
            # Retained duties need the same implicit stream derivatives. Replay
            # each unit independently, never overwriting another unit's inputs.
            if is_traced(parameters) or strategy == "simultaneous" or retain_results:
                for unit in self.units:
                    result = unit.fn(*(streams[k] for k in unit.inputs), theta)
                    _, record = _unit_outputs(result)
                    records[unit.name] = record
                    if retain_results:
                        unit_results[unit.name] = result
        else:
            # A feed-only graph has no implicit state to attach. Its supplied
            # streams still carry their ordinary input derivatives.
            streams = dict(self.feeds)
        for name, record in records.items():
            reports["unit:" + name] = record.report
        for name, stream in streams.items():
            reports["stream:" + name] = stream.report
        valid = jnp.all(jnp.asarray([r.converged for r in reports.values()]))
        streams, records, unit_results = gate_tree((streams, records, unit_results), valid)
        audits: dict[str, Any] = {}
        if audit and self.model is not None:
            from fugacio.sim.acceptance import audit_stream

            audits = {name: audit_stream(s, self.model) for name, s in streams.items()}
        return FlowsheetResult(streams, reports, records, audits, unit_results)

    def compile(
        self, streams: Mapping[str, Stream], *, linear_solver: str = "sparse"
    ) -> CompiledGraph:
        """Reuse structural layouts without retaining a numerical solution."""
        graph = self.graph
        key = (
            graph.feeds,
            tuple((u.name, id(u.fn), u.inputs, u.outputs) for u in graph.units),
            tuple((name, streams[name].components) for u in graph.units for name in u.outputs),
            linear_solver,
        )
        if key not in self._graphs:
            self._graphs[key] = graph.compile(streams, linear_solver=linear_solver)
        return self._graphs[key]

    def solve(
        self, theta: Any = None, *, check: bool = True, **tear_solve_kwargs: Any
    ) -> dict[str, Stream]:
        """Return named streams after checked recycle convergence and acceptance.

        Eager failures identify the responsible block, unit, or stream. Compiled
        failures produce nonfinite streams; :meth:`solve_with_info` retains the
        reports and best iterates for inspection. Set ``check=False`` to return
        best iterates.

        Raises:
            ConvergenceError: If an eager block, unit, or stream check fails.
            PhysicalAcceptanceError: If an audited stream fails acceptance.
        """
        result = self.solve_with_info(theta, **tear_solve_kwargs)
        if not check:
            return result.streams
        result.check()
        return jax.tree_util.tree_map(
            lambda x: jnp.where(result.accepted, x, jnp.nan), result.streams
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
        """The recycle map ``g(tears, (theta, upstream)) -> tears`` of one cyclic block.

        Maps are cached per partition so explicit enclosing transformations
        reuse function identities. Registering a unit or tear clears the cache.
        """
        if block in self._maps:
            return self._maps[block]

        def g(
            tear_tuple: tuple[Stream, ...], params: tuple[Any, dict[str, Stream]]
        ) -> tuple[Stream, ...]:
            th, base = params
            local = dict(base)
            local.update(zip(block.tears, tear_tuple, strict=True))
            self._run_units(block.units, local, th)
            return tuple(local[name] for name in block.tears)

        self._maps[block] = g
        return g


jax.tree_util.register_dataclass(UnitRecord, data_fields=["heat", "work", "report"], meta_fields=[])
jax.tree_util.register_dataclass(
    FlowsheetResult,
    data_fields=["streams", "reports", "units", "audits", "unit_results"],
    meta_fields=[],
)

__all__ = [
    "TEAR_METHODS",
    "Flowsheet",
    "FlowsheetResult",
    "Partition",
    "TearResult",
    "UnitRecord",
    "tear_solve",
    "tear_solve_with_info",
]
