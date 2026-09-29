"""Simultaneous execution of physical units and custom residual blocks.

Topology uses ProcessGraph validation. ResidualGraph assembles independently
compiled blocks into sparse Newton and implicit derivative systems. Built-in
blocks use the common physical unit kernels and stream coordinates; custom
blocks can introduce residual equations and auxiliary unknowns. Design
specifications enter as additional equations and manipulated variables.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from fugacio.sim.eo.blocks import Block, Context, Scales
from fugacio.sim.graph import ProcessGraph, ProcessUnit, ResidualGraph, stream_vector, vector_stream
from fugacio.sim.numerics import newton_iterations
from fugacio.sim.properties import Model, _resolve, resolve_package
from fugacio.sim.stream import Stream
from fugacio.thermo.diagnostics import SolveReport, SolveStatus, require_converged, with_status
from fugacio.thermo.implicit import gate_derivative
from fugacio.thermo.sparsity import SparsityPattern

#: A measurement read off the solved streams (for a design spec / objective).
Measure = Callable[[Mapping[str, Stream]], Array]


def _is_traced(*trees: Any) -> bool:
    """Whether numerical values are being traced by an enclosing transformation."""
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(trees))


class DOFReport(NamedTuple):
    """Degrees-of-freedom analysis of an EO flowsheet.

    Attributes:
        n_unknowns: Total scalar unknowns (internal stream variables + block
            auxiliaries + freed design-spec variables).
        n_equations: Total scalar equations (block residuals + design specs).
        degrees_of_freedom: ``n_unknowns - n_equations``. Zero means the
            equation count is balanced; rank and convergence still need checking; positive means
            under-specified (add specs); negative means over-specified.
        per_block: Equation count contributed by each block (keyed by the block's
            first outlet name).
    """

    n_unknowns: int
    n_equations: int
    degrees_of_freedom: int
    per_block: dict[str, int]


class _Plan(NamedTuple):
    """Cached structural assembly and local kernels for one process layout."""

    ctx: Context
    internal: tuple[str, ...]
    aux_scales: dict[str, float]
    unravel: Callable[[Array], Any]
    core: Callable[..., tuple[Array, SolveReport]]
    newton: Callable[..., tuple[Array, SolveReport]]
    n_unknowns: int
    n_equations: int
    #: One-slot warm-start cache: the most recent *concrete* converged unknown
    #: vector. Reused as the (detached) seed under autodiff so no sequential-modular
    #: sweeps run inside the grad trace (they are slow and recompile every call).
    seed: list[Array]


@dataclass(frozen=True)
class _DesignSpec:
    """A design spec freeing ``manipulated`` to drive ``measure`` to ``target``."""

    manipulated: str
    measure: Measure
    target: float | Array
    init: float | Array

    def scale(self) -> float:
        """Scale for the freed variable (its initial magnitude, floored)."""
        return float(max(abs(float(self.init)), 1.0))

    def residual_scale(self) -> float:
        """Scale for the spec residual (the target magnitude, floored)."""
        return float(max(abs(float(self.target)), 1.0))


@dataclass(frozen=True)
class EOSolution:
    """Converged equation-oriented flowsheet solution.

    Attributes:
        streams: All named streams (feeds plus solved internal streams).
        aux: Solved auxiliary variables declared by custom blocks.
        specs: Solved values of any freed design-spec manipulated variables.
        residual_norm: Max-norm of the (scaled) residual at the solution.
        n_unknowns: Number of scalar unknowns solved.
        n_equations: Number of scalar equations.
    """

    streams: dict[str, Stream]
    aux: dict[str, Array]
    specs: dict[str, Array]
    residual_norm: Array
    n_unknowns: int
    n_equations: int
    report: SolveReport

    @property
    def converged(self) -> Array:
        """Whether the full scaled equation system converged."""
        return self.report.converged

    def check(self) -> None:
        """Raise for a concrete failed simultaneous solve."""
        require_converged(self.report, "equation-oriented flowsheet")

    def __getitem__(self, name: str) -> Stream:
        """Return the solved stream ``name``."""
        return self.streams[name]


def _stream_scales(components: tuple[str, ...], sc: Scales) -> Array:
    c = len(components)
    return jnp.asarray([*[sc.flow] * c, sc.temperature, sc.pressure, *[sc.flow] * c])


def _pack_stream(s: Stream, sc: Scales, pkg: Any = None) -> Array:
    """Pack the common process stream coordinates using declared scales."""
    return stream_vector(s) / _stream_scales(s.components, sc)


def _unpack_stream(vec: Array, components: tuple[str, ...], sc: Scales, pkg: Any = None) -> Stream:
    """Restore material, temperature, pressure, and resolved vapor inventory."""
    return vector_stream(vec * _stream_scales(components, sc), components)


@dataclass
class EOFlowsheet:
    """Declarative equation-oriented flowsheet.

    Register feeds and blocks, optionally add design specs, then call `solve`.
    Streams are referenced by name; a block's outlet names become the system
    unknowns and recycles need no special handling (just reuse a downstream
    stream name as an upstream block's inlet).

    Example::

        fs = EOFlowsheet()
        fs.feed("fresh", fresh_stream)
        fs.add(Mixer(inlets=("fresh", "recycle"), outlets=("mixed",)))
        fs.add(Flash(inlets=("mixed",), outlets=("vapor", "liquid"), t="T", p="P"))
        fs.add(Splitter(inlets=("liquid",), outlets=("recycle", "purge"),
                        fractions=("r_recycle",)))
        sol = fs.solve({"T": 320.0, "P": 20e5, "r_recycle": [0.5, 0.5]})
        product = sol["vapor"]

    Attributes:
        scales: Residual/variable scales (auto-derived from the feeds by
            `solve` when left at the default).
        model: Property package (see `fugacio.sim.models.package_for`) used by
            every block; ``None`` selects Peng-Robinson over the feeds'
            components. Any method class (cubic, gamma-phi, PC-SAFT, reference
            fluid) can drive the whole simultaneous solve.
    """

    scales: Scales | None = None
    model: Model = None
    feeds: dict[str, Stream] = field(default_factory=dict)
    blocks: list[Block] = field(default_factory=list)
    specs: list[_DesignSpec] = field(default_factory=list)
    linear_solver: str = "sparse"
    _plans: dict[Any, _Plan] = field(default_factory=dict, init=False, repr=False, compare=False)

    # -- construction ------------------------------------------------------ #
    def feed(self, name: str, stream: Stream) -> EOFlowsheet:
        """Register a fresh feed stream by name. Returns ``self`` for chaining."""
        self.feeds[name] = stream
        return self

    def add(self, block: Block) -> EOFlowsheet:
        """Register a unit block. Returns ``self`` for chaining."""
        self.blocks.append(block)
        return self

    def spec(
        self,
        manipulated: str,
        measure: Measure,
        target: float | Array,
        *,
        init: float | Array,
    ) -> EOFlowsheet:
        """Add a design spec: free parameter ``manipulated`` to hit ``measure = target``.

        In EO form a design spec is simply one more equation (``measure(streams) -
        target = 0``) and one more unknown (the freed value of ``manipulated``,
        seeded at ``init``), solved simultaneously with the flowsheet, so coupled
        specs converge together with no nested loop.

        Args:
            manipulated: A parameter key read by some block (the degree of
                freedom). Its value becomes an unknown; the value passed in
                ``params`` for this key, if any, is ignored.
            measure: Reads the controlled variable from the solved streams.
            target: Desired value of the controlled variable.
            init: Initial guess for the manipulated variable.

        Returns:
            ``self`` for chaining.
        """
        if any(spec.manipulated == manipulated for spec in self.specs):
            raise ValueError(f"design variable {manipulated!r} is already freed by a specification")
        self.specs.append(_DesignSpec(manipulated, measure, target, init))
        return self

    # -- topology ---------------------------------------------------------- #
    def _components(self) -> tuple[str, ...]:
        """The shared component list, validated identical across all feeds."""
        if not self.feeds:
            raise ValueError("EOFlowsheet has no feeds; register at least one with .feed(...)")
        comps = next(iter(self.feeds.values())).components
        for name, s in self.feeds.items():
            if s.components != comps:
                raise ValueError(
                    f"feed {name!r} has components {s.components}, expected {comps}; "
                    "every stream in an EO flowsheet must share one component list"
                )
        return comps

    def _internal_names(self) -> list[str]:
        """Ordered (sorted) names of internal streams, validated for consistency.

        Raises:
            ValueError: if a stream is produced by more than one block, a block
                outlet shadows a feed, or a block inlet is neither a feed nor any
                block's outlet.
        """
        graph = ProcessGraph(
            tuple(self.feeds),
            tuple(
                ProcessUnit(str(i), block.forward, block.inlets, block.outlets)
                for i, block in enumerate(self.blocks)
            ),
        )
        graph.edges()
        return sorted(name for unit in graph.units for name in unit.outputs)

    def _aux_scales(self, ctx: Context) -> dict[str, float]:
        """Collect every block's auxiliary unknowns into one key -> scale map."""
        out: dict[str, float] = {}
        for blk in self.blocks:
            declared = blk.aux_scales(ctx)
            duplicate = out.keys() & declared.keys()
            if duplicate:
                raise ValueError(f"auxiliary variables have multiple owners: {sorted(duplicate)}")
            out.update(declared)
        return out

    def _incidence(self, ctx: Context) -> tuple[SparsityPattern, tuple[str, ...]]:
        """Declare scalar dependencies in the same a/d/s order as ravel_pytree."""
        internal = self._internal_names()
        aux = sorted(self._aux_scales(ctx))
        freed = sorted(spec.manipulated for spec in self.specs)
        if len(set(freed)) != len(freed):
            raise ValueError("a design variable cannot be freed by multiple specifications")
        variables = [*("aux:" + key for key in aux), *("parameter:" + key for key in freed)]
        auxiliaries = {key: i for i, key in enumerate(aux)}
        parameters = tuple(range(len(aux), len(aux) + len(freed)))
        streams: dict[str, tuple[int, ...]] = {}
        for name in internal:
            streams[name] = tuple(range(len(variables), len(variables) + 2 * ctx.n_components + 2))
            variables.extend(f"{name}:n[{component}]" for component in ctx.components)
            variables.extend(
                (
                    name + ":temperature",
                    name + ":pressure",
                )
            )
            variables.extend(f"{name}:vapor_n[{component}]" for component in ctx.components)
        rows = []
        for block in self.blocks:
            declaration = block.residual_dependencies(ctx)
            if declaration is None:
                columns = tuple(range(len(variables)))
            else:
                ports, own_aux = declaration
                if (
                    set(ports) - (streams.keys() | self.feeds.keys())
                    or set(own_aux) - auxiliaries.keys()
                ):
                    raise ValueError(
                        f"invalid residual dependency declaration for {block.outlets[0]!r}"
                    )
                columns = tuple(
                    sorted(
                        {
                            *parameters,
                            *(auxiliaries[key] for key in own_aux),
                            *(i for name in ports for i in streams.get(name, ())),
                        }
                    )
                )
            rows.extend([columns] * block.n_residuals(ctx))
        rows.extend([tuple(range(len(variables)))] * len(self.specs))
        return SparsityPattern(len(variables), tuple(rows)), tuple(variables)

    def diagnose_structure(self) -> dict[str, Any]:
        """Inspect declared incidence without compiling or solving the flowsheet.

        This identifies disconnected equation/variable counts and reports the
        coloring cost. Matching is an upper bound on rank. Use ``diagnose``
        afterward for state-dependent rank and conditioning.
        """
        ctx = self._context()
        pattern, variables = self._incidence(ctx)
        return {
            **pattern.diagnose(equations=self.equation_labels(ctx), variables=variables),
            "linear_solver": self.linear_solver,
            "stored_coefficients": sum(map(len, pattern.rows)),
            "blocks": [
                {
                    "name": block.outlets[0],
                    "equations": block.n_residuals(ctx),
                    "dependencies": "all_unknowns"
                    if block.residual_dependencies(ctx) is None
                    else "declared",
                }
                for block in self.blocks
            ],
        }

    def _context(self) -> Context:
        """Build the static solve context (components, EOS, scales)."""
        comps = self._components()
        scales = self.scales if self.scales is not None else _auto_scales(self.feeds)
        return Context(components=comps, scales=scales, model=self.model)

    def degrees_of_freedom(self) -> DOFReport:
        """Report the unknown/equation balance for the flowsheet (see `DOFReport`)."""
        ctx = self._context()
        internal = self._internal_names()
        per_block = {blk.outlets[0]: blk.n_residuals(ctx) for blk in self.blocks}
        n_aux = len(self._aux_scales(ctx))
        n_specs = len(self.specs)
        n_unknowns = len(internal) * (2 * ctx.n_components + 2) + n_aux + n_specs
        n_equations = sum(per_block.values()) + n_specs
        return DOFReport(
            n_unknowns=n_unknowns,
            n_equations=n_equations,
            degrees_of_freedom=n_unknowns - n_equations,
            per_block=per_block,
        )

    def equation_labels(self, ctx: Context | None = None) -> tuple[str, ...]:
        """Stable block and equation names in residual assembly order."""
        ctx = self._context() if ctx is None else ctx
        labels = [
            f"{block.outlets[0]}:equation[{i}]"
            for block in self.blocks
            for i in range(block.n_residuals(ctx))
        ]
        labels.extend(f"spec:{spec.manipulated}" for spec in self.specs)
        return tuple(labels)

    # -- initialisation ---------------------------------------------------- #
    def _seed(
        self,
        ctx: Context,
        internal: Sequence[str],
        feeds: Mapping[str, Stream],
        guess: Mapping[str, Stream] | None,
    ) -> dict[str, Stream]:
        """Default interior guess for every internal stream (overridable per stream)."""
        n_total = jnp.sum(jnp.stack([s.n for s in feeds.values()]), axis=0)
        t_avg = jnp.mean(jnp.stack([s.t for s in feeds.values()]))
        p_min = jnp.min(jnp.stack([s.p for s in feeds.values()]))
        default = Stream(n=0.5 * n_total, t=t_avg, p=p_min, components=ctx.components)
        seeded = {name: default for name in internal}
        if guess:
            seeded.update({k: v for k, v in guess.items() if k in seeded})
        return seeded

    def _initial_streams(
        self,
        ctx: Context,
        internal: Sequence[str],
        params: Mapping[str, Any],
        feeds: Mapping[str, Stream],
        guess: Mapping[str, Stream] | None,
        sweeps: int,
    ) -> dict[str, Stream]:
        """Build an initial guess by repeated forward (sequential-modular) sweeps.

        Starts from a default interior seed and re-evaluates every block in
        registration order ``sweeps`` times. Acyclic sections become exact; a
        recycle is only roughly closed (Newton finishes it), but the seed lands
        in the two-phase basin the equifugacity residuals need.
        """
        known: dict[str, Stream] = {**feeds, **self._seed(ctx, internal, feeds, guess)}
        resolved = set(feeds)
        for _ in range(sweeps):
            for blk in self.blocks:
                known.update(blk.forward(known, params, ctx))
                if all(name in resolved for name in blk.inlets):
                    resolved.update(blk.outlets)
            if all(name in resolved for name in internal):
                break
        return {name: known[name] for name in internal}

    def _initial_aux(
        self, ctx: Context, streams: Mapping[str, Stream], params: Mapping[str, Any]
    ) -> dict[str, Array]:
        """Initial values for all block auxiliary unknowns."""
        out: dict[str, Array] = {}
        for blk in self.blocks:
            out.update(blk.aux_init(streams, params, ctx))
        return out

    # -- residual assembly ------------------------------------------------- #
    def _assemble_streams(
        self,
        feeds: Mapping[str, Stream],
        ctx: Context,
        internal: Sequence[str],
        u: Mapping[str, Any],
        *,
        for_residual: bool = False,
    ) -> dict[str, Stream]:
        """Reconstruct the full physical streams dict (feeds + internal) from unknowns."""
        return {
            **feeds,
            **{name: _unpack_stream(u["s"][name], ctx.components, ctx.scales) for name in internal},
        }

    def _assemble_residual(
        self,
        streams: Mapping[str, Stream],
        u: Mapping[str, Any],
        params: Mapping[str, Any],
        aux_scales: Mapping[str, float],
        ctx: Context,
    ) -> Array:
        """Stack all block residuals and design-spec residuals into one vector."""
        params = dict(params)
        aux = {k: u["a"][k] * aux_scales[k] for k in aux_scales}
        for sp in self.specs:
            params[sp.manipulated] = u["d"][sp.manipulated] * sp.scale()
        parts = [blk.residuals(streams, aux, params, ctx) for blk in self.blocks]
        for sp in self.specs:
            val = sp.measure(streams)
            parts.append(((val - jnp.asarray(sp.target)) / sp.residual_scale())[None])
        return jnp.concatenate(parts)

    def _residual_fn(
        self,
        ctx: Context,
        internal: Sequence[str],
        aux_scales: Mapping[str, float],
        unravel: Callable[[Array], Any],
    ) -> Callable[[Array, Any], Array]:
        """Build the flat residual ``F(x, theta)`` for process Newton and rank diagnostics."""

        def residual(x: Array, theta: Any) -> Array:
            u = unravel(x)
            current_ctx = replace(ctx, model=theta["pkg"])
            streams = self._assemble_streams(
                theta["feeds"], current_ctx, internal, u, for_residual=True
            )
            return self._assemble_residual(streams, u, theta["params"], aux_scales, current_ctx)

        return residual

    def _initial_unknowns(
        self,
        ctx: Context,
        internal: Sequence[str],
        aux_scales: Mapping[str, float],
        params: Mapping[str, Any],
        feeds: Mapping[str, Stream],
        guess: Mapping[str, Stream] | None,
        sweeps: int,
    ) -> dict[str, Any]:
        """Assemble the scaled initial unknown pytree (streams, aux, spec vars)."""
        # A design spec frees its manipulated parameter, so it is absent from
        # ``params``; seed it at the spec's init value so the forward sweeps (which
        # evaluate the sequential-modular units) have a value to work with.
        params = {**{sp.manipulated: sp.init for sp in self.specs}, **dict(params)}
        streams0 = self._initial_streams(ctx, internal, params, feeds, guess, sweeps)
        aux0 = self._initial_aux(ctx, {**feeds, **streams0}, params)
        return {
            "s": {name: _pack_stream(streams0[name], ctx.scales, ctx.package) for name in internal},
            "a": {k: aux0[k] / aux_scales[k] for k in aux_scales},
            "d": {sp.manipulated: jnp.asarray(sp.init) / sp.scale() for sp in self.specs},
        }

    # -- compiled plan (cached per structure) ------------------------------ #
    def _structural_key(self, sweeps: int, tol: float, max_iter: int) -> tuple[Any, ...]:
        """A hashable key capturing everything the compiled residual depends on.

        Block ``repr`` captures every literal field (so a changed inline spec
        rebuilds); feeds contribute only their names/components/shapes (their
        *values* are dynamic inputs to the JIT, so changing them is free); specs
        contribute their manipulated key and measure identity.
        """
        feeds_sig = tuple(
            (name, self.feeds[name].components, tuple(self.feeds[name].n.shape))
            for name in sorted(self.feeds)
        )
        specs_sig = tuple(
            (sp.manipulated, id(sp.measure), repr(sp.target), repr(sp.init)) for sp in self.specs
        )
        scales_sig = None if self.scales is None else repr(self.scales)
        model_sig = resolve_package(self._components(), self.model).signature()
        return (
            tuple(repr(b) for b in self.blocks),
            feeds_sig,
            specs_sig,
            scales_sig,
            model_sig,
            self.linear_solver,
            int(sweeps),
            float(tol),
            int(max_iter),
        )

    def _get_plan(self, params: Mapping[str, Any], sweeps: int, tol: float, max_iter: int) -> _Plan:
        """Reuse local residual and Jacobian kernels for an unchanged structure."""
        key = self._structural_key(sweeps, tol, max_iter)
        cached = self._plans.get(key)
        if cached is not None:
            return cached

        ctx = self._context()
        internal = tuple(self._internal_names())
        aux_scales = self._aux_scales(ctx)
        if self.linear_solver not in ("sparse", "dense"):
            raise ValueError("EO linear_solver must be sparse or dense")
        pattern, _ = self._incidence(ctx)

        # Warm the process-global component-data cache with *concrete* arrays so
        # the JIT-compiled core below hits it, instead of caching a trace-time
        # array that then leaks across solves as an UnexpectedTracerError.
        # ``ensure_compile_time_eval`` keeps this concrete even when the very
        # first solve happens under ``jax.grad``.
        with jax.ensure_compile_time_eval():
            _resolve(ctx.components)

        # A structural template fixes the unknown layout (the unravel); its values
        # are irrelevant, so zero sweeps keeps it cheap.
        template = self._initial_unknowns(ctx, internal, aux_scales, params, self.feeds, None, 0)
        _, unravel = ravel_pytree(template)
        block_equations = []
        row = 0
        size = pattern.columns
        for index in range(len(self.blocks) + len(self.specs)):
            count = self.blocks[index].n_residuals(ctx) if index < len(self.blocks) else 1
            indices = tuple(
                sorted({c for columns in pattern.rows[row : row + count] for c in columns})
            )

            def local(v: Array, theta: Any, index: int = index, indices: Any = indices) -> Array:
                x = jnp.zeros(size, dtype=v.dtype).at[jnp.asarray(indices, dtype=int)].set(v)
                u = unravel(x)
                current_ctx = replace(ctx, model=theta["pkg"])
                streams = self._assemble_streams(theta["feeds"], current_ctx, internal, u)
                params = dict(theta["params"])
                for spec in self.specs:
                    params[spec.manipulated] = u["d"][spec.manipulated] * spec.scale()
                if index < len(self.blocks):
                    aux = {k: u["a"][k] * aux_scales[k] for k in aux_scales}
                    return self.blocks[index].residuals(streams, aux, params, current_ctx)
                spec = self.specs[index - len(self.blocks)]
                return jnp.atleast_1d((spec.measure(streams) - spec.target) / spec.residual_scale())

            block_equations.append((indices, count, local))
            row += count
        # Non-square systems remain available to rank diagnostics; solve rejects
        # their degrees of freedom before executing a plan.
        system = (
            ResidualGraph(size, block_equations, linear_solver=self.linear_solver)
            if len(pattern.rows) == size
            else None
        )
        lower_tree = jax.tree_util.tree_map(lambda x: jnp.full_like(x, -jnp.inf), template)
        for name in internal:
            lower_tree["s"][name] = jnp.concatenate(
                [
                    jnp.zeros(ctx.n_components),
                    jnp.array(
                        [
                            50.0 / ctx.scales.temperature,
                            1.0 / ctx.scales.pressure,
                        ]
                    ),
                    jnp.full(ctx.n_components, -1.0 / ctx.scales.flow),
                ]
            )
        lower = ravel_pytree(lower_tree)[0]

        def newton(x0: Array, params_: Any, feeds_: Any, pkg_: Any) -> tuple[Array, SolveReport]:
            """Newton solve from an externally supplied (detached) seed ``x0``."""
            theta = {"params": params_, "feeds": feeds_, "pkg": pkg_}
            if system is None:
                raise ValueError("flowsheet is not square")
            result = newton_iterations(
                system.residual,
                lambda x, p: system.linearize(x, p)[0],
                x0,
                theta,
                scale=jnp.ones_like(x0),
                tolerance=tol,
                max_iterations=max_iter,
                lower=lower,
            )
            value = system.attach(result.value, theta, result.report.converged)
            return value, result.report

        def core(params_: Any, feeds_: Any, pkg_: Any) -> tuple[Array, SolveReport]:
            raw_params, raw_feeds, raw_pkg = jax.lax.stop_gradient((params_, feeds_, pkg_))
            u0 = self._initial_unknowns(
                replace(ctx, model=raw_pkg),
                internal,
                aux_scales,
                raw_params,
                raw_feeds,
                None,
                sweeps,
            )
            return newton(ravel_pytree(u0)[0], params_, feeds_, pkg_)

        report = self.degrees_of_freedom()
        plan = _Plan(
            ctx=ctx,
            internal=internal,
            aux_scales=aux_scales,
            unravel=unravel,
            core=core,
            newton=newton,
            n_unknowns=report.n_unknowns,
            n_equations=report.n_equations,
            seed=[],
        )
        self._plans[key] = plan
        return plan

    def diagnose(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        guess: Mapping[str, Stream] | None = None,
        rank_tol: float | None = None,
    ) -> dict[str, Any]:
        """Inspect the scaled Jacobian at an initial or user-supplied state.

        Reports numerical rank, conditioning, zero rows and columns, and the
        corresponding unit/variable labels. Rank is local to the supplied state;
        a square equation count alone doesn't establish a solvable specification.
        This diagnostic runs on the host and doesn't mutate the warm-start cache.
        """
        import numpy as np

        params = dict(params or {})
        plan = self._get_plan(params, 0, 1e-10, 60)
        pkg = resolve_package(self._components(), self.model)
        ctx = replace(plan.ctx, model=pkg)
        u = self._initial_unknowns(
            ctx, plan.internal, plan.aux_scales, params, self.feeds, guess, 0
        )
        x = ravel_pytree(u)[0]
        residual = self._residual_fn(ctx, plan.internal, plan.aux_scales, plan.unravel)
        jac = np.asarray(
            jax.jacrev(residual)(x, {"params": params, "feeds": self.feeds, "pkg": pkg})
        )
        labels = self.equation_labels(ctx)
        variables = tuple(
            f"{jax.tree_util.keystr(path)}[{i}]"
            for path, leaf in jax.tree_util.tree_flatten_with_path(u)[0]
            for i in range(leaf.size)
        )
        finite = bool(np.isfinite(jac).all())
        singular_values = np.linalg.svd(jac, compute_uv=False) if finite else np.array([])
        threshold = (
            rank_tol
            if rank_tol is not None
            else (
                max(jac.shape) * np.finfo(float).eps * singular_values[0]
                if singular_values.size
                else 0.0
            )
        )
        rank = int(np.sum(singular_values > threshold))
        condition = (
            float(singular_values[0] / singular_values[-1])
            if singular_values.size and singular_values[-1] > threshold
            else None
        )
        return {
            "structure": self.diagnose_structure(),
            "n_unknowns": plan.n_unknowns,
            "n_equations": plan.n_equations,
            "degrees_of_freedom": plan.n_unknowns - plan.n_equations,
            "finite": finite,
            "rank": rank,
            "condition_number": condition,
            "full_rank": finite and rank == min(jac.shape),
            "zero_equations": [
                labels[i] for i in range(len(labels)) if np.all(np.abs(jac[i]) <= threshold)
            ],
            "unconstrained_variables": [
                variables[i]
                for i in range(len(variables))
                if np.all(np.abs(jac[:, i]) <= threshold)
            ],
        }

    def solve_path(
        self,
        start: Any,
        target: Any,
        *,
        solve_options: Mapping[str, Any] | None = None,
        **options: Any,
    ) -> Any:
        """Continue operating conditions, reusing only accepted stream guesses."""
        from fugacio.sim.continuation import continuation_solve

        def solve(params: Any, previous: Any) -> tuple[EOSolution, SolveReport]:
            settings = {**dict(solve_options or {}), "check": False, "warm_start": False}
            if previous is not None:
                settings["guess"] = previous.streams
            result = self.solve(params, **settings)
            return result, result.report

        return continuation_solve(solve, start, target, **options)

    # -- solve ------------------------------------------------------------- #
    def solve(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        guess: Mapping[str, Stream] | None = None,
        sweeps: int = 6,
        tol: float = 1e-10,
        max_iter: int = 60,
        check_dof: bool = True,
        check: bool = True,
        warm_start: bool = True,
    ) -> EOSolution:
        """Solve the flowsheet simultaneously and return all streams.

        Args:
            params: Parameter mapping read by blocks (operating conditions, split
                fractions, ...). Gradients with respect to these (and the feeds)
                flow through the converged solution by implicit differentiation.
            guess: Optional initial guesses for specific internal streams (useful
                to seed a recycle); other streams get a default interior seed.
            sweeps: Forward sweeps used to build the initial guess.
            tol: Maximum accepted scaled equation residual.
            max_iter: Newton iteration cap.
            check: Raise for concrete failures; mask compiled failed values with NaNs.
            warm_start: Reuse the most recent converged state for this structure.
            check_dof: If ``True``, raise when the flowsheet is not square
                (``degrees_of_freedom != 0``).

        Returns:
            An `EOSolution` with every solved stream, differentiable in ``params``
            and the feeds.

        Raises:
            ValueError: if ``check_dof`` and the flowsheet is under/over-specified.
        """
        params = dict(params or {})
        plan = self._get_plan(params, sweeps, tol, max_iter)
        pkg = resolve_package(self._components(), self.model)

        if check_dof:
            dof = plan.n_unknowns - plan.n_equations
            if dof != 0:
                raise ValueError(
                    f"flowsheet is not square: {plan.n_unknowns} unknowns vs "
                    f"{plan.n_equations} equations (degrees of freedom {dof}). "
                    "Add or remove a design spec, or fix/free an operating variable."
                )

        # Seeds never carry derivatives. Local kernels are reused whether the
        # orchestration is concrete or explicitly staged by the caller.
        if (
            guess is None
            and not _is_traced(params, self.feeds, pkg)
            and not (warm_start and plan.seed)
        ):
            x_star, solve_report = plan.core(params, self.feeds, pkg)
        else:
            # Warm-start from the last concrete solution when available (so no
            # sequential-modular sweeps run under autodiff); otherwise build the
            # seed eagerly this once.
            if guess is None and warm_start and plan.seed:
                x0 = plan.seed[0]
            else:
                raw_params, raw_feeds, raw_pkg, raw_guess = jax.lax.stop_gradient(
                    (params, self.feeds, pkg, guess)
                )
                u0 = self._initial_unknowns(
                    replace(plan.ctx, model=raw_pkg),
                    plan.internal,
                    plan.aux_scales,
                    raw_params,
                    raw_feeds,
                    raw_guess,
                    sweeps,
                )
                x0 = ravel_pytree(u0)[0]
            x0 = jax.lax.stop_gradient(x0)
            x_star, solve_report = plan.newton(x0, params, self.feeds, pkg)

        ctx = replace(plan.ctx, model=pkg)
        u = plan.unravel(x_star)
        streams = self._assemble_streams(self.feeds, ctx, plan.internal, u)
        aux = {k: u["a"][k] * plan.aux_scales[k] for k in plan.aux_scales}
        # A converged solution a block can't physically realize (an exchanger
        # temperature cross) is reported, not returned as an answer.
        solved_params = {
            **params,
            **{sp.manipulated: u["d"][sp.manipulated] * sp.scale() for sp in self.specs},
        }
        realizable = [
            jnp.asarray(b.feasible(streams, aux, solved_params, ctx)) for b in self.blocks
        ]
        realizable.extend(stream.report.converged for stream in streams.values())
        feasible = jnp.all(jnp.stack(realizable)) if realizable else jnp.asarray(True)
        solve_report = with_status(
            solve_report, solve_report.converged & ~feasible, SolveStatus.INFEASIBLE
        )
        if not _is_traced(x_star) and bool(solve_report.converged):
            plan.seed[:] = [jax.lax.stop_gradient(x_star)]
        x_star = gate_derivative(x_star, solve_report.converged)
        if check:
            require_converged(
                solve_report, "equation-oriented flowsheet", self.equation_labels(plan.ctx)
            )
            x_star = jnp.where(solve_report.converged, x_star, jnp.nan)
        u = plan.unravel(x_star)
        streams = self._assemble_streams(self.feeds, ctx, plan.internal, u)
        aux = {k: u["a"][k] * plan.aux_scales[k] for k in plan.aux_scales}
        specs = {sp.manipulated: u["d"][sp.manipulated] * sp.scale() for sp in self.specs}
        return EOSolution(
            streams=streams,
            aux=aux,
            specs=specs,
            residual_norm=solve_report.residual_norm,
            report=solve_report,
            n_unknowns=plan.n_unknowns,
            n_equations=plan.n_equations,
        )


def _auto_scales(feeds: Mapping[str, Stream]) -> Scales:
    """Derive characteristic scales from the feeds (a stable default conditioning)."""
    if _is_traced(feeds):
        return Scales()
    # Concrete host reductions stay concrete even when called while tracing an
    # enclosing optimization. JAX reductions would create tracers here.
    flow = max(sum(sum(s.n.tolist()) for s in feeds.values()), 1.0)
    t_avg = sum(float(s.t) for s in feeds.values()) / len(feeds)
    p_avg = sum(float(s.p) for s in feeds.values()) / len(feeds)
    return Scales(
        flow=flow,
        temperature=max(t_avg, 1.0),
        pressure=max(p_avg, 1.0e3),
        enthalpy_flow=flow * 1.0e4,
        enthalpy_molar=1.0e4,
        entropy_molar=10.0,
    )


jax.tree_util.register_dataclass(
    EOSolution,
    data_fields=["streams", "aux", "specs", "residual_norm", "report"],
    meta_fields=["n_unknowns", "n_equations"],
)
