"""Equation-oriented flowsheeting: solve the whole flowsheet as one system.

Where the sequential-modular engine (`fugacio.sim.flowsheet`) evaluates units in
order and converges recycles by tearing, the *equation-oriented* (EO) engine
collects every unit's equations, the stream connectivity, the recycles, and any
design specs into a single residual system ``F(x, theta) = 0`` and solves it
**simultaneously** with Newton's method. There is no tear stream and no unit
ordering: a recycle is just a stream that two blocks happen to share, and the
global solve closes it like any other equation.

This is the formulation a differentiable core is built for. The one expensive
ingredient of an EO solver, the Jacobian ``dF/dx``, is supplied *exactly* by JAX
autodiff rather than by finite differences or hand-coded analytic blocks, and the
converged solution is itself differentiable with respect to the parameters
``theta`` (operating conditions, feeds, prices, model parameters) by the implicit
function theorem (`fugacio.thermo.implicit.newton_system`). So a gradient of any
product spec, duty, or cost through the *entire converged plant*, recycles and
all, costs a single adjoint solve.

The unknowns are the flowsheet's internal streams (per-component molar flows plus
temperature and pressure) together with any block auxiliary variables and any
freed design-spec variables. Everything is carried in a non-dimensional form (see
`fugacio.sim.eo.blocks.Scales`) so the Newton system stays well conditioned.
`EOFlowsheet.degrees_of_freedom` reports the unknown/equation balance, the EO
analogue of a specification check.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from fugacio.sim.eo.blocks import Block, Context, Scales
from fugacio.sim.properties import _resolve
from fugacio.sim.stream import Stream
from fugacio.thermo import CubicEOS
from fugacio.thermo.eos import PR
from fugacio.thermo.implicit import newton_system

#: A measurement read off the solved streams (for a design spec / objective).
Measure = Callable[[Mapping[str, Stream]], Array]


def _is_traced(*trees: Any) -> bool:
    """Whether any array leaf is a JAX tracer (i.e. we are inside a transform).

    Used to pick the solve path: the fully JIT-compiled seed-and-solve core is
    only valid for concrete inputs; under ``jax.grad``/``jax.jvp`` the seed is
    instead built eagerly and detached, with only the Newton solve compiled.
    """
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(trees))


class DOFReport(NamedTuple):
    """Degrees-of-freedom analysis of an EO flowsheet.

    Attributes:
        n_unknowns: Total scalar unknowns (internal stream variables + block
            auxiliaries + freed design-spec variables).
        n_equations: Total scalar equations (block residuals + design specs).
        degrees_of_freedom: ``n_unknowns - n_equations``. Zero means the
            flowsheet is exactly specified and solvable; positive means
            under-specified (add specs); negative means over-specified.
        per_block: Equation count contributed by each block (keyed by the block's
            first outlet name).
    """

    n_unknowns: int
    n_equations: int
    degrees_of_freedom: int
    per_block: dict[str, int]


class _Plan(NamedTuple):
    """A compiled, cached solve plan for one flowsheet *structure*.

    Building the residual closure and JIT-compiling the Newton core is the
    expensive part of an EO solve (tens of seconds for a flash-bearing system),
    but it depends only on the flowsheet's *structure* (its blocks, connectivity,
    specs, and scales), not on the numeric parameter or feed *values*. Caching the
    plan on the flowsheet (keyed by that structure) means repeated solves, the
    forward sweeps of a finite-difference check, and the inner solves of an
    optimization all reuse one compilation and run essentially for free.
    """

    ctx: Context
    internal: tuple[str, ...]
    aux_scales: dict[str, float]
    unravel: Callable[[Array], Any]
    core: Callable[[Any, Any], tuple[Array, Array]]
    newton: Callable[[Array, Any, Any], tuple[Array, Array]]
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
        aux: Solved block auxiliary variables (e.g. isentropic temperatures).
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

    def __getitem__(self, name: str) -> Stream:
        """Return the solved stream ``name``."""
        return self.streams[name]


def _pack_stream(s: Stream, sc: Scales) -> Array:
    """Pack a stream into its scaled unknown vector ``[n/flow, t/T, p/P]``."""
    return jnp.concatenate([s.n / sc.flow, (s.t / sc.temperature)[None], (s.p / sc.pressure)[None]])


def _unpack_stream(vec: Array, components: tuple[str, ...], sc: Scales) -> Stream:
    """Reconstruct a physical stream from its scaled unknown vector."""
    c = len(components)
    return Stream(
        n=vec[:c] * sc.flow,
        t=vec[c] * sc.temperature,
        p=vec[c + 1] * sc.pressure,
        components=components,
    )


@dataclass
class EOFlowsheet:
    """Declarative equation-oriented flowsheet.

    Register feeds and blocks, optionally add design specs, then call `solve`.
    Streams are referenced by name; a block's outlet names become the system
    unknowns and recycles need no special handling (just reuse a downstream
    stream name as an upstream block's inlet).

    Example::

        fs = EOFlowsheet(eos=PR)
        fs.feed("fresh", fresh_stream)
        fs.add(Mixer(inlets=("fresh", "recycle"), outlets=("mixed",)))
        fs.add(Flash(inlets=("mixed",), outlets=("vapor", "liquid"), t="T", p="P"))
        fs.add(Splitter(inlets=("liquid",), outlets=("recycle", "purge"),
                        fractions=("r_recycle",)))
        sol = fs.solve({"T": 320.0, "P": 20e5, "r_recycle": [0.5, 0.5]})
        product = sol["vapor"]

    Attributes:
        eos: Cubic equation of state used everywhere in the flowsheet.
        kij: Optional binary-interaction matrix.
        scales: Residual/variable scales (auto-derived from the feeds by
            `solve` when left at the default).
    """

    eos: CubicEOS = PR
    kij: Array | None = None
    scales: Scales | None = None
    feeds: dict[str, Stream] = field(default_factory=dict)
    blocks: list[Block] = field(default_factory=list)
    specs: list[_DesignSpec] = field(default_factory=list)
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
        produced: dict[str, int] = {}
        for i, blk in enumerate(self.blocks):
            for name in blk.outlets:
                if name in self.feeds:
                    raise ValueError(f"block output {name!r} shadows a feed of the same name")
                if name in produced:
                    raise ValueError(
                        f"stream {name!r} is produced by two blocks "
                        f"(#{produced[name]} and #{i}); each stream needs exactly one source"
                    )
                produced[name] = i
        for i, blk in enumerate(self.blocks):
            for name in blk.inlets:
                if name not in self.feeds and name not in produced:
                    raise ValueError(
                        f"block #{i} reads undefined stream {name!r} "
                        "(not a feed and not produced by any block)"
                    )
        return sorted(produced)

    def _aux_scales(self, ctx: Context) -> dict[str, float]:
        """Collect every block's auxiliary unknowns into one key -> scale map."""
        out: dict[str, float] = {}
        for blk in self.blocks:
            out.update(blk.aux_scales(ctx))
        return out

    def _context(self) -> Context:
        """Build the static solve context (components, EOS, scales)."""
        comps = self._components()
        scales = self.scales if self.scales is not None else _auto_scales(self.feeds)
        return Context(components=comps, eos=self.eos, kij=self.kij, scales=scales)

    def degrees_of_freedom(self) -> DOFReport:
        """Report the unknown/equation balance for the flowsheet (see `DOFReport`)."""
        ctx = self._context()
        internal = self._internal_names()
        per_block = {blk.outlets[0]: blk.n_residuals(ctx) for blk in self.blocks}
        n_aux = len(self._aux_scales(ctx))
        n_specs = len(self.specs)
        n_unknowns = len(internal) * (ctx.n_components + 2) + n_aux + n_specs
        n_equations = sum(per_block.values()) + n_specs
        return DOFReport(
            n_unknowns=n_unknowns,
            n_equations=n_equations,
            degrees_of_freedom=n_unknowns - n_equations,
            per_block=per_block,
        )

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
        for _ in range(sweeps):
            for blk in self.blocks:
                known.update(blk.forward(known, params, ctx))
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
    @staticmethod
    def _assemble_streams(
        feeds: Mapping[str, Stream], ctx: Context, internal: Sequence[str], u: Mapping[str, Any]
    ) -> dict[str, Stream]:
        """Reconstruct the full physical streams dict (feeds + internal) from unknowns."""
        streams: dict[str, Stream] = dict(feeds)
        for name in internal:
            streams[name] = _unpack_stream(u["s"][name], ctx.components, ctx.scales)
        return streams

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
        """Build the flat residual ``F(x, theta)`` for `newton_system`."""

        def residual(x: Array, theta: Any) -> Array:
            u = unravel(x)
            streams = self._assemble_streams(theta["feeds"], ctx, internal, u)
            return self._assemble_residual(streams, u, theta["params"], aux_scales, ctx)

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
            "s": {name: _pack_stream(streams0[name], ctx.scales) for name in internal},
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
        specs_sig = tuple((sp.manipulated, id(sp.measure)) for sp in self.specs)
        kij_sig = None if self.kij is None else tuple(jnp.asarray(self.kij).shape)
        scales_sig = None if self.scales is None else repr(self.scales)
        return (
            tuple(repr(b) for b in self.blocks),
            feeds_sig,
            specs_sig,
            repr(self.eos),
            kij_sig,
            scales_sig,
            int(sweeps),
            float(tol),
            int(max_iter),
        )

    def _get_plan(self, params: Mapping[str, Any], sweeps: int, tol: float, max_iter: int) -> _Plan:
        """Return the compiled plan for this structure, building/caching on miss.

        The plan's JIT core (and its adjoint), the residual norm, and the
        initial-guess builder all compile once; every later solve with the same
        structure reuses them, so only the parameter/feed values vary. This is
        what makes finite-difference checks and optimization inner loops cheap.
        """
        key = self._structural_key(sweeps, tol, max_iter)
        cached = self._plans.get(key)
        if cached is not None:
            return cached

        ctx = self._context()
        internal = tuple(self._internal_names())
        aux_scales = self._aux_scales(ctx)

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
        residual = self._residual_fn(ctx, internal, aux_scales, unravel)

        def newton(x0: Array, params_: Any, feeds_: Any) -> tuple[Array, Array]:
            """Newton solve from an externally supplied (detached) seed ``x0``."""
            theta = {"params": params_, "feeds": feeds_}
            x_star = newton_system(residual, jax.lax.stop_gradient(x0), theta, tol, max_iter)
            norm = jax.lax.stop_gradient(jnp.max(jnp.abs(residual(x_star, theta))))
            return x_star, norm

        def core(params_: Any, feeds_: Any) -> tuple[Array, Array]:
            # The sequential-modular seed sweeps and the Newton solve share one
            # JIT, so a flowsheet compiles once and every later (non-differentiated)
            # solve reuses it. This is the fast path for plain solves and the
            # forward sweeps of a finite-difference check.
            u0 = self._initial_unknowns(ctx, internal, aux_scales, params_, feeds_, None, sweeps)
            return newton(ravel_pytree(u0)[0], params_, feeds_)

        report = self.degrees_of_freedom()
        plan = _Plan(
            ctx=ctx,
            internal=internal,
            aux_scales=aux_scales,
            unravel=unravel,
            core=jax.jit(core),
            newton=jax.jit(newton),
            n_unknowns=report.n_unknowns,
            n_equations=report.n_equations,
            seed=[],
        )
        self._plans[key] = plan
        return plan

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
    ) -> EOSolution:
        """Solve the flowsheet simultaneously and return all streams.

        Args:
            params: Parameter mapping read by blocks (operating conditions, split
                fractions, ...). Gradients with respect to these (and the feeds)
                flow through the converged solution by implicit differentiation.
            guess: Optional initial guesses for specific internal streams (useful
                to seed a recycle); other streams get a default interior seed.
            sweeps: Forward sweeps used to build the initial guess.
            tol: Newton convergence tolerance on the scaled step.
            max_iter: Newton iteration cap.
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

        if check_dof:
            dof = plan.n_unknowns - plan.n_equations
            if dof != 0:
                raise ValueError(
                    f"flowsheet is not square: {plan.n_unknowns} unknowns vs "
                    f"{plan.n_equations} equations (degrees of freedom {dof}). "
                    "Add or remove a design spec, or fix/free an operating variable."
                )

        # Fast path (plain solves, finite-difference sweeps): the seed and the
        # Newton solve share one cached JIT. Under autodiff, or with a
        # caller-supplied per-stream guess, the seed is built eagerly and only the
        # Newton solve is JIT-compiled: JIT-compiling the sequential-modular seed
        # and then differentiating *it* is both wasteful and unsupported (its phase
        # selections are non-differentiable), whereas the eager seed is detached
        # and differentiation flows solely through ``newton_system``.
        if guess is None and not _is_traced(params, self.feeds):
            x_star, norm = plan.core(params, self.feeds)
            plan.seed[:] = [jax.lax.stop_gradient(x_star)]
        else:
            # Warm-start from the last concrete solution when available (so no
            # sequential-modular sweeps run under autodiff); otherwise build the
            # seed eagerly this once.
            if guess is None and plan.seed:
                x0 = plan.seed[0]
            else:
                u0 = self._initial_unknowns(
                    plan.ctx, plan.internal, plan.aux_scales, params, self.feeds, guess, sweeps
                )
                x0 = ravel_pytree(u0)[0]
            x0 = jax.lax.stop_gradient(x0)
            x_star, norm = plan.newton(x0, params, self.feeds)

        u = plan.unravel(x_star)
        streams = self._assemble_streams(self.feeds, plan.ctx, plan.internal, u)
        aux = {k: u["a"][k] * plan.aux_scales[k] for k in plan.aux_scales}
        specs = {sp.manipulated: u["d"][sp.manipulated] * sp.scale() for sp in self.specs}
        return EOSolution(
            streams=streams,
            aux=aux,
            specs=specs,
            residual_norm=norm,
            n_unknowns=plan.n_unknowns,
            n_equations=plan.n_equations,
        )


def _auto_scales(feeds: Mapping[str, Stream]) -> Scales:
    """Derive characteristic scales from the feeds (a stable default conditioning)."""
    n_total = jnp.sum(jnp.stack([jnp.sum(s.n) for s in feeds.values()]))
    flow = float(jnp.maximum(n_total, 1.0))
    t_avg = float(jnp.mean(jnp.stack([s.t for s in feeds.values()])))
    p_avg = float(jnp.mean(jnp.stack([s.p for s in feeds.values()])))
    return Scales(
        flow=flow,
        temperature=max(t_avg, 1.0),
        pressure=max(p_avg, 1.0e3),
        enthalpy_flow=flow * 1.0e4,
        enthalpy_molar=1.0e4,
        entropy_molar=10.0,
    )
