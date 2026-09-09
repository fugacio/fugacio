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
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from fugacio.sim.eo.blocks import Block, Context, Scales
from fugacio.sim.properties import Model, _resolve, molar_enthalpy, resolve_package
from fugacio.sim.stream import Stream
from fugacio.thermo import CubicEOS
from fugacio.thermo.diagnostics import SolveReport, require_converged
from fugacio.thermo.eos import PR
from fugacio.thermo.implicit import newton_system_with_info
from fugacio.thermo.sparsity import SparsityPattern

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


def _pack_stream(s: Stream, sc: Scales, pkg: Any = None) -> Array:
    """Pack material, thermal, and pressure coordinates.

    Pure-fluid streams use enthalpy as the thermal coordinate because their
    temperature does not determine quality on the saturation line. Mixtures
    retain the historical temperature coordinate.
    """
    thermal = (
        molar_enthalpy(s, model=pkg) / sc.enthalpy_molar
        if len(s.components) == 1 and pkg is not None
        else s.t / sc.temperature
    )
    return jnp.concatenate(
        [s.n / sc.flow, jnp.atleast_1d(thermal), jnp.atleast_1d(s.p / sc.pressure)]
    )


@partial(jax.jit, static_argnames=("components", "sc"))
def _unpack_stream(vec: Array, components: tuple[str, ...], sc: Scales, pkg: Any = None) -> Stream:
    """Reconstruct streams, retaining pure-fluid phase amounts through PH state."""
    c = len(components)
    n, p = vec[:c] * sc.flow, vec[c + 1] * sc.pressure
    if c == 1 and pkg is not None:
        r = pkg.flash_ph(p, vec[c] * sc.enthalpy_molar, jnp.ones(1))
        return Stream(n, r.t, p, components, r.beta * n)
    return Stream(n, vec[c] * sc.temperature, p, components)


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
        eos: Cubic equation of state for the default property package.
        kij: Optional binary-interaction matrix for the default package.
        scales: Residual/variable scales (auto-derived from the feeds by
            `solve` when left at the default).
        model: Property package (see `fugacio.sim.models.package_for`) used by
            every block; ``None`` selects the cubic default built from ``eos`` /
            ``kij``. Any method class (cubic, gamma-phi, PC-SAFT, reference
            fluid) can drive the whole simultaneous solve.
    """

    eos: CubicEOS = PR
    kij: Array | None = None
    scales: Scales | None = None
    model: Model = None
    feeds: dict[str, Stream] = field(default_factory=dict)
    blocks: list[Block] = field(default_factory=list)
    specs: list[_DesignSpec] = field(default_factory=list)
    jacobian_mode: str = "colored"
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
        produced: dict[str, int] = {}
        for i, blk in enumerate(self.blocks):
            if not blk.outlets:
                raise ValueError(f"block #{i} must declare at least one outlet")
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
            streams[name] = tuple(range(len(variables), len(variables) + ctx.n_components + 2))
            variables.extend(f"{name}:n[{component}]" for component in ctx.components)
            variables.extend(
                (
                    name + (":enthalpy" if ctx.n_components == 1 else ":temperature"),
                    name + ":pressure",
                )
            )
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
            "jacobian_mode": self.jacobian_mode,
            "global_linear_solver": "pivoted_dense",
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
        return Context(
            components=comps, eos=self.eos, kij=self.kij, scales=scales, model=self.model
        )

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
        streams: dict[str, Stream] = dict(feeds)
        thermal_needed = set(internal)
        if for_residual and ctx.n_components == 1 and not self.specs:
            from fugacio.sim.eo.blocks import Flash, Heater, Mixer, Splitter, Valve

            thermal_needed = set()
            for block in self.blocks:
                if type(block) is Heater:
                    if block.t_out is not None:
                        thermal_needed.update(block.outlets)
                elif type(block) is Mixer:
                    if block.t is not None:
                        thermal_needed.update(block.outlets)
                elif type(block) not in (Splitter, Valve, Flash):
                    thermal_needed.update(block.inlets + block.outlets)
        for name in internal:
            if name in thermal_needed:
                streams[name] = _unpack_stream(
                    u["s"][name], ctx.components, ctx.scales, ctx.package
                )
            else:
                # These built-in equations read the H coordinate directly.
                # Don't trace unused PH inversions into their Newton system.
                values = u["s"][name]
                streams[name] = Stream(
                    values[:1] * ctx.scales.flow,
                    jnp.asarray(300.0),
                    values[2] * ctx.scales.pressure,
                    ctx.components,
                )
        if ctx.n_components > 1:
            from fugacio.sim.eo.blocks import Flash

            for block in self.blocks:
                if isinstance(block, Flash):
                    vapor, liquid = block.outlets
                    streams[vapor] = replace(streams[vapor], vapor_n=streams[vapor].n)
                    streams[liquid] = replace(
                        streams[liquid], vapor_n=jnp.zeros_like(streams[liquid].n)
                    )
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
        if ctx.n_components == 1:
            ctx = replace(
                ctx,
                enthalpy_coordinates={
                    id(streams[name]): values[1] * ctx.scales.enthalpy_molar
                    for name, values in u["s"].items()
                },
            )
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
        kij_sig = None if self.kij is None else tuple(jnp.asarray(self.kij).shape)
        scales_sig = None if self.scales is None else repr(self.scales)
        model_sig = (
            None
            if self.model is None
            else resolve_package(
                self._components(), self.model, eos=self.eos, kij=self.kij
            ).signature()
        )
        return (
            tuple(repr(b) for b in self.blocks),
            feeds_sig,
            specs_sig,
            repr(self.eos),
            kij_sig,
            scales_sig,
            model_sig,
            self.jacobian_mode,
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
        if self.jacobian_mode not in ("colored", "dense"):
            raise ValueError("EO jacobian_mode must be colored or dense")
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
        residual = self._residual_fn(ctx, internal, aux_scales, unravel)
        jacobian = (
            pattern.jacobian(residual)
            if self.jacobian_mode == "colored" and len(pattern.rows) == pattern.columns
            else None
        )
        lower_tree = jax.tree_util.tree_map(lambda x: jnp.full_like(x, -jnp.inf), template)
        for name in internal:
            lower_tree["s"][name] = jnp.concatenate(
                [
                    jnp.zeros(ctx.n_components),
                    jnp.array(
                        [
                            -jnp.inf if ctx.n_components == 1 else 50.0 / ctx.scales.temperature,
                            1.0 / ctx.scales.pressure,
                        ]
                    ),
                ]
            )
        lower = ravel_pytree(lower_tree)[0]

        def newton(x0: Array, params_: Any, feeds_: Any, pkg_: Any) -> tuple[Array, SolveReport]:
            """Newton solve from an externally supplied (detached) seed ``x0``."""
            theta = {"params": params_, "feeds": feeds_, "pkg": pkg_}
            result = newton_system_with_info(
                residual,
                jax.lax.stop_gradient(x0),
                theta,
                tol,
                max_iter,
                lower=lower,
                jacobian=jacobian,
            )
            return result.value, result.report

        def core(params_: Any, feeds_: Any, pkg_: Any) -> tuple[Array, SolveReport]:
            # The sequential-modular seed sweeps and the Newton solve share one
            # JIT, so a flowsheet compiles once and every later (non-differentiated)
            # solve reuses it. This is the fast path for plain solves and the
            # forward sweeps of a finite-difference check.
            u0 = self._initial_unknowns(
                replace(ctx, model=pkg_), internal, aux_scales, params_, feeds_, None, sweeps
            )
            return newton(ravel_pytree(u0)[0], params_, feeds_, pkg_)

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
        pkg = resolve_package(self._components(), self.model, eos=self.eos, kij=self.kij)
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
        pkg = resolve_package(self._components(), self.model, eos=self.eos, kij=self.kij)

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
                u0 = self._initial_unknowns(
                    replace(plan.ctx, model=pkg),
                    plan.internal,
                    plan.aux_scales,
                    params,
                    self.feeds,
                    guess,
                    sweeps,
                )
                x0 = ravel_pytree(u0)[0]
            x0 = jax.lax.stop_gradient(x0)
            x_star, solve_report = plan.newton(x0, params, self.feeds, pkg)

        if not _is_traced(x_star) and bool(solve_report.converged):
            plan.seed[:] = [jax.lax.stop_gradient(x_star)]
        if check:
            require_converged(
                solve_report, "equation-oriented flowsheet", self.equation_labels(plan.ctx)
            )
            x_star = jnp.where(solve_report.converged, x_star, jnp.nan)
        u = plan.unravel(x_star)
        streams = self._assemble_streams(self.feeds, replace(plan.ctx, model=pkg), plan.internal, u)
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
