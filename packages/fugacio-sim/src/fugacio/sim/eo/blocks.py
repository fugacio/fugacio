"""Physical unit declarations for simultaneous process equations.

Built-in blocks call the same unit kernels as sequential execution. Their
connection equations include material, temperature, pressure, and resolved
vapor inventory. Custom blocks can supply explicit residual equations and
auxiliary variables through the Block protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from jax import Array

from fugacio.sim.graph import stream_vector
from fugacio.sim.properties import Model, resolve_package
from fugacio.sim.stream import Stream
from fugacio.sim.units import (
    compressor as _compressor_unit,
)
from fugacio.sim.units import (
    flash_drum as _flash_drum_unit,
)
from fugacio.sim.units import (
    heater as _heater_unit,
)
from fugacio.sim.units import (
    pump as _pump_unit,
)
from fugacio.sim.units import (
    turbine as _turbine_unit,
)
from fugacio.sim.units import (
    valve as _valve_unit,
)
from fugacio.thermo import PropertyPackage

ArrayLike = Array | float

#: A unit specification that is either a literal value or a string key resolved
#: against the parameter mapping passed to the solver (so an operating condition
#: can be made a differentiable parameter, or the manipulated variable of a
#: design spec, simply by naming it).
Spec = float | Array | str


#: Temperature cross (K) an exchanger solution may show before it's infeasible.
_APPROACH_TOLERANCE = 1e-3


@dataclass(frozen=True)
class Scales:
    """Characteristic scales that non-dimensionalise the EO residual system.

    The unknowns and residuals are divided by these so the Newton system is
    well conditioned: molar flows and material balances by ``flow``, temperatures
    and temperature specs by ``temperature``, pressures and pressure specs by
    ``pressure``, total-enthalpy balances by ``enthalpy_flow``, molar-enthalpy
    balances by ``enthalpy_molar``, and molar-entropy relations by
    ``entropy_molar``. Phase-equilibrium (equifugacity) residuals are already
    dimensionless and are left unscaled.

    Attributes:
        flow: Characteristic molar flow (mol/s).
        temperature: Characteristic temperature (K).
        pressure: Characteristic pressure (Pa).
        enthalpy_flow: Characteristic total enthalpy flow (W).
        enthalpy_molar: Characteristic molar enthalpy (J/mol).
        entropy_molar: Characteristic molar entropy (J/mol/K).
    """

    flow: float = 1.0
    temperature: float = 100.0
    pressure: float = 1.0e5
    enthalpy_flow: float = 1.0e4
    enthalpy_molar: float = 1.0e4
    entropy_molar: float = 10.0


@dataclass(frozen=True)
class Context:
    """Static (non-differentiated) data shared by every block during a solve.

    Attributes:
        components: The flowsheet's component names (shared by all streams).
        scales: Residual / variable scales (see `Scales`).
        model: Property package used for every equilibrium / property call;
            ``None`` selects Peng-Robinson over ``components``.
    """

    components: tuple[str, ...]
    scales: Scales = field(default_factory=Scales)
    model: Model = None

    @property
    def n_components(self) -> int:
        """Number of components (the per-stream material-balance count)."""
        return len(self.components)

    @property
    def package(self) -> PropertyPackage:
        """The resolved property package every block evaluates properties with."""
        return resolve_package(self.components, self.model)


def resolve(spec: Spec, params: Mapping[str, Any]) -> Array:
    """Resolve a `Spec` to a JAX array.

    A string is looked up in ``params`` (so the value can be a differentiable
    parameter or a design-spec unknown); anything else is treated as a literal.

    Args:
        spec: A literal value or a key into ``params``.
        params: The parameter mapping passed to the solver.

    Returns:
        The resolved value as a float array.

    Raises:
        KeyError: if ``spec`` is a string with no entry in ``params``.
    """
    if isinstance(spec, str):
        return jnp.asarray(params[spec], dtype=float)
    return jnp.asarray(spec, dtype=float)


@dataclass(frozen=True)
class Block:
    """Base class for an equation-oriented unit block.

    A block names its inlet and outlet streams (by the keys used in the
    flowsheet) and supplies the equations relating them. Outlet streams are
    unknowns of the global EO system; each block "defines" its outlet streams by
    contributing exactly enough residual equations.

    Attributes:
        inlets: Inlet stream names (must already exist as feeds or other blocks'
            outlets).
        outlets: Outlet stream names defined by this block.
    """

    inlets: tuple[str, ...]
    outlets: tuple[str, ...]

    def residual_dependencies(self, ctx: Context) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        """Declare stream and auxiliary dependencies, or return None for all unknowns.

        This is a structural contract for every equation and operating point,
        not an observation of zeros at one state. Built-in blocks depend on
        their ports and owned auxiliaries. Custom classes, including subclasses
        of built-ins, default to all unknowns unless they override this method.
        Freed operating parameters are always included by the assembler.
        """
        if type(self) in (
            Mixer,
            Splitter,
            Heater,
            Valve,
            Pump,
            Compressor,
            Turbine,
            Flash,
            ComponentSeparator,
            HeatExchanger,
            StoichiometricReactor,
            Column,
        ):
            return self.inlets + self.outlets, tuple(self.aux_scales(ctx))
        return None

    def aux_scales(self, ctx: Context) -> dict[str, float]:
        """Auxiliary unknowns introduced by the block, mapped to their scale.

        Built-in blocks keep their internal solves in the common physical
        kernels. A custom block can declare an additional variable here so the
        flowsheet allocates an unknown, then add its defining equation in
        `residuals` and include it in `n_residuals`.
        """
        return {}

    def n_residuals(self, ctx: Context) -> int:
        """Number of residual equations this block contributes."""
        return len(self.outlets) * (2 * ctx.n_components + 2)

    def aux_init(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Array]:
        """Initial values for the block's auxiliary unknowns (unscaled)."""
        return {}

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Evaluate the unit explicitly, returning its outlet streams.

        Reuses the sequential-modular unit operation, so it is the reference the
        EO residuals are built to reproduce. Used to seed the EO solve.
        """
        raise NotImplementedError

    def residuals(
        self,
        streams: Mapping[str, Stream],
        aux: Mapping[str, Array],
        params: Mapping[str, Any],
        ctx: Context,
    ) -> Array:
        """Compare shared stream coordinates against the physical unit kernel.

        Custom residual units can override this and ``n_residuals``. Built-in
        units have one implementation of their physics, shared with ordinary
        Python functions and saved process cases.
        """
        predicted = self.forward(streams, params, ctx)
        c, sc = ctx.n_components, ctx.scales
        scales = jnp.asarray([*[sc.flow] * c, sc.temperature, sc.pressure, *[sc.flow] * c])
        return jnp.concatenate(
            [
                (stream_vector(streams[name]) - stream_vector(predicted[name])) / scales
                for name in self.outlets
            ]
        )

    def feasible(
        self,
        streams: Mapping[str, Stream],
        aux: Mapping[str, Array],
        params: Mapping[str, Any],
        ctx: Context,
    ) -> Array:
        """Whether a solved block state is physically realizable (checked after solving).

        The equations can have solutions no equipment achieves (a heat
        exchanger with a temperature cross). A block that can tell overrides
        this; the flowsheet reports an infeasible solution as not converged.
        """
        return jnp.asarray(True)

    # -- shared residual helpers ------------------------------------------- #
    def _material(self, out: Stream, inflow: Array, ctx: Context) -> Array:
        """Component material balance ``out.n - inflow`` (scaled by the flow scale)."""
        return (out.n - inflow) / ctx.scales.flow

    def _temperature(self, out: Stream, t_target: Array, ctx: Context) -> Array:
        """Temperature spec ``out.t - t_target`` (scaled)."""
        return ((out.t - t_target) / ctx.scales.temperature)[None]

    def _pressure(self, out: Stream, p_target: Array, ctx: Context) -> Array:
        """Pressure spec ``out.p - p_target`` (scaled)."""
        return ((out.p - p_target) / ctx.scales.pressure)[None]


@dataclass(frozen=True)
class Mixer(Block):
    """Adiabatic (or isothermal) mixer: combine inlet streams into one outlet.

    Flows add component-by-component. With ``t`` unset the outlet temperature is
    set by an *adiabatic* energy balance (total enthalpy in equals total enthalpy
    out); set ``t`` to fix the outlet temperature. The outlet pressure defaults to
    the lowest inlet pressure, or is fixed by ``p``.

    Attributes:
        t: Optional outlet temperature spec (literal or parameter key); ``None``
            selects the adiabatic energy balance.
        p: Optional outlet pressure spec; ``None`` uses the minimum inlet pressure.
    """

    t: Spec | None = None
    p: Spec | None = None

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Evaluate via `fugacio.sim.units.mix`."""
        from fugacio.sim.units import mix

        ins = [streams[name] for name in self.inlets]
        t = None if self.t is None else resolve(self.t, params)
        p = None if self.p is None else resolve(self.p, params)
        return {self.outlets[0]: mix(ins, t=t, p=p, model=ctx.model)}


@dataclass(frozen=True)
class Splitter(Block):
    """Flow splitter: one inlet to several outlets sharing composition and state.

    Attributes:
        fractions: Per-outlet split fractions (literal sequence/array or a
            parameter key), one per name in ``outlets``.
    """

    fractions: Spec = 1.0

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Evaluate via `fugacio.sim.units.splitter`."""
        from fugacio.sim.units import splitter

        feed = streams[self.inlets[0]]
        outs = splitter(feed, resolve(self.fractions, params))
        return dict(zip(self.outlets, outs, strict=True))


@dataclass(frozen=True)
class Heater(Block):
    """Heater/cooler on a temperature *or* a duty specification.

    Provide exactly one of ``t_out`` (outlet temperature) or ``duty`` (signed
    heat added, W). ``dp`` is the pressure drop across the block.

    Attributes:
        t_out: Outlet temperature spec, or ``None`` if a duty is given.
        duty: Heat-duty spec (W), or ``None`` if an outlet temperature is given.
        dp: Pressure drop (Pa).
    """

    t_out: Spec | None = None
    duty: Spec | None = None
    dp: Spec = 0.0

    def __post_init__(self) -> None:
        """Validate that exactly one of ``t_out`` / ``duty`` is specified."""
        if (self.t_out is None) == (self.duty is None):
            raise ValueError("Heater requires exactly one of t_out or duty")

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Evaluate via `fugacio.sim.units.heater`."""
        feed = streams[self.inlets[0]]
        t_out = None if self.t_out is None else resolve(self.t_out, params)
        duty = None if self.duty is None else resolve(self.duty, params)
        res = _heater_unit(
            feed,
            t_out=t_out,
            duty=duty,
            dp=resolve(self.dp, params),
            model=ctx.model,
        )
        return {self.outlets[0]: res.outlet}


@dataclass(frozen=True)
class Valve(Block):
    """Isenthalpic (Joule-Thomson) pressure letdown to ``p_out``.

    Attributes:
        p_out: Outlet pressure spec (literal or parameter key).
    """

    p_out: Spec = 1.0e5

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Evaluate via `fugacio.sim.units.valve`."""
        feed = streams[self.inlets[0]]
        out = _valve_unit(feed, resolve(self.p_out, params), model=ctx.model)
        return {self.outlets[0]: out}


@dataclass(frozen=True)
class Pump(Block):
    """Incompressible-liquid pump to ``p_out`` with an isentropic-equivalent efficiency.

    Attributes:
        p_out: Outlet pressure spec.
        efficiency: Pump efficiency in ``(0, 1]``; the lost work heats the outlet.
    """

    p_out: Spec = 1.0e5
    efficiency: Spec = 0.75

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Evaluate via `fugacio.sim.units.pump`."""
        feed = streams[self.inlets[0]]
        res = _pump_unit(
            feed,
            resolve(self.p_out, params),
            efficiency=resolve(self.efficiency, params),
            model=ctx.model,
        )
        return {self.outlets[0]: res.outlet}


@dataclass(frozen=True)
class _Machine(Block):
    """Shared physical kernel for `Compressor` and `Turbine`.

    Attributes:
        p_out: Outlet pressure spec.
        efficiency: Isentropic efficiency in ``(0, 1]``.
    """

    p_out: Spec = 1.0e5
    efficiency: Spec = 0.75
    _is_turbine: bool = False

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Evaluate via `fugacio.sim.units.compressor` / `turbine`."""
        feed = streams[self.inlets[0]]
        unit = _turbine_unit if self._is_turbine else _compressor_unit
        res = unit(
            feed,
            resolve(self.p_out, params),
            efficiency=resolve(self.efficiency, params),
            model=ctx.model,
        )
        return {self.outlets[0]: res.outlet}


@dataclass(frozen=True)
class Compressor(_Machine):
    """Isentropic compressor to ``p_out`` with an isentropic ``efficiency`` (< 1)."""

    _is_turbine: bool = False


@dataclass(frozen=True)
class Turbine(_Machine):
    """Isentropic turbine (expander) to ``p_out`` with an isentropic ``efficiency`` (< 1)."""

    _is_turbine: bool = True


@dataclass(frozen=True)
class Flash(Block):
    """Isothermal-isobaric two-phase flash: one inlet to vapour and liquid outlets.

    ``outlets`` must be ``(vapor_name, liquid_name)``. The common flash kernel
    determines the phase split and the connection equations retain both
    products' resolved inventories, temperatures, and pressures.

    Attributes:
        t: Drum temperature spec (literal or parameter key).
        p: Drum pressure spec (literal or parameter key).
    """

    t: Spec = 298.15
    p: Spec = 1.0e5

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Evaluate via `fugacio.sim.units.flash_drum`."""
        feed = streams[self.inlets[0]]
        vapor, liquid = _flash_drum_unit(
            feed,
            resolve(self.t, params),
            resolve(self.p, params),
            model=ctx.model,
        )
        return {self.outlets[0]: vapor, self.outlets[1]: liquid}


@dataclass(frozen=True)
class ComponentSeparator(Block):
    """Idealised separator with a per-component recovery to the top product.

    ``outlets`` must be ``(top_name, bottom_name)``. ``split_to_top`` is a
    per-component fraction sent to the top; the remainder leaves in the bottom.

    Attributes:
        split_to_top: Per-component recovery to the top (sequence/array or key).
        top_t: Optional top-product temperature (defaults to the feed's).
        top_p: Optional top-product pressure (defaults to the feed's).
        bottom_t: Optional bottom-product temperature (defaults to the feed's).
        bottom_p: Optional bottom-product pressure (defaults to the feed's).
    """

    split_to_top: Spec = 0.5
    top_t: Spec | None = None
    top_p: Spec | None = None
    bottom_t: Spec | None = None
    bottom_p: Spec | None = None

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Evaluate via `fugacio.sim.units.component_separator`."""
        from fugacio.sim.units import component_separator

        feed = streams[self.inlets[0]]
        top, bottom = component_separator(
            feed,
            resolve(self.split_to_top, params),
            top_t=None if self.top_t is None else resolve(self.top_t, params),
            top_p=None if self.top_p is None else resolve(self.top_p, params),
            bottom_t=None if self.bottom_t is None else resolve(self.bottom_t, params),
            bottom_p=None if self.bottom_p is None else resolve(self.bottom_p, params),
        )
        return {self.outlets[0]: top, self.outlets[1]: bottom}


@dataclass(frozen=True)
class HeatExchanger(Block):
    """Two-sided exchanger: a hot and a cold stream coupled by one duty.

    ``inlets`` must be ``(hot_in, cold_in)`` and ``outlets`` ``(hot_out,
    cold_out)``. The common segmented exchanger kernel determines duty and
    phase inventories from exactly one closing specification.

    Attributes:
        duty: Duty spec (W), or ``None``.
        t_hot_out: Hot outlet temperature spec (K), or ``None``.
        t_cold_out: Cold outlet temperature spec (K), or ``None``.
        ua: Size spec ``UA`` (W/K), or ``None``.
        dp_hot: Hot-side pressure drop (Pa).
        dp_cold: Cold-side pressure drop (Pa).
        flow: ``"counter"`` or ``"parallel"`` (used by the ``ua`` relation and
            the forward seed).
    """

    duty: Spec | None = None
    t_hot_out: Spec | None = None
    t_cold_out: Spec | None = None
    ua: Spec | None = None
    dp_hot: Spec = 0.0
    dp_cold: Spec = 0.0
    flow: str = "counter"

    def __post_init__(self) -> None:
        """Validate that exactly one closing specification is given."""
        given = [s for s in (self.duty, self.t_hot_out, self.t_cold_out, self.ua) if s is not None]
        if len(given) != 1:
            raise ValueError(
                "HeatExchanger requires exactly one of duty, t_hot_out, t_cold_out, ua"
            )
        if len(self.inlets) != 2 or len(self.outlets) != 2:
            raise ValueError(
                "HeatExchanger needs inlets (hot, cold) and outlets (hot_out, cold_out)"
            )
        if self.flow not in ("counter", "parallel"):
            raise ValueError("flow must be 'counter' or 'parallel'")

    def _spec_kwargs(self, params: Mapping[str, Any]) -> dict[str, Any]:
        for name in ("duty", "t_hot_out", "t_cold_out", "ua"):
            spec = getattr(self, name)
            if spec is not None:
                return {name: resolve(spec, params)}
        raise AssertionError("unreachable")  # pragma: no cover

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Evaluate via `fugacio.sim.heat_exchanger.heat_exchanger`."""
        from fugacio.sim.heat_exchanger import heat_exchanger

        res = heat_exchanger(
            streams[self.inlets[0]],
            streams[self.inlets[1]],
            dp_hot=resolve(self.dp_hot, params),
            dp_cold=resolve(self.dp_cold, params),
            flow=self.flow,
            model=ctx.model,
            **self._spec_kwargs(params),
        )
        return {self.outlets[0]: res.hot_out, self.outlets[1]: res.cold_out}


@dataclass(frozen=True)
class StoichiometricReactor(Block):
    """Fixed-conversion reactor with a formation-enthalpy energy balance.

    The reactions are given as a stoichiometric matrix ``nu`` of shape
    ``(n_reactions, n_components)`` (negative for reactants). Each reaction's
    extent is set by the fractional ``conversion`` of its ``key`` reactant, based
    on the inlet flow of that reactant, so the outlet is
    ``n_out = n_in + extent @ nu``. The energy balance uses the package's bulk
    enthalpy plus the ideal-gas standard enthalpies of formation, which carry the
    heat of reaction: the same basis as
    `fugacio.sim.reactors.stoichiometric_reactor`. Specify either the outlet
    temperature ``t_out`` or the ``duty`` (``0.0`` for adiabatic).

    Attributes:
        nu: Stoichiometric matrix (a sequence of rows or an array).
        key: Key-reactant index of every reaction.
        conversion: Fractional conversion spec(s) of the key reactants (a scalar
            for one reaction, else a sequence aligned with ``nu``).
        t_out: Outlet temperature spec, or ``None`` when ``duty`` is given.
        duty: Heat-duty spec (W, positive when added), or ``None``.
        dp: Pressure drop (Pa).
    """

    nu: Any = ()
    key: tuple[int, ...] = ()
    conversion: Any = 1.0
    t_out: Spec | None = None
    duty: Spec | None = 0.0
    dp: Spec = 0.0

    def __post_init__(self) -> None:
        """Validate the specification and stoichiometry shapes."""
        if (self.t_out is None) == (self.duty is None):
            raise ValueError("StoichiometricReactor requires exactly one of t_out or duty")
        nu = jnp.asarray(self.nu, dtype=float)
        if nu.ndim != 2:
            raise ValueError("nu must be an (n_reactions, n_components) matrix")
        if len(self.key) != nu.shape[0]:
            raise ValueError("one key-reactant index is needed per reaction")
        for r, k in enumerate(self.key):
            if not (0 <= k < nu.shape[1]) or float(nu[r, k]) >= 0.0:
                raise ValueError(f"key component {k} of reaction {r} must be a reactant (nu < 0)")

    def _extents(self, feed: Stream, params: Mapping[str, Any]) -> Array:
        nu = jnp.asarray(self.nu, dtype=float)
        conv = jnp.broadcast_to(
            jnp.asarray(
                resolve(self.conversion, params)
                if isinstance(self.conversion, str)
                else jnp.asarray(self.conversion, dtype=float)
            ),
            (nu.shape[0],),
        )
        keys = jnp.asarray(self.key)
        return conv * feed.n[keys] / (-nu[jnp.arange(nu.shape[0]), keys])

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Evaluate via `fugacio.sim.reactors.stoichiometric_reactor` (same energy basis)."""
        from fugacio.sim.reactors import stoichiometric_reactor
        from fugacio.thermo.reactions import Reaction

        feed = streams[self.inlets[0]]
        nu = jnp.asarray(self.nu, dtype=float)
        reactions = [Reaction(ctx.components, row) for row in nu]
        energy: dict[str, Any] = (
            {"t_out": resolve(self.t_out, params)}
            if self.t_out is not None
            else {"duty": resolve(self.duty, params)}  # type: ignore[arg-type]
        )
        result = stoichiometric_reactor(
            feed,
            reactions,
            extent=self._extents(feed, params),
            dp=resolve(self.dp, params),
            model=ctx.model,
            **energy,
        )
        return {self.outlets[0]: result.outlet}


@dataclass(frozen=True)
class Column(Block):
    """Rigorous MESH column embedded as a procedural block.

    ``outlets`` must be ``(distillate, bottoms)``; every inlet is a column feed
    whose stage is given by ``feed_stages`` (aligned with ``inlets``). The
    block's equations state that the outlet streams equal the products of
    `fugacio.sim.distillation.rigorous_column` at the current inlets and
    parameters. The column is converged inside every residual evaluation and
    differentiated implicitly, so the global Newton system stays small (the
    products only) while the column itself keeps its own well-conditioned
    simultaneous-correction solve.

    Attributes:
        feed_stages: Feed stage (1 = top) of every inlet.
        n_stages: Number of equilibrium stages including condenser and reboiler.
        p: Column pressure (Pa; literal or parameter key).
        specs: Column specifications as ``(kind, value)`` pairs, where ``kind``
            is one of ``"reflux_ratio"``, ``"distillate_rate"``,
            ``"bottoms_rate"``, ``"boilup_ratio"``, ``"reflux_rate"``,
            ``"condenser_duty"``, ``"reboiler_duty"`` and ``value`` is a `Spec`.
        condenser: ``"total"``, ``"partial"``, or ``None``.
        reboiler: ``"kettle"`` or ``None``.
        efficiency: Murphree vapour efficiency.
        options: Extra keyword arguments forwarded to `rigorous_column`.
    """

    feed_stages: tuple[int, ...] = ()
    n_stages: int = 10
    p: Spec = 1.0e5
    specs: tuple[tuple[str, Spec], ...] = ()
    condenser: str | None = "total"
    reboiler: str | None = "kettle"
    efficiency: Spec = 1.0
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the topology."""
        if len(self.outlets) != 2:
            raise ValueError("Column outlets must be (distillate, bottoms)")
        if len(self.feed_stages) != len(self.inlets):
            raise ValueError("one feed stage is needed per inlet")

    def _solve(self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context) -> Any:
        from fugacio.sim import distillation as dist

        feeds = [
            dist.ColumnFeed(streams[name], stage)
            for name, stage in zip(self.inlets, self.feed_stages, strict=True)
        ]
        specs = [
            dist.ColumnSpec(kind=kind, value=resolve(value, params)) for kind, value in self.specs
        ]
        return dist.rigorous_column(
            feeds,
            self.n_stages,
            p=resolve(self.p, params),
            condenser=self.condenser,
            reboiler=self.reboiler,
            specs=specs,
            efficiency=resolve(self.efficiency, params),
            model=ctx.model,
            **dict(self.options),
        )

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Evaluate via `fugacio.sim.distillation.rigorous_column`."""
        res = self._solve(streams, params, ctx)
        return {self.outlets[0]: res.distillate, self.outlets[1]: res.bottoms}
