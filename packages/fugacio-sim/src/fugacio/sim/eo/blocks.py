"""Equation-oriented unit blocks: each unit as a residual contribution.

A sequential-modular unit (`fugacio.sim.units`) is an *explicit function* of its
inlet streams: it computes outlets, internally converging any flash or
isentropic solve. An equation-oriented (EO) block is the same physics written as
*residual equations* instead. Its outlet streams are unknowns of a global system,
and the block contributes the equations that those unknowns must satisfy
(material balances, an energy balance, phase-equilibrium equifugacity, a
pressure spec, ...). The whole flowsheet, recycles and design specs included, is
then one residual system solved simultaneously by Newton's method, with the
Jacobian supplied exactly by JAX autodiff (see `fugacio.sim.eo.flowsheet`).

Every block carries two complementary methods:

* `Block.residuals` returns the block's residual vector for the EO solve. The
  residuals are *scaled* (material by a flow scale, energy by an enthalpy scale,
  pressures by a pressure scale, equifugacity left dimensionless) so the global
  Newton system is well conditioned regardless of the unit system.
* `Block.forward` evaluates the unit explicitly by reusing the corresponding
  sequential-modular unit operation. It is used to build a high-quality initial
  guess for the EO solve (a few forward sweeps) and lets a test cross-check the
  two formulations against each other.

The blocks mirror the sequential-modular units one-for-one, so the same physics
backs both engines and the EO solution must equal the sequential-modular
solution on any flowsheet both can express (the central differential test for
this layer).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.properties import Model, molar_enthalpy, molar_entropy, resolve_package
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
from fugacio.thermo import CubicEOS, PropertyPackage
from fugacio.thermo.eos import PR

ArrayLike = Array | float

#: A unit specification that is either a literal value or a string key resolved
#: against the parameter mapping passed to the solver (so an operating condition
#: can be made a differentiable parameter, or the manipulated variable of a
#: design spec, simply by naming it).
Spec = float | Array | str


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
        eos: Cubic equation of state for the default package (when ``model`` is
            unset).
        kij: Optional binary-interaction matrix for the default package.
        scales: Residual / variable scales (see `Scales`).
        model: Property package (or bare equilibrium model) used for every
            equilibrium / property call; ``None`` selects the cubic default.
    """

    components: tuple[str, ...]
    eos: CubicEOS = PR
    kij: Array | None = None
    scales: Scales = field(default_factory=Scales)
    model: Model = None
    # Local to one residual assembly. Pure-fluid enthalpy coordinates are
    # already known, including for an empty stream, and need no PH/PT round trip.
    enthalpy_coordinates: Mapping[int, Array] = field(default_factory=dict, repr=False)

    @property
    def n_components(self) -> int:
        """Number of components (the per-stream material-balance count)."""
        return len(self.components)

    @property
    def package(self) -> PropertyPackage:
        """The resolved property package every block evaluates properties with."""
        return resolve_package(self.components, self.model, eos=self.eos, kij=self.kij)


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


def _x(stream: Stream) -> Array:
    """Mole fractions guarded against a zero (empty) phase during iteration."""
    total = jnp.sum(stream.n)
    safe = jnp.where(total > 0.0, total, 1.0)
    return stream.n / safe


def _bulk_molar_enthalpy(stream: Stream, ctx: Context) -> Array:
    """Bulk molar enthalpy of a (possibly two-phase) stream (J/mol).

    The package's ``mixture_enthalpy`` blends the flashed phases with a
    `jax.lax.switch` on the phase regime, so a single-phase stream differentiates
    only the phase that exists (an absent cubic root would otherwise contribute a
    ``NaN`` derivative through ``0 * NaN``).
    """
    coordinate = ctx.enthalpy_coordinates.get(id(stream))
    return molar_enthalpy(stream, model=ctx.package) if coordinate is None else coordinate


def _bulk_molar_entropy(stream: Stream, ctx: Context) -> Array:
    """Bulk molar entropy of a (possibly two-phase) stream (J/mol/K)."""
    return molar_entropy(stream, model=ctx.package)


def _bulk_enthalpy_flow(stream: Stream, ctx: Context) -> Array:
    """Bulk total enthalpy flow of a stream (W)."""
    return jnp.sum(stream.n) * _bulk_molar_enthalpy(stream, ctx)


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

    def aux_scales(self, ctx: Context) -> dict[str, float]:
        """Auxiliary unknowns introduced by the block, mapped to their scale.

        Most blocks have none. A block with an internal implicit variable (for
        example a compressor's isentropic outlet temperature) declares it here so
        the flowsheet allocates an unknown and the block can add its defining
        equation in `residuals`.
        """
        return {}

    def n_residuals(self, ctx: Context) -> int:
        """Number of residual equations this block contributes."""
        raise NotImplementedError

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
        """Scaled residual vector for the block (length `n_residuals`)."""
        raise NotImplementedError

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

    def n_residuals(self, ctx: Context) -> int:
        """Material (per component) + pressure + energy: ``n_components + 2``."""
        return ctx.n_components + 2

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Evaluate via `fugacio.sim.units.mix`."""
        from fugacio.sim.units import mix

        ins = [streams[name] for name in self.inlets]
        t = None if self.t is None else resolve(self.t, params)
        p = None if self.p is None else resolve(self.p, params)
        return {self.outlets[0]: mix(ins, t=t, p=p, model=ctx.model, eos=ctx.eos, kij=ctx.kij)}

    def residuals(
        self,
        streams: Mapping[str, Stream],
        aux: Mapping[str, Array],
        params: Mapping[str, Any],
        ctx: Context,
    ) -> Array:
        """Material balance, pressure spec, and the (adiabatic or fixed-T) energy balance."""
        ins = [streams[name] for name in self.inlets]
        out = streams[self.outlets[0]]
        inflow = jnp.sum(jnp.stack([s.n for s in ins]), axis=0)
        mat = self._material(out, inflow, ctx)

        if self.p is None:
            p_target = jnp.min(jnp.stack([s.p for s in ins]))
        else:
            p_target = resolve(self.p, params)
        pres = self._pressure(out, p_target, ctx)

        if self.t is None:
            h_in = jnp.sum(jnp.stack([_bulk_enthalpy_flow(s, ctx) for s in ins]))
            h_out = _bulk_enthalpy_flow(out, ctx)
            energy = ((h_out - h_in) / ctx.scales.enthalpy_flow)[None]
        else:
            energy = self._temperature(out, resolve(self.t, params), ctx)
        return jnp.concatenate([mat, pres, energy])


@dataclass(frozen=True)
class Splitter(Block):
    """Flow splitter: one inlet to several outlets sharing composition and state.

    Attributes:
        fractions: Per-outlet split fractions (literal sequence/array or a
            parameter key), one per name in ``outlets``.
    """

    fractions: Spec = 1.0

    def n_residuals(self, ctx: Context) -> int:
        """``n_outlets * (n_components + 2)`` (each outlet fully defined)."""
        return len(self.outlets) * (ctx.n_components + 2)

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Evaluate via `fugacio.sim.units.splitter`."""
        from fugacio.sim.units import splitter

        feed = streams[self.inlets[0]]
        outs = splitter(feed, resolve(self.fractions, params))
        return dict(zip(self.outlets, outs, strict=True))

    def residuals(
        self,
        streams: Mapping[str, Stream],
        aux: Mapping[str, Array],
        params: Mapping[str, Any],
        ctx: Context,
    ) -> Array:
        """Each outlet carries its split fraction of the feed at the feed's T and P."""
        feed = streams[self.inlets[0]]
        fr = resolve(self.fractions, params)
        rows = []
        for i, name in enumerate(self.outlets):
            out = streams[name]
            rows.append(self._material(out, fr[i] * feed.n, ctx))
            rows.append(
                (
                    (_bulk_molar_enthalpy(out, ctx) - _bulk_molar_enthalpy(feed, ctx))
                    / ctx.scales.enthalpy_molar
                )[None]
                if ctx.n_components == 1
                else self._temperature(out, feed.t, ctx)
            )
            rows.append(self._pressure(out, feed.p, ctx))
        return jnp.concatenate(rows)


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

    def n_residuals(self, ctx: Context) -> int:
        """Material + pressure + energy: ``n_components + 2``."""
        return ctx.n_components + 2

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
            eos=ctx.eos,
            kij=ctx.kij,
        )
        return {self.outlets[0]: res.outlet}

    def residuals(
        self,
        streams: Mapping[str, Stream],
        aux: Mapping[str, Array],
        params: Mapping[str, Any],
        ctx: Context,
    ) -> Array:
        """Material, pressure drop, and the temperature or duty energy balance."""
        feed = streams[self.inlets[0]]
        out = streams[self.outlets[0]]
        mat = self._material(out, feed.n, ctx)
        pres = self._pressure(out, feed.p - resolve(self.dp, params), ctx)
        if self.t_out is not None:
            energy = self._temperature(out, resolve(self.t_out, params), ctx)
        else:
            h_in = _bulk_enthalpy_flow(feed, ctx)
            h_out = _bulk_enthalpy_flow(out, ctx)
            duty = resolve(self.duty, params)  # type: ignore[arg-type]
            energy = ((h_out - h_in - duty) / ctx.scales.enthalpy_flow)[None]
        return jnp.concatenate([mat, pres, energy])


@dataclass(frozen=True)
class Valve(Block):
    """Isenthalpic (Joule-Thomson) pressure letdown to ``p_out``.

    Attributes:
        p_out: Outlet pressure spec (literal or parameter key).
    """

    p_out: Spec = 1.0e5

    def n_residuals(self, ctx: Context) -> int:
        """Material + pressure + isenthalpic energy: ``n_components + 2``."""
        return ctx.n_components + 2

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Evaluate via `fugacio.sim.units.valve`."""
        feed = streams[self.inlets[0]]
        out = _valve_unit(
            feed, resolve(self.p_out, params), model=ctx.model, eos=ctx.eos, kij=ctx.kij
        )
        return {self.outlets[0]: out}

    def residuals(
        self,
        streams: Mapping[str, Stream],
        aux: Mapping[str, Array],
        params: Mapping[str, Any],
        ctx: Context,
    ) -> Array:
        """Material, outlet-pressure spec, and conserved molar enthalpy."""
        feed = streams[self.inlets[0]]
        out = streams[self.outlets[0]]
        mat = self._material(out, feed.n, ctx)
        pres = self._pressure(out, resolve(self.p_out, params), ctx)
        h_in = _bulk_molar_enthalpy(feed, ctx)
        h_out = _bulk_molar_enthalpy(out, ctx)
        energy = ((h_out - h_in) / ctx.scales.enthalpy_molar)[None]
        return jnp.concatenate([mat, pres, energy])


@dataclass(frozen=True)
class Pump(Block):
    """Incompressible-liquid pump to ``p_out`` with an isentropic-equivalent efficiency.

    Attributes:
        p_out: Outlet pressure spec.
        efficiency: Pump efficiency in ``(0, 1]``; the lost work heats the outlet.
    """

    p_out: Spec = 1.0e5
    efficiency: Spec = 0.75

    def n_residuals(self, ctx: Context) -> int:
        """Material + pressure + energy: ``n_components + 2``."""
        return ctx.n_components + 2

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
            eos=ctx.eos,
            kij=ctx.kij,
        )
        return {self.outlets[0]: res.outlet}

    def residuals(
        self,
        streams: Mapping[str, Stream],
        aux: Mapping[str, Array],
        params: Mapping[str, Any],
        ctx: Context,
    ) -> Array:
        """Material, outlet pressure, and the work-deposited enthalpy balance."""
        feed = streams[self.inlets[0]]
        out = streams[self.outlets[0]]
        p_out = resolve(self.p_out, params)
        eff = resolve(self.efficiency, params)
        v_l = ctx.package.volume(feed.t, feed.p, _x(feed), phase="liquid")
        w_actual = v_l * (p_out - feed.p) / eff
        mat = self._material(out, feed.n, ctx)
        pres = self._pressure(out, p_out, ctx)
        h_in = _bulk_molar_enthalpy(feed, ctx)
        h_out = _bulk_molar_enthalpy(out, ctx)
        energy = ((h_out - h_in - w_actual) / ctx.scales.enthalpy_molar)[None]
        return jnp.concatenate([mat, pres, energy])


@dataclass(frozen=True)
class _Machine(Block):
    """Shared isentropic-machine block for `Compressor` and `Turbine`.

    Introduces one auxiliary unknown, the isentropic outlet temperature
    ``t_iso``, defined by the constant-entropy relation ``s(t_iso, p_out) =
    s_in``. The real outlet enthalpy is ``h_in + w_actual`` with ``w_actual``
    derived from the isentropic work ``w_ideal = h(t_iso, p_out) - h_in`` and the
    efficiency.

    Attributes:
        p_out: Outlet pressure spec.
        efficiency: Isentropic efficiency in ``(0, 1]``.
    """

    p_out: Spec = 1.0e5
    efficiency: Spec = 0.75
    _is_turbine: bool = False

    def _aux_key(self) -> str:
        return f"{self.outlets[0]}::t_iso"

    def aux_scales(self, ctx: Context) -> dict[str, float]:
        """One auxiliary unknown: the isentropic outlet temperature."""
        return {
            self._aux_key(): ctx.scales.enthalpy_molar
            if ctx.n_components == 1
            else ctx.scales.temperature
        }

    def n_residuals(self, ctx: Context) -> int:
        """Material + pressure + entropy(aux) + energy: ``n_components + 3``."""
        return ctx.n_components + 3

    def aux_init(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Array]:
        """Seed ``t_iso`` from the forward isentropic solve for a tight initial guess."""
        feed = streams[self.inlets[0]]
        outs = self.forward(streams, params, ctx)
        # The forward outlet temperature is a good (slightly high) seed; the feed
        # temperature is an even safer interior seed for the isentropic state.
        return {
            self._aux_key(): _bulk_molar_enthalpy(feed, ctx)
            if ctx.n_components == 1
            else 0.5 * (feed.t + outs[self.outlets[0]].t)
        }

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
            eos=ctx.eos,
            kij=ctx.kij,
        )
        return {self.outlets[0]: res.outlet}

    def residuals(
        self,
        streams: Mapping[str, Stream],
        aux: Mapping[str, Array],
        params: Mapping[str, Any],
        ctx: Context,
    ) -> Array:
        """Material, pressure, the isentropic-state entropy match, and the work balance."""
        feed = streams[self.inlets[0]]
        out = streams[self.outlets[0]]
        p_out = resolve(self.p_out, params)
        eff = resolve(self.efficiency, params)
        t_iso = aux[self._aux_key()]

        mat = self._material(out, feed.n, ctx)
        pres = self._pressure(out, p_out, ctx)

        if ctx.n_components == 1:
            r_iso = ctx.package.flash_ph(p_out, t_iso, jnp.ones(1))
            iso = Stream(feed.n, r_iso.t, p_out, ctx.components, r_iso.beta * feed.n)
        else:
            iso = Stream(n=feed.n, t=t_iso, p=p_out, components=ctx.components)
        s_in = _bulk_molar_entropy(feed, ctx)
        s_iso = _bulk_molar_entropy(iso, ctx)
        entropy = ((s_iso - s_in) / ctx.scales.entropy_molar)[None]

        h_in = _bulk_molar_enthalpy(feed, ctx)
        h_iso = _bulk_molar_enthalpy(iso, ctx)
        w_ideal = h_iso - h_in
        w_actual = eff * w_ideal if self._is_turbine else w_ideal / eff
        h_out = _bulk_molar_enthalpy(out, ctx)
        energy = ((h_out - h_in - w_actual) / ctx.scales.enthalpy_molar)[None]
        return jnp.concatenate([mat, pres, entropy, energy])


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

    ``outlets`` must be ``(vapor_name, liquid_name)``. The block contributes the
    component material balances, the equifugacity equilibrium relations
    (``phi_i^L x_i = phi_i^V y_i``), and the temperature/pressure specs on both
    product streams, the same equations the sequential-modular `flash_drum`
    converges internally.

    Attributes:
        t: Drum temperature spec (literal or parameter key).
        p: Drum pressure spec (literal or parameter key).
    """

    t: Spec = 298.15
    p: Spec = 1.0e5

    def n_residuals(self, ctx: Context) -> int:
        """Material + equilibrium + 2 T-specs + 2 P-specs: ``2 * (n_components + 2)``."""
        return 2 * (ctx.n_components + 2)

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
            eos=ctx.eos,
            kij=ctx.kij,
        )
        return {self.outlets[0]: vapor, self.outlets[1]: liquid}

    def residuals(
        self,
        streams: Mapping[str, Stream],
        aux: Mapping[str, Array],
        params: Mapping[str, Any],
        ctx: Context,
    ) -> Array:
        """Equifugacity + material balance + temperature/pressure specs on both phases."""
        feed = streams[self.inlets[0]]
        vapor = streams[self.outlets[0]]
        liquid = streams[self.outlets[1]]
        t = resolve(self.t, params)
        p = resolve(self.p, params)
        pkg = ctx.package

        mat = (feed.n - vapor.n - liquid.n) / ctx.scales.flow

        regime = pkg.flash_pt(t, p, _x(feed)).beta

        def two_phase(_: None) -> Array:
            x = _x(liquid)
            y = _x(vapor)
            ln_phi_l = pkg.ln_phi(t, p, x, phase="liquid")
            ln_phi_v = pkg.ln_phi(t, p, y, phase="vapor")
            return (ln_phi_l + jnp.log(x)) - (ln_phi_v + jnp.log(y))

        # Equifugacity is an equality only when both phases exist. In a
        # single-phase region the absent product has zero component flows.
        index = jnp.where(regime <= 1e-9, 0, jnp.where(regime >= 1 - 1e-9, 2, 1)).astype(jnp.int32)
        equil = jax.lax.switch(
            index,
            [lambda _: vapor.n / ctx.scales.flow, two_phase, lambda _: liquid.n / ctx.scales.flow],
            None,
        )

        def thermal(stream: Stream, phase: str) -> Array:
            if ctx.n_components == 1:
                target = pkg.enthalpy(t, p, jnp.ones(1), phase=phase)
                return ((_bulk_molar_enthalpy(stream, ctx) - target) / ctx.scales.enthalpy_molar)[
                    None
                ]
            return self._temperature(stream, t, ctx)

        specs = jnp.concatenate(
            [
                thermal(vapor, "vapor"),
                thermal(liquid, "liquid"),
                self._pressure(vapor, p, ctx),
                self._pressure(liquid, p, ctx),
            ]
        )
        return jnp.concatenate([mat, equil, specs])


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

    def n_residuals(self, ctx: Context) -> int:
        """Two fully-defined outlets: ``2 * (n_components + 2)``."""
        return 2 * (ctx.n_components + 2)

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

    def residuals(
        self,
        streams: Mapping[str, Stream],
        aux: Mapping[str, Array],
        params: Mapping[str, Any],
        ctx: Context,
    ) -> Array:
        """Per-component split to the top/bottom with their temperature/pressure specs."""
        feed = streams[self.inlets[0]]
        top = streams[self.outlets[0]]
        bottom = streams[self.outlets[1]]
        frac = resolve(self.split_to_top, params)
        top_t = feed.t if self.top_t is None else resolve(self.top_t, params)
        top_p = feed.p if self.top_p is None else resolve(self.top_p, params)
        bot_t = feed.t if self.bottom_t is None else resolve(self.bottom_t, params)
        bot_p = feed.p if self.bottom_p is None else resolve(self.bottom_p, params)
        return jnp.concatenate(
            [
                self._material(top, frac * feed.n, ctx),
                self._temperature(top, top_t, ctx),
                self._pressure(top, top_p, ctx),
                self._material(bottom, (1.0 - frac) * feed.n, ctx),
                self._temperature(bottom, bot_t, ctx),
                self._pressure(bottom, bot_p, ctx),
            ]
        )


@dataclass(frozen=True)
class HeatExchanger(Block):
    """Two-sided exchanger: a hot and a cold stream coupled by one duty.

    ``inlets`` must be ``(hot_in, cold_in)`` and ``outlets`` ``(hot_out,
    cold_out)``. The block introduces one auxiliary unknown, the duty ``q``
    (W, hot to cold), and contributes both material balances, both pressure
    drops, the two energy balances that define ``q``, and one closing
    specification: the duty itself, either outlet temperature, or the rating
    relation ``q = UA * LMTD`` with the end approaches of a counter-current
    (default) or parallel arrangement.

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

    def _aux_key(self) -> str:
        return f"{self.outlets[0]}::q"

    def aux_scales(self, ctx: Context) -> dict[str, float]:
        """One auxiliary unknown: the duty."""
        return {self._aux_key(): ctx.scales.enthalpy_flow}

    def n_residuals(self, ctx: Context) -> int:
        """Two materials, two pressures, two energy balances, one spec: ``2 n_components + 5``."""
        return 2 * ctx.n_components + 5

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
            eos=ctx.eos,
            kij=ctx.kij,
            **self._spec_kwargs(params),
        )
        return {self.outlets[0]: res.hot_out, self.outlets[1]: res.cold_out}

    def aux_init(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Array]:
        """Seed the duty from the forward solution."""
        outs = self.forward(streams, params, ctx)
        hot_in = streams[self.inlets[0]]
        q = _bulk_enthalpy_flow(hot_in, ctx) - _bulk_enthalpy_flow(outs[self.outlets[0]], ctx)
        return {self._aux_key(): q}

    def residuals(
        self,
        streams: Mapping[str, Stream],
        aux: Mapping[str, Array],
        params: Mapping[str, Any],
        ctx: Context,
    ) -> Array:
        """Materials, pressure drops, the two energy balances, and the closing spec."""
        from fugacio.sim.economics import lmtd

        hot_in, cold_in = streams[self.inlets[0]], streams[self.inlets[1]]
        hot_out, cold_out = streams[self.outlets[0]], streams[self.outlets[1]]
        q = aux[self._aux_key()]
        sc = ctx.scales
        mat = jnp.concatenate(
            [self._material(hot_out, hot_in.n, ctx), self._material(cold_out, cold_in.n, ctx)]
        )
        pres = jnp.concatenate(
            [
                self._pressure(hot_out, hot_in.p - resolve(self.dp_hot, params), ctx),
                self._pressure(cold_out, cold_in.p - resolve(self.dp_cold, params), ctx),
            ]
        )
        energy = jnp.stack(
            [
                (_bulk_enthalpy_flow(hot_in, ctx) - _bulk_enthalpy_flow(hot_out, ctx) - q)
                / sc.enthalpy_flow,
                (_bulk_enthalpy_flow(cold_out, ctx) - _bulk_enthalpy_flow(cold_in, ctx) - q)
                / sc.enthalpy_flow,
            ]
        )
        if self.duty is not None:
            spec = (q - resolve(self.duty, params)) / sc.enthalpy_flow
        elif self.t_hot_out is not None:
            spec = (hot_out.t - resolve(self.t_hot_out, params)) / sc.temperature
        elif self.t_cold_out is not None:
            spec = (cold_out.t - resolve(self.t_cold_out, params)) / sc.temperature
        else:
            if self.flow == "counter":
                dt1, dt2 = hot_in.t - cold_out.t, hot_out.t - cold_in.t
            else:
                dt1, dt2 = hot_in.t - cold_in.t, hot_out.t - cold_out.t
            ua = resolve(self.ua, params)  # type: ignore[arg-type]
            spec = (q - ua * lmtd(dt1, dt2)) / sc.enthalpy_flow
        return jnp.concatenate([mat, pres, energy, jnp.reshape(spec, (1,))])


@dataclass(frozen=True)
class StoichiometricReactor(Block):
    """Fixed-conversion reactor on an ideal-gas (formation-based) energy balance.

    The reactions are given as a stoichiometric matrix ``nu`` of shape
    ``(n_reactions, n_components)`` (negative for reactants). Each reaction's
    extent is set by the fractional ``conversion`` of its ``key`` reactant, based
    on the inlet flow of that reactant, so the outlet is
    ``n_out = n_in + extent @ nu``. The energy balance uses absolute ideal-gas
    enthalpies (standard enthalpy of formation plus the ideal-gas heat-capacity
    integral, the same basis as `fugacio.sim.reactors.stoichiometric_reactor`),
    which is what carries the heat of reaction; specify either the outlet
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

    def n_residuals(self, ctx: Context) -> int:
        """Material + pressure + energy: ``n_components + 2``."""
        return ctx.n_components + 2

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

    def _absolute_enthalpy(self, stream: Stream, ctx: Context) -> Array:
        from fugacio.thermo.ideal import enthalpy_ig
        from fugacio.thermo.reactions import reaction_arrays

        hf, _gf, (a, b, c, d, e) = reaction_arrays(list(ctx.components))
        return jnp.sum(stream.n * (hf + enthalpy_ig(stream.t, a, b, c, d, e)))

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Explicit evaluation: stoichiometric outlet plus the isothermal or adiabatic balance."""
        from fugacio.thermo.implicit import bracketed_root

        feed = streams[self.inlets[0]]
        nu = jnp.asarray(self.nu, dtype=float)
        n_out = feed.n + self._extents(feed, params) @ nu
        p_out = feed.p - resolve(self.dp, params)
        if self.t_out is not None:
            t_out = resolve(self.t_out, params)
        else:
            h_target = self._absolute_enthalpy(feed, ctx) + resolve(self.duty, params)  # type: ignore[arg-type]

            def residual(t: Array, theta: tuple[Array, Array]) -> Array:
                n, h = theta
                probe = Stream(n=n, t=t, p=p_out, components=ctx.components)
                return (self._absolute_enthalpy(probe, ctx) - h) / ctx.scales.enthalpy_flow

            t_out = bracketed_root(
                residual, (n_out, h_target), jnp.asarray(200.0), jnp.asarray(4000.0), 1e-10, 200
            )
        return {self.outlets[0]: Stream(n=n_out, t=t_out, p=p_out, components=ctx.components)}

    def residuals(
        self,
        streams: Mapping[str, Stream],
        aux: Mapping[str, Array],
        params: Mapping[str, Any],
        ctx: Context,
    ) -> Array:
        """Stoichiometric material balance, pressure drop, and the energy balance."""
        feed = streams[self.inlets[0]]
        out = streams[self.outlets[0]]
        nu = jnp.asarray(self.nu, dtype=float)
        mat = self._material(out, feed.n + self._extents(feed, params) @ nu, ctx)
        pres = self._pressure(out, feed.p - resolve(self.dp, params), ctx)
        if self.t_out is not None:
            energy = self._temperature(out, resolve(self.t_out, params), ctx)
        else:
            h_in = self._absolute_enthalpy(feed, ctx)
            h_out = self._absolute_enthalpy(out, ctx)
            duty = resolve(self.duty, params)  # type: ignore[arg-type]
            energy = ((h_out - h_in - duty) / ctx.scales.enthalpy_flow)[None]
        return jnp.concatenate([mat, pres, energy])


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

    def n_residuals(self, ctx: Context) -> int:
        """Two fully defined products: ``2 * (n_components + 2)``."""
        return 2 * (ctx.n_components + 2)

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
            eos=ctx.eos,
            kij=ctx.kij,
            **dict(self.options),
        )

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Evaluate via `fugacio.sim.distillation.rigorous_column`."""
        res = self._solve(streams, params, ctx)
        return {self.outlets[0]: res.distillate, self.outlets[1]: res.bottoms}

    def residuals(
        self,
        streams: Mapping[str, Stream],
        aux: Mapping[str, Array],
        params: Mapping[str, Any],
        ctx: Context,
    ) -> Array:
        """Outlet streams equal the converged column products."""
        res = self._solve(streams, params, ctx)
        top, bottom = streams[self.outlets[0]], streams[self.outlets[1]]
        return jnp.concatenate(
            [
                self._material(top, res.distillate.n, ctx),
                self._temperature(top, res.distillate.t, ctx),
                self._pressure(top, res.distillate.p, ctx),
                self._material(bottom, res.bottoms.n, ctx),
                self._temperature(bottom, res.bottoms.t, ctx),
                self._pressure(bottom, res.bottoms.p, ctx),
            ]
        )
