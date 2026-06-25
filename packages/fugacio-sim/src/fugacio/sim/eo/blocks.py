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

import jax.numpy as jnp
from jax import Array, lax

from fugacio.sim.properties import _resolve
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
from fugacio.thermo import CubicEOS, component_arrays
from fugacio.thermo.eos import PR, ln_phi_mixture, molar_volume
from fugacio.thermo.equilibrium import flash_pt
from fugacio.thermo.properties import molar_enthalpy as _phase_molar_enthalpy
from fugacio.thermo.properties import molar_entropy as _phase_molar_entropy

ArrayLike = Array | float

# Threshold on the vapour fraction below/above which a stream is treated as a
# single phase for the purposes of differentiating its bulk enthalpy/entropy.
_SINGLE_PHASE_EPS = 1.0e-9

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
        eos: Cubic equation of state used for every equilibrium / property call.
        kij: Optional binary-interaction matrix for the EOS.
        scales: Residual / variable scales (see `Scales`).
    """

    components: tuple[str, ...]
    eos: CubicEOS = PR
    kij: Array | None = None
    scales: Scales = field(default_factory=Scales)

    @property
    def n_components(self) -> int:
        """Number of components (the per-stream material-balance count)."""
        return len(self.components)


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


def _bulk_molar_property(
    stream: Stream,
    ctx: Context,
    phase_property: Any,
) -> Array:
    """Bulk molar enthalpy or entropy of a stream, differentiable across phases.

    The bulk value blends the flashed vapour and liquid contributions,
    ``M = (1 - beta) M^L(x) + beta M^V(y)``. Differentiating that blend naively
    fails in a single-phase region: the *absent* phase forces a cubic root that
    does not exist there, whose derivative is ``NaN`` and which a plain
    ``jnp.where`` would still propagate (``0 * NaN``). A `jax.lax.switch` instead
    differentiates *only* the phase(s) that exist, so the gradient stays finite
    for a subcooled liquid, a superheated vapour, and a two-phase stream alike,
    while the value matches the two-phase blend exactly.
    """
    tc, pc, omega = _component_arrays(ctx.components)
    _, _, _, _, cp = _resolve(ctx.components)
    t, p, z = stream.t, stream.p, _x(stream)
    r = flash_pt(ctx.eos, t, p, z, tc, pc, omega, kij=ctx.kij)
    beta = r.beta

    def liquid(_: None) -> Array:
        return phase_property(
            t, p, r.x, tc, pc, omega, cp, eos=ctx.eos, phase="liquid", kij=ctx.kij
        )

    def vapor(_: None) -> Array:
        return phase_property(t, p, r.y, tc, pc, omega, cp, eos=ctx.eos, phase="vapor", kij=ctx.kij)

    def both(_: None) -> Array:
        return (1.0 - beta) * liquid(None) + beta * vapor(None)

    idx = jnp.where(
        beta <= _SINGLE_PHASE_EPS, 0, jnp.where(beta >= 1.0 - _SINGLE_PHASE_EPS, 2, 1)
    ).astype(jnp.int32)
    return lax.switch(idx, [liquid, both, vapor], None)


def _bulk_molar_enthalpy(stream: Stream, ctx: Context) -> Array:
    """Bulk molar enthalpy of a (possibly two-phase) stream (J/mol)."""
    return _bulk_molar_property(stream, ctx, _phase_molar_enthalpy)


def _bulk_molar_entropy(stream: Stream, ctx: Context) -> Array:
    """Bulk molar entropy of a (possibly two-phase) stream (J/mol/K)."""
    return _bulk_molar_property(stream, ctx, _phase_molar_entropy)


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
        return {self.outlets[0]: mix(ins, t=t, p=p, eos=ctx.eos, kij=ctx.kij)}

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
            rows.append(self._temperature(out, feed.t, ctx))
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
            feed, t_out=t_out, duty=duty, dp=resolve(self.dp, params), eos=ctx.eos, kij=ctx.kij
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
        out = _valve_unit(feed, resolve(self.p_out, params), eos=ctx.eos, kij=ctx.kij)
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
        arr = _component_arrays(ctx.components)
        v_l = molar_volume(
            ctx.eos, feed.t, feed.p, _x(feed), arr[0], arr[1], arr[2], phase="liquid", kij=ctx.kij
        )
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
        return {self._aux_key(): ctx.scales.temperature}

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
        return {self._aux_key(): 0.5 * (feed.t + outs[self.outlets[0]].t)}

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
            feed, resolve(self.t, params), resolve(self.p, params), eos=ctx.eos, kij=ctx.kij
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
        arr = _component_arrays(ctx.components)
        tc, pc, omega = arr[0], arr[1], arr[2]

        mat = (feed.n - vapor.n - liquid.n) / ctx.scales.flow

        x = _x(liquid)
        y = _x(vapor)
        ln_phi_l, _ = ln_phi_mixture(ctx.eos, t, p, x, tc, pc, omega, phase="liquid", kij=ctx.kij)
        ln_phi_v, _ = ln_phi_mixture(ctx.eos, t, p, y, tc, pc, omega, phase="vapor", kij=ctx.kij)
        # ln(phi_i^L x_i) - ln(phi_i^V y_i) = 0, i.e. equal component fugacities.
        equil = (ln_phi_l + jnp.log(x)) - (ln_phi_v + jnp.log(y))

        specs = jnp.concatenate(
            [
                self._temperature(vapor, t, ctx),
                self._temperature(liquid, t, ctx),
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


def _component_arrays(components: tuple[str, ...]) -> tuple[Array, Array, Array]:
    """Resolve component names to ``(tc, pc, omega)`` constant arrays.

    Built fresh on each call (not cached) so the constant arrays belong to the
    current JAX trace; caching JAX arrays would leak a tracer across the nested
    traces of `newton_system`'s forward and adjoint passes.
    """
    arr = component_arrays(list(components))
    return arr["tc"], arr["pc"], arr["omega"]
