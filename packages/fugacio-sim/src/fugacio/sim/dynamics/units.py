"""Dynamic unit operations: holdups carried as ODE states.

A steady-state unit (in :mod:`fugacio.sim.units`) maps inlet streams to outlet
streams instantaneously. A *dynamic* unit has memory -- inventory of material and
energy -- so its outlets depend on accumulated holdup, and its behaviour is an
ODE. Every model here follows the same recipe, which keeps the differentiable
index at one and reuses the existing thermodynamics:

* the **state** is a conserved inventory (component mole holdups, and an energy
  state where temperature moves): these are exactly the quantities with a clean
  balance ``d(holdup)/dt = in - out + generation``;
* the **constitutive relations** (phase split, density, reaction rate, pressure)
  are evaluated *instantaneously* from the current holdup using the steady-state
  kernels in :mod:`fugacio.thermo`, so a dynamic flash reuses :func:`flash_pt`, a
  dynamic reactor reuses the reaction thermochemistry and rate laws, and so on.

Each unit exposes :meth:`DynamicUnit.initial_state` and
:meth:`DynamicUnit.evaluate`; the latter returns the state derivative, the outlet
streams, and a dictionary of measurements (level, temperature, pressure,
composition) that controllers read. Manipulated variables (valve openings, duties,
jacket temperatures, product draws) are supplied per-call through a ``controls``
mapping, defaulting to each unit's configured value. The whole thing is
:mod:`jax.numpy`, so a dynamic flowsheet assembled from these units integrates and
differentiates as one system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import Array

from fugacio.sim.properties import _resolve
from fugacio.sim.stream import Stream
from fugacio.thermo import (
    PR,
    CubicEOS,
    flash_pt,
    liquid_density,
)
from fugacio.thermo.constants import R
from fugacio.thermo.ideal import cp_ig, enthalpy_ig
from fugacio.thermo.reactions import Reaction, delta_h_rxn, reaction_arrays

ArrayLike = Array | float
Controls = Mapping[str, Array]


class UnitStep(NamedTuple):
    """What a dynamic unit returns from :meth:`DynamicUnit.evaluate`.

    Attributes:
        dstate: Time derivative of the unit state (same pytree structure as state).
        outlets: Outlet streams, in the unit's documented order.
        measurements: Named scalar measurements (level, temperature, pressure,
            composition, ...) exposed to controllers and reporting.
    """

    dstate: Any
    outlets: tuple[Stream, ...]
    measurements: dict[str, Array]


def _warm_components(components: tuple[str, ...]) -> None:
    """Populate the cached component-constant table eagerly, at construction time.

    :func:`fugacio.sim.properties._resolve` memoizes its (static) physical-constant
    arrays. If its first call happened *inside* a traced integration the cache would
    capture tracers and leak; warming it here -- when the unit is built, outside any
    trace -- guarantees the cached arrays are plain constants by the time the dynamic
    solver runs.
    """
    _resolve(components)


def _control(controls: Controls | None, key: str, default: ArrayLike) -> Array:
    """Read a manipulated variable from ``controls`` (or fall back to ``default``)."""
    if controls is not None and key in controls:
        return jnp.asarray(controls[key])
    return jnp.asarray(default)


def _sum_inlets(inlets: Sequence[Stream], n_components: int) -> Array:
    """Total component molar inflow (mol/s) over the inlet streams."""
    if not inlets:
        return jnp.zeros((n_components,))
    return jnp.sum(jnp.stack([s.n for s in inlets]), axis=0)


class DynamicUnit(ABC):
    """Abstract base class for a dynamic unit operation."""

    name: str

    @abstractmethod
    def initial_state(self) -> Any:
        """Return the initial state pytree for this unit."""

    @abstractmethod
    def evaluate(
        self,
        state: Any,
        inlets: tuple[Stream, ...],
        controls: Controls | None = None,
        theta: Any = None,
    ) -> UnitStep:
        """Return the state derivative, outlet streams and measurements at ``state``."""

    @abstractmethod
    def measured(self, state: Any) -> dict[str, Array]:
        """State-only measurements (level, T, P, composition) available to controllers.

        These depend on the unit state alone -- not on the inlet streams -- so a
        controller can read them before the units are advanced, which keeps the
        instantaneous control algebra explicit (see :class:`DynamicFlowsheet`).
        """


# --------------------------------------------------------------------------- #
# Liquid surge / level tank (mass + composition dynamics)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LevelTank(DynamicUnit):
    """An isothermal, well-mixed liquid surge tank with level and composition dynamics.

    State is the per-component liquid holdup ``N`` (mol). The liquid leaves either
    at a commanded molar flow (``outlet="pump"``, manipulated variable ``"flow"``)
    or through a valve whose discharge follows a Torricelli law
    (``outlet="valve"``, manipulated variable ``"opening"`` in ``[0, 1]``); the
    outlet always carries the well-mixed holdup composition. The liquid level is
    inferred from the holdup volume and the tank cross-sectional ``area`` using the
    real liquid density.

    Manipulated variables: ``"flow"`` (mol/s, pump mode) or ``"opening"`` (-, valve
    mode). Measurements: ``level``, ``volume``, ``holdup``, ``outlet_flow``.
    """

    name: str
    components: tuple[str, ...]
    area: ArrayLike = 1.0
    t: ArrayLike = 298.15
    p: ArrayLike = 101325.0
    outlet: str = "pump"
    valve_cv: ArrayLike = 1.0
    flow_setpoint: ArrayLike = 0.0
    n0: Array | None = field(default=None)

    def __post_init__(self) -> None:
        _warm_components(self.components)

    def initial_state(self) -> Array:
        if self.n0 is not None:
            return jnp.asarray(self.n0, dtype=float)
        return jnp.ones((len(self.components),))

    def _molar_volume(self, n: Array) -> Array:
        _, _, _, mw, _ = _resolve(self.components)
        z = n / jnp.sum(n)
        rho_mass = liquid_density(list(self.components), jnp.asarray(self.t), z)
        molar_mass = jnp.sum(z * mw) * 1e-3
        return molar_mass / rho_mass

    def measured(self, state: Array) -> dict[str, Array]:
        n = jnp.clip(state, 0.0, None)
        total = jnp.sum(n) + 1e-12
        volume = total * self._molar_volume(n)
        return {"level": volume / jnp.asarray(self.area), "volume": volume, "holdup": total}

    def evaluate(
        self,
        state: Array,
        inlets: tuple[Stream, ...],
        controls: Controls | None = None,
        theta: Any = None,
    ) -> UnitStep:
        n = jnp.clip(state, 0.0, None)
        total = jnp.sum(n) + 1e-12
        z = n / total
        v_molar = self._molar_volume(n)
        volume = total * v_molar
        level = volume / jnp.asarray(self.area)

        if self.outlet == "pump":
            f_out = _control(controls, "flow", self.flow_setpoint)
        else:
            opening = jnp.clip(_control(controls, "opening", 0.0), 0.0, 1.0)
            vol_out = jnp.asarray(self.valve_cv) * opening * jnp.sqrt(jnp.clip(level, 0.0, None))
            f_out = vol_out / v_molar
        f_out = jnp.clip(f_out, 0.0, None)

        inflow = _sum_inlets(inlets, len(self.components))
        dn = inflow - f_out * z
        liquid = Stream(
            n=f_out * z, t=jnp.asarray(self.t), p=jnp.asarray(self.p), components=self.components
        )
        meas = {
            "level": level,
            "volume": volume,
            "holdup": total,
            "outlet_flow": f_out,
        }
        return UnitStep(dstate=dn, outlets=(liquid,), measurements=meas)


# --------------------------------------------------------------------------- #
# Constant-holdup blending tank (composition dynamics)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MixingTank(DynamicUnit):
    """A constant-holdup, perfectly mixed blending tank (composition first-order lag).

    With a fixed molar holdup ``holdup`` and equal in/out flow, the outlet
    composition relaxes toward the (instantaneous) inlet composition with a time
    constant equal to the residence time ``holdup / inlet_flow``. State is the
    mole-fraction vector ``x``.

    Measurements: ``residence_time``, ``inlet_flow``, ``x0`` ... (per-component
    fractions are also returned under ``frac``).
    """

    name: str
    components: tuple[str, ...]
    holdup: ArrayLike = 1.0
    t: ArrayLike = 298.15
    p: ArrayLike = 101325.0
    x0: Array | None = field(default=None)

    def initial_state(self) -> Array:
        if self.x0 is not None:
            return jnp.asarray(self.x0, dtype=float)
        return jnp.full((len(self.components),), 1.0 / len(self.components))

    def measured(self, state: Array) -> dict[str, Array]:
        return {"frac": state}

    def evaluate(
        self,
        state: Array,
        inlets: tuple[Stream, ...],
        controls: Controls | None = None,
        theta: Any = None,
    ) -> UnitStep:
        x = state
        inflow = _sum_inlets(inlets, len(self.components))
        f_in = jnp.sum(inflow) + 1e-12
        z_in = inflow / f_in
        dx = (f_in / jnp.asarray(self.holdup)) * (z_in - x)
        outlet = Stream(
            n=f_in * x, t=jnp.asarray(self.t), p=jnp.asarray(self.p), components=self.components
        )
        meas = {
            "residence_time": jnp.asarray(self.holdup) / f_in,
            "inlet_flow": f_in,
            "frac": x,
        }
        return UnitStep(dstate=dx, outlets=(outlet,), measurements=meas)


# --------------------------------------------------------------------------- #
# Stirred thermal mass (temperature dynamics)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ThermalMass(DynamicUnit):
    """A heated/cooled, well-mixed stirred tank: lumped temperature dynamics.

    A fixed molar ``holdup`` of liquid is heated by a duty ``Q`` (manipulated
    variable ``"duty"``, W), exchanges sensible heat with the inlet flow, and loses
    heat to ambient through ``ua`` (W/K). State is the bulk temperature ``T``;
    the outlet carries the inlet flows at ``T``. Heat capacities come from the
    ideal-gas correlations of :mod:`fugacio.thermo`.

    Manipulated variable: ``"duty"`` (W). Measurements: ``temperature``, ``duty``.
    """

    name: str
    components: tuple[str, ...]
    holdup: ArrayLike = 100.0
    ua: ArrayLike = 0.0
    t_ambient: ArrayLike = 298.15
    p: ArrayLike = 101325.0
    duty_setpoint: ArrayLike = 0.0
    t0: ArrayLike = 298.15

    def __post_init__(self) -> None:
        _warm_components(self.components)

    def initial_state(self) -> Array:
        return jnp.asarray(self.t0, dtype=float)

    def measured(self, state: Array) -> dict[str, Array]:
        return {"temperature": state}

    def evaluate(
        self,
        state: Array,
        inlets: tuple[Stream, ...],
        controls: Controls | None = None,
        theta: Any = None,
    ) -> UnitStep:
        temp = state
        _, _, _, _, cp = _resolve(self.components)
        a, b, c, d, e = cp
        inflow = _sum_inlets(inlets, len(self.components))
        total_in = jnp.sum(inflow) + 1e-12
        z = jnp.where(
            jnp.sum(inflow) > 0,
            inflow / total_in,
            jnp.full((len(self.components),), 1.0 / len(self.components)),
        )
        # Sensible enthalpy carried in by each inlet relative to the bulk temperature.
        h_in = jnp.asarray(0.0)
        for s in inlets:
            dh = enthalpy_ig(s.t, a, b, c, d, e) - enthalpy_ig(temp, a, b, c, d, e)
            h_in = h_in + jnp.sum(s.n * dh)
        q = _control(controls, "duty", self.duty_setpoint)
        c_th = jnp.asarray(self.holdup) * jnp.sum(z * cp_ig(temp, a, b, c, d, e))
        loss = jnp.asarray(self.ua) * (temp - jnp.asarray(self.t_ambient))
        dt = (h_in + q - loss) / c_th
        outlet = Stream(n=inflow, t=temp, p=jnp.asarray(self.p), components=self.components)
        return UnitStep(dstate=dt, outlets=(outlet,), measurements={"temperature": temp, "duty": q})


# --------------------------------------------------------------------------- #
# Continuous stirred-tank reactor (composition + temperature dynamics)
# --------------------------------------------------------------------------- #
class CSTRState(NamedTuple):
    """Dynamic CSTR state: concentrations (mol/m^3) and temperature (K)."""

    c: Array
    t: Array


@dataclass(frozen=True)
class DynamicCSTR(DynamicUnit):
    """A non-isothermal, constant-volume liquid CSTR -- the canonical control plant.

    Works in concentration space: state is the vector of species concentrations
    ``C`` (mol/m^3) and the reactor temperature ``T``. The reactor has volume
    ``volume`` and a constant volumetric throughput ``q``, so the residence time is
    ``volume / q``. The energy balance carries the heat of reaction (from
    :func:`fugacio.thermo.delta_h_rxn`) and a jacket term ``UA (T_jacket - T)`` with
    a constant volumetric heat capacity ``rho_cp`` (J/m^3/K). Exothermic operation
    reproduces the classic ignition/extinction and limit-cycle behaviour, which is
    why this is *the* reactor-control benchmark.

    Manipulated variables: ``"jacket_t"`` (K) and ``"q"`` (m^3/s). Measurements:
    ``temperature``, ``concentration``, ``heat_release``.
    """

    name: str
    components: tuple[str, ...]
    reactions: Reaction | Sequence[Reaction]
    rate_laws: Any
    volume: ArrayLike = 1.0
    q: ArrayLike = 1.0
    rho_cp: ArrayLike = 4.0e6
    ua: ArrayLike = 0.0
    jacket_t: ArrayLike = 298.15
    p: ArrayLike = 101325.0
    c0: Array | None = field(default=None)
    t0: ArrayLike = 298.15

    def _reactions(self) -> list[Reaction]:
        rxns = self.reactions
        return [rxns] if isinstance(rxns, Reaction) else list(rxns)

    def _nu(self) -> Array:
        return jnp.stack([jnp.asarray(r.nu) for r in self._reactions()])

    def initial_state(self) -> CSTRState:
        c0 = (
            jnp.zeros((len(self.components),))
            if self.c0 is None
            else jnp.asarray(self.c0, dtype=float)
        )
        return CSTRState(c=c0, t=jnp.asarray(self.t0, dtype=float))

    def measured(self, state: CSTRState) -> dict[str, Array]:
        return {"temperature": state.t, "concentration": state.c}

    def evaluate(
        self,
        state: CSTRState,
        inlets: tuple[Stream, ...],
        controls: Controls | None = None,
        theta: Any = None,
    ) -> UnitStep:
        nu = self._nu()
        laws = (
            list(self.rate_laws) if isinstance(self.rate_laws, (list, tuple)) else [self.rate_laws]
        )
        hf, _gf, coeffs = reaction_arrays(list(self.components))
        a, b, cc, d, e = coeffs
        q = _control(controls, "q", self.q)
        jacket_t = _control(controls, "jacket_t", self.jacket_t)
        v = jnp.asarray(self.volume)

        inflow = _sum_inlets(inlets, len(self.components))
        c_in = inflow / q
        t_in = inlets[0].t if inlets else state.t

        conc = jnp.clip(state.c, 0.0, None)
        rates = jnp.stack([law.rate(state.t, conc) for law in laws])
        gen = rates @ nu
        dc = (q / v) * (c_in - state.c) + gen

        dh = jnp.stack(
            [delta_h_rxn(nu[j], state.t, hf, a, b, cc, d, e) for j in range(nu.shape[0])]
        )
        heat_release = -v * jnp.sum(rates * dh)
        jacket = jnp.asarray(self.ua) * (jacket_t - state.t)
        c_th = jnp.asarray(self.rho_cp) * v
        dtemp = (q / v) * (t_in - state.t) + (heat_release + jacket) / c_th

        outlet = Stream(n=state.c * q, t=state.t, p=jnp.asarray(self.p), components=self.components)
        meas = {
            "temperature": state.t,
            "concentration": state.c,
            "heat_release": heat_release,
        }
        return UnitStep(dstate=CSTRState(c=dc, t=dtemp), outlets=(outlet,), measurements=meas)


# --------------------------------------------------------------------------- #
# Gas receiver / surge drum (pressure dynamics)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GasReceiver(DynamicUnit):
    """An isothermal gas surge drum: pressure builds and decays with the inventory.

    State is the component mole holdup ``N`` of gas in a fixed ``volume``; the
    pressure follows the ideal-gas law ``P = (sum N) R T / V``. Gas leaves either
    at a commanded flow (``outlet="flow"``, manipulated variable ``"flow"``) or
    through a valve to a downstream pressure ``p_down`` (``outlet="valve"``,
    manipulated variable ``"opening"``), giving the textbook pressure-control plant.

    Manipulated variables: ``"flow"`` (mol/s) or ``"opening"`` (-). Measurements:
    ``pressure``, ``holdup``, ``outlet_flow``.
    """

    name: str
    components: tuple[str, ...]
    volume: ArrayLike = 1.0
    t: ArrayLike = 298.15
    outlet: str = "valve"
    valve_cv: ArrayLike = 1.0e-3
    p_down: ArrayLike = 101325.0
    flow_setpoint: ArrayLike = 0.0
    n0: Array | None = field(default=None)

    def initial_state(self) -> Array:
        if self.n0 is not None:
            return jnp.asarray(self.n0, dtype=float)
        # Default: a small inventory at roughly atmospheric pressure.
        n_total = 101325.0 * float(self.volume) / (float(R) * float(self.t))
        return jnp.full((len(self.components),), n_total / len(self.components))

    def measured(self, state: Array) -> dict[str, Array]:
        n = jnp.clip(state, 0.0, None)
        total = jnp.sum(n) + 1e-12
        return {
            "pressure": total * R * jnp.asarray(self.t) / jnp.asarray(self.volume),
            "holdup": total,
        }

    def evaluate(
        self,
        state: Array,
        inlets: tuple[Stream, ...],
        controls: Controls | None = None,
        theta: Any = None,
    ) -> UnitStep:
        n = jnp.clip(state, 0.0, None)
        total = jnp.sum(n) + 1e-12
        y = n / total
        pressure = total * R * jnp.asarray(self.t) / jnp.asarray(self.volume)

        if self.outlet == "flow":
            f_out = _control(controls, "flow", self.flow_setpoint)
        else:
            opening = jnp.clip(_control(controls, "opening", 0.0), 0.0, 1.0)
            dp = jnp.clip(pressure - jnp.asarray(self.p_down), 0.0, None)
            f_out = jnp.asarray(self.valve_cv) * opening * jnp.sqrt(dp)
        f_out = jnp.clip(f_out, 0.0, None)

        inflow = _sum_inlets(inlets, len(self.components))
        dn = inflow - f_out * y
        outlet = Stream(n=f_out * y, t=jnp.asarray(self.t), p=pressure, components=self.components)
        meas = {"pressure": pressure, "holdup": total, "outlet_flow": f_out}
        return UnitStep(dstate=dn, outlets=(outlet,), measurements=meas)


# --------------------------------------------------------------------------- #
# Dynamic flash drum (separation dynamics)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DynamicFlash(DynamicUnit):
    """An isothermal-isobaric flash drum with liquid holdup and equilibrium vapor.

    State is the per-component liquid holdup ``M`` (mol). The vapour drawn off is in
    instantaneous phase equilibrium with the well-mixed holdup -- its composition is
    the equilibrium vapour of a :func:`flash_pt` on the holdup at ``(T, P)`` -- and
    leaves at a commanded rate; the liquid product leaves at the holdup composition.
    This captures the composition response of a separator to feed and draw
    disturbances while reusing the rigorous EOS equilibrium.

    Manipulated variables: ``"vapor_draw"`` (mol/s), ``"liquid_draw"`` (mol/s).
    Measurements: ``holdup``, ``x`` (liquid composition), ``y`` (vapour composition).
    """

    name: str
    components: tuple[str, ...]
    t: ArrayLike = 298.15
    p: ArrayLike = 101325.0
    eos: CubicEOS = PR
    kij: Array | None = None
    vapor_draw: ArrayLike = 0.0
    liquid_draw: ArrayLike = 0.0
    m0: Array | None = field(default=None)

    def __post_init__(self) -> None:
        _warm_components(self.components)

    def initial_state(self) -> Array:
        if self.m0 is not None:
            return jnp.asarray(self.m0, dtype=float)
        return jnp.ones((len(self.components),))

    def measured(self, state: Array) -> dict[str, Array]:
        m = jnp.clip(state, 0.0, None)
        total = jnp.sum(m) + 1e-12
        return {"holdup": total, "x": m / total}

    def evaluate(
        self,
        state: Array,
        inlets: tuple[Stream, ...],
        controls: Controls | None = None,
        theta: Any = None,
    ) -> UnitStep:
        m = jnp.clip(state, 0.0, None)
        total = jnp.sum(m) + 1e-12
        x = m / total
        arr = _resolve(self.components)
        tc, pc, omega = arr[0], arr[1], arr[2]
        result = flash_pt(
            self.eos, jnp.asarray(self.t), jnp.asarray(self.p), x, tc, pc, omega, kij=self.kij
        )
        # Equilibrium vapour composition in contact with the holdup (fall back to the
        # holdup composition in the single-phase limit, beta -> 0).
        y = jnp.where(result.beta > 1e-6, result.y, x)
        v_draw = jnp.clip(_control(controls, "vapor_draw", self.vapor_draw), 0.0, None)
        l_draw = jnp.clip(_control(controls, "liquid_draw", self.liquid_draw), 0.0, None)
        inflow = _sum_inlets(inlets, len(self.components))
        dm = inflow - v_draw * y - l_draw * x
        vapor = Stream(
            n=v_draw * y, t=jnp.asarray(self.t), p=jnp.asarray(self.p), components=self.components
        )
        liquid = Stream(
            n=l_draw * x, t=jnp.asarray(self.t), p=jnp.asarray(self.p), components=self.components
        )
        meas = {"holdup": total, "x": x, "y": y}
        return UnitStep(dstate=dm, outlets=(vapor, liquid), measurements=meas)


__all__ = [
    "CSTRState",
    "DynamicCSTR",
    "DynamicFlash",
    "DynamicUnit",
    "GasReceiver",
    "LevelTank",
    "MixingTank",
    "ThermalMass",
    "UnitStep",
]
