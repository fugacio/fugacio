"""Assemble dynamic units and control loops into one differentiable flowsheet ODE.

`DynamicFlowsheet` is the dynamic counterpart of
`fugacio.sim.Flowsheet`: register time-varying feeds, connect dynamic units
port-to-port, and close control loops, then `DynamicFlowsheet.simulate` over
a time horizon. Internally the whole plant (every unit holdup *and* every
controller's integral/derivative state) is concatenated into a single state
pytree with one global right-hand side, which is handed to the differentiable
`fugacio.sim.dynamics.odeint`. The result is end-to-end differentiable: you
can take a gradient of any trajectory feature (a settling time proxy, an off-spec
integral, a peak temperature) with respect to controller gains, setpoints, feed
schedules, equipment parameters carried in ``theta``, or the initial state.

Holdups break algebraic recycle loops (a recycled stream is an *output of a unit
state*, not the solution of a tear), so a dynamic flowsheet needs no tear solver;
units are simply evaluated in registration order each instant. Controlled
measurements (level, temperature, pressure, composition) are functions of unit
state alone, so the controller outputs for the instant are computed first, then the
units are advanced with those manipulated variables, keeping the instantaneous
system explicit and cheap.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.control.pid import PID
from fugacio.sim.dynamics.integrate import odeint
from fugacio.sim.dynamics.units import DynamicUnit
from fugacio.sim.stream import Stream

ArrayLike = Array | float

#: A feed is either a fixed `Stream` or a callable ``(t, theta) -> Stream``.
FeedSpec = Stream | Callable[[Array, Any], Stream]
#: A measurement reference: ``(unit, key)``, ``(unit, key, index)`` for a vector
#: measurement, or a callable ``measurements -> scalar``.
MeasurementSpec = Any
#: A setpoint: a constant or a callable ``(t, theta) -> value``.
SetpointSpec = ArrayLike | Callable[[Array, Any], Array]


@dataclass
class _Unit:
    unit: DynamicUnit
    inputs: tuple[str, ...]


@dataclass
class _Loop:
    pid: PID
    measurement: MeasurementSpec
    setpoint: SetpointSpec
    actuator: tuple[str, str]
    u0: ArrayLike | None


@dataclass
class _Manip:
    unit: str
    mv: str
    fn: Callable[[Array, Any], Array]


class DynamicResult(NamedTuple):
    """Trajectory returned by `DynamicFlowsheet.simulate`.

    Attributes:
        ts: Output times (shape ``(T,)``).
        states: Per-unit state trajectory (a dict ``name -> state`` with a leading
            time axis on every leaf).
        measurements: Per-unit measurement trajectory (a dict ``name -> {key: ...}``).
        controls: Manipulated-variable trajectory (a dict ``"unit.mv" -> values``).
    """

    ts: Array
    states: dict[str, Any]
    measurements: dict[str, dict[str, Array]]
    controls: dict[str, Array]

    def measurement(self, unit: str, key: str) -> Array:
        """Trajectory of a named measurement for a unit."""
        return self.measurements[unit][key]


def _eval_feed(spec: FeedSpec, t: Array, theta: Any) -> Stream:
    return spec(t, theta) if callable(spec) else spec


def _eval_setpoint(spec: SetpointSpec, t: Array, theta: Any) -> Array:
    return jnp.asarray(spec(t, theta)) if callable(spec) else jnp.asarray(spec)


def _read_measurement(spec: MeasurementSpec, meas: dict[str, dict[str, Array]]) -> Array:
    if callable(spec):
        return jnp.asarray(spec(meas))
    unit, key, *rest = spec
    value = meas[unit][key]
    if rest:
        return value[rest[0]]
    return value


@dataclass
class DynamicFlowsheet:
    """A declarative dynamic flowsheet: feeds, dynamic units, and control loops.

    Example::

        fs = DynamicFlowsheet()
        fs.feed("feed", feed_stream)                       # constant or (t, theta) -> Stream
        fs.add(tank, inputs=["feed"])                      # a LevelTank named "tank"
        fs.control(pid, measurement=("tank", "level"),     # level controller ...
                   setpoint=2.0, actuator=("tank", "flow"))# ... manipulating outlet flow
        result = fs.simulate(ts=jnp.linspace(0, 100, 201))
        level = result.measurement("tank", "level")

    Units are advanced in registration order; reference an upstream outlet as
    ``"unit.port"`` (port index) and a feed by its name.
    """

    feeds: dict[str, FeedSpec] = field(default_factory=dict)
    units: dict[str, _Unit] = field(default_factory=dict)
    loops: list[_Loop] = field(default_factory=list)
    manips: list[_Manip] = field(default_factory=list)
    _order: list[str] = field(default_factory=list)

    def feed(self, name: str, spec: FeedSpec) -> DynamicFlowsheet:
        """Register a feed stream (a fixed `Stream` or ``(t, theta) -> Stream``)."""
        self.feeds[name] = spec
        return self

    def add(self, unit: DynamicUnit, *, inputs: Sequence[str] = ()) -> DynamicFlowsheet:
        """Register a dynamic ``unit`` with its inlet sources (feeds or ``"unit.port"``)."""
        if unit.name in self.units:
            raise ValueError(f"duplicate unit name {unit.name!r}")
        self.units[unit.name] = _Unit(unit=unit, inputs=tuple(inputs))
        self._order.append(unit.name)
        return self

    def control(
        self,
        pid: PID,
        *,
        measurement: MeasurementSpec,
        setpoint: SetpointSpec,
        actuator: tuple[str, str],
        u0: ArrayLike | None = None,
    ) -> DynamicFlowsheet:
        """Close a feedback loop: ``pid`` drives ``actuator`` to hold ``measurement``."""
        self.loops.append(
            _Loop(pid=pid, measurement=measurement, setpoint=setpoint, actuator=actuator, u0=u0)
        )
        return self

    def manipulate(self, unit: str, mv: str, fn: Callable[[Array, Any], Array]) -> DynamicFlowsheet:
        """Drive a manipulated variable open-loop from a schedule ``fn(t, theta)``."""
        self.manips.append(_Manip(unit=unit, mv=mv, fn=fn))
        return self

    # ------------------------------------------------------------------ #
    # State assembly
    # ------------------------------------------------------------------ #
    def initial_state(self, t0: ArrayLike = 0.0, theta: Any = None) -> dict[str, Any]:
        """Build the global initial state (unit holdups + controller states)."""
        state: dict[str, Any] = {name: u.unit.initial_state() for name, u in self.units.items()}
        meas = self._measure(state)
        for i, loop in enumerate(self.loops):
            pv0 = _read_measurement(loop.measurement, meas)
            state[f"__ctrl_{i}"] = loop.pid.init_state(pv0, loop.u0)
        return state

    def _measure(self, state: dict[str, Any]) -> dict[str, dict[str, Array]]:
        return {name: u.unit.measured(state[name]) for name, u in self.units.items()}

    def _resolve_source(
        self, src: str, feeds_now: dict[str, Stream], outlets: dict[str, tuple[Stream, ...]]
    ) -> Stream:
        if src in feeds_now:
            return feeds_now[src]
        unit_name, _, port = src.partition(".")
        if unit_name not in outlets:
            raise ValueError(
                f"source {src!r} is not available yet; register {unit_name!r} before its consumer"
            )
        return outlets[unit_name][int(port) if port else 0]

    def _evaluate(
        self, t: Array, state: dict[str, Any], theta: Any
    ) -> tuple[dict[str, Any], dict[str, dict[str, Array]], dict[str, Array]]:
        """One instant: returns ``(dstate, measurements, control_outputs)``."""
        meas = self._measure(state)
        controls_by_unit: dict[str, dict[str, Array]] = {name: {} for name in self.units}
        control_outputs: dict[str, Array] = {}
        ctrl_deriv: dict[str, Any] = {}
        for i, loop in enumerate(self.loops):
            pid_state = state[f"__ctrl_{i}"]
            pv = _read_measurement(loop.measurement, meas)
            sp = _eval_setpoint(loop.setpoint, t, theta)
            u = loop.pid.output(pid_state, sp, pv)
            unit_name, mv = loop.actuator
            controls_by_unit[unit_name][mv] = u
            control_outputs[f"{unit_name}.{mv}"] = u
            ctrl_deriv[f"__ctrl_{i}"] = loop.pid.derivative(pid_state, sp, pv)
        for m in self.manips:
            val = jnp.asarray(m.fn(t, theta))
            controls_by_unit[m.unit][m.mv] = val
            control_outputs[f"{m.unit}.{m.mv}"] = val

        feeds_now = {name: _eval_feed(spec, t, theta) for name, spec in self.feeds.items()}
        outlets: dict[str, tuple[Stream, ...]] = {}
        dstate: dict[str, Any] = {}
        for name in self._order:
            unit_entry = self.units[name]
            inlets = tuple(
                self._resolve_source(src, feeds_now, outlets) for src in unit_entry.inputs
            )
            step = unit_entry.unit.evaluate(state[name], inlets, controls_by_unit[name], theta)
            outlets[name] = step.outlets
            dstate[name] = step.dstate
            meas[name] = step.measurements
        dstate.update(ctrl_deriv)
        return dstate, meas, control_outputs

    def rhs(self, t: Array, state: dict[str, Any], theta: Any = None) -> dict[str, Any]:
        """Global state derivative ``d(state)/dt`` (the function handed to the integrator)."""
        dstate, _, _ = self._evaluate(t, state, theta)
        return dstate

    def simulate(
        self,
        ts: Array,
        *,
        y0: dict[str, Any] | None = None,
        theta: Any = None,
        method: str = "rk4",
        substeps: int = 4,
    ) -> DynamicResult:
        """Integrate the flowsheet over the output grid ``ts`` and return a `DynamicResult`.

        Args:
            ts: Strictly increasing output times.
            y0: Optional global initial state (defaults to each unit's initial state
                and a bumpless controller start).
            theta: Differentiable parameter pytree threaded to feeds, setpoints,
                manipulations and units.
            method: Integration method (see `fugacio.sim.dynamics.FIXED_METHODS`).
            substeps: Inner steps per output interval.

        The returned trajectories of measurements and controller outputs are
        recomputed from the state trajectory, so they are differentiable in
        ``theta`` and ``y0`` as well.
        """
        ts = jnp.asarray(ts, dtype=float)
        state0 = self.initial_state(ts[0], theta) if y0 is None else y0
        traj = odeint(self.rhs, state0, ts, theta, method=method, substeps=substeps)

        def snapshot(
            t: Array, state: dict[str, Any]
        ) -> tuple[dict[str, dict[str, Array]], dict[str, Array]]:
            _, meas, controls = self._evaluate(t, state, theta)
            return meas, controls

        meas_traj, controls_traj = jax.vmap(snapshot)(ts, traj)
        return DynamicResult(ts=ts, states=traj, measurements=meas_traj, controls=controls_traj)


def simulate(
    rhs: Callable[[Array, Any, Any], Any],
    y0: Any,
    ts: Array,
    theta: Any = None,
    *,
    method: str = "rk4",
    substeps: int = 4,
) -> Any:
    """Integrate a bare ``rhs(t, y, theta)`` over ``ts`` (a thin `odeint` alias).

    For ad-hoc dynamic models that are not expressed as a `DynamicFlowsheet`.
    Returns the state trajectory pytree (leading time axis), differentiable in
    ``y0`` and ``theta``.
    """
    return odeint(rhs, y0, ts, theta, method=method, substeps=substeps)


__all__ = ["DynamicFlowsheet", "DynamicResult", "simulate"]
