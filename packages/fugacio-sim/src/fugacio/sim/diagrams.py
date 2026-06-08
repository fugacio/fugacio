"""Binary phase-diagram data and azeotrope finding for an equilibrium model.

Conceptual design and column screening lean on the binary picture: the P-x-y and
T-x-y envelopes and, above all, *where the azeotrope is* (it bounds what ordinary
distillation can reach). These helpers turn any binary
:class:`~fugacio.thermo.EquilibriumModel` into:

* :func:`pxy_diagram` / :func:`txy_diagram` -- the bubble (liquid) and the
  equilibrium-vapour curves on a composition grid, from one bubble sweep each; and
* :func:`azeotrope_pressure` / :func:`azeotrope_temperature` -- the azeotropic
  composition (where ``y_1 = x_1``) at fixed ``T`` or ``P``, returned with an
  ``exists`` flag so a non-azeotropic system is reported, not faked.

All outputs are differentiable: the diagram arrays through the bubble solves, and
the azeotrope locus through the bracketed root's implicit derivative -- so an
azeotrope's pressure sensitivity to a model parameter is a single
:func:`jax.grad` away.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.implicit import bracketed_root

ArrayLike = Array | float


class _BinaryModel(Protocol):
    """Structural type for the bubble calculations the diagrams need."""

    def bubble_pressure(self, t: ArrayLike, x: Array) -> tuple[Array, Array]: ...
    def bubble_temperature(
        self, p: ArrayLike, x: Array, *, t_min: float = ..., t_max: float = ...
    ) -> tuple[Array, Array]: ...


class _BubbleTemperatureModel(Protocol):
    """Structural type for the isobaric bubble calculation residue curves need."""

    def bubble_temperature(
        self, p: ArrayLike, x: Array, *, t_min: float = ..., t_max: float = ...
    ) -> tuple[Array, Array]: ...


class PxyDiagram(NamedTuple):
    """Isothermal P-x-y data for a binary.

    Attributes:
        x1: Liquid mole fraction of component 1 (the grid).
        y1: Equilibrium vapour mole fraction of component 1 (the bubble vapour).
        p: Bubble pressure at each ``x1`` (Pa).
        t: The (fixed) temperature (K).
    """

    x1: Array
    y1: Array
    p: Array
    t: Array


class TxyDiagram(NamedTuple):
    """Isobaric T-x-y data for a binary.

    Attributes:
        x1: Liquid mole fraction of component 1 (the grid).
        y1: Equilibrium vapour mole fraction of component 1.
        t: Bubble temperature at each ``x1`` (K).
        p: The (fixed) pressure (Pa).
    """

    x1: Array
    y1: Array
    t: Array
    p: Array


class AzeotropeResult(NamedTuple):
    """A binary azeotrope locus (or the best bracketed point if none exists).

    Attributes:
        exists: ``True`` if ``y_1 - x_1`` changes sign on the search bracket (a
            genuine azeotrope); ``False`` means the returned point is just a
            bracket end and should be ignored.
        x1: Azeotropic composition (``y_1 = x_1`` there).
        t: Temperature (K).
        p: Pressure (Pa).
    """

    exists: Array
    x1: Array
    t: Array
    p: Array


def pxy_diagram(model: _BinaryModel, t: ArrayLike, *, n: int = 51, eps: float = 1e-3) -> PxyDiagram:
    """Isothermal P-x-y curves for a binary on an ``n``-point composition grid.

    The grid is clipped to ``[eps, 1 - eps]`` to avoid the pure-component limits.
    Both curves come from a single bubble-pressure sweep: plot ``(x1, p)`` for the
    liquid line and ``(y1, p)`` for the vapour line.
    """
    x1 = jnp.linspace(eps, 1.0 - eps, n)

    def one(xx: Array) -> tuple[Array, Array]:
        p, y = model.bubble_pressure(t, jnp.array([xx, 1.0 - xx]))
        return p, y[0]

    p, y1 = jax.vmap(one)(x1)
    return PxyDiagram(x1=x1, y1=y1, p=p, t=jnp.asarray(t))


def txy_diagram(
    model: _BinaryModel,
    p: ArrayLike,
    *,
    n: int = 51,
    eps: float = 1e-3,
    t_min: float = 200.0,
    t_max: float = 600.0,
) -> TxyDiagram:
    """Isobaric T-x-y curves for a binary on an ``n``-point composition grid.

    Each grid point solves a bubble *temperature* (bracketed in ``[t_min, t_max]``)
    and returns the equilibrium vapour, giving both the liquid line ``(x1, t)`` and
    the vapour line ``(y1, t)`` from one sweep. Widen the bracket for very light or
    very heavy pairs.
    """
    x1 = jnp.linspace(eps, 1.0 - eps, n)

    def one(xx: Array) -> tuple[Array, Array]:
        t, y = model.bubble_temperature(p, jnp.array([xx, 1.0 - xx]), t_min=t_min, t_max=t_max)
        return t, y[0]

    t, y1 = jax.vmap(one)(x1)
    return TxyDiagram(x1=x1, y1=y1, t=t, p=jnp.asarray(p))


def azeotrope_pressure(
    model: _BinaryModel,
    t: ArrayLike,
    *,
    x_lo: float = 1e-3,
    x_hi: float = 1.0 - 1e-3,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> AzeotropeResult:
    """Find the binary azeotrope at fixed temperature ``t`` (where ``y_1 = x_1``).

    Brackets the root of ``y_1(x_1) - x_1`` from the bubble-pressure relation. The
    returned ``x1`` and ``p`` are differentiable with respect to the model
    parameters; check ``exists`` before trusting them.
    """

    def resid(x1: Array, m: _BinaryModel) -> Array:
        _, y = m.bubble_pressure(t, jnp.array([x1, 1.0 - x1]))
        return y[0] - x1

    f_lo = resid(jnp.asarray(x_lo), model)
    f_hi = resid(jnp.asarray(x_hi), model)
    exists = jnp.sign(f_lo) != jnp.sign(f_hi)
    x_az = bracketed_root(resid, model, jnp.asarray(x_lo), jnp.asarray(x_hi), tol, max_iter)
    p_az, _ = model.bubble_pressure(t, jnp.array([x_az, 1.0 - x_az]))
    return AzeotropeResult(exists=exists, x1=x_az, t=jnp.asarray(t), p=p_az)


def azeotrope_temperature(
    model: _BinaryModel,
    p: ArrayLike,
    *,
    x_lo: float = 1e-3,
    x_hi: float = 1.0 - 1e-3,
    t_min: float = 200.0,
    t_max: float = 600.0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> AzeotropeResult:
    """Find the binary azeotrope at fixed pressure ``p`` (where ``y_1 = x_1``).

    As :func:`azeotrope_pressure` but at constant pressure: each residual
    evaluation solves a bubble temperature (bracketed in ``[t_min, t_max]``).
    """

    def resid(x1: Array, m: _BinaryModel) -> Array:
        _, y = m.bubble_temperature(p, jnp.array([x1, 1.0 - x1]), t_min=t_min, t_max=t_max)
        return y[0] - x1

    f_lo = resid(jnp.asarray(x_lo), model)
    f_hi = resid(jnp.asarray(x_hi), model)
    exists = jnp.sign(f_lo) != jnp.sign(f_hi)
    x_az = bracketed_root(resid, model, jnp.asarray(x_lo), jnp.asarray(x_hi), tol, max_iter)
    t_az, _ = model.bubble_temperature(p, jnp.array([x_az, 1.0 - x_az]), t_min=t_min, t_max=t_max)
    return AzeotropeResult(exists=exists, x1=x_az, t=t_az, p=jnp.asarray(p))


class ResidueCurve(NamedTuple):
    """A simple-distillation residue curve: the still-liquid composition path.

    Attributes:
        x: Liquid composition along the curve, shape ``(steps + 1, n)`` (each row a
            mole-fraction vector on the composition simplex).
        t: Bubble (boiling) temperature at each point (K), shape ``(steps + 1,)``.
        p: The (fixed) pressure of the map (Pa).
    """

    x: Array
    t: Array
    p: Array


def _simplex(x: Array) -> Array:
    """Project a composition back onto the simplex (non-negative, sums to one)."""
    x = jnp.clip(x, 0.0, None)
    return x / jnp.sum(x)


@partial(jax.jit, static_argnames=("t_min", "t_max"))
def _bubble_ty(
    model: _BubbleTemperatureModel, p: ArrayLike, x: Array, t_min: float, t_max: float
) -> tuple[Array, Array]:
    """JIT entry point for one isobaric bubble solve ``x -> (T, y)``.

    Defined at module scope (with ``model`` as a pytree argument) so every residue
    curve drawn with the same model and bracket reuses a *single* compiled bubble
    solve, instead of recompiling the nested root-find for each curve/direction.
    """
    return model.bubble_temperature(p, x, t_min=t_min, t_max=t_max)


def residue_curve(
    model: _BubbleTemperatureModel,
    x0: Array,
    p: ArrayLike,
    *,
    steps: int = 200,
    ds: float = 0.05,
    direction: float = 1.0,
    t_min: float = 150.0,
    t_max: float = 700.0,
) -> ResidueCurve:
    """Integrate one residue curve of open (Rayleigh) distillation at fixed ``P``.

    Solves the residue-curve ODE ``dx_i/dtau = x_i - y_i(x)``, where ``y(x)`` is the
    bubble-point vapour in equilibrium with the still liquid ``x``. With
    ``direction = +1`` the curve advances toward the high-boiling stable node
    (the still-pot residue enriches in heavy components); ``direction = -1`` traces
    it back toward the low-boiling node.

    Integration is an explicit Euler march with the composition re-projected onto
    the simplex each step, so the path stays a valid composition. The expensive part
    -- the bubble-point ``(T, y)`` at each point -- is one compiled, differentiable
    solve (:func:`_bubble_ty`) reused across every step and every curve.

    Args:
        model: Any object with a ``bubble_temperature(p, x)`` method (e.g. an
            :class:`~fugacio.thermo.EquilibriumModel` from :mod:`fugacio.sim.models`).
        x0: Starting liquid composition, shape ``(n,)``.
        p: Pressure (Pa).
        steps: Number of integration steps.
        ds: Pseudo-time step size.
        direction: ``+1`` toward heavies, ``-1`` toward lights.
        t_min, t_max: Temperature bracket for the bubble-temperature solve.

    Returns:
        A :class:`ResidueCurve` with the composition and temperature path
        (``steps + 1`` points).
    """
    x0 = _simplex(jnp.asarray(x0, dtype=float))
    h = ds * direction
    p = jnp.asarray(p)
    xs = [x0]
    ts: list[Array] = []
    x = x0
    for _ in range(steps):
        t, y = _bubble_ty(model, p, x, t_min, t_max)
        ts.append(jnp.asarray(t))
        x = _simplex(x + h * (x - y))
        xs.append(x)
    t_last, _y = _bubble_ty(model, p, x, t_min, t_max)
    ts.append(jnp.asarray(t_last))
    return ResidueCurve(x=jnp.stack(xs), t=jnp.stack(ts), p=p)


def residue_curve_map(
    model: _BubbleTemperatureModel,
    starts: Array,
    p: ArrayLike,
    *,
    steps: int = 150,
    ds: float = 0.05,
    t_min: float = 150.0,
    t_max: float = 700.0,
) -> list[ResidueCurve]:
    """Trace a residue-curve map: one full curve through each starting composition.

    Each start is integrated *both* directions (toward the light and the heavy node)
    and the two halves are stitched into a single curve passing through the start.
    This is the ternary distillation designer's master diagram -- distillation
    boundaries and reachable products fall out of the family of curves.

    Args:
        model: Object exposing ``bubble_temperature(p, x)``.
        starts: Starting compositions, shape ``(k, n)``.
        p: Pressure (Pa).
        steps: Steps taken in *each* direction (the curve has ``2*steps + 1`` points).
        ds: Pseudo-time step size.
        t_min, t_max: Temperature bracket for the bubble-temperature solves.

    Returns:
        A list of ``k`` :class:`ResidueCurve` objects.
    """
    starts = jnp.atleast_2d(jnp.asarray(starts, dtype=float))
    curves: list[ResidueCurve] = []
    for x0 in starts:
        fwd = residue_curve(
            model, x0, p, steps=steps, ds=ds, direction=1.0, t_min=t_min, t_max=t_max
        )
        bwd = residue_curve(
            model, x0, p, steps=steps, ds=ds, direction=-1.0, t_min=t_min, t_max=t_max
        )
        # Stitch reversed backward half (dropping its duplicate start) onto the forward half.
        x_full = jnp.concatenate([bwd.x[::-1][:-1], fwd.x], axis=0)
        t_full = jnp.concatenate([bwd.t[::-1][:-1], fwd.t], axis=0)
        curves.append(ResidueCurve(x=x_full, t=t_full, p=jnp.asarray(p)))
    return curves


__all__ = [
    "AzeotropeResult",
    "PxyDiagram",
    "ResidueCurve",
    "TxyDiagram",
    "azeotrope_pressure",
    "azeotrope_temperature",
    "pxy_diagram",
    "residue_curve",
    "residue_curve_map",
    "txy_diagram",
]
