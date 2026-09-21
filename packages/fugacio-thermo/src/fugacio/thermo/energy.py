"""Energy-specified equilibrium: the shared result type and temperature solver.

The isothermal flash answers "what splits?" at a *given* temperature. Process
units instead fix an *energy* specification (a heat duty, an adiabatic mix, an
isentropic compression) and the temperature is unknown. Every property package
implements ``flash_ph`` / ``flash_ps`` on top of this module:

* `EnergyFlashResult`: the solved temperature and phase split;
* `_implicit_temperature`: a safeguarded Newton/bisection solve of a monotone
  energy residual (enthalpy or entropy minus a specification) for the
  temperature. The forward pass brackets the root in ``[t_min, t_max]`` and
  only evaluates residual values, falling back to bisection whenever a Newton
  step would leave the bracket, so it's robust when a trial temperature crosses
  a phase boundary. The converged temperature is differentiated by the implicit
  function theorem with no differentiation through the iteration itself, and
  an unconverged temperature has nonfinite derivatives.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.implicit import _residual_linearization

ArrayLike = Array | float


class EnergyFlashResult(NamedTuple):
    """Result of an energy-specified flash (PH or PS).

    Attributes:
        t: Solved temperature (K).
        beta: Vapour molar fraction.
        x: Liquid-phase mole fractions.
        y: Vapour-phase mole fractions.
        k: Equilibrium ratios at the solution.
    """

    t: Array
    beta: Array
    x: Array
    y: Array
    k: Array


@partial(jax.custom_jvp, nondiff_argnums=(0, 3, 4, 5, 6))
def _implicit_temperature(
    residual: Callable[[Array, Any], Array],
    params: Any,
    t_init: ArrayLike,
    t_min: float,
    t_max: float,
    tol: float,
    max_iter: int,
) -> Array:
    """Solve ``residual(T, params) = 0`` for the temperature ``T`` in ``[t_min, t_max]``.

    ``residual`` is a smooth, monotonically *increasing* energy residual (enthalpy
    or entropy minus a specification, since both rise with temperature); ``params``
    is the differentiable pytree it depends on.

    The forward pass is a safeguarded Newton iteration. It maintains a bracket
    ``[lo, hi]`` (initialised to ``[t_min, t_max]``) that always contains the root,
    using only residual *values*: the slope is estimated by a one-sided finite
    difference rather than ``jax.grad``, because a Newton trial can cross a phase
    boundary where the flash gradient is undefined (``NaN``). A Newton step is
    accepted only if it stays inside the bracket and the slope is usable; otherwise
    the step bisects. A trial where the residual is ``NaN`` (outside the model's
    domain, such as above a gamma-phi component's critical temperature) is
    excluded by moving the bracket end on its side of the last finite trial, so
    the search converges from any starting point without propagating a ``NaN``.

    The converged temperature is differentiated by the implicit function theorem
    in the ``custom_jvp`` rule below: ``dT* = -(dr/dparams . dparams) / (dr/dT)``,
    using only *first-order* sensitivities of the residual at the solution (so the
    flash's own implicit rules handle them and nothing differentiates through the
    iteration itself).
    """

    def cond(carry: tuple[Array, Array, Array, Array, Array, Array]) -> Array:
        _, _, _, _, i, err = carry
        return (err > tol) & (i < max_iter)

    def body(
        carry: tuple[Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        lo, hi, t, t_ok, i, _ = carry
        r = residual(t, params)
        # The residual increases with T, so the sign of r tells us which side of
        # the root we are on; tighten the bracket accordingly.
        lo = jnp.where(r <= 0.0, t, lo)
        hi = jnp.where(r > 0.0, t, hi)
        # An undefined residual excludes the trial's side of the last finite one.
        undefined = ~jnp.isfinite(r)
        hi = jnp.where(undefined & (t >= t_ok), t, hi)
        lo = jnp.where(undefined & (t < t_ok), t, lo)
        t_ok = jnp.where(undefined, t_ok, t)
        # One-sided finite-difference slope, probing *inside* the bracket so we
        # never evaluate the residual outside [t_min, t_max] (where it may be NaN).
        h = jnp.maximum(1e-4 * jnp.abs(t), 1e-4)
        t_probe = jnp.maximum(t - h, lo)
        dr = (r - residual(t_probe, params)) / jnp.maximum(t - t_probe, 1e-12)
        t_newton = t - r / dr
        usable = jnp.isfinite(t_newton) & (t_newton > lo) & (t_newton < hi) & (jnp.abs(dr) > 1e-12)
        t_next = jnp.where(usable, t_newton, 0.5 * (lo + hi))
        return lo, hi, t_next, t_ok, i + 1, jnp.abs(t_next - t)

    lo0 = jnp.asarray(t_min, dtype=float)
    hi0 = jnp.asarray(t_max, dtype=float)
    t0 = jnp.clip(jnp.asarray(t_init, dtype=float), lo0, hi0)
    init = (lo0, hi0, t0, t0, jnp.asarray(0), jnp.asarray(jnp.inf))
    _, _, t_star, _, _, _ = jax.lax.while_loop(cond, body, init)
    return t_star


@partial(_implicit_temperature.defjvp, symbolic_zeros=True)
def _implicit_temperature_jvp(
    residual: Callable[[Array, Any], Array],
    t_min: float,
    t_max: float,
    tol: float,
    max_iter: int,
    primals: tuple[Any, ArrayLike],
    tangents: tuple[Any, Any],
) -> tuple[Array, Array]:
    params, t_init = primals
    params_dot, _ = tangents
    t_star = _implicit_temperature(residual, params, t_init, t_min, t_max, tol, max_iter)
    value, push, r_dot = _residual_linearization(residual, t_star, params, params_dot)
    r_t = push(jnp.ones_like(t_star))
    error = jnp.abs(value)
    valid = jnp.isfinite(error) & (error <= jnp.maximum(4 * tol * jnp.abs(r_t), 1e-6))
    t_dot = (-r_dot / r_t) * jnp.where(valid, 1.0, jnp.nan)
    return t_star, t_dot
