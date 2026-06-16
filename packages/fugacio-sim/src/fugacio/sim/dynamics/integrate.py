"""Differentiable time integration of ordinary differential equations.

Everything in :mod:`fugacio.sim` up to here is *steady state*: a flash, a column,
a recycle loop is the solution of an algebraic system. Dynamics adds the missing
dimension -- time -- and with it the question every real plant asks: not just
*where* does the process settle, but *how* does it get there, how fast, and is
the path stable. This module is the engine for that: a small, self-contained,
end-to-end-differentiable ODE integrator written against :mod:`jax.numpy`.

Two complementary drivers are provided, and the split is deliberate:

* :func:`odeint` -- a **fixed output grid** integrator built on
  :func:`jax.lax.scan`. Because the step pattern is static, it is differentiable
  in *both* directions out of the box (forward- and reverse-mode), returns the
  whole trajectory at the requested times, and is the workhorse for simulating
  dynamic flowsheets where a uniform sampling grid is wanted anyway. Several
  steppers are available -- explicit Euler, classical RK4, the Dormand-Prince
  5(4) stages used as a fixed step, and the A-stable implicit Euler / trapezoidal
  methods (each implicit step solved by a fixed-count Newton iteration so the
  whole march stays reverse-differentiable) for stiff systems.
* :func:`integrate` -- an **adaptive-step** Dormand-Prince 5(4) integrator with a
  PI step controller, for when only the final state matters and efficiency or
  stiffness control does. Its data-dependent step count rules out naive
  reverse-mode (you cannot back-propagate through a :func:`jax.lax.while_loop`),
  so gradients are supplied by the *continuous adjoint* method: a
  hand-written ``custom_vjp`` integrates the adjoint ODE backwards, exactly the
  same "differentiate the converged solution, not the iteration" philosophy that
  :mod:`fugacio.thermo.implicit` and :func:`fugacio.sim.tear_solve` use for
  algebraic solves.

Both accept an arbitrary JAX pytree as the state ``y`` and an arbitrary pytree of
differentiable parameters ``theta``; they are flattened internally with
:func:`jax.flatten_util.ravel_pytree`, so you integrate in the natural shape of
your problem (a dict of unit holdups, a :class:`~fugacio.sim.stream.Stream`, a
bare vector) and differentiate with respect to parameters in their natural shape.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

ArrayLike = Array | float

#: A right-hand side ``f(t, y, theta) -> dy`` returning the time derivative of the
#: state pytree ``y`` (same structure as ``y``). ``t`` is a scalar and ``theta`` is
#: an arbitrary differentiable parameter pytree (or ``None``).
RHS = Callable[[Array, Any, Any], Any]

#: The flattened form used by the numeric core: ``f(t, y_flat, theta) -> dy_flat``.
_FlatRHS = Callable[[Array, Array, Any], Array]


# --------------------------------------------------------------------------- #
# Explicit steppers (operate on a flat state vector)
# --------------------------------------------------------------------------- #
def _euler_step(f: _FlatRHS, t: Array, y: Array, dt: Array, theta: Any) -> Array:
    """One explicit (forward) Euler step. First order; cheap but only for smooth, non-stiff RHS."""
    return y + dt * f(t, y, theta)


def _rk4_step(f: _FlatRHS, t: Array, y: Array, dt: Array, theta: Any) -> Array:
    """One classical four-stage Runge-Kutta step (fourth order)."""
    k1 = f(t, y, theta)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, theta)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, theta)
    k4 = f(t + dt, y + dt * k3, theta)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# Dormand-Prince 5(4) Butcher tableau (the method behind MATLAB's ``ode45``).
_DP_C = (1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0)
_DP_A = (
    (1.0 / 5.0,),
    (3.0 / 40.0, 9.0 / 40.0),
    (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
    (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
    (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
    (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0),
)
_DP_B5 = (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0)
_DP_B4 = (
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
)


def _dopri5_stages(f: _FlatRHS, t: Array, y: Array, dt: Array, theta: Any) -> list[Array]:
    """The seven Dormand-Prince stage derivatives at ``(t, y)`` over a step ``dt``."""
    k1 = f(t, y, theta)
    k2 = f(t + _DP_C[0] * dt, y + dt * (_DP_A[0][0] * k1), theta)
    k3 = f(t + _DP_C[1] * dt, y + dt * (_DP_A[1][0] * k1 + _DP_A[1][1] * k2), theta)
    k4 = f(
        t + _DP_C[2] * dt,
        y + dt * (_DP_A[2][0] * k1 + _DP_A[2][1] * k2 + _DP_A[2][2] * k3),
        theta,
    )
    k5 = f(
        t + _DP_C[3] * dt,
        y + dt * (_DP_A[3][0] * k1 + _DP_A[3][1] * k2 + _DP_A[3][2] * k3 + _DP_A[3][3] * k4),
        theta,
    )
    k6 = f(
        t + _DP_C[4] * dt,
        y
        + dt
        * (
            _DP_A[4][0] * k1
            + _DP_A[4][1] * k2
            + _DP_A[4][2] * k3
            + _DP_A[4][3] * k4
            + _DP_A[4][4] * k5
        ),
        theta,
    )
    k7 = f(
        t + _DP_C[5] * dt,
        y
        + dt
        * (
            _DP_A[5][0] * k1
            + _DP_A[5][1] * k2
            + _DP_A[5][2] * k3
            + _DP_A[5][3] * k4
            + _DP_A[5][4] * k5
            + _DP_A[5][5] * k6
        ),
        theta,
    )
    return [k1, k2, k3, k4, k5, k6, k7]


def _dopri5_step(f: _FlatRHS, t: Array, y: Array, dt: Array, theta: Any) -> Array:
    """One fixed Dormand-Prince step, taking the fifth-order solution."""
    k = _dopri5_stages(f, t, y, dt, theta)
    return y + dt * sum(_DP_B5[i] * k[i] for i in range(7))


def _dopri5_step_err(f: _FlatRHS, t: Array, y: Array, dt: Array, theta: Any) -> tuple[Array, Array]:
    """One Dormand-Prince step returning ``(y5, error)`` (5th-order step, embedded estimate)."""
    k = _dopri5_stages(f, t, y, dt, theta)
    y5 = y + dt * sum(_DP_B5[i] * k[i] for i in range(7))
    err = dt * sum((_DP_B5[i] - _DP_B4[i]) * k[i] for i in range(7))
    return y5, err


# --------------------------------------------------------------------------- #
# Implicit steppers for stiff systems (fixed-count Newton, reverse-safe)
# --------------------------------------------------------------------------- #
def _newton_implicit(g: Callable[[Array], Array], y_guess: Array, *, iters: int) -> Array:
    """Solve ``g(y) = 0`` with a fixed number of Newton steps (dense Jacobian).

    A *fixed* iteration count (not a tolerance ``while_loop``) is used on purpose:
    the unrolled iteration is plain differentiable JAX, so an implicit step inside
    a :func:`jax.lax.scan` march remains reverse-mode differentiable. The default
    count is comfortably enough for the well-conditioned, well-initialised steps a
    smooth implicit integrator produces.
    """

    def body(y: Array, _: Any) -> tuple[Array, None]:
        r = g(y)
        jac = jax.jacobian(g)(y)
        dy = jnp.linalg.solve(jac, -r)
        return y + dy, None

    y_star, _ = jax.lax.scan(body, y_guess, None, length=iters)
    return y_star


def _implicit_euler_step(
    f: _FlatRHS, t: Array, y: Array, dt: Array, theta: Any, *, newton_iters: int = 8
) -> Array:
    """One backward-Euler step (first order, L-stable). Robust for stiff systems."""

    def g(y_next: Array) -> Array:
        return y_next - y - dt * f(t + dt, y_next, theta)

    return _newton_implicit(g, y + dt * f(t, y, theta), iters=newton_iters)


def _trapezoidal_step(
    f: _FlatRHS, t: Array, y: Array, dt: Array, theta: Any, *, newton_iters: int = 8
) -> Array:
    """One trapezoidal (Crank-Nicolson) step (second order, A-stable)."""
    f0 = f(t, y, theta)

    def g(y_next: Array) -> Array:
        return y_next - y - 0.5 * dt * (f0 + f(t + dt, y_next, theta))

    return _newton_implicit(g, y + dt * f0, iters=newton_iters)


_FIXED_STEPPERS: dict[str, Callable[..., Array]] = {
    "euler": _euler_step,
    "rk4": _rk4_step,
    "dopri5": _dopri5_step,
    "implicit_euler": _implicit_euler_step,
    "trapezoidal": _trapezoidal_step,
}

#: Names of the implicit (stiff-capable) fixed-step methods.
IMPLICIT_METHODS = ("implicit_euler", "trapezoidal")
#: Names of every method understood by :func:`odeint`.
FIXED_METHODS = tuple(_FIXED_STEPPERS)


# --------------------------------------------------------------------------- #
# Fixed output-grid integrator (scan-based; differentiable both ways)
# --------------------------------------------------------------------------- #
def odeint(
    func: RHS,
    y0: Any,
    ts: Array,
    theta: Any = None,
    *,
    method: str = "rk4",
    substeps: int = 1,
) -> Any:
    """Integrate ``dy/dt = func(t, y, theta)`` over the output grid ``ts``.

    The state is advanced from ``ts[0]`` to ``ts[-1]``, taking ``substeps`` uniform
    inner steps of the chosen ``method`` between successive output points, and the
    state is recorded at every entry of ``ts``. Because the step pattern is static
    the whole integration is an ordinary :func:`jax.lax.scan`, hence differentiable
    in forward *and* reverse mode with respect to ``y0`` and ``theta`` -- no custom
    rule needed.

    Args:
        func: Right-hand side ``func(t, y, theta) -> dy`` (``dy`` matches ``y``'s
            pytree structure).
        y0: Initial state pytree at ``ts[0]``.
        ts: 1-D array of strictly increasing output times (length >= 2).
        theta: Optional differentiable parameter pytree forwarded to ``func``.
        method: One of :data:`FIXED_METHODS` -- ``"euler"``, ``"rk4"`` (default),
            ``"dopri5"``, or the stiff ``"implicit_euler"`` / ``"trapezoidal"``.
        substeps: Number of inner integration steps per output interval (>= 1);
            raise it to cut discretisation error without densifying ``ts``.

    Returns:
        The trajectory as a pytree of the same structure as ``y0`` with a leading
        time axis of length ``len(ts)`` on each leaf (so ``out[k]`` is the state at
        ``ts[k]``). Differentiable with respect to ``y0`` and ``theta``.
    """
    if method not in _FIXED_STEPPERS:
        raise ValueError(f"unknown method {method!r}; choose one of {FIXED_METHODS}")
    if substeps < 1:
        raise ValueError("substeps must be >= 1")
    stepper = _FIXED_STEPPERS[method]
    flat0, unravel = ravel_pytree(y0)
    ts = jnp.asarray(ts, dtype=float)

    def f_flat(t: Array, y: Array, th: Any) -> Array:
        dy = func(t, unravel(y), th)
        flat, _ = ravel_pytree(dy)
        return flat

    def interval(y: Array, bounds: tuple[Array, Array]) -> tuple[Array, Array]:
        t0, t1 = bounds
        dt = (t1 - t0) / substeps

        def inner(carry: tuple[Array, Array], _: Any) -> tuple[tuple[Array, Array], None]:
            t, yy = carry
            yy_new = stepper(f_flat, t, yy, dt, th_closure)
            return (t + dt, yy_new), None

        (_, y_next), _ = jax.lax.scan(inner, (t0, y), None, length=substeps)
        return y_next, y_next

    th_closure = theta
    bounds_seq = (ts[:-1], ts[1:])
    _, ys_rest = jax.lax.scan(interval, flat0, bounds_seq)
    ys_flat = jnp.concatenate([flat0[None, :], ys_rest], axis=0)
    return jax.vmap(unravel)(ys_flat)


def odeint_final(
    func: RHS,
    y0: Any,
    t0: ArrayLike,
    t1: ArrayLike,
    theta: Any = None,
    *,
    method: str = "rk4",
    steps: int = 100,
) -> Any:
    """Integrate from ``t0`` to ``t1`` and return only the final state.

    A convenience wrapper over :func:`odeint` for the common case of a single
    interval with ``steps`` uniform steps; differentiable in ``y0`` and ``theta``.
    """
    ts = jnp.array([float(t0), float(t1)])
    traj = odeint(func, y0, ts, theta, method=method, substeps=steps)
    return jax.tree_util.tree_map(lambda leaf: leaf[-1], traj)


# --------------------------------------------------------------------------- #
# Adaptive Dormand-Prince integrator with a continuous-adjoint custom_vjp
# --------------------------------------------------------------------------- #
class _AdaptiveStats(NamedTuple):
    n_steps: Array
    n_accepted: Array
    final_dt: Array


def _error_norm(err: Array, y0: Array, y1: Array, rtol: float, atol: float) -> Array:
    """RMS error norm scaled by ``atol + rtol * max(|y0|, |y1|)`` (target <= 1 to accept)."""
    scale = atol + rtol * jnp.maximum(jnp.abs(y0), jnp.abs(y1))
    return jnp.sqrt(jnp.mean((err / scale) ** 2))


def _adaptive_solve(
    f: _FlatRHS,
    y0: Array,
    theta: Any,
    t0: float,
    t1: float,
    dt0: float,
    rtol: float,
    atol: float,
    max_steps: int,
) -> tuple[Array, _AdaptiveStats]:
    """Forward adaptive Dormand-Prince march from ``t0`` to ``t1`` (returns final state)."""
    safety, min_factor, max_factor = 0.9, 0.2, 10.0
    direction = jnp.sign(jnp.asarray(t1 - t0))
    direction = jnp.where(direction == 0.0, 1.0, direction)

    def cond(carry: tuple[Array, Array, Array, Array, Array]) -> Array:
        t, _y, _dt, n, _acc = carry
        remaining = (t1 - t) * direction
        return (remaining > 1e-12) & (n < max_steps)

    def body(
        carry: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        t, y, dt, n, acc = carry
        remaining = (t1 - t) * direction
        dt = jnp.minimum(jnp.abs(dt), remaining) * direction
        y5, err = _dopri5_step_err(f, t, y, dt, theta)
        enorm = _error_norm(err, y, y5, rtol, atol)
        accept = enorm <= 1.0
        t_new = jnp.where(accept, t + dt, t)
        y_new = jnp.where(accept, y5, y)
        factor = safety * jnp.where(enorm > 0.0, enorm ** (-0.2), max_factor)
        factor = jnp.clip(factor, min_factor, max_factor)
        dt_new = dt * factor
        return t_new, y_new, dt_new, n + 1, acc + jnp.where(accept, 1, 0)

    t0a = jnp.asarray(t0, dtype=float)
    dt_init = jnp.asarray(dt0, dtype=float) * direction
    init = (t0a, y0, dt_init, jnp.asarray(0), jnp.asarray(0))
    _t_f, y_f, dt_f, n_f, acc_f = jax.lax.while_loop(cond, body, init)
    return y_f, _AdaptiveStats(n_steps=n_f, n_accepted=acc_f, final_dt=dt_f)


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4, 5, 6, 7, 8))
def _adaptive_flat(
    f: _FlatRHS,
    y0: Array,
    theta: Any,
    t0: float,
    t1: float,
    dt0: float,
    rtol: float,
    atol: float,
    max_steps: int,
) -> tuple[Array, Array, Array]:
    y_f, stats = _adaptive_solve(f, y0, theta, t0, t1, dt0, rtol, atol, max_steps)
    return y_f, stats.n_steps, stats.n_accepted


def _adaptive_flat_fwd(
    f: _FlatRHS,
    y0: Array,
    theta: Any,
    t0: float,
    t1: float,
    dt0: float,
    rtol: float,
    atol: float,
    max_steps: int,
) -> tuple[tuple[Array, Array, Array], tuple[Array, Any]]:
    y_f, stats = _adaptive_solve(f, y0, theta, t0, t1, dt0, rtol, atol, max_steps)
    return (y_f, stats.n_steps, stats.n_accepted), (y_f, theta)


def _adaptive_flat_bwd(
    f: _FlatRHS,
    t0: float,
    t1: float,
    dt0: float,
    rtol: float,
    atol: float,
    max_steps: int,
    res: tuple[Array, Any],
    cotangents: tuple[Array, Array, Array],
) -> tuple[Array, Any]:
    """Continuous-adjoint VJP: integrate the augmented adjoint ODE backwards.

    For ``y' = f(t, y, theta)`` the loss gradient is carried by the adjoint
    ``a = dL/dy(t)`` and the parameter accumulator ``g = dL/dtheta``, which satisfy
    ``da/dt = -(df/dy)^T a`` and ``dg/dt = -(df/dtheta)^T a``. Integrating the
    augmented state ``(y, a, g)`` from ``t1`` back to ``t0`` (recomputing ``y``
    along the way) yields ``a(t0) = y0_bar`` and ``g(t0) = theta_bar`` in a single
    backward solve, independent of how many forward steps were taken. The integer
    step-count outputs carry no gradient.
    """
    y1, theta = res
    y_bar, _, _ = cotangents
    theta_flat, unravel_theta = ravel_pytree(theta)
    n = y1.shape[0]
    p = theta_flat.shape[0]

    def aug_rhs(s: Array, z: Array, _: Any) -> Array:
        # Integrate forward in reversed time ``s = t1 - t`` (so the march runs from
        # ``t1`` back to ``t0`` as ``s`` grows); the sign flips below convert the
        # backward adjoint ODE into this forward-in-``s`` form.
        t = jnp.asarray(t1) - s
        y = z[:n]
        a = z[n : 2 * n]
        _, vjp = jax.vjp(lambda yy, th: f(t, yy, th), y, theta)
        jt_a, bt_a_tree = vjp(a)
        bt_a, _ = ravel_pytree(bt_a_tree)
        dy = -f(t, y, theta)
        return jnp.concatenate([dy, jt_a, bt_a])

    z1 = jnp.concatenate([y1, y_bar, jnp.zeros((p,))])
    z0, _ = _adaptive_solve(
        aug_rhs, z1, None, 0.0, float(t1) - float(t0), dt0, rtol, atol, max_steps
    )
    y0_bar = z0[n : 2 * n]
    theta_bar = unravel_theta(z0[2 * n :])
    return y0_bar, theta_bar


_adaptive_flat.defvjp(_adaptive_flat_fwd, _adaptive_flat_bwd)


class ODEResult(NamedTuple):
    """Outcome of an adaptive :func:`integrate` call.

    Attributes:
        y: Final state pytree at ``t1`` (differentiable w.r.t. ``y0`` and ``theta``).
        n_steps: Total attempted steps (accepted + rejected).
        n_accepted: Accepted steps.
        success: Whether the march reached ``t1`` within ``max_steps``.
    """

    y: Any
    n_steps: Array
    n_accepted: Array
    success: Array


def integrate(
    func: RHS,
    y0: Any,
    t0: ArrayLike,
    t1: ArrayLike,
    theta: Any = None,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    dt0: float = 1e-3,
    max_steps: int = 100_000,
) -> ODEResult:
    """Adaptively integrate ``dy/dt = func(t, y, theta)`` from ``t0`` to ``t1``.

    Uses a Dormand-Prince 5(4) embedded pair with a PI step-size controller and
    returns only the final state. Gradients with respect to ``y0`` and ``theta``
    are exact and come from the continuous-adjoint backward solve (see
    :func:`_adaptive_flat_bwd`), so they cost one adjoint integration regardless of
    how many forward steps the controller took.

    Prefer :func:`odeint` when you want the whole trajectory on a fixed grid;
    prefer :func:`integrate` when only the endpoint matters and adaptive control
    (efficiency, stiffness) is worth it.

    Returns:
        An :class:`ODEResult`.
    """
    flat0, unravel = ravel_pytree(y0)

    def f_flat(t: Array, y: Array, th: Any) -> Array:
        dy = func(t, unravel(y), th)
        flat, _ = ravel_pytree(dy)
        return flat

    y_f, n_steps, n_accepted = _adaptive_flat(
        f_flat, flat0, theta, float(t0), float(t1), dt0, rtol, atol, max_steps
    )
    return ODEResult(
        y=unravel(y_f),
        n_steps=n_steps,
        n_accepted=n_accepted,
        success=n_steps < max_steps,
    )


__all__ = [
    "FIXED_METHODS",
    "IMPLICIT_METHODS",
    "ODEResult",
    "integrate",
    "odeint",
    "odeint_final",
]
