"""Linearize a nonlinear dynamic model about an operating point, by autodiff.

A great deal of control analysis (poles, stability margins, Bode plots,
controllability) is *linear* analysis around a steady state. Fugacio gets the
linear model for free: the plant's right-hand side is already differentiable, so
the state-space matrices are simply its Jacobians,

    ``A = df/dy``, ``B = df/du``, ``C = dg/dy``, ``D = dg/du``,

evaluated at the operating point with `jax.jacobian`. No finite differences,
no hand-derived models: the same source of truth that runs the nonlinear
simulation produces its exact local linearization. From there everything is
standard dense linear algebra on ``(A, B, C, D)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

ArrayLike = Array | float


class StateSpace(NamedTuple):
    """A linear time-invariant state-space model ``x' = A x + B u``, ``y = C x + D u``.

    Attributes:
        a: State matrix ``(n, n)``.
        b: Input matrix ``(n, m)``.
        c: Output matrix ``(p, n)``.
        d: Feedthrough matrix ``(p, m)``.
    """

    a: Array
    b: Array
    c: Array
    d: Array


def _as_2d_cols(j: Array, n: int) -> Array:
    """Reshape a Jacobian to ``(n, m)`` columns (handles scalar-input case)."""
    if j.ndim == 1:
        return j.reshape(n, 1)
    return j.reshape(n, -1)


def linearize(
    f: Callable[..., Array],
    y_op: Array,
    u_op: ArrayLike,
    theta: Any = None,
    *,
    output: Callable[..., Array] | None = None,
) -> StateSpace:
    """Linearize ``y' = f(y, u, theta)`` about ``(y_op, u_op)`` into a `StateSpace`.

    Args:
        f: State derivative ``f(y, u, theta) -> dy`` (``theta`` optional/ignored if
            ``None``). ``y`` is a 1-D state vector; ``u`` is a scalar or 1-D input.
        y_op: Operating-point state (ideally a steady state, ``f(y_op, u_op) ~ 0``).
        u_op: Operating-point input.
        theta: Optional parameters held fixed during linearization.
        output: Optional measurement map ``g(y, u, theta) -> out``; defaults to the
            full state (``C = I``, ``D = 0``).

    Returns:
        The local `StateSpace` ``(A, B, C, D)``.
    """
    y_op = jnp.asarray(y_op, dtype=float)
    u_op = jnp.asarray(u_op, dtype=float)
    n = y_op.shape[0]

    def fy(y: Array) -> Array:
        return jnp.asarray(f(y, u_op, theta))

    def fu(u: Array) -> Array:
        return jnp.asarray(f(y_op, u, theta))

    a = jax.jacobian(fy)(y_op).reshape(n, n)
    b = _as_2d_cols(jax.jacobian(fu)(u_op), n)

    if output is None:
        c = jnp.eye(n)
        d = jnp.zeros((n, b.shape[1]))
        return StateSpace(a=a, b=b, c=c, d=d)

    def gy(y: Array) -> Array:
        return jnp.atleast_1d(jnp.asarray(output(y, u_op, theta)))

    def gu(u: Array) -> Array:
        return jnp.atleast_1d(jnp.asarray(output(y_op, u, theta)))

    p = gy(y_op).shape[0]
    c = jax.jacobian(gy)(y_op).reshape(p, n)
    d = _as_2d_cols(jax.jacobian(gu)(u_op), p)
    return StateSpace(a=a, b=b, c=c, d=d)


def poles(ss: StateSpace) -> Array:
    """Eigenvalues of ``A`` (the system poles)."""
    return jnp.linalg.eigvals(ss.a)


def is_stable(ss: StateSpace) -> Array:
    """Whether every pole has strictly negative real part (asymptotic stability)."""
    return jnp.all(jnp.real(poles(ss)) < 0.0)


def dc_gain(ss: StateSpace) -> Array:
    """Steady-state gain ``-C A^{-1} B + D`` (the response to a unit step at ``t -> inf``)."""
    return -ss.c @ jnp.linalg.solve(ss.a, ss.b) + ss.d


def frequency_response(ss: StateSpace, omega: Array) -> Array:
    """Complex frequency response ``H(j omega) = C (j omega I - A)^{-1} B + D``.

    Returns an array of shape ``(len(omega), p, m)`` of complex transfer-function
    values.
    """
    omega = jnp.asarray(omega, dtype=float)
    n = ss.a.shape[0]
    eye = jnp.eye(n, dtype=complex)
    a_c = ss.a.astype(complex)
    b_c = ss.b.astype(complex)

    def at(w: Array) -> Array:
        return ss.c.astype(complex) @ jnp.linalg.solve(1j * w * eye - a_c, b_c) + ss.d.astype(
            complex
        )

    return jax.vmap(at)(omega)


def bode(ss: StateSpace, omega: Array) -> tuple[Array, Array]:
    """Bode magnitude (dB) and phase (degrees) of a SISO system over ``omega``."""
    h = frequency_response(ss, omega)[:, 0, 0]
    mag_db = 20.0 * jnp.log10(jnp.abs(h) + 1e-300)
    phase_deg = jnp.angle(h) * 180.0 / jnp.pi
    return mag_db, phase_deg


def controllability(ss: StateSpace) -> Array:
    """Controllability matrix ``[B, A B, ..., A^{n-1} B]``."""
    n = ss.a.shape[0]
    cols = [ss.b]
    blk = ss.b
    for _ in range(1, n):
        blk = ss.a @ blk
        cols.append(blk)
    return jnp.concatenate(cols, axis=1)


def observability(ss: StateSpace) -> Array:
    """Observability matrix ``[C; C A; ...; C A^{n-1}]``."""
    n = ss.a.shape[0]
    rows = [ss.c]
    blk = ss.c
    for _ in range(1, n):
        blk = blk @ ss.a
        rows.append(blk)
    return jnp.concatenate(rows, axis=0)


def is_controllable(ss: StateSpace) -> Array:
    """Whether the controllability matrix has full state rank."""
    return jnp.linalg.matrix_rank(controllability(ss)) == ss.a.shape[0]


def is_observable(ss: StateSpace) -> Array:
    """Whether the observability matrix has full state rank."""
    return jnp.linalg.matrix_rank(observability(ss)) == ss.a.shape[0]


__all__ = [
    "StateSpace",
    "bode",
    "controllability",
    "dc_gain",
    "frequency_response",
    "is_controllable",
    "is_observable",
    "is_stable",
    "linearize",
    "observability",
    "poles",
]
