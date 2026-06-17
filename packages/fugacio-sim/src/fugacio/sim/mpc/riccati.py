"""Riccati equations, LQR, and the steady-state Kalman gain -- differentiably.

The algebraic Riccati equation is the algebraic heart of optimal control and
optimal estimation: the infinite-horizon LQR feedback law, the terminal cost that
makes a finite-horizon MPC stabilizing, and the steady-state Kalman filter all
reduce to its stabilizing solution. Fugacio needs those objects *inside* a
differentiable stack -- an LQR terminal weight that depends on a model parameter,
a Kalman gain whose covariances are being tuned -- so the solvers here are written
against `jax.numpy` and are differentiable end to end.

The trick that keeps them differentiable is the same one
`fugacio.sim.dynamics.odeint` uses for implicit integrator steps: a *fixed*
iteration count. Both Riccati solvers converge **quadratically**, so a fixed,
generous number of doubling / Newton steps reaches machine precision and -- being
a plain `jax.lax.scan` of dense linear algebra -- is reverse-mode
differentiable without any custom rule.

* **Discrete ARE** (`dare`) is solved by the structure-preserving doubling
  algorithm (SDA): each step squares the effective horizon, so ~30 iterations span
  a horizon of ``2**30`` and the solution is converged to roundoff.
* **Continuous ARE** (`care`) is solved by the matrix-sign-function Newton
  iteration on the Hamiltonian with determinantal scaling, then the stabilizing
  solution is recovered by Roberts' least-squares extraction from the sign matrix.

From the stabilizing solution the gains are one linear solve away
(`dlqr`, `lqr`), and the steady-state Kalman gain
(`kalman_gain`) is the dual discrete problem -- estimation is control run
backwards.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

ArrayLike = Array | float

#: Default fixed iteration count for the (quadratically convergent) Riccati solvers.
_DEFAULT_ITERS = 50


def _symm(m: Array) -> Array:
    """Symmetrize a matrix, suppressing the roundoff asymmetry the recursions accrue."""
    return 0.5 * (m + m.T)


def _as_matrix(m: ArrayLike) -> Array:
    """Coerce a scalar / 1-D / 2-D input to a 2-D float matrix."""
    a = jnp.asarray(m, dtype=float)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return jnp.diag(a)
    return a


# --------------------------------------------------------------------------- #
# Discrete-time algebraic Riccati equation (structure-preserving doubling)
# --------------------------------------------------------------------------- #
def dare(
    a: Array,
    b: Array,
    q: Array,
    r: Array,
    *,
    iters: int = _DEFAULT_ITERS,
) -> Array:
    """Stabilizing solution ``X`` of the discrete algebraic Riccati equation.

    Solves ``A^T X A - X - A^T X B (R + B^T X B)^{-1} B^T X A + Q = 0`` for the
    unique symmetric positive-semidefinite stabilizing solution by the
    structure-preserving doubling algorithm. The iteration is a fixed-length
    `jax.lax.scan`, so ``X`` is differentiable with respect to every input
    matrix.

    Args:
        a: State matrix ``(n, n)``.
        b: Input matrix ``(n, m)``.
        q: State weight ``(n, n)`` (symmetric PSD); a 1-D input is read as a
            diagonal, a scalar as ``q * I``.
        r: Input weight ``(m, m)`` (symmetric PD); same broadcasting as ``q``.
        iters: Number of doubling steps (each squares the horizon; the default is
            comfortably into the converged regime).

    Returns:
        The symmetric stabilizing solution ``X`` of shape ``(n, n)``.
    """
    a = jnp.asarray(a, dtype=float)
    b = jnp.asarray(b, dtype=float)
    n = a.shape[0]
    q = _as_matrix(q)
    r = _as_matrix(r)
    eye = jnp.eye(n)
    g0 = b @ jnp.linalg.solve(r, b.T)  # B R^{-1} B^T

    def step(
        carry: tuple[Array, Array, Array], _: object
    ) -> tuple[tuple[Array, Array, Array], None]:
        a_k, g_k, p_k = carry
        m = eye + g_k @ p_k
        minv_a = jnp.linalg.solve(m, a_k)  # (I + G P)^{-1} A
        minv_g = jnp.linalg.solve(m, g_k)  # (I + G P)^{-1} G
        a_next = a_k @ minv_a
        g_next = _symm(g_k + a_k @ minv_g @ a_k.T)
        p_next = _symm(p_k + a_k.T @ p_k @ minv_a)
        return (a_next, g_next, p_next), None

    (_, _, p_star), _ = jax.lax.scan(step, (a, g0, _symm(q)), None, length=iters)
    return p_star


def dlqr(
    a: Array,
    b: Array,
    q: Array,
    r: Array,
    *,
    iters: int = _DEFAULT_ITERS,
) -> tuple[Array, Array]:
    """Discrete infinite-horizon LQR gain and cost-to-go.

    Returns ``(K, X)`` for the optimal state feedback ``u = -K x`` that minimizes
    ``sum_k x_k^T Q x_k + u_k^T R u_k`` subject to ``x_{k+1} = A x_k + B u_k``,
    with ``X`` the `dare` solution (the cost-to-go ``x_0^T X x_0``) and
    ``K = (R + B^T X B)^{-1} B^T X A``.
    """
    a = jnp.asarray(a, dtype=float)
    b = jnp.asarray(b, dtype=float)
    r = _as_matrix(r)
    x = dare(a, b, q, r, iters=iters)
    k = jnp.linalg.solve(r + b.T @ x @ b, b.T @ x @ a)
    return k, x


# --------------------------------------------------------------------------- #
# Continuous-time algebraic Riccati equation (matrix sign function)
# --------------------------------------------------------------------------- #
def _matrix_sign(z: Array, *, iters: int) -> Array:
    """Matrix sign function via the scaled Newton iteration ``S <- (cS + S^{-1}/c)/2``.

    Determinantal scaling ``c = |det S|^{-1/n}`` accelerates the (quadratically
    convergent) iteration; a fixed step count keeps it differentiable.
    """
    n = z.shape[0]

    def step(s: Array, _: object) -> tuple[Array, None]:
        s_inv = jnp.linalg.inv(s)
        det = jnp.abs(jnp.linalg.det(s))
        c = jnp.where(det > 0.0, det ** (-1.0 / n), 1.0)
        return 0.5 * (c * s + s_inv / c), None

    s_star, _ = jax.lax.scan(step, z, None, length=iters)
    return s_star


def care(
    a: Array,
    b: Array,
    q: Array,
    r: Array,
    *,
    iters: int = 40,
) -> Array:
    """Stabilizing solution ``X`` of the continuous algebraic Riccati equation.

    Solves ``A^T X + X A - X B R^{-1} B^T X + Q = 0`` for the symmetric
    positive-semidefinite stabilizing solution via the matrix sign function of the
    Hamiltonian ``Z = [[A, -G], [-Q, -A^T]]`` (with ``G = B R^{-1} B^T``), followed
    by Roberts' least-squares extraction. Differentiable in every input matrix.
    """
    a = jnp.asarray(a, dtype=float)
    b = jnp.asarray(b, dtype=float)
    n = a.shape[0]
    q = _as_matrix(q)
    r = _as_matrix(r)
    g = b @ jnp.linalg.solve(r, b.T)
    ham = jnp.block([[a, -g], [-q, -a.T]])
    w = _matrix_sign(ham, iters=iters)
    eye = jnp.eye(n)
    w11, w12 = w[:n, :n], w[:n, n:]
    w21, w22 = w[n:, :n], w[n:, n:]
    lhs = jnp.concatenate([w12, w22 + eye], axis=0)
    rhs = -jnp.concatenate([w11 + eye, w21], axis=0)
    x = jnp.linalg.lstsq(lhs, rhs)[0]
    return _symm(x)


def lqr(
    a: Array,
    b: Array,
    q: Array,
    r: Array,
    *,
    iters: int = 40,
) -> tuple[Array, Array]:
    r"""Continuous infinite-horizon LQR gain and cost-to-go.

    Returns ``(K, X)`` for the optimal feedback ``u = -K x`` minimizing
    ``\int x^T Q x + u^T R u`` subject to ``x' = A x + B u``, with ``X`` the
    `care` solution and ``K = R^{-1} B^T X``.
    """
    a = jnp.asarray(a, dtype=float)
    b = jnp.asarray(b, dtype=float)
    r = _as_matrix(r)
    x = care(a, b, q, r, iters=iters)
    k = jnp.linalg.solve(r, b.T @ x)
    return k, x


# --------------------------------------------------------------------------- #
# Steady-state Kalman filter (the dual discrete problem)
# --------------------------------------------------------------------------- #
def kalman_gain(
    a: Array,
    c: Array,
    qn: Array,
    rn: Array,
    *,
    iters: int = _DEFAULT_ITERS,
) -> tuple[Array, Array]:
    """Steady-state discrete Kalman filter gain and prior covariance.

    For ``x_{k+1} = A x_k + w_k`` with process covariance ``Qn`` and measurement
    ``y_k = C x_k + v_k`` with covariance ``Rn``, returns ``(L, P)`` where ``P`` is
    the stabilizing solution of the filtering Riccati equation (the steady-state
    *prior* error covariance) and ``L = P C^T (C P C^T + Rn)^{-1}`` is the
    a-posteriori update gain (``x_hat <- x_prior + L (y - C x_prior)``).

    Estimation is control run backwards: ``P`` is `dare` applied to the dual
    pair ``(A^T, C^T, Qn, Rn)``, so this shares the doubling solver and its
    differentiability.
    """
    a = jnp.asarray(a, dtype=float)
    c = jnp.asarray(c, dtype=float)
    qn = _as_matrix(qn)
    rn = _as_matrix(rn)
    p = dare(a.T, c.T, qn, rn, iters=iters)
    s = c @ p @ c.T + rn
    gain = jnp.linalg.solve(s.T, (p @ c.T).T).T  # P C^T S^{-1}
    return gain, p


def riccati_residual_discrete(a: Array, b: Array, q: Array, r: Array, x: Array) -> Array:
    """Residual ``A^T X A - X - A^T X B (R + B^T X B)^{-1} B^T X A + Q`` (zero at the solution).

    Useful as a self-contained correctness oracle: a valid DARE solution drives
    this matrix to zero with no external reference.
    """
    a = jnp.asarray(a, dtype=float)
    b = jnp.asarray(b, dtype=float)
    q = _as_matrix(q)
    r = _as_matrix(r)
    x = jnp.asarray(x, dtype=float)
    btxb = r + b.T @ x @ b
    btxa = b.T @ x @ a
    return a.T @ x @ a - x - btxa.T @ jnp.linalg.solve(btxb, btxa) + q


def riccati_residual_continuous(a: Array, b: Array, q: Array, r: Array, x: Array) -> Array:
    """Residual ``A^T X + X A - X B R^{-1} B^T X + Q`` of the continuous ARE (zero at solution)."""
    a = jnp.asarray(a, dtype=float)
    b = jnp.asarray(b, dtype=float)
    q = _as_matrix(q)
    r = _as_matrix(r)
    x = jnp.asarray(x, dtype=float)
    g = b @ jnp.linalg.solve(r, b.T)
    return a.T @ x + x @ a - x @ g @ x + q


__all__ = [
    "care",
    "dare",
    "dlqr",
    "kalman_gain",
    "lqr",
    "riccati_residual_continuous",
    "riccati_residual_discrete",
]
