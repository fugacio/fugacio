"""Differentiable state estimation: Kalman, EKF, UKF, and moving-horizon estimation.

Control is only half of feedback; the other half is *estimation* -- reconstructing
the plant state (and the unmeasured disturbances) from noisy, partial
measurements, because MPC and every state-feedback law need a state to act on.
This module supplies the standard recursive estimators and their optimization-
based counterpart, all written against `jax.numpy` so they slot into the
differentiable stack:

* `KalmanFilter` -- the exact recursive Bayesian filter for a linear-
  Gaussian model (Joseph-form covariance update for numerical robustness).
* `ExtendedKalmanFilter` -- the Kalman filter on the *autodiff*
  linearization of a nonlinear model; the Jacobians are exact ``jax.jacobian``
  evaluations, not finite differences.
* `UnscentedKalmanFilter` -- the scaled unscented transform (Wan & van der
  Merwe), which propagates a deterministic set of sigma points through the true
  nonlinear maps and is typically more accurate than the EKF for strong
  nonlinearity, with no Jacobians at all.
* `moving_horizon_estimate` -- estimation as *optimization*: fit the state
  trajectory over a sliding window to the measurements subject to the model, with
  an arrival cost summarizing the past. It is the estimation dual of MPC, it can
  honor constraints (state positivity, bounds), and -- because it is solved with
  `fugacio.sim.argmin` -- it differentiates through the optimum. For a
  linear-Gaussian model with no constraints it reproduces the Kalman filter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.mpc.riccati import kalman_gain
from fugacio.sim.optimize import argmin

ArrayLike = Array | float


class GaussianState(NamedTuple):
    """A Gaussian belief over the state.

    Attributes:
        mean: State estimate ``(n,)``.
        cov: Estimate error covariance ``(n, n)``.
    """

    mean: Array
    cov: Array


def _as_matrix(m: ArrayLike, dim: int) -> Array:
    """Coerce a scalar / 1-D / 2-D covariance-like input to a ``(dim, dim)`` matrix."""
    a = jnp.asarray(m, dtype=float)
    if a.ndim == 0:
        return a * jnp.eye(dim)
    if a.ndim == 1:
        return jnp.diag(a)
    return a


def _joseph_update(
    x_prior: Array, cov_prior: Array, c: Array, r: Array, y: Array, y_pred: Array
) -> tuple[GaussianState, Array]:
    """Kalman measurement update in Joseph (symmetric, PSD-preserving) form."""
    n = x_prior.shape[0]
    s = c @ cov_prior @ c.T + r
    gain = jnp.linalg.solve(s.T, (cov_prior @ c.T).T).T  # P C^T S^{-1}
    innovation = y - y_pred
    x_post = x_prior + gain @ innovation
    ikc = jnp.eye(n) - gain @ c
    cov_post = ikc @ cov_prior @ ikc.T + gain @ r @ gain.T
    return GaussianState(mean=x_post, cov=0.5 * (cov_post + cov_post.T)), innovation


# --------------------------------------------------------------------------- #
# Linear Kalman filter
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KalmanFilter:
    """Recursive linear-Gaussian Kalman filter for ``x+ = A x + B u + w``, ``y = C x + D u + v``.

    Construct directly with the model and noise covariances (``q`` process, ``r``
    measurement; scalars/diagonals broadcast). A filter *step* corrects the prior
    with a prediction: `step` predicts from the previous posterior through
    the input and then updates with the new measurement.
    """

    a: Array
    b: Array
    c: Array
    q: Array
    r: Array
    d: Array | None = None

    def predict(self, state: GaussianState, u: ArrayLike = 0.0) -> GaussianState:
        """Time update: propagate the belief one step through the dynamics."""
        a = jnp.asarray(self.a, dtype=float)
        bu = jnp.asarray(self.b, dtype=float) @ jnp.atleast_1d(jnp.asarray(u, dtype=float))
        mean = a @ state.mean + bu
        cov = a @ state.cov @ a.T + _as_matrix(self.q, a.shape[0])
        return GaussianState(mean=mean, cov=0.5 * (cov + cov.T))

    def update(self, state: GaussianState, y: ArrayLike, u: ArrayLike = 0.0) -> GaussianState:
        """Measurement update: correct the belief with an observation ``y``."""
        c = jnp.asarray(self.c, dtype=float)
        y = jnp.atleast_1d(jnp.asarray(y, dtype=float))
        feed = (
            jnp.zeros_like(y)
            if self.d is None
            else jnp.asarray(self.d, dtype=float) @ jnp.atleast_1d(jnp.asarray(u, dtype=float))
        )
        y_pred = c @ state.mean + feed
        post, _ = _joseph_update(
            state.mean, state.cov, c, _as_matrix(self.r, c.shape[0]), y, y_pred
        )
        return post

    def step(self, state: GaussianState, u: ArrayLike, y: ArrayLike) -> GaussianState:
        """One predict-then-update cycle (previous posterior, input, new measurement)."""
        return self.update(self.predict(state, u), y, u)

    def filter(self, state0: GaussianState, us: Array, ys: Array) -> GaussianState:
        """Run the filter over input/measurement sequences; return the belief trajectory.

        ``us`` and ``ys`` have a leading time axis of equal length ``T``; the
        returned `GaussianState` has a leading time axis of length ``T`` on
        ``mean`` and ``cov``.
        """

        def body(
            state: GaussianState, uy: tuple[Array, Array]
        ) -> tuple[GaussianState, GaussianState]:
            u, y = uy
            new = self.step(state, u, y)
            return new, new

        _, traj = jax.lax.scan(body, state0, (jnp.asarray(us, float), jnp.asarray(ys, float)))
        return traj

    def steady_state_gain(self) -> Array:
        """The steady-state Kalman update gain (the dual-DARE solution)."""
        gain, _ = kalman_gain(
            jnp.asarray(self.a, float),
            jnp.asarray(self.c, float),
            _as_matrix(self.q, self.a.shape[0]),
            _as_matrix(self.r, self.c.shape[0]),
        )
        return gain


# --------------------------------------------------------------------------- #
# Extended Kalman filter (autodiff linearization)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExtendedKalmanFilter:
    """Kalman filter on the exact autodiff linearization of a nonlinear model.

    The discrete transition ``f(x, u) -> x+`` and measurement ``h(x) -> y`` are
    arbitrary differentiable callables; the filter linearizes them on the fly with
    `jax.jacobian` (``A = df/dx``, ``C = dh/dx``). For a continuous plant,
    pass a one-step integrator as ``f`` (e.g. a wrapped
    `fugacio.sim.dynamics.odeint_final`).
    """

    f: Callable[[Array, Array], Array]
    h: Callable[[Array], Array]
    q: Array
    r: Array

    def predict(self, state: GaussianState, u: ArrayLike = 0.0) -> GaussianState:
        """Time update: push the belief through ``f`` and propagate covariance via ``df/dx``."""
        u = jnp.atleast_1d(jnp.asarray(u, dtype=float))
        mean = self.f(state.mean, u)
        a = jax.jacobian(lambda x: self.f(x, u))(state.mean).reshape(mean.shape[0], -1)
        cov = a @ state.cov @ a.T + _as_matrix(self.q, mean.shape[0])
        return GaussianState(mean=mean, cov=0.5 * (cov + cov.T))

    def update(self, state: GaussianState, y: ArrayLike) -> GaussianState:
        """Measurement update: fold in ``y`` through the linearized ``dh/dx`` (Joseph form)."""
        y = jnp.atleast_1d(jnp.asarray(y, dtype=float))
        y_pred = jnp.atleast_1d(self.h(state.mean))
        c = jax.jacobian(lambda x: jnp.atleast_1d(self.h(x)))(state.mean).reshape(
            y_pred.shape[0], -1
        )
        post, _ = _joseph_update(
            state.mean, state.cov, c, _as_matrix(self.r, y_pred.shape[0]), y, y_pred
        )
        return post

    def step(self, state: GaussianState, u: ArrayLike, y: ArrayLike) -> GaussianState:
        """One predict-then-update cycle for input ``u`` and measurement ``y``."""
        return self.update(self.predict(state, u), y)

    def filter(self, state0: GaussianState, us: Array, ys: Array) -> GaussianState:
        """Run the EKF over input/measurement sequences; return the belief trajectory."""

        def body(
            state: GaussianState, uy: tuple[Array, Array]
        ) -> tuple[GaussianState, GaussianState]:
            u, y = uy
            new = self.step(state, u, y)
            return new, new

        _, traj = jax.lax.scan(body, state0, (jnp.asarray(us, float), jnp.asarray(ys, float)))
        return traj


# --------------------------------------------------------------------------- #
# Unscented Kalman filter (scaled unscented transform)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UnscentedKalmanFilter:
    """Sigma-point (unscented) Kalman filter for a nonlinear model.

    Propagates ``2n + 1`` deterministic sigma points through the true nonlinear
    transition ``f(x, u)`` and measurement ``h(x)`` and reconstructs the Gaussian
    by weighted moments -- no Jacobians, and accurate to higher order than the EKF
    for strongly nonlinear maps. ``alpha``/``beta``/``kappa`` are the standard
    spread parameters.
    """

    f: Callable[[Array, Array], Array]
    h: Callable[[Array], Array]
    q: Array
    r: Array
    alpha: float = field(default=1e-3)
    beta: float = field(default=2.0)
    kappa: float = field(default=0.0)

    def _weights(self, n: int) -> tuple[Array, Array, float]:
        lam = self.alpha**2 * (n + self.kappa) - n
        wm = jnp.concatenate(
            [jnp.array([lam / (n + lam)]), jnp.full((2 * n,), 1.0 / (2.0 * (n + lam)))]
        )
        wc = wm.at[0].add(1.0 - self.alpha**2 + self.beta)
        return wm, wc, lam

    def _sigma_points(self, mean: Array, cov: Array, lam: float) -> Array:
        n = mean.shape[0]
        sqrt = jnp.linalg.cholesky((n + lam) * (cov + 1e-12 * jnp.eye(n)))
        pts = [mean]
        for i in range(n):
            pts.append(mean + sqrt[:, i])
        for i in range(n):
            pts.append(mean - sqrt[:, i])
        return jnp.stack(pts, axis=0)

    def predict(self, state: GaussianState, u: ArrayLike = 0.0) -> GaussianState:
        """Time update: propagate the sigma points through ``f`` and rebuild the Gaussian."""
        u = jnp.atleast_1d(jnp.asarray(u, dtype=float))
        n = state.mean.shape[0]
        wm, wc, lam = self._weights(n)
        sig = self._sigma_points(state.mean, state.cov, lam)
        prop = jax.vmap(lambda x: self.f(x, u))(sig)
        mean = wm @ prop
        dev = prop - mean
        cov = jnp.einsum("i,ij,ik->jk", wc, dev, dev) + _as_matrix(self.q, n)
        return GaussianState(mean=mean, cov=0.5 * (cov + cov.T))

    def update(self, state: GaussianState, y: ArrayLike) -> GaussianState:
        """Measurement update: fold in ``y`` via the sigma-point output cross-covariance."""
        y = jnp.atleast_1d(jnp.asarray(y, dtype=float))
        n = state.mean.shape[0]
        wm, wc, lam = self._weights(n)
        sig = self._sigma_points(state.mean, state.cov, lam)
        ypts = jax.vmap(lambda x: jnp.atleast_1d(self.h(x)))(sig)
        y_mean = wm @ ypts
        dy = ypts - y_mean
        dx = sig - state.mean
        p_yy = jnp.einsum("i,ij,ik->jk", wc, dy, dy) + _as_matrix(self.r, y_mean.shape[0])
        p_xy = jnp.einsum("i,ij,ik->jk", wc, dx, dy)
        gain = jnp.linalg.solve(p_yy.T, p_xy.T).T
        mean = state.mean + gain @ (y - y_mean)
        cov = state.cov - gain @ p_yy @ gain.T
        return GaussianState(mean=mean, cov=0.5 * (cov + cov.T))

    def step(self, state: GaussianState, u: ArrayLike, y: ArrayLike) -> GaussianState:
        """One predict-then-update cycle for input ``u`` and measurement ``y``."""
        return self.update(self.predict(state, u), y)

    def filter(self, state0: GaussianState, us: Array, ys: Array) -> GaussianState:
        """Run the UKF over input/measurement sequences; return the belief trajectory."""

        def body(
            state: GaussianState, uy: tuple[Array, Array]
        ) -> tuple[GaussianState, GaussianState]:
            u, y = uy
            new = self.step(state, u, y)
            return new, new

        _, traj = jax.lax.scan(body, state0, (jnp.asarray(us, float), jnp.asarray(ys, float)))
        return traj


# --------------------------------------------------------------------------- #
# Moving-horizon estimation (estimation as optimization)
# --------------------------------------------------------------------------- #
class MHEResult(NamedTuple):
    """Outcome of `moving_horizon_estimate`.

    Attributes:
        x: Estimate of the state at the *end* of the window (the current state).
        trajectory: The full estimated state trajectory over the window ``(N, n)``.
        x0: The estimated state at the start of the window.
        noise: The estimated process-noise sequence ``(N - 1, n)``.
    """

    x: Array
    trajectory: Array
    x0: Array
    noise: Array


def moving_horizon_estimate(
    f: Callable[[Array, Array], Array],
    h: Callable[[Array], Array],
    us: Array,
    ys: Array,
    x_prior: Array,
    *,
    q: ArrayLike,
    r: ArrayLike,
    p0: ArrayLike,
    state_bounds: tuple[Any, Any] | None = None,
    max_iter: int = 200,
) -> MHEResult:
    """Estimate the current state by fitting the model to a window of measurements.

    Minimizes the arrival cost plus the stage costs over a window of ``N``
    measurements::

        ||x0 - x_prior||^2_{P0^{-1}}
            + sum_k ||w_k||^2_{Q^{-1}} + sum_k ||y_k - h(x_k)||^2_{R^{-1}}

    subject to ``x_{k+1} = f(x_k, u_k) + w_k``, over the decision variables ``x0``
    and the process-noise sequence ``w``. The optimum is found with
    `fugacio.sim.argmin`, so the estimate is differentiable (and optional
    box ``state_bounds`` are enforced). For a linear-Gaussian model with no bounds
    this reproduces the Kalman filter's posterior mean.

    Args:
        f: Discrete transition ``f(x, u) -> x+``.
        h: Measurement map ``h(x) -> y``.
        us: Window inputs ``(N - 1, m)``.
        ys: Window measurements ``(N, p)``.
        x_prior: Prior mean of the window's first state ``(n,)``.
        q: Process-noise covariance ``(n, n)`` (scalar/diagonal broadcast).
        r: Measurement-noise covariance ``(p, p)`` (scalar/diagonal broadcast).
        p0: Arrival-cost covariance ``(n, n)`` on ``x0 - x_prior``.
        state_bounds: Optional ``(lower, upper)`` box on every windowed state.
        max_iter: Optimizer iteration cap.

    Returns:
        An `MHEResult`.
    """
    us = jnp.atleast_2d(jnp.asarray(us, dtype=float))
    ys = jnp.atleast_2d(jnp.asarray(ys, dtype=float))
    x_prior = jnp.asarray(x_prior, dtype=float)
    n = x_prior.shape[0]
    horizon = ys.shape[0]
    q_inv = jnp.linalg.inv(_as_matrix(q, n))
    r_inv = jnp.linalg.inv(_as_matrix(r, ys.shape[1]))
    p0_inv = jnp.linalg.inv(_as_matrix(p0, n))

    def rollout(x0: Array, w: Array) -> Array:
        def body(x: Array, uw: tuple[Array, Array]) -> tuple[Array, Array]:
            u, wk = uw
            x_next = f(x, u) + wk
            return x_next, x_next

        _, rest = jax.lax.scan(body, x0, (us, w))
        return jnp.concatenate([x0[None, :], rest], axis=0)

    def objective(z: dict[str, Array], _: Any) -> Array:
        x0 = z["x0"]
        w = z["w"]
        traj = rollout(x0, w)
        e0 = x0 - x_prior
        cost = e0 @ p0_inv @ e0
        cost = cost + jnp.sum(jax.vmap(lambda wk: wk @ q_inv @ wk)(w))

        def meas_term(xk: Array, yk: Array) -> Array:
            e = yk - jnp.atleast_1d(h(xk))
            return e @ r_inv @ e

        cost = cost + jnp.sum(jax.vmap(meas_term)(traj, ys))
        return cost

    z0 = {"x0": x_prior, "w": jnp.zeros((horizon - 1, n))}
    bounds = None
    if state_bounds is not None:
        lo, hi = state_bounds
        bounds = (
            {"x0": jnp.broadcast_to(jnp.asarray(lo, float), (n,)), "w": -jnp.inf},
            {"x0": jnp.broadcast_to(jnp.asarray(hi, float), (n,)), "w": jnp.inf},
        )
    z_star = argmin(objective, z0, None, bounds=bounds, max_iter=max_iter)
    traj = rollout(z_star["x0"], z_star["w"])
    return MHEResult(x=traj[-1], trajectory=traj, x0=z_star["x0"], noise=z_star["w"])


__all__ = [
    "ExtendedKalmanFilter",
    "GaussianState",
    "KalmanFilter",
    "MHEResult",
    "UnscentedKalmanFilter",
    "moving_horizon_estimate",
]
