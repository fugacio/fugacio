"""Wertheim (TPT1) association for PC-SAFT, with a differentiable site solve.

Hydrogen-bonding fluids (water, alcohols, amines) carry short-range, directional
association that the hard-chain and dispersion terms cannot capture. Fugacio uses
the two-site-type model: every molecule has ``n_A`` electron-acceptor (type A) and
``n_B`` electron-donor (type B) sites, and only A-B bonds form. By the symmetry of
that model all A sites on a component share one bonded fraction ``X_A`` and all B
sites share ``X_B``, so the unknowns collapse to two per component. This scheme
reproduces the Huang-Radosz ``2B`` (alcohols, water), ``3B``, and ``4C`` cases; a
pure self-bonding ``1A`` (e.g. carboxylic-acid dimerisation) is out of scope.

The bonded fractions solve the Wertheim mass-action law

    X_{A,i} = 1 / (1 + rho_N sum_j x_j n_{B,j} X_{B,j} Delta_ij),
    X_{B,i} = 1 / (1 + rho_N sum_j x_j n_{A,j} X_{A,j} Delta_ij),

with the association strength
``Delta_ij = g_ij^hs sigma_ij^3 kappa_ij [exp(epsilon_ij^AB / T) - 1]`` and the
CR-1 combining rules for the unlike association energy and volume. That fixed
point is solved by successive substitution and differentiated by the implicit
function theorem in a `jax.custom_jvp` rule, so the bonded fractions (and the
association energy built from them) are differentiable, in both forward and
reverse mode, with respect to state *and* the PC-SAFT parameters, at machine
precision and independent of the iteration count.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.constants import N_A
from fugacio.thermo.saft.parameters import SaftParameters, segment_diameter

ArrayLike = Array | float


def association_strength(params: SaftParameters, rho: ArrayLike, t: ArrayLike, x: Array) -> Array:
    """Association strength matrix ``Delta_ij`` (m^3) for every component pair.

    ``Delta_ij = g_ij^hs sigma_ij^3 kappa_ij [exp(epsilon_ij^AB / T) - 1]`` with the
    contact radial distribution at the mixture packing fraction and the CR-1
    combining rules for the unlike association parameters.

    Args:
        params: PC-SAFT parameter set.
        rho: Molar density (mol/m^3).
        t: Temperature (K).
        x: Mole fractions, shape ``(n,)``.

    Returns:
        The symmetric ``(n, n)`` association-strength matrix.
    """
    t = jnp.asarray(t, dtype=float)
    x = jnp.asarray(x, dtype=float)
    d = segment_diameter(params, t)
    rho_n = N_A * jnp.asarray(rho, dtype=float)

    coeff = (jnp.pi / 6.0) * rho_n
    z2 = coeff * jnp.sum(x * params.m * d**2)
    z3 = coeff * jnp.sum(x * params.m * d**3)
    one_minus = 1.0 - z3

    dij = (d[:, None] * d[None, :]) / (d[:, None] + d[None, :])
    g_ij = 1.0 / one_minus + dij * 3.0 * z2 / one_minus**2 + dij**2 * 2.0 * z2**2 / one_minus**3

    sigma = params.sigma
    sigma_ij = 0.5 * (sigma[:, None] + sigma[None, :])
    # CR-1 combining rules for the unlike association energy and volume.
    eps_ab = params.epsilon_ab
    eps_ij = 0.5 * (eps_ab[:, None] + eps_ab[None, :])
    kappa = params.kappa_ab
    kappa_ij = (
        jnp.sqrt(kappa[:, None] * kappa[None, :])
        * (jnp.sqrt(sigma[:, None] * sigma[None, :]) / sigma_ij) ** 3
    )
    return g_ij * sigma_ij**3 * kappa_ij * (jnp.exp(eps_ij / t) - 1.0)


def _site_map(state: Array, theta: tuple[SaftParameters, Array, Array, Array]) -> Array:
    """One successive-substitution sweep of the Wertheim mass-action law."""
    params, rho, t, x = theta
    n = params.m.shape[0]
    xb = state[n:]  # the A-site update is Gauss-Seidel, so only X_B is read back
    rho_n = N_A * rho
    delta = association_strength(params, rho, t, x)
    # X_{A,i} couples to the B sites; X_{B,i} couples to the A sites.
    sum_b = rho_n * (delta @ (x * params.n_sites_b * xb))
    xa_new = 1.0 / (1.0 + sum_b)
    sum_a = rho_n * (delta @ (x * params.n_sites_a * xa_new))
    xb_new = 1.0 / (1.0 + sum_a)
    return jnp.concatenate([xa_new, xb_new])


@jax.custom_jvp
def _solve_sites(theta: tuple[SaftParameters, Array, Array, Array]) -> Array:
    """Solve the bonded-site fixed point ``X = G(X, theta)`` by successive substitution."""
    params = theta[0]
    n = params.m.shape[0]
    x0 = jnp.ones(2 * n)

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        prev, cur, i = carry
        return (jnp.max(jnp.abs(cur - prev)) > 1e-13) & (i < 200)

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        _, cur, i = carry
        return cur, _site_map(cur, theta), i + 1

    x1 = _site_map(x0, theta)
    _, x_star, _ = jax.lax.while_loop(cond, body, (x0, x1, jnp.asarray(1)))
    return x_star


@_solve_sites.defjvp
def _solve_sites_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[Array, Array]:
    (theta,) = primals
    (theta_dot,) = tangents
    x_star = _solve_sites(theta)
    # Implicit function theorem on the residual G(X, theta) - X = 0:
    # (I - dG/dX) X_dot = dG/dtheta . theta_dot.
    _, g_theta_dot = jax.jvp(lambda th: _site_map(x_star, th), (theta,), (theta_dot,))
    jac_x = jax.jacobian(lambda xx: _site_map(xx, theta))(x_star)
    a = jnp.eye(x_star.shape[0]) - jac_x
    x_dot = jnp.linalg.solve(a, g_theta_dot)
    return x_star, x_dot


def site_fractions(
    params: SaftParameters, rho: ArrayLike, t: ArrayLike, x: Array
) -> tuple[Array, Array]:
    """Bonded-site fractions ``(X_A, X_B)`` per component at ``(rho, T, x)``.

    Args:
        params: PC-SAFT parameter set.
        rho: Molar density (mol/m^3).
        t: Temperature (K).
        x: Mole fractions, shape ``(n,)``.

    Returns:
        ``(X_A, X_B)``: the fraction of type-A and type-B sites *not* bonded, each
        shape ``(n,)``. Differentiable with respect to state and parameters.
    """
    x = jnp.asarray(x, dtype=float)
    n = params.m.shape[0]
    state = _solve_sites((params, jnp.asarray(rho, dtype=float), jnp.asarray(t, dtype=float), x))
    return state[:n], state[n:]


def alpha_association(params: SaftParameters, rho: ArrayLike, t: ArrayLike, x: Array) -> Array:
    """Association contribution ``alpha_assoc`` to the reduced residual Helmholtz energy.

    ``sum_i x_i [ n_{A,i}(ln X_{A,i} - X_{A,i}/2) + n_{B,i}(ln X_{B,i} - X_{B,i}/2)
    + (n_{A,i} + n_{B,i}) / 2 ]`` (Chapman-Gubbins-Jackson-Radosz).
    """
    x = jnp.asarray(x, dtype=float)
    xa, xb = site_fractions(params, rho, t, x)
    na, nb = params.n_sites_a, params.n_sites_b
    per_component = na * (jnp.log(xa) - 0.5 * xa) + nb * (jnp.log(xb) - 0.5 * xb) + 0.5 * (na + nb)
    return jnp.sum(x * per_component)


__all__ = [
    "alpha_association",
    "association_strength",
    "site_fractions",
]
