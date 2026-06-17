"""Curated parameter lookups: UNIQUAC ``r``/``q`` and binary NRTL / UNIQUAC.

This is the convenience layer over the generated tables in
`fugacio.thermo._binary_params`. It turns component *names* into ready, fully
populated `NRTL` /
`UNIQUAC` model objects, handling the
pair-ordering bookkeeping (the tables are keyed by alphabetically sorted name
pairs) and assembling the ``n x n`` interaction matrices.

Pairs absent from the curated set default to athermal interaction (``b = 0``, so
``gamma -> 1`` for that pair) unless ``strict=True`` is requested, in which case a
missing pair raises. For systems with no curated parameters at all, fall back to
predictive UNIFAC (`fugacio.thermo.groupcontrib.unifac_activity`) or fit
parameters from data with `fugacio.thermo.regression`.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from fugacio.thermo._binary_params import NRTL_BINARY, UNIQUAC_BINARY, UNIQUAC_RQ
from fugacio.thermo._eos_params import PR_KIJ
from fugacio.thermo.activity.models import NRTL, UNIQUAC


def nrtl_params(name_i: str, name_j: str) -> tuple[float, float, float] | None:
    """Oriented binary NRTL params ``(b_ij, b_ji, alpha_ij)`` or ``None`` if absent.

    ``tau_ij = b_ij / T``. Lookup is order-insensitive; the result is oriented so
    that ``b_ij`` is the coefficient acting on the ``i -> j`` direction.
    """
    if (name_i, name_j) in NRTL_BINARY:
        return NRTL_BINARY[(name_i, name_j)]
    if (name_j, name_i) in NRTL_BINARY:
        b_ji, b_ij, alpha = NRTL_BINARY[(name_j, name_i)]
        return b_ij, b_ji, alpha
    return None


def uniquac_params(name_i: str, name_j: str) -> tuple[float, float] | None:
    """Oriented binary UNIQUAC params ``(b_ij, b_ji)`` or ``None`` if absent.

    ``tau_ij = exp(b_ij / T)``; lookup is order-insensitive (see `nrtl_params`).
    """
    if (name_i, name_j) in UNIQUAC_BINARY:
        return UNIQUAC_BINARY[(name_i, name_j)]
    if (name_j, name_i) in UNIQUAC_BINARY:
        b_ji, b_ij = UNIQUAC_BINARY[(name_j, name_i)]
        return b_ij, b_ji
    return None


def pr_kij(name_i: str, name_j: str) -> float | None:
    """Curated Peng-Robinson binary interaction coefficient ``k_ij`` or ``None``.

    Order-insensitive (``k_ij`` is symmetric for the standard one-fluid rule).
    """
    if (name_i, name_j) in PR_KIJ:
        return PR_KIJ[(name_i, name_j)]
    if (name_j, name_i) in PR_KIJ:
        return PR_KIJ[(name_j, name_i)]
    return None


def kij_from_database(components: list[str]) -> Array:
    """Assemble the symmetric Peng-Robinson ``k_ij`` matrix from curated pairs.

    Pairs without a curated value default to ``0`` (ideal van der Waals mixing).
    The result is an ``(n, n)`` array suitable for
    `fugacio.thermo.eos_model` / `fugacio.sim.eos_model_for`.
    """
    n = len(components)
    k = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            value = pr_kij(components[i], components[j])
            if value is not None:
                k[i][j] = k[j][i] = value
    return jnp.asarray(k)


def uniquac_rq(components: list[str]) -> tuple[Array, Array]:
    """Return the UNIQUAC ``(r, q)`` arrays for named components.

    Raises:
        KeyError: if any component lacks curated ``r``/``q`` data.
    """
    missing = [c for c in components if c not in UNIQUAC_RQ]
    if missing:
        raise KeyError(f"no UNIQUAC r/q for: {missing}")
    r = jnp.asarray([UNIQUAC_RQ[c][0] for c in components])
    q = jnp.asarray([UNIQUAC_RQ[c][1] for c in components])
    return r, q


def nrtl_from_database(
    components: list[str], *, strict: bool = False, alpha_default: float = 0.3
) -> NRTL:
    """Build an `NRTL` model for ``components`` from curated binaries.

    Args:
        components: Component names (any length ``>= 2``).
        strict: If ``True``, raise when a pair has no curated parameters; otherwise
            that pair defaults to athermal (``b = 0``).
        alpha_default: Non-randomness used for pairs without curated parameters.

    Returns:
        A fully populated `NRTL` whose ``b`` and ``alpha`` matrices are the
        curated coefficients (``a = e = 0``).
    """
    n = len(components)
    b = [[0.0] * n for _ in range(n)]
    alpha = [[0.0] * n for _ in range(n)]
    missing: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            params = nrtl_params(components[i], components[j])
            if params is None:
                missing.append((components[i], components[j]))
                alpha[i][j] = alpha[j][i] = alpha_default
                continue
            b_ij, b_ji, a_ij = params
            b[i][j] = b_ij
            b[j][i] = b_ji
            alpha[i][j] = alpha[j][i] = a_ij
    if strict and missing:
        raise KeyError(f"no curated NRTL parameters for pairs: {missing}")
    return NRTL(
        a=jnp.zeros((n, n)),
        b=jnp.asarray(b),
        alpha=jnp.asarray(alpha),
        e=jnp.zeros((n, n)),
    )


def uniquac_from_database(components: list[str], *, strict: bool = False) -> UNIQUAC:
    """Build a `UNIQUAC` model for ``components`` from curated data.

    Uses curated ``r``/``q`` and binary ``b`` coefficients. Pairs without data
    default to ``b = 0`` (``tau = 1``) unless ``strict=True``.

    Raises:
        KeyError: if any component lacks ``r``/``q`` (always), or any pair lacks
            interaction parameters (only when ``strict=True``).
    """
    r, q = uniquac_rq(components)
    n = len(components)
    b = [[0.0] * n for _ in range(n)]
    missing: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            params = uniquac_params(components[i], components[j])
            if params is None:
                missing.append((components[i], components[j]))
                continue
            b_ij, b_ji = params
            b[i][j] = b_ij
            b[j][i] = b_ji
    if strict and missing:
        raise KeyError(f"no curated UNIQUAC parameters for pairs: {missing}")
    return UNIQUAC(r=r, q=q, a=jnp.zeros((n, n)), b=jnp.asarray(b))


def has_nrtl(components: list[str]) -> bool:
    """Whether every binary pair in ``components`` has curated NRTL parameters."""
    return all(
        nrtl_params(components[i], components[j]) is not None
        for i in range(len(components))
        for j in range(i + 1, len(components))
    )


def has_uniquac(components: list[str]) -> bool:
    """Whether ``components`` have curated UNIQUAC ``r``/``q`` and all pair params."""
    if any(c not in UNIQUAC_RQ for c in components):
        return False
    return all(
        uniquac_params(components[i], components[j]) is not None
        for i in range(len(components))
        for j in range(i + 1, len(components))
    )
