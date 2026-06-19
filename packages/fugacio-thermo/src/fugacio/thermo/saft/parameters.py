"""The differentiable PC-SAFT parameter set and its combining rules.

`SaftParameters` bundles the pure-component PC-SAFT parameters of a mixture as a
JAX pytree: ``m`` (segment number), ``sigma`` (segment diameter, **metres**),
``epsilon`` (dispersion energy/k, kelvin), the association volume/energy
(``kappa_ab`` / ``epsilon_ab``), per-component association-site counts, and a
symmetric binary correction matrix ``kij``. All of these are differentiable
leaves, so any PC-SAFT property is differentiable with respect to the model
parameters themselves (the basis for `fugacio.thermo.saft.regression`); the
``associating`` flag and optional ``names`` are static metadata.

The unlike-pair combining rules are the standard Lorentz-Berthelot set,

    sigma_ij = (sigma_i + sigma_j) / 2,
    epsilon_ij = sqrt(epsilon_i epsilon_j) (1 - k_ij),

with the CR-1 rule for unlike association
(`fugacio.thermo.saft.association`). They are evaluated lazily inside the
energy terms (`fugacio.thermo.saft.pcsaft`) rather than stored, so a fitted
``kij`` flows straight into every downstream property.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.saft._data import ASSOCIATION_SITES, PURE_SAFT_PARAMS, saft_kij

ArrayLike = Array | float

#: Conversion from angstrom (literature unit for ``sigma``) to metres (SI).
ANGSTROM: float = 1.0e-10


@dataclass(frozen=True)
class SaftParameters:
    """PC-SAFT parameters of a mixture as a differentiable pytree.

    Attributes:
        m: Segment number per component, shape ``(n,)`` (dimensionless).
        sigma: Segment diameter per component, shape ``(n,)`` (**metres**).
        epsilon: Dispersion energy ``epsilon/k`` per component, ``(n,)`` (kelvin).
        kappa_ab: Association volume per component, ``(n,)`` (0 if non-associating).
        epsilon_ab: Association energy ``epsilon_AB/k`` per component, ``(n,)`` (kelvin).
        n_sites_a: Number of type-A (electron-acceptor) sites per component, ``(n,)``.
        n_sites_b: Number of type-B (electron-donor) sites per component, ``(n,)``.
        kij: Symmetric binary dispersion correction, shape ``(n, n)`` (zero diagonal).
        associating: Whether any A-B association is active in the mixture (static).
        names: Optional component labels (static), for diagnostics.
    """

    m: Array
    sigma: Array
    epsilon: Array
    kappa_ab: Array
    epsilon_ab: Array
    n_sites_a: Array
    n_sites_b: Array
    kij: Array
    associating: bool
    names: tuple[str, ...] | None

    @property
    def n_components(self) -> int:
        """Number of components in the parameter set."""
        return int(self.m.shape[0])


_META_FIELDS = ("associating", "names")

jax.tree_util.register_dataclass(
    SaftParameters,
    data_fields=[f.name for f in fields(SaftParameters) if f.name not in _META_FIELDS],
    meta_fields=list(_META_FIELDS),
)


def _as_matrix(kij: Array | None, n: int) -> Array:
    if kij is None:
        return jnp.zeros((n, n))
    arr = jnp.asarray(kij, dtype=float)
    if arr.shape != (n, n):
        raise ValueError(f"kij must have shape {(n, n)}, got {tuple(arr.shape)}")
    return arr


def saft_parameters(
    m: Array,
    sigma: Array,
    epsilon: Array,
    *,
    kappa_ab: Array | None = None,
    epsilon_ab: Array | None = None,
    n_sites_a: Array | None = None,
    n_sites_b: Array | None = None,
    kij: Array | None = None,
    names: Sequence[str] | None = None,
    sigma_in_angstrom: bool = False,
) -> SaftParameters:
    """Assemble a `SaftParameters` pytree from raw parameter arrays.

    Args:
        m: Segment number per component, shape ``(n,)``.
        sigma: Segment diameter per component, ``(n,)`` (metres, unless
            ``sigma_in_angstrom``).
        epsilon: Dispersion energy ``epsilon/k`` per component, ``(n,)`` (kelvin).
        kappa_ab: Association volume per component (``None`` => no association).
        epsilon_ab: Association energy ``epsilon_AB/k`` per component (kelvin).
        n_sites_a: Type-A site count per component (``None`` => zeros).
        n_sites_b: Type-B site count per component (``None`` => zeros).
        kij: Symmetric ``(n, n)`` binary correction (``None`` => zeros).
        names: Optional component labels.
        sigma_in_angstrom: If ``True``, ``sigma`` is given in angstrom and is
            converted to metres.

    Returns:
        A registered `SaftParameters` pytree.
    """
    m_arr = jnp.asarray(m, dtype=float)
    n = int(m_arr.shape[0])
    sigma_arr = jnp.asarray(sigma, dtype=float)
    if sigma_in_angstrom:
        sigma_arr = sigma_arr * ANGSTROM
    kappa = jnp.zeros(n) if kappa_ab is None else jnp.asarray(kappa_ab, dtype=float)
    eps_ab = jnp.zeros(n) if epsilon_ab is None else jnp.asarray(epsilon_ab, dtype=float)
    na = jnp.zeros(n) if n_sites_a is None else jnp.asarray(n_sites_a, dtype=float)
    nb = jnp.zeros(n) if n_sites_b is None else jnp.asarray(n_sites_b, dtype=float)
    associating = bool(float(jnp.sum(na)) > 0.0 and float(jnp.sum(nb)) > 0.0)
    return SaftParameters(
        m=m_arr,
        sigma=sigma_arr,
        epsilon=jnp.asarray(epsilon, dtype=float),
        kappa_ab=kappa,
        epsilon_ab=eps_ab,
        n_sites_a=na,
        n_sites_b=nb,
        kij=_as_matrix(kij, n),
        associating=associating,
        names=tuple(names) if names is not None else None,
    )


def saft_parameters_for(
    components: Sequence[str], *, kij: Array | None = None, use_database_kij: bool = True
) -> SaftParameters:
    """Build a `SaftParameters` set for named components from the curated bank.

    Args:
        components: Component names present in the PC-SAFT parameter bank
            (`fugacio.thermo.saft._data.PURE_SAFT_PARAMS`).
        kij: Explicit ``(n, n)`` binary-correction matrix; takes precedence over
            the curated bank.
        use_database_kij: Fill the binary corrections from the curated
            `fugacio.thermo.saft._data.SAFT_KIJ` set when ``kij`` is not given;
            pairs without a curated value stay at zero.

    Returns:
        The assembled `SaftParameters` pytree.

    Raises:
        KeyError: If any component lacks curated PC-SAFT parameters.
    """
    names = [c.strip().lower() for c in components]
    missing = [c for c in names if c not in PURE_SAFT_PARAMS]
    if missing:
        available = ", ".join(sorted(PURE_SAFT_PARAMS))
        raise KeyError(f"no PC-SAFT parameters for {missing}; available: {available}")
    records = [PURE_SAFT_PARAMS[c] for c in names]
    n = len(records)

    sites = [ASSOCIATION_SITES[r.scheme] for r in records]
    matrix = kij
    if matrix is None and use_database_kij:
        k = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                value = saft_kij(names[i], names[j])
                if value is not None:
                    k[i][j] = k[j][i] = value
        matrix = jnp.asarray(k)

    return saft_parameters(
        m=jnp.asarray([r.m for r in records]),
        sigma=jnp.asarray([r.sigma for r in records]),
        epsilon=jnp.asarray([r.epsilon for r in records]),
        kappa_ab=jnp.asarray([r.kappa_ab for r in records]),
        epsilon_ab=jnp.asarray([r.epsilon_ab for r in records]),
        n_sites_a=jnp.asarray([float(a) for a, _ in sites]),
        n_sites_b=jnp.asarray([float(b) for _, b in sites]),
        kij=matrix,
        names=tuple(names),
        sigma_in_angstrom=True,
    )


def segment_diameter(params: SaftParameters, t: ArrayLike) -> Array:
    """Temperature-dependent segment diameter ``d_i(T)`` (metres).

    The Chen-Kreglewski soft-repulsion diameter
    ``d_i = sigma_i [1 - 0.12 exp(-3 epsilon_i / (k T))]`` used by PC-SAFT.

    Args:
        params: PC-SAFT parameter set.
        t: Temperature (K).

    Returns:
        The per-component effective hard-sphere diameter, shape ``(n,)``.
    """
    t = jnp.asarray(t, dtype=float)
    return params.sigma * (1.0 - 0.12 * jnp.exp(-3.0 * params.epsilon / t))


def sigma_ij(params: SaftParameters) -> Array:
    """Lorentz combining rule ``sigma_ij = (sigma_i + sigma_j) / 2`` (metres)."""
    s = params.sigma
    return 0.5 * (s[:, None] + s[None, :])


def epsilon_ij(params: SaftParameters) -> Array:
    """Berthelot combining rule ``epsilon_ij = sqrt(eps_i eps_j) (1 - k_ij)`` (kelvin)."""
    e = params.epsilon
    geom = jnp.sqrt(e[:, None] * e[None, :])
    return geom * (1.0 - params.kij)


__all__ = [
    "ANGSTROM",
    "SaftParameters",
    "epsilon_ij",
    "saft_parameters",
    "saft_parameters_for",
    "segment_diameter",
    "sigma_ij",
]
