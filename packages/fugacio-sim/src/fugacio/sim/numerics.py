"""Process-level sparse algebra and orchestration with bounded compilation.

Unit physics stays in compiled JAX kernels. Concrete process iterations run on
the host; an enclosing JIT stages the same iteration with ``lax.while_loop``.
Sparse LU runs on the CPU and differentiates its matrix equation, including
transpose solves and higher derivatives. No process-sized dense fallback is
allocated when a sparse factorization fails.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from fugacio.thermo._iteration import is_traced as is_traced
from fugacio.thermo._iteration import while_loop as while_loop
from fugacio.thermo.diagnostics import SolveResult, SolveStatus, residual_report, with_status
from fugacio.thermo.linear import LinearReport, LinearResult

# Factorizations depend on every coefficient and incidence entry. This bounded
# numerical cache only avoids repeating CPU LU for the same linearization's
# multiple directions. It neither stores tracers nor supplies nonlinear seeds.
_FACTOR_CACHE: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
_FACTOR_LOCK = RLock()
_FACTOR_BYTES = 64 * 1024 * 1024
_FACTOR_COUNT = 8


def clear_factor_cache() -> None:
    """Release retained sparse LU factors without clearing compiled unit kernels."""
    with _FACTOR_LOCK:
        _FACTOR_CACHE.clear()


def _sparse_solve(
    values: np.ndarray,
    right: np.ndarray,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    size: int,
    transpose: bool,
) -> np.ndarray:
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(right)):
        return np.full_like(right, np.nan)
    key = (rows, columns, size, values.dtype.str, values.tobytes())
    # SuperLU factors aren't treated as thread safe. The lock protects both
    # eviction and solves; JAX callbacks can otherwise run on multiple threads.
    with _FACTOR_LOCK:
        try:
            cached = _FACTOR_CACHE.get(key)
            if cached is None:
                factor = splu(coo_matrix((values, (rows, columns)), shape=(size, size)).tocsc())
                nbytes = (
                    sum(
                        a.nbytes
                        for m in (factor.L, factor.U)
                        for a in (m.data, m.indices, m.indptr)
                    )
                    + values.nbytes
                )
                if nbytes <= _FACTOR_BYTES:
                    _FACTOR_CACHE[key] = (factor, nbytes)
                    while (
                        len(_FACTOR_CACHE) > _FACTOR_COUNT
                        or sum(item[1] for item in _FACTOR_CACHE.values()) > _FACTOR_BYTES
                    ):
                        _FACTOR_CACHE.popitem(last=False)
            else:
                factor = cached[0]
                _FACTOR_CACHE.move_to_end(key)
            return np.asarray(factor.solve(right, trans="T" if transpose else "N"))
        except (RuntimeError, ValueError):
            return np.full_like(right, np.nan)


@lru_cache(maxsize=64)
def _factor_callback(
    rows: tuple[int, ...], columns: tuple[int, ...], size: int, transpose: bool
) -> Any:
    # Callback identities are structural too. Rebuilding a callback at every
    # point otherwise invalidates JAX's executable cache for its primitive.
    def solve(values: Any, right: Any) -> np.ndarray:
        return _sparse_solve(np.asarray(values), np.asarray(right), rows, columns, size, transpose)

    return solve


@lru_cache(maxsize=64)
def _coalesced_indices(
    rows: tuple[int, ...], columns: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    entries = tuple(zip(rows, columns, strict=True))
    unique = dict.fromkeys(entries)
    groups = {entry: i for i, entry in enumerate(unique)}
    return (
        tuple(groups[entry] for entry in entries),
        tuple(row for row, _ in unique),
        tuple(column for _, column in unique),
    )


@dataclass(frozen=True)
class SparseJacobian:
    """A square coordinate matrix with static incidence and dynamic coefficients.

    Duplicate entries are summed. Storage is proportional to declared nonzeros.
    ``to_dense`` is an explicit diagnostic/reference operation only. The CPU
    solve transfers coefficients when used on an accelerator; it never claims
    an accelerator-native sparse factorization.
    """

    values: Array
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    size: int

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.rows),) or len(self.columns) != len(self.rows):
            raise ValueError("sparse coefficients and incidence have different sizes")
        if self.size < 1 or any(i < 0 or i >= self.size for i in (*self.rows, *self.columns)):
            raise ValueError("invalid sparse matrix index")

    def matvec(self, vector: Array) -> Array:
        """Apply the matrix to a vector or a batch of right-hand sides."""
        values, rows, columns = self._entries()
        weights = values.reshape((-1,) + (1,) * (vector.ndim - 1))
        return (
            jnp.zeros_like(vector)
            .at[jnp.asarray(rows, dtype=int)]
            .add(weights * vector[jnp.asarray(columns, dtype=int)])
        )

    def _entries(self) -> tuple[Array, tuple[int, ...], tuple[int, ...]]:
        groups, rows, columns = _coalesced_indices(self.rows, self.columns)
        if len(rows) == len(self.rows):
            return self.values, self.rows, self.columns
        values = (
            jnp.zeros(len(rows), dtype=self.values.dtype)
            .at[jnp.asarray(groups, dtype=int)]
            .add(self.values)
        )
        return values, rows, columns

    def transpose(self) -> SparseJacobian:
        """Return the transpose without moving or duplicating coefficient storage."""
        return SparseJacobian(self.values, self.columns, self.rows, self.size)

    def scaled(self, rows: Array, columns: Array) -> SparseJacobian:
        """Apply equation and variable scaling without densifying."""
        return SparseJacobian(
            self.values
            * rows[jnp.asarray(self.rows, dtype=int)]
            * columns[jnp.asarray(self.columns, dtype=int)],
            self.rows,
            self.columns,
            self.size,
        )

    def to_dense(self) -> Array:
        """Materialize an explicitly requested small reference matrix."""
        return (
            jnp.zeros((self.size, self.size), dtype=self.values.dtype)
            .at[jnp.asarray(self.rows, dtype=int), jnp.asarray(self.columns, dtype=int)]
            .add(self.values)
        )

    def _factor_solve(self, rhs: Array, *, transpose: bool = False) -> Array:
        return jax.pure_callback(
            _factor_callback(self.rows, self.columns, self.size, transpose),
            jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
            self.values,
            rhs,
            vmap_method="sequential",
        )

    def solve_with_info(self, rhs: Array) -> LinearResult:
        """Solve and check each RHS against the original matrix equation.

        LU pivots carry no derivative. JAX differentiates ``A x = b`` and uses
        the explicit transpose solve for adjoints. A singular or inaccurate
        result is nonfinite and retains independent linear residual evidence.
        """
        rhs = jnp.asarray(rhs, dtype=self.values.dtype)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.size:
            raise ValueError("right-hand side doesn't match the sparse matrix")
        numerical = jax.lax.stop_gradient(self)

        def checked(
            matrix: SparseJacobian, b: Array, transpose: bool = False
        ) -> tuple[Array, LinearReport]:
            value = numerical._factor_solve(b, transpose=transpose)
            residual = jnp.max(jnp.abs(matrix.matvec(value) - b), axis=0)
            tiny = jnp.finfo(b.dtype).tiny
            # COO duplicates must be summed before taking absolute values;
            # cancellation otherwise understates the reported backward error.
            coefficients, unique_rows, _ = matrix._entries()
            row_sums = (
                jnp.zeros(matrix.size, dtype=b.dtype)
                .at[jnp.asarray(unique_rows, dtype=int)]
                .add(jnp.abs(coefficients))
            )
            norm = jnp.max(row_sums)
            bnorm = jnp.max(jnp.abs(b), axis=0)
            xnorm = jnp.max(jnp.abs(value), axis=0)
            backward = jnp.max(residual / jnp.maximum(norm * xnorm + bnorm, tiny))
            relative = jnp.max(residual / jnp.maximum(bnorm, tiny))
            tolerance = max(1e-9, 100 * float(jnp.finfo(b.dtype).eps))
            accepted = (
                jnp.all(jnp.isfinite(value)) & (backward <= tolerance) & (relative <= tolerance)
            )
            report = LinearReport(accepted, backward, relative, jnp.asarray(False))
            return jnp.where(accepted, value, jnp.nan), report

        value, report = jax.lax.custom_linear_solve(
            self.matvec,
            rhs,
            solve=lambda _, b: checked(numerical, b),
            transpose_solve=lambda _, b: checked(numerical.transpose(), b, transpose=True),
            has_aux=True,
        )
        return LinearResult(value, jax.lax.stop_gradient(report))

    def solve(self, rhs: Array) -> Array:
        """Return a checked implicit solve, with NaN on failure."""
        return self.solve_with_info(rhs).value


jax.tree_util.register_dataclass(
    SparseJacobian, data_fields=["values"], meta_fields=["rows", "columns", "size"]
)


def newton_iterations(
    residual: Callable[[Array, Any], Array],
    jacobian: Callable[[Array, Any], Any],
    start: Array,
    parameters: Any,
    *,
    scale: Array | None = None,
    tolerance: float = 1e-9,
    max_iterations: int = 100,
    lower: Array | None = None,
    upper: Array | None = None,
) -> SolveResult:
    """Converge scaled equations without compiling the complete process.

    ``jacobian`` supplies a structured matrix. Bounds restrict trial points;
    residual-decreasing backtracking retains the best state on failure. This
    primal routine attaches no derivative to its iteration history.
    """
    start, parameters = jax.lax.stop_gradient((start, parameters))
    scale = jnp.maximum(jnp.abs(start), 1.0) if scale is None else scale
    lo = jnp.full_like(start, -jnp.inf) if lower is None else lower
    hi = jnp.full_like(start, jnp.inf) if upper is None else upper
    x = jnp.clip(start, lo, hi)
    r = residual(x, parameters) / scale

    def norm(r: Array) -> Array:
        return jnp.max(jnp.abs(r))

    def condition(state: Any) -> Array:
        _, r, iteration, _, failed = state
        return (
            (norm(r) > tolerance)
            & jnp.all(jnp.isfinite(r))
            & (iteration < max_iterations)
            & (failed == SolveStatus.CONVERGED)
        )

    def body(state: Any) -> Any:
        x, r, iteration, _, _ = state
        matrix = jacobian(x, parameters).scaled(1 / scale, scale)
        step = matrix.solve(-r) * scale
        finite = jnp.all(jnp.isfinite(step))
        safe = jnp.where(finite, step, 0.0)

        def search_condition(trial: Any) -> Array:
            return ~trial[-1]

        def search_body(trial: Any) -> Any:
            alpha, _, _, _ = trial
            candidate = jnp.clip(x + alpha * safe, lo, hi)
            candidate_r = residual(candidate, parameters) / scale
            accept = finite & jnp.isfinite(norm(candidate_r)) & (norm(candidate_r) < norm(r))
            done = accept | (alpha <= 1 / 128) | ~finite
            return (
                alpha / 2,
                jnp.where(accept, candidate, x),
                jnp.where(accept, candidate_r, r),
                done,
            )

        _, following, next_r, _ = while_loop(
            search_condition, search_body, (jnp.asarray(1.0), x, r, jnp.asarray(False))
        )
        size = norm((following - x) / scale)
        status = jnp.where(
            ~finite,
            SolveStatus.SINGULAR,
            jnp.where(size == 0, SolveStatus.STALLED, SolveStatus.CONVERGED),
        )
        return following, next_r, iteration + 1, size, status

    x, r, iterations, step, failed = while_loop(
        condition,
        body,
        (x, r, jnp.asarray(0), jnp.asarray(0.0), jnp.asarray(SolveStatus.CONVERGED)),
    )
    report = residual_report(r, tolerance, iterations=iterations, step_norm=step)
    report = with_status(report, (failed != SolveStatus.CONVERGED) & ~report.converged, failed)
    return SolveResult(x, jax.lax.stop_gradient(report))
