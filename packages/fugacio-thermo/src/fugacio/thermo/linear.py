"""Structured Jacobians and checked, implicitly differentiated linear solves.

The block representation stores a nearest-neighbor chain and a small dense
border. Coloring assembles its Jacobian without a stage-sized autodiff batch.
Block elimination pivots within each diagonal block; a residual check selects
a pivoted dense fallback when that elimination isn't adequate. The
fallback changes the linear algorithm, never the nonlinear equations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import lu_factor, lu_solve


class LinearReport(NamedTuple):
    """Independent linear residual and information about the actual algorithm."""

    accepted: Array
    backward_error: Array
    relative_residual: Array
    used_dense_fallback: Array

    def to_dict(self) -> dict[str, Any]:
        """Return concrete JSON data, retaining a nonfinite residual as null."""
        import math

        error = float(self.backward_error)
        residual = float(self.relative_residual)
        return {
            "accepted": bool(self.accepted),
            "backward_error": error if math.isfinite(error) else None,
            "relative_residual": residual if math.isfinite(residual) else None,
            "used_dense_fallback": bool(self.used_dense_fallback),
        }


class LinearResult(NamedTuple):
    """Solution and residual evidence; a rejected solution has nonfinite values."""

    value: Array
    report: LinearReport


class Jacobian(Protocol):
    """Matrix operations required by Newton and implicit differentiation."""

    def solve(self, rhs: Array) -> Array:
        """Solve for one or several right-hand sides."""
        ...

    def scaled(self, rows: Array, columns: Array) -> Jacobian:
        """Return ``diag(rows) @ self @ diag(columns)``."""
        ...

    def to_dense(self) -> Array:
        """Materialize the matrix for a reference solve or diagnosis."""
        ...


JacobianFn = Callable[[Array, Any], Jacobian]


def _residual_errors(
    matvec: Callable[[Array], Array],
    magnitude_matvec: Callable[[Array], Array],
    x: Array,
    rhs: Array,
) -> tuple[Array, Array]:
    ax = matvec(x)
    # Normwise errors for each RHS. Tiny individual equations or exactly zero
    # solution components mustn't impose componentwise relative accuracy on a
    # pivoted solve. Both scales are homogeneous in RHS/metric units.
    tiny = jnp.finfo(rhs.dtype).tiny
    matrix_norm = jnp.max(magnitude_matvec(jnp.ones_like(x)), axis=0)
    rhs_norm = jnp.max(jnp.abs(rhs), axis=0)
    scale = jnp.maximum(matrix_norm * jnp.max(jnp.abs(x), axis=0) + rhs_norm, tiny)
    residual = jnp.max(jnp.abs(ax - rhs), axis=0)
    # A backward-stable result can still leave an unusably large Newton
    # residual when the solution contains very large components. Also require
    # accuracy relative to each RHS as a whole. Zero individual RHS entries
    # don't impose spurious componentwise relative-accuracy requirements.
    return jnp.max(residual / scale), jnp.max(residual / jnp.maximum(rhs_norm, tiny))


def _tolerance(dtype: Any) -> float:
    return max(1e-10, 100 * float(jnp.finfo(dtype).eps))


class DenseJacobian(NamedTuple):
    """A dense Jacobian using JAX's pivoted linear solve."""

    matrix: Array

    def solve(self, rhs: Array) -> Array:
        """Solve using the differentiable dense reference implementation."""
        return jnp.linalg.solve(self.matrix, rhs)

    def scaled(self, rows: Array, columns: Array) -> DenseJacobian:
        """Apply independent equation and unknown scales."""
        return DenseJacobian(rows[:, None] * self.matrix * columns[None, :])

    def to_dense(self) -> Array:
        """Return the stored matrix."""
        return self.matrix


def dense_jacobian(
    residual: Callable[..., Array], x: Array, theta: Any, *, vectorize: bool = True
) -> DenseJacobian:
    """Linearize a square residual, optionally evaluating directions sequentially.

    ``vectorize=False`` retains one primal linearization and maps over tangent
    directions with a compiled loop. Nested unit solves then needn't acquire
    another tangent batch dimension. The resulting matrix and solve remain
    dense; this option bounds derivative working storage, not matrix storage.
    """
    if vectorize:
        matrix = jax.jacfwd(lambda u: residual(u, theta))(x).reshape(x.size, x.size)
    else:
        _, push = jax.linearize(lambda u: residual(u, theta), x)
        directions = jnp.eye(x.size, dtype=x.dtype)
        matrix = jax.lax.map(lambda v: push(v.reshape(x.shape)).ravel(), directions).T
    return DenseJacobian(matrix)


class _ChainFactor(NamedTuple):
    lu: Array
    pivots: Array
    upper_reduced: Array
    lower: Array

    def solve(self, rhs: Array) -> Array:
        def forward(previous: Array, item: tuple[Array, ...]) -> tuple[Array, Array]:
            lu, pivots, lower, b = item
            value = lu_solve((lu, pivots), b - lower @ previous)
            return value, value

        _, reduced = jax.lax.scan(
            forward,
            jnp.zeros_like(rhs[0]),
            (self.lu, self.pivots, self.lower, rhs),
        )

        def backward(following: Array, item: tuple[Array, Array]) -> tuple[Array, Array]:
            upper, value = item
            x = value - upper @ following
            return x, x

        _, values = jax.lax.scan(
            backward, jnp.zeros_like(rhs[0]), (self.upper_reduced, reduced), reverse=True
        )
        return values


def _chain_factor(lower: Array, diagonal: Array, upper: Array) -> _ChainFactor:
    def eliminate(previous: Array, item: tuple[Array, ...]) -> tuple[Array, tuple[Array, ...]]:
        lo, diag, up = item
        lu, pivots = lu_factor(diag - lo @ previous)
        reduced = lu_solve((lu, pivots), up)
        return reduced, (lu, pivots, reduced)

    _, (lu, pivots, reduced) = jax.lax.scan(
        eliminate, jnp.zeros_like(diagonal[0]), (lower, diagonal, upper)
    )
    return _ChainFactor(lu, pivots, reduced, lower)


class BorderedBlockJacobian(NamedTuple):
    """Block tridiagonal matrix with a small, unrestricted border.

    ``lower``, ``diagonal``, and ``upper`` have shape ``(n, b, b)``. The first
    lower and last upper blocks are zero. Border columns have shape
    ``(n, b, k)``, border rows ``(k, n, b)``, and the corner ``(k, k)``.
    Unknowns and residuals are ordered by block, followed by border entries.
    """

    lower: Array
    diagonal: Array
    upper: Array
    border_columns: Array
    border_rows: Array
    corner: Array

    @property
    def shape(self) -> tuple[int, int]:
        """Square matrix shape."""
        n, b, _ = self.diagonal.shape
        size = n * b + self.corner.shape[0]
        return size, size

    def matvec(self, x: Array) -> Array:
        """Multiply a vector or a matrix of right-hand sides without densifying."""
        n, b, _ = self.diagonal.shape
        tail = x.shape[1:]
        core = x[: n * b].reshape(n, b, -1)
        edge = x[n * b :].reshape(self.corner.shape[0], core.shape[-1])
        previous = jnp.concatenate((jnp.zeros_like(core[:1]), core[:-1]))
        following = jnp.concatenate((core[1:], jnp.zeros_like(core[:1])))
        result = (
            self.lower @ previous
            + self.diagonal @ core
            + self.upper @ following
            + self.border_columns @ edge
        )
        border = (
            self.border_rows.reshape(self.corner.shape[0], n * b) @ core.reshape(n * b, -1)
            + self.corner @ edge
        )
        return jnp.concatenate((result.reshape(n * b, -1), border)).reshape((self.shape[0], *tail))

    def transpose(self) -> BorderedBlockJacobian:
        """Transpose both the chain and its border."""
        zeros = jnp.zeros_like(self.diagonal[:1])
        return BorderedBlockJacobian(
            jnp.swapaxes(jnp.concatenate((zeros, self.upper[:-1])), 1, 2),
            jnp.swapaxes(self.diagonal, 1, 2),
            jnp.swapaxes(jnp.concatenate((self.lower[1:], zeros)), 1, 2),
            jnp.transpose(self.border_rows, (1, 2, 0)),
            jnp.transpose(self.border_columns, (2, 0, 1)),
            self.corner.T,
        )

    def scaled(self, rows: Array, columns: Array) -> BorderedBlockJacobian:
        """Apply row and column scales while preserving the sparsity pattern."""
        n, b, _ = self.diagonal.shape
        r, c = rows[: n * b].reshape(n, b), columns[: n * b].reshape(n, b)
        before = jnp.concatenate((jnp.ones_like(c[:1]), c[:-1]))
        after = jnp.concatenate((c[1:], jnp.ones_like(c[:1])))
        return BorderedBlockJacobian(
            r[:, :, None] * self.lower * before[:, None, :],
            r[:, :, None] * self.diagonal * c[:, None, :],
            r[:, :, None] * self.upper * after[:, None, :],
            r[:, :, None] * self.border_columns * columns[n * b :],
            rows[n * b :, None, None] * self.border_rows * c[None, :, :],
            rows[n * b :, None] * self.corner * columns[None, n * b :],
        )

    def to_dense(self) -> Array:
        """Materialize a reference matrix without differentiating the residual again."""
        n, b, _ = self.diagonal.shape
        size = n * b
        row = jnp.arange(size).reshape(n, b)
        out = jnp.zeros(self.shape, dtype=self.diagonal.dtype)
        out = out.at[row[:, :, None], row[:, None, :]].set(self.diagonal)
        if n > 1:
            out = out.at[row[1:, :, None], row[:-1, None, :]].set(self.lower[1:])
            out = out.at[row[:-1, :, None], row[1:, None, :]].set(self.upper[:-1])
        out = out.at[:size, size:].set(self.border_columns.reshape(size, -1))
        out = out.at[size:, :size].set(self.border_rows.reshape(self.corner.shape[0], size))
        return out.at[size:, size:].set(self.corner)

    def _factor(self) -> tuple[_ChainFactor, Array, tuple[Array, Array] | None]:
        n, b, _ = self.diagonal.shape
        k = self.corner.shape[0]
        factor = _chain_factor(self.lower, self.diagonal, self.upper)
        inverse_border = factor.solve(self.border_columns) if k else self.border_columns
        schur = None
        if k:
            rows = self.border_rows.reshape(k, n * b)
            schur = lu_factor(self.corner - rows @ inverse_border.reshape(n * b, k))
        return factor, inverse_border, schur

    def _block_solve(
        self, rhs: Array, factors: tuple[_ChainFactor, Array, tuple[Array, Array] | None]
    ) -> Array:
        n, b, _ = self.diagonal.shape
        k = self.corner.shape[0]
        factor, inverse_border, schur = factors
        core = rhs[: n * b].reshape(n, b, -1)
        q = factor.solve(core)
        if k:
            rows = self.border_rows.reshape(k, n * b)
            assert schur is not None
            edge = lu_solve(schur, rhs[n * b :].reshape(k, -1) - rows @ q.reshape(n * b, -1))
            q = q - inverse_border @ edge
        else:
            edge = jnp.empty((0, core.shape[2]), dtype=rhs.dtype)
        return jnp.concatenate((q.reshape(n * b, -1), edge)).reshape(rhs.shape)

    def _checked_solve(
        self, rhs: Array, factors: tuple[_ChainFactor, Array, tuple[Array, Array] | None]
    ) -> tuple[Array, LinearReport]:
        candidate = self._block_solve(rhs, factors)
        tol = _tolerance(rhs.dtype)
        magnitude = jax.tree_util.tree_map(jnp.abs, self)
        initial_error, initial_residual = _residual_errors(
            self.matvec, magnitude.matvec, candidate, rhs
        )
        fallback = (
            ~jnp.all(jnp.isfinite(candidate)) | ~(initial_error <= tol) | ~(initial_residual <= tol)
        )
        value = jax.lax.cond(
            fallback, lambda _: jnp.linalg.solve(self.to_dense(), rhs), lambda _: candidate, None
        )
        error, residual = _residual_errors(self.matvec, magnitude.matvec, value, rhs)
        accepted = jnp.all(jnp.isfinite(value)) & (error <= tol) & (residual <= tol)
        return jnp.where(accepted, value, jnp.nan), LinearReport(
            accepted, error, residual, fallback
        )

    def solve_with_info(self, rhs: Array) -> LinearResult:
        """Solve with implicit gradients and independently checked linear residuals.

        The transposed solve has its own factorization and residual check.
        Neither the block pivot choices nor dense fallback choices are
        differentiated. Singular or unresolved equations return NaNs.
        """
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.shape[0]:
            raise ValueError("right-hand side must have shape (size,) or (size, nrhs)")
        # Assemble the forward factors outside the RHS-dependent primitive.
        # jax.linearize can then retain them once for every requested JVP.
        # The primitive differentiates A x = b through matvec, not through LU.
        # Detach before factorization so eager AD doesn't build unused pivot
        # and factor derivatives outside that primitive. matvec below keeps
        # the original matrix, including its higher-order dependence.
        numerical = jax.lax.stop_gradient(self)
        factors = numerical._factor()
        transposed = numerical.transpose()
        transpose_factors = transposed._factor()
        value, report = jax.lax.custom_linear_solve(
            self.matvec,
            rhs,
            solve=lambda _, b: numerical._checked_solve(b, factors),
            transpose_solve=lambda _, b: transposed._checked_solve(b, transpose_factors),
            has_aux=True,
        )
        return LinearResult(value, jax.lax.stop_gradient(report))

    def solve(self, rhs: Array) -> Array:
        """Return the checked, differentiable linear solution."""
        return self.solve_with_info(rhs).value


@dataclass(frozen=True)
class BlockLayout:
    """Static declaration of a chain's exact Jacobian sparsity pattern.

    Each core residual block may depend on its own and neighboring unknown
    blocks and every border unknown. Border equations may depend on every
    unknown. This is a structural contract, not a pattern inferred from zeros
    at one operating point. Use ``check`` against a dense Jacobian when adding
    a model adapter.
    """

    blocks: int
    block_size: int
    border_size: int = 0

    def __post_init__(self) -> None:
        if any(
            isinstance(v, bool) or not isinstance(v, int)
            for v in (self.blocks, self.block_size, self.border_size)
        ):
            raise ValueError("layout sizes must be integers")
        if self.blocks < 1 or self.block_size < 1 or self.border_size < 0:
            raise ValueError("invalid block layout sizes")

    @property
    def size(self) -> int:
        """Total number of unknowns and equations."""
        return self.blocks * self.block_size + self.border_size

    @property
    def directions(self) -> int:
        """Forward coloring directions, independent of chain length beyond three blocks."""
        return min(self.blocks, 3) * self.block_size + self.border_size

    def linearize(
        self, residual: Callable[..., Array], x: Array, theta: Any
    ) -> BorderedBlockJacobian:
        """Assemble the exact block Jacobian using colored JVPs and border VJPs."""
        if x.ndim != 1 or x.size != self.size:
            raise ValueError("unknown shape doesn't match the declared block layout")
        n, b, k = self.blocks, self.block_size, self.border_size
        colors = min(n, 3)
        _, push = jax.linearize(lambda u: residual(u, theta), x)
        slots = jnp.arange(n * b) % (colors * b)
        seeds = jnp.eye(self.directions, dtype=x.dtype)[:, slots]
        seeds = jnp.concatenate(
            (seeds, jnp.eye(self.directions, dtype=x.dtype)[:, colors * b :]), axis=1
        )
        compressed = jax.vmap(push)(seeds).T
        local = compressed[: n * b].reshape(n, b, self.directions)

        def band(offset: int) -> Array:
            neighbor = jnp.arange(n) + offset
            indices = (neighbor % colors)[:, None, None] * b + jnp.arange(b)[None, None, :]
            values = jnp.take_along_axis(local, indices, axis=2)
            return jnp.where(((neighbor >= 0) & (neighbor < n))[:, None, None], values, 0.0)

        if k:
            pull = jax.linear_transpose(push, x)
            border_seeds = jnp.pad(jnp.eye(k, dtype=x.dtype), ((0, 0), (n * b, 0)))
            rows = jax.vmap(lambda v: pull(v)[0])(border_seeds)[:, : n * b].reshape(k, n, b)
        else:
            rows = jnp.empty((0, n, b), dtype=x.dtype)
        return BorderedBlockJacobian(
            band(-1),
            band(0),
            band(1),
            local[:, :, colors * b :],
            rows,
            compressed[n * b :, colors * b :],
        )

    def check(
        self, residual: Callable[..., Array], x: Array, theta: Any, *, tolerance: float = 1e-9
    ) -> dict[str, Any]:
        """Compare declared structure with dense autodiff at a concrete point.

        This diagnostic detects missing couplings at the supplied point. It
        isn't a proof of a pattern's validity throughout the model domain.
        """
        dense = dense_jacobian(residual, x, theta).matrix
        structured = self.linearize(residual, x, theta).to_dense()
        error = float(jnp.max(jnp.abs(dense - structured) / jnp.maximum(jnp.abs(dense), 1.0)))
        return {
            "accepted": error <= tolerance,
            "relative_error": error,
            "size": self.size,
            "directions": self.directions,
        }
