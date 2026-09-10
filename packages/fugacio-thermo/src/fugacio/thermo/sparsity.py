"""Declared residual incidence, deterministic coloring, and structural diagnostics."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.thermo.linear import DenseJacobian


@dataclass(frozen=True)
class SparsityPattern:
    """Conservative row dependencies in a fixed scalar unknown order.

    Every true derivative entry must appear in ``rows``. Extra entries cost
    work but don't change the equations. Never infer this contract from zeros
    at a single operating point. Unknown custom blocks should declare every
    variable until their dependencies are known.
    """

    columns: int
    rows: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if isinstance(self.columns, bool) or not isinstance(self.columns, int) or self.columns < 1:
            raise ValueError("sparsity pattern needs a positive integer column count")
        if not self.rows:
            raise ValueError("sparsity pattern needs at least one equation")
        for row in self.rows:
            if len(set(row)) != len(row) or any(
                isinstance(c, bool) or not isinstance(c, int) or not 0 <= c < self.columns
                for c in row
            ):
                raise ValueError("row dependencies must be unique valid column indices")

    def coloring(self) -> tuple[int, ...]:
        """Color the column-intersection graph in a deterministic greedy order."""
        neighbors: list[set[int]] = [set() for _ in range(self.columns)]
        for row in self.rows:
            group = set(row)
            for column in row:
                neighbors[column].update(group - {column})
        colors = [-1] * self.columns
        for column in sorted(range(self.columns), key=lambda c: (-len(neighbors[c]), c)):
            forbidden = {colors[c] for c in neighbors[column]}
            color = 0
            while color in forbidden:
                color += 1
            colors[column] = color
        return tuple(colors)

    def jacobian(self, residual: Callable[..., Array]) -> Callable[[Array, Any], DenseJacobian]:
        """Create an exact colored AD assembler with a dense reference solve.

        The result still stores a dense global matrix. This adapter reduces
        differentiation work for sparse flowsheet connections; it doesn't
        claim a sparse factorization or expand a procedural unit's equations.
        """
        if len(self.rows) != self.columns:
            raise ValueError("Newton Jacobian assembly requires a square pattern")
        colors = self.coloring()
        directions = max(colors) + 1
        row_indices = tuple(r for r, columns in enumerate(self.rows) for _ in columns)
        column_indices = tuple(c for columns in self.rows for c in columns)

        def assemble(x: Array, theta: Any) -> DenseJacobian:
            if x.ndim != 1 or x.size != self.columns:
                raise ValueError("unknown vector doesn't match the sparsity pattern")
            _, push = jax.linearize(lambda value: residual(value, theta), x)
            seeds = jax.nn.one_hot(jnp.asarray(colors), directions, dtype=x.dtype).T
            compressed = jax.vmap(push)(seeds).T
            rows, cols = jnp.asarray(row_indices, dtype=int), jnp.asarray(column_indices, dtype=int)
            values = compressed[rows, jnp.asarray(colors)[cols]]
            matrix = (
                jnp.zeros((len(self.rows), self.columns), dtype=x.dtype).at[rows, cols].set(values)
            )
            return DenseJacobian(matrix)

        return assemble

    def matching(self) -> tuple[int, ...]:
        """Find a maximum equation-to-variable matching; unmatched rows contain -1.

        Breadth-first augmenting paths avoid Python recursion limits on long
        process trains. A complete matching is only a structural upper bound
        on numerical rank, especially with conservative block dependencies.
        """
        owners = [-1] * self.columns
        matched = [-1] * len(self.rows)
        for initial in range(len(self.rows)):
            queue = deque([initial])
            visited_rows = {initial}
            parent: dict[int, int] = {}
            free = -1
            while queue and free < 0:
                row = queue.popleft()
                for column in self.rows[row]:
                    if column in parent:
                        continue
                    parent[column] = row
                    owner = owners[column]
                    if owner < 0:
                        free = column
                        break
                    if owner not in visited_rows:
                        visited_rows.add(owner)
                        queue.append(owner)
            while free >= 0:
                row = parent[free]
                previous = matched[row]
                matched[row], owners[free] = free, row
                free = previous
        return tuple(matched)

    def diagnose(
        self, *, equations: Sequence[str] | None = None, variables: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Describe incidence and unmatched equations/variables without a numerical solve."""
        equations = (
            tuple(equations) if equations is not None else tuple(map(str, range(len(self.rows))))
        )
        variables = (
            tuple(variables) if variables is not None else tuple(map(str, range(self.columns)))
        )
        if len(equations) != len(self.rows) or len(variables) != self.columns:
            raise ValueError("diagnostic labels don't match the incidence dimensions")
        match = self.matching()
        rank = sum(c >= 0 for c in match)
        used = set(match)
        nonzero = sum(map(len, self.rows))
        return {
            "n_unknowns": self.columns,
            "n_equations": len(self.rows),
            "degrees_of_freedom": self.columns - len(self.rows),
            "structural_rank_upper_bound": rank,
            "structurally_square_and_matched": rank == self.columns == len(self.rows),
            "declared_nonzeros": nonzero,
            "density": nonzero / (len(self.rows) * self.columns),
            "colored_directions": max(self.coloring()) + 1,
            "unmatched_equations": [equations[r] for r, c in enumerate(match) if c < 0],
            "unmatched_variables": [variables[c] for c in range(self.columns) if c not in used],
            "scope": (
                "Declared incidence only; a matching doesn't establish numerical rank "
                "or physical validity."
            ),
        }
