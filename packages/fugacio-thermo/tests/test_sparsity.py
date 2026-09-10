import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fugacio.thermo.sparsity import SparsityPattern


def test_sparse_coloring_reconstructs_and_differentiates_exact_jacobian():
    size = 11
    rows = tuple(tuple(j for j in range(size) if abs(i - j) <= 1) for i in range(size))
    pattern = SparsityPattern(size, rows)
    colors = pattern.coloring()
    assert max(colors) < 4
    for row in rows:
        assert len({colors[c] for c in row}) == len(row)
    matrix = np.zeros((size, size))
    rng = np.random.default_rng(12)
    for i, row in enumerate(rows):
        matrix[i, list(row)] = rng.normal(size=len(row))

    def residual(x, theta):
        return jnp.array(matrix) @ jnp.sin(theta * x) + x**3

    assemble = pattern.jacobian(residual)
    x = jnp.linspace(0.1, 0.8, size)
    for transform in (lambda f: f, jax.jacfwd):
        actual = jax.jit(transform(lambda t: assemble(x, t).matrix))(0.3)
        expected = transform(lambda t: jax.jacfwd(residual)(x, t))(0.3)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_maximum_matching_against_all_small_patterns():
    for bits in itertools.product((False, True), repeat=9):
        rows = tuple(tuple(j for j in range(3) if bits[i * 3 + j]) for i in range(3))
        pattern = SparsityPattern(3, rows)
        actual = sum(c >= 0 for c in pattern.matching())
        expected = max(
            sum(c in rows[i] for i, c in enumerate(p)) for p in itertools.permutations(range(3))
        )
        assert actual == expected


def test_structural_deficiency_and_conservative_full_pattern():
    pattern = SparsityPattern(4, ((0, 1), (0, 1), (0, 1), (2,)))
    report = pattern.diagnose(equations=("a", "b", "c", "d"), variables=("x", "y", "z", "free"))
    assert report["structural_rank_upper_bound"] == 3
    assert not report["structurally_square_and_matched"]
    assert report["unmatched_equations"] == ["c"]
    assert report["unmatched_variables"] == ["free"]
    assert SparsityPattern(4, (tuple(range(4)),) * 4).diagnose()["colored_directions"] == 4


@pytest.mark.parametrize(
    "columns,rows", [(0, ((),)), (2, ()), (2, ((0, 0),)), (2, ((2,),)), (True, ((0,),))]
)
def test_invalid_patterns(columns, rows):
    with pytest.raises(ValueError):
        SparsityPattern(columns, rows)
