"""Curated parameter database: lookups, orientation, and model construction."""

import jax.numpy as jnp
import pytest

from fugacio.thermo import data
from fugacio.thermo.activity.models import nrtl


def test_nrtl_lookup_is_orientation_aware() -> None:
    fwd = data.nrtl_params("ethanol", "water")
    rev = data.nrtl_params("water", "ethanol")
    assert fwd is not None and rev is not None
    # b_ij / b_ji swap; alpha is symmetric.
    assert fwd[0] == pytest.approx(rev[1])
    assert fwd[1] == pytest.approx(rev[0])
    assert fwd[2] == pytest.approx(rev[2])


def test_missing_pair_returns_none() -> None:
    assert data.nrtl_params("argon", "helium") is None
    assert not data.has_nrtl(["argon", "helium"])


def test_uniquac_rq_lookup() -> None:
    r, q = data.uniquac_rq(["ethanol", "water"])
    assert float(r[1]) == pytest.approx(0.92, abs=1e-3)
    assert float(q[1]) == pytest.approx(1.4, abs=1e-3)


def test_uniquac_rq_missing_raises() -> None:
    with pytest.raises(KeyError):
        data.uniquac_rq(["ethanol", "unobtanium"])


def test_nrtl_from_database_matches_manual_model() -> None:
    model = data.nrtl_from_database(["ethanol", "water"])
    b_ij, b_ji, alpha = data.nrtl_params("ethanol", "water")
    manual = nrtl(
        a=jnp.zeros((2, 2)),
        b=jnp.array([[0.0, b_ij], [b_ji, 0.0]]),
        alpha=jnp.array([[0.0, alpha], [alpha, 0.0]]),
    )
    x = jnp.array([0.4, 0.6])
    assert float(jnp.max(jnp.abs(model.ln_gamma(x, 350.0) - manual.ln_gamma(x, 350.0)))) < 1e-6


def test_strict_missing_pair_raises() -> None:
    with pytest.raises(KeyError):
        data.nrtl_from_database(["argon", "helium"], strict=True)


def test_pr_kij_lookup_is_orientation_insensitive() -> None:
    fwd = data.pr_kij("carbon dioxide", "benzene")
    rev = data.pr_kij("benzene", "carbon dioxide")
    assert fwd is not None
    assert fwd == pytest.approx(rev)
    assert fwd == pytest.approx(0.0774, abs=1e-4)


def test_pr_kij_missing_pair_returns_none() -> None:
    assert data.pr_kij("argon", "helium") is None


def test_kij_from_database_is_symmetric_with_zero_diagonal() -> None:
    comps = ["carbon dioxide", "methane", "benzene"]
    k = data.kij_from_database(comps)
    assert k.shape == (3, 3)
    assert bool(jnp.allclose(k, k.T))
    assert float(jnp.max(jnp.abs(jnp.diag(k)))) == 0.0
    # The curated CO2/benzene coefficient lands in the matrix.
    assert float(k[0, 2]) == pytest.approx(0.0774, abs=1e-4)
