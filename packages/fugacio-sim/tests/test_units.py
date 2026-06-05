"""Differentiable unit operations: stream pytree, flash drum, and mixer balances."""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import Stream, flash_drum, mix


def test_stream_total_and_fractions() -> None:
    s = Stream.from_fractions(("methane", "propane"), jnp.array([0.7, 0.3]), 10.0, 300.0, 5e5)
    assert float(s.total) == pytest.approx(10.0)
    assert float(s.z[0]) == pytest.approx(0.7)
    assert float(s.n[1]) == pytest.approx(3.0)


def test_stream_is_a_pytree() -> None:
    s = Stream.from_fractions(("methane", "propane"), jnp.array([0.7, 0.3]), 10.0, 300.0, 5e5)
    leaves, treedef = jax.tree_util.tree_flatten(s)
    rebuilt = jax.tree_util.tree_unflatten(treedef, [leaf * 2 for leaf in leaves])
    assert float(rebuilt.total) == pytest.approx(20.0)
    assert rebuilt.components == ("methane", "propane")


def test_flash_drum_conserves_mass() -> None:
    feed = Stream.from_fractions(
        ("methane", "propane", "n-pentane"), jnp.array([0.5, 0.3, 0.2]), 100.0, 320.0, 20e5
    )
    vapor, liquid = flash_drum(feed, 320.0, 20e5)
    assert float(vapor.total + liquid.total) == pytest.approx(float(feed.total), rel=1e-9)
    per_component = vapor.n + liquid.n - feed.n
    assert float(jnp.max(jnp.abs(per_component))) < 1e-6


def test_flash_drum_recovery_is_differentiable_in_temperature() -> None:
    feed = Stream.from_fractions(
        ("methane", "propane", "n-pentane"), jnp.array([0.5, 0.3, 0.2]), 100.0, 320.0, 20e5
    )

    def vapor_flow(t: float) -> jax.Array:
        vapor, _ = flash_drum(feed, t, 20e5)
        return vapor.total

    grad = float(jax.grad(vapor_flow)(320.0))
    fd = float((vapor_flow(320.5) - vapor_flow(319.5)) / 1.0)
    assert grad == pytest.approx(fd, rel=1e-3)
    assert grad > 0.0  # heating the drum makes more vapor


def test_mix_material_balance() -> None:
    a = Stream.from_fractions(("methane", "propane"), jnp.array([1.0, 0.0]), 4.0, 300.0, 5e5)
    b = Stream.from_fractions(("methane", "propane"), jnp.array([0.0, 1.0]), 6.0, 310.0, 4e5)
    out = mix([a, b])
    assert float(out.total) == pytest.approx(10.0)
    assert float(out.z[0]) == pytest.approx(0.4)
    assert float(out.p) == pytest.approx(4e5)  # lowest inlet pressure
    assert float(out.t) == pytest.approx((4.0 * 300.0 + 6.0 * 310.0) / 10.0)


def test_mix_rejects_mismatched_components() -> None:
    a = Stream.from_fractions(("methane",), jnp.array([1.0]), 1.0, 300.0, 5e5)
    b = Stream.from_fractions(("propane",), jnp.array([1.0]), 1.0, 300.0, 5e5)
    with pytest.raises(ValueError):
        mix([a, b])
