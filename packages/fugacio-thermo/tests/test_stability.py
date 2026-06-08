"""Tangent-plane stability: detect (and not over-detect) liquid-liquid splits."""

import jax.numpy as jnp
import pytest

from fugacio.thermo.activity.models import margules
from fugacio.thermo.stability import liquid_stability, tangent_plane_distance


def test_strong_positive_deviation_is_unstable() -> None:
    # Symmetric Margules with A > 2 has a miscibility gap.
    model = margules(2.5, 2.5)
    res = liquid_stability(model, 300.0, jnp.array([0.5, 0.5]))
    assert not bool(res.stable)
    assert float(res.tpd) < 0.0
    # The detected split is a genuinely different composition than the feed.
    assert abs(float(res.split[0]) - 0.5) > 0.1


def test_mild_deviation_is_stable() -> None:
    model = margules(1.0, 1.0)
    res = liquid_stability(model, 300.0, jnp.array([0.5, 0.5]))
    assert bool(res.stable)
    assert float(res.tpd) >= -1e-6


def test_tangent_plane_distance_zero_at_feed() -> None:
    model = margules(2.0, 1.5)
    z = jnp.array([0.4, 0.6])
    tpd = tangent_plane_distance(lambda w: model.ln_gamma(w, 300.0), z, z)
    assert float(tpd) == pytest.approx(0.0, abs=1e-6)
