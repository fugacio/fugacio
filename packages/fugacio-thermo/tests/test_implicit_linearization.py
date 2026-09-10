from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fugacio.thermo.implicit import bracketed_root, newton_root
from fugacio.thermo.package import _phase_classification


def test_phase_classification_detaches_inputs_before_flash_linearization():
    derivative_calls = []

    @jax.custom_jvp
    def locate(value):
        return jnp.sin(value)

    @locate.defjvp
    def locate_jvp(primals, tangents):
        derivative_calls.append(True)
        (value,), (direction,) = primals, tangents
        return locate(value), jnp.cos(value) * direction

    class Split(NamedTuple):
        beta: jax.Array

    @jax.tree_util.register_dataclass
    @dataclass
    class Package:
        gain: jax.Array

        def flash_pt(self, temperature, pressure, composition):
            return Split(locate(self.gain * temperature + pressure + composition.sum()))

    def classify(point):
        return _phase_classification(Package(point[0]), point[1], point[2], point[3:]).beta

    point = jnp.array([0.3, 0.7, 0.2, 0.6, 0.4])
    value, push = jax.linearize(classify, point)
    assert value == pytest.approx(jnp.sin(1.41))
    assert push(jnp.ones_like(point)) == 0.0
    assert derivative_calls == []
    np.testing.assert_array_equal(jax.jit(jax.jacrev(classify))(point), jnp.zeros_like(point))
    assert derivative_calls == []
    # The fixture's regular flash does require its custom derivative.
    assert jax.grad(lambda t: Package(point[0]).flash_pt(t, point[2], point[3:]).beta)(
        point[1]
    ) == pytest.approx(0.3 * jnp.cos(1.41))
    assert derivative_calls == [True]


@pytest.mark.parametrize("method", ["bracketed", "newton"])
def test_nested_scalar_roots_share_state_and_parameter_linearizations(method):
    def root(residual, theta):
        if method == "bracketed":
            return bracketed_root(residual, theta, jnp.array(0.1), jnp.array(8.0))
        return newton_root(residual, theta, jnp.array(3.0))

    def solve(parameters):
        def residual(x, theta):
            inside = root(lambda y, value: y**2 - value, x + theta["offset"])
            return inside - theta["target"]

        return root(
            residual,
            {"offset": parameters[0], "target": parameters[1], "fixed": jnp.arange(5.0)},
        )

    # sqrt(x + offset) = target, hence x = target**2 - offset.
    point = jnp.array([1.2, 2.1])
    assert jax.jit(solve)(point) == pytest.approx(3.21, abs=1e-10)
    for transform in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(jax.jit(transform(solve))(point), [-1.0, 4.2], atol=1e-10)
    np.testing.assert_allclose(
        jax.jit(jax.hessian(solve))(point), [[0.0, 0.0], [0.0, 2.0]], atol=1e-9
    )
