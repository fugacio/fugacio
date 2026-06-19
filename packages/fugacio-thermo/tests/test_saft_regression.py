"""Differentiable PC-SAFT parameter recovery.

These tests generate synthetic data *from the model itself* at known parameters,
then check that the gradient-based fitters of `fugacio.thermo.saft.regression`
recover those parameters by differentiating straight through the saturation and
bubble-point solvers. Because the data sit exactly on the model, the global
optimum has zero residual at the true parameters, so a successful recovery is
strong evidence that the parameter gradients (through the implicit solvers) are
correct and usable for optimisation.
"""

from dataclasses import replace

import jax.numpy as jnp

from fugacio.thermo import component_arrays
from fugacio.thermo.saft import (
    bubble_pressure_saft,
    fit_saft_kij,
    fit_saft_pure,
    molar_density,
    psat_saft,
    saft_parameters_for,
)


def _wilson_guess(component: str, t: float) -> float:
    arr = component_arrays([component])
    tc, pc, omega = float(arr["tc"][0]), float(arr["pc"][0]), float(arr["omega"][0])
    return pc * float(jnp.exp(5.373 * (1.0 + omega) * (1.0 - tc / t)))


def test_fit_saft_pure_recovers_known_parameters() -> None:
    true = saft_parameters_for(["n-pentane"])
    x = jnp.ones(1)
    temps = jnp.array([280.0, 330.0])
    psat_exp = jnp.array(
        [float(psat_saft(true, float(t), _wilson_guess("n-pentane", float(t)))) for t in temps]
    )
    rho_exp = jnp.array(
        [
            float(molar_density(true, float(t), float(p), x, phase="liquid"))
            for t, p in zip(temps, psat_exp, strict=True)
        ]
    )

    seed = replace(true, m=true.m * 1.04, sigma=true.sigma * 0.985, epsilon=true.epsilon * 1.025)
    fitted, cost = fit_saft_pure(seed, temps, psat_exp, rho_exp, max_iter=40)

    assert float(cost) < 1e-6
    assert abs(float(fitted.m[0]) / float(true.m[0]) - 1.0) < 0.02
    assert abs(float(fitted.sigma[0]) / float(true.sigma[0]) - 1.0) < 0.02
    assert abs(float(fitted.epsilon[0]) / float(true.epsilon[0]) - 1.0) < 0.02


def test_fit_saft_kij_recovers_known_binary_correction() -> None:
    components = ["ethane", "n-heptane"]
    arr = component_arrays(components)
    tc, pc, omega = arr["tc"], arr["pc"], arr["omega"]
    base = saft_parameters_for(components, use_database_kij=False)

    true_kij = 0.02
    true = replace(base, kij=jnp.array([[0.0, true_kij], [true_kij, 0.0]]))
    t = 320.0
    x = jnp.array([0.3, 0.5, 0.7])
    p_exp = jnp.array(
        [
            float(bubble_pressure_saft(true, t, jnp.array([xi, 1.0 - xi]), tc, pc, omega)[0])
            for xi in x
        ]
    )

    fitted_kij, cost = fit_saft_kij(base, t, x, p_exp, tc, pc, omega, max_iter=30)

    assert float(cost) < 1e-8
    assert abs(float(fitted_kij) - true_kij) < 2e-3
