"""Chemical-reaction equilibrium: feasibility, Le Chatelier, and exact gradients.

The solved composition is checked three ways: the activity quotient must equal
``K(T)`` at the root, the extent must stay inside the physically feasible range,
and the qualitative pressure/temperature responses must follow Le Chatelier's
principle. Gradients of conversion (via implicit differentiation through the
solver) are checked against finite differences.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo.components import component_arrays
from fugacio.thermo.reaction_equilibrium import conversion, equilibrium
from fugacio.thermo.reactions import Reaction

NH3 = ["nitrogen", "hydrogen", "ammonia"]
NH3_RX = Reaction.parse("nitrogen + 3 hydrogen = 2 ammonia", NH3)
FEED = jnp.array([1.0, 3.0, 0.0])

SMR = ["methane", "water", "carbon monoxide", "hydrogen", "carbon dioxide"]


def _quotient_residual(rx: Reaction, y: jnp.ndarray, p: float) -> float:
    from fugacio.thermo.constants import P_REF
    from fugacio.thermo.reactions import reaction_properties

    ln_a = jnp.log(y) + jnp.log(jnp.asarray(p) / P_REF)
    ln_k = reaction_properties(rx, _T).ln_k
    return float(jnp.sum(rx.nu * ln_a) - ln_k)


_T = 700.0


def test_composition_normalised_and_feasible() -> None:
    r = equilibrium(NH3_RX, FEED, _T, 100e5)
    assert float(jnp.sum(r.y)) == pytest.approx(1.0, abs=1e-9)
    assert all(float(v) >= 0.0 for v in r.moles)
    # Extent is bounded by complete conversion of the limiting reactant (1 mol N2).
    assert 0.0 < float(r.extent[0]) < 1.0


def test_quotient_equals_K_at_solution() -> None:
    p = 100e5
    r = equilibrium(NH3_RX, FEED, _T, p)
    assert _quotient_residual(NH3_RX, r.y, p) == pytest.approx(0.0, abs=1e-7)


def test_pressure_raises_conversion_for_mole_reducing_reaction() -> None:
    x_lo = float(conversion(equilibrium(NH3_RX, FEED, _T, 10e5), FEED, 0))
    x_hi = float(conversion(equilibrium(NH3_RX, FEED, _T, 300e5), FEED, 0))
    assert x_hi > x_lo  # Delta_n = -2: higher P favours products


def test_temperature_lowers_conversion_for_exothermic_reaction() -> None:
    x_cold = float(conversion(equilibrium(NH3_RX, FEED, 600.0, 100e5), FEED, 0))
    x_hot = float(conversion(equilibrium(NH3_RX, FEED, 800.0, 100e5), FEED, 0))
    assert x_cold > x_hot  # exothermic: heat shifts equilibrium back to reactants


def test_phi_basis_differs_from_ideal_at_high_pressure() -> None:
    arr = component_arrays(NH3)
    y_ideal = float(equilibrium(NH3_RX, FEED, _T, 300e5).y[2])
    y_phi = float(
        equilibrium(
            NH3_RX, FEED, _T, 300e5, basis="phi", tc=arr["tc"], pc=arr["pc"], omega=arr["omega"]
        ).y[2]
    )
    assert y_phi != pytest.approx(y_ideal, abs=1e-3)


def test_conversion_gradient_matches_finite_difference() -> None:
    def x_of_p(p: float) -> jax.Array:
        return conversion(equilibrium(NH3_RX, FEED, _T, p), FEED, 0)

    p0 = 100e5
    g = float(jax.grad(x_of_p)(p0))
    dp = 1e3
    fd = (float(x_of_p(p0 + dp)) - float(x_of_p(p0 - dp))) / (2 * dp)
    assert g == pytest.approx(fd, rel=1e-4)
    assert g > 0.0  # conversion rises with pressure


def test_multireaction_steam_reforming_plus_shift() -> None:
    smr = Reaction.of(SMR, {"methane": 1, "water": 1}, {"carbon monoxide": 1, "hydrogen": 3})
    wgs = Reaction.of(SMR, {"carbon monoxide": 1, "water": 1}, {"carbon dioxide": 1, "hydrogen": 1})
    feed = jnp.array([1.0, 3.0, 1e-6, 1e-6, 1e-6])
    r = equilibrium([smr, wgs], feed, 1100.0, 1e5, max_iter=100)
    assert float(jnp.sum(r.y)) == pytest.approx(1.0, abs=1e-9)
    assert all(float(v) > 0.0 for v in r.moles)

    # Both reaction quotients match their equilibrium constants at the solution.
    from fugacio.thermo.constants import P_REF

    ln_a = jnp.log(r.y) + jnp.log(jnp.asarray(1e5) / P_REF)
    res = jnp.stack([smr.nu @ ln_a, wgs.nu @ ln_a]) - jnp.log(r.k)
    assert float(jnp.max(jnp.abs(res))) < 1e-8


def test_mismatched_components_raises() -> None:
    a = Reaction.parse(
        "carbon monoxide + water = carbon dioxide + hydrogen",
        ["carbon monoxide", "water", "carbon dioxide", "hydrogen"],
    )
    b = Reaction.parse("methane + water = carbon monoxide + 3 hydrogen", SMR)
    with pytest.raises(ValueError, match="same component ordering"):
        equilibrium([a, b], jnp.ones(5), 1000.0, 1e5)
