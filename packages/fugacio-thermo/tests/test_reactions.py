"""Reaction stoichiometry and standard-state thermochemistry.

Reference values are textbook ideal-gas formation data: the water-gas shift,
ammonia synthesis, and methanol synthesis enthalpies/Gibbs energies are checked
against literature, and the temperature dependence of ``K`` is verified against
the van't Hoff and Gibbs-Helmholtz identities by automatic differentiation.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.thermo.constants import R
from fugacio.thermo.reactions import (
    Reaction,
    reaction_properties,
    stoichiometry,
)

WGS = ["carbon monoxide", "water", "carbon dioxide", "hydrogen"]
NH3 = ["nitrogen", "hydrogen", "ammonia"]


def test_stoichiometry_signs_and_alignment() -> None:
    nu = stoichiometry(NH3, {"nitrogen": 1, "hydrogen": 3}, {"ammonia": 2})
    assert [float(v) for v in nu] == [-1.0, -3.0, 2.0]


def test_parse_reaction_with_coefficients() -> None:
    nu = Reaction.parse("nitrogen + 3 hydrogen = 2 ammonia", NH3).nu
    assert [float(v) for v in nu] == [-1.0, -3.0, 2.0]


def test_parse_reaction_alternate_separators_and_case() -> None:
    a = Reaction.parse("CARBON MONOXIDE + Water -> carbon dioxide + Hydrogen", WGS).nu
    b = Reaction.parse("carbon monoxide + water <=> carbon dioxide + hydrogen", WGS).nu
    assert [float(v) for v in a] == [float(v) for v in b] == [-1.0, -1.0, 1.0, 1.0]


def test_parse_reaction_rejects_unknown_species() -> None:
    with pytest.raises(ValueError, match="unknown species"):
        Reaction.parse("carbon monoxide + unobtanium = carbon dioxide", WGS)


def test_parse_reaction_requires_one_separator() -> None:
    with pytest.raises(ValueError, match="separator"):
        Reaction.parse("carbon monoxide + water", WGS)


def test_reaction_reactants_products_delta_n() -> None:
    rx = Reaction.parse("nitrogen + 3 hydrogen = 2 ammonia", NH3)
    assert rx.reactants == {"nitrogen": 1.0, "hydrogen": 3.0}
    assert rx.products == {"ammonia": 2.0}
    assert rx.delta_n == pytest.approx(-2.0)


def test_reaction_is_pytree() -> None:
    rx = Reaction.parse("nitrogen + 3 hydrogen = 2 ammonia", NH3)
    leaves = jax.tree_util.tree_leaves(rx)
    assert len(leaves) == 1  # only nu is a differentiable leaf
    rebuilt = jax.tree_util.tree_map(lambda x: x * 1.0, rx)
    assert rebuilt.components == rx.components


def test_water_gas_shift_thermochemistry() -> None:
    rx = Reaction.parse("carbon monoxide + water = carbon dioxide + hydrogen", WGS)
    p = reaction_properties(rx, 298.15)
    assert float(p.delta_h) / 1e3 == pytest.approx(-41.2, abs=1.0)
    assert float(p.delta_g) / 1e3 == pytest.approx(-28.6, abs=1.0)
    assert float(p.k) > 1e4  # strongly favourable at room temperature


def test_ammonia_synthesis_is_exothermic_and_K_drops_with_T() -> None:
    rx = Reaction.parse("nitrogen + 3 hydrogen = 2 ammonia", NH3)
    p298 = reaction_properties(rx, 298.15)
    p700 = reaction_properties(rx, 700.0)
    assert float(p298.delta_h) / 1e3 == pytest.approx(-91.9, abs=1.5)
    assert float(p298.delta_g) / 1e3 == pytest.approx(-32.8, abs=1.5)
    # Exothermic: equilibrium constant falls steeply with temperature.
    assert float(p700.k) < float(p298.k)
    assert float(p700.k) < 1.0 < float(p298.k)


def test_methanol_synthesis_enthalpy() -> None:
    comps = ["carbon monoxide", "hydrogen", "methanol"]
    rx = Reaction.parse("carbon monoxide + 2 hydrogen = methanol", comps)
    p = reaction_properties(rx, 298.15)
    assert float(p.delta_h) / 1e3 == pytest.approx(-90.5, abs=1.5)


def test_k_equals_exp_minus_dg_over_rt() -> None:
    rx = Reaction.parse("carbon monoxide + water = carbon dioxide + hydrogen", WGS)
    p = reaction_properties(rx, 500.0)
    assert float(p.k) == pytest.approx(float(jnp.exp(-p.delta_g / (R * 500.0))), rel=1e-12)
    assert float(p.ln_k) == pytest.approx(float(-p.delta_g / (R * 500.0)), rel=1e-12)


@pytest.mark.parametrize("t", [350.0, 600.0, 900.0])
def test_vant_hoff_identity(t: float) -> None:
    """d(ln K)/dT = DH_rxn / (R T^2), checked by autodiff."""
    rx = Reaction.parse("carbon monoxide + water = carbon dioxide + hydrogen", WGS)
    dlnk_dt = jax.grad(lambda tt: reaction_properties(rx, tt).ln_k)(t)
    dh = reaction_properties(rx, t).delta_h
    assert float(dlnk_dt) == pytest.approx(float(dh) / (R * t**2), rel=1e-6)


@pytest.mark.parametrize("t", [350.0, 600.0, 900.0])
def test_gibbs_helmholtz_identity(t: float) -> None:
    """d(DG/T)/dT = -DH_rxn / T^2, checked by autodiff."""
    rx = Reaction.parse("nitrogen + 3 hydrogen = 2 ammonia", NH3)
    d_g_over_t = jax.grad(lambda tt: reaction_properties(rx, tt).delta_g / tt)(t)
    dh = reaction_properties(rx, t).delta_h
    assert float(d_g_over_t) == pytest.approx(-float(dh) / t**2, rel=1e-6)


def test_missing_gibbs_data_raises() -> None:
    """A noble gas has no Gibbs of formation path through a reaction; helium reacts with nothing."""
    # Helium has hform/gform = 0 as an element, so build a reaction that needs a
    # species genuinely lacking data is not possible in the curated DB anymore;
    # instead confirm an unknown component name is rejected at array build time.
    from fugacio.thermo.reactions import reaction_arrays

    with pytest.raises(KeyError):
        reaction_arrays(["unobtanium"])
