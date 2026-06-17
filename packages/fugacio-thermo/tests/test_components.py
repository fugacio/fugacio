"""Database self-consistency: every curated record must be internally coherent."""

import jax.numpy as jnp
import pytest

from fugacio.thermo import components as comp
from fugacio.thermo.constants import ATM, R

# Species for which "Antoine reproduces 1 atm at the normal boiling point" does not
# physically hold, so they are excluded from that consistency check:
#   * helium:      a quantum cryogen (Tb = 4.2 K); the Antoine fit cannot be trusted
#                  in that regime.
#   * acetylene:   its triple point (~1.27 atm) lies above 1 atm, so it sublimes and
#                  has no normal (1 atm) boiling point; the tabulated "Tb" is a
#                  sublimation temperature.
_ANTOINE_TB_EXCLUDE = {"helium", "acetylene"}
WITH_ANTOINE = [
    name
    for name, c in comp.DATABASE.items()
    if c.antoine is not None and c.tb is not None and name not in _ANTOINE_TB_EXCLUDE
]
WITH_CP = [name for name, c in comp.DATABASE.items() if c.cp_ig is not None]
WITH_VC_ZC = [name for name, c in comp.DATABASE.items() if c.vc is not None and c.zc is not None]


def test_database_non_empty() -> None:
    assert len(comp.DATABASE) >= 20


@pytest.mark.parametrize("name", WITH_ANTOINE)
def test_antoine_normal_boiling_point_is_one_atm(name: str) -> None:
    c = comp.get(name)
    a = c.antoine
    assert a is not None and c.tb is not None
    p_bar = 10.0 ** (a.a - a.b / (c.tb + a.c))
    assert p_bar * 1e5 == pytest.approx(ATM, rel=0.03)


@pytest.mark.parametrize("name", WITH_VC_ZC)
def test_critical_compressibility_consistency(name: str) -> None:
    c = comp.get(name)
    assert c.vc is not None and c.zc is not None
    zc_calc = c.pc * c.vc / (R * c.tc)
    assert zc_calc == pytest.approx(c.zc, abs=0.01)


@pytest.mark.parametrize("name", WITH_CP)
def test_ideal_gas_cp_is_physical(name: str) -> None:
    c = comp.get(name)
    cp = c.cp_ig
    assert cp is not None
    value = R * (cp.a + cp.b * 298.15 + cp.c * 298.15**2 + cp.d / 298.15**2 + cp.e * 298.15**3)
    # Cp must not dip below the monatomic ideal-gas floor (5/2 R; the fitted
    # polynomials for the noble gases sit exactly on it, so allow float wiggle)
    # and must stay sane: the database tops out near n-eicosane's ~466 J/mol/K.
    assert 2.5 * R * (1.0 - 1e-6) <= value < 600.0


def test_all_components_have_positive_critical_constants() -> None:
    for c in comp.DATABASE.values():
        assert c.tc > 0.0 and c.pc > 0.0 and c.mw > 0.0


def test_known_cp_values() -> None:
    cases = {"nitrogen": 29.1, "carbon dioxide": 37.1, "water": 33.6, "methane": 35.1}
    for name, expected in cases.items():
        cp = comp.get(name).cp_ig
        assert cp is not None
        value = R * (cp.a + cp.b * 298.15 + cp.c * 298.15**2 + cp.d / 298.15**2)
        assert value == pytest.approx(expected, abs=1.0)


def test_get_is_case_insensitive() -> None:
    assert comp.get("WATER").name == "water"
    assert comp.get("  Benzene ").name == "benzene"


def test_get_unknown_raises() -> None:
    with pytest.raises(KeyError):
        comp.get("unobtanium")


def test_component_arrays_shapes() -> None:
    arr = comp.component_arrays(["methane", "propane", "n-pentane"])
    assert arr["tc"].shape == (3,)
    assert jnp.all(arr["pc"] > 0.0)
