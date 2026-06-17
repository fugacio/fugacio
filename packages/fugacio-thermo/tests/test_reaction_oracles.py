"""Differential tests: Fugacio reaction thermochemistry / equilibrium vs Cantera.

These are opt-in *oracle* tests (marker: ``oracle``), excluded from the default
suite and skipped unless Cantera is importable. The Cantera ideal-gas phase is
built from Fugacio's *own* standard formation data and ideal-gas ``Cp``
coefficients (see :func:`fugacio.thermo.oracles._cantera_ideal_gas`), so the
comparison isolates the temperature-integration kernel and the equilibrium
*solver* rather than a difference in reference data. Agreement is therefore
expected to (near) machine precision, not merely in the ballpark.
"""

import jax.numpy as jnp
import pytest

from fugacio.thermo import oracles
from fugacio.thermo.reaction_equilibrium import equilibrium
from fugacio.thermo.reactions import Reaction, reaction_properties

pytestmark = [
    pytest.mark.oracle,
    pytest.mark.skipif(not oracles.HAVE_CANTERA, reason="cantera not installed"),
]

# Ammonia synthesis N2 + 3 H2 = 2 NH3 (Delta_n = -2, strongly pressure sensitive).
_NH3 = ["nitrogen", "hydrogen", "ammonia"]
_NH3_RX = Reaction.parse("nitrogen + 3 hydrogen = 2 ammonia", _NH3)

# Esterification AcOH + EtOH = EtOAc + H2O (Delta_n = 0, pressure independent).
_ESTER = ["acetic acid", "ethanol", "ethyl acetate", "water"]
_ESTER_RX = Reaction.parse("acetic acid + ethanol = ethyl acetate + water", _ESTER)

# Steam-methane reforming + water-gas shift (two simultaneous reactions).
_SMR = ["methane", "water", "carbon monoxide", "hydrogen", "carbon dioxide"]


@pytest.mark.parametrize("t", [298.15, 400.0, 600.0, 800.0])
def test_reaction_properties_match_cantera_ammonia(t: float) -> None:
    """Standard DH/DS/DG and K(T) for ammonia synthesis agree with Cantera."""
    fp = reaction_properties(_NH3_RX, t)
    cp = oracles.cantera_reaction_properties(_NH3_RX, t)
    assert float(fp.delta_h) == pytest.approx(cp["delta_h"], rel=1e-6, abs=1e-2)
    assert float(fp.delta_s) == pytest.approx(cp["delta_s"], rel=1e-6, abs=1e-4)
    assert float(fp.delta_g) == pytest.approx(cp["delta_g"], rel=1e-6, abs=1e-2)
    assert float(fp.ln_k) == pytest.approx(cp["ln_k"], rel=1e-6, abs=1e-6)
    assert float(fp.k) == pytest.approx(cp["k"], rel=1e-6)


@pytest.mark.parametrize("t", [350.0, 500.0])
def test_reaction_properties_match_cantera_esterification(t: float) -> None:
    """Standard DG and K(T) for liquid-free esterification thermochemistry agree."""
    fp = reaction_properties(_ESTER_RX, t)
    cp = oracles.cantera_reaction_properties(_ESTER_RX, t)
    assert float(fp.delta_h) == pytest.approx(cp["delta_h"], rel=1e-6, abs=1e-2)
    assert float(fp.delta_g) == pytest.approx(cp["delta_g"], rel=1e-6, abs=1e-2)
    assert float(fp.ln_k) == pytest.approx(cp["ln_k"], rel=1e-6, abs=1e-6)


@pytest.mark.parametrize(("t", "p"), [(700.0, 5e5), (700.0, 50e5), (550.0, 100e5)])
def test_equilibrium_composition_matches_cantera_ammonia(t: float, p: float) -> None:
    """Ammonia equilibrium composition matches Cantera's Gibbs minimiser at P up to 100 bar."""
    feed = [1.0, 3.0, 0.0]
    y_fug = equilibrium(_NH3_RX, jnp.asarray(feed), t, p).y
    y_ct = oracles.cantera_equilibrium_composition(_NH3, feed, t, p)
    for got, want in zip([float(v) for v in y_fug], y_ct, strict=True):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-8)


def test_equilibrium_composition_matches_cantera_esterification() -> None:
    """Equimolar esterification equilibrium composition matches Cantera."""
    feed = [1.0, 1.0, 0.0, 0.0]
    y_fug = equilibrium(_ESTER_RX, jnp.asarray(feed), 355.0, 1e5).y
    y_ct = oracles.cantera_equilibrium_composition(_ESTER, feed, 355.0, 1e5)
    for got, want in zip([float(v) for v in y_fug], y_ct, strict=True):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-8)


def test_equilibrium_composition_matches_cantera_multireaction() -> None:
    """Simultaneous SMR + water-gas-shift equilibrium matches Cantera."""
    smr = Reaction.of(_SMR, {"methane": 1, "water": 1}, {"carbon monoxide": 1, "hydrogen": 3})
    wgs = Reaction.of(
        _SMR, {"carbon monoxide": 1, "water": 1}, {"carbon dioxide": 1, "hydrogen": 1}
    )
    feed = [1.0, 3.0, 1e-6, 1e-6, 1e-6]
    y_fug = equilibrium([smr, wgs], jnp.asarray(feed), 1100.0, 1e5, max_iter=100).y
    y_ct = oracles.cantera_equilibrium_composition(_SMR, feed, 1100.0, 1e5)
    for got, want in zip([float(v) for v in y_fug], y_ct, strict=True):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-8)
