"""Differential testing against open reference codes (opt-in; ``pytest -m oracle``).

These tests grade Fugacio's Peng-Robinson core against independent open-source
references (CoolProp's Helmholtz equations of state and the ``chemicals`` Wagner
correlations). They are marked ``oracle`` and excluded from the default suite
because the reference libraries are heavy optional dependencies; run them with
``just oracles``. Agreement is asserted to a tolerance because the references use
higher-accuracy models than a cubic EOS -- the goal is to catch gross errors and
regressions, not to assert bit-for-bit equality.
"""

import jax.numpy as jnp
import pytest

from fugacio.thermo import components as comp
from fugacio.thermo import oracles
from fugacio.thermo.eos import PR, compressibility
from fugacio.thermo.equilibrium import psat_eos

pytestmark = [
    pytest.mark.oracle,
    pytest.mark.skipif(not oracles.has_coolprop(), reason="CoolProp not installed"),
]

SAT_CASES = [("propane", 300.0), ("n-hexane", 341.88), ("benzene", 353.24), ("n-pentane", 309.22)]
Z_CASES = [("nitrogen", 300.0, 50e5), ("methane", 300.0, 50e5), ("propane", 350.0, 5e5)]


@pytest.mark.parametrize("name,t", SAT_CASES)
def test_psat_matches_coolprop(name: str, t: float) -> None:
    c = comp.get(name)
    ours = float(psat_eos(PR, t, c.tc, c.pc, c.omega))
    ref = oracles.coolprop_psat(name, t)
    assert ours == pytest.approx(ref, rel=0.05)


@pytest.mark.parametrize("name,t,p", Z_CASES)
def test_compressibility_matches_coolprop(name: str, t: float, p: float) -> None:
    arr = comp.component_arrays([name])
    z_ours = float(
        compressibility(
            PR, t, p, jnp.array([1.0]), arr["tc"], arr["pc"], arr["omega"], phase="vapor"
        )
    )
    z_ref = oracles.coolprop_compressibility(name, t, p)
    assert z_ours == pytest.approx(z_ref, rel=0.03)


@pytest.mark.skipif(not oracles.has_thermo(), reason="thermo library not installed")
@pytest.mark.parametrize("name,t", [("benzene", 353.24), ("n-hexane", 341.88)])
def test_psat_matches_thermo_library(name: str, t: float) -> None:
    c = comp.get(name)
    ours = float(psat_eos(PR, t, c.tc, c.pc, c.omega))
    ref = oracles.thermo_psat(name, t)
    assert ours == pytest.approx(ref, rel=0.10)
