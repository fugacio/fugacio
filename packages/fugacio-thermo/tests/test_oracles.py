"""Differential tests: Fugacio activity / bubble-point results vs the ``thermo`` library.

These are *oracle* tests (marker: ``oracle``) and are excluded from the default
suite. Run them with ``just oracles`` (``uv run --group oracles pytest -m oracle``).
Each test is skipped if its reference backend is not installed, so the file always
collects cleanly.

The activity-coefficient oracles reuse Fugacio's own UNIFAC group assignments, so a
discrepancy points at the kernel rather than at differing group splits. NRTL/UNIQUAC
oracles pass identical interaction parameters to both implementations, isolating the
algebra and Fugacio's ``tau`` convention.
"""

import jax.numpy as jnp
import pytest

from fugacio.thermo import oracles
from fugacio.thermo.activity.models import nrtl, uniquac
from fugacio.thermo.activity.wilson import wilson_lambda, wilson_ln_gamma
from fugacio.thermo.components import component_arrays
from fugacio.thermo.gammaphi import bubble_pressure_gamma
from fugacio.thermo.groupcontrib.dortmund import modified_unifac_activity
from fugacio.thermo.groupcontrib.unifac import unifac_activity

pytestmark = [
    pytest.mark.oracle,
    pytest.mark.skipif(not oracles.HAVE_THERMO, reason="thermo not installed"),
]

# Representative liquid compositions and temperatures for ethanol(1)/water(2).
_X_GRID = [
    jnp.array([0.1, 0.9]),
    jnp.array([0.3, 0.7]),
    jnp.array([0.5, 0.5]),
    jnp.array([0.8, 0.2]),
]
_T_GRID = [313.15, 343.15, 373.15]

# NRTL ethanol(1)/water(2): tau_ij = b_ij / T, constant non-randomness alpha.
_NRTL_B = [[0.0, 670.0], [310.0, 0.0]]
_NRTL_ALPHA = [[0.0, 0.3], [0.3, 0.0]]

# UNIQUAC ethanol(1)/water(2): r/q from UNIFAC, tau_ij = exp(a_ij + b_ij / T).
_UNIQUAC_R = [2.1055, 0.92]
_UNIQUAC_Q = [1.972, 1.4]
_UNIQUAC_A = [[0.0, 0.5], [-0.3, 0.0]]
_UNIQUAC_B = [[0.0, -50.0], [40.0, 0.0]]

# Wilson ethanol(1)/water(2): molar volumes (cm3/mol) and energy differences (J/mol).
_WILSON_VOLUME = [58.68, 18.07]
_WILSON_ENERGY = [[0.0, 1000.0], [2500.0, 0.0]]


def _nrtl_model() -> object:
    return nrtl(a=jnp.zeros((2, 2)), b=jnp.array(_NRTL_B), alpha=jnp.array(_NRTL_ALPHA))


def _uniquac_model() -> object:
    return uniquac(
        r=jnp.array(_UNIQUAC_R),
        q=jnp.array(_UNIQUAC_Q),
        a=jnp.array(_UNIQUAC_A),
        b=jnp.array(_UNIQUAC_B),
    )


@pytest.mark.parametrize("x", _X_GRID)
@pytest.mark.parametrize("t", _T_GRID)
def test_nrtl_gamma_matches_thermo(x: jnp.ndarray, t: float) -> None:
    model = _nrtl_model()
    gamma = jnp.exp(model.ln_gamma(x, t))  # type: ignore[attr-defined]
    ref = oracles.thermo_nrtl_gamma([float(v) for v in x], t, _NRTL_B, _NRTL_ALPHA)
    for got, want in zip([float(g) for g in gamma], ref, strict=True):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("x", _X_GRID)
@pytest.mark.parametrize("t", _T_GRID)
def test_uniquac_gamma_matches_thermo(x: jnp.ndarray, t: float) -> None:
    model = _uniquac_model()
    gamma = jnp.exp(model.ln_gamma(x, t))  # type: ignore[attr-defined]
    ref = oracles.thermo_uniquac_gamma(
        [float(v) for v in x], t, _UNIQUAC_R, _UNIQUAC_Q, _UNIQUAC_A, _UNIQUAC_B
    )
    for got, want in zip([float(g) for g in gamma], ref, strict=True):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("x", _X_GRID)
@pytest.mark.parametrize("t", _T_GRID)
def test_unifac_classic_gamma_matches_thermo(x: jnp.ndarray, t: float) -> None:
    comps = ["ethanol", "water"]
    gamma = jnp.exp(unifac_activity(comps, x, t))
    ref = oracles.thermo_unifac_gamma(comps, [float(v) for v in x], t, dortmund=False)
    for got, want in zip([float(g) for g in gamma], ref, strict=True):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("x", _X_GRID)
@pytest.mark.parametrize("t", _T_GRID)
def test_unifac_dortmund_gamma_matches_thermo(x: jnp.ndarray, t: float) -> None:
    comps = ["ethanol", "water"]
    gamma = jnp.exp(modified_unifac_activity(comps, x, t))
    ref = oracles.thermo_unifac_gamma(comps, [float(v) for v in x], t, dortmund=True)
    for got, want in zip([float(g) for g in gamma], ref, strict=True):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("x", _X_GRID)
@pytest.mark.parametrize("t", [313.15, 343.15])
def test_bubble_pressure_gamma_matches_modified_raoult(x: jnp.ndarray, t: float) -> None:
    """Fugacio's gamma-phi bubble pressure equals an independent modified-Raoult sum.

    The reference combines ``thermo``'s NRTL activity coefficients with Fugacio's own
    EOS saturation pressures, so this checks the gamma-phi bubble assembly with a
    gamma source that is independent of Fugacio's kernel.
    """
    comps = ["ethanol", "water"]
    arr = component_arrays(comps)
    model = _nrtl_model()
    p_fugacio, _ = bubble_pressure_gamma(model, t, x, arr["tc"], arr["pc"], arr["omega"])

    gamma_ref = oracles.thermo_nrtl_gamma([float(v) for v in x], t, _NRTL_B, _NRTL_ALPHA)
    p_ref = oracles.modified_raoult_bubble_pressure(comps, [float(v) for v in x], t, gamma_ref)

    assert float(p_fugacio) == pytest.approx(p_ref, rel=1e-6)


@pytest.mark.parametrize("x", _X_GRID)
@pytest.mark.parametrize("t", _T_GRID)
def test_wilson_gamma_matches_thermo(x: jnp.ndarray, t: float) -> None:
    """Fugacio's Wilson kernel matches ``thermo`` given the same ``Lambda`` matrix."""
    lam = wilson_lambda(t, jnp.array(_WILSON_VOLUME), jnp.array(_WILSON_ENERGY))
    gamma = jnp.exp(wilson_ln_gamma(x, lam))
    ref = oracles.thermo_wilson_gamma(
        [float(v) for v in x], [[float(c) for c in row] for row in lam]
    )
    for got, want in zip([float(g) for g in gamma], ref, strict=True):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-8)


@pytest.mark.skipif(not oracles.HAVE_CLAPEYRON, reason="juliacall/Clapeyron.jl not installed")
def test_clapeyron_unifac_gamma_matches_fugacio() -> None:
    """Clapeyron.jl UNIFAC activity coefficients agree with Fugacio's (loose tolerance).

    Group parameters can differ slightly between the libraries, so this only checks
    they are in the same ballpark. Skipped unless ``juliacall`` (with Clapeyron.jl)
    is available, so it never runs in the hermetic CI suite.
    """
    comps = ["ethanol", "water"]
    x = jnp.array([0.3, 0.7])
    t = 343.15
    gamma = jnp.exp(unifac_activity(comps, x, t))
    ref = oracles.clapeyron_gamma(comps, [float(v) for v in x], t, model="UNIFAC")
    for got, want in zip([float(g) for g in gamma], ref, strict=True):
        assert got == pytest.approx(want, rel=0.05)


@pytest.mark.skipif(not oracles.HAVE_CHEMICALS, reason="chemicals not installed")
def test_cas_lookup_matches_known_values() -> None:
    assert oracles.cas_for("water") == "7732-18-5"
    assert oracles.cas_for("ethanol") == "64-17-5"
    assert oracles.cas_for("benzene") == "71-43-2"
