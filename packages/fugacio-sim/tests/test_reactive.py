"""Reactive flash: simultaneous reaction and phase equilibrium.

The esterification ``acetic acid + ethanol <=> ethyl acetate + water`` is the
canonical reactive-distillation system and is equimolar (no net mole change).
`reactive_flash` must return a state where the liquid satisfies reaction
equilibrium (activity quotient equals ``K(T)``) and the V/L material balance
(total moles conserved for the equimolar reaction). Energy-balanced reactive
columns are covered in ``test_reactive_mesh.py``.
"""

import jax.numpy as jnp
import pytest

from fugacio.sim import Stream, reactive_flash
from fugacio.thermo import component_arrays, get
from fugacio.thermo.activity.models import nrtl
from fugacio.thermo.constants import P_REF, R
from fugacio.thermo.ideal import ideal_gas_coeffs
from fugacio.thermo.package import GammaPhiPackage, gamma_phi_package
from fugacio.thermo.reactions import Reaction, delta_g_rxn, reaction_arrays
from fugacio.thermo.reference import liquid_reference_fugacity

COMPS = ("acetic acid", "ethanol", "ethyl acetate", "water")
RX = Reaction.of(COMPS, {"acetic acid": 1, "ethanol": 1}, {"ethyl acetate": 1, "water": 1})
P = 101325.0


def _model() -> GammaPhiPackage:
    arr = component_arrays(list(COMPS))
    alpha = jnp.full((4, 4), 0.3) - jnp.eye(4) * 0.3
    activity = nrtl(a=jnp.zeros((4, 4)), b=jnp.zeros((4, 4)), alpha=alpha)
    cp = ideal_gas_coeffs([get(c) for c in COMPS])
    return gamma_phi_package(activity, arr["tc"], arr["pc"], arr["omega"], cp)


def _feed() -> Stream:
    return Stream(jnp.array([1.0, 1.0, 1e-4, 1e-4]), jnp.asarray(360.0), jnp.asarray(P), COMPS)


def _ln_k(t: float) -> float:
    hf, gf, (a, b, c, d, e) = reaction_arrays(list(COMPS))
    return -float(delta_g_rxn(RX.nu, t, hf, gf, a, b, c, d, e)) / (R * t)


def _ln_quotient(model: object, t: float, x: jnp.ndarray) -> float:
    """Liquid-activity reaction quotient ``sum_i nu_i ln(x_i gamma_i f_i^0/P_ref)``."""
    f_ref, _ = liquid_reference_fugacity(
        model.eos,
        t,
        P,
        model.tc,
        model.pc,
        model.omega,
        poynting=model.poynting,
        phi_saturation=model.phi_saturation,
    )
    ln_a = jnp.log(x) + model.activity.ln_gamma(x, t) + jnp.log(f_ref) - jnp.log(P_REF)
    return float(RX.nu @ ln_a)


def _ln_quotient_vapor(t: float, y: jnp.ndarray) -> float:
    """Ideal-vapour reaction quotient ``sum_i nu_i ln(y_i P/P_ref)`` (equals the
    liquid quotient at VLE, but stays finite when the flash is all-vapour)."""
    ln_a = jnp.log(y) + jnp.log(P / P_REF)
    return float(RX.nu @ ln_a)


# --------------------------------------------------------------------------- #
# Reactive flash
# --------------------------------------------------------------------------- #
def test_reactive_flash_satisfies_reaction_and_phase_equilibrium() -> None:
    model = _model()
    feed = _feed()
    res = reactive_flash(feed, RX, 355.0, P, model)
    # Reaction equilibrium: liquid activity quotient equals K(T).
    assert _ln_quotient(model, 355.0, res.liquid.z) == pytest.approx(_ln_k(355.0), abs=1e-7)
    # The forward reaction proceeds (ester is produced from acid + alcohol).
    assert float(res.extent[0]) > 0.1
    # Equimolar reaction conserves total moles through the flash.
    total = float(jnp.sum(res.vapor.n) + jnp.sum(res.liquid.n))
    assert total == pytest.approx(float(feed.total), rel=1e-9)
    # Valid phase compositions.
    assert 0.0 <= float(res.beta) <= 1.0
    assert bool(jnp.all(res.liquid.n >= -1e-9))


def test_reactive_flash_vapor_fraction_increases_with_temperature() -> None:
    model = _model()
    feed = _feed()
    lo = reactive_flash(feed, RX, 350.0, P, model)
    hi = reactive_flash(feed, RX, 372.0, P, model)
    assert float(hi.beta) >= float(lo.beta)
    assert float(hi.beta) > 0.0  # warmer flash has vaporised
    # Reaction equilibrium holds at both temperatures (checked on the phase that
    # is present: liquid for the cool all-liquid flash, vapour for the warm one).
    assert _ln_quotient(model, 350.0, lo.liquid.z) == pytest.approx(_ln_k(350.0), abs=1e-7)
    assert _ln_quotient_vapor(372.0, hi.vapor.z) == pytest.approx(_ln_k(372.0), abs=1e-7)


def test_reactive_flash_rejects_mismatched_components() -> None:
    model = _model()
    feed = _feed()
    bad = Reaction.of(("water", "ethanol"), {"water": 1}, {"ethanol": 1})
    with pytest.raises(ValueError, match="same order"):
        reactive_flash(feed, bad, 355.0, P, model)
