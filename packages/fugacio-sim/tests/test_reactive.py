"""Reactive separations: simultaneous reaction + phase equilibrium.

The esterification ``acetic acid + ethanol <=> ethyl acetate + water`` is the
canonical reactive-distillation system and is *equimolar* (no net mole change), so
constant molar overflow is exact and the column balances close to machine
precision. Tests check three things:

* :func:`reactive_flash` returns a state where the liquid simultaneously satisfies
  reaction equilibrium (activity quotient equals ``K(T)``) and the V/L material
  balance (total moles conserved for the equimolar reaction);
* :func:`reactive_distillation` produces ester, concentrates the light ester into
  the distillate, converts the acid beyond zero, conserves every atom (the
  component balance ``feed + generation = distillate + bottoms`` closes tightly),
  and reduces to a non-reactive column at zero holdup;
* more catalyst holdup gives more conversion.
"""

import jax.numpy as jnp
import pytest

from fugacio.sim import Stream, reactive_distillation, reactive_flash
from fugacio.thermo import component_arrays, gamma_phi_model
from fugacio.thermo.activity.models import nrtl
from fugacio.thermo.constants import P_REF, R
from fugacio.thermo.kinetics import MassActionReversible
from fugacio.thermo.reactions import Reaction, delta_g_rxn, reaction_arrays
from fugacio.thermo.reference import liquid_reference_fugacity

COMPS = ("acetic acid", "ethanol", "ethyl acetate", "water")
RX = Reaction.of(COMPS, {"acetic acid": 1, "ethanol": 1}, {"ethyl acetate": 1, "water": 1})
P = 101325.0


def _model() -> object:
    arr = component_arrays(list(COMPS))
    alpha = jnp.full((4, 4), 0.3) - jnp.eye(4) * 0.3
    activity = nrtl(a=jnp.zeros((4, 4)), b=jnp.zeros((4, 4)), alpha=alpha)
    return gamma_phi_model(activity, arr["tc"], arr["pc"], arr["omega"])


def _feed() -> Stream:
    return Stream(jnp.array([1.0, 1.0, 1e-4, 1e-4]), jnp.asarray(360.0), jnp.asarray(P), COMPS)


def _law(a_f: float = 2.0, a_r: float = 1.0) -> MassActionReversible:
    return MassActionReversible(
        a_f=jnp.asarray(a_f),
        ea_f=jnp.asarray(0.0),
        a_r=jnp.asarray(a_r),
        ea_r=jnp.asarray(0.0),
        nu=RX.nu,
    )


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


# --------------------------------------------------------------------------- #
# Reactive distillation (single 6-stage / 4-component shape, reused)
# --------------------------------------------------------------------------- #
def _column(holdup: float, a_f: float = 2.0) -> object:
    return reactive_distillation(
        _feed(),
        _model(),
        RX,
        _law(a_f=a_f),
        holdup,
        n_stages=6,
        feed_stage=3,
        reflux=2.0,
        distillate_rate=1.0,
        t_min=300.0,
        t_max=420.0,
    )


def test_reactive_distillation_converts_and_concentrates_ester() -> None:
    feed = _feed()
    res = _column(0.1)
    d = res.distillate.n
    b = res.bottoms.n
    # Acid is converted.
    conversion = 1.0 - float((d[0] + b[0]) / feed.n[0])
    assert conversion > 0.05
    # The light ester is enriched in the distillate relative to the feed.
    assert float(res.distillate.z[2]) > float(feed.z[2]) + 0.05
    # Atom balance: feed + net generation = distillate + bottoms (to ~machine eps).
    generation = jnp.sum(res.generation, axis=0)
    leak = (feed.n + generation) - (d + b)
    assert float(jnp.max(jnp.abs(leak))) < 1e-6
    # Equimolar reaction conserves total moles.
    assert float(jnp.sum(d + b)) == pytest.approx(float(feed.total), rel=1e-7)
    # Valid composition profiles.
    assert jnp.allclose(jnp.sum(res.x, axis=1), 1.0, atol=1e-8)
    assert bool(jnp.all(res.x >= -1e-9))


def test_reactive_distillation_zero_holdup_is_non_reactive() -> None:
    feed = _feed()
    res = _column(0.0)
    # No holdup -> no reaction anywhere.
    assert float(jnp.max(jnp.abs(res.generation))) == 0.0
    # Acid passes through unconverted; essentially no ester is made.
    assert float(res.distillate.n[0] + res.bottoms.n[0]) == pytest.approx(
        float(feed.n[0]), rel=1e-6
    )
    assert float(res.distillate.n[2] + res.bottoms.n[2]) < 1e-3


def test_reactive_distillation_more_holdup_more_conversion() -> None:
    feed = _feed()
    low = _column(0.05)
    high = _column(0.15)

    def acid_conversion(res: object) -> float:
        return 1.0 - float((res.distillate.n[0] + res.bottoms.n[0]) / feed.n[0])

    assert acid_conversion(high) > acid_conversion(low) > 0.0
