"""Reactor unit operations: material balances, energy balances, and gradients.

The reactor blocks are checked against independent references:

* the *equilibrium* reactor must reproduce
  :func:`fugacio.thermo.reaction_equilibrium.equilibrium` and satisfy an adiabatic
  enthalpy balance when run adiabatically;
* the *stoichiometric* reactor must place the outlet exactly at the requested
  extent / conversion;
* the *kinetic* reactors (CSTR, PFR, batch) are validated against the closed-form
  solutions of a first-order gas-phase isomerisation, where the algebra is exact;
* duties are checked to equal the heat of reaction, and conversions are
  differentiated through the solvers and compared with finite differences.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import (
    Stream,
    batch_reactor,
    conversion,
    cstr,
    equilibrium_reactor,
    pfr,
    stoichiometric_reactor,
)
from fugacio.thermo.constants import P_REF, R
from fugacio.thermo.ideal import enthalpy_ig
from fugacio.thermo.kinetics import PowerLaw, arrhenius
from fugacio.thermo.reaction_equilibrium import equilibrium
from fugacio.thermo.reactions import Reaction, delta_g_rxn, delta_h_rxn, reaction_arrays

ISOM = ("n-butane", "isobutane")
ISOM_RX = Reaction.of(ISOM, {"n-butane": 1}, {"isobutane": 1})

SMR = ("methane", "water", "carbon monoxide", "hydrogen", "carbon dioxide")
SMR_RX = Reaction.of(SMR, {"methane": 1, "water": 1}, {"carbon monoxide": 1, "hydrogen": 3})
WGS_RX = Reaction.of(SMR, {"carbon monoxide": 1, "water": 1}, {"carbon dioxide": 1, "hydrogen": 1})


def _isom_feed(fa: float = 10.0, fb: float = 0.0, t: float = 350.0, p: float = 3e5) -> Stream:
    return Stream(jnp.array([fa, fb]), jnp.asarray(t), jnp.asarray(p), ISOM)


def _h_total(n: jnp.ndarray, t: float, comps: tuple[str, ...]) -> float:
    hf, _gf, (a, b, c, d, e) = reaction_arrays(list(comps))
    return float(jnp.sum(n * (hf + enthalpy_ig(t, a, b, c, d, e))))


# --------------------------------------------------------------------------- #
# Equilibrium reactor
# --------------------------------------------------------------------------- #
def test_equilibrium_reactor_matches_thermo_solver() -> None:
    feed = _isom_feed(t=330.0, p=5e5)
    res = equilibrium_reactor(feed, ISOM_RX)
    ref = equilibrium(ISOM_RX, feed.n, 330.0, 5e5)
    assert jnp.allclose(res.outlet.n, ref.moles, atol=1e-8)
    assert float(res.extent[0]) == pytest.approx(float(ref.extent[0]), rel=1e-6)
    assert float(res.outlet.t) == pytest.approx(330.0)
    assert float(res.outlet.p) == pytest.approx(5e5)
    # n-butane -> isobutane is exothermic, so holding T requires heat removal.
    assert float(res.duty) < 0.0


def test_equilibrium_reactor_isothermal_duty_equals_heat_of_reaction() -> None:
    feed = _isom_feed(t=360.0)
    res = equilibrium_reactor(feed, ISOM_RX)
    hf, _gf, (a, b, c, d, e) = reaction_arrays(list(ISOM))
    dh = float(delta_h_rxn(ISOM_RX.nu, 360.0, hf, a, b, c, d, e))
    assert float(res.duty) == pytest.approx(float(res.extent[0]) * dh, rel=1e-6)


def test_equilibrium_reactor_adiabatic_balances_energy_and_equilibrium() -> None:
    feed = _isom_feed(fa=10.0, t=300.0, p=2e5)
    res = equilibrium_reactor(feed, ISOM_RX, adiabatic=True)
    t_out = float(res.outlet.t)
    assert t_out > 300.0  # exothermic temperature rise
    assert float(res.duty) == 0.0
    # Adiabatic enthalpy balance: outlet enthalpy equals the feed enthalpy.
    assert _h_total(res.outlet.n, t_out, ISOM) == pytest.approx(
        _h_total(feed.n, 300.0, ISOM), rel=1e-7
    )
    # Equilibrium holds at the solved outlet temperature.
    hf, gf, (a, b, c, d, e) = reaction_arrays(list(ISOM))
    ln_k = -float(delta_g_rxn(ISOM_RX.nu, t_out, hf, gf, a, b, c, d, e)) / (R * t_out)
    y = res.outlet.n / jnp.sum(res.outlet.n)
    ln_q = float(ISOM_RX.nu @ (jnp.log(y) + jnp.log(jnp.asarray(2e5) / P_REF)))
    assert ln_q == pytest.approx(ln_k, abs=1e-6)


def test_equilibrium_reactor_multireaction_matches_solver() -> None:
    feed = Stream(
        jnp.array([1.0, 3.0, 1e-6, 1e-6, 1e-6]), jnp.asarray(1100.0), jnp.asarray(1e5), SMR
    )
    res = equilibrium_reactor(feed, [SMR_RX, WGS_RX], max_iter=100)
    ref = equilibrium([SMR_RX, WGS_RX], feed.n, 1100.0, 1e5, max_iter=100)
    assert jnp.allclose(res.outlet.n, ref.moles, atol=1e-6)


# --------------------------------------------------------------------------- #
# Stoichiometric reactor
# --------------------------------------------------------------------------- #
def test_stoichiometric_conversion_sets_outlet() -> None:
    feed = _isom_feed(fa=10.0, t=350.0)
    res = stoichiometric_reactor(feed, ISOM_RX, conversion=0.4)
    assert jnp.allclose(res.outlet.n, jnp.array([6.0, 4.0]), atol=1e-9)
    assert float(res.extent[0]) == pytest.approx(4.0)
    assert float(res.duty) < 0.0


def test_stoichiometric_extent_and_adiabatic_temperature_rise() -> None:
    feed = _isom_feed(fa=10.0, t=300.0)
    res = stoichiometric_reactor(feed, ISOM_RX, extent=[3.0], adiabatic=True)
    assert jnp.allclose(res.outlet.n, jnp.array([7.0, 3.0]), atol=1e-9)
    assert float(res.outlet.t) > 300.0
    assert _h_total(res.outlet.n, float(res.outlet.t), ISOM) == pytest.approx(
        _h_total(feed.n, 300.0, ISOM), rel=1e-7
    )


def test_stoichiometric_requires_exactly_one_spec() -> None:
    feed = _isom_feed()
    with pytest.raises(ValueError, match="exactly one"):
        stoichiometric_reactor(feed, ISOM_RX)
    with pytest.raises(ValueError, match="exactly one"):
        stoichiometric_reactor(feed, ISOM_RX, extent=[1.0], conversion=0.5)


def test_stoichiometric_conversion_rejects_multireaction() -> None:
    feed = Stream(jnp.ones(5), jnp.asarray(1000.0), jnp.asarray(1e5), SMR)
    with pytest.raises(ValueError, match="single reaction"):
        stoichiometric_reactor(feed, [SMR_RX, WGS_RX], conversion=0.5)


# --------------------------------------------------------------------------- #
# Kinetic reactors vs. analytic first-order isomerisation
# --------------------------------------------------------------------------- #
_T = 350.0
_P = 3e5
_FA0 = 5.0
_LAW = PowerLaw(a=jnp.asarray(2.0e3), ea=jnp.asarray(30e3), orders=jnp.array([1.0, 0.0]))


def _alpha() -> float:
    k = float(arrhenius(_T, 2.0e3, 30e3))
    return k * (_P / (R * _T)) / _FA0  # 1/m^3, isomerisation keeps total flow at F_A0


def test_cstr_first_order_matches_analytic() -> None:
    volume = 0.5
    feed = _isom_feed(fa=_FA0, fb=0.0, t=_T, p=_P)
    res = cstr(feed, ISOM_RX, _LAW, volume)
    av = _alpha() * volume
    x_analytic = av / (1.0 + av)
    x = float(conversion(feed, res.outlet, 0))
    assert 0.2 < x < 0.8
    assert x == pytest.approx(x_analytic, rel=1e-6)
    # Steady-state mole balance residual is satisfied at the returned outlet.
    c = res.outlet.n / jnp.sum(res.outlet.n) * (_P / (R * _T))
    r = _LAW.rate(_T, c)
    assert jnp.allclose(res.outlet.n - feed.n - volume * (r * ISOM_RX.nu), 0.0, atol=1e-9)


def test_pfr_first_order_matches_analytic() -> None:
    volume = 0.5
    feed = _isom_feed(fa=_FA0, fb=0.0, t=_T, p=_P)
    res = pfr(feed, ISOM_RX, _LAW, volume, steps=400)
    x_analytic = 1.0 - jnp.exp(-_alpha() * volume)
    x = float(conversion(feed, res.outlet, 0))
    assert x == pytest.approx(float(x_analytic), rel=1e-5)


def test_pfr_outperforms_cstr_for_positive_order() -> None:
    volume = 0.5
    feed = _isom_feed(fa=_FA0, t=_T, p=_P)
    x_pfr = float(conversion(feed, pfr(feed, ISOM_RX, _LAW, volume, steps=400).outlet, 0))
    x_cstr = float(conversion(feed, cstr(feed, ISOM_RX, _LAW, volume).outlet, 0))
    assert x_pfr > x_cstr


def test_batch_first_order_matches_analytic() -> None:
    # Constant-volume batch isomerisation: N_A = N_A0 exp(-k t), independent of V.
    n0, vol, time = 4.0, 0.01, 30.0
    feed = Stream(jnp.array([n0, 0.0]), jnp.asarray(_T), jnp.asarray(_P), ISOM)
    res = batch_reactor(feed, ISOM_RX, _LAW, vol, time, steps=400)
    k = float(arrhenius(_T, 2.0e3, 30e3))
    x_analytic = 1.0 - jnp.exp(-k * time)
    x = float(conversion(feed, res.outlet, 0))
    assert 0.2 < x < 0.95
    assert x == pytest.approx(float(x_analytic), rel=1e-5)


def test_pfr_adiabatic_temperature_rises() -> None:
    feed = _isom_feed(fa=_FA0, t=_T, p=_P)
    res = pfr(feed, ISOM_RX, _LAW, 0.5, adiabatic=True, steps=400)
    assert float(res.outlet.t) > _T  # exothermic
    assert float(conversion(feed, res.outlet, 0)) > 0.0
    assert float(res.duty) == 0.0


# --------------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------------- #
def test_cstr_conversion_gradient_wrt_volume_matches_fd() -> None:
    feed = _isom_feed(fa=_FA0, t=_T, p=_P)

    def x_of_v(v: jax.Array) -> jax.Array:
        res = cstr(feed, ISOM_RX, _LAW, v)
        return conversion(feed, res.outlet, 0)

    v0 = jnp.asarray(0.5)
    g = float(jax.grad(x_of_v)(v0))
    dv = 1e-4
    fd = (float(x_of_v(v0 + dv)) - float(x_of_v(v0 - dv))) / (2 * dv)
    assert g == pytest.approx(fd, rel=1e-4)
    assert g > 0.0  # more volume -> more conversion


def test_pfr_conversion_differentiable_wrt_rate_constant() -> None:
    feed = _isom_feed(fa=_FA0, t=_T, p=_P)

    def x_of_a(a: jax.Array) -> jax.Array:
        law = PowerLaw(a=a, ea=jnp.asarray(30e3), orders=jnp.array([1.0, 0.0]))
        return conversion(feed, pfr(feed, ISOM_RX, law, 0.5, steps=200).outlet, 0)

    g = float(jax.grad(x_of_a)(jnp.asarray(2.0e3)))
    assert g > 0.0  # a faster reaction converts more
