"""Reactor unit operations: material balances, energy balances, and gradients.

The reactors run on a property package (Peng-Robinson by default), so
equilibrium uses real-fluid fugacities and energy uses real-fluid enthalpies
plus ideal-gas formation enthalpies. They're checked against independent
references:

* the *equilibrium* reactor must satisfy each reaction's equilibrium constant
  with the package's fugacity coefficients, close the energy balance, and
  approach the ideal-gas solver where the gas is nearly ideal;
* the *stoichiometric* reactor must place the outlet exactly at the requested
  extent or conversion;
* the *kinetic* reactors (CSTR, PFR, batch) must satisfy their own balances
  exactly and match the closed-form first-order isomerization at low pressure,
  where the concentration is the ideal-gas one;
* conversions are differentiated through the solvers and compared with finite
  differences.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import (
    Stream,
    batch_reactor,
    conversion,
    cstr,
    enthalpy_flow,
    equilibrium_reactor,
    package_for,
    pfr,
    stoichiometric_reactor,
)
from fugacio.thermo.constants import P_REF, R
from fugacio.thermo.kinetics import PowerLaw, arrhenius
from fugacio.thermo.reaction_equilibrium import equilibrium
from fugacio.thermo.reactions import Reaction, delta_g_rxn, reaction_arrays

ISOM = ("n-butane", "isobutane")
ISOM_RX = Reaction.of(ISOM, {"n-butane": 1}, {"isobutane": 1})

SMR = ("methane", "water", "carbon monoxide", "hydrogen", "carbon dioxide")
SMR_RX = Reaction.of(SMR, {"methane": 1, "water": 1}, {"carbon monoxide": 1, "hydrogen": 3})
WGS_RX = Reaction.of(SMR, {"carbon monoxide": 1, "water": 1}, {"carbon dioxide": 1, "hydrogen": 1})


def _isom_feed(fa: float = 10.0, fb: float = 0.0, t: float = 350.0, p: float = 3e5) -> Stream:
    return Stream(jnp.array([fa, fb]), jnp.asarray(t), jnp.asarray(p), ISOM)


def _energy(stream: Stream) -> float:
    """Real-fluid enthalpy flow plus formation enthalpy (W)."""
    hf, _gf, _cp = reaction_arrays(list(stream.components))
    return float(enthalpy_flow(stream) + stream.n @ hf)


def _equilibrium_gap(stream: Stream, reactions: list[Reaction]) -> float:
    """Largest ``|ln K_j - sum_i nu_ji ln(y_i phi_i P / P_ref)|`` at the stream state."""
    hf, gf, coeffs = reaction_arrays(list(stream.components))
    y = stream.n / jnp.sum(stream.n)
    ln_phi = package_for(stream.components).ln_phi(stream.t, stream.p, y, phase="vapor")
    ln_a = jnp.log(y) + ln_phi + jnp.log(stream.p / P_REF)
    gaps = [
        -delta_g_rxn(rx.nu, stream.t, hf, gf, *coeffs) / (R * stream.t) - rx.nu @ ln_a
        for rx in reactions
    ]
    return float(jnp.max(jnp.abs(jnp.array(gaps))))


# --------------------------------------------------------------------------- #
# Equilibrium reactor
# --------------------------------------------------------------------------- #
def test_equilibrium_reactor_satisfies_real_fluid_equilibrium() -> None:
    feed = _isom_feed(t=330.0, p=5e5)
    res = equilibrium_reactor(feed, ISOM_RX)
    assert bool(res.converged)
    assert _equilibrium_gap(res.outlet, [ISOM_RX]) < 1e-8
    assert float(res.outlet.t) == pytest.approx(330.0)
    assert float(res.outlet.p) == pytest.approx(5e5)
    assert float(jnp.sum(res.outlet.n)) == pytest.approx(10.0, rel=1e-12)
    # n-butane -> isobutane is exothermic, so holding T requires heat removal.
    assert float(res.duty) < 0.0


def test_equilibrium_reactor_approaches_the_ideal_gas_solver_at_low_pressure() -> None:
    feed = _isom_feed(t=330.0, p=1e3)
    res = equilibrium_reactor(feed, ISOM_RX)
    ideal = equilibrium(ISOM_RX, feed.n, 330.0, 1e3)
    assert jnp.allclose(res.outlet.n, ideal.moles, rtol=1e-4)


def test_equilibrium_reactor_isothermal_duty_closes_the_energy_balance() -> None:
    feed = _isom_feed(t=360.0)
    res = equilibrium_reactor(feed, ISOM_RX)
    assert float(res.duty) == pytest.approx(_energy(res.outlet) - _energy(feed), rel=1e-8)


def test_adiabatic_equilibrium_reactor_balances_energy_and_equilibrium() -> None:
    feed = _isom_feed(fa=10.0, t=300.0, p=2e5)
    res = equilibrium_reactor(feed, ISOM_RX, duty=0.0)
    assert float(res.outlet.t) > 300.0  # exothermic temperature rise
    assert float(res.duty) == 0.0
    assert _energy(res.outlet) == pytest.approx(_energy(feed), rel=1e-8)
    assert _equilibrium_gap(res.outlet, [ISOM_RX]) < 1e-8


def test_equilibrium_reactor_multireaction_satisfies_both_equilibria() -> None:
    feed = Stream(
        jnp.array([1.0, 3.0, 1e-6, 1e-6, 1e-6]), jnp.asarray(1100.0), jnp.asarray(1e5), SMR
    )
    res = equilibrium_reactor(feed, [SMR_RX, WGS_RX], max_iter=100)
    assert _equilibrium_gap(res.outlet, [SMR_RX, WGS_RX]) < 1e-7
    # At 1 bar and 1100 K the gas is nearly ideal.
    ideal = equilibrium([SMR_RX, WGS_RX], feed.n, 1100.0, 1e5, max_iter=100)
    assert jnp.allclose(res.outlet.n, ideal.moles, rtol=1e-3, atol=1e-5)


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
    res = stoichiometric_reactor(feed, ISOM_RX, extent=[3.0], duty=0.0)
    assert jnp.allclose(res.outlet.n, jnp.array([7.0, 3.0]), atol=1e-9)
    assert float(res.outlet.t) > 300.0
    assert _energy(res.outlet) == pytest.approx(_energy(feed), rel=1e-8)


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
# Kinetic reactors and the first-order isomerization
# --------------------------------------------------------------------------- #
_T = 350.0
_P_LOW = 1e3  # near-ideal gas, so c = y P / (R T) to about 1e-4
_FA0 = 0.01
_LAW = PowerLaw(a=jnp.asarray(2.0e3), ea=jnp.asarray(30e3), orders=jnp.array([1.0, 0.0]))


def _alpha(p: float = _P_LOW) -> float:
    """First-order ``k c_total / F_total`` (1/m^3); the isomerization keeps F constant."""
    k = float(arrhenius(_T, 2.0e3, 30e3))
    return k * (p / (R * _T)) / _FA0


def _volume(target: float = 1.0) -> float:
    """Reactor volume with ``alpha V = target``."""
    return target / _alpha()


def test_cstr_satisfies_its_mole_balance_exactly() -> None:
    feed = _isom_feed(fa=_FA0, fb=0.0, t=_T, p=3e5)
    volume = 1e-3
    res = cstr(feed, ISOM_RX, volume, _LAW)
    assert bool(res.converged)
    y = res.outlet.n / jnp.sum(res.outlet.n)
    v = package_for(ISOM).volume(_T, 3e5, y, phase="vapor")
    rate = _LAW.rate(_T, y / v)
    balance = res.outlet.n - feed.n - volume * (rate * ISOM_RX.nu)
    assert float(jnp.max(jnp.abs(balance))) < 1e-12


def test_cstr_first_order_matches_analytic() -> None:
    feed = _isom_feed(fa=_FA0, fb=0.0, t=_T, p=_P_LOW)
    res = cstr(feed, ISOM_RX, _volume(), _LAW)
    x = float(conversion(feed, res.outlet, 0))
    assert x == pytest.approx(0.5, rel=1e-3)  # alpha V / (1 + alpha V)


def test_pfr_first_order_matches_analytic() -> None:
    feed = _isom_feed(fa=_FA0, fb=0.0, t=_T, p=_P_LOW)
    res = pfr(feed, ISOM_RX, _volume(), _LAW, steps=128)
    x = float(conversion(feed, res.outlet, 0))
    assert x == pytest.approx(1.0 - float(jnp.exp(-1.0)), rel=1e-3)


def test_pfr_outperforms_cstr_for_positive_order() -> None:
    feed = _isom_feed(fa=_FA0, t=_T, p=_P_LOW)
    x_pfr = float(conversion(feed, pfr(feed, ISOM_RX, _volume(), _LAW).outlet, 0))
    x_cstr = float(conversion(feed, cstr(feed, ISOM_RX, _volume(), _LAW).outlet, 0))
    assert x_pfr > x_cstr


def test_batch_first_order_matches_analytic() -> None:
    # Constant-volume batch isomerization: N_A = N_A0 exp(-k t), independent of V.
    n0, vol, time = 4.0, 0.01, 30.0
    feed = Stream(jnp.array([n0, 0.0]), jnp.asarray(_T), jnp.asarray(3e5), ISOM)
    res = batch_reactor(feed, ISOM_RX, _LAW, vol, time, steps=400)
    k = float(arrhenius(_T, 2.0e3, 30e3))
    x = float(conversion(feed, res.contents, 0))
    assert 0.2 < x < 0.95
    assert x == pytest.approx(float(1.0 - jnp.exp(-k * time)), rel=1e-5)


def test_adiabatic_pfr_temperature_rises() -> None:
    feed = _isom_feed(fa=_FA0, t=_T, p=_P_LOW)
    res = pfr(feed, ISOM_RX, _volume(), _LAW, duty=0.0)
    assert float(res.outlet.t) > _T  # exothermic
    assert float(conversion(feed, res.outlet, 0)) > 0.0
    assert float(res.duty) == 0.0
    assert _energy(res.outlet) == pytest.approx(_energy(feed), rel=1e-6)


# --------------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------------- #
def test_cstr_conversion_gradient_wrt_volume_matches_fd() -> None:
    feed = _isom_feed(fa=_FA0, t=_T, p=_P_LOW)

    def x_of_v(v: jax.Array) -> jax.Array:
        return conversion(feed, cstr(feed, ISOM_RX, v, _LAW).outlet, 0)

    v0 = jnp.asarray(_volume())
    g = float(jax.grad(x_of_v)(v0))
    dv = 1e-4 * float(v0)
    fd = (float(x_of_v(v0 + dv)) - float(x_of_v(v0 - dv))) / (2 * dv)
    assert g == pytest.approx(fd, rel=1e-4)
    assert g > 0.0  # more volume, more conversion


def test_pfr_conversion_differentiable_wrt_rate_constant() -> None:
    feed = _isom_feed(fa=_FA0, t=_T, p=_P_LOW)

    def x_of_a(a: jax.Array) -> jax.Array:
        law = PowerLaw(a=a, ea=jnp.asarray(30e3), orders=jnp.array([1.0, 0.0]))
        return conversion(feed, pfr(feed, ISOM_RX, _volume(), law).outlet, 0)

    g = float(jax.grad(x_of_a)(jnp.asarray(2.0e3)))
    assert g > 0.0  # a faster reaction converts more
