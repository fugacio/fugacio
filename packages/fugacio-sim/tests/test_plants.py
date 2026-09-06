"""Whole-plant case studies: recycle loops, rigorous columns, exchangers, reactors together.

Three small plants exercise the pieces end to end, the way a flowsheet author
would combine them:

1. **HDA-lite**: toluene hydrodealkylation with a hydrogen recycle loop (mixer,
   furnace, fixed-conversion reactor, cooler, flash, purge split), the loop
   partitioned and torn automatically, followed by a stabiliser and a rigorous
   benzene/toluene column. The overall material balance must close, the benzene
   made must equal the toluene converted, and the column must deliver its spec.
2. **Depropanizer**: a C3/nC4/nC5 column on Peng-Robinson with recovery and
   purity specifications, its feed preheated against the hot bottoms in a
   two-sided exchanger. Checks the specs, the balances, and that the preheat
   duty is consistent on both sides.
3. **Ethanol/water train (NRTL)**: an economiser exchanger between the feed and
   the column bottoms closes a heat-integration loop *around* a rigorous column
   on a gamma-phi package; the loop is converged by Broyden and the distillate
   must stay below the azeotrope while the bottoms is nearly pure water.

These are the slowest tests in the suite (each plant compiles several nested
implicit solvers); they are marked ``plant`` so they can be selected or
deselected explicitly.
"""

import gc

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import (
    ColumnFeed,
    Flowsheet,
    Stream,
    component_separator,
    distillate_rate,
    enthalpy_flow,
    flash_drum,
    heat_exchanger,
    heater,
    mix,
    package_for,
    purity,
    recovery,
    reflux_ratio,
    rigorous_column,
    splitter,
    valve,
)
from fugacio.sim.reactors import stoichiometric_reactor
from fugacio.thermo.reactions import Reaction

pytestmark = pytest.mark.plant


# --------------------------------------------------------------------------- #
# 1. HDA-lite
# --------------------------------------------------------------------------- #
HDA = ("hydrogen", "methane", "benzene", "toluene")


def _hda_flowsheet() -> Flowsheet:
    toluene = Stream.from_fractions(HDA, jnp.array([0.0, 0.0, 0.0, 1.0]), 100.0, 300.0, 30e5)
    makeup = Stream.from_fractions(HDA, jnp.array([0.95, 0.05, 0.0, 0.0]), 250.0, 300.0, 30e5)
    pkg = package_for(HDA)
    rxn = Reaction(components=HDA, nu=jnp.array([-1.0, 1.0, 1.0, -1.0]))

    fs = Flowsheet()
    fs.feed("toluene", toluene)
    fs.feed("makeup", makeup)
    fs.unit(
        "mixer",
        lambda tol, h2, rec, th: mix([tol, h2, rec], t=310.0),
        inputs=("toluene", "makeup", "recycle"),
        outputs=("mixed",),
    )
    fs.unit(
        "furnace",
        lambda s, th: heater(s, t_out=th["T_rx"]).outlet,
        inputs=("mixed",),
        outputs=("hot",),
    )
    fs.unit(
        "reactor",
        lambda s, th: (
            stoichiometric_reactor(s, rxn, conversion=th["X"], t_out=th["T_rx"], model=pkg).outlet
        ),
        inputs=("hot",),
        outputs=("effluent",),
    )
    fs.unit(
        "cooler",
        lambda s, th: heater(s, t_out=310.0).outlet,
        inputs=("effluent",),
        outputs=("cooled",),
    )
    fs.unit(
        "flash",
        lambda s, th: flash_drum(s, 310.0, 30e5),
        inputs=("cooled",),
        outputs=("gas", "crude"),
    )
    fs.unit(
        "purge_split",
        lambda s, th: splitter(s, jnp.array([th["purge"], 1.0 - th["purge"]])),
        inputs=("gas",),
        outputs=("purge", "recycle"),
    )
    fs.unit(
        "stabilizer",
        lambda s, th: component_separator(s, jnp.array([1.0, 1.0, 0.0, 0.0])),
        inputs=("crude",),
        outputs=("offgas", "aromatics"),
    )
    fs.unit(
        "column",
        lambda s, th: _bt_column(s, th["R"]),
        inputs=("aromatics",),
        outputs=("benzene", "toluene_rec"),
    )
    return fs


def _bt_column(aromatics: Stream, r: float) -> tuple[Stream, Stream]:
    liquid = valve(aromatics, 1.5e5)
    res = rigorous_column(
        [ColumnFeed(liquid, 6)],
        12,
        p=1.5e5,
        specs=[reflux_ratio(r), recovery(2, "distillate", 0.99)],
    )
    return res.distillate, res.bottoms


def test_hda_lite_plant_closes_and_meets_specs() -> None:
    fs = _hda_flowsheet()
    parts = fs.partition()
    # One recycle loop (six units, one tear) followed by the acyclic separation train.
    assert [p.cyclic for p in parts] == [True, False, False]
    assert len(parts[0].tears) == 1 and len(parts[0].units) == 6

    theta = {"T_rx": 900.0, "X": 0.75, "purge": 0.10, "R": 2.5}
    solved = fs.solve_with_info(theta, method="wegstein")
    solved.check()
    s = solved.streams

    fresh = s["toluene"].n + s["makeup"].n
    out = s["purge"].n + s["offgas"].n + s["benzene"].n + s["toluene_rec"].n
    # Overall material balance: the reaction is a 1:1 swap, so total moles are conserved.
    assert float(jnp.sum(fresh)) == pytest.approx(float(jnp.sum(out)), rel=1e-8)
    # Atom balances: carbon and hydrogen.
    carbon = jnp.array([0.0, 1.0, 6.0, 7.0])
    hydrogen = jnp.array([2.0, 4.0, 6.0, 8.0])
    assert float(carbon @ fresh) == pytest.approx(float(carbon @ out), rel=1e-8)
    assert float(hydrogen @ fresh) == pytest.approx(float(hydrogen @ out), rel=1e-8)

    # Benzene made equals the toluene converted at the reactor inlet (the recycle gas
    # carries a little toluene back, so the inlet exceeds the fresh 100 mol/s).
    benzene_made = float(
        s["benzene"].n[2] + s["toluene_rec"].n[2] + s["purge"].n[2] + s["offgas"].n[2]
    )
    assert benzene_made == pytest.approx(0.75 * float(s["hot"].n[3]), rel=1e-6)
    assert float(s["hot"].n[3]) > 100.0
    # The recycle carries hydrogen and methane back; the loop is genuinely closed.
    assert float(s["recycle"].total) > 500.0
    assert float(jnp.abs(s["recycle"].n - s["purge"].n * 9.0).max()) < 1e-6
    # Column delivers 99 % benzene recovery.
    assert float(s["benzene"].n[2] / s["aromatics"].n[2]) == pytest.approx(0.99, abs=1e-6)
    assert float(s["benzene"].z[2]) > 0.95

    # Close total energy on one reference, including formation enthalpies at
    # the reacting boundary and the ideal separator's required heat exchange.
    from fugacio.thermo.reactions import reaction_arrays

    pkg = package_for(HDA)
    hf, _, _ = reaction_arrays(list(HDA))
    rxn = Reaction(components=HDA, nu=jnp.array([-1.0, 1.0, 1.0, -1.0]))
    reactor = stoichiometric_reactor(
        s["hot"], rxn, conversion=theta["X"], t_out=theta["T_rx"], model=pkg
    )
    col = rigorous_column(
        [ColumnFeed(valve(s["aromatics"], 1.5e5), 6)],
        12,
        p=1.5e5,
        specs=[reflux_ratio(theta["R"]), recovery(2, "distillate", 0.99)],
    )

    def h(name):
        return enthalpy_flow(s[name], model=pkg)

    mixer_duty = h("mixed") - h("toluene") - h("makeup") - h("recycle")
    separator_duty = h("offgas") + h("aromatics") - h("crude")
    duties = mixer_duty + heater(s["mixed"], t_out=theta["T_rx"], model=pkg).duty + reactor.duty
    duties += (
        heater(s["effluent"], t_out=310.0, model=pkg).duty
        + separator_duty
        + col.condenser_duty
        + col.reboiler_duty
    )
    inlet_energy = h("toluene") + h("makeup") + fresh @ hf
    outlet_energy = h("purge") + h("offgas") + h("benzene") + h("toluene_rec") + out @ hf
    assert float(inlet_energy + duties - outlet_energy) == pytest.approx(0.0, abs=1.0)


# --------------------------------------------------------------------------- #
# 2. Depropanizer with a feed/bottoms economiser loop
# --------------------------------------------------------------------------- #
C3 = ("propane", "n-butane", "n-pentane")
_C3_SPECS = [recovery(0, "distillate", 0.98), purity("distillate", 0, 0.95)]


def _c3_column(feed: Stream) -> tuple[Stream, Stream]:
    res = rigorous_column([ColumnFeed(feed, 8)], 16, p=16e5, specs=_C3_SPECS)
    return res.distillate, res.bottoms


def _c3_flowsheet() -> tuple[Flowsheet, Stream]:
    cold_feed = Stream.from_fractions(C3, jnp.array([0.40, 0.35, 0.25]), 100.0, 300.0, 16e5)

    fs = Flowsheet()
    fs.feed("feed", cold_feed)
    fs.unit(
        "economiser",
        lambda cold, hot, th: _swap(heat_exchanger(hot, cold, min_approach=th["dt"])),
        inputs=("feed", "bottoms"),
        outputs=("preheated", "bottoms_cooled"),
    )
    fs.unit(
        "column",
        lambda s, th: _c3_column(s),
        inputs=("preheated",),
        outputs=("distillate", "bottoms"),
    )
    return fs, cold_feed


def test_depropanizer_with_economiser_loop() -> None:
    fs, cold_feed = _c3_flowsheet()
    (block,) = fs.partition()
    assert block.cyclic and block.tears in (("bottoms",), ("preheated",))

    # The tear is seeded automatically by one pass with an empty recycle (an
    # empty hot side exchanges nothing), then converged by Broyden.
    s = fs.solve({"dt": 15.0}, method="broyden", tol=1e-8)
    col = rigorous_column([ColumnFeed(s["preheated"], 8)], 16, p=16e5, specs=_C3_SPECS)
    assert float(col.residual_norm) < 1e-8

    # Specs.
    assert float(s["distillate"].n[0] / cold_feed.n[0]) == pytest.approx(0.98, abs=1e-8)
    assert float(s["distillate"].z[0]) == pytest.approx(0.95, abs=1e-8)
    assert float(s["bottoms"].z[0]) < 0.02
    # Balances around the whole train.
    assert jnp.allclose(s["distillate"].n + s["bottoms_cooled"].n, cold_feed.n, atol=1e-6)
    # Saturated product phases are preserved, so the energy balance can be
    # checked directly through the public stream property interface.
    h_in = enthalpy_flow(s["preheated"])
    h_out = enthalpy_flow(col.distillate) + enthalpy_flow(col.bottoms)
    assert float(h_in + col.condenser_duty + col.reboiler_duty - h_out) == pytest.approx(
        0.0, abs=1e-7 * float(col.reboiler_duty)
    )
    # Exchanger: the recovered heat is consistent on both sides and pinches at 15 K.
    hx = heat_exchanger(s["bottoms"], cold_feed, min_approach=15.0)
    q_cold = enthalpy_flow(s["preheated"]) - enthalpy_flow(cold_feed)
    q_hot = enthalpy_flow(s["bottoms"]) - enthalpy_flow(s["bottoms_cooled"])
    assert float(q_cold) == pytest.approx(float(hx.duty), rel=1e-6)
    assert float(q_hot) == pytest.approx(float(hx.duty), rel=1e-6)
    assert float(hx.min_approach) == pytest.approx(15.0, abs=1e-6)
    assert float(s["preheated"].t) > 330.0
    # The torn stream is self-consistent: one more pass reproduces it.
    assert float(hx.cold_out.t) == pytest.approx(float(s["preheated"].t), abs=1e-5)
    assert jnp.allclose(_c3_column(hx.cold_out)[1].n, s["bottoms"].n, atol=1e-6)
    # Preheating the feed lowers the reboiler duty by about the recovered heat
    # (reflux ratio and distillate rate are what the specs pin down, so the
    # condenser duty barely moves).
    col0 = rigorous_column([ColumnFeed(cold_feed, 8)], 16, p=16e5, specs=_C3_SPECS)
    saving = float(col0.reboiler_duty - col.reboiler_duty)
    assert 0.5 * float(hx.duty) < saving < 1.5 * float(hx.duty)


def test_depropanizer_economiser_sensitivity() -> None:
    # CI runs this in a fresh process so the plant and column verification
    # executables aren't retained while compiling the full plant derivative.
    fs, cold_feed = _c3_flowsheet()
    s = fs.solve({"dt": 15.0}, method="broyden", tol=1e-8)
    jax.block_until_ready(s)
    # The seed solve, adjoint, and finite-difference evaluations compile
    # different graphs. Retaining all three sets exhausts CI runner memory.
    # Only release in-memory caches; the persistent compilation cache remains.
    jax.clear_caches()
    gc.collect()

    # A sensitivity of the converged, heat-integrated train agrees with an
    # independent operating-condition perturbation through both units.
    def recovered_heat(approach):
        result = fs.solve({"dt": approach}, method="broyden", tol=1e-9, guess=s)
        return enthalpy_flow(result["preheated"]) - enthalpy_flow(cold_feed)

    derivative = float(jax.grad(recovered_heat)(jnp.asarray(15.0)))
    jax.clear_caches()
    gc.collect()
    finite_difference = (recovered_heat(15.01) - recovered_heat(14.99)) / 0.02
    assert jnp.isfinite(derivative)
    assert float(derivative) == pytest.approx(float(finite_difference), rel=5e-3, abs=1e-2)


def _swap(res) -> tuple[Stream, Stream]:
    return res.cold_out, res.hot_out


# --------------------------------------------------------------------------- #
# 3. Ethanol/water train with heat recovery (NRTL)
# --------------------------------------------------------------------------- #
def test_ethanol_water_train_recovers_bottoms_heat() -> None:
    comps = ("ethanol", "water")
    pkg = package_for(comps, "nrtl")
    feed = Stream.from_fractions(comps, jnp.array([0.10, 0.90]), 100.0, 300.0, 1.5e5)

    def column(s: Stream):
        liquid = valve(s, 1.013e5, model=pkg)
        return rigorous_column(
            [ColumnFeed(liquid, 6)],
            12,
            p=1.013e5,
            specs=[reflux_ratio(3.0), distillate_rate(11.5)],
            model=pkg,
        )

    # Base case on the cold feed.
    col0 = column(feed)
    assert float(col0.residual_norm) < 1e-8
    assert jnp.allclose(col0.distillate.n + col0.bottoms.n, feed.n, atol=1e-6)
    # Distillate below the azeotrope (~0.89), bottoms nearly pure water, the
    # NRTL temperature profile monotone between the two boiling points.
    assert 0.75 < float(col0.distillate.z[0]) < 0.9
    assert float(col0.bottoms.z[0]) < 0.01
    assert 351.0 < float(col0.t[0]) < 355.0
    assert 371.0 < float(col0.t[-1]) < 375.0

    # Heat recovery: cool the bottoms to 10 K above the feed and give that duty
    # to the feed (NRTL enthalpies with heat of mixing on both sides).
    cooled = heater(col0.bottoms, t_out=310.0, model=pkg)
    recovered = -float(cooled.duty)
    assert recovered > 0.0
    preheated = heater(feed, duty=recovered, model=pkg).outlet
    assert float(preheated.t) > 340.0
    assert float(
        enthalpy_flow(preheated, model=pkg) - enthalpy_flow(feed, model=pkg)
    ) == pytest.approx(recovered, rel=1e-8)

    # The reboiler saving equals the recovered heat (the products and condenser
    # duty are pinned by the specifications, so the overall energy balance moves
    # it all to the reboiler) to within the small shift in product enthalpies.
    col = column(preheated)
    assert float(col.residual_norm) < 1e-8
    saving = float(col0.reboiler_duty - col.reboiler_duty)
    assert saving == pytest.approx(recovered, rel=0.05)
    assert float(col.distillate.z[0]) == pytest.approx(float(col0.distillate.z[0]), abs=0.02)

    # Close the economizer loop against the current column bottoms.
    fs = Flowsheet().feed("feed", feed)
    fs.unit(
        "economizer",
        lambda cold, hot, th: _swap(heat_exchanger(hot, cold, min_approach=th["dt"], model=pkg)),
        inputs=("feed", "bottoms"),
        outputs=("preheated", "cooled"),
    )

    def column_products(stream, th):
        result = column(stream)
        return result.distillate, result.bottoms

    fs.unit("column", column_products, inputs=("preheated",), outputs=("distillate", "bottoms"))
    solved = fs.solve_with_info({"dt": 10.0}, method="broyden", guess={"bottoms": col.bottoms})
    solved.check()
    final = column(solved["preheated"])
    assert jnp.allclose(solved["distillate"].n + solved["cooled"].n, feed.n, atol=1e-6)
    balance = enthalpy_flow(feed, model=pkg) + final.condenser_duty + final.reboiler_duty
    balance -= enthalpy_flow(solved["distillate"], model=pkg) + enthalpy_flow(
        solved["cooled"], model=pkg
    )
    assert float(balance) == pytest.approx(0.0, abs=1.0)
