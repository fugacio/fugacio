"""Compile once: new operating points, re-solves, and new case runners reuse kernels.

A listener on JAX's backend-compile event counts compilations. After one
warm-up call, evaluating a unit, a package calculation, or a flowsheet at a new
operating point must not compile anything, and neither must a second case
runner built from an identical case.
"""

import jax
import jax.numpy as jnp
import pytest

from fugacio.sim import (
    Flowsheet,
    Stream,
    adiabatic_flash,
    compressor,
    flash_drum,
    heater,
    mix,
    package_for,
    pump,
    splitter,
    turbine,
    valve,
)
from fugacio.sim.cases.examples import example_case
from fugacio.sim.cases.runtime import CaseRunner
from fugacio.sim.cases.schema import ProcessCase

_COMPILES = [0]


def _count(event: str, duration: float, **kwargs: object) -> None:
    if event.endswith("backend_compile_duration"):
        _COMPILES[0] += 1


jax.monitoring.register_event_duration_secs_listener(_count)


def _compiles_during(call, *args) -> int:
    before = _COMPILES[0]
    jax.block_until_ready(jax.tree_util.tree_leaves(call(*args)))
    return _COMPILES[0] - before


LIGHT = ("methane", "propane", "n-pentane")
Z = jnp.array([0.5, 0.3, 0.2])
PKG = package_for(LIGHT)
FEED = Stream.from_fractions(LIGHT, Z, 10.0, 320.0, 20e5)
GAS = Stream.from_fractions(("methane",), jnp.ones(1), 1.0, 300.0, 5e5)
LIQUID = Stream.from_fractions(("n-pentane",), jnp.ones(1), 1.0, 300.0, 2e5)

CALLS = {
    "flash_drum": (lambda t: flash_drum(FEED, t, 20e5), 320.0, 325.0),
    "adiabatic_flash": (lambda p: adiabatic_flash(FEED, p).outlets, 10e5, 12e5),
    "heater_t_out": (lambda t: heater(FEED, t_out=t).outlet, 350.0, 360.0),
    "heater_duty": (lambda q: heater(FEED, duty=q).outlet, 1e4, 2e4),
    "valve": (lambda p: valve(FEED, p), 10e5, 12e5),
    "pump": (lambda p: pump(LIQUID, p).outlet, 10e5, 12e5),
    "compressor": (lambda p: compressor(GAS, p).outlet, 10e5, 12e5),
    "turbine": (lambda p: turbine(GAS, p).outlet, 2e5, 3e5),
    "mix": (
        lambda t: mix([FEED, Stream.from_fractions(LIGHT, Z, 5.0, t, 20e5)]),
        330.0,
        340.0,
    ),
    "package_flash_pt": (lambda t: PKG.flash_pt(t, 20e5, Z), 320.0, 330.0),
    "package_flash_ph": (
        lambda h: PKG.flash_ph(20e5, h, Z),
        float(PKG.mixture_enthalpy(320.0, 20e5, Z)),
        float(PKG.mixture_enthalpy(330.0, 20e5, Z)),
    ),
    "package_bubble_pressure": (lambda t: PKG.bubble_pressure(t, Z), 250.0, 260.0),
}


@pytest.mark.parametrize("name", list(CALLS))
def test_a_new_operating_point_reuses_the_compiled_kernel(name) -> None:
    call, first, second = CALLS[name]
    _compiles_during(call, first)
    assert _compiles_during(call, second) == 0


def _recycle() -> Flowsheet:
    fs = Flowsheet()
    fs.feed("fresh", FEED)
    fs.unit(
        "mixer",
        lambda fresh, recycle, th: mix([fresh, recycle]),
        inputs=("fresh", "recycle"),
        outputs=("mixed",),
    )
    fs.unit(
        "drum",
        lambda mixed, th: flash_drum(mixed, th["T"], 20e5),
        inputs=("mixed",),
        outputs=("vapor", "liquid"),
    )
    fs.unit(
        "split",
        lambda liquid, th: splitter(liquid, jnp.array([th["r"], 1.0 - th["r"]])),
        inputs=("liquid",),
        outputs=("recycle", "purge"),
    )
    return fs


def test_a_flowsheet_resolve_reuses_its_recycle_map() -> None:
    fs = _recycle()

    def solve(t, r):
        result = fs.solve_with_info({"T": jnp.asarray(t), "r": jnp.asarray(r)})
        assert bool(result.converged)
        return result.streams

    _compiles_during(solve, 320.0, 0.5)
    assert _compiles_during(solve, 325.0, 0.6) == 0


def test_a_new_case_runner_reuses_unit_templates() -> None:
    case = example_case("heater")
    CaseRunner(case).evaluate()
    assert _compiles_during(lambda: CaseRunner(case).evaluate().metrics) == 0


def test_templates_that_differ_in_a_literal_setting_never_share_a_kernel() -> None:
    def literal(t_out: float) -> ProcessCase:
        document = example_case("heater").to_dict()
        document["units"][0]["settings"]["t_out"] = {"value": t_out, "unit": "K"}
        return ProcessCase.from_dict(document)

    for t_out in (350.0, 360.0):
        evaluation = CaseRunner(literal(t_out)).evaluate()
        assert float(evaluation.streams["product"].t) == pytest.approx(t_out)
