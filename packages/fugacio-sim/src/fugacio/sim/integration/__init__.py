"""Differentiable heat integration and pinch analysis for Fugacio.

Heat integration asks how to recover heat between a process's hot and cold
streams so the least possible external utility is bought, and what network of
exchangers achieves it. This subpackage supplies the whole pinch-technology
workflow, kept end-to-end differentiable so the targets compose with the rest of
the engine and can be optimised by gradients:

* :mod:`~fugacio.sim.integration.streams` -- the :class:`HeatStream` model and
  extraction of hot/cold streams (and their ``CP``) from process
  :class:`~fugacio.sim.stream.Stream` objects via the real, two-phase-aware
  enthalpy;
* :mod:`~fugacio.sim.integration.targeting` -- the problem table algorithm for
  the minimum hot/cold utilities and the pinch, plus the composite and grand
  composite curves;
* :mod:`~fugacio.sim.integration.area` -- the Bath-formula area target, the
  minimum-units target, capital and total-annual-cost targets, and the
  capital-energy trade-off (:func:`optimal_dt_min`, "supertargeting");
* :mod:`~fugacio.sim.integration.network` -- heat-exchanger-network synthesis by
  the pinch design method, with rigorous feasibility/MER verification.
"""

from __future__ import annotations

from fugacio.sim.integration.area import (
    DEFAULT_AREA_COST,
    OptimalDtMin,
    SuperTargetResult,
    UnitsTarget,
    area_target,
    capital_cost_target,
    optimal_dt_min,
    supertarget,
    total_annual_cost_target,
    units_target,
)
from fugacio.sim.integration.network import (
    Exchanger,
    HeatExchangerNetwork,
    synthesize_network,
    verify_network,
)
from fugacio.sim.integration.streams import (
    DEFAULT_FILM_COEFFICIENT,
    HeatStream,
    heat_stream,
    make_stream,
)
from fugacio.sim.integration.targeting import (
    CompositeCurves,
    GrandComposite,
    HeatCascade,
    PinchResult,
    composite_curves,
    grand_composite_curve,
    heat_cascade,
    minimum_utilities,
    pinch_analysis,
)

__all__ = [
    "DEFAULT_AREA_COST",
    "DEFAULT_FILM_COEFFICIENT",
    "CompositeCurves",
    "Exchanger",
    "GrandComposite",
    "HeatCascade",
    "HeatExchangerNetwork",
    "HeatStream",
    "OptimalDtMin",
    "PinchResult",
    "SuperTargetResult",
    "UnitsTarget",
    "area_target",
    "capital_cost_target",
    "composite_curves",
    "grand_composite_curve",
    "heat_cascade",
    "heat_stream",
    "make_stream",
    "minimum_utilities",
    "optimal_dt_min",
    "pinch_analysis",
    "supertarget",
    "synthesize_network",
    "total_annual_cost_target",
    "units_target",
    "verify_network",
]
