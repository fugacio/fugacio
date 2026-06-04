"""Differentiable process-simulation layer for Fugacio (depends on ``fugacio.thermo``)."""

from fugacio.sim.vle import antoine_psat, bubble_pressure

__all__ = ["antoine_psat", "bubble_pressure"]

__version__ = "0.0.1"
