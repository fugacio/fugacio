"""Differentiable process-simulation layer for Fugacio (depends on ``fugacio.thermo``).

The layer provides:

* :class:`Stream` -- the differentiable material stream passed between units;
* unit operations (:func:`flash_drum`, :func:`mix`) built on the EOS phase
  equilibrium in :mod:`fugacio.thermo`;
* the original lightweight modified-Raoult helpers (:func:`bubble_pressure`,
  :func:`antoine_psat`).
"""

from fugacio.sim.stream import Stream
from fugacio.sim.units import flash_drum, mix
from fugacio.sim.vle import antoine_psat, bubble_pressure

__all__ = [
    "Stream",
    "antoine_psat",
    "bubble_pressure",
    "flash_drum",
    "mix",
]

__version__ = "0.0.1"
