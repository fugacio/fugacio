"""Differentiable thermodynamics and physical-property engine for Fugacio.

Everything here is written against :mod:`jax.numpy`, so gradients flow cleanly
through the rest of the Fugacio stack (``fugacio.sim``, ``fugacio.copilot``).
"""

from fugacio.thermo.activity import (
    margules_excess_gibbs,
    margules_gamma,
    margules_ln_gamma,
)

__all__ = [
    "margules_excess_gibbs",
    "margules_gamma",
    "margules_ln_gamma",
]

__version__ = "0.0.1"
