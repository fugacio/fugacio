"""Enable 64-bit JAX arithmetic when ``fugacio.thermo`` is imported.

Cubic-root selection, Newton iterations to ``1e-10`` residuals, and the
implicit-function-theorem gradients throughout Fugacio assume double precision;
in JAX's default 32-bit mode they lose digits or fail to converge. This module
runs before any other ``fugacio.thermo`` import and flips ``jax_enable_x64``
on, unless the environment variable ``FUGACIO_X64`` is set to ``"0"`` (for
users who deliberately run single precision and accept the consequences).
"""

from __future__ import annotations

import os

import jax

ENABLED: bool = os.environ.get("FUGACIO_X64", "1") != "0"
"""Whether this import enabled 64-bit mode."""

if ENABLED:
    jax.config.update("jax_enable_x64", True)
