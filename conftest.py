"""Pytest session configuration.

Thermodynamic calculations want double precision, so enable JAX's float64 mode
for the whole test suite. (Application code should opt in the same way.)

The persistent compilation cache makes the solver-heavy tests cheap after the
first run: the reference Helmholtz state functions (``fugacio.thermo.helmholtz``)
embed Newton/bisection loops whose XLA compilation takes tens of seconds per
fluid, and the cache reuses those binaries across pytest sessions (and across
CI runs when ``.jax_cache`` is restored from the actions cache).
"""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", str(Path(__file__).parent / ".jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
