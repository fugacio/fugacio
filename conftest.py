"""Pytest session configuration.

Thermodynamic calculations want double precision, so enable JAX's float64 mode
for the whole test suite. (Application code should opt in the same way.)
"""

import jax

jax.config.update("jax_enable_x64", True)
