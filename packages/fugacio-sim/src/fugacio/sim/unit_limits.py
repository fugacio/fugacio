"""Operating limits shared by the unit kernels and the process-case audit.

Each rule is a traceable predicate, so a unit evaluated inside a compiled
flowsheet reports a violated limit (and returns NaN outputs) while an eager
call raises a `ValueError` naming the rule. The saved-case audit
(`fugacio.sim.cases.results`) evaluates the same predicates on solved
streams, so a limit can't be enforced in one place and forgotten in another.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

ArrayLike = Array | float

#: Pressure tolerance (Pa) for "no pressure rise" and "no pressure drop" rules.
PRESSURE_TOLERANCE = 1e-4

#: Tolerance on a set of split fractions summing to one.
FRACTION_TOLERANCE = 1e-10

#: Largest vapor fraction a liquid-only machine (a pump) accepts at its inlet.
LIQUID_INLET_VAPOR_FRACTION = 1e-7

NO_PRESSURE_RISE = "a unit without shaft work can't raise pressure (outlet above inlet)"
NO_PRESSURE_DROP = "compression equipment requires an outlet pressure at least the inlet pressure"
EXPANSION_ONLY = "an expander requires an outlet pressure at most the inlet pressure"
SPLIT_FRACTIONS = "split fractions must each lie in [0, 1] and sum to one"
RECOVERIES = "component recoveries must each lie in [0, 1]"
LIQUID_INLET = "the liquid-pump model requires a liquid inlet; use a compressor for vapor"


def pressure_not_raised(p_in: ArrayLike, p_out: ArrayLike) -> Array:
    """Valves, expanders, separators, and mixers can't raise pressure."""
    return jnp.asarray(p_out) <= jnp.asarray(p_in) + PRESSURE_TOLERANCE


def pressure_not_lowered(p_in: ArrayLike, p_out: ArrayLike) -> Array:
    """Pumps and compressors can't lower pressure."""
    return jnp.asarray(p_out) >= jnp.asarray(p_in) - PRESSURE_TOLERANCE


def split_fractions_valid(fractions: ArrayLike) -> Array:
    """Conservative split: fractions in ``[0, 1]`` summing to one."""
    f = jnp.asarray(fractions)
    return jnp.all((f >= 0) & (f <= 1)) & (jnp.abs(jnp.sum(f) - 1.0) <= FRACTION_TOLERANCE)


def recoveries_valid(recoveries: ArrayLike) -> Array:
    """Per-component recoveries in ``[0, 1]`` (the complement leaves the other outlet)."""
    r = jnp.asarray(recoveries)
    return jnp.all((r >= 0) & (r <= 1))


def liquid_inlet(vapor_fraction: ArrayLike) -> Array:
    """A liquid-only machine's inlet is (within roundoff) all liquid."""
    return jnp.asarray(vapor_fraction) <= LIQUID_INLET_VAPOR_FRACTION


def require(ok: Array, message: str) -> Array:
    """Raise `ValueError` for a concrete violated rule; return the flag for tracing.

    Raises:
        ValueError: If ``ok`` is concretely false.
    """
    if not isinstance(ok, jax.core.Tracer) and not bool(ok):
        raise ValueError(message)
    return ok


__all__ = [
    "EXPANSION_ONLY",
    "FRACTION_TOLERANCE",
    "LIQUID_INLET",
    "LIQUID_INLET_VAPOR_FRACTION",
    "NO_PRESSURE_DROP",
    "NO_PRESSURE_RISE",
    "PRESSURE_TOLERANCE",
    "RECOVERIES",
    "SPLIT_FRACTIONS",
    "liquid_inlet",
    "pressure_not_lowered",
    "pressure_not_raised",
    "recoveries_valid",
    "require",
    "split_fractions_valid",
]
