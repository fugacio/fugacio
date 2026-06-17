"""Surface tension of pure liquids and mixtures.

Pure components dispatch between the curated Mulero-Cachadina fits (REFPROP-grade
multi-term ``(1-Tr)^n`` expansions transcribed/refitted from open data) and the
Brock-Bird corresponding-states estimate (`brock_bird_surface_tension`)
when no fit exists. Mixtures use the Winterfeld-Scriven-Davis combination rule
(`winterfeld_scriven_davis`), which weights component tensions by their
liquid molar volumes.

All tensions are in N/m. Tension vanishes at the (mixture) critical point by
construction of both routes.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from fugacio.thermo._property_data import SIGMA_MULERO_CACHADINA
from fugacio.thermo.components import Component, get
from fugacio.thermo.correlations import mulero_cachadina
from fugacio.thermo.volumetric import liquid_molar_volumes

ArrayLike = Array | float


def brock_bird_surface_tension(t: ArrayLike, tb: ArrayLike, tc: ArrayLike, pc: ArrayLike) -> Array:
    """Brock-Bird corresponding-states surface tension (N/m).

    ``sigma = Pc^(2/3) * Tc^(1/3) * Q * (1 - Tr)^(11/9)`` with the Miller
    factor::

        Q = 0.1196 * (1 + Tbr * ln(Pc/1 atm) / (1 - Tbr)) - 0.279

    where ``Pc`` enters the prefactor in bar and ``sigma`` emerges in mN/m
    (converted to N/m here); typical accuracy is ~5% for non-polar and weakly
    polar liquids, worse for alcohols and water (Poling et al., 5th ed.,
    eq. 12-3.5).
    """
    tc = jnp.asarray(tc)
    pc = jnp.asarray(pc)
    tr = jnp.clip(jnp.asarray(t) / tc, 0.0, 1.0)
    tbr = jnp.asarray(tb) / tc
    q = 0.1196 * (1.0 + tbr * jnp.log(pc / 101325.0) / (1.0 - tbr)) - 0.279
    pc_bar = pc * 1.0e-5
    sigma_mn_per_m = pc_bar ** (2.0 / 3.0) * tc ** (1.0 / 3.0) * q * (1.0 - tr) ** (11.0 / 9.0)
    return sigma_mn_per_m * 1.0e-3


def winterfeld_scriven_davis(x: Array, sigma: Array, v_liquid: Array) -> Array:
    """Winterfeld-Scriven-Davis mixture surface tension (N/m).

    ``sigma_m = sum_i sum_j x_i x_j V_i V_j sqrt(sigma_i sigma_j)
    / (sum_i x_i V_i)^2`` -- volume-fraction-squared weighting of the geometric
    pair means (Winterfeld, Scriven & Davis 1978). Exact in the pure limits.
    """
    x = jnp.asarray(x)
    sigma = jnp.asarray(sigma)
    v = jnp.asarray(v_liquid)
    num = jnp.sum((x * v)[:, None] * (x * v)[None, :] * jnp.sqrt(sigma[:, None] * sigma[None, :]))
    return num / jnp.sum(x * v) ** 2


# --- Name-based dispatchers -------------------------------------------------------


def _resolve(components: list[str] | list[Component]) -> list[Component]:
    return [get(c) if isinstance(c, str) else c for c in components]


def surface_tensions(components: list[str] | list[Component], t: ArrayLike) -> Array:
    """Per-component surface tensions ``sigma_i(T)`` (N/m).

    Curated Mulero-Cachadina fit where available; Brock-Bird otherwise (which
    needs a normal boiling point; components lacking ``tb`` raise).
    """
    values: list[Array] = []
    for comp in _resolve(components):
        fit = SIGMA_MULERO_CACHADINA.get(comp.name)
        if fit is not None:
            tc_fit, s0, n0, s1, n1, s2, n2, _tmin, _tmax = fit
            values.append(mulero_cachadina(t, tc_fit, s0, n0, s1, n1, s2, n2))
        elif comp.tb is not None:
            values.append(brock_bird_surface_tension(t, comp.tb, comp.tc, comp.pc))
        else:
            raise ValueError(
                f"component {comp.name!r} has neither a surface-tension fit nor a "
                f"boiling point for the Brock-Bird estimate"
            )
    return jnp.stack(values)


def mixture_surface_tension(
    components: list[str] | list[Component], t: ArrayLike, x: Array
) -> Array:
    """Liquid-mixture surface tension by Winterfeld-Scriven-Davis (N/m)."""
    resolved = _resolve(components)
    sigma = surface_tensions(resolved, t)
    v = liquid_molar_volumes(resolved, t)
    return winterfeld_scriven_davis(jnp.asarray(x), sigma, v)
