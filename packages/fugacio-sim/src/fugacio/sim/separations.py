"""Liquid-liquid and three-phase separators on an activity-coefficient package.

These complement the vapor-liquid blocks in `fugacio.sim.units` with the
non-ideal phase behavior an activity-coefficient (gamma-phi) package describes:

* `decanter`: a liquid-liquid separator (settling tank) that splits one feed
  into two conjugate liquid products via the stability-first isoactivity flash;
* `three_phase_flash`: a vapor + two-liquid (V-L-L) separator for heterogeneous
  systems (water/organic decantation, heteroazeotropic columns).

Both follow the unit contract: an eager call raises on failure (with a hint
naming the separator that does apply), and a traced call returns NaN products.
Every product is a differentiable `Stream`; flows carry gradients with respect
to the operating ``T``, ``P``, the feed, and the package's activity parameters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.stream import Stream
from fugacio.thermo.diagnostics import (
    ConvergenceError,
    SolveStatus,
    nan_unless_converged,
    require_converged,
)
from fugacio.thermo.lle import flash_lle_with_info
from fugacio.thermo.package import GammaPhiPackage
from fugacio.thermo.vlle import flash_vlle_with_info

ArrayLike = Array | float


def _activity_package(model: object, unit: str) -> GammaPhiPackage:
    if not isinstance(model, GammaPhiPackage):
        raise TypeError(
            f"{unit} needs an activity-coefficient (gamma-phi) package, for example "
            "package_for(components, 'nrtl'); got " + type(model).__name__
        )
    return model


def decanter(
    feed: Stream,
    model: GammaPhiPackage,
    *,
    t: ArrayLike | None = None,
    tol: float = 1e-12,
    max_iter: int = 400,
) -> tuple[Stream, Stream]:
    """Liquid-liquid settler: split ``feed`` into two conjugate liquid products.

    Solves the stability-first isoactivity LLE flash
    (`fugacio.thermo.lle.flash_lle_with_info`) with the package's activity model
    at temperature ``t`` (default: the feed temperature) and the feed pressure.
    A feed outside any miscibility gap is one liquid: ``liquid_I`` carries it all
    and ``liquid_II`` is empty.

    Returns:
        ``(liquid_I, liquid_II)`` product streams. The two isoactivity roots are
        symmetric, so the order is made deterministic by returning the product
        richest in the first component (index 0) as ``liquid_I``.

    Raises:
        TypeError: If ``model`` isn't a gamma-phi package.
        ConvergenceError: If an eager split fails.
    """
    pkg = _activity_package(model, "decanter")
    t_arr = feed.t if t is None else jnp.asarray(t)
    solved = flash_lle_with_info(pkg.activity, t_arr, feed.z, tol=tol, max_iter=max_iter)
    res = solved.value
    total = feed.total
    n_a = res.x_i * (1.0 - res.psi) * total
    n_b = res.x_ii * res.psi * total
    # Canonical phase order: the component-0-rich product is liquid_I. Swapping the
    # whole stream (not just the composition) preserves the material balance.
    i_first = (res.x_i[0] >= res.x_ii[0]) | (res.psi <= 0)
    n_i = jnp.where(i_first, n_a, n_b)
    n_ii = jnp.where(i_first, n_b, n_a)
    liquid_i = Stream(
        n=n_i, vapor_n=jnp.zeros_like(n_i), t=t_arr, p=feed.p, components=feed.components
    )
    liquid_ii = Stream(
        n=n_ii, vapor_n=jnp.zeros_like(n_ii), t=t_arr, p=feed.p, components=feed.components
    )
    require_converged(solved.report, "decanter liquid-liquid split")
    return nan_unless_converged((liquid_i, liquid_ii), solved.report)


def three_phase_flash(
    feed: Stream,
    t: ArrayLike,
    p: ArrayLike,
    model: GammaPhiPackage,
    *,
    tol: float = 1e-11,
    max_iter: int = 300,
) -> tuple[Stream, Stream, Stream]:
    """Vapor-liquid-liquid (V-L-L) flash of ``feed`` at ``(T, P)``.

    Drives the three-phase flash (`fugacio.thermo.vlle.flash_vlle_with_info`)
    with the package's activity liquid and its EOS or ideal vapor. Use it where a
    vapor coexists with two liquids (water/organic decantation, heteroazeotropic
    distillation).

    Returns:
        ``(vapor, liquid_I, liquid_II)`` product streams.

    Raises:
        TypeError: If ``model`` isn't a gamma-phi package.
        ConvergenceError: If an eager flash fails. A feed that isn't three-phase
            at ``(T, P)`` is reported as infeasible, with a hint to use
            `fugacio.sim.flash_drum` (vapor-liquid) or `decanter` (liquid-liquid).
    """
    pkg = _activity_package(model, "three-phase flash")
    solved = flash_vlle_with_info(
        pkg.activity,
        t,
        p,
        feed.z,
        pkg.tc,
        pkg.pc,
        pkg.omega,
        eos=pkg.eos,
        kij=pkg.kij,
        vapor=pkg.vapor,
        poynting=pkg.poynting,
        phi_saturation=pkg.phi_saturation,
        tol=tol,
        max_iter=max_iter,
    )
    report = solved.report
    traced = isinstance(report.status, jax.core.Tracer)
    if not traced and int(report.status) == int(SolveStatus.INFEASIBLE):
        raise ConvergenceError(
            report,
            "three-phase flash: the feed isn't three-phase at these conditions; use "
            "flash_drum for a vapor-liquid split or decanter for a liquid-liquid split",
        )
    require_converged(report, "three-phase flash")
    res = solved.value
    total = feed.total
    t_arr = jnp.asarray(t, dtype=float)
    p_arr = jnp.asarray(p, dtype=float)
    vapor = Stream(
        n=res.y * res.beta_v * total,
        vapor_n=res.y * res.beta_v * total,
        t=t_arr,
        p=p_arr,
        components=feed.components,
    )
    liquid_i = Stream(
        n=res.x_i * res.beta_l1 * total,
        vapor_n=jnp.zeros_like(feed.n),
        t=t_arr,
        p=p_arr,
        components=feed.components,
    )
    liquid_ii = Stream(
        n=res.x_ii * res.beta_l2 * total,
        vapor_n=jnp.zeros_like(feed.n),
        t=t_arr,
        p=p_arr,
        components=feed.components,
    )
    return nan_unless_converged((vapor, liquid_i, liquid_ii), report)


__all__ = ["decanter", "three_phase_flash"]
