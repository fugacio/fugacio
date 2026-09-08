"""Rigorous equilibrium-stage columns: simultaneous-correction MESH solve.

The shortcut and constant-molar-overflow columns in `fugacio.sim.column` are
design-screening tools. This module is the rigorous counterpart a process
simulator is judged by: every stage carries its own **M**aterial balances,
**E**quilibrium relations, **S**ummation constraints, and energy (**H**eat)
balance, and the whole set is solved *simultaneously* by Newton's method in the
manner of Naphtali and Sandholm (1971), with the Jacobian supplied exactly by JAX
autodiff. Because the stage energy balances are included, the internal vapour
and liquid traffic varies up the column as the real heats of vaporisation and
mixing dictate, which is what the constant-molar-overflow assumption discards.

The column is described generically enough to cover the separations that make
up a plant:

* any number of feeds, each on its own stage (`ColumnFeed`), entering at
  whatever thermal condition the feed stream carries;
* liquid or vapour side draws (`SideDraw`) as a fraction of the phase leaving
  a stage, and per-stage heat duties (`StageDuty`) for intercoolers and
  interreboilers;
* a total or partial condenser (or none, for an absorber), a kettle reboiler
  (or none, for a stripper), and an optional subcooled-reflux temperature;
* Murphree vapour efficiencies per stage or for the whole column;
* a linear pressure profile between the top and bottom stages;
* any two column specifications (`ColumnSpec`) to close the condenser and
  reboiler degrees of freedom: reflux ratio, distillate or bottoms rate, boilup
  ratio, a duty, a product purity, a component recovery, or a stage temperature;
* any property package (`fugacio.sim.models.package_for`), so an ethanol/water
  column runs on NRTL and a demethaniser on Peng-Robinson with the same code.

Unknowns are the logarithms of the component liquid and vapour flows leaving
every stage (so flows stay positive through the Newton iteration) plus the stage
temperatures and the condenser/reboiler duties; the equilibrium relations are
written in logarithmic form so trace components are as well conditioned as bulk
ones. A bubble-point (Wang-Henke) sweep seeds the iteration, then
`fugacio.thermo.implicit.newton_system` converges it with a backtracking line
search. The converged column is differentiable with respect to the feeds, the
pressure profile, the specifications, the side-draw fractions, the duties, and the
property package by the implicit function theorem, so column design variables
can be optimised by gradient descent through a fully rigorous model.

`absorber` and `stripper` are thin wrappers for the two-feed, no-condenser,
no-reboiler configurations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.properties import Model, molar_enthalpy, resolve_package
from fugacio.sim.stream import Stream
from fugacio.thermo import PR, CubicEOS, PropertyPackage
from fugacio.thermo.activity.models import NRTL
from fugacio.thermo.diagnostics import SolveReport, SolveResult, SolveStatus, require_converged
from fugacio.thermo.implicit import (
    fixed_point_with_info,
    implicit_solution,
    newton_system_with_info,
)
from fugacio.thermo.package import GammaPhiPackage

ArrayLike = Array | float

#: Temperature scale (K) that non-dimensionalises the stage temperatures.
_T_SCALE = 100.0
#: Molar enthalpy scale (J/mol) for the stage energy balances.
_H_MOLAR_SCALE = 1.0e4
#: Floor on any component flow used to seed the logarithmic unknowns (mol/s).
_FLOW_FLOOR = 1.0e-12


# --------------------------------------------------------------------------- #
# Column description
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ColumnFeed:
    """A feed stream entering the column on a given stage.

    Attributes:
        stream: The feed `Stream` (its temperature and pressure set its
            enthalpy, so a subcooled, saturated, or partially vaporised feed is
            handled without a separate quality parameter).
        stage: Stage number, counted from the top with the condenser (when
            present) as stage 1 and the reboiler (when present) as stage
            ``n_stages``.
    """

    stream: Stream
    stage: int


@dataclass(frozen=True)
class SideDraw:
    """A side product withdrawn from a stage.

    Attributes:
        stage: Stage number (1 = top).
        phase: ``"liquid"`` or ``"vapor"``.
        fraction: Fraction of that phase *leaving the stage* that is withdrawn
            (a ratio keeps the material balance well posed for any column
            traffic; differentiable).
    """

    stage: int
    phase: str = "liquid"
    fraction: ArrayLike = 0.1


@dataclass(frozen=True)
class StageDuty:
    """An external heat duty on an intermediate stage (intercooler / interreboiler).

    Attributes:
        stage: Stage number (1 = top).
        duty: Heat added to the stage (W); negative removes heat.
    """

    stage: int
    duty: ArrayLike


@dataclass(frozen=True)
class ColumnSpec:
    """A column specification closing one condenser/reboiler degree of freedom.

    Build these with the helper constructors (`reflux_ratio`, `distillate_rate`,
    `purity`, ...) rather than by hand.

    Attributes:
        kind: One of ``"reflux_ratio"``, ``"reflux_rate"``, ``"distillate_rate"``,
            ``"bottoms_rate"``, ``"boilup_ratio"``, ``"condenser_duty"``,
            ``"reboiler_duty"``, ``"mole_fraction"``, ``"recovery"``,
            ``"component_flow"``, ``"stage_temperature"``.
        value: Target value (differentiable).
        component: Component index for composition / recovery / flow specs.
        product: ``"distillate"``, ``"bottoms"``, or ``"draw:<k>"`` (the ``k``-th
            side draw, zero-based) for product-based specs.
        stage: Stage number for ``"stage_temperature"``.
    """

    kind: str
    value: ArrayLike
    component: int | None = None
    product: str = "distillate"
    stage: int | None = None


def reflux_ratio(value: ArrayLike) -> ColumnSpec:
    """Spec the molar reflux ratio ``L_1 / D``."""
    return ColumnSpec("reflux_ratio", value)


def reflux_rate(value: ArrayLike) -> ColumnSpec:
    """Spec the molar reflux flow ``L_1`` (mol/s)."""
    return ColumnSpec("reflux_rate", value)


def distillate_rate(value: ArrayLike) -> ColumnSpec:
    """Spec the total distillate flow ``D`` (mol/s)."""
    return ColumnSpec("distillate_rate", value)


def bottoms_rate(value: ArrayLike) -> ColumnSpec:
    """Spec the total bottoms flow ``B`` (mol/s)."""
    return ColumnSpec("bottoms_rate", value)


def boilup_ratio(value: ArrayLike) -> ColumnSpec:
    """Spec the boilup ratio ``V_N / B``."""
    return ColumnSpec("boilup_ratio", value)


def condenser_duty(value: ArrayLike) -> ColumnSpec:
    """Spec the condenser duty (W, negative for heat removed)."""
    return ColumnSpec("condenser_duty", value)


def reboiler_duty(value: ArrayLike) -> ColumnSpec:
    """Spec the reboiler duty (W, positive for heat added)."""
    return ColumnSpec("reboiler_duty", value)


def purity(product: str, component: int, value: ArrayLike) -> ColumnSpec:
    """Spec the mole fraction of ``component`` in ``product``."""
    return ColumnSpec("mole_fraction", value, component=component, product=product)


def recovery(component: int, product: str, value: ArrayLike) -> ColumnSpec:
    """Spec the fraction of the fed ``component`` that leaves in ``product``."""
    return ColumnSpec("recovery", value, component=component, product=product)


def component_flow(product: str, component: int, value: ArrayLike) -> ColumnSpec:
    """Spec the molar flow of ``component`` in ``product`` (mol/s)."""
    return ColumnSpec("component_flow", value, component=component, product=product)


def stage_temperature(stage: int, value: ArrayLike) -> ColumnSpec:
    """Spec the temperature of stage ``stage`` (K)."""
    return ColumnSpec("stage_temperature", value, stage=stage)


class RigorousColumnResult(NamedTuple):
    """Converged rigorous column.

    Attributes:
        distillate: Top product (liquid for a total condenser, vapour for a
            partial condenser, the overhead vapour for an absorber).
        bottoms: Bottom liquid product.
        side_draws: Side products in the order the `SideDraw` entries were given.
        condenser_duty: Condenser duty (W, negative = heat removed); zero if none.
        reboiler_duty: Reboiler duty (W); zero if none.
        t: Stage temperatures (K), top to bottom.
        p: Stage pressures (Pa).
        x: Liquid mole fractions per stage, shape ``(n_stages, n_components)``.
        y: Vapour mole fractions per stage.
        k: K-values per stage.
        liquid_flow: Total liquid leaving each stage (mol/s), including any draw.
        vapor_flow: Total vapour leaving each stage (mol/s), including any draw.
        reflux_ratio: ``L_1 / D`` at the solution.
        boilup_ratio: ``V_N / B`` at the solution.
        residual_norm: Max-norm of the scaled MESH residual at the solution.
    """

    distillate: Stream
    bottoms: Stream
    side_draws: tuple[Stream, ...]
    condenser_duty: Array
    reboiler_duty: Array
    t: Array
    p: Array
    x: Array
    y: Array
    k: Array
    liquid_flow: Array
    vapor_flow: Array
    reflux_ratio: Array
    boilup_ratio: Array
    residual_norm: Array
    report: SolveReport

    def warm_start(self) -> dict[str, Array]:
        """Full stage state for ``rigorous_column(..., guess=result.warm_start())``."""
        return {
            "liquid": self.liquid_flow[:, None] * self.x,
            "vapor": self.vapor_flow[:, None] * self.y,
            "t": self.t,
            "condenser_duty": self.condenser_duty,
            "reboiler_duty": self.reboiler_duty,
        }

    @property
    def converged(self) -> Array:
        """Whether the stage balances and specifications converged."""
        return self.report.converged


# --------------------------------------------------------------------------- #
# Static structure and differentiable parameters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Structure:
    """Everything about the column that is fixed during a solve (Python-static)."""

    n: int
    c: int
    components: tuple[str, ...]
    condenser: str | None
    reboiler: str | None
    feed_stages: tuple[int, ...]
    draw_stages: tuple[int, ...]
    draw_phases: tuple[str, ...]
    duty_stages: tuple[int, ...]
    spec_kinds: tuple[str, ...]
    spec_components: tuple[int | None, ...]
    spec_products: tuple[str, ...]
    spec_stages: tuple[int | None, ...]
    subcooled: bool

    @property
    def has_condenser(self) -> bool:
        return self.condenser is not None

    @property
    def has_reboiler(self) -> bool:
        return self.reboiler is not None

    @property
    def n_unknowns(self) -> int:
        return self.n * (2 * self.c + 1) + int(self.has_condenser) + int(self.has_reboiler)


class _Unknowns(NamedTuple):
    ln_l: Array
    ln_v: Array
    t: Array
    q_c: Array
    q_r: Array


def _unpack(u: Array, st: _Structure) -> _Unknowns:
    n, c = st.n, st.c
    ln_l = u[: n * c].reshape(n, c)
    ln_v = u[n * c : 2 * n * c].reshape(n, c)
    t = u[2 * n * c : 2 * n * c + n] * _T_SCALE
    k = 2 * n * c + n
    q_c = u[k] if st.has_condenser else jnp.asarray(0.0)
    if st.has_condenser:
        k += 1
    q_r = u[k] if st.has_reboiler else jnp.asarray(0.0)
    return _Unknowns(ln_l, ln_v, t, q_c, q_r)


def _pack(ln_l: Array, ln_v: Array, t: Array, q_c: Array, q_r: Array, st: _Structure) -> Array:
    parts = [ln_l.reshape(-1), ln_v.reshape(-1), t / _T_SCALE]
    if st.has_condenser:
        parts.append(jnp.reshape(q_c, (1,)))
    if st.has_reboiler:
        parts.append(jnp.reshape(q_r, (1,)))
    return jnp.concatenate(parts)


def _stage_index(stage: int, n: int, what: str) -> int:
    if not 1 <= stage <= n:
        raise ValueError(f"{what} stage {stage} is outside 1..{n}")
    return stage - 1


def _product_flows(
    name: str, liq: Array, v: Array, s_l: Array, s_v: Array, st: _Structure
) -> Array:
    """Component molar flows of a named product at the current iterate."""
    if name == "distillate":
        return v[0]
    if name == "bottoms":
        return liq[st.n - 1]
    if name.startswith("draw:"):
        k = int(name.split(":", 1)[1])
        j = st.draw_stages[k]
        return s_l[j] * liq[j] if st.draw_phases[k] == "liquid" else s_v[j] * v[j]
    raise ValueError(f"unknown product {name!r}; use 'distillate', 'bottoms', or 'draw:<k>'")


# --------------------------------------------------------------------------- #
# Residuals
# --------------------------------------------------------------------------- #


@jax.jit
def _incipient_vapor_k(pkg: PropertyPackage, t: Array, p: Array, x: Array) -> Array:
    """Self-consistent incipient-vapor K-values above a specified liquid.

    A total condenser's distillate composition is liquid, so it cannot also be
    used as the vapor composition in a phi-phi bubble equation. Converge that
    absent vapor separately and differentiate its fixed point implicitly.
    Ideal-vapor gamma-phi packages need no composition iteration.
    """
    if isinstance(pkg, GammaPhiPackage) and pkg.vapor == "ideal":
        return pkg.k_values(t, p, x, x)
    seed = pkg.k_seed(t, p, x) * x
    seed = seed / jnp.sum(seed)

    def update(y: Array, theta: Any) -> Array:
        model, temperature, pressure, liquid = theta
        proposed = model.k_values(temperature, pressure, liquid, y) * liquid
        return proposed / jnp.sum(proposed)

    solved = fixed_point_with_info(update, seed, (pkg, t, p, x), tol=1e-12, max_iter=100)
    k = pkg.k_values(t, p, x, solved.value)
    return jnp.where(solved.report.converged, k, jnp.nan)


@jax.custom_jvp
def _bubble_sum(pkg: PropertyPackage, t: Array, p: Array, x: Array) -> Array:
    """Bubble closure with a Gibbs-Duhem reduction of its implicit derivative."""
    return jnp.sum(_incipient_vapor_k(pkg, t, p, x) * x)


@_bubble_sum.defjvp
def _bubble_sum_jvp(primals: tuple, tangents: tuple) -> tuple[Array, Array]:
    pkg, t, p, x = primals
    k = _incipient_vapor_k(pkg, t, p, x)
    value = jnp.sum(k * x)
    y = k * x / value
    # At the normalized fixed point, K*x = S*y. The contribution from
    # changing y to dS is -S * sum(y_i * d ln(phi_i^V)), which vanishes by
    # Gibbs-Duhem. Hold y fixed in this first directional derivative to avoid
    # nesting another implicit Jacobian inside the MESH and recycle Jacobians.
    # Keep y differentiable above: higher derivatives still include its motion.
    _, tangent = jax.jvp(
        lambda model, tt, pp, xx: jnp.sum(model.k_values(tt, pp, xx, y) * xx),
        primals,
        tangents,
    )
    return value, tangent


def _stage_properties(
    pkg: PropertyPackage, t: Array, p: Array, x: Array, y: Array, st: _Structure
) -> tuple[Array, Array, Array]:
    """K-values and phase enthalpies on every stage (vectorised over stages)."""
    k = jax.vmap(lambda tj, pj, xj, yj: pkg.k_values(tj, pj, xj, yj))(t, p, x, y)
    h_l = jax.vmap(lambda tj, pj, xj: pkg.enthalpy(tj, pj, xj, phase="liquid"))(t, p, x)
    h_v = jax.vmap(lambda tj, pj, yj: pkg.enthalpy(tj, pj, yj, phase="vapor"))(t, p, y)
    if st.condenser == "total":
        # The "vapour" leaving a total condenser is the liquid distillate.
        h_d = pkg.enthalpy(t[0], p[0], y[0], phase="liquid")
        h_v = h_v.at[0].set(h_d)
    return k, h_l, h_v


def _residuals(u: Array, theta: dict[str, Any], st: _Structure) -> Array:
    """Scaled MESH residual vector plus the column specifications."""
    pkg: PropertyPackage = theta["pkg"]
    n, c = st.n, st.c
    un = _unpack(u, st)
    liq = jnp.exp(un.ln_l)
    v = jnp.exp(un.ln_v)
    big_l = jnp.sum(liq, axis=1)
    big_v = jnp.sum(v, axis=1)
    x = liq / big_l[:, None]
    y = v / big_v[:, None]
    t = un.t
    p = theta["p"]
    s_l = theta["s_l"]
    s_v = theta["s_v"]
    eta = theta["eta"]
    f = theta["f"]
    hf = theta["hf"]
    f_total = theta["f_total"]
    q = theta["q"]
    if st.has_condenser:
        q = q.at[0].add(un.q_c)
    if st.has_reboiler:
        q = q.at[n - 1].add(un.q_r)

    k, h_l, h_v = _stage_properties(pkg, t, p, x, y, st)

    # Streams entering from the neighbouring stages (zero beyond the ends).
    l_above = jnp.concatenate([jnp.zeros((1, c)), liq[:-1]], axis=0)
    v_below = jnp.concatenate([v[1:], jnp.zeros((1, c))], axis=0)
    big_l_above = jnp.sum(l_above, axis=1)
    big_v_below = jnp.sum(v_below, axis=1)
    h_l_above = jnp.concatenate([jnp.zeros(1), h_l[:-1]])
    h_v_below = jnp.concatenate([h_v[1:], jnp.zeros(1)])

    # M: component material balances (scaled by the total feed).
    mat = (l_above + v_below + f - (1.0 + s_l)[:, None] * liq - (1.0 + s_v)[:, None] * v) / f_total

    # E: Murphree-corrected equilibrium in log form. The vapour from below is
    # the composition leaving the stage underneath; the bottom stage has none, so
    # its efficiency is effectively one.
    y_below = jnp.where(
        (big_v_below > 0.0)[:, None],
        v_below / jnp.where(big_v_below > 0.0, big_v_below, 1.0)[:, None],
        k * x,
    )
    y_eq = y_below + eta[:, None] * (k * x - y_below)
    equil = un.ln_v - jnp.log(big_v)[:, None] - jnp.log(jnp.clip(y_eq, 1e-300, None))
    if st.condenser == "total":
        # Total condenser: distillate and reflux share one composition (C - 1
        # independent equalities) and the reflux is saturated (bubble point) or
        # at the specified subcooled temperature.
        equality = (un.ln_v[0] - jnp.log(big_v[0])) - (un.ln_l[0] - jnp.log(big_l[0]))
        top = (
            (t[0] - theta["t_reflux"]) / _T_SCALE
            if st.subcooled
            else _bubble_sum(pkg, t[0], p[0], x[0]) - 1.0
        )
        equil = equil.at[0].set(jnp.concatenate([equality[: c - 1], jnp.reshape(top, (1,))]))

    # H: stage energy balances (scaled by the feed enthalpy-flow scale).
    energy = (
        big_l_above * h_l_above
        + big_v_below * h_v_below
        + hf
        + q
        - (1.0 + s_l) * big_l * h_l
        - (1.0 + s_v) * big_v * h_v
    ) / (f_total * _H_MOLAR_SCALE)

    # Column specifications.
    specs = []
    for kind, comp, prod, stage, value in zip(
        st.spec_kinds,
        st.spec_components,
        st.spec_products,
        st.spec_stages,
        theta["specs"],
        strict=True,
    ):
        if kind == "reflux_ratio":
            specs.append((big_l[0] / big_v[0] - value) / jnp.maximum(jnp.abs(value), 1.0))
        elif kind == "reflux_rate":
            specs.append((big_l[0] - value) / f_total)
        elif kind == "distillate_rate":
            specs.append((big_v[0] - value) / f_total)
        elif kind == "bottoms_rate":
            specs.append((big_l[n - 1] - value) / f_total)
        elif kind == "boilup_ratio":
            specs.append((big_v[n - 1] / big_l[n - 1] - value) / jnp.maximum(jnp.abs(value), 1.0))
        elif kind == "condenser_duty":
            specs.append((un.q_c - value) / (f_total * _H_MOLAR_SCALE))
        elif kind == "reboiler_duty":
            specs.append((un.q_r - value) / (f_total * _H_MOLAR_SCALE))
        elif kind == "stage_temperature":
            specs.append((t[stage] - value) / _T_SCALE)
        else:
            flows = _product_flows(prod, liq, v, s_l, s_v, st)
            if kind == "mole_fraction":
                specs.append(flows[comp] / jnp.sum(flows) - value)
            elif kind == "recovery":
                specs.append(flows[comp] / jnp.sum(f[:, comp]) - value)
            elif kind == "component_flow":
                specs.append((flows[comp] - value) / f_total)
            else:
                raise ValueError(f"unknown column spec kind {kind!r}")

    parts = [mat.reshape(-1), equil.reshape(-1), energy]
    if specs:
        parts.append(jnp.stack([jnp.reshape(s, ()) for s in specs]))
    return jnp.concatenate(parts)


# --------------------------------------------------------------------------- #
# Initialisation (bubble-point sweeps)
# --------------------------------------------------------------------------- #


def _k_estimate(pkg: PropertyPackage, t: Array, p: Array, x: Array) -> Array:
    """K-values of a liquid ``x`` against the vapour its own seed K-values imply.

    A phi-phi package evaluated with ``y == x`` returns ``K = 1`` wherever the
    EOS has a single density root, so the vapour composition is first estimated
    from the package's composition-light `k_seed` (Wilson for a cubic) and the
    rigorous K-values are then evaluated against that vapour.
    """
    y = pkg.k_seed(t, p, x) * x
    y = y / jnp.sum(y)
    return pkg.k_values(t, p, x, y)


def _bubble_newton(pkg: PropertyPackage, t: Array, p: Array, x: Array, steps: int = 2) -> Array:
    """A few damped Newton steps on ``sum_i K_i(T) x_i = 1`` for every stage."""

    def one(tj: Array, pj: Array, xj: Array) -> Array:
        def g(tt: Array) -> Array:
            return jnp.log(jnp.sum(_k_estimate(pkg, tt, pj, xj) * xj))

        def update(_: int, tj: Array) -> Array:
            val, slope = jax.value_and_grad(g)(tj)
            step = -val / jnp.where(jnp.abs(slope) > 1e-12, slope, 1e-12)
            return tj + jnp.clip(step, -25.0, 25.0)

        return jax.lax.fori_loop(0, steps, update, tj)

    return jax.vmap(one)(t, p, x)


def _seed(
    pkg: PropertyPackage,
    st: _Structure,
    theta: dict[str, Any],
    feeds: Sequence[Stream],
    guess: dict[str, Any],
    sweeps: int,
) -> Array:
    """Initial unknown vector from constant-molar-overflow flows and bubble-point sweeps."""
    n, c = st.n, st.c
    f = theta["f"]
    f_total = theta["f_total"]
    z = jnp.sum(f, axis=0) / f_total
    p = theta["p"]
    s_l, s_v = theta["s_l"], theta["s_v"]

    # Vapour / liquid portions of every feed at its own conditions.
    f_vap = jnp.zeros(n)
    f_liq = jnp.zeros(n)
    for fd, j in zip(feeds, st.feed_stages, strict=True):
        beta = pkg.flash_pt(fd.t, fd.p, fd.z).beta
        f_vap = f_vap.at[j].add(beta * fd.total)
        f_liq = f_liq.at[j].add((1.0 - beta) * fd.total)

    # Column traffic guesses.
    d_guess = jnp.asarray(guess.get("distillate_rate", 0.5 * f_total), dtype=float)
    r_guess = jnp.asarray(guess.get("reflux_ratio", 1.5), dtype=float)
    cum_liq = jnp.cumsum(f_liq)
    vap_from_below = jnp.cumsum(f_vap[::-1])[::-1]
    if st.has_condenser:
        big_v = jnp.full(n, d_guess * (r_guess + 1.0))
        big_v = big_v.at[0].set(d_guess)
        big_l = r_guess * d_guess + cum_liq
        big_l = big_l.at[n - 1].set(jnp.maximum(f_total - d_guess, 0.05 * f_total))
    else:
        big_v = jnp.maximum(vap_from_below, 0.05 * f_total)
        big_l = jnp.maximum(cum_liq, 0.05 * f_total)
    big_l = jnp.maximum(big_l, 0.02 * f_total)
    big_v = jnp.maximum(big_v, 0.02 * f_total)

    # Temperature seed: the flow-weighted feed temperature everywhere, refined
    # stage by stage by the bubble-point sweeps below (a bracketed bubble/dew
    # solve of the combined feed would be more elegant, but a cubic EOS has no
    # saturation point for a component above its critical temperature and the
    # brackets would have to be guessed anyway).
    refine_t = st.has_condenser or st.has_reboiler
    if "t" in guess:
        t = jnp.asarray(guess["t"], dtype=float)
    elif refine_t:
        t_feed = sum(fd.total * jnp.asarray(fd.t, dtype=float) for fd in feeds) / f_total
        t = jnp.full(n, t_feed)
        t = _bubble_newton(pkg, t, p, jnp.tile(z, (n, 1)), steps=6)
    else:
        # Absorber / stripper: the stages sit between the two feed temperatures;
        # the combined feed has no meaningful bubble point (it is mostly gas).
        order = sorted(range(len(feeds)), key=lambda i: st.feed_stages[i])
        top_t = jnp.asarray(feeds[order[0]].t, dtype=float)
        bot_t = jnp.asarray(feeds[order[-1]].t, dtype=float)
        t = jnp.linspace(top_t, bot_t, n)

    def sweep(_: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
        t, x = state
        k = jax.vmap(lambda tj, pj, xj: _k_estimate(pkg, tj, pj, xj))(t, p, x)
        if st.condenser == "total":
            k = k.at[0].set(jnp.ones(c))
        # Tridiagonal component balances with v_ji = K_ji (V_j / L_j) l_ji.
        ratio = k * (big_v / big_l)[:, None]
        l_new = []
        for i in range(c):
            diag = -((1.0 + s_l) + (1.0 + s_v) * ratio[:, i])
            upper = ratio[1:, i]  # coefficient of l_{j+1}
            a = jnp.diag(diag) + jnp.diag(jnp.ones(n - 1), -1) + jnp.diag(upper, 1)
            l_new.append(jnp.linalg.solve(a, -f[:, i]))
        l_mat = jnp.clip(jnp.stack(l_new, axis=1), _FLOW_FLOOR, None)
        x = l_mat / jnp.sum(l_mat, axis=1)[:, None]
        if refine_t:
            t = _bubble_newton(pkg, t, p, x, steps=3)
        return t, x

    # Keep initialization loops compact: unrolling the nested EOS derivatives
    # duplicates large graphs and can exhaust a cold CI runner during compilation.
    t, x = jax.lax.fori_loop(0, sweeps, sweep, (t, jnp.tile(z, (n, 1))))

    k = jax.vmap(lambda tj, pj, xj: _k_estimate(pkg, tj, pj, xj))(t, p, x)
    if st.condenser == "total":
        k = k.at[0].set(jnp.ones(c))
    y = k * x
    y = y / jnp.sum(y, axis=1)[:, None]
    l_mat = jnp.clip(x * big_l[:, None], _FLOW_FLOOR, None)
    v_mat = jnp.clip(y * big_v[:, None], _FLOW_FLOOR, None)

    # Duty seeds from the top/bottom energy balances at the seeded state.
    q_c = jnp.asarray(0.0)
    q_r = jnp.asarray(0.0)
    if st.has_condenser:
        h_v1 = pkg.enthalpy(t[1], p[1], y[1], phase="vapor") if n > 1 else 0.0
        h_top = pkg.enthalpy(t[0], p[0], x[0], phase="liquid")
        q_c = -(big_v[1] * (h_v1 - h_top)) if n > 1 else jnp.asarray(0.0)
    if st.has_reboiler:
        h_l_prev = pkg.enthalpy(t[n - 2], p[n - 2], x[n - 2], phase="liquid") if n > 1 else 0.0
        h_v_bot = pkg.enthalpy(t[n - 1], p[n - 1], y[n - 1], phase="vapor")
        q_r = big_v[n - 1] * (h_v_bot - h_l_prev) if n > 1 else jnp.asarray(0.0)

    return _pack(jnp.log(l_mat), jnp.log(v_mat), t, q_c, q_r, st)


@partial(jax.jit, static_argnames=("st", "sweeps", "tol", "max_iter", "homotopy"))
def _solve(
    theta: dict[str, Any],
    feeds: list[Stream],
    hints: dict[str, Any],
    st: _Structure,
    sweeps: int,
    tol: float,
    max_iter: int,
    homotopy: bool,
) -> tuple[Array, SolveReport]:
    """Seed and converge the MESH system; compiled once per column *structure*.

    Everything that varies between calls (feeds, pressures, spec values, the
    package parameters) is a pytree argument, so a column of the same shape,
    say the same unit evaluated at another point of a recycle iteration or an
    optimisation, reuses the compiled Newton solver instead of retracing it.
    """
    pkg = theta["pkg"]
    theta_seed = jax.lax.stop_gradient(theta)
    if "liquid" in hints and "vapor" in hints:
        u0 = _pack(
            jnp.log(jnp.maximum(hints["liquid"], 1e-300)),
            jnp.log(jnp.maximum(hints["vapor"], 1e-300)),
            hints["t"],
            hints["condenser_duty"],
            hints["reboiler_duty"],
            st,
        )
    else:
        u0 = _seed(pkg, st, theta_seed, feeds, hints, sweeps)
    u0 = jax.lax.stop_gradient(u0)

    def residual(u: Array, th: dict[str, Any]) -> Array:
        return _residuals(u, th, st)

    lower = jnp.full_like(u0, -jnp.inf).at[: 2 * st.n * st.c].set(-700.0)
    upper = jnp.full_like(u0, jnp.inf).at[: 2 * st.n * st.c].set(700.0)
    lower = lower.at[2 * st.n * st.c : 2 * st.n * st.c + st.n].set(50.0 / _T_SCALE)
    upper = upper.at[2 * st.n * st.c : 2 * st.n * st.c + st.n].set(2000.0 / _T_SCALE)

    def converge(seed: Array, th: dict[str, Any]) -> SolveResult:
        return newton_system_with_info(residual, seed, th, tol, max_iter, lower=lower, upper=upper)

    if (
        homotopy
        and max_iter > 0
        and isinstance(pkg, GammaPhiPackage)
        and isinstance(pkg.activity, NRTL)
    ):
        feed_seed = jax.lax.stop_gradient(feeds)

        def softened(fraction: Array) -> dict[str, Any]:
            target = theta_seed["pkg"]
            activity = jax.tree_util.tree_map(lambda a: fraction * a, target.activity)
            model = replace(target, activity=activity)
            hf = jnp.zeros(st.n)
            for feed, stage in zip(feed_seed, st.feed_stages, strict=True):
                hf = hf.at[stage].add(feed.total * molar_enthalpy(feed, model=model))
            return {**theta_seed, "pkg": model, "hf": hf}

        def cond(state: tuple[Array, SolveResult, SolveResult]) -> Array:
            attempt, direct, _ = state
            return (attempt == 0) | ((attempt < 6) & ~direct.report.converged)

        def step(
            state: tuple[Array, SolveResult, SolveResult],
        ) -> tuple[Array, SolveResult, SolveResult]:
            attempt, direct, previous = state
            # Attempt 0 uses the original model. If it fails, restart at ideal
            # activity, then restore NRTL in four increments through attempt 5.
            th = jax.lax.cond(
                attempt == 0,
                lambda _: theta_seed,
                lambda _: softened((attempt - 1) / 4.0),
                None,
            )
            seed = jax.lax.cond(
                attempt == 1,
                lambda _: _seed(th["pkg"], st, th, feed_seed, jax.lax.stop_gradient(hints), sweeps),
                lambda _: previous.value,
                None,
            )
            solved = converge(seed, th)
            solved = solved._replace(
                report=solved.report._replace(
                    iterations=previous.report.iterations + solved.report.iterations
                )
            )
            direct = jax.tree_util.tree_map(
                lambda new, old: jnp.where(attempt == 0, new, old), solved, direct
            )
            return attempt + 1, direct, solved

        initial = SolveResult(
            u0,
            SolveReport(
                jnp.asarray(SolveStatus.MAX_ITERATIONS),
                jnp.asarray(0),
                jnp.asarray(jnp.inf),
                jnp.asarray(0.0),
                jnp.asarray(0),
            ),
        )
        # Share one Newton graph between the direct solve and recovery. A
        # data-dependent loop also prevents XLA from unrolling five copies of
        # the nested saturation solvers during cold NRTL compilation.
        _, direct, recovered = jax.lax.while_loop(cond, step, (jnp.asarray(0), initial, initial))
        # Recovery always ends at the original model, so only endpoint reports
        # compete with the direct result, never intermediate activity models.
        better = recovered.report.converged | (
            recovered.report.residual_norm < direct.report.residual_norm
        )
        result = jax.tree_util.tree_map(
            lambda good, original: jnp.where(better, good, original), recovered, direct
        )
    else:
        result = converge(u0, theta_seed)
    report = jax.lax.stop_gradient(result.report)
    value = implicit_solution(
        residual, jax.lax.stop_gradient(result.value), theta, report.converged
    )
    return value, report


# --------------------------------------------------------------------------- #
# Public solver
# --------------------------------------------------------------------------- #


def rigorous_column(
    feeds: Sequence[ColumnFeed],
    n_stages: int,
    *,
    p: ArrayLike | None = None,
    p_top: ArrayLike | None = None,
    p_bottom: ArrayLike | None = None,
    condenser: str | None = "total",
    reboiler: str | None = "kettle",
    specs: Sequence[ColumnSpec] = (),
    side_draws: Sequence[SideDraw] = (),
    stage_duties: Sequence[StageDuty] = (),
    efficiency: ArrayLike = 1.0,
    reflux_temperature: ArrayLike | None = None,
    model: Model = None,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    guess: dict[str, Any] | None = None,
    sweeps: int = 4,
    tol: float = 1e-9,
    max_iter: int = 80,
    check: bool = True,
    homotopy: bool = True,
) -> RigorousColumnResult:
    """Solve a rigorous multistage column by simultaneous correction (MESH).

    Args:
        feeds: One or more `ColumnFeed` entries (stream + stage).
        n_stages: Total number of equilibrium stages *including* the condenser
            and reboiler when present (stage 1 is the top).
        p: Uniform column pressure (Pa); or give ``p_top`` and ``p_bottom`` for a
            linear profile.
        p_top: Top-stage pressure (Pa) when a profile is wanted.
        p_bottom: Bottom-stage pressure (Pa) when a profile is wanted.
        condenser: ``"total"`` (liquid distillate, saturated or subcooled
            reflux), ``"partial"`` (vapour distillate from an equilibrium
            stage), or ``None`` (no condenser; the top stage is an ordinary
            stage, as in an absorber).
        reboiler: ``"kettle"`` (an equilibrium stage with a heat duty) or ``None``.
        specs: Column specifications; exactly one per condenser/reboiler present
            (two for a conventional column, one for a reboiled absorber, none
            for an absorber or stripper).
        side_draws: Side products as fractions of the phase leaving a stage.
        stage_duties: Intermediate heat duties (W).
        efficiency: Murphree vapour efficiency, a scalar or one value per stage.
        reflux_temperature: Subcooled reflux temperature (K) for a total
            condenser; ``None`` returns saturated reflux.
        model: Property package (see `fugacio.sim.models.package_for`); defaults
            to Peng-Robinson.
        eos: Cubic EOS for the default package.
        kij: Binary interaction matrix for the default package.
        guess: Optional seeding hints: ``"reflux_ratio"``, ``"distillate_rate"``
            (used for the internal-traffic seed) and ``"t"`` (a stage temperature
            profile). Specs of those kinds are used automatically.
        sweeps: Bubble-point sweeps used to seed the Newton solve.
        check: Raise for a failed concrete solve; compiled failures return NaNs.
        tol: Maximum accepted scaled equation residual.
        max_iter: Iteration cap for each Newton solve, including homotopy increments.
        homotopy: Retry a failed NRTL solve by gradually restoring activity effects
            from an ideal-activity solution. The final report checks the original model.

    Returns:
        A `RigorousColumnResult` with the products, duties, and stage profiles, all
        differentiable with respect to the feeds, pressures, specifications,
        draw fractions, duties, and the package parameters.

    Raises:
        ValueError: for an inconsistent description (bad stage numbers, wrong
            number of specs, unknown condenser/reboiler type, mismatched
            components).
    """
    if not feeds:
        raise ValueError("a column needs at least one feed")
    if n_stages < 1:
        raise ValueError("n_stages must be at least 1")
    if condenser not in (None, "total", "partial"):
        raise ValueError(f"unknown condenser type {condenser!r}; use 'total', 'partial', or None")
    if reboiler not in (None, "kettle"):
        raise ValueError(f"unknown reboiler type {reboiler!r}; use 'kettle' or None")
    components = feeds[0].stream.components
    for fd in feeds:
        if fd.stream.components != components:
            raise ValueError("all column feeds must share one component list")
    n_specs_needed = int(condenser is not None) + int(reboiler is not None)
    if len(specs) != n_specs_needed:
        raise ValueError(
            f"this column has {n_specs_needed} degree(s) of freedom "
            f"(condenser: {condenser}, reboiler: {reboiler}) but {len(specs)} spec(s) were given"
        )
    for sd in side_draws:
        if sd.phase not in ("liquid", "vapor"):
            raise ValueError(f"side draw phase must be 'liquid' or 'vapor', got {sd.phase!r}")
    if reflux_temperature is not None and condenser != "total":
        raise ValueError("reflux_temperature applies to a total condenser only")

    pkg = resolve_package(components, model, eos=eos, kij=kij)
    n, c = n_stages, len(components)
    st = _Structure(
        n=n,
        c=c,
        components=components,
        condenser=condenser,
        reboiler=reboiler,
        feed_stages=tuple(_stage_index(fd.stage, n, "feed") for fd in feeds),
        draw_stages=tuple(_stage_index(sd.stage, n, "side draw") for sd in side_draws),
        draw_phases=tuple(sd.phase for sd in side_draws),
        duty_stages=tuple(_stage_index(sq.stage, n, "stage duty") for sq in stage_duties),
        spec_kinds=tuple(sp.kind for sp in specs),
        spec_components=tuple(sp.component for sp in specs),
        spec_products=tuple(sp.product for sp in specs),
        spec_stages=tuple(
            None if sp.stage is None else _stage_index(sp.stage, n, "spec") for sp in specs
        ),
        subcooled=reflux_temperature is not None,
    )

    # Pressure profile.
    if p is not None:
        p_prof = jnp.full(n, jnp.asarray(p, dtype=float))
    elif p_top is not None and p_bottom is not None:
        p_prof = jnp.linspace(
            jnp.asarray(p_top, dtype=float), jnp.asarray(p_bottom, dtype=float), n
        )
    else:
        raise ValueError("give either p (uniform) or both p_top and p_bottom")

    # Differentiable parameter pytree.
    f = jnp.zeros((n, c))
    hf = jnp.zeros(n)
    feed_streams = [fd.stream for fd in feeds]
    for stream, j in zip(feed_streams, st.feed_stages, strict=True):
        f = f.at[j].add(stream.n)
        hf = hf.at[j].add(stream.total * molar_enthalpy(stream, model=pkg))
    # A component absent from every feed has no finite log-flow; a trace of it
    # (far below any tolerance) keeps the unknowns finite without perturbing the
    # balances of the components that are present.
    absent = jnp.sum(f, axis=0) <= _FLOW_FLOOR
    f = f.at[st.feed_stages[0]].add(jnp.where(absent, _FLOW_FLOOR, 0.0))
    s_l = jnp.zeros(n)
    s_v = jnp.zeros(n)
    for sd, j in zip(side_draws, st.draw_stages, strict=True):
        frac = jnp.asarray(sd.fraction, dtype=float)
        if sd.phase == "liquid":
            s_l = s_l.at[j].add(frac)
        else:
            s_v = s_v.at[j].add(frac)
    q = jnp.zeros(n)
    for sq, j in zip(stage_duties, st.duty_stages, strict=True):
        q = q.at[j].add(jnp.asarray(sq.duty, dtype=float))
    eta = jnp.broadcast_to(jnp.asarray(efficiency, dtype=float), (n,))
    theta: dict[str, Any] = {
        "pkg": pkg,
        "p": p_prof,
        "f": f,
        "hf": hf,
        "f_total": jnp.sum(f),
        "s_l": s_l,
        "s_v": s_v,
        "q": q,
        "eta": eta,
        "specs": [jnp.asarray(sp.value, dtype=float) for sp in specs],
        "t_reflux": (
            jnp.asarray(0.0)
            if reflux_temperature is None
            else jnp.asarray(reflux_temperature, dtype=float)
        ),
    }

    # Seed (detached: the iteration's starting point carries no gradient).
    hints = dict(guess or {})
    for sp in specs:
        if sp.kind in ("reflux_ratio", "distillate_rate") and sp.kind not in hints:
            hints[sp.kind] = sp.value
        if sp.kind == "bottoms_rate" and "distillate_rate" not in hints:
            hints["distillate_rate"] = theta["f_total"] - jnp.asarray(sp.value, dtype=float)
    # Canonical dtypes so repeated calls with the same structure hit the jit cache
    # whether the caller passed Python floats or arrays.
    canon = [
        Stream(
            n=jnp.asarray(s.n, dtype=float),
            t=jnp.asarray(s.t, dtype=float),
            p=jnp.asarray(s.p, dtype=float),
            components=s.components,
            vapor_n=jnp.asarray(s.vapor_n, dtype=float),
        )
        for s in feed_streams
    ]
    hints = {k: jnp.asarray(v, dtype=float) for k, v in hints.items()}
    u_star, solve_report = _solve(theta, canon, hints, st, sweeps, tol, max_iter, homotopy)
    if check:
        labels = tuple(
            [f"stage {stage + 1}: material {comp}" for stage in range(n) for comp in components]
            + [
                f"stage {stage + 1}: equilibrium {comp}"
                for stage in range(n)
                for comp in components
            ]
            + [f"stage {stage + 1}: energy" for stage in range(n)]
            + [f"specification: {sp.kind}" for sp in specs]
        )
        require_converged(solve_report, "rigorous column", labels)
        u_star = jnp.where(solve_report.converged, u_star, jnp.nan)

    # Assemble the result.
    un = _unpack(u_star, st)
    liq = jnp.exp(un.ln_l)
    v = jnp.exp(un.ln_v)
    big_l = jnp.sum(liq, axis=1)
    big_v = jnp.sum(v, axis=1)
    x = liq / big_l[:, None]
    y = v / big_v[:, None]
    k = jax.vmap(lambda tj, pj, xj, yj: pkg.k_values(tj, pj, xj, yj))(un.t, p_prof, x, y)
    if condenser == "total" and not st.subcooled:
        k = k.at[0].set(_incipient_vapor_k(pkg, un.t[0], p_prof[0], x[0]))
    distillate = Stream(
        n=v[0],
        t=un.t[0],
        p=p_prof[0],
        components=components,
        vapor_n=jnp.zeros_like(v[0]) if condenser == "total" else v[0],
    )
    bottoms = Stream(
        n=liq[n - 1],
        t=un.t[n - 1],
        p=p_prof[n - 1],
        components=components,
        vapor_n=jnp.zeros_like(liq[n - 1]),
    )
    draws = []
    for kdx, j in enumerate(st.draw_stages):
        flows = _product_flows(f"draw:{kdx}", liq, v, s_l, s_v, st)
        draws.append(
            Stream(
                n=flows,
                t=un.t[j],
                p=p_prof[j],
                components=components,
                vapor_n=flows if side_draws[kdx].phase == "vapor" else jnp.zeros_like(flows),
            )
        )
    return RigorousColumnResult(
        distillate=distillate,
        bottoms=bottoms,
        side_draws=tuple(draws),
        condenser_duty=un.q_c,
        reboiler_duty=un.q_r,
        t=un.t,
        p=p_prof,
        x=x,
        y=y,
        k=k,
        liquid_flow=(1.0 + s_l) * big_l,
        vapor_flow=(1.0 + s_v) * big_v,
        reflux_ratio=big_l[0] / big_v[0],
        boilup_ratio=big_v[n - 1] / big_l[n - 1],
        residual_norm=solve_report.residual_norm,
        report=solve_report,
    )


def absorber(
    gas: Stream,
    liquid: Stream,
    n_stages: int,
    *,
    p: ArrayLike | None = None,
    p_top: ArrayLike | None = None,
    p_bottom: ArrayLike | None = None,
    efficiency: ArrayLike = 1.0,
    model: Model = None,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    **kwargs: Any,
) -> RigorousColumnResult:
    """Countercurrent absorber: lean solvent to the top stage, rich gas to the bottom.

    No condenser, no reboiler, no specifications: the column is fully determined
    by its two feeds. ``distillate`` in the result is the treated gas leaving the
    top and ``bottoms`` the rich solvent leaving the bottom.
    """
    return rigorous_column(
        [ColumnFeed(liquid, 1), ColumnFeed(gas, n_stages)],
        n_stages,
        p=p,
        p_top=p_top,
        p_bottom=p_bottom,
        condenser=None,
        reboiler=None,
        efficiency=efficiency,
        model=model,
        eos=eos,
        kij=kij,
        **kwargs,
    )


def stripper(
    liquid: Stream,
    gas: Stream,
    n_stages: int,
    *,
    p: ArrayLike | None = None,
    p_top: ArrayLike | None = None,
    p_bottom: ArrayLike | None = None,
    efficiency: ArrayLike = 1.0,
    model: Model = None,
    eos: CubicEOS = PR,
    kij: Array | None = None,
    **kwargs: Any,
) -> RigorousColumnResult:
    """Countercurrent stripper: rich liquid to the top stage, stripping gas to the bottom.

    The same topology as `absorber` with the roles of the feeds swapped in the
    signature; ``distillate`` is the loaded stripping gas and ``bottoms`` the
    stripped liquid.
    """
    return rigorous_column(
        [ColumnFeed(liquid, 1), ColumnFeed(gas, n_stages)],
        n_stages,
        p=p,
        p_top=p_top,
        p_bottom=p_bottom,
        condenser=None,
        reboiler=None,
        efficiency=efficiency,
        model=model,
        eos=eos,
        kij=kij,
        **kwargs,
    )


__all__ = [
    "ColumnFeed",
    "ColumnSpec",
    "RigorousColumnResult",
    "SideDraw",
    "StageDuty",
    "absorber",
    "boilup_ratio",
    "bottoms_rate",
    "component_flow",
    "condenser_duty",
    "distillate_rate",
    "purity",
    "reboiler_duty",
    "recovery",
    "reflux_rate",
    "reflux_ratio",
    "rigorous_column",
    "stage_temperature",
    "stripper",
]
