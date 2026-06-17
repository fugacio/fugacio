"""Differentiable reactor unit operations with material *and* energy balances.

These blocks turn the reaction thermochemistry and kinetics in
`fugacio.thermo` into flowsheet units that consume and produce
`Stream` objects, just like the separation units in
`fugacio.sim.units`. Two families are provided:

* *Equilibrium* / *stoichiometric* reactors: the conversion is set by chemical
  equilibrium (`fugacio.thermo.reaction_equilibrium.equilibrium`) or by a
  specified extent/conversion; no rate law is needed.
* *Kinetic* reactors: `cstr`, `pfr`, and `batch_reactor`
  integrate the rate laws of `fugacio.thermo.kinetics` over reactor volume
  (CSTR/PFR) or time (batch).

Every reactor supports an **energy balance**: run it isothermally at a specified
temperature and the heat *duty* required to hold that temperature is returned
(it carries the heat of reaction), or run it ``adiabatic=True`` and the outlet
temperature is solved from an adiabatic enthalpy balance. The enthalpy
bookkeeping is the ideal-gas absolute enthalpy ``Hf_i(298) + integral Cp_i dT``
that underlies `fugacio.thermo.reactions.delta_h_rxn`, so reaction heat and
sensible heat are accounted for consistently. Kinetic-reactor concentrations use
the ideal-gas relation ``c_i = y_i P / (R T)``.

Because the underlying solves (equilibrium root-finds, the CSTR Newton system,
the explicit RK4 marches) are differentiable, a reactor's conversion, outlet
temperature, and duty are differentiable in the feed, the operating conditions,
*and* the reaction/kinetic parameters, ready for gradient-based design.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.stream import Stream
from fugacio.thermo import component_arrays
from fugacio.thermo.constants import P_REF, R
from fugacio.thermo.eos import PR, CubicEOS
from fugacio.thermo.ideal import cp_ig, enthalpy_ig
from fugacio.thermo.implicit import bracketed_root, newton_system
from fugacio.thermo.reaction_equilibrium import equilibrium
from fugacio.thermo.reactions import (
    CpCoeffs,
    Reaction,
    delta_g_rxn,
    delta_h_rxn,
    reaction_arrays,
)

ArrayLike = Array | float

# Scale (J) used to non-dimensionalise the adiabatic energy residual so it sits
# on the same O(1) footing as the dimensionless mole / equilibrium residuals.
_H_SCALE = 1.0e4


class ReactorResult(NamedTuple):
    """Outcome of a reactor calculation.

    Attributes:
        outlet: Product `Stream` (at the reactor outlet
            temperature, solved for adiabatic operation).
        duty: Heat duty (W) to hold an isothermal reactor at temperature; positive
            means heat *added*. Zero for an adiabatic reactor.
        extent: Extent of each reaction (mol/s for flow reactors, mol for batch),
            shape ``(n_reactions,)``.
    """

    outlet: Stream
    duty: Array
    extent: Array


def _as_reactions(reactions: Reaction | Sequence[Reaction]) -> list[Reaction]:
    return [reactions] if isinstance(reactions, Reaction) else list(reactions)


def _as_rate_laws(rate_laws: Any, n_reactions: int) -> list[Any]:
    laws = list(rate_laws) if isinstance(rate_laws, (list, tuple)) else [rate_laws]
    if len(laws) != n_reactions:
        raise ValueError(f"expected {n_reactions} rate law(s), got {len(laws)}")
    return laws


def _stack_nu(reactions: Sequence[Reaction], components: tuple[str, ...]) -> Array:
    """Stack reaction stoichiometries into an ``(n_reactions, n_components)`` matrix."""
    rows = []
    for r in reactions:
        if tuple(r.components) != tuple(components):
            raise ValueError(
                "each reaction must be defined over the reactor feed's components in the same order"
            )
        rows.append(jnp.asarray(r.nu))
    return jnp.stack(rows)


def _ig_data(components: tuple[str, ...]) -> tuple[Array, Array, CpCoeffs]:
    """Ideal-gas formation enthalpy/Gibbs and ``Cp`` coefficients for the components."""
    return reaction_arrays(list(components))


def _h_total(n: Array, t: ArrayLike, hf: Array, coeffs: CpCoeffs) -> Array:
    """Absolute ideal-gas enthalpy flow ``sum_i n_i (Hf_i + integral Cp_i dT)`` (W)."""
    a, b, c, d, e = coeffs
    return jnp.sum(n * (hf + enthalpy_ig(t, a, b, c, d, e)))


def _cp_species(t: ArrayLike, coeffs: CpCoeffs) -> Array:
    a, b, c, d, e = coeffs
    return cp_ig(t, a, b, c, d, e)


def _dh_rxn_rows(nu: Array, t: ArrayLike, hf: Array, coeffs: CpCoeffs) -> Array:
    a, b, c, d, e = coeffs
    return jnp.stack([delta_h_rxn(nu[j], t, hf, a, b, c, d, e) for j in range(nu.shape[0])])


def _ln_k_rows(nu: Array, t: ArrayLike, hf: Array, gf: Array, coeffs: CpCoeffs) -> Array:
    a, b, c, d, e = coeffs
    t = jnp.asarray(t)
    return jnp.stack(
        [-delta_g_rxn(nu[j], t, hf, gf, a, b, c, d, e) / (R * t) for j in range(nu.shape[0])]
    )


def _ideal_concentration(n: Array, t: ArrayLike, p: ArrayLike) -> Array:
    """Ideal-gas concentration ``c_i = y_i P / (R T)`` (mol/m^3)."""
    y = n / jnp.sum(n)
    return y * jnp.asarray(p) / (R * jnp.asarray(t))


def _rate_vector(rate_laws: list[Any], t: ArrayLike, c: Array) -> Array:
    return jnp.stack([law.rate(t, c) for law in rate_laws])


def _extent_from_moles(nu: Array, dn: Array) -> Array:
    """Least-squares extents reproducing a mole change ``dn = extent @ nu``."""
    extent, *_ = jnp.linalg.lstsq(nu.T, dn, rcond=None)
    return extent


def _solve_adiabatic_t(
    n_out: Array, h_target: Array, hf: Array, coeffs: CpCoeffs, *, t_lo: float, t_hi: float
) -> Array:
    """Outlet temperature where the product enthalpy equals ``h_target`` (adiabatic)."""

    def residual(t: Array, theta: tuple[Array, Array]) -> Array:
        n, h = theta
        return _h_total(n, t, hf, coeffs) - h

    return bracketed_root(residual, (n_out, h_target), jnp.asarray(t_lo), jnp.asarray(t_hi), 1e-8)


def equilibrium_reactor(
    feed: Stream,
    reactions: Reaction | Sequence[Reaction],
    *,
    t_out: ArrayLike | None = None,
    adiabatic: bool = False,
    basis: str = "ideal-gas",
    eos: CubicEOS = PR,
    kij: Array | None = None,
    tol: float = 1e-10,
    max_iter: int = 80,
) -> ReactorResult:
    """Reactor whose outlet is the chemical-equilibrium composition.

    Isothermal (default, or with ``t_out``): the equilibrium composition at the
    operating temperature is found and the duty to hold that temperature returned.
    Adiabatic (``adiabatic=True``): the extents and outlet temperature are solved
    *together* from the equilibrium conditions plus an adiabatic energy balance
    (ideal-gas basis).

    Args:
        feed: Inlet stream; reactions must be defined over ``feed.components``.
        reactions: One reaction or several sharing the feed's component ordering.
        t_out: Isothermal operating temperature (K); defaults to ``feed.t``.
        adiabatic: Solve the outlet temperature from an adiabatic balance instead.
        basis: ``"ideal-gas"`` or ``"phi"`` (EOS fugacity coefficients) for the
            isothermal equilibrium; adiabatic operation uses the ideal-gas basis.
        eos: Cubic EOS used when ``basis="phi"``.
        kij: Optional binary interaction matrix for the EOS.
        tol: Convergence tolerance on the reaction extents.
        max_iter: Maximum number of solver iterations.

    Returns:
        A `ReactorResult`.
    """
    comps = feed.components
    rxns = _as_reactions(reactions)
    nu = _stack_nu(rxns, comps)
    hf, gf, coeffs = _ig_data(comps)
    p = jnp.asarray(feed.p)
    n_feed = feed.n
    t_in = jnp.asarray(feed.t)
    h_in = _h_total(n_feed, t_in, hf, coeffs)

    if not adiabatic:
        t = jnp.asarray(feed.t if t_out is None else t_out)
        tc = pc = omega = None
        if basis == "phi":
            arr = component_arrays(list(comps))
            tc, pc, omega = arr["tc"], arr["pc"], arr["omega"]
        res = equilibrium(
            rxns,
            n_feed,
            t,
            p,
            basis=basis,
            eos=eos,
            tc=tc,
            pc=pc,
            omega=omega,
            kij=kij,
            tol=tol,
            max_iter=max_iter,
        )
        duty = _h_total(res.moles, t, hf, coeffs) - h_in
        outlet = Stream(n=res.moles, t=t, p=p, components=comps)
        return ReactorResult(outlet=outlet, duty=duty, extent=res.extent)

    n_rxn = nu.shape[0]
    reactant = nu < 0.0
    cap = jnp.min(
        jnp.where(reactant, n_feed[None, :] / jnp.where(reactant, -nu, 1.0), jnp.inf), axis=1
    )

    def residual(z: Array, theta: tuple[Array, Array, Array]) -> Array:
        nf, hin, pres = theta
        xi = z[:n_rxn]
        t = z[n_rxn]
        n = nf + xi @ nu
        y = n / jnp.sum(n)
        ln_a = jnp.log(jnp.clip(y, 1e-300, None)) + jnp.log(pres / P_REF)
        eq = nu @ ln_a - _ln_k_rows(nu, t, hf, gf, coeffs)
        energy = (_h_total(n, t, hf, coeffs) - hin) / _H_SCALE
        return jnp.concatenate([eq, jnp.reshape(energy, (1,))])

    z0 = jnp.concatenate([0.1 * cap, jnp.reshape(t_in, (1,))])
    z = newton_system(residual, z0, (n_feed, h_in, p), tol, max_iter)
    xi = z[:n_rxn]
    t_solved = z[n_rxn]
    n_out = n_feed + xi @ nu
    outlet = Stream(n=n_out, t=t_solved, p=p, components=comps)
    return ReactorResult(outlet=outlet, duty=jnp.asarray(0.0), extent=xi)


def stoichiometric_reactor(
    feed: Stream,
    reactions: Reaction | Sequence[Reaction],
    *,
    extent: ArrayLike | None = None,
    conversion: ArrayLike | None = None,
    t_out: ArrayLike | None = None,
    adiabatic: bool = False,
    t_lo: float = 200.0,
    t_hi: float = 6000.0,
) -> ReactorResult:
    """Reactor with a *specified* extent or key-reactant conversion (no equilibrium).

    Provide exactly one of ``extent`` (per reaction, mol/s) or ``conversion`` (a
    single-reaction fractional conversion of its limiting reactant). The outlet is
    ``n = n_feed + extent @ nu``; the energy balance is the same isothermal-duty /
    adiabatic-temperature treatment as `equilibrium_reactor`.

    Raises:
        ValueError: if not exactly one of ``extent`` / ``conversion`` is given, or
            ``conversion`` is used with more than one reaction.
    """
    if (extent is None) == (conversion is None):
        raise ValueError("provide exactly one of 'extent' or 'conversion'")
    comps = feed.components
    rxns = _as_reactions(reactions)
    nu = _stack_nu(rxns, comps)
    hf, _gf, coeffs = _ig_data(comps)
    n_feed = feed.n
    t_in = jnp.asarray(feed.t)
    p = jnp.asarray(feed.p)

    if conversion is not None:
        if nu.shape[0] != 1:
            raise ValueError("'conversion' is only defined for a single reaction")
        nu_row = nu[0]
        reactant = nu_row < 0.0
        extent_max = jnp.min(
            jnp.where(reactant, n_feed / jnp.where(reactant, -nu_row, 1.0), jnp.inf)
        )
        extent_arr = jnp.reshape(jnp.asarray(conversion) * extent_max, (1,))
    else:
        extent_arr = jnp.atleast_1d(jnp.asarray(extent, dtype=float))

    n_out = n_feed + extent_arr @ nu
    h_in = _h_total(n_feed, t_in, hf, coeffs)
    if adiabatic:
        t_solved = _solve_adiabatic_t(n_out, h_in, hf, coeffs, t_lo=t_lo, t_hi=t_hi)
        outlet = Stream(n=n_out, t=t_solved, p=p, components=comps)
        return ReactorResult(outlet=outlet, duty=jnp.asarray(0.0), extent=extent_arr)
    t = jnp.asarray(feed.t if t_out is None else t_out)
    duty = _h_total(n_out, t, hf, coeffs) - h_in
    outlet = Stream(n=n_out, t=t, p=p, components=comps)
    return ReactorResult(outlet=outlet, duty=duty, extent=extent_arr)


def cstr(
    feed: Stream,
    reactions: Reaction | Sequence[Reaction],
    rate_laws: Any,
    volume: ArrayLike,
    *,
    t_out: ArrayLike | None = None,
    adiabatic: bool = False,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> ReactorResult:
    """Continuous stirred-tank reactor (perfectly mixed) at steady state.

    Solves the steady-state mole balance ``F_out = F_in + V (r . Nu)`` with the
    outlet-condition rates ``r`` (one per reaction, from ``rate_laws``) and
    ideal-gas concentrations. Isothermal by default (duty returned); with
    ``adiabatic=True`` the outlet temperature is solved jointly with the flows.

    Args:
        feed: Inlet stream (``feed.n`` are molar flows, mol/s).
        reactions: Reaction(s) over ``feed.components``.
        rate_laws: One rate law per reaction (a kinetics object with ``rate(T, c)``).
        volume: Reactor volume (m^3).
        t_out: Isothermal temperature (K); defaults to ``feed.t``.
        adiabatic: Solve the outlet temperature from the energy balance.
        tol: Convergence tolerance on the steady-state mole balance.
        max_iter: Maximum number of Newton iterations.
    """
    comps = feed.components
    rxns = _as_reactions(reactions)
    nu = _stack_nu(rxns, comps)
    laws = _as_rate_laws(rate_laws, nu.shape[0])
    hf, _gf, coeffs = _ig_data(comps)
    f_in = feed.n
    t_in = jnp.asarray(feed.t)
    p = jnp.asarray(feed.p)
    vol = jnp.asarray(volume)
    h_in = _h_total(f_in, t_in, hf, coeffs)
    n = len(comps)

    if not adiabatic:
        t = jnp.asarray(feed.t if t_out is None else t_out)

        # Every differentiable input (feed, volume, T, P, kinetic parameters) is
        # threaded through ``theta`` so the implicit-diff solver can propagate
        # gradients; closing over them would leak tracers out of the solve.
        def residual(f: Array, theta: tuple[Array, Array, Array, Array, Any]) -> Array:
            f0, v, temp, pres, klaws = theta
            c = _ideal_concentration(f, temp, pres)
            r = _rate_vector(klaws, temp, c)
            return f - f0 - v * (r @ nu)

        f_out = newton_system(residual, f_in, (f_in, vol, t, p, laws), tol, max_iter)
        extent = _extent_from_moles(nu, f_out - f_in)
        duty = _h_total(f_out, t, hf, coeffs) - h_in
        return ReactorResult(Stream(f_out, t, p, comps), duty, extent)

    def residual_ad(z: Array, theta: tuple[Array, Array, Array, Array, Any]) -> Array:
        f0, hin, v, pres, klaws = theta
        f = z[:n]
        t = z[n]
        c = _ideal_concentration(f, t, pres)
        r = _rate_vector(klaws, t, c)
        mole = f - f0 - v * (r @ nu)
        energy = (_h_total(f, t, hf, coeffs) - hin) / _H_SCALE
        return jnp.concatenate([mole, jnp.reshape(energy, (1,))])

    z0 = jnp.concatenate([f_in, jnp.reshape(t_in, (1,))])
    z = newton_system(residual_ad, z0, (f_in, h_in, vol, p, laws), tol, max_iter)
    f_out = z[:n]
    t_solved = z[n]
    extent = _extent_from_moles(nu, f_out - f_in)
    return ReactorResult(Stream(f_out, t_solved, p, comps), jnp.asarray(0.0), extent)


def _march(
    state0: tuple[Array, Array],
    deriv: Any,
    step_size: Array,
    steps: int,
) -> tuple[Array, Array]:
    """Explicit RK4 march of ``(flows/moles, T)`` over ``steps`` of ``step_size``."""

    def rk4(state: tuple[Array, Array], _: None) -> tuple[tuple[Array, Array], None]:
        x, t = state
        dx1, dt1 = deriv(x, t)
        dx2, dt2 = deriv(x + 0.5 * step_size * dx1, t + 0.5 * step_size * dt1)
        dx3, dt3 = deriv(x + 0.5 * step_size * dx2, t + 0.5 * step_size * dt2)
        dx4, dt4 = deriv(x + step_size * dx3, t + step_size * dt3)
        x = x + (step_size / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
        t = t + (step_size / 6.0) * (dt1 + 2.0 * dt2 + 2.0 * dt3 + dt4)
        return (jnp.clip(x, 0.0, None), t), None

    (x_final, t_final), _ = jax.lax.scan(rk4, state0, None, length=steps)
    return x_final, t_final


def pfr(
    feed: Stream,
    reactions: Reaction | Sequence[Reaction],
    rate_laws: Any,
    volume: ArrayLike,
    *,
    t_out: ArrayLike | None = None,
    adiabatic: bool = False,
    steps: int = 200,
) -> ReactorResult:
    """Plug-flow reactor: integrate the species balances along the reactor volume.

    Marches ``dF_i/dV = (r . Nu)_i`` (ideal-gas concentrations, isobaric) from the
    feed to ``volume`` with explicit RK4. Isothermal by default; with
    ``adiabatic=True`` the temperature is integrated alongside via
    ``dT/dV = -(sum_j r_j DH_rxn,j) / (sum_i F_i Cp_i)``.
    """
    comps = feed.components
    rxns = _as_reactions(reactions)
    nu = _stack_nu(rxns, comps)
    laws = _as_rate_laws(rate_laws, nu.shape[0])
    hf, _gf, coeffs = _ig_data(comps)
    f_in = feed.n
    t_in = jnp.asarray(feed.t)
    p = jnp.asarray(feed.p)
    t_fixed = jnp.asarray(feed.t if t_out is None else t_out)
    dv = jnp.asarray(volume) / steps

    def deriv(f: Array, t: Array) -> tuple[Array, Array]:
        c = _ideal_concentration(f, t, p)
        r = _rate_vector(laws, t, c)
        df = r @ nu
        if adiabatic:
            dh = _dh_rxn_rows(nu, t, hf, coeffs)
            dt = -jnp.sum(r * dh) / jnp.sum(f * _cp_species(t, coeffs))
        else:
            dt = jnp.asarray(0.0)
        return df, dt

    f_out, t_final = _march((f_in, t_fixed), deriv, dv, steps)
    extent = _extent_from_moles(nu, f_out - f_in)
    if adiabatic:
        return ReactorResult(Stream(f_out, t_final, p, comps), jnp.asarray(0.0), extent)
    duty = _h_total(f_out, t_fixed, hf, coeffs) - _h_total(f_in, t_in, hf, coeffs)
    return ReactorResult(Stream(f_out, t_fixed, p, comps), duty, extent)


def batch_reactor(
    feed: Stream,
    reactions: Reaction | Sequence[Reaction],
    rate_laws: Any,
    volume: ArrayLike,
    time: ArrayLike,
    *,
    t_out: ArrayLike | None = None,
    adiabatic: bool = False,
    steps: int = 200,
) -> ReactorResult:
    """Constant-volume batch reactor: integrate the mole balances over time.

    Here ``feed.n`` are the *initial moles* (mol). Marches
    ``dN_i/dt = V (r . Nu)_i`` with concentrations ``c_i = N_i / V`` by explicit
    RK4. Isothermal by default; with ``adiabatic=True`` the temperature is
    integrated via ``(sum_i N_i Cp_i) dT/dt = -V sum_j r_j DH_rxn,j``. The returned
    ``extent`` is in moles and ``duty`` (isothermal) is the cumulative heat (J).
    """
    comps = feed.components
    rxns = _as_reactions(reactions)
    nu = _stack_nu(rxns, comps)
    laws = _as_rate_laws(rate_laws, nu.shape[0])
    hf, _gf, coeffs = _ig_data(comps)
    n0 = feed.n
    t_in = jnp.asarray(feed.t)
    p = jnp.asarray(feed.p)
    vol = jnp.asarray(volume)
    t_fixed = jnp.asarray(feed.t if t_out is None else t_out)
    dt_step = jnp.asarray(time) / steps

    def deriv(n: Array, t: Array) -> tuple[Array, Array]:
        c = jnp.clip(n, 0.0, None) / vol
        r = _rate_vector(laws, t, c)
        dn = vol * (r @ nu)
        if adiabatic:
            dh = _dh_rxn_rows(nu, t, hf, coeffs)
            dt = -vol * jnp.sum(r * dh) / jnp.sum(n * _cp_species(t, coeffs))
        else:
            dt = jnp.asarray(0.0)
        return dn, dt

    n_out, t_final = _march((n0, t_fixed), deriv, dt_step, steps)
    extent = _extent_from_moles(nu, n_out - n0)
    if adiabatic:
        return ReactorResult(Stream(n_out, t_final, p, comps), jnp.asarray(0.0), extent)
    duty = _h_total(n_out, t_fixed, hf, coeffs) - _h_total(n0, t_in, hf, coeffs)
    return ReactorResult(Stream(n_out, t_fixed, p, comps), duty, extent)


def conversion(feed: Stream, outlet: Stream, component_index: int) -> Array:
    """Fractional conversion of a feed component, ``(n_in - n_out) / n_in``."""
    n0 = feed.n[component_index]
    return (n0 - outlet.n[component_index]) / n0


__all__ = [
    "ReactorResult",
    "batch_reactor",
    "conversion",
    "cstr",
    "equilibrium_reactor",
    "pfr",
    "stoichiometric_reactor",
]
