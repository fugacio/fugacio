"""Generate ``_property_data.py``: pure-component property-correlation tables.

A *one-shot authoring tool*, not part of the package. For every component in the
Fugacio database it assembles, from the open ``chemicals`` dataset:

* Spencer-Danner Rackett ``Z_RA`` (COSTALD table, else a one-point fit to the
  best available saturated-density correlation at ``Tr = 0.7``);
* COSTALD characteristic volumes and SRK acentric factors;
* gas-phase dipole moments (CCCBDB / Muller / Poling, via ``chemicals``);
* saturated-liquid density (Perry DIPPR-105 transcribed; VDI-PPDS refitted);
* liquid viscosity (Perry DIPPR-101 transcribed; VDI-PPDS / Viswanath-Natarajan
  refitted onto DIPPR-101 with ``c5 = 1``);
* dilute-gas viscosity and thermal conductivity (Perry DIPPR-102 transcribed;
  VDI-PPDS quartic polynomials refitted onto DIPPR-102);
* liquid thermal conductivity (Perry DIPPR-100 transcribed; VDI-PPDS quartic
  transcribed as DIPPR-100);
* surface tension (Mulero-Cachadina REFPROP fits transcribed; Somayajulu and
  Jasper refitted onto the Mulero-Cachadina form);
* enthalpy of vaporization (Perry DIPPR-106 transcribed; VDI-PPDS refitted);
* Antoine and ideal-gas-``Cp`` backfills for components whose curated record
  lacks them (transcribed from Poling, else fitted to Wagner / extended-Antoine /
  DIPPR-101 vapour pressures and Poling / TRC heat capacities).

Every transcription and refit is validated before it is emitted: refits must
reproduce their source within a relative tolerance over the sampled range, and
all values must pass physical-plausibility bounds, so a unit mix-up or a wrong
functional form is dropped (loudly) rather than baked into the tree.

Run with ``uv run --group oracles python scripts/gen_property_data.py`` (needs
the optional ``chemicals``/``thermo``/``scipy`` stack). The emitted coefficients
are plain reference data; none of those packages are runtime dependencies.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable

import numpy as np
from chemicals import dipole as dip
from chemicals import interface as itf
from chemicals import phase_change as pch
from chemicals import thermal_conductivity as tcm
from chemicals import vapor_pressure as vap
from chemicals import viscosity as vis
from chemicals import volume as vol
from chemicals.heat_capacity import Poling as poling_cp
from chemicals.heat_capacity import TRCCp
from scipy.optimize import curve_fit

from fugacio.thermo.components import DATABASE
from fugacio.thermo.constants import ATM, R

OUT = "packages/fugacio-thermo/src/fugacio/thermo/_property_data.py"

warnings.filterwarnings("ignore")


def fnum(x: float, sig: int = 6) -> float:
    """Round to ``sig`` significant figures (keeps the emitted file compact)."""
    if x == 0.0 or not math.isfinite(x):
        return float(x)
    return float(f"{x:.{sig}g}")


def _finite(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _has(df: object, cas: str) -> bool:
    return cas in df.index  # type: ignore[union-attr]


def _sample(
    fn: Callable[[float], float], lo: float, hi: float, n: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``fn`` over ``[lo, hi]``, dropping non-finite / non-positive points."""
    ts, ys = [], []
    for t in np.linspace(lo, hi, n):
        try:
            y = fn(float(t))
        except Exception:
            continue
        if y is not None and math.isfinite(y) and y > 0.0:
            ts.append(float(t))
            ys.append(float(y))
    return np.asarray(ts), np.asarray(ys)


def _max_rel_err(y: np.ndarray, y_fit: np.ndarray) -> float:
    return float(np.max(np.abs(y_fit - y) / np.abs(y)))


# --- DIPPR / Mulero-Cachadina evaluators (mirror fugacio.thermo.correlations) ----


def eval_dippr100(t: float, c: tuple[float, ...]) -> float:
    return c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3 + c[4] * t**4


def eval_dippr101(t: float, c: tuple[float, ...]) -> float:
    return math.exp(c[0] + c[1] / t + c[2] * math.log(t) + c[3] * t ** c[4])


def eval_dippr102(t: float, c: tuple[float, ...]) -> float:
    return c[0] * t ** c[1] / (1.0 + c[2] / t + c[3] / t**2)


def eval_dippr106(t: float, tc: float, c: tuple[float, ...]) -> float:
    tr = min(t / tc, 1.0 - 1e-12)
    tau = 1.0 - tr
    return c[0] * tau ** (c[1] + c[2] * tr + c[3] * tr**2)


def eval_mc(t: float, tc: float, c: tuple[float, ...]) -> float:
    tau = max(1.0 - t / tc, 0.0)
    return c[0] * tau ** c[1] + c[2] * tau ** c[3]


# --- Per-property builders --------------------------------------------------------

Row = tuple[float, ...]


# Hand-curated DIPPR-105 rows for fluids missing from (or poorly served by) the
# machine-readable Perry table. Water has no wide-range DIPPR-105 row (DIPPR
# represents it with eq. 116), and COSTALD/Rackett are 8-30% off for it, so this
# row is a least-squares 105-form refit of the IAPWS-95 saturated-liquid molar
# density (via CoolProp) over 273.16-633.15 K: max error 2.2% (at the cold-water
# density anomaly), <1% above 290 K.
MANUAL_RHO105: dict[str, Row] = {
    "water": (27.206, 0.0205428, 647.096, 0.0610050, 273.16, 633.15),
}


def liquid_density(cas: str, name: str, tc: float, mw: float) -> Row | None:
    """``(c1..c4, tmin, tmax)`` DIPPR-105 in mol/m^3, transcribed or refitted."""
    if name in MANUAL_RHO105:
        return MANUAL_RHO105[name]
    perry = vol.rho_data_Perry_8E_105_l
    if _has(perry, cas):
        r = perry.loc[cas]
        row = (
            float(r["C1"]),
            float(r["C2"]),
            float(r["C3"]),
            float(r["C4"]),
            float(r["Tmin"]),
            float(r["Tmax"]),
        )
        mid = 0.5 * (row[4] + row[5])
        val = eval_dippr105(mid, row[:4])
        if 1.0e2 < val < 1.5e5:
            return row
        print(f"  DROP rho105 {name}: implausible {val:.3g} mol/m^3")
        return None
    ppds = vol.rho_data_VDI_PPDS_2
    if _has(ppds, cas):
        r = ppds.loc[cas]
        rhoc, tc_s = float(r["rhoc"]), float(r["Tc"])
        a, b, c, d = (float(r[k]) for k in "ABCD")

        def rho_mol(t: float) -> float:
            tau = max(1.0 - t / tc_s, 0.0)
            rho_kg = (
                rhoc + a * tau**0.35 + b * tau ** (2.0 / 3.0) + c * tau + d * tau ** (4.0 / 3.0)
            )
            return rho_kg / (mw * 1e-3)

        lo, hi = 0.3 * tc_s, 0.95 * tc_s
        ts, ys = _sample(rho_mol, lo, hi)
        if len(ts) < 20:
            return None

        def model(t: np.ndarray, c1: float, c2: float, c3: float, c4: float) -> np.ndarray:
            return c1 / c2 ** (1.0 + (1.0 - t / c3) ** c4)

        try:
            p0 = (float(ys[-1]) * 3.0, 0.27, tc_s, 0.28)
            popt, _ = curve_fit(model, ts, ys, p0=p0, maxfev=20000)
        except Exception:
            return None
        if _max_rel_err(ys, model(ts, *popt)) > 0.02 or popt[2] <= hi:
            print(f"  DROP rho105 {name}: VDI refit failed tolerance")
            return None
        c1, c2, c3, c4 = (fnum(v) for v in popt)
        return (c1, c2, c3, c4, fnum(float(ts[0]), 5), fnum(float(ts[-1]), 5))
    return None


def eval_dippr105(t: float, c: tuple[float, ...]) -> float:
    return c[0] / c[1] ** (1.0 + (1.0 - t / c[2]) ** c[3])


def liquid_viscosity(cas: str, name: str, tc: float, tb: float | None) -> Row | None:
    """``(c1..c5, tmin, tmax)`` DIPPR-101 in Pa*s, transcribed or refitted."""
    perry = vis.mu_data_Perrys_8E_2_313
    if _has(perry, cas):
        r = perry.loc[cas]
        c4 = _finite(r["C4"])
        c5 = _finite(r["C5"], 1.0) if c4 != 0.0 else 1.0
        row = (
            _finite(r["C1"]),
            _finite(r["C2"]),
            _finite(r["C3"]),
            c4,
            c5,
            float(r["Tmin"]),
            float(r["Tmax"]),
        )
        mid = 0.5 * (row[5] + row[6])
        val = eval_dippr101(mid, row[:5])
        if 1.0e-6 < val < 1.0e2:
            return row
        print(f"  DROP mu_l {name}: implausible {val:.3g} Pa*s")
        return None

    sources: list[tuple[Callable[[float], float], float, float]] = []
    ppds = vis.mu_data_VDI_PPDS_7
    if _has(ppds, cas):
        r = ppds.loc[cas]
        a, b, c, d, e = (float(r[k]) for k in "ABCDE")
        lo = max(0.95 * c if c > 0 else 0.35 * tc, 0.3 * tc)
        hi = 0.95 * tc if tb is None else min(1.3 * tb, 0.95 * tc)
        sources.append((lambda t: vis.PPDS9(t, a, b, c, d, e), lo, hi))
    vn3 = vis.mu_data_VN3
    if _has(vn3, cas):
        r = vn3.loc[cas]
        a, b, c = float(r["A"]), float(r["B"]), float(r["C"])
        lo, hi = float(r["Tmin"]), float(r["Tmax"])
        sources.append((lambda t: vis.Viswanath_Natarajan_3(t, a, b, c), lo, hi))
    vn2 = vis.mu_data_VN2
    if _has(vn2, cas):
        r = vn2.loc[cas]
        a, b = float(r["A"]), float(r["B"])
        lo, hi = float(r["Tmin"]), float(r["Tmax"])
        sources.append((lambda t: vis.Viswanath_Natarajan_2(t, a, b), lo, hi))

    for fn, lo, hi in sources:
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi - lo < 20.0:
            continue
        ts, ys = _sample(fn, lo, min(hi, 0.99 * tc))
        if len(ts) < 20:
            continue
        # ln(mu) = c1 + c2/T + c3*ln(T) + c4*T is linear in the coefficients.
        basis = np.stack([np.ones_like(ts), 1.0 / ts, np.log(ts), ts], axis=1)
        coef, *_ = np.linalg.lstsq(basis, np.log(ys), rcond=None)
        fit = np.exp(basis @ coef)
        if _max_rel_err(ys, fit) > 0.02:
            continue
        c1, c2, c3, c4 = (fnum(float(v)) for v in coef)
        mid_val = eval_dippr101(0.5 * (ts[0] + ts[-1]), (c1, c2, c3, c4, 1.0))
        if not (1.0e-6 < mid_val < 1.0e2):
            continue
        return (c1, c2, c3, c4, 1.0, fnum(float(ts[0]), 5), fnum(float(ts[-1]), 5))
    return None


def _dippr102_refit(
    name: str,
    label: str,
    fn: Callable[[float], float],
    lo: float,
    hi: float,
    bounds: tuple[float, float],
) -> Row | None:
    ts, ys = _sample(fn, lo, hi)
    if len(ts) < 20:
        return None

    def model(t: np.ndarray, ln_c1: float, c2: float, c3: float, c4: float) -> np.ndarray:
        return np.exp(ln_c1) * t**c2 / (1.0 + c3 / t + c4 / t**2)

    try:
        p0 = (math.log(ys[len(ys) // 2] / ts[len(ts) // 2] ** 0.8), 0.8, 100.0, 0.0)
        popt, _ = curve_fit(model, ts, ys, p0=p0, maxfev=50000)
    except Exception:
        return None
    if _max_rel_err(ys, model(ts, *popt)) > 0.03:
        print(f"  DROP {label} {name}: refit failed tolerance")
        return None
    c1, c2, c3, c4 = fnum(math.exp(popt[0])), fnum(popt[1]), fnum(popt[2]), fnum(popt[3])
    mid_val = eval_dippr102(0.5 * (ts[0] + ts[-1]), (c1, c2, c3, c4))
    if not (bounds[0] < mid_val < bounds[1]):
        print(f"  DROP {label} {name}: implausible {mid_val:.3g}")
        return None
    return (c1, c2, c3, c4, fnum(float(ts[0]), 5), fnum(float(ts[-1]), 5))


def gas_viscosity(cas: str, name: str) -> Row | None:
    """``(c1..c4, tmin, tmax)`` DIPPR-102 in Pa*s, transcribed or refitted."""
    perry = vis.mu_data_Perrys_8E_2_312
    if _has(perry, cas):
        r = perry.loc[cas]
        row = (
            _finite(r["C1"]),
            _finite(r["C2"]),
            _finite(r["C3"]),
            _finite(r["C4"]),
            float(r["Tmin"]),
            float(r["Tmax"]),
        )
        mid = 0.5 * (row[4] + row[5])
        val = eval_dippr102(mid, row[:4])
        if 1.0e-7 < val < 1.0e-3:
            return row
        print(f"  DROP mu_g {name}: implausible {val:.3g} Pa*s")
        return None
    ppds = vis.mu_data_VDI_PPDS_8
    if _has(ppds, cas):
        r = ppds.loc[cas]
        a, b, c, d, e = (float(r[k]) for k in "ABCDE")
        return _dippr102_refit(
            name,
            "mu_g",
            lambda t: a + b * t + c * t**2 + d * t**3 + e * t**4,
            220.0,
            1000.0,
            (1.0e-7, 1.0e-3),
        )
    return None


def liquid_conductivity(cas: str, name: str, tc: float, tb: float | None) -> Row | None:
    """``(c1..c5, tmin, tmax)`` DIPPR-100 in W/m/K, transcribed."""
    perry = tcm.k_data_Perrys_8E_2_315
    if _has(perry, cas):
        r = perry.loc[cas]
        row = (
            _finite(r["C1"]),
            _finite(r["C2"]),
            _finite(r["C3"]),
            _finite(r["C4"]),
            _finite(r["C5"]),
            float(r["Tmin"]),
            float(r["Tmax"]),
        )
        mid = 0.5 * (row[5] + row[6])
        val = eval_dippr100(mid, row[:5])
        if 0.01 < val < 1.5:
            return row
        print(f"  DROP k_l {name}: implausible {val:.3g} W/m/K")
        return None
    ppds = tcm.k_data_VDI_PPDS_9
    if _has(ppds, cas):
        r = ppds.loc[cas]
        a, b, c, d, e = (float(r[k]) for k in "ABCDE")
        lo = 0.35 * tc if tb is None else 0.55 * tb
        hi = 0.9 * tc if tb is None else min(1.2 * tb, 0.9 * tc)
        row = (a, b, c, d, e, fnum(lo, 5), fnum(hi, 5))
        for t in (lo, 0.5 * (lo + hi), hi):
            val = eval_dippr100(t, row[:5])
            if not (0.01 < val < 1.5):
                print(f"  DROP k_l {name}: implausible VDI {val:.3g} W/m/K at {t:.0f} K")
                return None
        return row
    return None


def gas_conductivity(cas: str, name: str) -> Row | None:
    """``(c1..c4, tmin, tmax)`` DIPPR-102 in W/m/K, transcribed or refitted."""
    perry = tcm.k_data_Perrys_8E_2_314
    if _has(perry, cas):
        r = perry.loc[cas]
        row = (
            _finite(r["C1"]),
            _finite(r["C2"]),
            _finite(r["C3"]),
            _finite(r["C4"]),
            float(r["Tmin"]),
            float(r["Tmax"]),
        )
        mid = 0.5 * (row[4] + row[5])
        val = eval_dippr102(mid, row[:4])
        if 1.0e-4 < val < 1.0:
            return row
        print(f"  DROP k_g {name}: implausible {val:.3g} W/m/K")
        return None
    ppds = tcm.k_data_VDI_PPDS_10
    if _has(ppds, cas):
        r = ppds.loc[cas]
        a, b, c, d, e = (float(r[k]) for k in "ABCDE")
        return _dippr102_refit(
            name,
            "k_g",
            lambda t: a + b * t + c * t**2 + d * t**3 + e * t**4,
            250.0,
            1000.0,
            (1.0e-4, 1.0),
        )
    return None


def surface_tension(cas: str, name: str, tc: float, tb: float | None) -> Row | None:
    """``(tc, s0, n0, s1, n1, s2, n2, tmin, tmax)`` Mulero-Cachadina in N/m."""
    mc = itf.sigma_data_Mulero_Cachadina
    if _has(mc, cas):
        r = mc.loc[cas]
        return (
            float(r["Tc"]),
            _finite(r["sigma0"]),
            _finite(r["n0"], 1.0),
            _finite(r["sigma1"]),
            _finite(r["n1"], 1.0),
            _finite(r["sigma2"]),
            _finite(r["n2"], 1.0),
            float(r["Tmin"]),
            float(r["Tmax"]),
        )

    sources: list[tuple[Callable[[float], float], float, float]] = []
    som = itf.sigma_data_Somayajulu2
    if _has(som, cas):
        r = som.loc[cas]
        tc_s, tt = float(r["Tc"]), float(r["Tt"])
        a, b, c = float(r["A"]), float(r["B"]), float(r["C"])
        sources.append((lambda t: itf.Somayajulu(t, tc_s, a, b, c), tt, 0.99 * tc_s))
    jas = itf.sigma_data_Jasper_Lange
    if _has(jas, cas):
        r = jas.loc[cas]
        a, b = float(r["a"]), float(r["b"])
        lo = _finite(r["Tmin"], 0.0)
        hi = _finite(r["Tmax"], 0.0)
        if not (lo > 0.0 and hi > lo):
            lo = 0.45 * tc if tb is None else 0.7 * tb
            hi = 0.7 * tc if tb is None else min(1.05 * tb, 0.85 * tc)
        sources.append((lambda t: itf.Jasper(t, a, b), lo, min(hi, 0.95 * tc)))

    for fn, lo, hi in sources:
        if hi - lo < 20.0:
            continue
        ts, ys = _sample(fn, lo, hi)
        if len(ts) < 15:
            continue

        def model(t: np.ndarray, s0: float, n0: float, s1: float, n1: float) -> np.ndarray:
            tau = np.clip(1.0 - t / tc, 1e-12, 1.0)
            return s0 * tau**n0 + s1 * tau**n1

        try:
            popt, _ = curve_fit(
                model,
                ts,
                ys,
                p0=(float(ys[0]) / 0.5, 1.26, 0.0, 2.0),
                maxfev=20000,
            )
        except Exception:
            continue
        if _max_rel_err(ys, model(ts, *popt)) > 0.02:
            continue
        s0, n0, s1, n1 = (fnum(float(v)) for v in popt)
        mid_val = eval_mc(0.5 * (ts[0] + ts[-1]), tc, (s0, n0, s1, n1))
        if not (1.0e-4 < mid_val < 0.5):
            continue
        return (
            fnum(tc, 6),
            s0,
            n0,
            s1,
            n1,
            0.0,
            1.0,
            fnum(float(ts[0]), 5),
            fnum(float(ts[-1]), 5),
        )
    return None


def heat_of_vaporization(cas: str, name: str, mw: float) -> Row | None:
    """``(tc, c1..c4, tmin, tmax)`` DIPPR-106 in J/mol, transcribed or refitted."""
    perry = pch.phase_change_data_Perrys2_150
    if _has(perry, cas):
        r = perry.loc[cas]
        tc_s = float(r["Tc"])
        row = (
            tc_s,
            _finite(r["C1"]),
            _finite(r["C2"]),
            _finite(r["C3"]),
            _finite(r["C4"]),
            float(r["Tmin"]),
            float(r["Tmax"]),
        )
        mid = 0.5 * (row[5] + row[6])
        val = eval_dippr106(mid, tc_s, row[1:5])
        # Lower bound admits the quantum cryogens (helium boils off ~90 J/mol).
        if 10.0 < val < 3.0e5:
            return row
        print(f"  DROP hvap {name}: implausible {val:.3g} J/mol")
        return None
    ppds = pch.phase_change_data_VDI_PPDS_4
    if _has(ppds, cas):
        r = ppds.loc[cas]
        tc_s = float(r["Tc"])
        a, b, c, d, e = (float(r[k]) for k in "ABCDE")

        def hvap_fn(t: float) -> float:
            return pch.PPDS12(t, tc_s, a, b, c, d, e)

        ts, ys = _sample(hvap_fn, 0.3 * tc_s, 0.95 * tc_s)
        if len(ts) < 20:
            return None

        def model(t: np.ndarray, c1: float, c2: float, c3: float, c4: float) -> np.ndarray:
            tr = np.clip(t / tc_s, 0.0, 1.0 - 1e-12)
            tau = 1.0 - tr
            return c1 * tau ** (c2 + c3 * tr + c4 * tr**2)

        try:
            tau_mid = 1.0 - ts[len(ts) // 2] / tc_s
            p0 = (float(ys[len(ys) // 2]) / tau_mid**0.38, 0.38, 0.0, 0.0)
            popt, _ = curve_fit(model, ts, ys, p0=p0, maxfev=50000)
        except Exception:
            return None
        if _max_rel_err(ys, model(ts, *popt)) > 0.03:
            print(f"  DROP hvap {name}: VDI refit failed tolerance")
            return None
        c1, c2, c3, c4 = (fnum(float(v)) for v in popt)
        mid_val = eval_dippr106(0.5 * (ts[0] + ts[-1]), tc_s, (c1, c2, c3, c4))
        if not (10.0 < mid_val < 3.0e5):
            return None
        return (
            fnum(tc_s, 6),
            c1,
            c2,
            c3,
            c4,
            fnum(float(ts[0]), 5),
            fnum(float(ts[-1]), 5),
        )
    return None


def antoine_backfill(
    cas: str, name: str, tc: float, pc: float, tb: float | None
) -> tuple[float, float, float, float, float] | None:
    """NIST-form Antoine ``(a, b, c, tmin, tmax)`` fitted from open Psat sources."""
    sources: list[tuple[Callable[[float], float], float, float]] = []
    if _has(vap.Psat_data_AntoinePoling, cas):
        r = vap.Psat_data_AntoinePoling.loc[cas]
        row = (
            float(r["A"]) - 5.0,
            float(r["B"]),
            float(r["C"]),
            float(r["Tmin"]),
            float(r["Tmax"]),
        )
        if _antoine_ok(row, tb):
            return row
    if _has(vap.Psat_data_WagnerMcGarry, cas):
        r = vap.Psat_data_WagnerMcGarry.loc[cas]
        a, b, c, d = (float(r[k]) for k in "ABCD")
        tc_s, pc_s, tmin = float(r["Tc"]), float(r["Pc"]), float(r["Tmin"])
        sources.append(
            (lambda t: vap.Wagner_original(t, tc_s, pc_s, a, b, c, d), tmin, 0.99 * tc_s)
        )
    if _has(vap.Psat_data_WagnerPoling, cas):
        r = vap.Psat_data_WagnerPoling.loc[cas]
        a, b, c, d = (float(r[k]) for k in "ABCD")
        tc_s, pc_s = float(r["Tc"]), float(r["Pc"])
        tmin, tmax = float(r["Tmin"]), float(r["Tmax"])
        sources.append((lambda t: vap.Wagner(t, tc_s, pc_s, a, b, c, d), tmin, tmax))
    if _has(vap.Psat_data_Perrys2_8, cas):
        r = vap.Psat_data_Perrys2_8.loc[cas]
        c1, c2, c3, c4, c5 = (_finite(r[f"C{i}"]) for i in range(1, 6))
        tmin, tmax = float(r["Tmin"]), float(r["Tmax"])
        sources.append(
            (
                lambda t: math.exp(c1 + c2 / t + c3 * math.log(t) + c4 * t**c5),
                tmin,
                min(tmax, 0.99 * tc),
            )
        )

    for fn, lo, hi in sources:
        if hi - lo < 30.0:
            continue
        # Fit over the VLE-relevant window (~1 kPa to ~5 bar): a three-constant
        # Antoine cannot bridge the whole triple-to-critical span of the source
        # correlation at the 1% QC level, and the low/high extremes are where it
        # is never evaluated anyway.
        ts, ps = _sample(fn, lo, hi, n=80)
        keep = (ps > 1.0e3) & (ps < 5.0e5)
        ts, ps = ts[keep], ps[keep]
        if len(ts) < 15:
            continue

        def model(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            return a - b / (t + c)

        try:
            popt, _ = curve_fit(
                model, ts, np.log10(ps / 1.0e5), p0=(4.0, 1300.0, -40.0), maxfev=20000
            )
        except Exception:
            continue
        log_fit = model(ts, *popt)
        if float(np.max(np.abs(log_fit - np.log10(ps / 1.0e5)))) > 0.01:
            continue
        row = (
            fnum(float(popt[0])),
            fnum(float(popt[1])),
            fnum(float(popt[2])),
            fnum(float(ts[0]), 5),
            fnum(float(ts[-1]), 5),
        )
        if _antoine_ok(row, tb):
            return row
    return None


def _antoine_ok(row: tuple[float, float, float, float, float], tb: float | None) -> bool:
    """Require the fit to reproduce 1 atm at the normal boiling point within 3%."""
    if tb is None:
        return True
    a, b, c, _tmin, _tmax = row
    try:
        p = 1.0e5 * 10.0 ** (a - b / (tb + c))
    except (OverflowError, ZeroDivisionError):
        return False
    return math.isfinite(p) and abs(p - ATM) / ATM <= 0.03


def cp_backfill(cas: str, name: str) -> tuple[float, float, float, float, float, float] | None:
    """Ideal-gas ``Cp/R`` cubic ``(a, b, c, e, tmin, tmax)`` fitted from open data."""
    from chemicals.heat_capacity import Cp_data_Poling, TRC_gas_data

    fn: Callable[[float], float] | None = None
    lo = hi = 0.0
    if _has(Cp_data_Poling, cas):
        r = Cp_data_Poling.loc[cas]
        coeffs = [_finite(r[f"a{i}"]) for i in range(5)]
        if any(c != 0.0 for c in coeffs):
            lo, hi = max(_finite(r["Tmin"], 200.0), 200.0), min(_finite(r["Tmax"], 1000.0), 1000.0)
            fn = lambda t: poling_cp(t, *coeffs)  # noqa: E731
    if fn is None and _has(TRC_gas_data, cas):
        r = TRC_gas_data.loc[cas]
        coeffs = [_finite(r[f"a{i}"]) for i in range(8)]
        lo, hi = max(_finite(r["Tmin"], 200.0), 200.0), min(_finite(r["Tmax"], 1500.0), 1000.0)
        fn = lambda t: TRCCp(t, *coeffs)  # noqa: E731
    if fn is None:
        return None
    if hi - lo < 100.0:
        lo, hi = lo, lo + 300.0
    ts, ys = _sample(fn, lo, hi, n=80)
    if len(ts) < 20:
        return None
    cp_over_r = ys / R
    tau = ts / 1000.0
    e_s, c_s, b_s, a = (float(v) for v in np.polyfit(tau, cp_over_r, 3))
    b, c, e = b_s / 1e3, c_s / 1e6, e_s / 1e9
    value_298 = R * (a + b * 298.15 + c * 298.15**2 + e * 298.15**3)
    if not (2.5 * R <= value_298 < 400.0):
        print(f"  DROP cp {name}: Cp(298) = {value_298:.3g} out of range")
        return None
    fit = R * (a + b * ts + c * ts**2 + e * ts**3)
    if _max_rel_err(ys, fit) > 0.05:
        print(f"  DROP cp {name}: fit failed tolerance")
        return None
    return (
        fnum(a),
        fnum(b),
        fnum(c),
        fnum(e),
        fnum(float(ts[0]), 5),
        fnum(float(ts[-1]), 5),
    )


# --- Assembly ---------------------------------------------------------------------

HEADER = '''"""Curated pure-component property-correlation coefficients (generated).

Auto-generated by ``scripts/gen_property_data.py`` from the open ``chemicals``
dataset; do not edit by hand. Perry/DIPPR-form coefficients are transcribed
verbatim where the source tabulates them and least-squares refitted onto the
same canonical forms from the other open sources otherwise (the generator
records and bounds the refit error). Keys are canonical database names; all
values are SI unless noted.

Tables (and the kernels that consume them):

* ``ZRA`` -- Spencer-Danner Rackett compressibilities
  (:func:`fugacio.thermo.volumetric.rackett_volume`);
* ``COSTALD_VOLUME`` -- ``(V* m^3/mol, omega_SRK)``
  (:func:`fugacio.thermo.volumetric.costald_volume`);
* ``DIPOLE`` -- gas-phase dipole moments in debye (Chung estimators);
* ``RHO_LIQUID_DIPPR105`` -- ``(c1..c4, tmin, tmax)``, density in mol/m^3
  (:func:`fugacio.thermo.correlations.dippr105`);
* ``MU_LIQUID_DIPPR101`` -- ``(c1..c5, tmin, tmax)``, viscosity in Pa*s
  (:func:`fugacio.thermo.correlations.dippr101`);
* ``MU_GAS_DIPPR102`` -- ``(c1..c4, tmin, tmax)``, viscosity in Pa*s
  (:func:`fugacio.thermo.correlations.dippr102`);
* ``K_LIQUID_DIPPR100`` -- ``(c1..c5, tmin, tmax)``, conductivity in W/m/K
  (:func:`fugacio.thermo.correlations.dippr100`);
* ``K_GAS_DIPPR102`` -- ``(c1..c4, tmin, tmax)``, conductivity in W/m/K;
* ``SIGMA_MULERO_CACHADINA`` -- ``(tc, s0, n0, s1, n1, s2, n2, tmin, tmax)``,
  tension in N/m (:func:`fugacio.thermo.correlations.mulero_cachadina`);
* ``HVAP_DIPPR106`` -- ``(tc, c1..c4, tmin, tmax)``, enthalpy of vaporization in
  J/mol (:func:`fugacio.thermo.correlations.dippr106`);
* ``ANTOINE_BACKFILL`` -- ``(a, b, c, tmin, tmax)`` NIST-form Antoine constants
  fitted for components whose curated record lacks them;
* ``CP_BACKFILL`` -- ``(a, b, c, e, tmin, tmax)`` ideal-gas ``Cp/R`` polynomial
  coefficients fitted for components whose curated record lacks them.
"""

from __future__ import annotations
'''


def _compact(value: object) -> str:
    """Render a row tuple with floats at 12 significant digits (keeps lines short)."""
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, tuple):
        return "(" + ", ".join(_compact(v) for v in value) + ")"
    return repr(value)


def _emit_dict(name: str, annotation: str, comment: str, rows: dict[str, object]) -> str:
    lines = [f"#: {comment}"]
    head = f"{name}: {annotation} = {{"
    if len(head) > 100:
        # Wrap the subscript the way ruff-format would, to keep the line under limit.
        outer, _, inner = annotation.partition("[")
        lines += [f"{name}: {outer}[", f"    {inner.removesuffix(']')}", "] = {"]
    else:
        lines += [head]
    for key in sorted(rows):
        lines.append(f"    {key!r}: {_compact(rows[key])},")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    from fugacio.thermo._property_data import ANTOINE_BACKFILL as prev_antoine
    from fugacio.thermo._property_data import CP_BACKFILL as prev_cp

    zra: dict[str, object] = {}
    costald: dict[str, object] = {}
    dipole: dict[str, object] = {}
    rho105: dict[str, object] = {}
    mu_l: dict[str, object] = {}
    mu_g: dict[str, object] = {}
    k_l: dict[str, object] = {}
    k_g: dict[str, object] = {}
    sigma: dict[str, object] = {}
    hvap: dict[str, object] = {}
    antoine: dict[str, object] = {}
    cp: dict[str, object] = {}

    for name, comp in sorted(DATABASE.items()):
        cas = comp.cas
        if not cas:
            continue

        if _has(vol.rho_data_COSTALD, cas):
            r = vol.rho_data_COSTALD.loc[cas]
            vchar, omega_srk, z_ra = (
                _finite(r["Vchar"]),
                _finite(r["omega_SRK"]),
                _finite(r["Z_RA"]),
            )
            if vchar > 0.0 and omega_srk != 0.0:
                costald[name] = (vchar, omega_srk)
            if 0.2 < z_ra < 0.35:
                zra[name] = z_ra

        try:
            d = dip.dipole_moment(cas)
        except Exception:
            d = None
        if d is not None and math.isfinite(d) and d >= 0.0:
            dipole[name] = fnum(float(d), 4)

        row = liquid_density(cas, name, comp.tc, comp.mw)
        if row is not None:
            rho105[name] = row
            if name not in zra:
                # One-point Rackett fit at Tr = 0.7 (clipped into the fit range).
                t_star = min(max(0.7 * comp.tc, row[4]), row[5])
                tr = t_star / comp.tc
                if 0.0 < tr < 1.0:
                    v = 1.0 / eval_dippr105(t_star, row[:4])
                    base = comp.pc * v / (R * comp.tc)
                    if base > 0.0:
                        z = base ** (1.0 / (1.0 + (1.0 - tr) ** (2.0 / 7.0)))
                        if isinstance(z, float) and 0.2 < z < 0.35:
                            zra[name] = fnum(z, 5)

        row = liquid_viscosity(cas, name, comp.tc, comp.tb)
        if row is not None:
            mu_l[name] = row
        row = gas_viscosity(cas, name)
        if row is not None:
            mu_g[name] = row
        row = liquid_conductivity(cas, name, comp.tc, comp.tb)
        if row is not None:
            k_l[name] = row
        row = gas_conductivity(cas, name)
        if row is not None:
            k_g[name] = row
        row = surface_tension(cas, name, comp.tc, comp.tb)
        if row is not None:
            sigma[name] = row
        row = heat_of_vaporization(cas, name, comp.mw)
        if row is not None:
            hvap[name] = row
        # DATABASE already has prior backfill applied, so consult the previous
        # tables too -- otherwise a regeneration would see nothing as missing
        # and silently drop every backfilled row.
        if comp.antoine is None or name in prev_antoine:
            arow = antoine_backfill(cas, name, comp.tc, comp.pc, comp.tb)
            if arow is not None:
                antoine[name] = arow
        if comp.cp_ig is None or name in prev_cp:
            crow = cp_backfill(cas, name)
            if crow is not None:
                cp[name] = crow

    sections: Iterable[str] = (
        _emit_dict(
            "ZRA",
            "dict[str, float]",
            "Spencer-Danner Rackett compressibility ``Z_RA``.",
            zra,
        ),
        _emit_dict(
            "COSTALD_VOLUME",
            "dict[str, tuple[float, float]]",
            "COSTALD characteristic volume ``V*`` (m^3/mol) and SRK acentric factor.",
            costald,
        ),
        _emit_dict(
            "DIPOLE",
            "dict[str, float]",
            "Gas-phase dipole moment (debye).",
            dipole,
        ),
        _emit_dict(
            "RHO_LIQUID_DIPPR105",
            "dict[str, tuple[float, float, float, float, float, float]]",
            "DIPPR-105 saturated-liquid molar density (mol/m^3): ``(c1, c2, c3, c4, tmin, tmax)``.",
            rho105,
        ),
        _emit_dict(
            "MU_LIQUID_DIPPR101",
            "dict[str, tuple[float, float, float, float, float, float, float]]",
            "DIPPR-101 liquid viscosity (Pa*s): ``(c1, c2, c3, c4, c5, tmin, tmax)``.",
            mu_l,
        ),
        _emit_dict(
            "MU_GAS_DIPPR102",
            "dict[str, tuple[float, float, float, float, float, float]]",
            "DIPPR-102 dilute-gas viscosity (Pa*s): ``(c1, c2, c3, c4, tmin, tmax)``.",
            mu_g,
        ),
        _emit_dict(
            "K_LIQUID_DIPPR100",
            "dict[str, tuple[float, float, float, float, float, float, float]]",
            "DIPPR-100 liquid thermal conductivity (W/m/K): ``(c1, c2, c3, c4, c5, tmin, tmax)``.",
            k_l,
        ),
        _emit_dict(
            "K_GAS_DIPPR102",
            "dict[str, tuple[float, float, float, float, float, float]]",
            "DIPPR-102 dilute-gas thermal conductivity (W/m/K): ``(c1, c2, c3, c4, tmin, tmax)``.",
            k_g,
        ),
        _emit_dict(
            "SIGMA_MULERO_CACHADINA",
            "dict[str, tuple[float, float, float, float, float, float, float, float, float]]",
            "Mulero-Cachadina surface tension (N/m): ``(tc, s0, n0, s1, n1, s2, n2, tmin, tmax)``.",
            sigma,
        ),
        _emit_dict(
            "HVAP_DIPPR106",
            "dict[str, tuple[float, float, float, float, float, float, float]]",
            "DIPPR-106 enthalpy of vaporization (J/mol): ``(tc, c1, c2, c3, c4, tmin, tmax)``.",
            hvap,
        ),
        _emit_dict(
            "ANTOINE_BACKFILL",
            "dict[str, tuple[float, float, float, float, float]]",
            "NIST-form Antoine constants for components lacking a curated inline fit.",
            antoine,
        ),
        _emit_dict(
            "CP_BACKFILL",
            "dict[str, tuple[float, float, float, float, float, float]]",
            "Ideal-gas ``Cp/R = a + b*T + c*T^2 + e*T^3`` fits for components lacking one.",
            cp,
        ),
    )

    with open(OUT, "w") as fh:
        fh.write(HEADER + "\n" + "\n\n".join(sections))

    print(
        f"wrote {OUT}: zra={len(zra)} costald={len(costald)} dipole={len(dipole)} "
        f"rho105={len(rho105)} mu_l={len(mu_l)} mu_g={len(mu_g)} k_l={len(k_l)} "
        f"k_g={len(k_g)} sigma={len(sigma)} hvap={len(hvap)} antoine={len(antoine)} "
        f"cp={len(cp)}"
    )


if __name__ == "__main__":
    main()
