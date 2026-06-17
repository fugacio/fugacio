"""Generate vendored multiparameter Helmholtz-EOS coefficient tables.

A one-shot authoring tool (not part of the package), companion to
``scripts/gen_components.py``. It extracts the published reference
(multiparameter Helmholtz-energy) equation-of-state coefficients, the
saturation ancillary equations, and the surface-tension correlations for a
curated list of process fluids from CoolProp's open fluid library
(``get_fluid_param_string(fluid, "JSON")``), and writes them to
``packages/fugacio-thermo/src/fugacio/thermo/helmholtz/_data.py``.

CoolProp is itself a faithful transcription of the peer-reviewed reference
formulations (IAPWS-95 for water, Span & Wagner for CO2, Setzmann & Wagner for
methane, ...); the BibTeX key of the source paper is recorded per fluid. Only
term families that :mod:`fugacio.thermo.helmholtz.terms` implements are
accepted; the script fails loudly if a fluid needs an unsupported term, so a
regeneration can never silently change the model class.

Run with ``uv run --group oracles python scripts/gen_helmholtz.py`` (requires
the optional ``coolprop`` package).
"""

from __future__ import annotations

import json
from typing import Any

from CoolProp.CoolProp import get_fluid_param_string

DATA_PY = "packages/fugacio-thermo/src/fugacio/thermo/helmholtz/_data.py"

#: CoolProp fluid name -> Fugacio registry name. Names match the Fugacio
#: component database where the species exists there; refrigerants keep their
#: ASHRAE designations.
FLUIDS: dict[str, str] = {
    "Water": "water",
    "CarbonDioxide": "carbon dioxide",
    "Nitrogen": "nitrogen",
    "Oxygen": "oxygen",
    "Argon": "argon",
    "CarbonMonoxide": "carbon monoxide",
    "Hydrogen": "hydrogen",
    "HydrogenSulfide": "hydrogen sulfide",
    "SulfurDioxide": "sulfur dioxide",
    "Ammonia": "ammonia",
    "Methane": "methane",
    "Ethane": "ethane",
    "n-Propane": "propane",
    "n-Butane": "n-butane",
    "IsoButane": "isobutane",
    "n-Pentane": "n-pentane",
    "n-Hexane": "n-hexane",
    "n-Octane": "n-octane",
    "Ethylene": "ethylene",
    "Propylene": "propylene",
    "Benzene": "benzene",
    "Toluene": "toluene",
    "Ethanol": "ethanol",
    "R134a": "R134a",
    "R32": "R32",
    "R1234yf": "R1234yf",
}

SUPPORTED_ALPHA0 = {
    "IdealGasHelmholtzLead",
    "IdealGasHelmholtzLogTau",
    "IdealGasHelmholtzPower",
    "IdealGasHelmholtzPlanckEinstein",
    "IdealGasHelmholtzPlanckEinsteinFunctionT",
    "IdealGasHelmholtzEnthalpyEntropyOffset",
}

SUPPORTED_ALPHAR = {
    "ResidualHelmholtzPower",
    "ResidualHelmholtzGaussian",
    "ResidualHelmholtzNonAnalytic",
    "ResidualHelmholtzGaoB",
}


def _ideal_terms(alpha0: list[dict[str, Any]], t_reducing: float) -> dict[str, Any]:
    """Normalize CoolProp ``alpha0`` blocks into Fugacio's four ideal families."""
    a1 = 0.0
    a2 = 0.0
    log_tau = 0.0
    power: list[tuple[float, float]] = []
    planck: list[tuple[float, float]] = []
    for term in alpha0:
        kind = term["type"]
        if kind not in SUPPORTED_ALPHA0:
            raise ValueError(f"unsupported alpha0 term {kind!r}")
        if kind in ("IdealGasHelmholtzLead", "IdealGasHelmholtzEnthalpyEntropyOffset"):
            # Lead is a1 + a2*tau + ln(delta); the offset is the same without
            # ln(delta). Exactly one Lead per fluid supplies the ln(delta).
            a1 += float(term["a1"])
            a2 += float(term["a2"])
        elif kind == "IdealGasHelmholtzLogTau":
            log_tau += float(term["a"])
        elif kind == "IdealGasHelmholtzPower":
            power += [(float(n), float(t)) for n, t in zip(term["n"], term["t"], strict=True)]
        elif kind == "IdealGasHelmholtzPlanckEinstein":
            planck += [(float(n), float(t)) for n, t in zip(term["n"], term["t"], strict=True)]
        else:  # PlanckEinsteinFunctionT: theta given in kelvin -> divide by Tc.
            tcrit = float(term["Tcrit"])
            if abs(tcrit - t_reducing) > 1e-6 * t_reducing:
                raise ValueError("PlanckEinsteinFunctionT Tcrit != reducing temperature")
            planck += [
                (float(n), float(v) / tcrit) for n, v in zip(term["n"], term["v"], strict=True)
            ]
    leads = sum(1 for t in alpha0 if t["type"] == "IdealGasHelmholtzLead")
    if leads != 1:
        raise ValueError(f"expected exactly one Lead term, found {leads}")
    return {
        "lead": (a1, a2),
        "log_tau": log_tau,
        "ideal_power": tuple(power),
        "planck_einstein": tuple(planck),
    }


def _residual_terms(alphar: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize CoolProp ``alphar`` blocks into Fugacio's four residual families."""
    power: list[tuple[float, ...]] = []
    gaussian: list[tuple[float, ...]] = []
    nonanalytic: list[tuple[float, ...]] = []
    gaob: list[tuple[float, ...]] = []
    for term in alphar:
        kind = term["type"]
        if kind not in SUPPORTED_ALPHAR:
            raise ValueError(f"unsupported alphar term {kind!r}")
        if kind == "ResidualHelmholtzPower":
            power += [
                (float(n), float(t), float(d), float(el))
                for n, t, d, el in zip(term["n"], term["t"], term["d"], term["l"], strict=True)
            ]
        elif kind == "ResidualHelmholtzGaussian":
            gaussian += [
                tuple(float(v) for v in row)
                for row in zip(
                    term["n"],
                    term["t"],
                    term["d"],
                    term["eta"],
                    term["beta"],
                    term["gamma"],
                    term["epsilon"],
                    strict=True,
                )
            ]
        elif kind == "ResidualHelmholtzNonAnalytic":
            nonanalytic += [
                tuple(float(v) for v in row)
                for row in zip(
                    term["n"],
                    term["a"],
                    term["b"],
                    term["beta"],
                    term["A"],
                    term["B"],
                    term["C"],
                    term["D"],
                    strict=True,
                )
            ]
        else:  # GaoB
            gaob += [
                tuple(float(v) for v in row)
                for row in zip(
                    term["n"],
                    term["t"],
                    term["d"],
                    term["eta"],
                    term["beta"],
                    term["gamma"],
                    term["epsilon"],
                    term["b"],
                    strict=True,
                )
            ]
    return {
        "power": tuple(power),
        "gaussian": tuple(gaussian),
        "nonanalytic": tuple(nonanalytic),
        "gaob": tuple(gaob),
    }


def _ancillary(block: dict[str, Any]) -> tuple[float, int, int, tuple[tuple[float, float], ...]]:
    """Normalize one saturation ancillary to ``(reducing, using_tau_r, noexp, ((n, t), ...))``.

    The exponential families evaluate ``reducing * exp(f * sum(n_i theta^t_i))``
    with ``f = Tc/T`` when ``using_tau_r`` else 1; the ``noexp`` family is
    ``reducing * (1 + sum(n_i theta^t_i))``, all with ``theta = 1 - T/Tc``.
    """
    kind = block["type"]
    noexp = 1 if kind.endswith("noexp") else 0
    using_tau_r = 1 if block.get("using_tau_r", False) else 0
    coeffs = tuple((float(n), float(t)) for n, t in zip(block["n"], block["t"], strict=True))
    return (float(block["reducing_value"]), using_tau_r, noexp, coeffs)


def _anc_rho_liquid(
    anc: tuple[float, int, int, tuple[tuple[float, float], ...]], t: float, tc: float
) -> float:
    """Evaluate a saturated-liquid density ancillary (for the rho_max bound)."""
    import math

    reducing, using_tau_r, noexp, coeffs = anc
    theta = 1.0 - t / tc
    total = sum(n * theta**texp for n, texp in coeffs)
    if noexp:
        return reducing * (1.0 + total)
    factor = tc / t if using_tau_r else 1.0
    return reducing * math.exp(factor * total)


def build() -> None:
    fluids: dict[str, dict[str, Any]] = {}
    for cp_name, name in FLUIDS.items():
        doc = json.loads(get_fluid_param_string(cp_name, "JSON"))[0]
        eos = doc["EOS"][0]
        info = doc["INFO"]
        anc = doc["ANCILLARIES"]
        reducing = eos["STATES"]["reducing"]
        critical = doc["STATES"]["critical"]

        t_reducing = float(reducing["T"])
        entry: dict[str, Any] = {
            "coolprop_name": cp_name,
            "cas": str(info["CAS"]),
            "bibtex_eos": str(eos.get("BibTeX_EOS", "")),
            "molar_mass": float(eos["molar_mass"]),
            "gas_constant": float(eos["gas_constant"]),
            "t_reducing": t_reducing,
            "rho_reducing": float(reducing["rhomolar"]),
            "t_critical": float(critical["T"]),
            "p_critical": float(critical["p"]),
            "rho_critical": float(critical["rhomolar"]),
            "t_triple": float(eos["Ttriple"]),
            "p_triple": float(doc["STATES"]["triple_liquid"]["p"]),
            "t_max": float(eos["T_max"]),
            "p_max": float(eos["p_max"]),
            "acentric": float(eos["acentric"]),
        }
        entry.update(_ideal_terms(eos["alpha0"], t_reducing))
        entry.update(_residual_terms(eos["alphar"]))
        entry["anc_psat"] = _ancillary(anc["pS"])
        entry["anc_rho_liquid"] = _ancillary(anc["rhoL"])
        entry["anc_rho_vapor"] = _ancillary(anc["rhoV"])
        # Density search bound: comfortably above the densest liquid state the
        # EOS is published for (the saturated liquid at the triple point).
        rho_triple = _anc_rho_liquid(entry["anc_rho_liquid"], entry["t_triple"], t_reducing)
        entry["rho_max"] = 1.5 * max(rho_triple, 3.0 * entry["rho_reducing"])

        sigma = anc.get("surface_tension")
        if sigma is None:
            raise ValueError(f"{cp_name} ships no surface-tension correlation")
        entry["sigma_tc"] = float(sigma["Tc"])
        entry["sigma"] = tuple(
            (float(a), float(n)) for a, n in zip(sigma["a"], sigma["n"], strict=True)
        )
        entry["bibtex_sigma"] = str(sigma.get("BibTeX", ""))

        fluids[name] = entry

    _write(fluids)
    n_terms = sum(
        len(f["power"]) + len(f["gaussian"]) + len(f["nonanalytic"]) + len(f["gaob"])
        for f in fluids.values()
    )
    print(f"Helmholtz data: {len(fluids)} fluids, {n_terms} residual terms.")


def _format(value: Any, indent: int) -> str:
    pad = " " * indent
    if isinstance(value, tuple):
        if not value:
            return "()"
        if isinstance(value[0], tuple):
            rows = ",\n".join(pad + "    " + repr(row) for row in value)
            return f"(\n{rows},\n{pad})"
        return repr(value)
    return repr(value)


def _write(fluids: dict[str, dict[str, Any]]) -> None:
    blocks: list[str] = []
    for name, entry in fluids.items():
        lines = [f"    {name!r}: {{"]
        for key, value in entry.items():
            lines.append(f"        {key!r}: {_format(value, 8)},")
        lines.append("    },")
        blocks.append("\n".join(lines))
    body = "\n".join(blocks)
    src = f'''"""Reference multiparameter Helmholtz-EOS coefficients (generated).

Generated by ``scripts/gen_helmholtz.py``. Each entry holds the published
reference equation of state for one pure fluid in the reduced-Helmholtz form

    alpha(delta, tau) = alpha0(delta, tau) + alphar(delta, tau),

with ``delta = rho/rho_reducing`` and ``tau = t_reducing/T``, plus the
saturation ancillary equations (initial guesses for the Maxwell solve) and the
surface-tension correlation. Coefficients are extracted from CoolProp's open
fluid library, which transcribes the peer-reviewed source formulations; the
source paper of each EOS is recorded in ``bibtex_eos`` (e.g. IAPWS-95 for
water is ``Wagner-JPCRD-2002``). The per-fluid ``gas_constant`` is the value
the EOS was fitted with and must be used in place of the CODATA constant when
evaluating it. Do not edit by hand; regenerate instead.
"""

from __future__ import annotations

from typing import Any

FLUID_DATA: dict[str, dict[str, Any]] = {{
{body}
}}
'''
    with open(DATA_PY, "w") as f:
        f.write(src)


if __name__ == "__main__":
    build()
