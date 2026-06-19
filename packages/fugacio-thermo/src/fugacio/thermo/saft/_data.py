"""Pure-component PC-SAFT parameters and curated binary corrections.

The three pure-component parameters of non-associating PC-SAFT, the segment
number ``m``, the segment diameter ``sigma`` (angstrom), and the dispersion
energy ``epsilon/k`` (kelvin), are tabulated by Gross & Sadowski, "Perturbed-Chain
SAFT: An Equation of State Based on a Perturbation Theory for Chain Molecules",
*Ind. Eng. Chem. Res.* **2001**, 40, 1244. Associating species carry two extra
parameters, the association energy ``epsilon_AB/k`` (kelvin) and the dimensionless
association volume ``kappa_AB``, with the assignment of the Huang-Radosz scheme;
those are from Gross & Sadowski, "Application of the Perturbed-Chain SAFT Equation
of State to Associating Systems", *Ind. Eng. Chem. Res.* **2002**, 41, 5510.

Each entry maps a Fugacio component name to a `PureSaft` record. ``sigma`` is in
angstrom here (the unit the literature tabulates); `fugacio.thermo.saft.parameters`
converts it to metres when it builds the differentiable parameter pytree. The
``scheme`` is a Huang-Radosz association label (``"none"``, ``"2B"``, ``"3B"``,
``"4C"``) that fixes the number of electron-acceptor (type A) and electron-donor
(type B) sites; only A-B bonding is modelled (see
`fugacio.thermo.saft.association`).
"""

from __future__ import annotations

from dataclasses import dataclass

#: Number of (acceptor, donor) association sites for each Huang-Radosz scheme.
ASSOCIATION_SITES: dict[str, tuple[int, int]] = {
    "none": (0, 0),
    "1A": (1, 0),
    "2B": (1, 1),
    "3B": (2, 1),
    "4C": (2, 2),
}


@dataclass(frozen=True)
class PureSaft:
    """Pure-component PC-SAFT parameters as tabulated in the literature.

    Attributes:
        m: Segment number (dimensionless).
        sigma: Temperature-independent segment diameter (angstrom).
        epsilon: Dispersion energy ``epsilon/k`` (kelvin).
        scheme: Huang-Radosz association scheme (``"none"``, ``"2B"``, ...).
        kappa_ab: Dimensionless association volume (0 for non-associating).
        epsilon_ab: Association energy ``epsilon_AB/k`` (kelvin; 0 if none).
        source: Short provenance tag for the parameter row.
    """

    m: float
    sigma: float
    epsilon: float
    scheme: str = "none"
    kappa_ab: float = 0.0
    epsilon_ab: float = 0.0
    source: str = "GS2001"


def _p(m: float, sigma: float, epsilon: float) -> PureSaft:
    return PureSaft(m=m, sigma=sigma, epsilon=epsilon)


def _a(
    m: float, sigma: float, epsilon: float, scheme: str, kappa_ab: float, epsilon_ab: float
) -> PureSaft:
    return PureSaft(
        m=m,
        sigma=sigma,
        epsilon=epsilon,
        scheme=scheme,
        kappa_ab=kappa_ab,
        epsilon_ab=epsilon_ab,
        source="GS2002",
    )


#: Curated pure-component PC-SAFT parameters keyed by Fugacio component name.
PURE_SAFT_PARAMS: dict[str, PureSaft] = {
    # Non-associating: n-alkanes (Gross & Sadowski 2001, Table 1).
    "methane": _p(1.0000, 3.7039, 150.03),
    "ethane": _p(1.6069, 3.5206, 191.42),
    "propane": _p(2.0020, 3.6184, 208.11),
    "n-butane": _p(2.3316, 3.7086, 222.88),
    "isobutane": _p(2.2616, 3.7574, 216.53),
    "n-pentane": _p(2.6896, 3.7729, 231.20),
    "n-hexane": _p(3.0576, 3.7983, 236.77),
    "n-heptane": _p(3.4831, 3.8049, 238.40),
    "n-octane": _p(3.8176, 3.8373, 242.78),
    "n-nonane": _p(4.2079, 3.8448, 244.51),
    "n-decane": _p(4.6627, 3.8384, 243.87),
    # Non-associating: gases and ring/unsaturated species.
    "nitrogen": _p(1.2053, 3.3130, 90.96),
    "oxygen": _p(1.1217, 3.2098, 114.96),
    "carbon dioxide": _p(2.0729, 2.7852, 169.21),
    "carbon monoxide": _p(1.3097, 3.2507, 92.150),
    "ethylene": _p(1.5930, 3.4450, 176.47),
    "propylene": _p(1.9597, 3.5356, 207.19),
    "benzene": _p(2.4653, 3.6478, 287.35),
    "toluene": _p(2.8149, 3.7169, 285.69),
    "cyclohexane": _p(2.5303, 3.8499, 278.11),
    # Associating (Gross & Sadowski 2002, Table 1): water and 1-alcohols, 2B scheme.
    "water": _a(1.0656, 3.0007, 366.51, "2B", 0.034868, 2500.7),
    "methanol": _a(1.5255, 3.2300, 188.90, "2B", 0.035176, 2899.5),
    "ethanol": _a(2.3827, 3.1771, 198.24, "2B", 0.032384, 2653.4),
    "1-propanol": _a(2.9997, 3.2522, 233.40, "2B", 0.015268, 2276.8),
    "1-butanol": _a(2.7515, 3.6139, 259.59, "2B", 0.006692, 2544.6),
}


#: Curated symmetric PC-SAFT binary correction parameters ``k_ij`` (dispersion).
#:
#: Keyed by an alphabetically sorted name pair. Pairs absent here default to
#: ``k_ij = 0`` (Berthelot geometric mean). Values are representative literature
#: fits and are intended as sensible defaults, not a comprehensive bank.
SAFT_KIJ: dict[tuple[str, str], float] = {
    ("carbon dioxide", "methane"): 0.0650,
    ("carbon dioxide", "n-butane"): 0.1150,
    ("carbon dioxide", "propane"): 0.1135,
    ("methane", "n-butane"): 0.0220,
    ("ethane", "n-heptane"): 0.0070,
    ("methanol", "water"): -0.0660,
}


def saft_kij(name_i: str, name_j: str) -> float | None:
    """Curated PC-SAFT binary correction ``k_ij`` for a pair, or ``None``.

    Order-insensitive: ``k_ij`` is symmetric for the standard one-fluid mixing
    rule of the dispersion term.

    Args:
        name_i: First component name.
        name_j: Second component name.

    Returns:
        The curated ``k_ij`` if the pair is tabulated, else ``None``.
    """
    key = (name_i, name_j) if name_i <= name_j else (name_j, name_i)
    return SAFT_KIJ.get(key)


def has_saft_params(name: str) -> bool:
    """Whether curated PC-SAFT parameters exist for a component name."""
    return name.strip().lower() in PURE_SAFT_PARAMS


__all__ = [
    "ASSOCIATION_SITES",
    "PURE_SAFT_PARAMS",
    "SAFT_KIJ",
    "PureSaft",
    "has_saft_params",
    "saft_kij",
]
