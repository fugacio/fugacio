"""Generate UNIFAC group tables + curated binary parameters from open datasets.

A one-shot authoring tool (not part of the package). It pulls, for every species
in the Fugacio component database that can be resolved:

* its UNIFAC subgroup assignment (DDBST, via the molecule's InChI key), the
  subgroup ``R_k``/``Q_k`` (UNIFAC ``UFSG``) and main-group interaction matrix
  (``UFIP``) -- written to ``groupcontrib/_unifac_data.py``;
* the UNIQUAC pure-component ``r``/``q`` derived from the same group assignment
  (``r_i = sum nu_k R_k``, ``q_i = sum nu_k Q_k``); and
* curated binary NRTL and UNIQUAC interaction coefficients from the open ChemSep
  database (``thermo.interaction_parameters``) -- both written to
  ``_binary_params.py``.

The unit conventions were verified to reproduce ``thermo`` exactly:
NRTL ``tau_ij = b_ij/T`` with ``a = 0``; UNIQUAC ``tau_ij = exp(b_ij/T)`` with the
group-derived ``r``/``q``. Run with ``uv run python scripts/gen_parameters.py``
(requires the optional ``chemicals`` and ``thermo`` packages).
"""

from __future__ import annotations

import chemicals as ch
import thermo.unifac as u
from thermo import interaction_parameters as ipm

u.load_group_assignments_DDBST()
u.load_unifac_ip()
IPDB = ipm.IPDB

UNIFAC_DATA_PY = "packages/fugacio-thermo/src/fugacio/thermo/groupcontrib/_unifac_data.py"
DORTMUND_DATA_PY = "packages/fugacio-thermo/src/fugacio/thermo/groupcontrib/_dortmund_data.py"
BINARY_PY = "packages/fugacio-thermo/src/fugacio/thermo/_binary_params.py"


def inchikey(cas: str) -> str | None:
    try:
        return ch.search_chemical(cas).InChI_key
    except Exception:
        return None


def build() -> None:
    import fugacio.thermo.components as comp

    db = comp.DATABASE
    names = sorted(db)

    # 1. UNIFAC group assignments per component (classic UNIFAC, via DDBST).
    assignments: dict[str, dict[int, int]] = {}
    cas_of: dict[str, str] = {}
    for name in names:
        c = db[name]
        if not c.cas:
            continue
        ik = inchikey(c.cas)
        if ik is None:
            continue
        groups = u.DDBST_UNIFAC_assignments.get(ik)
        if groups:
            assignments[name] = {int(k): int(v) for k, v in groups.items()}
            cas_of[name] = c.cas

    used_subgroups = sorted({sg for g in assignments.values() for sg in g})
    subgroups = {}
    for sg in used_subgroups:
        obj = u.UFSG[sg]
        subgroups[sg] = (obj.group, int(obj.main_group_id), float(obj.R), float(obj.Q))

    used_main = sorted({mg for _, mg, _, _ in subgroups.values()})
    interactions: dict[tuple[int, int], float] = {}
    for m in used_main:
        for n in used_main:
            if m == n:
                continue
            try:
                val = u.UFIP[m][n]
            except KeyError:
                continue
            if val is not None:
                interactions[(m, n)] = float(val)

    # 2. UNIQUAC r/q from the group assignment.
    rq: dict[str, tuple[float, float]] = {}
    for name, groups in assignments.items():
        r = sum(count * subgroups[sg][2] for sg, count in groups.items())
        q = sum(count * subgroups[sg][3] for sg, count in groups.items())
        rq[name] = (round(r, 4), round(q, 4))

    # 3. Curated binary NRTL / UNIQUAC parameters from ChemSep.
    nrtl: dict[tuple[str, str], tuple[float, float, float]] = {}
    uniquac: dict[tuple[str, str], tuple[float, float]] = {}
    named = [n for n in names if db[n].cas]
    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            a, b = named[i], named[j]
            cas_pair = [db[a].cas, db[b].cas]
            if IPDB.has_ip_specific("ChemSep NRTL", cas_pair, "bij"):
                bm = IPDB.get_ip_asymmetric_matrix("ChemSep NRTL", cas_pair, "bij")
                am = IPDB.get_ip_asymmetric_matrix("ChemSep NRTL", cas_pair, "alphaij")
                nrtl[(a, b)] = (round(bm[0][1], 4), round(bm[1][0], 4), round(am[0][1], 4))
            if IPDB.has_ip_specific("ChemSep UNIQUAC", cas_pair, "bij"):
                bm = IPDB.get_ip_asymmetric_matrix("ChemSep UNIQUAC", cas_pair, "bij")
                uniquac[(a, b)] = (round(bm[0][1], 4), round(bm[1][0], 4))

    # 4. Modified UNIFAC (Dortmund): subgroups, T-dependent interactions, assignments.
    do_assignments: dict[str, dict[int, int]] = {}
    for name in names:
        c = db[name]
        if not c.cas:
            continue
        ik = inchikey(c.cas)
        if ik is None:
            continue
        groups = u.DDBST_MODIFIED_UNIFAC_assignments.get(ik)
        if groups:
            do_assignments[name] = {int(k): int(v) for k, v in groups.items()}

    do_used_sg = sorted({sg for g in do_assignments.values() for sg in g})
    do_subgroups = {}
    for sg in do_used_sg:
        obj = u.DOUFSG[sg]
        do_subgroups[sg] = (obj.group, int(obj.main_group_id), float(obj.R), float(obj.Q))

    do_used_main = sorted({mg for _, mg, _, _ in do_subgroups.values()})
    do_interactions: dict[tuple[int, int], tuple[float, float, float]] = {}
    for m in do_used_main:
        for n in do_used_main:
            if m == n:
                continue
            try:
                val = u.DOUFIP2006[m][n]
            except KeyError:
                continue
            a, b, cc = (float(v) for v in val)
            do_interactions[(m, n)] = (round(a, 5), round(b, 6), round(cc, 9))

    _write_unifac(subgroups, interactions, assignments)
    _write_dortmund(do_subgroups, do_interactions, do_assignments)
    _write_binary(rq, nrtl, uniquac)
    print(
        f"UNIFAC: {len(assignments)} components, {len(subgroups)} subgroups, "
        f"{len(interactions)} interaction pairs."
    )
    print(
        f"Dortmund: {len(do_assignments)} components, {len(do_subgroups)} subgroups, "
        f"{len(do_interactions)} interaction pairs."
    )
    print(f"UNIQUAC r/q: {len(rq)} components.")
    print(f"Binary params: {len(nrtl)} NRTL pairs, {len(uniquac)} UNIQUAC pairs.")


def _fmt_dict(items: list[str], indent: str = "    ") -> str:
    return "\n".join(indent + line for line in items)


def _write_unifac(
    subgroups: dict[int, tuple[str, int, float, float]],
    interactions: dict[tuple[int, int], float],
    assignments: dict[str, dict[int, int]],
) -> None:
    sg_lines = [f"{sg}: ({name!r}, {mg}, {r}, {q})," for sg, (name, mg, r, q) in subgroups.items()]
    ip_lines = [f"({m}, {n}): {v}," for (m, n), v in interactions.items()]
    cg_lines = [
        f"{name!r}: {{{', '.join(f'{sg}: {c}' for sg, c in groups.items())}}},"
        for name, groups in sorted(assignments.items())
    ]
    src = f'''"""UNIFAC group tables, generated by ``scripts/gen_parameters.py``.

Subgroup ``R_k``/``Q_k`` and main-group interaction energies are the public
UNIFAC (Hansen VLE) parameters bundled in the open ``thermo`` package; the
component-to-subgroup assignments are from the DDBST public set. Do not edit by
hand -- regenerate instead.
"""

from __future__ import annotations

# subgroup id -> (name, main-group id, R_k, Q_k)
SUBGROUPS: dict[int, tuple[str, int, float, float]] = {{
{_fmt_dict(sg_lines)}
}}

# main-group interaction parameters a_mn (Kelvin); missing pairs default to 0.
INTERACTIONS: dict[tuple[int, int], float] = {{
{_fmt_dict(ip_lines)}
}}

# component name -> {{subgroup id: count}}
COMPONENT_GROUPS: dict[str, dict[int, int]] = {{
{_fmt_dict(cg_lines)}
}}
'''
    with open(UNIFAC_DATA_PY, "w") as f:
        f.write(src)


def _write_dortmund(
    subgroups: dict[int, tuple[str, int, float, float]],
    interactions: dict[tuple[int, int], tuple[float, float, float]],
    assignments: dict[str, dict[int, int]],
) -> None:
    sg_lines = [f"{sg}: ({name!r}, {mg}, {r}, {q})," for sg, (name, mg, r, q) in subgroups.items()]
    ip_lines = [f"({m}, {n}): ({a}, {b}, {c})," for (m, n), (a, b, c) in interactions.items()]
    cg_lines = [
        f"{name!r}: {{{', '.join(f'{sg}: {c}' for sg, c in groups.items())}}},"
        for name, groups in sorted(assignments.items())
    ]
    src = f'''"""Modified UNIFAC (Dortmund) group tables (generated by gen_parameters.py).

Dortmund subgroup ``R_k``/``Q_k`` and the temperature-dependent main-group
interaction parameters ``(a_mn, b_mn, c_mn)`` (psi = exp(-(a + b T + c T^2)/T))
are the published modified-UNIFAC values bundled in the open ``thermo`` package
(``DOUFSG`` / ``DOUFIP2006``); assignments are the DDBST modified-UNIFAC set. Do
not edit by hand -- regenerate instead.
"""

from __future__ import annotations

# subgroup id -> (name, main-group id, R_k, Q_k)
DO_SUBGROUPS: dict[int, tuple[str, int, float, float]] = {{
{_fmt_dict(sg_lines)}
}}

# main-group interaction parameters (a_mn, b_mn, c_mn); missing pairs default to 0.
DO_INTERACTIONS: dict[tuple[int, int], tuple[float, float, float]] = {{
{_fmt_dict(ip_lines)}
}}

# component name -> {{subgroup id: count}}
DO_COMPONENT_GROUPS: dict[str, dict[int, int]] = {{
{_fmt_dict(cg_lines)}
}}
'''
    with open(DORTMUND_DATA_PY, "w") as f:
        f.write(src)


def _write_binary(
    rq: dict[str, tuple[float, float]],
    nrtl: dict[tuple[str, str], tuple[float, float, float]],
    uniquac: dict[tuple[str, str], tuple[float, float]],
) -> None:
    rq_lines = [f"{name!r}: ({r}, {q})," for name, (r, q) in sorted(rq.items())]
    nrtl_lines = [
        f"({a!r}, {b!r}): ({bij}, {bji}, {al})," for (a, b), (bij, bji, al) in sorted(nrtl.items())
    ]
    uq_lines = [
        f"({a!r}, {b!r}): ({bij}, {bji})," for (a, b), (bij, bji) in sorted(uniquac.items())
    ]
    src = f'''"""Curated UNIQUAC r/q and binary NRTL / UNIQUAC parameters (generated).

Generated by ``scripts/gen_parameters.py``. The ``r``/``q`` come from the UNIFAC
group assignment (``r_i = sum nu_k R_k``, ``q_i = sum nu_k Q_k``). The binary
interaction coefficients are from the open ChemSep database; both use the
``tau_ij`` conventions of :mod:`fugacio.thermo.activity.models`
(NRTL ``tau = b/T``; UNIQUAC ``tau = exp(b/T)``), keyed by name pairs sorted
alphabetically. Do not edit by hand -- regenerate instead.
"""

from __future__ import annotations

# component name -> (r, q)
UNIQUAC_RQ: dict[str, tuple[float, float]] = {{
{_fmt_dict(rq_lines)}
}}

# (name_i, name_j) sorted -> (b_ij, b_ji, alpha_ij) for NRTL, tau_ij = b_ij / T.
NRTL_BINARY: dict[tuple[str, str], tuple[float, float, float]] = {{
{_fmt_dict(nrtl_lines)}
}}

# (name_i, name_j) sorted -> (b_ij, b_ji) for UNIQUAC, tau_ij = exp(b_ij / T).
UNIQUAC_BINARY: dict[tuple[str, str], tuple[float, float]] = {{
{_fmt_dict(uq_lines)}
}}
'''
    with open(BINARY_PY, "w") as f:
        f.write(src)


if __name__ == "__main__":
    build()
