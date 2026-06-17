"""Differential tests: the problem table algorithm vs an independent utility LP.

Opt-in *oracle* tests (marker: ``oracle``) that pit Fugacio's analytic problem
table cascade against a completely different formulation of the same target: the
Papoulias-Grossmann **transshipment linear program** solved by SciPy's
``linprog``. The LP minimises the hot utility subject to a non-negative heat
cascade through the temperature intervals; its optimum is, by construction, the
minimum-utility target. Agreement across a battery of random stream sets (and the
textbook four-stream problem) validates the targeting independently of the
in-house cascade arithmetic. SciPy ships as a JAX dependency, so the reference is
always importable.
"""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest

from fugacio.sim.integration import make_stream, pinch_analysis

HAVE_SCIPY = find_spec("scipy") is not None

pytestmark = [
    pytest.mark.oracle,
    pytest.mark.skipif(not HAVE_SCIPY, reason="scipy not installed"),
]


def _linprog_utilities(streams: list, dt_min: float) -> tuple[float, float]:
    """Minimum (hot, cold) utility from the transshipment LP via ``scipy.linprog``.

    Variables are the cascaded residual heat flows ``R_0..R_K`` (``R_0`` is the
    hot utility fed at the top, ``R_K`` the cold utility leaving the bottom), all
    non-negative; the equalities ``R_i - R_{i-1} = D_i`` enforce the interval heat
    balance, and the objective minimises ``R_0``.
    """
    from scipy.optimize import linprog

    half = 0.5 * dt_min
    lo, hi, signed = [], [], []
    for s in streams:
        ts, tt = float(s.t_supply), float(s.t_target)
        cp = float(s.cp)
        is_hot = ts > tt
        shift = -half if is_hot else half
        a, b = ts + shift, tt + shift
        lo.append(min(a, b))
        hi.append(max(a, b))
        signed.append(cp if is_hot else -cp)
    lo_a, hi_a, signed_a = np.array(lo), np.array(hi), np.array(signed)

    boundaries = np.unique(np.concatenate([lo_a, hi_a]))[::-1]  # descending
    upper = boundaries[:-1]
    lower = boundaries[1:]
    mid = 0.5 * (upper + lower)
    present = (lo_a[None, :] <= mid[:, None]) & (mid[:, None] <= hi_a[None, :])
    interval_cp = (present * signed_a[None, :]).sum(axis=1)
    d = interval_cp * (upper - lower)  # interval surplus, shape (K,)

    k = d.shape[0]
    n = k + 1  # residuals R_0..R_K
    c = np.zeros(n)
    c[0] = 1.0  # minimise the hot utility R_0
    a_eq = np.zeros((k, n))
    for i in range(k):
        a_eq[i, i] = -1.0
        a_eq[i, i + 1] = 1.0
    res = linprog(c, A_eq=a_eq, b_eq=d, bounds=[(0.0, None)] * n, method="highs")
    assert res.success
    return float(res.x[0]), float(res.x[-1])


def _check(streams: list, dt_min: float) -> None:
    res = pinch_analysis(streams, dt_min)
    q_h, q_c = _linprog_utilities(streams, dt_min)
    assert float(res.hot_utility) == pytest.approx(q_h, abs=1e-6, rel=1e-6)
    assert float(res.cold_utility) == pytest.approx(q_c, abs=1e-6, rel=1e-6)


def test_linprog_matches_four_stream() -> None:
    streams = [
        make_stream(20.0, 135.0, 2.0, name="C1"),
        make_stream(170.0, 60.0, 3.0, name="H1"),
        make_stream(80.0, 140.0, 4.0, name="C2"),
        make_stream(150.0, 30.0, 1.5, name="H2"),
    ]
    _check(streams, 10.0)


def test_linprog_matches_random_problems() -> None:
    rng = np.random.default_rng(0)
    for _ in range(25):
        n_hot = int(rng.integers(1, 4))
        n_cold = int(rng.integers(1, 4))
        streams = []
        for i in range(n_hot):
            t_hi = float(rng.uniform(150.0, 320.0))
            t_lo = float(rng.uniform(20.0, t_hi - 10.0))
            streams.append(make_stream(t_hi, t_lo, float(rng.uniform(0.5, 5.0)), name=f"H{i}"))
        for i in range(n_cold):
            t_lo = float(rng.uniform(20.0, 250.0))
            t_hi = float(rng.uniform(t_lo + 10.0, 340.0))
            streams.append(make_stream(t_lo, t_hi, float(rng.uniform(0.5, 5.0)), name=f"C{i}"))
        _check(streams, float(rng.uniform(5.0, 25.0)))
