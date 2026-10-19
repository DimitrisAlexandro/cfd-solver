"""Low-Reynolds-number mode: regime switch, bubble transition length, trip."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerolab.airfoil import polar as pl  # noqa: E402
from aerolab.physics import transition as T  # noqa: E402


def test_regime_switch():
    assert pl.uses_rans(3e4) and pl.uses_rans(1e5) and pl.uses_rans(3e6)
    assert not pl.uses_rans(1e4) and not pl.uses_rans(1000)
    assert not pl.uses_rans(5e4, "laminar") and pl.uses_rans(5e3, "rans")


def test_bubble_length_models():
    re, ue = 2e5, 1.3
    # Horton: constant Re_l = 4e4
    assert T.bubble_length(re, ue, 1e-3, "horton") == pytest.approx(4e4 / (re * ue))
    # Mayle: Re_l = K Re_theta^0.7
    th = 6e-4
    ret = re * ue * th
    assert T.bubble_length(re, ue, th, "mayle") == pytest.approx(T.BUBBLE_K * ret**0.7 / (re * ue))
    assert T.bubble_length(re, ue, th, "mayle:300") == pytest.approx(300 * ret**0.7 / (re * ue))
    # Mayle's long-bubble length scales with the boundary layer: shorter than
    # Horton's at low Re, about equal near Re_theta,s 200 (the low-Re mode limit)
    lo = T.bubble_length(1e5, 1.3, 1.1e-3, "mayle")
    assert lo < T.bubble_length(1e5, 1.3, 1.1e-3, "horton")
    r200 = 200.0
    assert 1000 * r200**0.7 == pytest.approx(4e4, rel=0.05)
    # the low-Re default: the longer of the two (leading-edge bubbles: Horton)
    assert T.bubble_length(2e5, 1.8, 1e-4, "max") == pytest.approx(T.bubble_length(2e5, 1.8, 1e-4, "horton"))
    assert T.bubble_length(1e5, 1.3, 3e-3, "max") == pytest.approx(T.bubble_length(1e5, 1.3, 3e-3, "mayle"))
    # no theta (nan): a finite length
    assert math.isfinite(T.bubble_length(1e5, 1.2, float("nan"), "mayle"))


def _case(re, alpha=4.0):
    from aerolab.core.simple_solver import SimpleSolver
    fvm, mesh, patches, bcs, U, turb, info = pl.make_case(str(Path(__file__).resolve().parents[1] /
                                                              "validation" / "reference" / "e387.dat"),
                                                          re, alpha, level=0.6)
    s = SimpleSolver(fvm, bcs, 1.0 / re, U0=U, turbulence=turb)
    return s, mesh


@pytest.mark.parametrize("re, expect", [(1e5, True), (2e6, False)])
def test_trip_only_in_low_re_mode(re, expect):
    """the trip seeds the shear layer behind a separation-induced transition,
    only in the low-Re mode; N_crit transitions and high Re are untouched"""
    s, mesh = _case(re)
    tr = T.Transition(s, mesh, re, mode="free", x_tr={"upper": 0.5, "lower": 1.0})
    tr.est = {"upper": {"how": "laminar separation"}, "lower": {"how": "laminar to the trailing edge"}}
    tr.apply()
    if expect:
        idx, val = s.turb.seed
        assert len(idx) > 0 and val == pytest.approx(T.SEED_CHI / re)
        xc = s.m.xc[idx]
        assert np.all(xc[:, 0] >= 0.5 - 0.02) and np.all(xc[:, 0] <= 0.5 + T.SEED_LENGTH + 0.02)
        assert np.all(xc[:, 1] > -0.01)                       # upper surface only
        assert np.all(s.turb.gamma[idx] > 0.5)                # never inside the laminar region
        # the floor holds after an SA solve
        s.turb.solve()
        assert np.all(s.turb.nuT[idx] >= val * (1 - 1e-5))        # imposed implicitly (penalty)
    else:
        assert s.turb.seed is None
    tr.est = {"upper": {"how": "N_crit"}, "lower": {"how": "N_crit"}}
    tr.apply()
    assert s.turb.seed is None
    # a leading-edge bubble is not tripped
    tr.est = {"upper": {"how": "laminar separation", "x_sep": 0.05}, "lower": {"how": "N_crit"}}
    tr.apply()
    assert s.turb.seed is None
