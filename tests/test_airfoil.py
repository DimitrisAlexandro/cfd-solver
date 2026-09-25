"""Regression test for solver step A (laminar airfoil). Full study: validate_airfoil.py"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import validate_airfoil as va   # noqa: E402


@pytest.fixture(scope="module")
def coarse_a2():
    return va.run(re=1000, alpha=2, refine=0.7)


def test_laminar_naca0012_vs_xfoil(coarse_a2):
    r = coarse_a2
    xp = va.xfoil_polar(1000)
    x = xp[xp[:, 0] == 2][0]                     # XFOIL: alpha, Cl, Cd, Cm
    assert r["converged"]
    # coarse test grid (fast): its Cl is ~5 % above the grid-converged value;
    # the validated medium/fine grids agree with XFOIL within 1 % (validate_airfoil.py)
    assert r["Cl"] == pytest.approx(x[1], rel=0.08)
    assert r["Cd"] == pytest.approx(0.119, rel=0.04)   # literature ~0.119-0.12 near alpha 0-2
    assert r["x_sep"] is None or r["x_sep"] > 0.9       # attached (or tiny TE bubble) at 2 deg


def test_symmetric_airfoil_zero_lift():
    r = va.run(re=1000, alpha=0, refine=0.6)
    assert abs(r["Cl"]) < 2e-3
    up = np.array(r["wall"]["side"]) == 1
    cp = np.array(r["wall"]["Cp"])
    assert np.isclose(cp[up].max(), cp[~up].max(), atol=0.02)   # symmetric stagnation
