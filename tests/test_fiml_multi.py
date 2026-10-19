"""Measured data behind the multi-airfoil field inversion."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerolab.ml import fiml_multi as F  # noqa: E402


def test_data_loaders():
    x, cp, side = F.coles_wadcock_cp()
    assert len(x) == 51 and set(side) == {1, -1}
    assert cp[side == 1].min() < -5 and cp[side == -1].min() > -1.5   # suction side up
    assert abs(F.s809_osu(8.1) - 0.90) < 1e-9 and abs(F.s809_osu(14.2) - 1.01) < 1e-9
    xg, cg, sg = F.gregory_cp(15)
    assert (sg == 1).all() and cg.min() < -8
    assert abs(F.gregory_cl(15.0) - 1.505) < 0.01


def test_s809_geometry():
    d = np.loadtxt(F.S809, skiprows=4)
    le = np.argmin(d[:, 0])
    xs = np.linspace(0.02, 0.98, 300)
    up = np.interp(xs, d[:le + 1, 0][::-1], d[:le + 1, 1][::-1])
    lo = np.interp(xs, d[le:, 0], d[le:, 1])
    t = up - lo
    assert abs(t.max() - 0.21) < 0.003 and 0.33 < xs[np.argmax(t)] < 0.43


def test_cases_have_data():
    for c in F.training_cases():
        assert F.target_cl(c) is not None or F.target_cp(c) is not None
    assert {c["family"] for c in F.training_cases()} == set(F.test_points())
