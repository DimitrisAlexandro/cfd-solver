"""Tests for the e^N transition model (Phase 3, E4)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import transition as T  # noqa: E402


def test_envelope_blasius_e9():
    """Classic e^9 result for the Blasius boundary layer: Re_x,tr ~ 3e6."""
    re = 1e6
    x = np.linspace(1e-4, 10, 100000)
    th = 0.664 * x / np.sqrt(re * x)
    N, _ = T.integrate_N(x, np.full_like(x, 2.591), re * th, th)
    rex = re * x[np.argmax(N >= 9)]
    assert 2.5e6 < rex < 3.6e6
    assert np.all(np.diff(N) >= 0)                          # N never decreases


def test_laminar_bl_classic_solutions():
    re = 1e6
    s = np.linspace(0, 2, 400)
    th, H, _, sep = T.laminar_bl(s, np.ones_like(s), re)         # Blasius
    k = 200
    assert sep is None
    assert H[k] == pytest.approx(2.59, abs=0.03)
    assert th[k] == pytest.approx(0.664 * s[k] / np.sqrt(re * s[k]), rel=0.01)
    s = np.linspace(0, 1.2, 1200)                                 # Howarth: Ue = 1 - x/8
    th, H, _, sep = T.laminar_bl(s, 1 - s / 8, re)
    assert s[sep] / 8 == pytest.approx(0.1199, rel=0.05)
    s = np.r_[0, np.geomspace(1e-5, 0.5, 400)]                    # Hiemenz stagnation
    th, H, _, sep = T.laminar_bl(s, 4 * s, re)
    assert H[200] == pytest.approx(2.216, abs=0.05)
    assert th[200] * np.sqrt(re * 4) == pytest.approx(0.2923, rel=0.02)


def test_reproduces_xfoil_on_xfoil_pressure():
    """Same pressure distribution -> same transition point as XFOIL."""
    pytest.importorskip("xfoil")
    import ctypes
    from xfoil import XFoil
    from xfoil.model import Airfoil
    import airfoil_geom as ag
    xf = XFoil()
    xf.print = False
    loop, _ = ag.load_airfoil("naca0012", n_side=120, n_te=2)
    up = int(np.argmax(np.where(loop[:, 0] > 0.999, loop[:, 1], -1)))
    lo = int(np.argmin(np.where(loop[:, 0] > 0.999, loop[:, 1], 1)))
    pts = np.roll(loop, -up, axis=0)[:(lo - up) % len(loop) + 1]
    xf.airfoil = Airfoil(pts[:, 0].copy(), pts[:, 1].copy())
    xf.repanel(n_nodes=240)
    xf.Re = 3e6
    xf.max_iter = 200
    for a in (0.0, 4.0):
        xf.a(a)
        x, y, cp = xf.get_cp_distribution()
        xt = (ctypes.c_double * 2).in_dll(xf._lib, "__i_xfoil_MOD_xoctr")
        ist = np.argmax(cp)
        for side, (xs, ys, cs) in enumerate(((x[:ist + 1][::-1], y[:ist + 1][::-1], cp[:ist + 1][::-1]),
                                             (x[ist:], y[ist:], cp[ist:]))):
            s = np.r_[0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
            th, H, ret, sep = T.laminar_bl(s, np.sqrt(np.maximum(1 - cs, 1e-9)), 3e6)
            end = sep if sep else len(th)
            N, _ = T.integrate_N(s[:end], H[:end], ret[:end], th[:end])
            assert (N >= 9).any()
            assert xs[np.argmax(N >= 9)] == pytest.approx(xt[side], abs=0.02)


def test_rans_profile_extraction_blasius():
    import validate_flatplate as vf
    from simple_solver import SimpleSolver
    lv = 0.5
    m = vf.plate_mesh(lv)
    nx_up, nx_pl, ny = int(24 * lv), int(112 * lv), int(96 * lv)
    s = SimpleSolver(m, vf.BCS, 1 / vf.RE, U0=(1.0, 0.0))
    s.solve(tol=1e-8, max_iter=3000, verbose=False)
    ii = np.arange(nx_up, nx_up + nx_pl)
    cells = np.arange(ny)[:, None] + ii[None, :] * ny
    xw = np.c_[m.xc[ii * ny, 0], np.zeros(len(ii))]
    cols = T.structured_columns(xw, np.tile([1.0, 0], (len(ii), 1)), np.tile([0, 1.0], (len(ii), 1)), cells)
    bl = T.extract(s, cols, vf.RE)
    k = np.argmin(abs(bl["x"] - 1.0))
    assert bl["H"][k] == pytest.approx(2.591, rel=0.02)
    assert bl["theta"][k] == pytest.approx(0.664 * bl["x"][k] / np.sqrt(vf.RE * bl["x"][k]), rel=0.02)


def test_coupled_naca0012_alpha0():
    """Coarse mesh, Re 3e6, alpha 0: symmetric transition near mid-chord
    (XFOIL 0.51) and drag far below the fully turbulent value."""
    import polar
    r, _ = polar.run_point("naca0012", 3e6, 0.0, level=0.6, transition="free")
    assert r["status"] == "converged" and r["transition_converged"]
    assert 0.42 < r["x_tr_upper"] < 0.6 and 0.42 < r["x_tr_lower"] < 0.6
    assert abs(r["x_tr_upper"] - r["x_tr_lower"]) < 0.03
    assert r["Cd"] < 0.0065
    rt, _ = polar.run_point("naca0012", 3e6, 0.0, level=0.6, transition="off")
    assert rt["Cd"] > 1.4 * r["Cd"] and rt["x_tr_upper"] is None
