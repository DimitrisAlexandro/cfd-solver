"""Tests for multi-element airfoils (Phase 3, E5): geometry, panel method, mesher."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import multi_element as me  # noqa: E402
import panel  # noqa: E402
import boundary_layer as bl  # noqa: E402
from geometry import signed_area  # noqa: E402


def karman_trefftz(xc=-0.08, yc=0.06, tau_deg=12.0, n=200):
    """Exact Karman-Trefftz airfoil: open CCW outline from the TE, chord,
    exact lift curve."""
    b = 1.0
    R = math.hypot(b - xc, yc)
    beta = math.atan2(yc, b - xc)
    k = 2 - math.radians(tau_deg) / math.pi
    th = np.linspace(0, 2 * math.pi, n + 1)
    zeta = complex(xc, yc) + R * np.exp(1j * (-beta + th))
    z = k * b * ((zeta + b) ** k + (zeta - b) ** k) / ((zeta + b) ** k - (zeta - b) ** k)
    z[0] = z[-1] = k * b
    P = np.c_[z.real, z.imag]
    c = float(np.linalg.norm(P - P[0], axis=1).max())
    return P, c, lambda a: 8 * math.pi * R / c * math.sin(math.radians(a) + beta)


@pytest.mark.parametrize("alpha", [0.0, 8.0])
def test_panel_exact_karman_trefftz(alpha):
    errs = []
    for n in (100, 200, 400):
        P, c, cl = karman_trefftz(n=n)
        r = panel.solve([P], alpha, ref_chord=c)
        errs.append(abs(r["Cl"] / cl(alpha) - 1))
    assert errs[-1] < 2e-4
    assert errs[1] / errs[2] > 3.0                   # second order


def test_panel_multi_element_consistent():
    el, info = me.load_multi("30p30n")
    r = panel.solve([e["outline"] for e in el], 8.0, names=[e["name"] for e in el])
    # pressure lift = Kutta-Joukowski lift from the element circulations
    assert r["Cl"] == pytest.approx(r["Cl_kj"], rel=5e-3)
    assert r["Cl"] == pytest.approx(sum(e["Cl"] for e in r["elements"]), rel=1e-12)
    assert 3.6 < r["Cl"] < 3.9
    assert all(e["Cl"] > 0 for e in r["elements"])


def test_30p30n_geometry():
    el, info = me.load_multi("30p30n")
    assert [e["name"] for e in el] == ["slat", "main", "flap"]
    # nominal slot gaps 2.95 % (slat) and 1.27 % (flap); the 0.1 % TE base
    # thickening may only close them slightly
    assert info["min_gaps"]["slat-main"] == pytest.approx(0.0295, abs=0.0005)
    assert info["min_gaps"]["main-flap"] == pytest.approx(0.0127, abs=0.0006)
    for e in el:
        assert signed_area(e["loop"]) > 0
        L = e["loop"]
        d = np.roll(L, -1, axis=0) - L
        dn = np.roll(d, -1, axis=0)
        turn = np.degrees(np.abs(np.arctan2(d[:, 0] * dn[:, 1] - d[:, 1] * dn[:, 0],
                                            (d * dn).sum(1))))
        assert turn.max() < 35.0                    # all corners rounded: no fans


def test_spec_string_and_overlap():
    el, info = me.load_multi("naca23012+naca23012:0.3,0.98,-0.04,25")
    assert [e["name"] for e in el] == ["main", "flap"]
    assert el[1]["chord"] == pytest.approx(0.3, rel=2e-3)
    with pytest.raises(ValueError, match="overlap"):
        me.load_multi("naca0012+naca0012:0.3,0.5,0.0,0")


def test_round_corners():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    r = me.round_corners(sq, 0.1)
    assert signed_area(r) == pytest.approx(1 - (4 - math.pi) * 0.01, rel=1e-3)
    assert np.all(np.linalg.norm(r - 0.5, axis=1) <= math.hypot(0.5, 0.5) - 0.04)


def test_limited_boundary_layer():
    """thickness limit: same layer count, ends at the limit, cells valid."""
    th = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    wall = np.c_[np.cos(th), 0.3 * np.sin(th)]
    lim = lambda P: np.where(P[:, 1] > 0, 0.004, 0.02)
    layers, canon = bl.extrude(wall, 1e-5, growth=1.15, n_layers=80, verbose=False,
                               max_thickness=lim, repair=True)
    T = np.linalg.norm(layers[-1] - layers[0], axis=1)
    up = wall[:, 1] > 0.1
    assert T[up].max() <= 0.0045
    assert T[~up & (wall[:, 1] < -0.1)].min() > 0.01
    assert len(layers) > 30


def test_multi_mesh_valid(tmp_path):
    import hybrid_mesher as hm
    import mesh_check
    el, info = me.load_multi("30p30n", level=0.7)
    m = hm.multi_mesh(el, 9e6, level=0.7, alpha_deg=8.0)
    assert set(m["patches"]) == {"wall_slat", "wall_main", "wall_flap", "farfield"}
    hm.save_npz(m, tmp_path / "m.npz")
    errors, stats = mesh_check.check(tmp_path / "m.npz", verbose=False)
    assert not errors, errors
    from fvmesh import FVMesh
    fvm = FVMesh(m["points"], m["cells"], m["patches"])
    assert fvm.n_clamped == 0                       # no near-90 deg faces
