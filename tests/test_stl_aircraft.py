"""STL import / export of the aircraft model."""
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import aircraft as A  # noqa: E402
import stl_aircraft as S  # noqa: E402


def _compare(a, b, tol_len=0.012, tol_deg=0.15):
    for s0 in a.surfaces:
        s1 = next(s for s in b.surfaces if s.name == s0.name)
        d0, d1 = asdict(s0), asdict(s1)
        for k in ("span", "root_chord", "tip_chord", "x", "z"):
            assert d1[k] == pytest.approx(d0[k], abs=tol_len * max(a.b, 1)), (s0.name, k)
        for k in ("sweep", "dihedral", "twist", "incidence"):
            assert d1[k] == pytest.approx(d0[k], abs=tol_deg), (s0.name, k)
        assert s1.symmetric == s0.symmetric and s1.vertical == s0.vertical


def test_roundtrip_binary(tmp_path):
    ac = A.demo_aircraft()
    S.write_stl(S.aircraft_mesh(ac), tmp_path / "a.stl")
    b, warns, info = S.to_aircraft(S.read_stl((tmp_path / "a.stl").read_bytes()), af_dir=tmp_path)
    assert [s.name for s in b.surfaces] == ["wing", "horizontal tail", "fin"]
    _compare(ac, b)
    assert b.wing.airfoil == "naca2412" and b.surfaces[1].airfoil == "naca0012"
    f0, f1 = ac.fuselage, b.fuselage
    assert f1.length == pytest.approx(f0.length, rel=0.01)
    assert f1.diameter == pytest.approx(f0.diameter, rel=0.02)
    assert f1.x_max == pytest.approx(f0.x_max, abs=0.03)
    assert {c.name for s in b.surfaces for c in s.controls} == {"aileron", "flap", "elevator", "rudder"}


def test_swept_ttail_other_axes_mm_ascii(tmp_path):
    ac = A.demo_aircraft()
    w, h = ac.surfaces[0], ac.surfaces[1]
    w.sweep, w.dihedral, w.twist, w.airfoil, w.tip_airfoil = 25.0, 6.0, -3.0, "naca4415", "naca0012"
    h.z, h.x, h.sweep = 1.6, 6.9, 15.0
    m = S.aircraft_mesh(ac)
    m = np.stack([-m[..., 0], m[..., 2], m[..., 1]], -1) * 1000 + [500, 20, 30]   # mm, y up, nose to +x
    txt = "solid t\n" + "".join("facet normal 0 0 0\nouter loop\n" + "".join(
        f"vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n" for p in t) + "endloop\nendfacet\n" for t in m) + "endsolid t\n"
    tris = S.read_stl(txt.encode())
    units = S.guess_units(tris)
    la, nose, up = S.guess_orientation(tris, units)
    assert (units, la, nose, up) == ("mm", "x", "max", "+y")
    b, _, _ = S.to_aircraft(tris, la, nose, up, units, af_dir=tmp_path)
    _compare(ac, b, tol_deg=0.2)
    assert b.wing.airfoil == "naca4415" and b.wing.tip_airfoil == "naca0012"


def test_open_sheets_and_cut_sections(tmp_path):
    """a mesh of separate open left / right halves (not closed at y = 0)"""
    ac = A.demo_aircraft()
    m = S.aircraft_mesh(ac).copy()
    left = m.mean(1)[:, 1] < 0
    m[left] += [1e-6, 0, 0]          # left half no longer shares vertices with the right half
    V, T = S.indexed(m)
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    assert (np.unique(E, axis=0, return_counts=True)[1] == 1).sum() > 100      # really open
    b, _, _ = S.to_aircraft(m, sections="cut", af_dir=tmp_path)
    assert [s.name for s in b.surfaces] == ["wing", "horizontal tail", "fin"]
    _compare(ac, b)
    import airfoil_geom as ag
    _, inf = ag.load_airfoil(b.wing.airfoil)
    assert inf["max_thickness"] == pytest.approx(0.12, abs=0.006)


def test_stl_bytes_roundtrip():
    m = S.aircraft_mesh(A.demo_aircraft())
    t = S.read_stl(S.stl_bytes(m))
    assert t.shape == m.shape and np.allclose(t, m, atol=1e-5)
    with pytest.raises(ValueError):
        S.read_stl(b"hello")
