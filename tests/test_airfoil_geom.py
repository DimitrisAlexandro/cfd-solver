"""Tests for the airfoil geometry input (Phase 3, E2)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerolab.airfoil import airfoil_geom as ag  # noqa: E402


def _dist(P, C):
    a, b = C[:-1], C[1:]
    ab = b - a
    out = []
    for p in P:
        t = np.clip(((p - a) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
        out.append(np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))
    return np.array(out)


@pytest.mark.parametrize("code,t,xt", [("0012", 0.12, 0.30), ("2412", 0.12, 0.30),
                                         ("23012", 0.12, 0.30), ("4415", 0.15, 0.30)])
def test_naca_shapes(code, t, xt):
    loop, info = ag.load_airfoil("naca" + code)
    assert info["max_thickness"] == pytest.approx(t, abs=2e-4)
    assert info["x_max_thickness"] == pytest.approx(xt, abs=0.02)
    ref, _ = ag.normalise(ag.naca_raw(code, 6000))
    assert _dist(loop[loop[:, 0] < 0.999], ref).max() < 1e-6


def test_naca23012_camber():
    raw = ag.naca_raw("23012", 4000)
    i = np.argmin(raw[:, 0])
    up, lo = raw[:i + 1][::-1], raw[i:]
    x = np.linspace(0.01, 0.99, 981)
    yc = 0.5 * (np.interp(x, up[:, 0], up[:, 1]) + np.interp(x, lo[:, 0], lo[:, 1]))
    assert yc.max() == pytest.approx(0.0184, abs=2e-4)          # NACA: 1.84 % at 15 % c
    assert x[yc.argmax()] == pytest.approx(0.15, abs=0.01)


@pytest.mark.parametrize("fmt", ["selig", "lednicer"])
def test_dat_roundtrip(tmp_path, fmt):
    raw = ag.naca_raw("2412", 200)
    f = tmp_path / f"n2412_{fmt}.dat"
    ag.write_dat(f, raw, "NACA 2412", fmt=fmt)
    pts, name = ag.read_dat(f)
    assert name == "NACA 2412"
    loop_file, _ = ag.load_airfoil(str(f))
    loop_code, _ = ag.load_airfoil("naca2412")
    assert np.abs(loop_file - loop_code).max() < 2e-5          # 8-digit file vs 200-point spline


def test_scaled_shifted_and_rotated_input(tmp_path):
    raw = ag.naca_raw("0012", 200)
    a = math.radians(5.0)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    moved = (raw * 250.0) @ R.T + [30.0, -7.0]                  # mm, rotated, offset
    f = tmp_path / "moved.dat"
    ag.write_dat(f, moved, "moved")
    loop, info = ag.load_airfoil(str(f), derotate=True)
    ref, _ = ag.load_airfoil("naca0012")
    assert info["chord_input"] == pytest.approx(250.0, rel=1e-4)
    assert info["chord_angle_deg"] == pytest.approx(5.0, abs=1e-3)
    assert np.abs(loop - ref).max() < 1e-4
    # without derotate the 5 deg stays in the geometry (angle of attack reference)
    loop2, info2 = ag.load_airfoil(str(f))
    assert abs(loop2[:, 1].max() - ref[:, 1].max()) > 0.01


def test_sharp_te_gets_small_base():
    loop, info = ag.load_airfoil("tmr0012")
    assert info["te_thickness"] == pytest.approx(ag.TE_THICK, rel=1e-2)   # rescaled after the cut
    assert loop[:, 0].max() == pytest.approx(1.0, abs=1e-9) and loop[:, 0].min() == pytest.approx(0.0, abs=1e-9)
    from aerolab.validation import validate_naca0012_sa as v
    assert np.abs(loop - v.tmr0012(160)).max() < 5e-4          # same as the validated TMR loop


def test_bad_input(tmp_path):
    f = tmp_path / "junk.dat"
    f.write_text("not an airfoil\n1 2\n3 4\n")
    with pytest.raises(ValueError):
        ag.load_airfoil(str(f))
    with pytest.raises(FileNotFoundError):
        ag.load_airfoil("no_such_foil.dat")
    with pytest.raises(ValueError):
        ag.load_airfoil("naca23112")                            # reflexed 5-digit


def test_mesh_from_dat(tmp_path):
    from aerolab.mesh import hybrid_mesher as hm
    from aerolab.mesh.geometry import first_cell_height
    f = tmp_path / "n2412.dat"
    ag.write_dat(f, ag.naca_raw("2412", 160), "NACA 2412")
    loop, _ = ag.load_airfoil(str(f), n_side=80)
    m = hm.build_mesh(wall=loop, reynolds=3e6, h1=first_cell_height(3e6, 1.0), growth=1.2,
                      n_layers=40, farfield_radius=20, n_farfield=32, verbose=False)
    assert len(m["cells"]) > 1000 and m["n_bl"] > 0
