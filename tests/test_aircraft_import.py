"""Aircraft import: AVL geometry + mass files, JSON."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import aircraft as A  # noqa: E402
import aircraft_import as AI  # noqa: E402

AVL = """Test glider
#Mach
0.0
#IYsym IZsym Zsym
0 0 0.0
#Sref Cref Bref
0.50 0.20 2.6
#Xref Yref Zref
0.07 0.0 0.0
0.010
#--------------------------------
SURFACE
Wing
8 1.0 20 -2.0
YDUPLICATE
0.0
ANGLE
2.0
SECTION
0.0 0.0 0.0 0.22 0.0
AFILE
clarky.dat
CONTROL
flap 1.0 0.75 0 0 0 1
SECTION
0.0 0.6 0.03 0.22 0.0
AFILE
clarky.dat
CONTROL
flap 1.0 0.75 0 0 0 1
CONTROL
aileron -1.0 0.75 0 0 0 -1
SECTION
0.06 1.3 0.10 0.12 -2.0
AFILE
clarky.dat
CONTROL
aileron -1.0 0.75 0 0 0 -1
#--------------------------------
SURFACE
Stab
6 1.0 8 -2.0
YDUPLICATE
0.0
TRANSLATE
0.9 0.0 0.05
SECTION
0.0 0.0 0.0 0.12 0.0
NACA
0010
CONTROL
Elevator 1.0 0.6 0 0 0 1
SECTION
0.02 0.3 0.0 0.09 0.0
NACA
0010
CONTROL
Elevator 1.0 0.6 0 0 0 1
#--------------------------------
SURFACE
Fin
6 1.0 6 -2.0
SECTION
0.92 0.0 0.05 0.14 0.0
NACA
0009
CONTROL
rudder 1.0 0.6 0 0 0 1
SECTION
0.98 0.0 0.30 0.09 0.0
NACA
0009
CONTROL
rudder 1.0 0.6 0 0 0 1
#--------------------------------
BODY
Fuse
12 1.0
BFILE
fuse.dat
"""

MASS = """Lunit = 0.01 m
Munit = 0.001 kg
Tunit = 1.0 s
g = 9.81
#  mass   x     y    z    Ixx  Iyy  Izz
   800    7.0   0.0  0.0  0    0    0     ! fuselage pod
   400    9.0  10.0  0.0  0    0    0     ! right wing
   400    9.0 -10.0  0.0  0    0    0     ! left wing
"""



def clarky():
    """a cambered section in Selig order (NACA 4412 outline stands in for a Clark Y)"""
    import airfoil_geom as ag
    o = ag.sharp_outline("naca4412")
    return ("clark y\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in o) + "\n").encode()


def test_avl_import(tmp_path):
    files = {"glider.avl": AVL.encode(), "glider.mass": MASS.encode(), "clarky.dat": clarky()}
    ac, warns = AI.read_files(files, tmp_path)
    w, h, f = ac.surfaces
    assert (w.name, h.name, f.name) == ("Wing", "Stab", "Fin")
    assert w.symmetric and not w.vertical and f.vertical and not f.symmetric
    assert w.span == pytest.approx(2.6) and w.root_chord == pytest.approx(0.22) and w.tip_chord == pytest.approx(0.12)
    assert w.incidence == pytest.approx(2.0) and w.twist == pytest.approx(-2.0)
    assert w.airfoil.endswith("clarky.dat") and Path(w.airfoil).exists()
    assert h.x == pytest.approx(0.9) and h.z == pytest.approx(0.05) and h.airfoil == "naca0010"
    assert f.span == pytest.approx(0.25) and f.sweep > 0
    ctl = {c.name: c for s in ac.surfaces for c in s.controls}
    assert set(ctl) == {"flap", "aileron", "elevator", "rudder"}
    assert ctl["aileron"].mode == "antisymmetric" and ctl["flap"].mode == "symmetric"
    assert ctl["flap"].eta1 == pytest.approx(0.6 / 1.3, abs=1e-3)
    assert ctl["aileron"].eta0 == pytest.approx(0.6 / 1.3, abs=1e-3) and ctl["aileron"].eta1 == 1.0
    assert ctl["elevator"].cf == pytest.approx(0.4)
    assert ac.mass == pytest.approx(1.6) and ac.cg[0] == pytest.approx(0.08)
    assert ac.inertias()[0] == pytest.approx(0.8 * 0.1**2)
    assert ac.cd_extra == pytest.approx(0.01)
    assert any("kink" in m for m in warns) and any("BODY" in m for m in warns)
    # the crank is kept: one kink at the inner/outer panel break, exact planform area
    assert len(w.kinks) == 1
    k = w.kinks[0]
    assert (k["y"], k["chord"], k["dx"], k["dz"], k["twist"]) == pytest.approx((0.6, 0.22, 0.0, 0.03, 0.0))
    assert w.area == pytest.approx(2 * (0.22 * 0.6 + 0.5 * (0.22 + 0.12) * 0.7))
    assert w.x_le(1.0) == pytest.approx(0.06, abs=1e-5) and w.rise(1.0) == pytest.approx(0.10, abs=1e-5)
    assert not h.kinks and not f.kinks
    # the imported aircraft analyses
    d = A.Analysis(ac, 12.0, 0.0, None).derivatives(2.0)
    assert d["Cm_a"] < 0 and d["Cn_b"] > 0


def test_avl_roundtrip(tmp_path):
    import aero_database as D
    ac = A.demo_aircraft()
    D.write_avl(ac, tmp_path)
    b, _ = AI.read_files({"a.avl": (tmp_path / "aircraft.avl").read_bytes(),
                          "a.mass": (tmp_path / "aircraft.mass").read_bytes()}, tmp_path)
    for s0, s1 in zip(ac.surfaces, b.surfaces):
        d0, d1 = asdict(s0), asdict(s1)
        for k, v in d0.items():
            if isinstance(v, float):
                assert d1[k] == pytest.approx(v, abs=2e-3), (s0.name, k)
            elif k != "controls":
                assert d1[k] == v, (s0.name, k)
        assert [(c.name, c.mode) for c in s0.controls] == [(c.name, c.mode) for c in s1.controls]
    assert b.mass == pytest.approx(ac.mass) and b.cg == pytest.approx(ac.cg)


def test_json_import():
    ac = A.demo_aircraft()
    b, w = AI.read_files({"x.json": json.dumps(ac.to_dict()).encode()})
    assert b.to_dict() == ac.to_dict() and w == []
    with pytest.raises(ValueError):
        AI.read_files({"x.txt": b"hello"})


def test_avl_roundtrip_kinks(tmp_path):
    import aero_database as D
    import validate_aircraft as VA
    ac = VA.test_cranked()
    D.write_avl(ac, tmp_path)                 # also writes sections at the control edges
    b, _ = AI.read_files({"a.avl": (tmp_path / "aircraft.avl").read_bytes()}, tmp_path)
    for s0, s1 in zip(ac.surfaces, b.surfaces):
        assert len(s1.kinks) == len(s0.kinks) == 1, s0.name          # collinear sections dropped
        for k in ("y", "chord", "dx", "dz", "twist"):
            assert s1.kinks[0][k] == pytest.approx(s0.kinks[0].get(k, 0.0), abs=1e-3)
        assert s1.area == pytest.approx(s0.area, rel=1e-4)
