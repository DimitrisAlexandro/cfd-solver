"""Tests for the polar driver and the airfoil tool (Phase 3, E1/E3).
Fast laminar cases (Re 500, coarse mesh); the turbulent validation is
validation/phase3_*."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import airfoil  # noqa: E402
import polar  # noqa: E402


def test_parse_alpha():
    assert airfoil.parse_alpha("-4:4:2") == [-4, -2, 0, 2, 4]
    assert airfoil.parse_alpha("0,5,10") == [0, 5, 10]
    assert airfoil.parse_alpha("0:1:0.25") == [0, 0.25, 0.5, 0.75, 1.0]


def test_moment_sign_convention():
    """Lift acts near the quarter chord: about the leading edge the moment is
    nose-down, Cm_LE ~ -Cl/4; about c/4 it is small."""
    from postproc import wall_data
    r, st = polar.run_point("naca0012", 500.0, 4.0, level=0.5)
    fvm, mesh, patches, bcs, U, turb, info = polar.make_case("naca0012", 500.0, 4.0, 0.5)
    from simple_solver import SimpleSolver
    s = SimpleSolver(fvm, bcs, 1 / 500.0, U0=U)
    s.set_initial(st["U"], st["p"])
    s.solve(tol=1e-7, max_iter=3000, verbose=False)
    c = wall_data(s, "wall", U)[1]
    cm_le = -s.moment("wall", (0.0, 0.0)) / 0.5
    assert c["Cl"] > 0.1
    assert cm_le == pytest.approx(-0.25 * c["Cl"], abs=0.25 * 0.25 * c["Cl"])
    assert abs(c["Cm"]) < 0.25 * abs(cm_le)


def test_polar_cli(tmp_path):
    out = tmp_path / "p"
    res = airfoil.main(["polar", "naca0012", "--re", "500", "--alpha", "-2:2:2", "--refine", "0.5",
                        "--jobs", "2", "--out", str(out)])
    assert [r["alpha"] for r in res] == [-2, 0, 2]
    assert all(r["status"] == "converged" for r in res)
    cl = {r["alpha"]: r["Cl"] for r in res}
    assert abs(cl[0]) < 2e-3 and cl[2] > 0.05
    assert cl[-2] == pytest.approx(-cl[2], rel=0.03)            # symmetric airfoil
    cd = {r["alpha"]: r["Cd"] for r in res}
    assert cd[-2] == pytest.approx(cd[2], rel=0.03)
    assert sum(r["warm_start"] for r in res) >= 1
    for f in ("polar.csv", "polar.png", "distributions.png", "report.txt", "polar.json",
              "geometry.png"):
        assert (out / f).exists(), f
    assert (out / "alpha_+02.00" / "results.vtm").exists()
    d = json.loads((out / "polar.json").read_text())
    assert len(d["results"]) == 3
    h = np.genfromtxt(out / "polar.csv", delimiter=",", names=True, dtype=None, encoding=None)
    assert len(h) == 3 and "Cm" in h.dtype.names


def test_show(tmp_path):
    airfoil.main(["show", "naca23012", "--out", str(tmp_path)])
    assert (tmp_path / "geometry.png").exists()


def test_prandtl_glauert():
    res = [{"Cl": 1.0, "Cm": 0.01, "Cd_p": 0.004, "Cd_v": 0.006, "Cd": 0.010,
            "wall": {"Cp": [1.0, -2.0]}}]
    airfoil.apply_prandtl_glauert(res, 0.15)
    b = 1 / np.sqrt(1 - 0.15**2)
    r = res[0]
    assert r["Cl"] == pytest.approx(b) and r["Cl_incompressible"] == 1.0
    assert r["Cd"] == pytest.approx(0.004 * b + 0.006)
    assert r["wall"]["Cp"][1] == pytest.approx(-2 * b, rel=1e-6)
    with pytest.raises(ValueError):
        airfoil.apply_prandtl_glauert(res, 0.8)
