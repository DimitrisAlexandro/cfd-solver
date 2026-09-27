"""Regression tests for the Spalart-Allmaras model (full studies:
validate_flatplate.py, validate_naca0012_sa.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import turbulence as sa  # noqa: E402
from fvmesh import FVMesh, rect_quad  # noqa: E402


def test_constants():
    assert sa.CW1 == pytest.approx(0.1355 / 0.41**2 + 1.622 / (2 / 3))


def test_log_layer_equilibrium():
    """In the log layer nuT = kappa u_tau y, S = u_tau/(kappa y) solves SA exactly:
    production - destruction + (1+cb2)/sigma |grad nuT|^2 = 0 (r = 1, fw = 1)."""
    t = object.__new__(sa.SpalartAllmaras)
    t.nu = 1e-7
    y = np.array([1e-2, 5e-2, 2e-1])              # chi = 1.6e3 .. 3e4
    t.d = y
    ut = 0.04
    nuT = sa.KAPPA * ut * y                          # chi >> 1 -> fv2 ~ 0
    S = t.source(nuT, ut / (sa.KAPPA * y))
    diff = (1 + sa.CB2) / sa.SIGMA * (sa.KAPPA * ut) ** 2
    assert np.abs(S + diff).max() < 3e-3 * diff    # fv2 term ~ 1/chi


def test_wall_distance_channel():
    m = FVMesh(*rect_quad(20, 10, 0.0, 2.0, 0.0, 1.0))
    walls = np.r_[m.patch_faces["bottom"], m.patch_faces["top"]]
    d = sa.wall_distance(m, walls)
    assert np.allclose(d, np.minimum(m.xc[:, 1], 1.0 - m.xc[:, 1]))


def test_flat_plate_coarse():
    """TMR flat plate, coarsest grid: converges, Cf(0.97) within 2 % of the
    incompressible SA reference 0.002729, laminar Blasius limit exact."""
    import validate_flatplate as vf
    from simple_solver import SimpleSolver
    r = vf.run(0.5, tol=1e-7, max_iter=3000, verbose=False)
    assert r["converged"]
    assert abs(r["cf97"] / vf.REF_CF_INCOMP - 1) < 0.02
    lam = SimpleSolver(vf.plate_mesh(0.5), vf.BCS, 1.0 / vf.RE, U0=(1.0, 0.0))
    lam.solve(tol=1e-9, max_iter=2000, verbose=False)
    x, cf, _ = vf.plate_cf(lam)
    blasius = 0.664 / np.sqrt(vf.RE * vf.X_CF)
    assert np.interp(vf.X_CF, x, cf) == pytest.approx(blasius, rel=0.01)


def test_turbulent_airfoil_case_mesh(tmp_path):
    import case
    out = tmp_path / "n0012"
    case.main(["mesh", "naca", "--naca", "tmr", "--re", "6e6", "--refine", "0.5",
               "--out", str(out)])
    meta = json.loads((out / "case.json").read_text())
    assert meta["turbulence"].startswith("Spalart-Allmaras")
    assert "vortex" in meta["bcs"]["inlet"]
    assert meta["n_bl_cells"] > 0 and meta["quality"]["nonorth_max"] < 90


def test_venkat_limiter():
    """Limiter (threshold 0.1 U_ref): inactive for small smooth variations,
    clips an O(U_ref) spurious extremum;
    by default only triangles are limited (quads keep psi = 1)."""
    from fvmesh import rect_tri
    from simple_solver import SimpleSolver
    m = FVMesh(*rect_tri(1 / 16))
    bcs = {k: {"type": "wall"} for k in ("bottom", "right", "left")}
    bcs["top"] = {"type": "wall", "U": (1.0, 0.0)}          # velocity scale U_ref = 1
    s = SimpleSolver(m, bcs, 1e-3, turbulence="sa")
    assert s.limiter == "venkat" and s.limiter_mask.all()
    smooth = 0.01 * m.xc[:, 0]
    g = m.grad(smooth, 0.01 * m.bxf[:, 0])
    assert s._venkat(smooth, g, 0.01 * m.bxf[:, 0]).min() > 0.99
    spike = np.zeros(m.ncells)
    i = np.argmin(np.linalg.norm(m.xc - 0.5, axis=1))
    spike[i] = 1.0
    g = m.grad(spike, np.zeros(len(m.bcell)))
    nb = np.r_[m.nei[m.own == i], m.own[m.nei == i]]
    psi = s._venkat(spike, g, np.zeros(len(m.bcell)))
    assert psi[nb].max() < 0.5                        # neighbours may not overshoot
    q = FVMesh(*rect_quad(8, 8))
    sq = SimpleSolver(q, bcs, 1e-3, turbulence="sa")
    assert not sq.limiter_mask.any()
    assert SimpleSolver(q, bcs, 1e-3).limiter is None  # laminar: unlimited
