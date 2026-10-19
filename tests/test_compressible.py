"""Phase 5: compressible solver."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerolab.core.fvmesh import FVMesh, rect_quad  # noqa: E402
from aerolab.core.simple_solver import SimpleSolver  # noqa: E402
from aerolab.core.compressible import CompressibleSolver  # noqa: E402

NAMES = ("lower", "outlet", "upper", "inlet")


def _channel():
    from aerolab.validation import validate_channel as vch
    m = FVMesh(*rect_quad(30, 10, 0, 3, -vch.H, vch.H, names=vch.NAMES))
    bcs = {"wallB": {"type": "wall"}, "wallT": {"type": "wall"},
           "inlet": {"type": "inlet", "U": vch.profile}, "outlet": {"type": "outlet"}}
    return m, bcs, vch


def test_transpose_stress_consistent():
    """div(nu (grad U)^T) vanishes for constant nu and div U = 0: with the
    outlet faces included the channel solution must not change"""
    m, bcs, vch = _channel()
    a = SimpleSolver(m, bcs, vch.NU, U0=(0.5, 0))
    a.solve(tol=1e-10, max_iter=5000, verbose=False)
    b = SimpleSolver(m, bcs, vch.NU, U0=(0.5, 0))
    b._transpose_stress = lambda: True
    b.solve(tol=1e-10, max_iter=5000, verbose=False)
    assert np.abs(a.U - b.U).max() < 5e-4           # was 4e-2 without the outlet faces


def test_low_mach_limit():
    """M -> 0: the compressible solver reproduces the incompressible one"""
    m, bcs, vch = _channel()
    s0 = SimpleSolver(m, bcs, vch.NU, U0=(0.5, 0))
    s0.solve(tol=1e-11, max_iter=5000, verbose=False)
    s1 = CompressibleSolver(m, bcs, vch.NU, 0.001, U0=(0.5, 0), sutherland=None,
                            alpha_u=0.9, alpha_p=1.0, limiter=None)
    s1.solve(tol=1e-11, max_iter=3000, verbose=False)
    assert s1.converged
    assert np.abs(s1.U - s0.U).max() < 5e-4            # -2/3 mu div U term only
    assert abs(s1.rho.max() - 1) < 1e-5


def test_density_correction_stable_m05():
    """uniform M 0.5 channel, perturbed pressure: the compressible pressure
    correction (isentropic psi, velocity-only Majumdar flux) must converge"""
    m = FVMesh(*rect_quad(30, 10, 0, 3, 0, 1, names=NAMES))
    bcs = {"lower": {"type": "wall"}, "upper": {"type": "wall"},
           "inlet": {"type": "inlet", "U": (1.0, 0.0)}, "outlet": {"type": "outlet"}}
    s = CompressibleSolver(m, bcs, 0.0, 0.5, inviscid=True, alpha_p=1.0)
    s.incompressible_start = 0
    s.p += 1e-2 * np.random.default_rng(0).standard_normal(m.ncells)
    s.rho = (s.P0 + s.p) / (s.R * s.T)
    r = [max(s.iterate().values()) for _ in range(60)]
    assert r[-1] < 1e-3 * r[0] and np.all(np.isfinite(r))


@pytest.mark.parametrize("mach", [0.5, 0.675])
def test_ni_bump(mach):
    from aerolab.validation import validate_bump as vb
    r, s, _ = vb.run(mach, nx=60, ny=20, max_iter=3000, tol=1e-6)
    assert r["converged"]
    if mach == 0.5:                     # subsonic: symmetric, isentropic, M_max ~ 0.69
        assert r["supersonic_cells"] == 0
        assert 0.66 < r["M_wall_max"] < 0.71
        assert r["asym"] < 0.04
        assert r["entropy_max"] < 0.01
    else:                               # transonic: shock at ~72 % of the bump
        assert 1.2 < r["M_wall_max"] < 1.45
        assert r["x_shock"] is not None and 1.65 < r["x_shock"] < 1.80


def test_naca0012_euler_transonic():
    """NACA 0012, M 0.8, alpha 1.25, Euler (coarse): Cl ~0.35, Cd ~0.022"""
    from aerolab.validation import validate_naca0012_sa as v
    from aerolab.validation import validate_transonic as vt
    mesh = vt.euler_mesh(v.tmr0012(120), 1.25, level=0.6)
    fvm, bcs, U = vt.fv_case(mesh, 1.25)
    r, s, w = vt.solve(fvm, bcs, U, 1.25, 0.8, max_iter=2500)
    assert 0.32 < r["Cl"] < 0.38
    assert 0.019 < r["Cd"] < 0.027
    assert r["x_shock_upper"] is not None and 0.5 < r["x_shock_upper"] < 0.7


def test_polar_point_compressible():
    """polar tool: mach > 0 runs the compressible solver (RANS, warm start)"""
    from aerolab.airfoil import polar as pl
    r, st = pl.run_point("naca0012", 3e6, 2.0, level=0.7, mach=0.7, transition="off")
    assert r["solver"] == "compressible" and r["status"] == "converged"
    assert r["M_max"] > 1.0                                  # supersonic pocket
    assert 0.30 < r["Cl"] < 0.38 and 0.009 < r["Cd"] < 0.014
    assert st["T"] is not None and st["mach"] == 0.7
