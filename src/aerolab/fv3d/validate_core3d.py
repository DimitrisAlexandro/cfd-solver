"""3D-1 validation: the 3D finite-volume core (SIMPLEC + SA, C++ kernels).

    python -m fv3d.validate_core3d            # all cases (~1 h on 2 cores)
    python -m fv3d.validate_core3d --cases duct,sphere

Cases
  cavity   lid-driven cavity Re 100 as a one-cell-thick extrusion with symmetry
           planes: must reproduce the 2D solver on the same grid (same discretisation)
  airfoil  NACA 0012 Re 6e6 alpha 10 (SA), extruded TMR mesh: Cl/Cd versus the 2D
           solver and CFL3D
  duct     laminar square duct, Re_Dh 50: developed u_max/u_mean = 2.0962 and
           f Re_Dh = 56.91 (Shah & London 1978)
  sphere   Re 100: Cd ~ 1.087-1.09, recirculation length Ls/D ~ 0.87-0.88
           (Johnson & Patel 1999; Clift, Grace & Weber 1978)
Outputs: validation/core3d_validation.json, core3d_report.md
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fv3d import meshgen3d as G  # noqa: E402
from fv3d.mesh3d import FVMesh3D  # noqa: E402
from fv3d.solver3d import Solver3D  # noqa: E402


def case_cavity(n=32, Re=100.0):
    from fvmesh import FVMesh, rect_quad
    from simple_solver import SimpleSolver
    pm = G.box(n, n, 1, lz=1.0 / n, names={"xmin": "left", "xmax": "right", "ymin": "bottom",
                                          "ymax": "top", "zmin": "front", "zmax": "back"})
    m3 = FVMesh3D(pm)
    bc = {"top": {"type": "wall", "U": (1.0, 0.0, 0.0)}, "bottom": {"type": "wall"}, "left": {"type": "wall"},
          "right": {"type": "wall"}, "front": {"type": "symmetry"}, "back": {"type": "symmetry"}}
    t = time.time()
    s3 = Solver3D(m3, bc, 1.0 / Re)
    it3 = s3.solve(tol=1e-9, max_iter=6000, verbose=False)
    t3 = time.time() - t
    pts, cells, patches = rect_quad(n, n)
    m2 = FVMesh(pts, cells, patches)
    s2 = SimpleSolver(m2, {"top": {"type": "wall", "U": (1.0, 0.0)}, "bottom": {"type": "wall"},
                           "left": {"type": "wall"}, "right": {"type": "wall"}}, 1.0 / Re)
    it2 = s2.solve(tol=1e-9, max_iter=6000, verbose=False)
    r2, r3 = np.round(m2.xc, 9), np.round(m3.xc, 9)
    o2, o3 = np.lexsort((r2[:, 1], r2[:, 0])), np.lexsort((r3[:, 1], r3[:, 0]))
    return {"cells": int(m3.ncells), "iterations_3d": it3, "iterations_2d": it2, "time_3d_s": round(t3, 1),
            "max_dU_3d_vs_2d": float(np.abs(s3.U[o3, :2] - s2.U[o2]).max()),
            "max_Uz": float(np.abs(s3.U[:, 2]).max()), "pass": bool(np.abs(s3.U[o3, :2] - s2.U[o2]).max() < 1e-5)}


def case_airfoil(level=0.7):
    import polar
    from simple_solver import SimpleSolver
    Re, a = 6e6, 10.0
    fvm, mesh, patches, bcs, U, turb, info = polar.make_case("tmr0012", Re, a, level)
    s2 = SimpleSolver(fvm, bcs, 1 / Re, turbulence="sa", U0=U)
    s2.solve(tol=1e-9, max_iter=3000, verbose=False)
    P2 = mesh["points"]
    ep = {(min(x, y), max(x, y)): nm for nm, E in patches.items() for x, y in E}
    m3 = FVMesh3D(G.extrude(P2, mesh["cells"], ep, 2, xyz=lambda l: np.c_[P2, np.full(len(P2), 0.01 * l)]))
    bc3 = {k: ({"type": v["type"], "U": tuple(v["U"]) + (0.0,)} if "U" in v else dict(v)) for k, v in bcs.items()}
    bc3 |= {"zmin": {"type": "symmetry"}, "zmax": {"type": "symmetry"}}
    t = time.time()
    s3 = Solver3D(m3, bc3, 1 / Re, turbulence="sa", U0=tuple(U) + (0.0,))
    s3.solve(tol=1e-9, max_iter=3000, verbose=False)
    t3 = time.time() - t
    ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))

    def clcd(Fp, Fv, span):
        Fx, Fy = (Fp + Fv)[0], (Fp + Fv)[1]
        return (-Fx * sa + Fy * ca) / (0.5 * span), (Fx * ca + Fy * sa) / (0.5 * span)
    cl2, cd2 = clcd(*s2.force("wall"), 1.0)
    cl3, cd3 = clcd(*s3.force("wall"), 0.01)
    return {"cells": int(m3.ncells), "Cl_3d": cl3, "Cd_3d": cd3, "Cl_2d": cl2, "Cd_2d": cd2,
            "Cl_CFL3D": 1.0909, "Cd_CFL3D": 0.01231, "iterations_3d": len(s3.history), "time_3d_s": round(t3, 1),
            "pass": bool(abs(cl3 - cl2) < 1e-3 * abs(cl2) and abs(cd3 - cd2) < 1e-2 * cd2)}


def case_duct(Re=50.0, n=14, nx=80, L=8.0):
    m = FVMesh3D(G.duct(nx, n, n, L, 1.0, 1.0, beta=1.2))
    s = Solver3D(m, {"inlet": {"type": "inlet", "U": (1.0, 0.0, 0.0)}, "outlet": {"type": "outlet"},
                     "wall": {"type": "wall"}}, 1.0 / Re, U0=(1.0, 0.0, 0.0))
    t = time.time()
    it = s.solve(tol=1e-7, max_iter=3000, verbose=False)
    # developed section: cells in a slab near x = 0.75 L
    xs = np.unique(np.round(m.xc[:, 0], 9))
    x0 = xs[np.argmin(np.abs(xs - 0.75 * L))]
    sl = np.abs(m.xc[:, 0] - x0) < 1e-9
    umean = (s.U[sl, 0] * m.vol[sl]).sum() / m.vol[sl].sum()
    umax_ratio = s.U[sl, 0].max() / umean
    # f Re from the mean pressure gradient between 0.6 L and 0.9 L (Dh = 1)
    xa, xb = xs[np.argmin(np.abs(xs - 0.6 * L))], xs[np.argmin(np.abs(xs - 0.9 * L))]
    pa = s.p[np.abs(m.xc[:, 0] - xa) < 1e-9].mean()
    pb = s.p[np.abs(m.xc[:, 0] - xb) < 1e-9].mean()
    dpdx = (pb - pa) / (xb - xa)
    fRe = -dpdx * 1.0 / (0.5 * umean ** 2) * Re * umean      # Darcy f = -dp/dx Dh / (rho U^2/2)
    # (the inlet velocity is the mean velocity; Re_Dh = U Dh / nu)
    return {"cells": int(m.ncells), "iterations": it, "time_s": round(time.time() - t, 1),
            "umax_over_umean": umax_ratio, "ref_umax_over_umean": 2.0962,
            "fRe": fRe, "ref_fRe": 56.91,
            "pass": bool(abs(umax_ratio / 2.0962 - 1) < 0.03 and abs(fRe / 56.91 - 1) < 0.05)}


def case_sphere(n=16, nr=40, Re=100.0):
    from scipy.spatial import cKDTree
    m = FVMesh3D(G.sphere(n, nr, 0.5, 20.0, first=0.01))
    s = Solver3D(m, {"sphere": {"type": "wall"}, "farfield": {"type": "farfield", "U": (1, 0, 0)}}, 1 / Re,
                 U0=(1, 0, 0))
    t = time.time()
    it = s.solve(tol=1e-6, max_iter=3000, verbose=False)
    Fp, Fv = s.force("sphere")
    A = math.pi / 4
    cd = (Fp + Fv)[0] / (0.5 * A)
    xa = np.linspace(0.505, 4.0, 700)
    Pa = np.c_[xa, np.zeros_like(xa), np.zeros_like(xa)]
    _, j = cKDTree(m.xc).query(Pa)
    ux = s.U[j, 0] + np.einsum("ij,ij->i", s.gU[0][j], Pa - m.xc[j])
    neg = np.where(ux < 0)[0]
    Ls = float(xa[neg[-1] + 1] - 0.5) if len(neg) and neg[-1] + 1 < len(xa) else float("nan")
    return {"cells": int(m.ncells), "iterations": it, "time_s": round(time.time() - t, 1), "Cd": cd,
            "Cd_pressure": Fp[0] / (0.5 * A), "Cd_friction": Fv[0] / (0.5 * A), "ref_Cd": [1.087, 1.09],
            "Ls_over_D": Ls, "ref_Ls_over_D": [0.87, 0.88],
            "pass": bool(abs(cd / 1.088 - 1) < 0.04 and abs(Ls / 0.875 - 1) < 0.1)}


CASES = {"cavity": case_cavity, "airfoil": case_airfoil, "duct": case_duct, "sphere": case_sphere}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="cavity,airfoil,duct,sphere")
    ap.add_argument("--out", default=str(ROOT / "validation"))
    a = ap.parse_args(argv)
    from fv3d import native
    out = {"native": native.status(), "cases": {}}
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    for nm in a.cases.split(","):
        t = time.time()
        r = CASES[nm]()
        r["wall_time_s"] = round(time.time() - t, 1)
        out["cases"][nm] = r
        print(nm, json.dumps(r), flush=True)
        (o / "core3d_validation.json").write_text(json.dumps(out, indent=1, default=float))
    L = ["# 3D finite-volume core validation (3D-1)", "", f"Native kernels: {out['native']}", ""]
    for nm, r in out["cases"].items():
        L.append(f"## {nm}: {'PASS' if r.get('pass') else 'CHECK'}")
        L += [f"- {k}: {v:.6g}" if isinstance(v, float) else f"- {k}: {v}" for k, v in r.items() if k != "pass"]
        L.append("")
    (o / "core3d_report.md").write_text("\n".join(L))
    print("report:", o / "core3d_report.md")


if __name__ == "__main__":
    main()
