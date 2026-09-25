"""
Case runner: geometry -> mesh -> (solve) -> files for the viewer / ParaView.

A case is a folder:
    mesh.vtm (+ mesh.vtu, mesh_boundary.vtp)   written BEFORE solving:
                                               mesh + per-cell quality fields
    mesh.npz                                   mesh in solver format
    history.csv                                written LIVE while solving:
                                               residuals, Cd, Cl per iteration
    results.vtm (+ results.vtu, results_boundary.vtp)
                                               flow fields + quality, wall Cp/Cf/y+
    wall_<patch>.csv                           wall distributions (x, y, Cp, Cf, y+)
    case.json                                  parameters, coefficients, run info

Usage
    python case.py mesh naca --naca 0012 --out ../cases/naca0012
    python case.py run cylinder --re 40 --out ../cases/cyl_re40
    python case.py run naca --naca 0012 --re 5000 --alpha 2 --out ../cases/n0012_re5k
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import hybrid_mesher as hm
from fvmesh import FVMesh
from geometry import circle
from io_vtk import write_vtm, write_vtp_lines, write_vtu
from postproc import flow_fields, mesh_quality, wall_data
from simple_solver import SimpleSolver


# ----------------------------------------------------------------------
# geometry + mesh
# ----------------------------------------------------------------------
def build(kind, args):
    ang = math.radians(getattr(args, "alpha", 0.0))
    U_inf = (math.cos(ang), math.sin(ang))
    if kind == "naca":
        mesh = hm.build_mesh(naca=args.naca, n_side=args.n_side, reynolds=args.re,
                             yplus=args.yplus, alpha_deg=args.alpha,
                             farfield_radius=args.farfield, verbose=True)
        label = f"NACA {args.naca}"
    elif kind == "cylinder":
        r = args.refine
        mesh = hm.build_mesh(wall=circle((0.5, 0.0), 0.5, int(round(200 * r))), h1=0.004 / r,
                             growth=1.1 ** (1 / r), farfield_radius=args.farfield,
                             n_farfield=int(80 * r), h_near=0.03 / r, grad=0.1 / r,
                             wake_length=4.0, wake_h=0.04 / r, wake_grad=0.03 / r,
                             alpha_deg=args.alpha, label="cylinder D=1", verbose=True)
        label = "cylinder"
    else:
        raise ValueError(kind)
    P = mesh["points"]
    ff = mesh["patches"]["farfield"]
    t = P[ff[:, 1]] - P[ff[:, 0]]
    inflow = t[:, 1] * U_inf[0] - t[:, 0] * U_inf[1] < 0    # outward normal . U_inf < 0
    patches = {"wall": mesh["patches"]["airfoil"], "inlet": ff[inflow], "outlet": ff[~inflow]}
    fvm = FVMesh(P, mesh["cells"], patches)
    return fvm, mesh, patches, U_inf, label


# ----------------------------------------------------------------------
# writers
# ----------------------------------------------------------------------
def _boundary(fvm, patches, extra=None):
    """Boundary lines + patch id (+ optional per-face arrays on some faces)."""
    names = list(patches)
    lines = np.vstack([patches[n] for n in names])
    pid = np.concatenate([[i] * len(patches[n]) for i, n in enumerate(names)]).astype(np.int32)
    data = {"patch_id": pid}
    if extra:
        for k, (face_rows, vals) in extra.items():
            arr = np.full(len(lines), np.nan)
            arr[face_rows] = vals
            data[k] = arr
    return lines, data, names


def write_mesh(folder, fvm, mesh, patches, meta):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    q = mesh_quality(fvm, mesh["cells"], mesh["n_bl"])
    write_vtu(folder / "mesh.vtu", fvm.points, mesh["cells"], cell_data=q)
    lines, bdata, names = _boundary(fvm, patches)
    write_vtp_lines(folder / "mesh_boundary.vtp", fvm.points, lines, bdata)
    write_vtm(folder / "mesh.vtm", {"cells": "mesh.vtu", "boundary": "mesh_boundary.vtp"})
    hm.save_npz(mesh, folder / "mesh.npz")
    meta.update(patches=names, n_cells=fvm.ncells, n_bl_cells=mesh["n_bl"],
                quality={"min_angle_min": float(q["min_angle"][mesh["n_bl"]:].min())
                         if fvm.ncells > mesh["n_bl"] else None,
                         "nonorth_max": float(q["nonorthogonality"].max()),
                         "size_ratio_max_core": float(q["size_ratio"][mesh["n_bl"]:].max())
                         if fvm.ncells > mesh["n_bl"] else None})
    (folder / "case.json").write_text(json.dumps(meta, indent=2))
    return q


def run_case(folder, fvm, mesh, patches, U_inf, nu, meta, tol=1e-7, max_iter=20000,
             monitor_every=10):
    folder = Path(folder)
    q = write_mesh(folder, fvm, mesh, patches, meta)
    print(f"[case] mesh written -> {folder / 'mesh.vtm'} (open it now to inspect)")

    bcs = {"wall": {"type": "wall"}, "inlet": {"type": "inlet", "U": U_inf},
           "outlet": {"type": "outlet"}}
    s = SimpleSolver(fvm, bcs, nu, U0=U_inf)
    hist = open(folder / "history.csv", "w", buffering=1)          # line-buffered: live
    hist.write("iteration,cont,Ux,Uy,Cd,Cl,time_s\n")
    t0 = time.time()

    def monitor(it, r):
        if it % monitor_every == 0 or it == 1:
            _, c = wall_data(s, "wall", U_inf)
            cd, cl = f"{c['Cd']:.8g}", f"{c['Cl']:.8g}"
        else:
            cd = cl = ""
        hist.write(f"{it},{r['cont']:.4e},{r['Ux']:.4e},{r['Uy']:.4e},{cd},{cl},"
                   f"{time.time() - t0:.2f}\n")

    it = s.solve(tol=tol, max_iter=max_iter, verbose=True, every=200, callback=monitor)
    hist.close()

    # results
    f = flow_fields(s, U_inf=np.linalg.norm(U_inf))
    f.update(q)
    write_vtu(folder / "results.vtu", fvm.points, mesh["cells"], cell_data=f)
    w, coeffs = wall_data(s, "wall", U_inf)
    lines, bdata, names = _boundary(fvm, patches)
    # boundary faces are stored patch by patch in _boundary's order
    start = {n: sum(len(patches[k]) for k in names[:i]) for i, n in enumerate(names)}
    wall_rows = start["wall"] + np.searchsorted(fvm.patch_faces["wall"], w["face"])
    lines, bdata, _ = _boundary(fvm, patches, extra={k: (wall_rows, w[k]) for k in
                                                     ("Cp", "Cf", "yplus")})
    write_vtp_lines(folder / "results_boundary.vtp", fvm.points, lines, bdata)
    write_vtm(folder / "results.vtm", {"cells": "results.vtu", "boundary": "results_boundary.vtp"})
    cols = ("x", "y", "side", "Cp", "Cf", "yplus", "wall_distance")
    np.savetxt(folder / "wall_wall.csv", np.c_[tuple(w[c] for c in cols)], delimiter=",",
               header=",".join(cols), comments="", fmt="%.8g")
    meta.update(iterations=it, converged=bool(s.converged), runtime_s=round(time.time() - t0, 1),
                coefficients={k: float(v) for k, v in coeffs.items()},
                yplus_max=float(w["yplus"].max()), yplus_mean=float(w["yplus"].mean()))
    (folder / "case.json").write_text(json.dumps(meta, indent=2))
    print(f"[case] results -> {folder / 'results.vtm'}  Cd={coeffs['Cd']:.4f} Cl={coeffs['Cl']:.4f}  "
          f"y+ max {w['yplus'].max():.2f}")
    return s, coeffs


# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["mesh", "run"])
    ap.add_argument("kind", choices=["naca", "cylinder"])
    ap.add_argument("--out", required=True, help="case folder")
    ap.add_argument("--naca", default="0012")
    ap.add_argument("--n-side", type=int, default=120)
    ap.add_argument("--re", type=float, default=None,
                    help="Reynolds number, chord / diameter (default: naca 1e6, cylinder 40)")
    ap.add_argument("--yplus", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.0, help="angle of attack (deg)")
    ap.add_argument("--farfield", type=float, default=None, help="farfield radius")
    ap.add_argument("--refine", type=float, default=1.0, help="cylinder mesh refinement")
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--max-iter", type=int, default=20000)
    a = ap.parse_args(argv)
    if a.re is None:
        a.re = 1e6 if a.kind == "naca" else 40.0
    if a.farfield is None:
        a.farfield = 20.0 if a.kind == "naca" else 60.0

    fvm, mesh, patches, U_inf, label = build(a.kind, a)
    meta = {"case": label, "kind": a.kind, "Re": a.re, "alpha_deg": a.alpha,
            "U_inf": list(U_inf), "nu": 1.0 / a.re, "ref_length": 1.0,
            "bcs": {"wall": "no-slip wall", "inlet": "fixed velocity U_inf",
                    "outlet": "zero-gradient U, p = 0"}}
    if a.action == "mesh":
        write_mesh(a.out, fvm, mesh, patches, meta)
        print(f"[case] mesh written -> {Path(a.out) / 'mesh.vtm'}")
    else:
        run_case(a.out, fvm, mesh, patches, U_inf, 1.0 / a.re, meta, tol=a.tol,
                 max_iter=a.max_iter)


if __name__ == "__main__":
    main()
