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
    results.pvd + snapshots/                   transient runs: time series (ParaView)
    plots.png, U_mag.png, Cp.png, vorticity.png, mesh.png (+ nut_ratio.png, vorticity.gif)
                                               pictures, written automatically
    case.json                                  parameters, coefficients, run info

Usage
    python case.py mesh naca --naca 0012 --out ../cases/naca0012
    python case.py run cylinder --re 40 --out ../cases/cyl_re40
    python case.py run naca --naca 0012 --re 1000 --alpha 4 --out ../cases/n0012_re1k
    python case.py run cylinder --re 100 --transient --dt 0.05 --t-end 150 --out ../cases/cyl100
    python case.py run naca --naca 0012 --re 6e6 --alpha 10 --out ../cases/n0012_re6m
    Airfoils with Re < 1e5: laminar, validated laminar mesh settings.
    Airfoils with Re >= 1e5: Spalart-Allmaras (fully turbulent), y+ < 1 mesh,
    farfield radius 100 c with a point-vortex correction (validate_naca0012_sa.py).
    --naca tmr uses the NASA TMR NACA 0012 definition; --turbulence laminar|sa
    overrides the automatic choice; results then also hold nut, nut_ratio,
    nuTilde, wall_distance.
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
from io_vtk import write_pvd, write_vtm, write_vtp_lines, write_vtu
from postproc import flow_fields, mesh_quality, wall_data
from simple_solver import SimpleSolver


LAMINAR_RE = 1e5     # below this, airfoil cases use the laminar mesh settings

# ----------------------------------------------------------------------
# geometry + mesh
# ----------------------------------------------------------------------
def build(kind, args):
    ang = math.radians(getattr(args, "alpha", 0.0))
    U_inf = (math.cos(ang), math.sin(ang))
    if kind == "naca" and args.re < LAMINAR_RE:
        # laminar regime: the settings validated in validate_airfoil.py
        from validate_airfoil import airfoil_case
        fvm, mesh, patches, U = airfoil_case(code=args.naca, re=args.re, alpha=args.alpha,
                                             refine=args.refine, farfield=args.farfield,
                                             verbose=True)
        return fvm, mesh, patches, U, f"NACA {args.naca} (laminar mesh)"
    if kind == "naca":
        # turbulent regime: the settings validated in validate_naca0012_sa.py
        from validate_naca0012_sa import airfoil_mesh
        naca = None if args.naca.lower() in ("0012tmr", "tmr") else args.naca
        mesh = airfoil_mesh(level=args.refine, alpha=args.alpha, verbose=True, naca=naca,
                            re=args.re, farfield=args.farfield, yplus=args.yplus)
        label = "TMR NACA 0012" if naca is None else f"NACA {args.naca}"
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


def write_results(folder, s, fvm, mesh, patches, U_inf, q):
    """Final flow fields + wall distributions (results.vtm, wall_wall.csv)."""
    f = flow_fields(s, U_inf=np.linalg.norm(U_inf))
    f.update(q)
    write_vtu(folder / "results.vtu", fvm.points, mesh["cells"], cell_data=f)
    w, coeffs = wall_data(s, "wall", U_inf)
    names = list(patches)
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
    return w, coeffs


def export_pngs(folder, transient=False):
    """Pictures of every finished case, saved in the case folder."""
    try:
        import viewer
    except Exception as e:                     # pyvista missing: files are still complete
        print(f"[case] PNG export skipped ({e})")
        return
    folder = Path(folder)
    viewer.plots(folder, show=False, save=folder / "plots.png")
    fields = ["U_mag", "Cp", "vorticity"]
    meta = folder / "case.json"
    if meta.exists() and json.loads(meta.read_text()).get("turbulence", "laminar") != "laminar":
        fields.append("nut_ratio")                      # eddy viscosity / nu
    for field in fields:
        v = viewer.Viewer(folder, field=field, off_screen=True,
                          show_streamlines=(field == "U_mag" and not transient))
        v.save(folder / f"{field}.png")
        v.pl.close()
    v = viewer.Viewer(folder / "mesh.vtm", field="min_angle", edges=True, off_screen=True)
    v.save(folder / "mesh.png")
    v.pl.close()
    if transient and (folder / "snapshots").exists():
        viewer.animate(folder, field="vorticity", out=folder / "vorticity.gif")
    print(f"[case] pictures -> {folder}/*.png")


def run_case(folder, fvm, mesh, patches, U_inf, nu, meta, tol=1e-7, max_iter=20000,
             monitor_every=10, pngs=True, turbulence=None, vortex=False):
    folder = Path(folder)
    q = write_mesh(folder, fvm, mesh, patches, meta)
    print(f"[case] mesh written -> {folder / 'mesh.vtm'} (open it now to inspect)")

    bcs = {"wall": {"type": "wall"}, "inlet": {"type": "inlet", "U": U_inf},
           "outlet": {"type": "outlet"}}
    s = SimpleSolver(fvm, bcs, nu, U0=U_inf, turbulence=turbulence)
    hist = open(folder / "history.csv", "w", buffering=1)          # line-buffered: live
    hist.write("iteration,cont,Ux,Uy," + ("nuT," if turbulence else "") + "Cd,Cl,time_s\n")
    t0 = time.time()
    vf = None
    if vortex:                    # point-vortex farfield correction (lifting airfoils)
        from validate_naca0012_sa import VortexFarfield
        alpha = math.degrees(math.atan2(U_inf[1], U_inf[0]))
        vf = VortexFarfield(s, alpha)

    def monitor(it, r):
        if vf is not None:
            vf.update(it)
        if it % monitor_every == 0 or it == 1:
            _, c = wall_data(s, "wall", U_inf)
            cd, cl = f"{c['Cd']:.8g}", f"{c['Cl']:.8g}"
        else:
            cd = cl = ""
        nt = f"{r['nuT']:.4e}," if turbulence else ""
        hist.write(f"{it},{r['cont']:.4e},{r['Ux']:.4e},{r['Uy']:.4e},{nt}{cd},{cl},"
                   f"{time.time() - t0:.2f}\n")

    it = s.solve(tol=tol, max_iter=max_iter, verbose=True, every=200, callback=monitor)
    hist.close()

    w, coeffs = write_results(folder, s, fvm, mesh, patches, U_inf, q)
    meta.update(iterations=it, converged=bool(s.converged), runtime_s=round(time.time() - t0, 1),
                coefficients={k: float(v) for k, v in coeffs.items()},
                yplus_max=float(w["yplus"].max()), yplus_mean=float(w["yplus"].mean()))
    if turbulence:
        meta.update(nut_ratio_max=float((s.turb.nu_t() / nu).max()))
    (folder / "case.json").write_text(json.dumps(meta, indent=2))
    print(f"[case] results -> {folder / 'results.vtm'}  Cd={coeffs['Cd']:.4f} Cl={coeffs['Cl']:.4f}  "
          f"y+ max {w['yplus'].max():.2f}")
    if pngs:
        export_pngs(folder)
    return s, coeffs


def run_transient(folder, fvm, mesh, patches, U_inf, nu, meta, dt, t_end, write_every=1.0,
                  kick=False, max_outer=20, tol=1e-5, pngs=True):
    """Time-accurate run (BDF2 + outer iterations per step).
    history.csv: one row per time step. snapshots/: fields every write_every
    time units, tied together by results.pvd (ParaView: open it, press play).
    kick: spin the body briefly early on (t = 2..6) to trigger vortex shedding
    from a symmetric start, like a small physical disturbance."""
    from validate_unsteady import shedding_stats
    folder = Path(folder)
    q = write_mesh(folder, fvm, mesh, patches, meta)
    print(f"[case] mesh written -> {folder / 'mesh.vtm'}")
    wall_U = (0.0, 0.0)
    if kick:
        P = fvm.points
        cx, cy = P[np.unique(patches["wall"])].mean(0)

        def wall_U(x, y, t):
            if 2.0 <= t <= 6.0:
                return np.c_[-0.5 * (y - cy), 0.5 * (x - cx)]
            return np.zeros((len(x), 2))
    bcs = {"wall": {"type": "wall", "U": wall_U}, "inlet": {"type": "inlet", "U": U_inf},
           "outlet": {"type": "outlet"}}
    s = SimpleSolver(fvm, bcs, nu, U0=U_inf, alpha_u=0.9)
    snap_dir = folder / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    for old in snap_dir.glob("*.vtu"):
        old.unlink()
    hist = open(folder / "history.csv", "w", buffering=1)
    hist.write("step,time,outer,cont,Ux,Uy,Cd,Cl,time_s\n")
    t0 = time.time()
    entries, T, CD, CL = [], [], [], []
    nsteps = int(round(t_end / dt))
    next_write = write_every
    for i in range(1, nsteps + 1):
        k, r = s.advance(dt, max_outer=max_outer, tol=tol)
        _, c = wall_data(s, "wall", U_inf)
        T.append(s.time); CD.append(c["Cd"]); CL.append(c["Cl"])
        hist.write(f"{i},{s.time:.6g},{k},{r['cont']:.3e},{r['Ux']:.3e},{r['Uy']:.3e},"
                   f"{c['Cd']:.8g},{c['Cl']:.8g},{time.time() - t0:.1f}\n")
        if s.time >= next_write - 0.5 * dt:
            name = f"snapshots/fields_{len(entries):05d}.vtu"
            write_vtu(folder / name, fvm.points, mesh["cells"],
                      cell_data=flow_fields(s, U_inf=np.linalg.norm(U_inf)))
            entries.append((s.time, name))
            write_pvd(folder / "results.pvd", entries)
            next_write += write_every
        if i % 200 == 0 or i == 1:
            print(f"  step {i:6d}  t {s.time:8.3f}  outer {k:2d}  Cd {c['Cd']:.4f}  "
                  f"Cl {c['Cl']:+.4f}  ({time.time() - t0:.0f}s)", flush=True)
    hist.close()
    w, coeffs = write_results(folder, s, fvm, mesh, patches, U_inf, q)
    st = shedding_stats(np.array(T), np.array(CD), np.array(CL))
    meta.update(transient={"dt": dt, "t_end": t_end, "steps": nsteps, "snapshots": len(entries),
                           "kick": kick}, runtime_s=round(time.time() - t0, 1),
                coefficients_final={k: float(v) for k, v in coeffs.items()},
                shedding={k: v for k, v in st.items()},
                yplus_max=float(w["yplus"].max()))
    (folder / "case.json").write_text(json.dumps(meta, indent=2))
    if st["St"] is not None:
        print(f"[case] shedding: Strouhal {st['St']:.4f}, mean Cd {st['Cd_mean']:.4f}, "
              f"Cl amplitude {st['Cl_amp']:.4f}")
    else:
        print(f"[case] no periodic shedding detected ({st['periods']} lift up-crossings)")
    print(f"[case] snapshots -> {folder / 'results.pvd'} ({len(entries)} frames)")
    if pngs:
        export_pngs(folder, transient=True)
    return s, st


# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["mesh", "run"])
    ap.add_argument("kind", choices=["naca", "cylinder"])
    ap.add_argument("--out", required=True, help="case folder")
    ap.add_argument("--naca", default="0012")
    ap.add_argument("--re", type=float, default=None,
                    help="Reynolds number, chord / diameter (default: naca 1e6, cylinder 40)")
    ap.add_argument("--yplus", type=float, default=0.7, help="turbulent airfoil: target y+")
    ap.add_argument("--alpha", type=float, default=0.0, help="angle of attack (deg)")
    ap.add_argument("--farfield", type=float, default=None, help="farfield radius")
    ap.add_argument("--refine", type=float, default=1.0,
                    help="mesh refinement factor (cylinder, and laminar airfoils)")
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--transient", action="store_true", help="time-accurate (unsteady) run")
    ap.add_argument("--dt", type=float, default=0.05, help="time step (units of L/U)")
    ap.add_argument("--t-end", type=float, default=100.0, help="end time (units of L/U)")
    ap.add_argument("--write-every", type=float, default=1.0,
                    help="snapshot interval in time units (transient)")
    ap.add_argument("--no-kick", action="store_true",
                    help="transient cylinder: do not spin it briefly to trigger shedding")
    ap.add_argument("--no-png", action="store_true", help="skip the automatic pictures")
    ap.add_argument("--turbulence", choices=["auto", "laminar", "sa"], default="auto",
                    help="auto: Spalart-Allmaras for airfoils at Re >= 1e5, else laminar")
    ap.add_argument("--no-vortex", action="store_true",
                    help="turbulent airfoil: no point-vortex farfield correction")
    a = ap.parse_args(argv)
    if a.re is None:
        a.re = 1e6 if a.kind == "naca" else 40.0
    if a.farfield is None:
        a.farfield = (40.0 if a.re < LAMINAR_RE else 100.0) if a.kind == "naca" else 60.0
    if a.turbulence == "auto":
        a.turbulence = "sa" if (a.kind == "naca" and a.re >= LAMINAR_RE) else "laminar"
    turb = None if a.turbulence == "laminar" else a.turbulence

    fvm, mesh, patches, U_inf, label = build(a.kind, a)
    meta = {"case": label, "kind": a.kind, "Re": a.re, "alpha_deg": a.alpha,
            "U_inf": list(U_inf), "nu": 1.0 / a.re, "ref_length": 1.0,
            "bcs": {"wall": "no-slip wall", "inlet": "fixed velocity U_inf",
                    "outlet": "zero-gradient U, p = 0"},
            "turbulence": "Spalart-Allmaras (SA-noft2), nuTilde_inf = 3 nu" if turb else "laminar"}
    vortex = turb is not None and a.kind == "naca" and not a.no_vortex
    if vortex:
        meta["bcs"]["inlet"] = "U_inf + point-vortex correction (Gamma = Cl U c / 2)"
    if a.action == "mesh":
        write_mesh(a.out, fvm, mesh, patches, meta)
        print(f"[case] mesh written -> {Path(a.out) / 'mesh.vtm'}")
    elif a.transient:
        run_transient(a.out, fvm, mesh, patches, U_inf, 1.0 / a.re, meta, dt=a.dt,
                      t_end=a.t_end, write_every=a.write_every,
                      kick=(a.kind == "cylinder" and not a.no_kick), pngs=not a.no_png)
    else:
        run_case(a.out, fvm, mesh, patches, U_inf, 1.0 / a.re, meta, tol=a.tol,
                 max_iter=a.max_iter, pngs=not a.no_png, turbulence=turb, vortex=vortex)


if __name__ == "__main__":
    main()
