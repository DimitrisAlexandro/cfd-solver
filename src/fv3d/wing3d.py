"""3D RANS of a wing: mesh, solve, forces and span loading.

    python -m fv3d.wing3d --airfoil naca0012 --semi-span 3 --re 3e6 --alpha 0,4,8 --level 0.5 --out ../cases/wing3d_demo

Outputs (out/): wing3d.json (CL, CD, Cm, per angle: span loading, residual history),
report.txt. Coefficients on the half-wing planform area (= full-wing coefficients).
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from .mesh3d import FVMesh3D
from .solver3d import Solver3D
from .wingmesh import wing_mesh


def coefficients(s, info, alpha):
    Fp, Fv = s.force("wing")
    F = Fp + Fv
    ca, sa = math.cos(math.radians(alpha)), math.sin(math.radians(alpha))
    q = 0.5
    S = info["S_half"]
    L = -F[0] * sa + F[1] * ca
    D = F[0] * ca + F[1] * sa
    Dp = Fp[0] * ca + Fp[1] * sa
    # pitching moment about the root quarter chord (nose up +): -Mz
    M = s.moment("wing", ref=(info["x_ref"], 0.0, 0.0))
    return {"CL": L / (q * S), "CD": D / (q * S), "CDp": Dp / (q * S), "CDv": (D - Dp) / (q * S),
            "Cm": -M[2] / (q * S * info["mac"]), "CS": F[2] / (q * S)}


def span_loading(s, info, alpha, nbins=None):
    """sectional lift coefficient c_l(eta) and c_l c / c_ref from strips of wing faces"""
    m = s.m
    idx, pb, tau, S, Smag = s.wall_values("wing")
    F = pb[:, None] * S + tau * Smag[:, None]
    zf = m.bxf[idx, 2]
    ca, sa = math.cos(math.radians(alpha)), math.sin(math.radians(alpha))
    lift = -F[:, 0] * sa + F[:, 1] * ca
    zl = np.array(info["z_levels"][:info["itip"] + 1])
    k = np.clip(np.searchsorted(zl, zf) - 1, 0, len(zl) - 2)
    on_side = zf < zl[-1] - 1e-9                            # exclude the tip cap
    Ls = np.bincount(k[on_side], lift[on_side], len(zl) - 1)
    dz = np.diff(zl)
    zc = 0.5 * (zl[1:] + zl[:-1])
    c = info["chord"](zc)
    ccl = Ls / dz / 0.5                                      # c_l * c  (c_ref = root chord = 1)
    return {"eta": (zc / info["semi_span"]).tolist(), "cl": (ccl / c).tolist(), "clc": ccl.tolist()}


def section_start(info, alpha, re, max_iter=1500, log=print):
    """2D SA solution of the root section at alpha on the wing's own section mesh,
    mapped onto the 3D cells: (U, p, nuTilde) to start the 3D run from. Inboard of
    the tip every cell takes its 2D cell's values; beyond the tip they fade to the
    free stream over one chord; the tip fill starts at the free stream. A cold start
    from a uniform field drives the blunt trailing-edge wake through a violent
    transient that can diverge; the 2D field has it already settled."""
    from fvmesh import FVMesh
    from simple_solver import SimpleSolver
    sm = info["section_mesh"]
    P, ff = sm["points"], sm["farfield"]
    ca, sa = math.cos(math.radians(alpha)), math.sin(math.radians(alpha))
    t = P[ff[:, 1]] - P[ff[:, 0]]
    inflow = t[:, 1] * ca - t[:, 0] * sa < 0
    fvm = FVMesh(P, sm["cells"], {"wall": sm["airfoil"], "inlet": ff[inflow], "outlet": ff[~inflow]})
    bc2 = {"wall": {"type": "wall"}, "inlet": {"type": "inlet", "U": (ca, sa)}, "outlet": {"type": "outlet"}}
    t0 = time.time()
    # first-order upwind settles the wake of the blunt trailing edge from the uniform
    # start (the coarse wing section mesh diverges otherwise), then second order
    s2 = SimpleSolver(fvm, bc2, 1.0 / re, turbulence="sa", U0=(ca, sa), second_order=False)
    it = s2.solve(tol=1e-6, max_iter=300, verbose=False)
    keep = (s2.U.copy(), s2.p.copy(), s2.turb.nuT.copy())
    s2.second_order = True
    it += s2.solve(tol=1e-6, max_iter=max_iter, verbose=False)
    if not (np.isfinite(s2.U).all() and np.abs(s2.U).max() < 10):
        s2.U, s2.p, s2.turb.nuT = keep                    # second order did not settle: first order
    log(f"[wing3d] 2D section start at alpha {alpha:g}: {it} it, {time.time() - t0:.0f}s")
    c2, zc = info["cell_2d"], info["z_cell"]
    sec = c2 < info["n_section_cells"]
    j = np.where(sec, c2, 0)
    fade = np.where(sec, np.exp(-np.maximum(zc - info["semi_span"], 0.0) / max(info["taper"], 0.2)), 0.0)
    Uinf = np.array([ca, sa])
    U = np.zeros((len(c2), 3))
    U[:, :2] = fade[:, None] * s2.U[j] + (1 - fade[:, None]) * Uinf
    p = fade * s2.p[j]
    nut_inf = 3.0 / re
    nuT = fade * s2.turb.nuT[j] + (1 - fade) * nut_inf
    return U, p, nuT


def save_fields(path, s, info, pm, alpha, coeffs):
    """the solution for the results viewer (postview.Case3D): points, the extrusion map,
    cell fields (float32) and the wing surface with Cp / Cf"""
    m = s.m
    c2 = info["cells2"]
    W = max(len(c) for c in c2)
    C2 = np.full((len(c2), W), -1, np.int32)
    for i, c in enumerate(c2):
        C2[i, :len(c)] = c
    idx, pb, tau, S, Smag = s.wall_values("wing")
    faces = m.bfaces[idx]
    g = s.gU                                               # g[k][:, j] = d u_k / d x_j
    vort = np.c_[g[2][:, 1] - g[1][:, 2], g[0][:, 2] - g[2][:, 0], g[1][:, 0] - g[0][:, 1]]
    keep = {k: v for k, v in info.items() if isinstance(v, (int, float, str))}
    meta = dict(keep, alpha=float(alpha), nu=float(s.nu), **{k: float(v) for k, v in coeffs.items()})
    np.savez_compressed(
        path, info=json.dumps(meta), points=np.asarray(pm["points"], np.float32), n2=info["n2"],
        cells2=C2, cell_layer=info["cell_layer"].astype(np.int32), cell_2d=info["cell_2d"].astype(np.int32),
        z_levels=np.asarray(info["z_levels"], np.float32), centres=m.xc.astype(np.float32),
        U=s.U.astype(np.float32), p=s.p.astype(np.float32), vorticity=vort.astype(np.float32),
        nut=(s.turb.nu_t() if s.turb is not None else np.zeros(m.ncells)).astype(np.float32),
        wall_faces=faces.astype(np.int32), wall_centres=m.bxf[idx].astype(np.float32),
        wall_Cp=(2.0 * pb).astype(np.float32), wall_Cf=(2.0 * np.linalg.norm(tau, axis=1)).astype(np.float32))


def converged(hist, force_tol):
    """forces settled: the mean CL and CD of the last 200 iterations differ from those
    of the 200 before by less than force_tol (relative, floors 0.3 and 0.01). Means,
    not the min-max band: a steady RANS solution often ends in a tiny limit cycle
    (here +-0.0001 in CL) that a band test never accepts."""
    w = 8                                                  # samples every 25 iterations
    a, b = hist[-2 * w:-w], hist[-w:]
    m = lambda h, k: sum(x[k] for x in h) / len(h)
    band = max(x["CL"] for x in b) - min(x["CL"] for x in b)
    return (abs(m(b, "CL") - m(a, "CL")) < force_tol * max(abs(m(b, "CL")), 0.3)
            and abs(m(b, "CD") - m(a, "CD")) < force_tol * max(abs(m(b, "CD")), 0.01)
            and band < 0.02 * max(abs(m(b, "CL")), 0.3))


def run(airfoil="naca0012", semi_span=3.0, taper=1.0, sweep_c4=0.0, dihedral=0.0, twist=0.0, re=3e6,
        alphas=(0.0, 4.0, 8.0), level=0.5, farfield=12.0, max_iter=3000, tol=1e-6, force_tol=1e-3,
        out=None, log=print, start_2d=True):
    t0 = time.time()
    pm, info = wing_mesh(airfoil, semi_span, taper, sweep_c4, dihedral, twist, re=re, level=level,
                         farfield=farfield)
    m = FVMesh3D(pm)
    q = m.quality()
    log(f"[wing3d] mesh {m.ncells} cells ({info['layers']} spanwise layers), {time.time() - t0:.1f}s; "
        f"non-orthogonality max {q['nonortho_max']:.0f} deg")
    res, init = [], None
    for a in alphas:
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        U = (ca, sa, 0.0)
        s = Solver3D(m, {"wing": {"type": "wall"}, "symmetry": {"type": "symmetry"},
                         "farfield": {"type": "farfield", "U": U}}, 1.0 / re, U0=U, turbulence="sa")
        if init is None and start_2d:
            s.set_initial(*section_start(info, a, re, log=log))
        if init is not None:                                 # warm start, velocity rotated
            ca0, sa0 = math.cos(math.radians(init[0])), math.sin(math.radians(init[0]))
            Ux, Uy = init[1][:, 0], init[1][:, 1]
            # rotate by the angle difference
            da = math.radians(a - init[0])
            Ur = np.c_[Ux * math.cos(da) - Uy * math.sin(da), Ux * math.sin(da) + Uy * math.cos(da), init[1][:, 2]]
            s.set_initial(Ur, init[2], init[3])
        hist = []
        t1 = time.time()

        def cb(it, r):
            if it % 25 == 0:
                hist.append({"it": it, **{k: float(v) for k, v in r.items()}, **coefficients(s, info, a)})
                if it % 100 == 0:
                    h = hist[-1]
                    log(f"  alpha {a:g}: it {it}  CL {h['CL']:.5f}  CD {h['CD']:.5f}  cont {r['cont']:.1e}  "
                        f"nuT {r.get('nuT', 0):.1e}  ({time.time() - t1:.0f}s)")
                if len(hist) >= 16 and r["cont"] < 1e-3 and converged(hist, force_tol):
                    return True
            return False
        it = s.solve(tol=tol, max_iter=max_iter, verbose=False, callback=cb)
        c = coefficients(s, info, a)
        sl = span_loading(s, info, a)
        ok = s.converged
        res.append({"alpha": a, **c, "iterations": it, "time_s": round(time.time() - t1, 1),
                    "status": "converged" if ok else "not converged", "span": sl, "history": hist})
        log(f"[wing3d] alpha {a:g}: CL {c['CL']:.4f}  CD {c['CD']:.5f} (pressure {c['CDp']:.5f})  Cm {c['Cm']:+.4f}  "
            f"{it} it, {time.time() - t1:.0f}s, {'converged' if ok else 'NOT converged'}")
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
            save_fields(Path(out) / f"fields_a{a:g}.npz", s, info, pm, a, c)
        init = (a, s.U.copy(), s.p.copy(), s.turb.nuT.copy())
    info_out = {k: v for k, v in info.items()
                if k not in ("chord", "section", "z_levels", "section_mesh", "cell_layer", "cell_2d", "z_cell", "cells2")}
    out_d = {"info": info_out, "mesh": q, "results": res}
    if out:
        o = Path(out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "wing3d.json").write_text(json.dumps(out_d, indent=1))
        lines = [f"3D RANS (SA) half wing: {airfoil}, AR {info['aspect_ratio']:.2f}, taper {taper:g}, sweep {sweep_c4:g}, "
                 f"dihedral {dihedral:g}, twist {twist:g}; Re {re:.3g} (root chord); {m.ncells} cells", "",
                 "alpha      CL        CD       CDp       Cm     iterations  status"]
        lines += [f"{r['alpha']:5.1f}  {r['CL']:8.4f}  {r['CD']:8.5f}  {r['CDp']:8.5f}  {r['Cm']:+8.4f}  {r['iterations']:6d}  "
                  f"{r['status']}" for r in res]
        (o / "report.txt").write_text("\n".join(lines) + "\n")
    return out_d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--airfoil", default="naca0012")
    ap.add_argument("--semi-span", type=float, default=3.0, help="in root chords")
    ap.add_argument("--taper", type=float, default=1.0)
    ap.add_argument("--sweep", type=float, default=0.0, help="quarter-chord sweep (deg)")
    ap.add_argument("--dihedral", type=float, default=0.0)
    ap.add_argument("--twist", type=float, default=0.0, help="tip incidence relative to root (deg)")
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--alpha", default="0,4,8")
    ap.add_argument("--level", type=float, default=0.5)
    ap.add_argument("--farfield", type=float, default=12.0)
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=0,
                    help="solver threads (default: half the logical cores, i.e. the physical ones on most PCs)")
    a = ap.parse_args(argv)
    # one thread team only: the C++ kernels use OpenMP; BLAS / MKL thread pools that
    # also spin up (numpy, pypardiso) would oversubscribe the cores (WSL: ~2x slower)
    if a.threads:
        os.environ["AEROLAB_THREADS"] = str(a.threads)
    elif "AEROLAB_THREADS" not in os.environ:
        os.environ["AEROLAB_THREADS"] = str(max(1, (os.cpu_count() or 2) // 2))
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    run(a.airfoil, a.semi_span, a.taper, a.sweep, a.dihedral, a.twist, a.re,
        [float(x) for x in a.alpha.split(",")], a.level, a.farfield, a.max_iter, out=a.out)


if __name__ == "__main__":
    main()
