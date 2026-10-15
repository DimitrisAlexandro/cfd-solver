"""
Validation (solver step C): turbulent NACA 0012, Re = 6e6, Spalart-Allmaras,
NASA TMR case "2D NACA 0012 Airfoil Validation"
(tmbwg.github.io/turbmodels/naca0012_val.html).

  alpha = 0, 10, 15 deg, fully turbulent (SA-noft2, nuT_inf = 3 nu),
  compared with CFL3D SA on the 897x257 grid (validation/reference/tmr/
  n0012*_cfl3d_sa.dat) and the experiments of Ladson (Cl, Cd, tripped,
  Re 6e6) and Gregory & O'Reilly (Cp, Re 2.88e6).

Geometry: the TMR NACA 0012 definition (sharp trailing edge, chord 1),
    y = +-0.594689181 [0.298222773 sqrt(x) - 0.127125232 x - 0.357907906 x^2
                       + 0.291984971 x^3 - 0.105174606 x^4]
truncated where its thickness is 0.1 % chord and rescaled back to chord 1
(a tiny blunt base keeps the boundary-layer extrusion clean; the change
in shape is < 0.3 % of chord at the trailing edge).

Farfield: circle of radius 100 c. The inflow half has a Dirichlet velocity
U_inf plus the point-vortex correction Gamma = Cl U_inf c / 2 at the quarter
chord (updated during the iterations, as in the CFL3D reference results).

    python validate_naca0012_sa.py               alpha 0, 10, 15 on level 1
    python validate_naca0012_sa.py --grid        + levels 0.7 / 1.4 at alpha 0 and 10
    python validate_naca0012_sa.py --plot        plots only
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import hybrid_mesher as hm
from fvmesh import FVMesh
from geometry import first_cell_height, signed_area
from postproc import wall_data
from simple_solver import SimpleSolver
from validate_flatplate import tmr_zones

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "validation"
TMR = VAL / "reference" / "tmr"
RESULTS = VAL / "naca0012_sa_results.json"
RE = 6e6
FARFIELD = 100.0
TE_THICK = 0.001
# grid-converged SA values at alpha 10 (CFL3D / FUN3D, grid family II, point-
# vortex farfield; tmbwg.github.io/turbmodels/naca0012numerics_val_sa_withpv.html)
CONTINUUM_A10 = {"Cl": 1.0912, "Cd": 0.01221, "Cd_p": 0.00601, "Cd_v": 0.006204}


def tmr_thickness(x):
    return 0.594689181 * (0.298222773 * np.sqrt(x) - 0.127125232 * x - 0.357907906 * x**2
                          + 0.291984971 * x**3 - 0.105174606 * x**4)


def tmr0012(n_side=160, te_blend=0.85, n_te=3, te_thick=TE_THICK):
    """Closed CCW loop of the TMR NACA 0012 with a te_thick blunt base."""
    xt = brentq(lambda x: 2 * tmr_thickness(x) - te_thick, 0.9, 1.0)
    beta = np.linspace(0.0, np.pi, n_side)
    s = te_blend * 0.5 * (1 - np.cos(beta)) + (1 - te_blend) * (1 - np.cos(beta / 2))
    x = s * xt
    y = tmr_thickness(x)
    x = x / xt                                           # chord 1 again
    upper = np.c_[x, y][::-1]
    lower = np.c_[x, -y][1:]
    t = np.linspace(0, 1, n_te + 1)[1:-1]
    base = np.c_[np.ones_like(t), -y[-1] + t * 2 * y[-1]]
    loop = np.vstack([upper, lower, base])
    return loop if signed_area(loop) > 0 else loop[::-1]


def airfoil_mesh(level=1.0, alpha=0.0, verbose=False, wall=None, naca=None, re=RE,
                 farfield=FARFIELD, yplus=0.7, growth=1.15, size_scale=None):
    """Turbulent (y+ < 1) hybrid airfoil mesh, the settings validated here.
    Default body: the TMR NACA 0012; pass naca="2412" or wall=(M,2) for others."""
    r = level
    if wall is None and naca is None:
        wall, label = tmr0012(int(round(160 * r))), "TMR NACA 0012"
    elif wall is None:
        from geometry import naca4
        wall, label = naca4(naca, int(round(160 * r))), f"NACA {naca}"
    else:
        label = "custom airfoil"
    return hm.build_mesh(wall=wall, reynolds=re,
                         h1=first_cell_height(re, yplus) / r, growth=growth ** (1 / r),
                         n_layers=int(round(60 * r * math.log(1.15) / math.log(growth))),
                         bl_ar_cap=1.0,
                         farfield_radius=farfield, n_farfield=int(round(48 * r)),
                         h_near=0.03 / r, grad=0.15 / r, wake_length=3.0,
                         wake_h=0.015 / r, wake_grad=0.05 / r, alpha_deg=alpha,
                         label=f"{label} (level {r:g})", verbose=verbose,
                         size_scale=size_scale)


class VortexFarfield:
    """Point-vortex farfield correction, Gamma = Cl U c / 2 at the quarter
    chord: velocity U_inf + u_vortex on the inflow faces, Bernoulli pressure
    p = (|U_inf|^2 - |U_inf + u_vortex|^2)/2 on the outflow faces (without the
    outlet part the lift still depended on the farfield radius: +0.4 % from
    100 c to 400 c). Call update(it) from the solver callback."""
    def __init__(self, solver, alpha, patch="inlet", outlet="outlet", wall="wall", every=50,
                 center=(0.25, 0.0)):
        """wall: one wall patch or a list (multi-element airfoil: the vortex
        carries the total lift; center = the moment reference point)."""
        self.s, self.alpha, self.every = solver, alpha, every
        self.walls = [wall] if isinstance(wall, str) else list(wall)
        self.center = center
        self.idx = solver.m.patch_faces[patch]
        self.out = solver.m.patch_faces.get(outlet, np.zeros(0, int))
        self.U = (math.cos(math.radians(alpha)), math.sin(math.radians(alpha)))
        self.gamma = 0.0

    def update(self, it, res=None):
        if it % self.every == 0:
            self.gamma = 0.5 * sum(wall_data(self.s, w, self.U)[1]["Cl"] for w in self.walls)
            self.s.Ub[self.idx] = vortex_velocity(self.s.m.bxf[self.idx], self.alpha, self.gamma,
                                                  self.center)
            uo = vortex_velocity(self.s.m.bxf[self.out], self.alpha, self.gamma, self.center)
            self.s.pb_out[self.out] = 0.5 * (1.0 - (uo**2).sum(1))


def vortex_velocity(xy, alpha, gamma, center=(0.25, 0.0)):
    """U_inf + point vortex (circulation gamma, clockwise for lift > 0) at center."""
    a = math.radians(alpha)
    dx, dy = xy[:, 0] - center[0], xy[:, 1] - center[1]
    r2 = dx**2 + dy**2
    k = gamma / (2 * math.pi * r2)
    return np.c_[math.cos(a) + k * dy, math.sin(a) - k * dx]


def run(alpha=10.0, level=1.0, tol=1e-7, max_iter=12000, verbose=True, vortex=True,
        return_solver=False):
    t0 = time.time()
    mesh = airfoil_mesh(level, alpha, verbose=verbose)
    U = (math.cos(math.radians(alpha)), math.sin(math.radians(alpha)))
    P = mesh["points"]
    ff = mesh["patches"]["farfield"]
    t = P[ff[:, 1]] - P[ff[:, 0]]
    inflow = t[:, 1] * U[0] - t[:, 0] * U[1] < 0
    patches = {"wall": mesh["patches"]["airfoil"], "inlet": ff[inflow], "outlet": ff[~inflow]}
    fvm = FVMesh(P, mesh["cells"], patches)
    bcs = {"wall": {"type": "wall"}, "inlet": {"type": "inlet", "U": U},
           "outlet": {"type": "outlet"}}
    s = SimpleSolver(fvm, bcs, 1.0 / RE, U0=U, turbulence="sa")
    vf = VortexFarfield(s, alpha) if vortex else None
    it = s.solve(tol=tol, max_iter=max_iter, verbose=verbose, every=500,
                 callback=vf.update if vf else None)
    w, c = wall_data(s, "wall", U)
    o = np.lexsort((w["x"], w["side"]))
    res = {"alpha": alpha, "level": level, "cells": int(fvm.ncells), "n_bl": int(mesh["n_bl"]),
           "iterations": int(it), "converged": bool(s.converged),
           "time_s": round(time.time() - t0, 1), "Cl": float(c["Cl"]), "Cd": float(c["Cd"]),
           "Cd_p": float(c["Cd_pressure"]), "Cd_v": float(c["Cd_viscous"]),
           "yplus_max": float(w["yplus"].max()), "yplus_mean": float(w["yplus"].mean()),
           "nut_ratio_max": float((s.turb.nu_t() * RE).max()),
           "wall": {k: np.asarray(w[k])[o].round(7).tolist() for k in ("x", "side", "Cp", "Cf")}}
    if return_solver:
        return res, s, mesh
    return res


def reference():
    clcd = next(iter(tmr_zones(TMR / "n0012clcd_cfl3d_sa.dat").values()))
    return {float(a): (cl, cd) for a, cl, cd in clcd}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    db = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    jobs = [(a, 1.0) for a in (0.0, 10.0, 15.0)]
    if "--grid" in sys.argv:
        jobs += [(a, lv) for a in (0.0, 10.0) for lv in (0.7, 1.4)]
    if "--plot" not in sys.argv:
        for a, lv in jobs:
            k = f"a{a:g}_L{lv:g}"
            if k in db:
                continue
            print(f"== alpha {a:g}, level {lv:g}", flush=True)
            db[k] = run(a, lv)
            RESULTS.write_text(json.dumps(db))
    ref = reference()
    print(f"alpha 10 grid-converged SA (CFL3D/FUN3D): Cl {CONTINUUM_A10['Cl']}  Cd {CONTINUUM_A10['Cd']}"
          f"  (Cd_p {CONTINUUM_A10['Cd_p']}, Cd_v {CONTINUUM_A10['Cd_v']})")
    for k in sorted(db, key=lambda k: (db[k]["alpha"], db[k]["cells"])):
        r = db[k]
        print(f"  {k:>10s}: Cd_p {r['Cd_p']:.5f}  Cd_v {r['Cd_v']:.5f}")
    print(f"{'case':>12s} {'cells':>7s} {'Cl':>8s} {'CFL3D':>8s} {'dCl':>7s} {'Cd':>9s} "
          f"{'CFL3D':>9s} {'dCd':>7s} {'y+max':>6s} {'iters':>6s}")
    for k in sorted(db, key=lambda k: (db[k]["alpha"], db[k]["cells"])):
        r = db[k]
        cl0, cd0 = ref[r["alpha"]]
        dcl = f"{100 * (r['Cl'] / cl0 - 1):+6.1f}%" if abs(cl0) > 1e-3 else f"{r['Cl'] - cl0:+.4f}"
        print(f"{k:>12s} {r['cells']:7d} {r['Cl']:8.4f} {cl0:8.4f} {dcl:>7s} {r['Cd']:9.5f} "
              f"{cd0:9.5f} {100 * (r['Cd'] / cd0 - 1):+6.1f}% {r['yplus_max']:6.2f} "
              f"{r['iterations']:6d}{'' if r['converged'] else ' (not converged)'}")

    # ---- plots ---------------------------------------------------------------
    cps = tmr_zones(TMR / "n0012cp_cfl3d_sa.dat")
    cfs = tmr_zones(TMR / "n0012cf_cfl3d_sa.dat")
    gcp = tmr_zones(TMR / "CP_Gregory_expdata.dat")
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    for j, a in enumerate((0.0, 10.0, 15.0)):
        k = f"a{a:g}_L1"
        zc = cps.get(f"alpha={a:g}")
        if zc is not None:
            ax[0, j].plot(zc[:, 0], zc[:, 1], "k-", lw=2, alpha=.4, label="CFL3D SA (897x257)")
        ze = gcp.get(f"alpha={a:g}")
        if ze is not None:
            ax[0, j].plot(ze[:, 0], ze[:, 1], "ks", mfc="none", ms=5, label="Gregory & O'Reilly (exp.)")
        zf = cfs.get(f"alpha={a:g}, upper surface")
        if zf is not None:
            ax[1, j].plot(zf[:, 0], zf[:, 1], "k-", lw=2, alpha=.4, label="CFL3D SA, upper")
        if k in db:
            w = {q: np.asarray(v) for q, v in db[k]["wall"].items()}
            for side, st, nm in ((1, "C0-", "upper"), (-1, "C1-", "lower")):
                m = (w["side"] == side) & (w["x"] < 0.9999)     # base faces excluded
                ax[0, j].plot(w["x"][m], w["Cp"][m], st, lw=1.2, label=f"this solver, {nm}")
                ax[1, j].plot(w["x"][m], w["Cf"][m], st, lw=1.2, label=f"this solver, {nm}")
        ax[0, j].invert_yaxis(); ax[0, j].set_title(f"Cp, alpha = {a:g} deg")
        ax[1, j].set_title(f"Cf, alpha = {a:g} deg"); ax[1, j].set_ylim(-0.002, 0.012)
        for q in (0, 1):
            ax[q, j].set_xlabel("x/c"); ax[q, j].grid(alpha=.3); ax[q, j].legend(fontsize=7)
    fig.suptitle("NACA 0012, Re = 6e6, Spalart-Allmaras")
    fig.tight_layout(); fig.savefig(VAL / "naca0012_sa_cp_cf.png", dpi=130)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    lad = tmr_zones(TMR / "CLCD_Ladson_expdata.dat")
    for (name, z), mk in zip(lad.items(), ("o", "s", "^")):
        ax[0].plot(z[:, 0], z[:, 1], mk, mfc="none", color="gray", ms=5, label=f"Ladson {name}")
        ax[1].plot(z[:, 1], z[:, 2], mk, mfc="none", color="gray", ms=5, label=f"Ladson {name}")
    ra = sorted(ref)
    ax[0].plot(ra, [ref[a][0] for a in ra], "k*-", ms=10, label="CFL3D SA")
    ax[1].plot([ref[a][0] for a in ra], [ref[a][1] for a in ra], "k*-", ms=10, label="CFL3D SA")
    mine = sorted((db[k] for k in db if db[k]["level"] == 1.0), key=lambda r: r["alpha"])
    ax[0].plot([r["alpha"] for r in mine], [r["Cl"] for r in mine], "rD-", label="this solver")
    ax[1].plot([r["Cl"] for r in mine], [r["Cd"] for r in mine], "rD-", label="this solver")
    ax[0].set_xlabel("alpha (deg)"); ax[0].set_ylabel("Cl"); ax[0].set_xlim(-1, 18)
    ax[1].set_xlabel("Cl"); ax[1].set_ylabel("Cd"); ax[1].set_xlim(-0.1, 1.8); ax[1].set_ylim(0, 0.03)
    # grid convergence of Cd (h = 1/sqrt(cells)), relative to CFL3D
    for a, st in ((0.0, "o-"), (10.0, "s-")):
        rs = sorted((r for r in db.values() if r["alpha"] == a), key=lambda r: r["cells"])
        if len(rs) > 1:
            ax[2].plot([1 / math.sqrt(r["cells"]) for r in rs],
                       [100 * (r["Cd"] / ref[a][1] - 1) for r in rs], st,
                       label=f"Cd, alpha = {a:g}")
            ax[2].plot([1 / math.sqrt(r["cells"]) for r in rs],
                       [100 * (r["Cl"] / ref[a][0] - 1) if a else np.nan for r in rs], st[0] + "--",
                       mfc="none", label=f"Cl, alpha = {a:g}" if a else None)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].plot(0, 100 * (CONTINUUM_A10["Cd"] / ref[10.0][1] - 1), "g*", ms=14,
               label="alpha 10: NASA grid-converged Cd (h -> 0)")
    ax[2].plot(0, 100 * (CONTINUUM_A10["Cl"] / ref[10.0][0] - 1), "r*", ms=14, mfc="none",
               label="alpha 10: NASA grid-converged Cl (h -> 0)")
    ax[2].set_xlim(left=0); ax[2].set_xlabel("h = 1/sqrt(cells)")
    ax[2].set_ylabel("difference to CFL3D 897x257 (%)"); ax[2].set_title("Grid convergence")
    for q in ax:
        q.grid(alpha=.3); q.legend(fontsize=7)
    fig.suptitle("NACA 0012, Re = 6e6: lift curve, drag polar, grid convergence")
    fig.tight_layout(); fig.savefig(VAL / "naca0012_sa_polar.png", dpi=130)
    print("saved", VAL / "naca0012_sa_cp_cf.png", "and", VAL / "naca0012_sa_polar.png")


if __name__ == "__main__":
    main()
