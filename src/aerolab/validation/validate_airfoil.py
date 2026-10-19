"""
Validation 4 (solver step A): steady laminar flow over a NACA 0012 airfoil,
using our hybrid mesher + our SIMPLEC solver.

Main case Re = 1000 (steady below alpha ~ 8 deg, attached below ~ 4 deg),
plus Re = 2000 and 5000 at small alpha.

References
  XFOIL (Drela), run on the SAME blunt-TE coordinates     -> xfoil_ref.py
  Di Ilio et al. (2020), arXiv:2006.10487 - lattice-Boltzmann Navier-Stokes,
      Cd(alpha=0): Re 1000: 0.119, Re 2000: 0.084, Re 5000: 0.052
      (their XFOIL: 0.119, 0.084, 0.054); steady below alpha = 8 deg,
      no separation below 4 deg
  Kurtulus (2015), Int. J. Micro Air Vehicles 7(3) - Fluent, Re 1000:
      Cd(0) ~ 0.12, lift slope ~ pi per radian near alpha = 0

Studies
  domain : farfield radius 20 / 40 / 80 chords        (Re 1000, alpha 4)
  grid   : three mesh levels                          (Re 1000, alpha 0 and 4)
  sweep  : alpha = 0..7 deg                           (Re 1000)
  extra  : Re 2000 and 5000                           (alpha 0, 2)

All results are stored in ../validation/airfoil_results.json, so plots can
be remade without re-running:
    python src/run.py validate_airfoil                 # everything (~2-3 h)
    python src/run.py validate_airfoil --study grid    # one study
    python src/run.py validate_airfoil --plot          # plots from saved results
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from aerolab.mesh import hybrid_mesher as hm
from aerolab.core.fvmesh import FVMesh
from aerolab.core.postproc import wall_data
from aerolab.core.simple_solver import SimpleSolver

ROOT = Path(__file__).resolve().parents[3]
VAL = ROOT / "validation"
REF = VAL / "reference"
RESULTS = VAL / "airfoil_results.json"

LITERATURE_CD0 = {  # Cd at alpha = 0
    1000: {"Di Ilio et al. 2020 (LBM)": 0.119, "Kurtulus 2015 (Fluent)": 0.12},
    2000: {"Di Ilio et al. 2020 (LBM)": 0.084},
    5000: {"Di Ilio et al. 2020 (LBM)": 0.052},
}

DEFAULT_FARFIELD = 40.0


# ----------------------------------------------------------------------
def laminar_h1(re):
    """First layer height for a LAMINAR boundary layer (delta ~ 5c/sqrt(Re)):
    ~ delta/80 at the trailing edge. (The y+ formula in geometry.py assumes a
    turbulent boundary layer and is far too coarse at Re ~ 1000.)"""
    return 2e-3 * math.sqrt(1000.0 / re)


def airfoil_case(code="0012", re=1000.0, alpha=0.0, refine=1.0, farfield=DEFAULT_FARFIELD,
                 verbose=False):
    r = refine
    # Uniform layer heights (no aspect-ratio cap): for a thick laminar boundary
    # layer the capped version produced a notch at the trailing-edge fan junction
    # and a BL depth that changed between grid levels (0.11 / 0.13 / 0.07 c).
    # Uniform layers keep the BL region identical on every level (~0.08 c).
    mesh = hm.build_mesh(naca=code, n_side=int(round(120 * r)), h1=laminar_h1(re) / r,
                         growth=1.1 ** (1 / r), n_layers=int(round(17 * r)),
                         bl_ar_cap=float("inf"), farfield_radius=farfield,
                         n_farfield=int(round(64 * r)), h_near=0.02 / r, grad=0.12 / r,
                         wake_length=5.0, wake_h=0.01 / r, wake_grad=0.03 / r,
                         alpha_deg=alpha, label=f"NACA {code} (refine x{r:g})",
                         verbose=verbose)
    U = (math.cos(math.radians(alpha)), math.sin(math.radians(alpha)))
    P = mesh["points"]
    ff = mesh["patches"]["farfield"]
    t = P[ff[:, 1]] - P[ff[:, 0]]
    inflow = t[:, 1] * U[0] - t[:, 0] * U[1] < 0
    patches = {"wall": mesh["patches"]["airfoil"], "inlet": ff[inflow], "outlet": ff[~inflow]}
    return FVMesh(P, mesh["cells"], patches), mesh, patches, U


def separation_x(w):
    """Upper-surface separation: first x (from the LE, x > 0.05) where Cf < 0.
    Faces at x > 0.998 are excluded: that is the blunt trailing-edge BASE,
    where the flow is always locally reversed (not boundary-layer separation)."""
    w = {k: np.asarray(v) for k, v in w.items()}
    up = (w["side"] == 1) & (w["x"] > 0.05) & (w["x"] < 0.998)
    o = np.argsort(w["x"][up])
    x, cf = w["x"][up][o], w["Cf"][up][o]
    neg = np.where(cf < 0)[0]
    if len(neg) == 0:
        return None
    i = neg[0]
    if i == 0:
        return float(x[0])
    return float(x[i - 1] - cf[i - 1] * (x[i] - x[i - 1]) / (cf[i] - cf[i - 1]))


def run(re=1000.0, alpha=0.0, refine=1.0, farfield=DEFAULT_FARFIELD, tol=1e-7,
        max_iter=20000, verbose=False, return_solver=False):
    t0 = time.time()
    fvm, mesh, patches, U = airfoil_case(re=re, alpha=alpha, refine=refine, farfield=farfield)
    bcs = {"wall": {"type": "wall"}, "inlet": {"type": "inlet", "U": U},
           "outlet": {"type": "outlet"}}
    s = SimpleSolver(fvm, bcs, 1.0 / re, U0=U)
    it = s.solve(tol=tol, max_iter=max_iter, verbose=verbose, every=500)
    w, c = wall_data(s, "wall", U)
    o = np.lexsort((w["x"], w["side"]))
    res = {"Re": re, "alpha": alpha, "refine": refine, "farfield": farfield,
           "cells": int(fvm.ncells), "iterations": int(it), "converged": bool(s.converged),
           "time_s": round(time.time() - t0, 1),
           "Cd": float(c["Cd"]), "Cl": float(c["Cl"]), "Cd_p": float(c["Cd_pressure"]),
           "Cd_v": float(c["Cd_viscous"]), "x_sep": separation_x(w),
           "yplus_max": float(w["yplus"].max()),
           "wall": {k: np.asarray(w[k])[o].round(6).tolist() for k in ("x", "side", "Cp", "Cf")}}
    if return_solver:
        return res, s, mesh
    return res


# ----------------------------------------------------------------------
def _load():
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def _save(db):
    VAL.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(db))


def _key(r):
    return f"Re{r['Re']:g}_a{r['alpha']:g}_r{r['refine']:g}_R{r['farfield']:g}"


def run_many(specs, db):
    for sp in specs:
        k = f"Re{sp['re']:g}_a{sp['alpha']:g}_r{sp.get('refine', 1.0):g}_R{sp.get('farfield', DEFAULT_FARFIELD):g}"
        if k in db:
            print(f"  (cached) {k}")
            continue
        r = run(re=sp["re"], alpha=sp["alpha"], refine=sp.get("refine", 1.0),
                farfield=sp.get("farfield", DEFAULT_FARFIELD))
        db[k] = r
        _save(db)
        print(f"  {k}: {r['cells']} cells, {r['iterations']} it{'' if r['converged'] else ' (NOT CONVERGED)'}, "
              f"{r['time_s']:.0f}s -> Cd {r['Cd']:.5f}  Cl {r['Cl']:.5f}  x_sep {r['x_sep']}",
              flush=True)
    return db


STUDIES = {
    "domain": [{"re": 1000, "alpha": 4, "farfield": R} for R in (20.0, 40.0, 80.0)],
    "grid": [{"re": 1000, "alpha": a, "refine": r} for a in (0, 4) for r in (0.7, 1.0, 1.4)],
    "sweep": [{"re": 1000, "alpha": a} for a in range(0, 8)],
    "extra": [{"re": re, "alpha": a} for re in (2000, 5000) for a in (0, 2)],
}


# ----------------------------------------------------------------------
def xfoil_polar(re):
    f = REF / f"xfoil_naca0012_re{int(re)}_polar.txt"
    return np.loadtxt(f) if f.exists() else None


def xfoil_cp(re, alpha):
    f = REF / f"xfoil_naca0012_re{int(re)}_cp_a{alpha:g}.txt"
    return np.loadtxt(f) if f.exists() else None


def _refresh_sep(db):
    for r in db.values():
        r["x_sep"] = separation_x(r["wall"])
    return db


def report(db):
    _refresh_sep(db)
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    get = lambda **kw: db.get(f"Re{kw['re']:g}_a{kw['alpha']:g}_r{kw.get('refine', 1.0):g}"
                              f"_R{kw.get('farfield', DEFAULT_FARFIELD):g}")
    P("DOMAIN SIZE (Re 1000, alpha 4):")
    for R in (20.0, 40.0, 80.0):
        r = get(re=1000, alpha=4, farfield=R)
        if r:
            P(f"  R = {R:4.0f} c   Cd {r['Cd']:.5f}   Cl {r['Cl']:.5f}")
    P("GRID CONVERGENCE (Re 1000):")
    for a in (0, 4):
        vals = [get(re=1000, alpha=a, refine=f) for f in (0.7, 1.0, 1.4)]
        for f, r in zip((0.7, 1.0, 1.4), vals):
            if r:
                P(f"  alpha {a}  refine x{f:3.1f} ({r['cells']:6d} cells)   Cd {r['Cd']:.5f}   "
                  f"Cl {r['Cl']:.5f}   x_sep {r['x_sep']}")
        if all(vals):
            # Roache's Grid Convergence Index on the two finest grids, using the
            # scheme's FORMAL order p = 2 (the observed order is only meaningful
            # when convergence is monotonic, which is checked and reported)
            f1, f2, f3 = (v["Cd"] for v in vals)          # coarse, medium, fine
            rr, p = 1.4, 2.0
            mono = (f2 - f1) * (f3 - f2) > 0
            gci = 1.25 * abs((f3 - f2) / f3) / (rr ** p - 1) * 100
            ext = f3 + (f3 - f2) / (rr ** p - 1)
            trend = "monotonic" if mono else "oscillatory (not monotonic)"
            P(f"    Cd: medium->fine change {abs(f3 - f2) / f3 * 100:.3f} %, "
              f"coarse->medium {abs(f2 - f1) / f2 * 100:.2f} %, {trend}; "
              f"GCI(fine, p=2) {gci:.3f} %, Richardson estimate {ext:.5f}")
    P("ALPHA SWEEP (Re 1000) vs XFOIL:")
    xp = xfoil_polar(1000)
    P("  alpha   Cl       Cl_xfoil   Cd       Cd_xfoil   x_sep")
    for a in range(0, 8):
        r = get(re=1000, alpha=a)
        if r:
            xr = xp[xp[:, 0] == a][0] if xp is not None and (xp[:, 0] == a).any() else [a, np.nan, np.nan]
            sep = f"{r['x_sep']:.3f}" if r["x_sep"] is not None else "attached"
            P(f"  {a:3d}   {r['Cl']:7.4f}   {xr[1]:7.4f}   {r['Cd']:7.5f}   {xr[2]:7.5f}   {sep}"
              f"{'' if r['converged'] else '  (not converged)'}")
    rs = [get(re=1000, alpha=a) for a in (0, 1, 2)]
    if all(rs):
        slope = np.polyfit([0, 1, 2], [r["Cl"] for r in rs], 1)[0] * 180 / math.pi
        P(f"  lift slope near alpha 0: {slope:.3f} per rad  (Kurtulus 2015: ~pi = 3.142; "
          f"XFOIL: {np.polyfit(xp[:3, 0], xp[:3, 1], 1)[0] * 180 / math.pi:.3f})")
    P("Cd AT ALPHA 0 vs LITERATURE:")
    for re in (1000, 2000, 5000):
        r = get(re=re, alpha=0)
        xp = xfoil_polar(re)
        if r:
            lit = ", ".join(f"{k} {v}" for k, v in LITERATURE_CD0[re].items())
            xf = xp[xp[:, 0] == 0][0][2] if xp is not None else float("nan")
            P(f"  Re {re:5d}: this solver {r['Cd']:.4f} | XFOIL {xf:.4f} | {lit}")
    (VAL / "airfoil_report.txt").write_text("\n".join(lines) + "\n")


def plots(db):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    get = lambda **kw: db.get(f"Re{kw['re']:g}_a{kw['alpha']:g}_r{kw.get('refine', 1.0):g}"
                              f"_R{kw.get('farfield', DEFAULT_FARFIELD):g}")

    # 1) polar
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    rs = [get(re=1000, alpha=a) for a in range(8)]
    rs = [r for r in rs if r]
    xp = xfoil_polar(1000)
    a = [r["alpha"] for r in rs]
    ax[0].plot(a, [r["Cl"] for r in rs], "o-", label="this solver")
    ax[1].plot(a, [r["Cd"] for r in rs], "o-", label="this solver")
    if xp is not None:
        ax[0].plot(xp[:, 0], xp[:, 1], "s--", mfc="none", label="XFOIL")
        ax[1].plot(xp[:, 0], xp[:, 2], "s--", mfc="none", label="XFOIL")
    aa = np.linspace(0, 7, 10)
    ax[0].plot(aa, np.radians(aa) * math.pi, "k:", label="slope pi/rad (Kurtulus 2015)")
    for k, v in LITERATURE_CD0[1000].items():
        ax[1].plot(0, v, "^", ms=9, label=k)
    ax[0].set_title("NACA 0012, Re = 1000: lift"); ax[0].set_xlabel("alpha (deg)"); ax[0].set_ylabel("Cl")
    ax[1].set_title("NACA 0012, Re = 1000: drag"); ax[1].set_xlabel("alpha (deg)"); ax[1].set_ylabel("Cd")
    for re, mk in ((1000, "o"), (2000, "s"), (5000, "D")):
        r = get(re=re, alpha=0)
        xq = xfoil_polar(re)
        if r:
            ax[2].plot(re, r["Cd"], mk, color="C0", ms=8, label="this solver" if re == 1000 else None)
        if xq is not None:
            ax[2].plot(re, xq[0, 2], mk, color="C1", mfc="none", ms=10, label="XFOIL" if re == 1000 else None)
        for k, v in LITERATURE_CD0[re].items():
            ax[2].plot(re, v, "^", color="C2", ms=9, label=k if re == 1000 and "Di" in k else None)
    ax[2].set_xscale("log"); ax[2].set_title("Cd at alpha = 0 vs Reynolds number")
    ax[2].set_xlabel("Re"); ax[2].set_ylabel("Cd")
    for x in ax:
        x.grid(alpha=.3); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(VAL / "airfoil_polar.png", dpi=140)

    # 2) Cp and Cf distributions vs XFOIL
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    for j, alpha in enumerate((0, 4, 7)):
        r = get(re=1000, alpha=alpha)
        if not r:
            continue
        w = {k: np.array(v) for k, v in r["wall"].items()}
        for side, ls, lab in ((1, "-", "upper"), (-1, "--", "lower")):
            s = w["side"] == side
            ax[0, j].plot(w["x"][s], w["Cp"][s], ls, color="C0", label=f"this solver, {lab}")
            ax[1, j].plot(w["x"][s], w["Cf"][s], ls, color="C0", label=f"this solver, {lab}")
        xc = xfoil_cp(1000, alpha)
        if xc is not None:
            ax[0, j].plot(xc[:, 0], xc[:, 1], ".", color="C1", ms=3, label="XFOIL")
        ax[0, j].invert_yaxis()
        ax[0, j].set_title(f"Cp, Re 1000, alpha {alpha} deg")
        ax[1, j].axhline(0, color="k", lw=0.7)
        sep = f"separation x = {r['x_sep']:.3f}" if r["x_sep"] else "attached"
        ax[1, j].set_title(f"Cf, alpha {alpha} deg ({sep})")
        for x in ax[:, j]:
            x.set_xlabel("x/c"); x.grid(alpha=.3); x.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(VAL / "airfoil_distributions.png", dpi=140)

    # 3) grid + domain convergence
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a_, mk in ((0, "o-"), (4, "s-")):
        vals = [get(re=1000, alpha=a_, refine=f) for f in (0.7, 1.0, 1.4)]
        vals = [v for v in vals if v]
        if vals:
            ax[0].plot([v["cells"] for v in vals], [v["Cd"] for v in vals], mk, label=f"Cd, alpha {a_}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("cells"); ax[0].set_ylabel("Cd")
    ax[0].set_title("Grid convergence, Re 1000"); ax[0].grid(alpha=.3); ax[0].legend()
    vals = [get(re=1000, alpha=4, farfield=R) for R in (20.0, 40.0, 80.0)]
    vals = [v for v in vals if v]
    if vals:
        ax[1].plot([v["farfield"] for v in vals], [v["Cd"] for v in vals], "o-", label="Cd")
        ax2 = ax[1].twinx()
        ax2.plot([v["farfield"] for v in vals], [v["Cl"] for v in vals], "s-", color="C1", label="Cl")
        ax2.set_ylabel("Cl")
    ax[1].set_xscale("log"); ax[1].set_xlabel("farfield radius (chords)"); ax[1].set_ylabel("Cd")
    ax[1].set_title("Domain-size effect, Re 1000, alpha 4"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(VAL / "airfoil_convergence.png", dpi=140)
    print("plots saved in", VAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=list(STUDIES) + ["all"], default="all")
    ap.add_argument("--plot", action="store_true", help="only report + plots from saved results")
    a = ap.parse_args()
    db = _load()
    if not a.plot:
        for name, specs in STUDIES.items():
            if a.study in ("all", name):
                print(f"== study: {name}", flush=True)
                db = run_many(specs, db)
    report(db)
    plots(db)


if __name__ == "__main__":
    main()
