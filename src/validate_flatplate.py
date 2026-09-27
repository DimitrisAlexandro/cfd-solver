"""
Verification (solver step C): Spalart-Allmaras on NASA's 2D zero-pressure-
gradient flat plate (tmbwg.github.io/turbmodels/flatplate.html).

  Re = 5e6 per unit length, plate from x = 0 to x = 2 (Re_L = 1e7),
  symmetry plane -1/3 < x < 0, domain height 1, SA-noft2, nuT_inf = 3 nu.

Reference (NASA TMR data files, validation/reference/tmr/fp_*):
  incompressible SA (FUN3D-incompressible, SC/Tetra), finest grid 545x385:
      Cf(x = 0.97) = 0.002729
  compressible SA at M = 0.2 (CFL3D, FUN3D), finest grid:
      Cf(x = 0.97) = 0.002706, CD = 0.002860 / 0.002852
  (Our solver is incompressible, so the incompressible codes are the direct
   comparison; the M = 0.2 values are ~1 % lower.)

    python validate_flatplate.py          -> ../validation/flatplate_*.png
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from fvmesh import FVMesh
from simple_solver import SimpleSolver

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "validation"
TMR = VAL / "reference" / "tmr"
RESULTS = VAL / "flatplate_results.json"
RE = 5e6
X_CF = 0.970084071
REF_CF_INCOMP = 0.002729           # FUN3D-incompressible / SC/Tetra, 545x385
REF_CF_M02 = 0.0027056             # CFL3D, M = 0.2, 545x385
REF_CD_M02 = 0.0028599             # CFL3D, M = 0.2, 545x385


def geometric(n, length, first):
    """n cells over `length`, first cell size `first`, constant growth ratio."""
    if abs(first * n - length) < 1e-12 * length:
        return np.linspace(0, length, n + 1)
    from scipy.optimize import brentq
    # sum_{k<n} r^k = length/first, solved in log form (robust for large r)
    target = math.log(length / first)
    f = lambda lr: (np.logaddexp.reduce(lr * np.arange(n)) - target)
    lr = brentq(f, -5.0, 5.0, xtol=1e-15) if f(0.0) != 0 else 0.0
    h = first * np.exp(lr * np.arange(n))
    return np.r_[0.0, np.cumsum(h)] * (length / h.sum())


def plate_mesh(level=1.0):
    """Structured quads, 136f x 96f cells (level 0.5 / 1 / 2 ~ TMR 69x49 / 137x97 / 273x193);
    geometric clustering at the leading edge (x = 0) and the wall."""
    f = level
    nx_up, nx_pl, ny = int(24 * f), int(112 * f), int(96 * f)
    dx0, dy0 = 4e-4 / f, 5e-6 / f
    xu = -geometric(nx_up, 1.0 / 3.0, dx0)[::-1]
    xp = geometric(nx_pl, 2.0, dx0)
    xs = np.r_[xu[:-1], xp]
    ys = geometric(ny, 1.0, dy0)
    nx = len(xs) - 1
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = np.c_[X.ravel(), Y.ravel()]
    idx = lambda i, j: i * (ny + 1) + j
    cells = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
             for i in range(nx) for j in range(ny)]
    patches = {"symmetry": [], "plate": [], "top": [], "outlet": [], "inlet": []}
    for i in range(nx):
        xm = 0.5 * (xs[i] + xs[i + 1])
        patches["plate" if xm > 0 else "symmetry"].append((idx(i, 0), idx(i + 1, 0)))
        patches["top"].append((idx(i + 1, ny), idx(i, ny)))
    for j in range(ny):
        patches["outlet"].append((idx(nx, j), idx(nx, j + 1)))
        patches["inlet"].append((idx(0, j + 1), idx(0, j)))
    return FVMesh(pts, cells, {k: np.array(v) for k, v in patches.items()})


BCS = {"inlet": {"type": "inlet", "U": (1.0, 0.0)}, "symmetry": {"type": "symmetry"},
       "plate": {"type": "wall"}, "top": {"type": "outlet"}, "outlet": {"type": "outlet"}}


def tmr_zones(path):
    """Read a Tecplot-style TMR .dat file -> {zone title: (n, 2) array}."""
    zones, cur, name = {}, None, None
    for line in Path(path).read_text().splitlines():
        t = line.strip()
        if not t or t.startswith("#") or t.lower().startswith("variables"):
            continue
        if t.lower().startswith("zone"):
            name = t.split("=", 1)[1].strip().strip('"') if "=" in t else f"zone{len(zones)}"
            cur = zones.setdefault(name, [])
            continue
        if cur is None:                              # data before any zone header
            cur = zones.setdefault("data", [])
        cur.append([float(v) for v in t.split()])
    return {k: np.array(v) for k, v in zones.items()}


def plate_cf(s):
    idx, pb, tau, S, Sm = s.wall_values("plate")
    x = s.m.bxf[idx, 0]
    o = np.argsort(x)
    return x[o], (2.0 * tau[:, 0])[o], Sm[o]           # Cf = tau / (0.5 rho U^2)


def run(level=1.0, tol=1e-8, max_iter=20000, verbose=True):
    m = plate_mesh(level)
    s = SimpleSolver(m, BCS, 1.0 / RE, U0=(1.0, 0.0), turbulence="sa", alpha_u=0.8,
                     alpha_turb=0.8)
    t0 = time.time()
    it = s.solve(tol=tol, max_iter=max_iter, verbose=verbose, every=500)
    x, cf, dx = plate_cf(s)
    cf97 = float(np.interp(X_CF, x, cf))
    cd = float((cf * dx).sum() / 2.0)                 # mean Cf over the plate (TMR CD)
    # velocity profile at x = 0.97 (column of cells)
    xc = m.xc
    col = np.abs(xc[:, 0] - xc[np.argmin(np.abs(xc[:, 0] - X_CF)), 0]) < 1e-12
    y = xc[col, 1]; o = np.argsort(y)
    u = s.U[col, 0][o]; y = y[o]
    utau = math.sqrt(cf97 / 2.0)
    return {"level": level, "cells": int(m.ncells), "iterations": int(it),
            "converged": bool(s.converged), "time_s": round(time.time() - t0, 1),
            "cf97": cf97, "cd": cd, "x": x.tolist(), "cf": cf.tolist(),
            "yplus": (y * utau * RE).tolist(), "uplus": (u / utau).tolist(),
            "nut_ratio_max": float((s.turb.nu_t() * RE).max()),
            "yplus_first": float(xc[col, 1].min() * utau * RE)}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    db = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    levels = [0.5, 1.0, 2.0] if "--fine" in sys.argv else [0.5, 1.0]
    if "--plot" not in sys.argv:
        for lv in levels:
            k = f"L{lv:g}"
            if k in db:
                continue
            print(f"== level {lv}", flush=True)
            db[k] = run(lv)
            RESULTS.write_text(json.dumps(db))
    print(f"{'grid':>18s} {'Cf(0.97)':>10s} {'vs incomp. ref':>15s} {'CD':>10s} {'iters':>6s} {'y+1':>6s}")
    for k in sorted(db, key=lambda k: db[k]["cells"]):
        r = db[k]
        print(f"{r['cells']:10d} cells {r['cf97']:10.6f} {100 * (r['cf97'] / REF_CF_INCOMP - 1):+14.2f}% "
              f"{r['cd']:10.6f} {r['iterations']:6d} {r['yplus_first']:6.2f}")
    print(f"reference: Cf(0.97) incompressible SA {REF_CF_INCOMP} ; M=0.2 SA {REF_CF_M02}, CD {REF_CD_M02}")

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    for (name, ref), st in zip(tmr_zones(TMR / "fp_cf_plate.dat").items(), ("k-", "k--")):
        ax[0].plot(ref[:, 0], ref[:, 1], st, lw=2, alpha=.5, label=f"NASA {name} (M 0.2, 545x385)")
    for k in sorted(db, key=lambda k: db[k]["cells"]):
        r = db[k]
        ax[0].plot(r["x"], r["cf"], label=f"this solver ({r['cells']} cells)")
    ax[0].plot(X_CF, REF_CF_INCOMP, "r*", ms=12, label="incompressible SA reference, x = 0.97")
    ax[0].set_ylim(0.0015, 0.006); ax[0].set_xlim(0, 2)
    ax[0].set_xlabel("x"); ax[0].set_ylabel("Cf"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[0].set_title("Flat plate skin friction (SA)")
    yp = np.logspace(-1, 4, 200)
    ax[1].semilogx(yp, np.where(yp < 11, yp, np.log(yp) / 0.41 + 5.0), "k:", label="u+ = y+ ; log law")
    uy = next(iter(tmr_zones(TMR / "fp_flatplate_u+y+.dat").values()))      # x = 0.97008
    ax[1].semilogx(10.0 ** uy[:, 0], uy[:, 1], "k-", lw=2, alpha=.4, label="NASA SA (CFL3D), x = 0.97")
    for k in sorted(db, key=lambda k: db[k]["cells"]):
        r = db[k]
        ax[1].semilogx(r["yplus"], r["uplus"], ".-", ms=3, label=f"this solver ({r['cells']} cells)")
    ax[1].set_xlim(0.1, 1e4); ax[1].set_ylim(0, 30); ax[1].set_xlabel("y+"); ax[1].set_ylabel("u+")
    ax[1].set_title("Velocity profile at x = 0.97"); ax[1].grid(alpha=.3, which="both")
    ax[1].legend(fontsize=8)
    # grid convergence vs the incompressible codes on the same grid family
    for (name, z), mk in zip(tmr_zones(TMR / "fp_cf_incomp_results_sa.dat").items(), ("ks--", "k^:")):
        ax[2].plot(z[:, 2], z[:, 3], mk, mfc="none", label=f"NASA {name}")
    rs = sorted(db.values(), key=lambda r: r["cells"])
    ax[2].plot([1 / math.sqrt(r["cells"]) for r in rs], [r["cf97"] for r in rs], "rD-",
               label="this solver")
    ax[2].axhline(REF_CF_INCOMP, color="gray", lw=0.8)
    ax[2].set_xlim(0, 0.02); ax[2].set_ylim(0.00250, 0.00280)
    ax[2].set_xlabel("h = 1/sqrt(cells)"); ax[2].set_ylabel("Cf(x = 0.97)")
    ax[2].set_title("Grid convergence (TMR grids 69x49 ... 273x193)")
    ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(VAL / "flatplate_sa.png", dpi=140)
    print("saved", VAL / "flatplate_sa.png")


if __name__ == "__main__":
    main()
