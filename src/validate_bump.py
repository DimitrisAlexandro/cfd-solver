"""
Phase 5 validation 1: Ni's circular-arc bump in a channel (inviscid).

Channel 3 x 1, 10 % thick circular arc on the lower wall (1 <= x <= 2),
slip walls, inlet velocity (M_in), outlet static pressure. Classical test
of compressible solvers (Ni 1982; Eidelman et al. 1984):

  M_in = 0.5   subsonic: smooth, fore-aft symmetric, isentropic flow; no
               drag; entropy only from discretisation error
  M_in = 0.675 transonic: supersonic pocket over the bump terminated by a
               shock at ~72 % of the bump chord, M_max ~ 1.3-1.4

    python validate_bump.py            -> ../validation/bump_*.png, bump_report.txt
"""

import json
import math
import time
from pathlib import Path

import numpy as np

from compressible import CompressibleSolver
from fvmesh import FVMesh, rect_quad

OUT = Path(__file__).resolve().parents[1] / "validation"
NAMES = ("lower", "outlet", "upper", "inlet")


def bump_y(x, t=0.1):
    R = (0.25 + t * t) / (2 * t)
    yb = np.sqrt(np.maximum(R * R - (x - 1.5) ** 2, 0.0)) - (R - t)
    return np.where((x > 1.0) & (x < 2.0), np.maximum(yb, 0.0), 0.0)


def bump_mesh(nx=90, ny=30):
    """structured quads: x clustered on the bump, uniform in y"""
    P, cells, patches = rect_quad(nx, ny, 0, 1, 0, 1, names=NAMES)
    s = P[:, 0]
    # piecewise: 1/4 of the cells before, 1/2 on, 1/4 after the bump
    x = np.where(s < 0.25, 4 * s, np.where(s < 0.75, 1 + 2 * (s - 0.25), 2 + 4 * (s - 0.75)))
    yb = bump_y(x)
    y = yb + P[:, 1] * (1 - yb)
    return FVMesh(np.c_[x, y], cells, patches)


def bump_bcs(mach, g=1.4):
    """total pressure / temperature at the inlet, static pressure at the
    exit, chosen so that isentropic flow without the bump has Mach `mach`
    (non-dimensional: static rho = T = 1 there, U = 1)"""
    P0 = 1.0 / (g * mach**2)
    f = 1 + 0.5 * (g - 1) * mach**2
    return {"lower": {"type": "wall"}, "upper": {"type": "wall"},
            "inlet": {"type": "inlet", "p0": P0 * (f ** (g / (g - 1)) - 1.0), "T0": f},
            "outlet": {"type": "outlet"}}


def run(mach, nx=90, ny=30, max_iter=4000, tol=1e-7, verbose=False, **kw):
    t0 = time.time()
    m = bump_mesh(nx, ny)
    s = CompressibleSolver(m, bump_bcs(mach), 0.0, mach, inviscid=True, **kw)
    it = s.solve(tol=tol, max_iter=max_iter, verbose=verbose, every=200)
    Mf = s.mach_field()
    pa = s.p_abs()
    # entropy (p / rho^gamma relative to the inlet)
    ent = pa / s.rho ** s.gamma / (s.P0 / 1.0) - 1.0
    # lower-wall Mach number
    idx = m.patch_faces["lower"]
    c = m.bcell[idx]
    xw = m.bxf[idx, 0]
    o = np.argsort(xw)
    on = (xw[o] > 1) & (xw[o] < 2)
    Mw = Mf[c][o]
    res = {"mach_in": mach, "cells": m.ncells, "iterations": it, "converged": bool(s.converged),
           "time_s": round(time.time() - t0, 1), "M_max": float(Mf.max()),
           "M_wall_max": float(Mw.max()), "x_M_max": float(xw[o][Mw.argmax()]),
           "entropy_max": float(np.abs(ent).max()),
           "supersonic_cells": int((Mf > 1).sum())}
    # fore-aft symmetry of the wall Mach number on the bump (subsonic)
    xs, Ms = xw[o][on], Mw[on]
    res["asym"] = float(np.abs(Ms - np.interp(3.0 - xs, xs, Ms)).max())
    # shock position: steepest Mach drop along the lower wall
    dM = np.diff(Mw)
    k = int(np.argmin(dM))
    res["x_shock"] = float(0.5 * (xw[o][k] + xw[o][k + 1])) if dM[k] < -0.05 else None
    return res, s, (xw[o], Mw)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    OUT.mkdir(exist_ok=True)
    out, lines = {}, ["Ni bump (10 % circular arc), inviscid, compressible solver", ""]
    fig, axs = plt.subplots(2, 2, figsize=(13, 7))
    for j, M in enumerate((0.5, 0.675)):
        r, s, (xw, Mw) = run(M, verbose=True)
        out[str(M)] = r
        lines.append(json.dumps(r))
        m = s.m
        tri = mtri.Triangulation(m.xc[:, 0], m.xc[:, 1])
        ax = axs[0, j]
        cs = ax.tricontourf(tri, s.mach_field(), 40, cmap="jet")
        fig.colorbar(cs, ax=ax)
        ax.set_aspect("equal"); ax.set_title(f"Mach, M_in = {M}")
        axs[1, j].plot(xw, Mw, "k.-", ms=3)
        axs[1, j].axhline(1.0, color="r", lw=0.6)
        axs[1, j].set_xlabel("x"); axs[1, j].set_ylabel("M (lower wall)")
        axs[1, j].set_title(f"M_max {r['M_wall_max']:.3f}" +
                            (f", shock x = {r['x_shock']:.3f}" if r["x_shock"] else ""))
    fig.tight_layout()
    fig.savefig(OUT / "bump_mach.png", dpi=130)
    (OUT / "bump_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
