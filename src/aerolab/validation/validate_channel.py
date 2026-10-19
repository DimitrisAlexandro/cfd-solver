"""
Validation 1: plane Poiseuille flow (exact solution).

Channel y in [-h, h], length 3, parabolic inflow u = Umax (1 - (y/h)^2),
outlet p = 0, no-slip walls. The exact solution is the same parabola
everywhere with dp/dx = -2 nu Umax / h^2.

What it proves: the discretisation is second-order accurate on BOTH quads
and unstructured triangles (the observed order of the error must be ~2).

    python src/run.py validate_channel            -> ../validation/channel.png
"""

from pathlib import Path

import numpy as np

from aerolab.core.fvmesh import FVMesh, rect_quad, rect_tri
from aerolab.core.simple_solver import SimpleSolver

NU, H = 0.01, 0.5
NAMES = ("wallB", "outlet", "wallT", "inlet")


def profile(x, y):
    return np.c_[1 - (y / H) ** 2, 0 * y]


def run(kind, n_across, verbose=False):
    if kind == "quad":
        m = FVMesh(*rect_quad(3 * n_across, n_across, 0, 3, -H, H, names=NAMES))
    else:
        m = FVMesh(*rect_tri(1.0 / n_across, 0, 3, -H, H, names=NAMES))
    bcs = {"wallB": {"type": "wall"}, "wallT": {"type": "wall"},
           "inlet": {"type": "inlet", "U": profile}, "outlet": {"type": "outlet"}}
    s = SimpleSolver(m, bcs, NU, U0=(0.5, 0))
    it = s.solve(tol=1e-9, max_iter=5000, verbose=verbose)
    ex = profile(m.xc[:, 0], m.xc[:, 1])
    err = np.sqrt((((s.U - ex) ** 2).sum(1) * m.vol).sum() / m.vol.sum())
    mid = np.abs(m.xc[:, 0] - 1.5) < 1.0
    dpdx = np.polyfit(m.xc[mid, 0], s.p[mid], 1)[0]
    return {"kind": kind, "n": n_across, "cells": m.ncells, "iters": it, "err": err,
            "dpdx": dpdx, "dpdx_exact": -2 * NU / H**2, "mesh": m, "solver": s}


def main(out="../validation"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out).mkdir(parents=True, exist_ok=True)
    results = {}
    for kind in ("quad", "tri"):
        results[kind] = [run(kind, n) for n in (10, 20, 40)]

    print(f"{'mesh':5s} {'n':>4s} {'cells':>7s} {'iters':>6s} {'L2 error':>10s} "
          f"{'order':>6s} {'dp/dx error':>12s}")
    for kind, rs in results.items():
        for i, r in enumerate(rs):
            order = "" if i == 0 else f"{np.log(rs[i-1]['err'] / r['err']) / np.log(2):6.2f}"
            e = abs(r["dpdx"] / r["dpdx_exact"] - 1) * 100
            print(f"{kind:5s} {r['n']:4d} {r['cells']:7d} {r['iters']:6d} {r['err']:10.3e} "
                  f"{order:>6s} {e:11.3f}%")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for kind, mk in (("quad", "o-"), ("tri", "s-")):
        rs = results[kind]
        hs = [1 / r["n"] for r in rs]
        ax[0].loglog(hs, [r["err"] for r in rs], mk, label=f"{kind} meshes")
    h = np.array([1 / 10, 1 / 40])
    ax[0].loglog(h, 3e-3 * (h / h[0]) ** 2, "k--", label="slope 2 (2nd order)")
    ax[0].set_xlabel("cell size h / (channel height)"); ax[0].set_ylabel("L2 velocity error")
    ax[0].set_title("Grid convergence (exact solution)"); ax[0].legend(); ax[0].grid(alpha=.3, which="both")

    r = results["tri"][1]
    m, s = r["mesh"], r["solver"]
    sel = np.abs(m.xc[:, 0] - 2.0) < 0.05
    yy = np.linspace(-H, H, 100)
    ax[1].plot(1 - (yy / H) ** 2, yy, "k-", label="exact")
    ax[1].plot(s.U[sel, 0], m.xc[sel, 1], "s", ms=4, label=f"solver, triangles ({m.ncells} cells)")
    ax[1].set_xlabel("u"); ax[1].set_ylabel("y"); ax[1].set_title("Velocity profile at x = 2")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/channel.png", dpi=140)
    return results


if __name__ == "__main__":
    main()
