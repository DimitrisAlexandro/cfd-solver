"""
Speed benchmark (solver step D). Fixed cases, run with a chosen linear solver:

  cavity    lid-driven cavity Re 1000, 129 x 129 quads (vs Botella & Peyret)
  cylinder  steady cylinder Re 40, hybrid mesh (Cd, recirculation length)
  naca_sa   NACA 0012 Re 6e6 alpha 10, Spalart-Allmaras, 29k cells (Cl, Cd)
  multi     30P30N (JAXA) Re 1.71e6 alpha 5.5, SA, 75k cells, as the polar tool
            runs it (force-based convergence)

Phase 4: --mode numba (compiled kernels, default when numba is installed) or
--mode baseline (the step-D numpy path with its pressure tolerance 0.01).

    python src/run.py benchmark --linear direct             reference (exact sparse LU)
    python src/run.py benchmark --linear iterative          Krylov + multigrid
    python src/run.py benchmark --linear iterative --cases cavity cylinder
    python src/run.py benchmark --compare                   table: direct vs iterative

Results -> ../validation/benchmark_<linear>.json
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from aerolab.core import kernels
from aerolab.core import linsolve
from aerolab.core import simple_solver
from aerolab.core.simple_solver import SimpleSolver

VAL = Path(__file__).resolve().parents[3] / "validation"


def case_cavity():
    from aerolab.validation import validate_cavity as vc
    from aerolab.core.fvmesh import FVMesh, rect_quad
    m = FVMesh(*rect_quad(128, 128))
    s = SimpleSolver(m, vc.BCS, 1.0 / 1000.0)
    return s, dict(tol=1e-7), lambda s: {k: float(v) for k, v in vc.botella_errors(s).items()
                                          if np.isscalar(v)}


def case_cylinder():
    from aerolab.validation import validate_cylinder as vcy
    fvm, mesh = vcy.cylinder_mesh()
    bcs = {"cylinder": {"type": "wall"}, "inlet": {"type": "inlet", "U": (1.0, 0.0)},
           "outlet": {"type": "outlet"}}
    s = SimpleSolver(fvm, bcs, 1.0 / 40.0, U0=(1.0, 0.0))
    return s, dict(tol=1e-7), lambda s: {k: float(v) for k, v in vcy.analyse(s).items()
                                          if np.isscalar(v) and v is not None}


def case_naca_sa():
    from aerolab.validation import validate_naca0012_sa as v
    from aerolab.core.fvmesh import FVMesh
    from aerolab.core.postproc import wall_data
    a = 10.0
    mesh = v.airfoil_mesh(1.0, a)
    U = (math.cos(math.radians(a)), math.sin(math.radians(a)))
    P = mesh["points"]
    ff = mesh["patches"]["farfield"]
    t = P[ff[:, 1]] - P[ff[:, 0]]
    inflow = t[:, 1] * U[0] - t[:, 0] * U[1] < 0
    fvm = FVMesh(P, mesh["cells"], {"wall": mesh["patches"]["airfoil"], "inlet": ff[inflow],
                                    "outlet": ff[~inflow]})
    bcs = {"wall": {"type": "wall"}, "inlet": {"type": "inlet", "U": U},
           "outlet": {"type": "outlet"}}
    s = SimpleSolver(fvm, bcs, 1.0 / v.RE, U0=U, turbulence="sa")
    vf = v.VortexFarfield(s, a)

    def res(s):
        c = wall_data(s, "wall", U)[1]
        return {"Cl": float(c["Cl"]), "Cd": float(c["Cd"])}
    return s, dict(tol=1e-7, callback=vf.update), res


CASES = {"cavity": case_cavity, "cylinder": case_cylinder, "naca_sa": case_naca_sa}


def set_mode(mode):
    """numba: compiled kernels + pressure tolerance 0.1 (Phase 4);
    baseline: the step-D numpy path, pressure tolerance 0.01."""
    kernels.USE_NUMBA = (mode == "numba") and kernels.HAVE_NUMBA
    init = linsolve.LinearSolver.__init__
    if not hasattr(linsolve.LinearSolver, "_orig_init"):
        linsolve.LinearSolver._orig_init = init
    orig = linsolve.LinearSolver._orig_init
    rtol_p = 0.1 if mode == "numba" else 0.01

    def patched(self, kind="iterative", **k):
        k.setdefault("rtol_p", rtol_p)
        orig(self, kind, **k)
    linsolve.LinearSolver.__init__ = patched


def run_multi(mode):
    from aerolab.airfoil import polar as pl
    set_mode(mode)
    t0 = time.time()
    r, _ = pl.run_point("30p30n_jaxa", 1.71e6, 5.5, level=1.0, transition="off")
    out = {"cells": r["cells"], "iterations": r["iterations"], "wall_s": round(time.time() - t0, 1),
           "s_per_iter": (time.time() - t0) / max(r["iterations"], 1), "converged": True,
           "results": {"Cl": r["Cl"], "Cd": r["Cd"], "Cm": r["Cm"]}}
    print(f"[{mode:8s}] multi     {out['cells']:6d} cells  {out['s_per_iter']:7.3f} s/iter  "
          f"{out['iterations']:5d} it  {out['wall_s']:7.1f} s  {out['results']}", flush=True)
    return out


def run(name, linear, timing_iters=30, full=True, mode=None):
    simple_solver.DEFAULT_LINEAR_SOLVER = linear
    if mode is not None:
        if name == "multi":
            return run_multi(mode)
        set_mode(mode)
    s, kw, results = CASES[name]()
    s.iterate()                                       # warm-up (first iteration differs)
    t0 = time.time()
    for _ in range(timing_iters):
        s.iterate()
    per_it = (time.time() - t0) / timing_iters
    out = {"cells": int(s.m.ncells), "s_per_iter": per_it}
    if full:
        s, kw, results = CASES[name]()
        t0 = time.time()
        it = s.solve(max_iter=20000, verbose=False, **kw)
        out.update(iterations=int(it), converged=bool(s.converged),
                   wall_s=round(time.time() - t0, 1), results=results(s))
    print(f"[{mode or linear:9s}] {name:9s} {out['cells']:6d} cells  {per_it:7.3f} s/iter"
          + (f"  {out['iterations']:5d} it  {out['wall_s']:7.1f} s  {out['results']}" if full else ""),
          flush=True)
    return out


def compare():
    d = json.loads((VAL / "benchmark_direct.json").read_text())
    i = json.loads((VAL / "benchmark_iterative.json").read_text())
    print(f"{'case':9s} {'cells':>6s} | {'s/iter direct':>13s} {'iterative':>9s} {'x':>5s} | "
          f"{'wall direct':>11s} {'iterative':>9s} {'x':>5s} | result differences")
    for k in d:
        if k not in i:
            continue
        a, b = d[k], i[k]
        diffs = ", ".join(f"{q} {abs(b['results'][q] - a['results'][q]) / max(abs(a['results'][q]), 1e-12):.1e}"
                          for q in a.get("results", {}) if q in b.get("results", {}))
        print(f"{k:9s} {a['cells']:6d} | {a['s_per_iter']:13.3f} {b['s_per_iter']:9.3f} "
              f"{a['s_per_iter'] / b['s_per_iter']:5.1f} | {a.get('wall_s', float('nan')):11.1f} "
              f"{b.get('wall_s', float('nan')):9.1f} {a.get('wall_s', 1) / b.get('wall_s', 1):5.1f} | {diffs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", choices=["direct", "iterative"], default="iterative")
    ap.add_argument("--mode", choices=["numba", "baseline"], default=None,
                    help="Phase 4: compiled kernels vs the step-D numpy path")
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        return compare()
    path = VAL / (f"benchmark_{a.linear}.json" if a.mode is None else f"benchmark_{a.mode}.json")
    db = json.loads(path.read_text()) if path.exists() else {}
    for c in a.cases:
        db[c] = run(c, a.linear, full=not a.timing_only, mode=a.mode)
        path.write_text(json.dumps(db, indent=1))


if __name__ == "__main__":
    main()
