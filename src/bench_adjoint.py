"""Adjoint cost vs mesh level, SuperLU vs MUMPS (NACA 0012, Re 3e6, alpha 4):
flow, Jacobian, factorisation and solve times, accuracy of the adjoint.
    python src/bench_adjoint.py --levels 0.5 0.7 1.0 --out validation/adjoint_speed.json"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adjoint as A  # noqa: E402


def run(level, backends=("superlu", "mumps"), log=print):
    t = time.time()
    pr = A.solve_primal("naca0012", 3e6, 4.0, level)
    t_flow = time.time() - t
    adj = A.Adjoint(pr["solver"], 4.0, pr["vortex"], pr["walls"]).build(factorise=False)
    g = adj.res.objective_gradient(adj.X, "Cl")
    AT = adj.J.T.tocsr()
    rec = {"level": level, "cells": int(pr["solver"].m.ncells), "unknowns": int(adj.res.N),
           "nnz": int(adj.J.nnz), "flow_s": round(t_flow, 1),
           "jacobian_s": round(adj.timing["jacobian"] + adj.timing["colouring"], 1)}
    psi = {}
    for b in backends:
        if b == "mumps" and not A.mumps_available():
            continue
        t = time.time()
        lu = A.factorize(AT, b)
        tf = time.time() - t
        t = time.time()
        x = lu.solve(-g)
        ts = time.time() - t
        psi[b] = x
        rec[b] = {"factorise_s": round(tf, 1), "solve_s": round(ts, 2),
                  "rel_residual": float(np.abs(AT @ x + g).max() / np.abs(g).max())}
        log(f"level {level}: {b} factorise {tf:.1f} s, solve {ts:.2f} s")
        del lu
    if len(psi) == 2:
        a, b = psi.values()
        rec["psi_rel_difference"] = float(np.abs(a - b).max() / np.abs(a).max())
    rec["peak_rss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=float, nargs="+", default=[0.5, 0.7, 1.0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "validation" / "adjoint_speed.json"))
    a = ap.parse_args()
    out = [run(lv) for lv in a.levels]
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
