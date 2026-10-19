"""
Phase 4b benchmark: coupled (u, v, p) solver vs SIMPLEC on the polar tool's
own cases, timed on THIS machine (the coupled solver's cost is dominated by
the MKL PARDISO factorization, which scales with cores; SIMPLEC does not).

    python src/run.py benchmark_coupled                  NACA 0012 Re 6e6, 4 angles, cold starts
    python src/run.py benchmark_coupled --polar          + full warm-started polar -4..16 (both solvers)
    python src/run.py benchmark_coupled --multi          + 30P30N alpha 5.5

If the coupled column is clearly faster here, make it the default:
    export CFD_SOLVER=coupled                    (polar tool / airfoil.py --solver auto)

Results -> ../validation/benchmark_coupled.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from aerolab.airfoil import polar as pl

VAL = Path(__file__).resolve().parents[3] / "validation"


def point(spec, re, a, solver, **kw):
    t = time.time()
    r, _ = pl.run_point(spec, re, a, transition="off", solver=solver, **kw)
    return {"alpha": a, "Cl": r["Cl"], "Cd": r["Cd"], "iterations": r["iterations"],
            "status": r["status"], "time_s": round(time.time() - t, 1)}


def show(rows, label):
    print(f"\n{label}")
    print(f"{'alpha':>6s} | {'simple: Cl':>10s} {'Cd':>9s} {'it':>5s} {'s':>6s} | "
          f"{'coupled: Cl':>11s} {'Cd':>9s} {'it':>5s} {'s':>6s} | speed-up")
    for s, c in rows:
        print(f"{s['alpha']:6.1f} | {s['Cl']:10.5f} {s['Cd']:9.6f} {s['iterations']:5d} "
              f"{s['time_s']:6.1f} | {c['Cl']:11.5f} {c['Cd']:9.6f} {c['iterations']:5d} "
              f"{c['time_s']:6.1f} | {s['time_s'] / c['time_s']:5.2f}x")
    ts, tc = sum(s["time_s"] for s, _ in rows), sum(c["time_s"] for _, c in rows)
    print(f"total  simple {ts:.0f} s, coupled {tc:.0f} s -> {ts / tc:.2f}x")
    return ts, tc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--polar", action="store_true")
    ap.add_argument("--multi", action="store_true")
    a = ap.parse_args()
    out = {"cpu_count": os.cpu_count()}
    rows = [(point("tmr0012", 6e6, al, "simple"), point("tmr0012", 6e6, al, "coupled"))
            for al in (0.0, 6.0, 12.0, 16.0)]
    out["naca_cold"] = {"rows": rows, "totals": show(rows, "NACA 0012 Re 6e6, cold starts")}
    if a.polar:
        res = {}
        for sv in ("simple", "coupled"):
            t = time.time()
            res[sv] = (pl.polar("tmr0012", 6e6, np.arange(-4, 17, 2.0), jobs=1,
                                transition="off", solver=sv, log=False), time.time() - t)
        keep = ("alpha", "Cl", "Cd", "iterations", "status", "time_s")
        rows = [({k: s[k] for k in keep}, {k: c[k] for k in keep})
                for s, c in zip(res["simple"][0], res["coupled"][0])]
        out["naca_polar"] = {"rows": rows, "totals": show(rows, "NACA 0012 polar (warm starts)")}
    if a.multi:
        rows = [(point("30p30n_jaxa", 1.71e6, 5.5, "simple"),
                 point("30p30n_jaxa", 1.71e6, 5.5, "coupled"))]
        out["multi"] = {"rows": rows, "totals": show(rows, "30P30N Re 1.71e6")}
    VAL.mkdir(exist_ok=True)
    (VAL / "benchmark_coupled.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
