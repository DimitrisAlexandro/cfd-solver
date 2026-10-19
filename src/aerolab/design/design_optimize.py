"""
Airfoil optimizer: change the shape until the CFD says it is better.

    python src/run.py design_optimize --start naca2412 --re 3e6 --cl 0.6 --t-min 0.12 --evals 30
    python src/run.py design_optimize --start naca0012 --re 6e6 --cl 0.4,0.7 --cm-min -0.08 ...

Shape: CST (Kulfan) fit of the starting airfoil (8th order per side) plus
smooth perturbations: n_var design variables per side, linearly spread
over the Bernstein coefficients (so the surface stays smooth).
Objective: drag at the target lift coefficient(s) (= max L/D there;
several Cl values -> their average, a robust design); every evaluation
finds the angle for each Cl with the RANS solver (secant, warm start).
Constraints (penalties): thickness >= t_min, Cm >= cm_min, no crossing
surfaces.
Search: surrogate-based (radial-basis-function model of the CFD results
+ trust region, Jones-style); the initial samples are a Latin hypercube.
The best shape is finally re-run on the validated mesh (level 1).
Output: optimize.json (history), best.dat, optimize.png.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from aerolab.airfoil import airfoil_geom as ag
from aerolab.airfoil import cst
from aerolab.airfoil import polar as pl


class Shape:
    def __init__(self, start, n_var=4, order=8):
        self.au0, self.al0, self.fit_err = cst.fit(ag.sharp_outline(start), order)
        self.n_var = n_var
        k = order + 1
        idx = np.linspace(0, k - 1, n_var)
        # hat functions over the coefficient index (smooth surface changes)
        self.B = np.array([np.interp(np.arange(k), idx, np.eye(n_var)[j]) for j in range(n_var)]).T

    def coeffs(self, v):
        v = np.asarray(v, float)
        return self.au0 + self.B @ v[:self.n_var], self.al0 + self.B @ v[self.n_var:]

    def write(self, v, path, name):
        au, al = self.coeffs(v)
        cst.write_dat(path, au, al, name)
        return path


def evaluate(shape, v, re, cls, level, transition, workdir, init=None, cm_min=None, t_min=None,
             log=print):
    au, al = shape.coeffs(v)
    t, xt, tmin = cst.thickness(au, al)
    pen = 0.0
    if tmin < 0:
        return {"ok": False, "obj": 1.0, "t": t, "reason": "surfaces cross"}, init
    if t_min and t < t_min:
        pen += 1.0 * (t_min - t)            # ~ Cd scale: keeps the surrogate smooth
    f = shape.write(v, Path(workdir) / "candidate.dat", "candidate")
    pts, state, slope = [], init, 0.11
    alpha = None
    for cl in cls:
        a = alpha if alpha is not None else cl / slope
        hist = []
        for k in range(5):
            try:
                r, s = pl.run_point(str(f), re, a, level=level, transition=transition, init=state)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
                return {"ok": False, "obj": 1.0, "t": t, "reason": f"diverged ({e})"[:80]}, init
            if r["status"].startswith("converged"):
                state = s
            hist.append((a, r["Cl"]))
            if abs(r["Cl"] - cl) < 0.005:
                break
            if len(hist) >= 2 and abs(hist[-1][0] - hist[-2][0]) > 1e-6:
                sl = (hist[-1][1] - hist[-2][1]) / (hist[-1][0] - hist[-2][0])
                if sl > 0.02:
                    slope = sl
            a = a + (cl - r["Cl"]) / slope
        alpha = r["alpha"]
        pts.append({"cl_target": cl, "alpha": r["alpha"], "Cl": r["Cl"], "Cd": r["Cd"],
                    "Cm": r["Cm"], "status": r["status"], "x_tr_upper": r.get("x_tr_upper"),
                    "x_tr_lower": r.get("x_tr_lower")})
        if cm_min is not None and r["Cm"] < cm_min:
            pen += 0.1 * (cm_min - r["Cm"])
    cd = float(np.mean([p["Cd"] for p in pts]))
    return {"ok": True, "obj": cd + pen, "Cd": cd, "penalty": pen, "t": t, "x_t": xt,
            "points": pts}, state


def optimize(start, re, cls, evals=30, n_var=4, bound=0.04, level=0.7, transition="off",
             t_min=None, cm_min=None, out="../cases/optimize", seed=1, log=print):
    from scipy.interpolate import RBFInterpolator
    from scipy.optimize import minimize
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shape = Shape(start, n_var)
    t0_, *_ = cst.thickness(shape.au0, shape.al0)
    if t_min is None:
        t_min = t0_ * 0.98
    nd = 2 * n_var
    rng = np.random.default_rng(seed)
    X, Y, H = [], [], []
    state = None
    t0 = time.time()

    def ev(v, tag):
        nonlocal state
        r, state_new = evaluate(shape, v, re, cls, level, transition, out, state, cm_min, t_min, log)
        if r["ok"]:
            state = state_new
        X.append(np.array(v)); Y.append(r["obj"])
        H.append({"i": len(H), "tag": tag, "v": list(map(float, v)), **{k: r[k] for k in r if k != "ok"}})
        best = min(Y)
        log(f"  eval {len(H):3d} [{tag:8s}] obj {r['obj']:.5f}"
            + (f" Cd {r['Cd']:.5f} t/c {r['t']:.4f}" if r["ok"] else f" ({r.get('reason')})")
            + f"  best {best:.5f}  ({(time.time() - t0) / 60:.1f} min)")
        (out / "optimize.json").write_text(json.dumps(_summary(), indent=1, default=float))
        return r["obj"]

    def _summary():
        i = int(np.argmin(Y)) if Y else 0
        return {"start": str(start), "re": re, "cl": list(cls), "level": level,
                "transition": transition, "t_min": t_min, "cm_min": cm_min, "n_var": n_var,
                "bound": bound, "history": H, "best": H[i] if H else None,
                "au0": shape.au0.tolist(), "al0": shape.al0.tolist(), "B": shape.B.tolist()}

    ev(np.zeros(nd), "start")
    # Latin hypercube
    n0 = max(nd, 6)
    lhs = (np.argsort(rng.random((n0, nd)), axis=0) + rng.random((n0, nd))) / n0
    for p in lhs:
        if len(X) >= evals:
            break
        ev(bound * (2 * p - 1) * 0.6, "LHS")
    radius = 0.5 * bound
    while len(X) < evals:
        Xa, Ya = np.array(X), np.array(Y)
        ok = Ya < 0.9
        yb = Ya.copy()
        yb[~ok] = Ya[ok].max() * 1.5 if ok.any() else 1.0
        model = RBFInterpolator(Xa, yb, kernel="thin_plate_spline", smoothing=1e-9, degree=1)
        xb = Xa[int(np.argmin(Ya))]
        cands = []
        for _ in range(12):
            x0 = np.clip(xb + rng.uniform(-radius, radius, nd), -bound, bound)
            lo = np.maximum(xb - radius, -bound)
            hi = np.minimum(xb + radius, bound)
            r = minimize(lambda x: float(model(x[None])[0]), x0, method="L-BFGS-B",
                         bounds=list(zip(lo, hi)))
            cands.append((r.fun, r.x))
        cands.sort(key=lambda c: c[0])
        xn = cands[0][1]
        if np.min(np.linalg.norm(Xa - xn, axis=1)) < 1e-3 * bound:      # already sampled
            xn = np.clip(xb + rng.uniform(-radius, radius, nd), -bound, bound)
        before = min(Y)
        ev(xn, "model")
        if Y[-1] < before - 1e-6:
            radius = min(radius * 1.5, bound)
        else:
            radius = max(radius * 0.6, 0.05 * bound)
    summ = _summary()
    best = summ["best"]
    vb = np.array(best["v"])
    shape.write(vb, out / "best.dat", f"optimized from {Path(str(start)).stem}")
    shape.write(np.zeros(nd), out / "start_cst.dat", f"{Path(str(start)).stem} (CST fit)")
    # verification on the validated mesh
    log("[optimize] verification on mesh level 1 (start and best)")
    ver = {}
    for tag, v in (("start", np.zeros(nd)), ("best", vb)):
        r, _ = evaluate(shape, v, re, cls, 1.0, transition, out, None, cm_min, t_min, log)
        ver[tag] = r
        log(f"  {tag}: Cd {r.get('Cd', float('nan')):.5f}")
    summ["verification_level1"] = ver
    (out / "optimize.json").write_text(json.dumps(summ, indent=1, default=float))
    return summ, shape


def plots(summ, shape, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    H = summ["history"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    y = [h["obj"] if h["obj"] < 0.9 else np.nan for h in H]
    axs[0].plot(range(1, len(y) + 1), y, "o", ms=3, color="gray", label="evaluations")
    axs[0].plot(range(1, len(y) + 1), np.fmin.accumulate(np.nan_to_num(y, nan=1.0)), "b-",
                label="best so far")
    axs[0].set_xlabel("CFD evaluation"); axs[0].set_ylabel("objective (Cd at target Cl)")
    axs[0].legend(); axs[0].grid(alpha=.3)
    o0 = cst.outline(*shape.coeffs(np.zeros(2 * shape.n_var)))
    ob = cst.outline(*shape.coeffs(summ["best"]["v"]))
    axs[1].plot(o0[:, 0], o0[:, 1], "k--", label="start")
    axs[1].plot(ob[:, 0], ob[:, 1], "r-", label="optimized")
    axs[1].set_aspect("equal"); axs[1].legend(); axs[1].grid(alpha=.3)
    v = summ.get("verification_level1", {})
    tx = []
    for tag in ("start", "best"):
        r = v.get(tag, {})
        if r.get("points"):
            p = r["points"][0]
            tx.append(f"{tag}: Cd {r['Cd']:.5f}, alpha {p['alpha']:.2f}, Cm {p['Cm']:+.4f}, "
                      f"t/c {r['t']:.4f}")
    axs[1].set_title("\n".join(tx) or "shape", fontsize=8)
    pb = cst.outline(*shape.coeffs(summ["best"]["v"]))
    dy = pb[:, 1] - o0[:, 1]
    axs[2].plot(o0[:, 0], dy * 100, "r-")
    axs[2].set_xlabel("x/c"); axs[2].set_ylabel("shape change (% c)"); axs[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "optimize.png", dpi=120); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="starting airfoil")
    ap.add_argument("--re", type=float, required=True)
    ap.add_argument("--cl", default="0.5", help="target Cl, or several: 0.4,0.7")
    ap.add_argument("--t-min", type=float, help="min thickness/chord (default: 98 %% of start)")
    ap.add_argument("--cm-min", type=float, help="min Cm (c/4), e.g. -0.08")
    ap.add_argument("--evals", type=int, default=30, help="CFD evaluations (each = 2-3 runs per Cl)")
    ap.add_argument("--n-var", type=int, default=4, help="design variables per side")
    ap.add_argument("--bound", type=float, default=0.04, help="max change of a CST coefficient")
    ap.add_argument("--refine", type=float, default=0.7, help="mesh level during the search")
    ap.add_argument("--transition", choices=["off", "free"], default="off")
    ap.add_argument("--out", default="../cases/optimize")
    a = ap.parse_args(argv)
    summ, shape = optimize(a.start, a.re, [float(x) for x in a.cl.split(",")], a.evals, a.n_var,
                           a.bound, a.refine, a.transition, a.t_min, a.cm_min, a.out)
    plots(summ, shape, a.out)
    v = summ["verification_level1"]
    if v["start"].get("Cd") and v["best"].get("Cd"):
        print(f"Cd at Cl {a.cl}: start {v['start']['Cd']:.5f} -> optimized {v['best']['Cd']:.5f} "
              f"({100 * (v['best']['Cd'] / v['start']['Cd'] - 1):+.1f} %) on the validated mesh")
    print(f"-> {a.out}/ (best.dat, optimize.png, optimize.json)")


if __name__ == "__main__":
    main()
