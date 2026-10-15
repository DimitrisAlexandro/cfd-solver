"""
Data-driven correction of the laminar-separation-bubble transition length
(low-Re mode), learned from wind-tunnel data with uncertainty.

    python bubble_ml.py scan            RANS at a ladder of bubble constants K
                                        for every training case (~25 min, 3 jobs)
    python bubble_ml.py fit             K* per case, GP model of log K*, leave-
                                        one-out validation -> validation/bubble_ml_report.md

Model being corrected (transition.py): transition at laminar separation +
Mayle's long-bubble length, Re_l = K Re_theta,s^0.7 with one constant
K = 1000 for every bubble (leading-edge bubbles: not shorter than Horton's).
The Eppler 387 validation (validation/lowre_report.md) shows that one
constant is not right everywhere: Cd within a few % at Re 3e5, up to -22 %
at Re 1e5.

1. Scan: for every training case (Re, alpha) the airfoil polar is run with
   K in a geometric ladder; each run records Cl, Cd and the state of the
   laminar layer at separation (transition.separation_features: Re, Re_theta
   at separation, H, x_sep, U_e, theta dlnU_e/ds).
2. Inversion per case: K* = the constant at which the RANS drag equals the
   measured drag (NASA LTPT, McGhee et al. TM-4062), log-interpolated in K;
   cases whose drag never crosses the measurement keep the best K and are
   flagged (they bound K* but say it is not the only error).
3. Learning: a Gaussian process (surrogate.GP: ARD squared exponential +
   noise) of log K* on the separation features (log Re_theta,s, H_s, x_sep,
   log Re) -> K and its uncertainty for a new bubble.
4. Validation, leave one case out: K predicted from the other cases; the
   drag at that K from the case's own scan (interpolated in log K: no new
   RANS run), against LTPT, compared with K = 1000.
Install for production runs: bubble_ml.install() sets
transition.BUBBLE_K_FN to the model's mean prediction.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "validation"
SCAN = VAL / "bubble_ml_scan.json"
MODEL = ROOT / "data" / "bubble_ml_model.json"
AIRFOIL = str(VAL / "reference" / "e387.dat")
CASES = [(1e5, 0.0), (1e5, 2.0), (2e5, 0.0), (2e5, 2.0), (2e5, 8.0), (3e5, 0.0), (3e5, 2.0),
         (3e5, 4.0), (3e5, 8.0)]
KS = [250.0, 400.0, 650.0, 1000.0, 1500.0, 2300.0]
FEATS = ["log_re_theta_s", "H_s", "x_sep", "log_re"]


def _run(args):
    re, alpha, K, level, max_iter = args
    import transition as T
    import polar as pl
    T.BUBBLE_K = K
    t0 = time.time()
    try:
        r, _ = pl.run_point(AIRFOIL, re, alpha, level=level, max_iter=max_iter, transition="free")
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
        return {"re": re, "alpha": alpha, "K": K, "error": str(e)[:80]}
    est = r.get("transition_estimates") or []
    feats = {}
    for sd in ("upper", "lower"):
        for e in reversed(est):
            f = (e.get(sd) or {}).get("features")
            if f and (e.get(sd) or {}).get("how") == "laminar separation":
                feats[sd] = f
                break
    return {"re": re, "alpha": alpha, "K": K, "Cl": r["Cl"], "Cd": r["Cd"], "status": r["status"],
            "x_tr_upper": r["x_tr_upper"], "x_tr_lower": r["x_tr_lower"], "features": feats,
            "time_s": round(time.time() - t0, 1)}


def scan(jobs=3, level=1.0, max_iter=4000, log=print):
    from multiprocessing import get_context
    done = json.loads(SCAN.read_text()) if SCAN.exists() else []
    have = {(d["re"], d["alpha"], d["K"]) for d in done}
    todo = [(re, a, K, level, max_iter) for re, a in CASES for K in KS if (re, a, K) not in have]
    log(f"[bubble_ml] {len(todo)} runs ({len(done)} already in {SCAN.name})")
    os.environ["CFD_JOBS"] = str(jobs)
    with get_context("fork").Pool(jobs) as pool:
        for r in pool.imap_unordered(_run, todo):
            done.append(r)
            SCAN.write_text(json.dumps(done, indent=1))
            log(f"  Re {r['re']:.0e} alpha {r['alpha']:4.1f} K {r['K']:6.0f}: "
                + (f"Cd {r['Cd']:.5f} Cl {r['Cl']:.3f} x_tr {r['x_tr_upper']} {r['status']} "
                   f"({r['time_s']:.0f} s)" if "Cd" in r else r["error"]), flush=True)
    return done


def ltpt(re, alpha):
    import validate_lowre as vl
    return vl.ltpt_at(re, alpha, 1), vl.ltpt_at(re, alpha, 2)


def invert(rows):
    """K* per case from the scan. Only steady solutions count: past a
    critical K the bubble bursts, the RANS solution turns unsteady and its
    (time-averaged) drag jumps - matching the measured drag on that branch
    would be meaningless. Not bracketed -> linear extrapolation in log K by
    at most one ladder step, flagged."""
    out = []
    for re, a in CASES:
        rr = sorted([r for r in rows if r["re"] == re and r["alpha"] == a and "Cd" in r
                     and str(r["status"]).startswith("converged")], key=lambda r: r["K"])
        if len(rr) < 2:
            continue
        cl_e, cd_e = ltpt(re, a)
        if cd_e is None:
            continue
        lk = np.log([r["K"] for r in rr])
        cd = np.array([r["Cd"] for r in rr])
        f = cd - cd_e
        kstar, how = None, "bracketed"
        for i in range(len(f) - 1):
            if f[i] == 0 or f[i] * f[i + 1] < 0:
                t = f[i] / (f[i] - f[i + 1]) if f[i] != f[i + 1] else 0.0
                kstar = math.exp(lk[i] + t * (lk[i + 1] - lk[i]))
                break
        if kstar is None:
            j = int(np.argmin(np.abs(f)))
            j2 = j + 1 if j + 1 < len(f) else j - 1
            slope = (f[j2] - f[j]) / (lk[j2] - lk[j])
            step = math.log(KS[1] / KS[0])
            dl = float(np.clip(-f[j] / slope, -step, step)) if slope != 0 else 0.0
            kstar, how = math.exp(lk[j] + dl), "extrapolated"
        near = min(rr, key=lambda r: abs(math.log(r["K"] / kstar)))
        ft = near["features"].get("upper") or near["features"].get("lower")
        if not ft:
            continue
        base = next((r for r in rows if r["re"] == re and r["alpha"] == a and r.get("K") == 1000.0
                     and "Cd" in r), None)
        out.append({"re": re, "alpha": a, "K_star": kstar, "crossed": how == "bracketed",
                    "how": how, "Cd_LTPT": cd_e, "Cl_LTPT": cl_e, "features": ft,
                    "lk": lk.tolist(), "cd": cd.tolist(),
                    "Cd_K1000": base["Cd"] if base else None,
                    "K1000_status": base["status"] if base else None})
    return out


def _X(cases):
    return np.array([[math.log10(max(c["features"]["re_theta_s"], 1.0)), c["features"]["H_s"],
                      c["features"]["x_sep"], math.log10(c["re"])] for c in cases])


def fit(log=print):
    from surrogate import GP
    rows = json.loads(SCAN.read_text())
    cases = invert(rows)
    log(f"[bubble_ml] {len(cases)} cases with K*: " + ", ".join(
        f"Re {c['re']:.0e} a {c['alpha']:g}: K* {c['K_star']:.0f}{'' if c['crossed'] else ' (bound)'}"
        for c in cases))
    X = _X(cases)
    y = np.log([c["K_star"] for c in cases])
    # leave one case out
    loo = []
    for i, c in enumerate(cases):
        k = np.arange(len(cases)) != i
        g = GP().fit(X[k], y[k], restarts=3)
        mu, sd = g.predict(X[i:i + 1])
        Kp = float(np.exp(mu[0]))
        cd_p = float(np.interp(math.log(Kp), c["lk"], c["cd"]))      # drag at the predicted K
        loo.append({**{q: c[q] for q in ("re", "alpha", "K_star", "crossed", "Cd_LTPT", "Cd_K1000")},
                    "K_pred": Kp, "K_pred_sd_log": float(sd[0]), "Cd_pred": cd_p,
                    "err_pred": cd_p / c["Cd_LTPT"] - 1,
                    "err_K1000": (c["Cd_K1000"] / c["Cd_LTPT"] - 1) if c["Cd_K1000"] else None})
    g = GP().fit(X, y, restarts=4)
    MODEL.parent.mkdir(exist_ok=True)
    MODEL.write_text(json.dumps({"features": FEATS, "gp": g.to_json(), "cases": cases}, indent=1))
    report(cases, loo, g)
    return cases, loo, g


def report(cases, loo, g):
    L = ["# Learned laminar-separation-bubble length (Eppler 387, LTPT)", "",
         "K* = the Mayle constant (Re_l = K Re_theta,s^0.7) at which the RANS drag equals the "
         "measured drag; a Gaussian process of log K* on the state of the laminar layer at "
         "separation (log Re_theta,s, H_s, x_sep, log Re). Leave-one-out: K predicted from the "
         "other cases, the drag at that K interpolated from the case's own K scan. "
         "Reproduce: `python src/bubble_ml.py scan` then `fit`.", "",
         "| Re | α | K* | K predicted (LOO) | Cd LTPT | Cd, K = 1000 | Cd, learned K | error K=1000 | error learned |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in loo:
        L.append(f"| {r['re']:.0e} | {r['alpha']:g} | {r['K_star']:.0f}{'' if r['crossed'] else ' (extrap.)'} | "
                 f"{r['K_pred']:.0f} (x/÷ {math.exp(2 * r['K_pred_sd_log']):.2f}) | {r['Cd_LTPT']:.4f} | "
                 f"{r['Cd_K1000']:.5f} | {r['Cd_pred']:.5f} | {100 * r['err_K1000']:+.0f} % | "
                 f"{100 * r['err_pred']:+.0f} % |")
    e0 = [abs(r["err_K1000"]) for r in loo if r["err_K1000"] is not None]
    e1 = [abs(r["err_pred"]) for r in loo]
    # sensitivity of the steady drag to the bubble constant: d ln Cd / d ln K
    sens = []
    for c in cases:
        lk, cd = np.array(c["lk"]), np.array(c["cd"])
        if len(lk) >= 2:
            sens.append(float(np.polyfit(lk, np.log(cd), 1)[0]))
    nb = sum(1 for c in cases if c["crossed"])
    L += ["", f"Mean |ΔCd| vs LTPT: K = 1000 {100 * np.mean(e0):.1f} %, learned (leave-one-out) "
          f"{100 * np.mean(e1):.1f} %.", "",
          "## Findings", "",
          f"* Only {nb} of {len(cases)} cases bracket the measured drag on the steady branch. The "
          f"steady drag responds weakly to the bubble constant: d ln Cd / d ln K = "
          f"{np.median(sens):+.3f} (median; range {min(sens):+.3f} to {max(sens):+.3f}) - a factor 2 "
          f"in the bubble length changes Cd by a few per cent at most before the bubble bursts.",
          "* At Re 3e5 the fixed K = 1000 is already within 1-2 % of LTPT; at Re 1e5 no bubble "
          "length on the steady branch reaches the measured drag (-13 to -22 %).",
          "* So the learned K does not beat the constant: the transition location inside the "
          "bubble is not what limits the accuracy. What is left is how the separated shear "
          "layer reattaches and recovers under SA (and the unsteadiness of the bubble that "
          "steady RANS averages away) - a correction of the turbulence model in the shear "
          "layer (the FIML route, fiml.py) is the indicated next step, not a better "
          "transition-length correlation.",
          "* The learned model is therefore not installed by default (bubble_ml.install() "
          "opts in).", "",
          "The band is the GP's 2-sigma factor on K. Only steady RANS solutions enter the "
          "inversion: beyond a critical K the bubble bursts and the solution turns unsteady "
          "(time-averaged drag 50-150 % high), a branch on which matching the measured drag "
          "would be meaningless. (extrap.): the measured drag is not bracketed by the steady "
          "runs; K* is extrapolated by at most one ladder step."]
    (VAL / "bubble_ml_report.md").write_text("\n".join(L) + "\n")


def install(path=MODEL):
    """use the learned K (mean) in transition.py from now on"""
    import transition as T
    from surrogate import GP
    d = json.loads(Path(path).read_text())
    g = GP.from_json(d["gp"])

    def fn(f):
        x = np.array([[math.log10(max(f["re_theta_s"], 1.0)), f["H_s"], f["x_sep"],
                       math.log10(f["re"])]])
        return float(np.clip(np.exp(g.predict(x)[0][0]), 200.0, 5000.0))
    T.BUBBLE_K_FN = fn
    return fn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["scan", "fit"])
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(argv)
    if a.cmd == "scan":
        scan(a.jobs)
    else:
        cases, loo, _ = fit()
        print((VAL / "bubble_ml_report.md").read_text())


if __name__ == "__main__":
    main()
