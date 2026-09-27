"""
Validation 5 (solver step B): time-accurate (unsteady) flow.

Part 1 - Taylor-Green vortex (exact Navier-Stokes solution)
    u = -cos x sin y e^(-2 nu t),  v = sin x cos y e^(-2 nu t)
    on [0, 2pi]^2 with the exact velocity imposed on the boundary (time-
    dependent). Measures the TIME accuracy: error vs time step must fall
    with slope 2 (BDF2). The spatial error is removed by comparing against
    a very small time step on the same mesh; the exact-solution error is
    reported too.

Part 2 - vortex shedding behind a circular cylinder, Re = 100, hybrid mesh
    from our mesher. Strouhal number, mean drag, lift amplitude.
    References (as commonly cited; verify against the originals):
      Williamson (1996), Annu. Rev. Fluid Mech. 28 - experiments,
          St(Re) fit: St = -3.3265/Re + 0.1816 + 1.6e-4 Re  -> 0.164
      Park, Kwon & Choi (1998), KSME Int. J. 12 - numerical:
          Cd 1.33, Cl amplitude 0.33, St 0.165
      Liu, Zheng & Sung (1998), J. Comput. Phys. 139 - numerical:
          Cd 1.35, Cl amplitude 0.339, St 0.164

    python validate_unsteady.py tgv
    python validate_unsteady.py cylinder
    python validate_unsteady.py --plot          # plots from saved results
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from fvmesh import FVMesh, rect_quad
from simple_solver import SimpleSolver

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "validation"
RESULTS = VAL / "unsteady_results.json"

CYL_REF = {
    "St": {"Williamson 1996 (exp. fit)": 0.164, "Park et al. 1998": 0.165,
           "Liu et al. 1998": 0.164},
    "Cd_mean": {"Park et al. 1998": 1.33, "Liu et al. 1998": 1.35},
    "Cl_amp": {"Park et al. 1998": 0.33, "Liu et al. 1998": 0.339},
}


# ----------------------------------------------------------------------
# Taylor-Green vortex
# ----------------------------------------------------------------------
def tgv_exact(x, y, t, nu):
    F = math.exp(-2 * nu * t)
    return np.c_[-np.cos(x) * np.sin(y) * F, np.sin(x) * np.cos(y) * F]


def tgv_run(n=48, dt=0.1, t_end=2.0, nu=0.05, tol=1e-9, max_outer=50):
    m = FVMesh(*rect_quad(n, n, 0, 2 * math.pi, 0, 2 * math.pi))
    f = lambda x, y, t: tgv_exact(x, y, t, nu)
    bcs = {k: {"type": "wall", "U": f} for k in ("bottom", "right", "top", "left")}
    s = SimpleSolver(m, bcs, nu, alpha_u=0.95)
    s.U = tgv_exact(m.xc[:, 0], m.xc[:, 1], 0.0, nu)
    s.F = np.einsum('ij,ij->i', m.interp(s.U), m.Sf)
    s.Ubar_S_old = s.F.copy()
    # consistent initial pressure field (exact), up to a constant
    s.p = -0.25 * (np.cos(2 * m.xc[:, 0]) + np.cos(2 * m.xc[:, 1]))
    s.p -= s.p[0]
    nsteps = int(round(t_end / dt))
    outer = []
    for _ in range(nsteps):
        k, _ = s.advance(dt, max_outer=max_outer, tol=tol)
        outer.append(k)
    ex = tgv_exact(m.xc[:, 0], m.xc[:, 1], s.time, nu)
    err = np.sqrt((((s.U - ex) ** 2).sum(1) * m.vol).sum() / m.vol.sum())
    return {"dt": dt, "n": n, "U": s.U, "err_exact": float(err), "t": s.time,
            "outer_mean": float(np.mean(outer)), "vol": m.vol}


def study_tgv(db, n=48):
    dts = [0.4, 0.2, 0.1, 0.05]
    ref = tgv_run(n=n, dt=0.00625)
    rows = []
    for dt in dts:
        r = tgv_run(n=n, dt=dt)
        e_t = np.sqrt((((r["U"] - ref["U"]) ** 2).sum(1) * r["vol"]).sum() / r["vol"].sum())
        rows.append({"dt": dt, "err_time": float(e_t), "err_exact": r["err_exact"],
                     "outer_mean": r["outer_mean"]})
        print(f"  dt {dt:6.3f}: time error {e_t:.3e}  (vs exact {r['err_exact']:.3e}), "
              f"outer iterations/step {r['outer_mean']:.1f}", flush=True)
    for i in range(1, len(rows)):
        rows[i]["order"] = math.log(rows[i - 1]["err_time"] / rows[i]["err_time"]) / math.log(2)
    print("  observed temporal order: " +
          ", ".join(f"{r['order']:.2f}" for r in rows[1:]))
    db["tgv"] = {"n": n, "rows": rows, "err_exact_ref_dt": ref["err_exact"]}
    return db


# ----------------------------------------------------------------------
# cylinder vortex shedding
# ----------------------------------------------------------------------
def cylinder_run(re=100.0, refine=1.0, dt=0.05, t_end=200.0, t_kick=(2.0, 6.0),
                 max_outer=20, tol=1e-5, verbose=True, record_every=1, snapshot_times=()):
    """Start from uniform flow; a short counter-clockwise spin of the cylinder
    (t_kick) breaks the symmetry so shedding starts quickly, as a physical
    disturbance would."""
    import validate_cylinder as vc
    from postproc import wall_data
    fvm, mesh = vc.cylinder_mesh(refine=refine)
    omega = 0.5

    def wall_U(x, y, t):
        if t_kick[0] <= t <= t_kick[1]:
            return np.c_[-omega * y, omega * (x - 0.5)]     # rigid rotation about (0.5, 0)
        return np.zeros((len(x), 2))

    bcs = {"cylinder": {"type": "wall", "U": wall_U},
           "inlet": {"type": "inlet", "U": (1.0, 0.0)}, "outlet": {"type": "outlet"}}
    s = SimpleSolver(fvm, bcs, 1.0 / re, U0=(1.0, 0.0), alpha_u=0.9)
    t_hist, cd, cl, outer = [], [], [], []
    snaps = {}
    t0 = time.time()
    nsteps = int(round(t_end / dt))
    for i in range(1, nsteps + 1):
        k, r = s.advance(dt, max_outer=max_outer, tol=tol)
        outer.append(k)
        if i % record_every == 0:
            _, c = wall_data(s, "cylinder", (1.0, 0.0))
            t_hist.append(s.time); cd.append(c["Cd"]); cl.append(c["Cl"])
        for ts in snapshot_times:
            if abs(s.time - ts) < 0.5 * dt:
                snaps[ts] = {"U": s.U.copy(), "p": s.p.copy(), "gU": [g.copy() for g in s.gU]}
        if verbose and i % 200 == 0:
            print(f"    t {s.time:7.2f}  Cd {cd[-1]:.4f}  Cl {cl[-1]:+.4f}  outer {np.mean(outer[-200:]):.1f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    res = {"Re": re, "refine": refine, "dt": dt, "cells": int(fvm.ncells),
           "t": t_hist, "Cd": cd, "Cl": cl, "outer_mean": float(np.mean(outer)),
           "time_s": round(time.time() - t0, 1)}
    res.update(shedding_stats(np.array(t_hist), np.array(cd), np.array(cl)))
    return res, s, mesh, snaps


def shedding_stats(t, cd, cl, n_periods=8):
    """Strouhal number from the lift-coefficient period (zero up-crossings, linear
    interpolation), mean Cd and Cl amplitude over the last n_periods."""
    up = np.where((cl[:-1] < 0) & (cl[1:] >= 0))[0]
    if len(up) < n_periods + 2:
        return {"St": None, "Cd_mean": None, "Cl_amp": None, "Cd_amp": None, "periods": len(up)}
    tc = t[up] - cl[up] * (t[up + 1] - t[up]) / (cl[up + 1] - cl[up])
    tc = tc[-(n_periods + 1):]
    T = np.diff(tc)
    sel = (t >= tc[0]) & (t <= tc[-1])
    return {"St": float(1.0 / T.mean()), "St_spread": float(np.ptp(1.0 / T)),
            "Cd_mean": float(cd[sel].mean()), "Cd_amp": float(0.5 * np.ptp(cd[sel])),
            "Cl_amp": float(0.5 * np.ptp(cl[sel])), "periods": int(len(up))}


def study_cylinder(db, cases=((0.7, 0.1), (0.7, 0.05), (1.0, 0.05)), t_end=200.0):
    for refine, dt in cases:
        key = f"cyl_r{refine:g}_dt{dt:g}"
        if key in db:
            print(f"  (cached) {key}")
            continue
        print(f"  cylinder Re 100, refine x{refine}, dt {dt}:", flush=True)
        res, _, _, _ = cylinder_run(refine=refine, dt=dt, t_end=t_end)
        db[key] = res
        _save(db)
        print(f"  -> St {res['St']}, Cd mean {res['Cd_mean']}, Cl amp {res['Cl_amp']} "
              f"({res['cells']} cells, {res['time_s']:.0f}s)", flush=True)
    return db


# ----------------------------------------------------------------------
def _load():
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def _save(db):
    VAL.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(db))


def report(db):
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    if "tgv" in db:
        P(f"TAYLOR-GREEN VORTEX ({db['tgv']['n']}x{db['tgv']['n']} quads, t = 2):")
        for r in db["tgv"]["rows"]:
            o = f"   order {r['order']:.2f}" if "order" in r else ""
            P(f"  dt {r['dt']:6.3f}   time error {r['err_time']:.3e}{o}   "
              f"(error vs exact {r['err_exact']:.3e})")
    keys = sorted(k for k in db if k.startswith("cyl_"))
    if keys:
        P("CYLINDER Re = 100 (vortex shedding):")
        for k in keys:
            r = db[k]
            if r["St"] is None:
                P(f"  {k}: shedding not established ({r['periods']} periods)")
                continue
            P(f"  refine x{r['refine']:<4g} dt {r['dt']:<5g} ({r['cells']:6d} cells): "
              f"St {r['St']:.4f}   Cd mean {r['Cd_mean']:.4f} (+-{r['Cd_amp']:.4f})   "
              f"Cl amp {r['Cl_amp']:.4f}")
        a, b = db.get("cyl_r0.7_dt0.05"), db.get("cyl_r1_dt0.05")
        if a and b and a["St"] and b["St"]:
            rr = 1.0 / 0.7
            ext = {q: b[q] + (b[q] - a[q]) / (rr ** 2 - 1) for q in ("St", "Cd_mean", "Cl_amp")}
            P(f"  Richardson estimate (2 meshes, assumed order 2 - indicative only): "
              f"St {ext['St']:.4f}   Cd mean {ext['Cd_mean']:.4f}   Cl amp {ext['Cl_amp']:.4f}")
        t1, t2 = db.get("cyl_r0.7_dt0.1"), db.get("cyl_r0.7_dt0.05")
        if t1 and t2 and t1["St"] and t2["St"]:
            P(f"  time-step effect (dt 0.1 -> 0.05): St {abs(t2['St'] - t1['St']) / t2['St'] * 100:.2f} %, "
              f"Cd {abs(t2['Cd_mean'] - t1['Cd_mean']) / t2['Cd_mean'] * 100:.2f} %, "
              f"Cl amp {abs(t2['Cl_amp'] - t1['Cl_amp']) / t2['Cl_amp'] * 100:.2f} %")
        for q, lab in (("St", "Strouhal"), ("Cd_mean", "mean Cd"), ("Cl_amp", "Cl amplitude")):
            P(f"  reference {lab:13s}: " + ", ".join(f"{a} {v}" for a, v in CYL_REF[q].items()))
    VAL.mkdir(exist_ok=True)
    (VAL / "unsteady_report.txt").write_text("\n".join(lines) + "\n")


def plots(db):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "tgv" in db:
        rows = db["tgv"]["rows"]
        fig, ax = plt.subplots(figsize=(6, 4.8))
        dts = np.array([r["dt"] for r in rows])
        ax.loglog(dts, [r["err_time"] for r in rows], "o-", label="time error (vs dt -> 0)")
        ax.loglog(dts, [r["err_exact"] for r in rows], "s--", label="total error vs exact")
        ax.loglog(dts, rows[0]["err_time"] * (dts / dts[0]) ** 2, "k:", label="slope 2")
        ax.set_xlabel("time step"); ax.set_ylabel("L2 velocity error at t = 2")
        ax.set_title("Taylor-Green vortex: temporal convergence (BDF2)")
        ax.grid(alpha=.3, which="both"); ax.legend()
        fig.tight_layout(); fig.savefig(VAL / "unsteady_tgv.png", dpi=140)
    keys = sorted(k for k in db if k.startswith("cyl_") and db[k]["St"] is not None)
    if keys:
        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for k in keys:
            r = db[k]
            lab = f"refine x{r['refine']:g}, dt {r['dt']:g}"
            ax[0].plot(r["t"], r["Cd"], lw=0.9, label=lab)
            ax[1].plot(r["t"], r["Cl"], lw=0.9, label=lab)
        for v in CYL_REF["Cd_mean"].values():
            ax[0].axhline(v, color="k", ls=":", lw=0.8)
        ax[0].set_ylabel("Cd"); ax[1].set_ylabel("Cl"); ax[1].set_xlabel("time (D/U)")
        ax[0].set_title("Cylinder Re = 100: force history (dotted: literature mean Cd)")
        ax[0].set_ylim(1.2, 1.6)
        for a in ax:
            a.grid(alpha=.3); a.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(VAL / "unsteady_cylinder_forces.png", dpi=140)
    print("plots saved in", VAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="*", default=["tgv", "cylinder"])
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    db = _load()
    if not a.plot:
        if "tgv" in a.what:
            print("== Taylor-Green vortex", flush=True)
            db = study_tgv(db); _save(db)
        if "cylinder" in a.what:
            print("== cylinder Re 100", flush=True)
            db = study_cylinder(db)
    report(db)
    plots(db)


if __name__ == "__main__":
    main()
