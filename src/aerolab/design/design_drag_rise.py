"""
Drag-rise / transonic explorer: Mach sweep at a fixed lift coefficient.

    python src/run.py design_drag_rise --airfoil naca0012 --re 6e6 --cl 0.3 --mach 0.5:0.82:0.04
    python src/run.py design_drag_rise --airfoil ../validation/reference/rae2822.dat --re 6.5e6 --cl 0.5

For every Mach number (compressible RANS solver, warm-started from the
previous Mach): the angle of attack that gives the target Cl is found by
secant iteration (|dCl| < 0.005, usually 2-3 runs), then Cd, Cm, the
largest local Mach and the flow fields are stored. From Cd(M):
  * drag-divergence Mach, Douglas criterion dCd/dM = 0.1
  * drag-divergence Mach, Boeing criterion: Cd 20 counts above its
    low-speed value
  * Korn's equation (Mason): M_dd = kappa - Cl/10 - t/c, kappa 0.87 for
    conventional and 0.95 for supercritical sections - an independent
    textbook estimate to compare with.
Output: drag_rise.json, drag_rise.png (Cd-M with M_dd, alpha-M, Cm-M,
M_max-M) and drag_rise_fields.png (Mach field with the sonic line per Mach).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from aerolab.airfoil import polar as pl


def find_alpha(spec, re, cl_t, mach, alpha0, slope, init, level, transition, out, tol=0.005,
               max_runs=5, log=print):
    """secant iteration alpha -> Cl = cl_t; returns (result, state, runs)"""
    a, runs, hist = alpha0, 0, []
    st = init
    while runs < max_runs:
        r, s = pl.run_point(spec, re, a, level=level, mach=mach, transition=transition,
                            init=st, out=out)
        runs += 1
        hist.append((a, r["Cl"]))
        log(f"    M {mach:.3f} alpha {a:7.3f}: Cl {r['Cl']:.4f} Cd {r['Cd']:.5f} "
            f"({r['status']}, {r['iterations']} it, {r['time_s']:.0f} s)")
        st = s if r["status"].startswith("converged") else st
        if abs(r["Cl"] - cl_t) < tol:
            break
        if len(hist) >= 2 and abs(hist[-1][0] - hist[-2][0]) > 1e-6:
            sl = (hist[-1][1] - hist[-2][1]) / (hist[-1][0] - hist[-2][0])
            if sl > 0.02:
                slope = sl
        a = a + (cl_t - r["Cl"]) / slope
    return r, st, runs, slope


def run(spec, re, cl_t, machs, level=1.0, transition="off", out="../cases/drag_rise", log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    from aerolab.airfoil import airfoil_geom as ag
    _, info = ag.load_airfoil(spec)
    tc = info["max_thickness"]
    res = []
    state, alpha = None, None
    slope = 0.105                                   # per deg, incompressible estimate
    t0 = time.time()
    for M in machs:
        beta = math.sqrt(max(1 - M**2, 0.05))
        sl = slope if res else 0.105 / beta
        if alpha is None:
            alpha = cl_t / sl
        log(f"[drag rise] M {M:.3f}")
        fdir = out / f"M{M:.3f}"
        r, state, runs, slope = find_alpha(spec, re, cl_t, M, alpha, sl, state, level, transition,
                                           fdir, log=log)
        alpha = r["alpha"]
        res.append({"mach": M, "alpha": r["alpha"], "Cl": r["Cl"], "Cd": r["Cd"], "Cd_p": r["Cd_p"],
                    "Cd_v": r["Cd_v"], "Cm": r["Cm"], "M_max": r.get("M_max"),
                    "supersonic_cells": r.get("supersonic_cells"), "status": r["status"],
                    "runs": runs, "fields": str(fdir / f"alpha_{r['alpha']:+06.2f}")})
        (out / "drag_rise.json").write_text(json.dumps(_summary(spec, re, cl_t, tc, res), indent=1))
    log(f"[drag rise] {len(res)} Mach numbers in {(time.time() - t0) / 60:.1f} min")
    return _summary(spec, re, cl_t, tc, res)


def _summary(spec, re, cl_t, tc, res):
    # points where the target lift could not be reached (shock-induced
    # separation: the lift curve collapses) are reported but not used for M_dd
    for r in res:
        r["cl_reached"] = bool(abs(r.get("Cl", cl_t) - cl_t) < 0.02)
    ok = [r for r in res if r["cl_reached"]]
    M = np.array([r["mach"] for r in ok])
    Cd = np.array([r["Cd"] for r in ok])
    out = {"airfoil": str(spec), "re": re, "cl": cl_t, "thickness": tc, "points": res}
    if len(M) >= 3:
        # Douglas: first Mach where dCd/dM reaches 0.1 (linear interpolation)
        Mm = 0.5 * (M[1:] + M[:-1])
        d = np.diff(Cd) / np.diff(M)
        k = np.where(d >= 0.1)[0]
        if len(k):
            j = k[0]
            out["M_dd_douglas"] = float(Mm[j] if j == 0 else
                                        Mm[j - 1] + (0.1 - d[j - 1]) / (d[j] - d[j - 1]) * (Mm[j] - Mm[j - 1]))
        base = Cd[0]
        k = np.where(Cd >= base + 0.002)[0]
        if len(k) and k[0] > 0:
            j = k[0]
            out["M_dd_boeing"] = float(M[j - 1] + (base + 0.002 - Cd[j - 1]) / (Cd[j] - Cd[j - 1])
                                       * (M[j] - M[j - 1]))
    out["M_dd_korn_conventional"] = 0.87 - cl_t / 10 - tc
    out["M_dd_korn_supercritical"] = 0.95 - cl_t / 10 - tc
    return out


def plots(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from aerolab.core.io_vtk import read_case_mesh, read_vtu_cell_data
    out = Path(out)
    P = summary["points"]
    M = [p["mach"] for p in P]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax = axs[0, 0]
    ax.plot(M, [p["Cd"] for p in P], "bo-", label="Cd")
    bad = [p for p in P if not p.get("cl_reached", True)]
    if bad:
        ax.plot([p["mach"] for p in bad], [p["Cd"] for p in bad], "rx", ms=10,
                label="target Cl not reached (shock stall)")
    ax.plot(M, [p["Cd_p"] for p in P], "g--", lw=0.8, label="pressure (incl. wave)")
    ax.plot(M, [p["Cd_v"] for p in P], "k:", lw=0.8, label="friction")
    for key, col, lab in (("M_dd_douglas", "r", "M_dd (dCd/dM = 0.1)"),
                          ("M_dd_boeing", "m", "M_dd (+20 counts)"),
                          ("M_dd_korn_conventional", "gray", "Korn, conventional")):
        if summary.get(key):
            ax.axvline(summary[key], color=col, ls="--", lw=0.8, label=f"{lab}: {summary[key]:.3f}")
    ax.set_xlabel("Mach"); ax.set_ylabel("Cd"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_title(f"{Path(summary['airfoil']).stem}, Cl = {summary['cl']:g}, Re {summary['re']:.3g}")
    axs[0, 1].plot(M, [p["alpha"] for p in P], "bo-"); axs[0, 1].set_ylabel("alpha for target Cl (deg)")
    axs[1, 0].plot(M, [p["Cm"] for p in P], "bo-"); axs[1, 0].set_ylabel("Cm (c/4)")
    axs[1, 1].plot(M, [p["M_max"] for p in P], "bo-"); axs[1, 1].axhline(1, color="r", lw=0.6)
    axs[1, 1].set_ylabel("max local Mach")
    for a in axs.ravel()[1:]:
        a.set_xlabel("Mach"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "drag_rise.png", dpi=120); plt.close(fig)
    # Mach fields
    sel = [p for p in P if Path(p["fields"], "results.vtu").exists()]
    if sel:
        n = len(sel)
        cols = min(n, 3)
        rows = math.ceil(n / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 2.8 * rows), squeeze=False)
        for ax, p in zip(axs.ravel(), sel):
            Pm, cells = read_case_mesh(p["fields"])
            v = read_vtu_cell_data(Path(p["fields"]) / "results.vtu")["Mach"]
            c = np.array([Pm[k].mean(0) for k in cells])
            s = (c[:, 0] > -0.4) & (c[:, 0] < 1.6) & (np.abs(c[:, 1]) < 0.7)
            t = mtri.Triangulation(c[s, 0], c[s, 1])
            cs = ax.tricontourf(t, v[s], np.linspace(0, 1.5, 31), cmap="jet", extend="max")
            try:
                ax.tricontour(t, v[s], [1.0], colors="k", linewidths=0.8)
            except Exception:
                pass
            ax.set_xlim(-0.3, 1.5); ax.set_ylim(-0.5, 0.6); ax.set_aspect("equal")
            ax.set_title(f"M {p['mach']:.3f}, alpha {p['alpha']:.2f}: Cd {p['Cd']:.4f}", fontsize=9)
        for ax in axs.ravel()[n:]:
            ax.axis("off")
        fig.colorbar(cs, ax=axs, shrink=0.6, label="Mach")
        fig.savefig(out / "drag_rise_fields.png", dpi=110); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--airfoil", required=True)
    ap.add_argument("--re", type=float, required=True)
    ap.add_argument("--cl", type=float, required=True)
    ap.add_argument("--mach", default="0.5:0.82:0.04", help="start:stop:step or list")
    ap.add_argument("--refine", type=float, default=1.0)
    ap.add_argument("--transition", default="off", help="off, or fixed trips 'xu,xl'")
    ap.add_argument("--out", default="../cases/drag_rise")
    a = ap.parse_args(argv)
    if ":" in a.mach:
        s, e, d = (float(x) for x in a.mach.split(":"))
        machs = list(np.round(np.arange(s, e + 0.5 * d, d), 4))
    else:
        machs = [float(x) for x in a.mach.split(",")]
    tr = a.transition
    if tr != "off":
        xu, xl = (float(x) for x in tr.split(","))
        tr = {"upper": xu, "lower": xl}
    summ = run(a.airfoil, a.re, a.cl, machs, a.refine, tr, a.out)
    plots(summ, a.out)
    for k in ("M_dd_douglas", "M_dd_boeing", "M_dd_korn_conventional", "M_dd_korn_supercritical"):
        if k in summ:
            print(f"  {k:26s} {summ[k]:.3f}")
    print(f"-> {a.out}/ (drag_rise.json, drag_rise.png, drag_rise_fields.png)")


if __name__ == "__main__":
    main()
