"""
Phase 3, E5 validation: 30P30N three-element high-lift airfoil (JAXA model)
against the JAXA LWT2 wind-tunnel pressure data, SA, fully turbulent.

    python src/run.py validate_multi            runs (~1-2 h on 2 cores) + report
    python src/run.py validate_multi --plots    report/plots from validation/multi_30p30n.json

Case (AIAA BANC-III/IV Category 7, JAXA APC-IV): M 0.17, Re 1.71e6 on the
stowed chord; the JAXA README gives the open-air CFD (SA) angles that match
the tunnel data: 5.5, 9.5 and 14 deg (uncorrected 7 / 11 / 16 deg,
Kevlar-wall test 2017; 6 / 10 deg hard-wall test 2013). Geometry and data:
validation/reference/30p30n_jaxa (SOURCE.txt).

1. SA at alpha 5.5, 9.5, 14 (mesh level 1), Cp x Prandtl-Glauert (M 0.17,
   factor 1.0148) against the measured Cp at every pressure tap: each tap is
   placed on the element surface (taps follow each other around the surface)
   and the SA Cp is interpolated there -> mean / max |dCp| per element
2. grid study at alpha 5.5: levels 0.7 / 1 / 1.4
3. inviscid panel method on the same outlines (the slot effects without the
   boundary layers)
Writes validation/multi_30p30n.json, multi_30p30n_report.txt,
multi_30p30n_cp.png, multi_30p30n_grid.png.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from aerolab.airfoil import multi_element as me
from aerolab.airfoil import panel
from aerolab.airfoil import polar as pl

ROOT = Path(__file__).resolve().parents[3]
VAL = ROOT / "validation"
REF = VAL / "reference" / "30p30n_jaxa"
OUT = VAL / "multi_30p30n.json"
SPEC = "30p30n_jaxa"
RE = 1.71e6
MACH = 0.17
PG = 1.0 / math.sqrt(1 - MACH**2)
ALPHAS = (5.5, 9.5, 14.0)
GRID_ALPHA = 5.5
# SU2 V&V (compressible SA-noft2, Roe + van Albada, JAXA APC-IV grids L1-L5):
# grid-converged values at alpha 5.5 from the quadratic fits on
# su2code.github.io/vandv/30p30n (lift.png / drag.png): Cl 2.89602, Cd 0.04281
SU2_A55 = {"Cl": 2.89602, "Cd": 0.04281}
LEVELS = (0.7, 1.4)                     # + level 1 from the sweep


def run():
    res = {"Re": RE, "Mach": MACH, "spec": SPEC, "sweep": [], "grid": [], "inviscid": []}
    t0 = time.time()
    res["sweep"] = pl.polar(SPEC, RE, ALPHAS, level=1.0, jobs=2, transition="off",
                            max_iter=10000)
    for r in res["sweep"]:
        print(f"[sweep] alpha {r['alpha']:5.1f}: Cl {r['Cl']:.4f}  Cd {r['Cd']:.5f}  "
              f"Cm {r['Cm']:+.4f}  {r['status']}  {r['iterations']} it  {r['time_s']:.0f} s",
              flush=True)
    OUT.write_text(json.dumps(res))
    for lv in LEVELS:
        r, _ = pl.run_point(SPEC, RE, GRID_ALPHA, level=lv, transition="off", max_iter=12000)
        r["level"] = lv
        res["grid"].append(r)
        print(f"[grid] level {lv}: {r['cells']} cells  Cl {r['Cl']:.4f}  Cd {r['Cd']:.5f}  "
              f"{r['status']}  {r['iterations']} it  {r['time_s']:.0f} s", flush=True)
        OUT.write_text(json.dumps(res))
    el, _ = me.load_multi(SPEC, level=2.0)
    for a in ALPHAS:
        pr = panel.solve([e["outline"] for e in el], a, names=[e["name"] for e in el])
        res["inviscid"].append({"alpha": a, "Cl": pr["Cl"], "Cm": pr["Cm"],
                                "elements": {e["name"]: {"Cl": e["Cl"]} for e in pr["elements"]}})
    res["wall_time_s"] = round(time.time() - t0)
    OUT.write_text(json.dumps(res))
    return res


# ----------------------------------------------------------------------
# taps on the surface
# ----------------------------------------------------------------------
def _arc(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def _crossings(P, s, x):
    """Arc positions where the open outline P crosses the vertical line x."""
    out = []
    for i in range(len(P) - 1):
        a, b = P[i, 0] - x, P[i + 1, 0] - x
        if a == 0.0:
            out.append(s[i])
        elif a * b < 0:
            out.append(s[i] + a / (a - b) * (s[i + 1] - s[i]))
    if P[-1, 0] == x:
        out.append(s[-1])
    return out


def place_taps(outline, xs):
    """Taps given only by x/c. The JAXA/BANC files list them from the
    trailing edge along the pressure side (slat: the cove side) to the
    leading edge and back along the suction side; for our counter-clockwise
    outlines (s = 0 at the upper TE end) that is DEcreasing arc length.
    (From x alone the mirrored path is just as consistent, so the direction
    must be fixed.) Each tap takes the next crossing in that direction; only
    the first step may pass the trailing edge (a tap on the TE base).
    Returns (arc positions, outline arc lengths)."""
    s = _arc(outline)
    L = s[-1]
    cand = [_crossings(outline, s, x) or [s[np.argmin(np.abs(outline[:, 0] - x))]] for x in xs]
    best = None
    for c0 in cand[0]:
        path, cur, cost, ok = [c0], c0, 0.0, True
        for k, cs in enumerate(cand[1:]):
            steps = [((cur - c) % L if k == 0 else cur - c, c) for c in cs]
            steps = [(d, c) for d, c in steps if d >= -1e-9]
            if not steps:
                ok = False
                break
            d, c = min(steps)
            path.append(c); cost += d; cur = c
        if ok and (best is None or cost < best[0]):
            best = (cost, path)
    if best is None:
        raise ValueError("could not place the pressure taps on the surface")
    return np.array(best[1]), s


def compare_cp(r, el, exp):
    """SA Cp (x PG) at the taps; returns per element dict of arrays."""
    w = {k: np.asarray(v) for k, v in r["wall"].items()}
    out = {}
    for k, e in enumerate(el):
        rows = exp[exp["element"] == e["name"]]
        if len(rows) == 0:
            continue
        st, s = place_taps(e["outline"], rows["x"])
        m = w["element"] == k
        P = np.c_[w["x"][m], w["y"][m]]
        # wall faces -> arc position on the outline (nearest outline segment)
        a, b = e["outline"][:-1], e["outline"][1:]
        ab = b - a
        t = np.clip(np.einsum('pij,ij->pi', P[:, None] - a[None], ab) /
                    np.maximum((ab**2).sum(1), 1e-300)[None], 0, 1)
        q = a[None] + t[..., None] * ab[None]
        j = np.argmin(np.linalg.norm(P[:, None] - q, axis=2), axis=1)
        sw = s[j] + t[np.arange(len(j)), j] * np.linalg.norm(ab[j], axis=1)
        o = np.argsort(sw)
        cp = np.interp(st, sw[o], w["Cp"][m][o] * PG)
        out[e["name"]] = {"x": rows["x"], "exp": rows["Cp"], "sa": cp}
    return out


def _read_exp(alpha, test):
    p = REF / f"exp_cp_a{alpha:g}_{test}.csv"
    if not p.exists():
        return None
    return np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding=None,
                         skip_header=1)


def report(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    el, _ = me.load_multi(SPEC, level=1.0)
    sw = sorted(res["sweep"], key=lambda r: r["alpha"])
    inv = {r["alpha"]: r for r in res["inviscid"]}
    el_inv = None
    for r in sw:                                   # (partial results file)
        if r["alpha"] not in inv:
            el_inv = el_inv or me.load_multi(SPEC, level=2.0)[0]
            inv[r["alpha"]] = {"Cl": panel.solve([e["outline"] for e in el_inv], r["alpha"])["Cl"]}
    names = sw[0]["element_names"]
    lines = [f"30P30N JAXA model (thick TEs), SA-noft2 fully turbulent, Re {RE:.3g} (stowed chord), "
             f"incompressible; Cp compared x {PG:.4f} (Prandtl-Glauert, M {MACH:g})",
             "experiment: JAXA LWT2 (Murayama et al. 2014, 2018), open-air CFD angles per the "
             "JAXA README", ""]
    lines.append(f"{'alpha':>6s} {'cells':>7s} {'Cl':>8s} {'Cl_PG':>8s} {'Cd':>9s} {'Cm':>8s} "
                 f"{'Cl_inv':>8s}  " + "  ".join(f"{n:>7s}" for n in names) + "  status")
    for r in sw:
        lines.append(f"{r['alpha']:6.1f} {r['cells']:7d} {r['Cl']:8.4f} {r['Cl'] * PG:8.4f} "
                     f"{r['Cd']:9.5f} {r['Cm']:+8.4f} {inv[r['alpha']]['Cl']:8.4f}  " +
                     "  ".join(f"{r['elements'][n]['Cl']:7.4f}" for n in names) +
                     f"  {r['status']}, {r['iterations']} it, {r['time_s']:.0f} s")
    lines += ["", "Cp at the pressure taps: mean |Cp_SA - Cp_exp| (max) per element"]
    cmp_all = {}
    for r in sw:
        for test in ("kevlar2017", "hardwall2013"):
            exp = _read_exp(r["alpha"], test)
            if exp is None:
                continue
            c = compare_cp(r, el, exp)
            cmp_all[(r["alpha"], test)] = c
            lines.append(f"  alpha {r['alpha']:4.1f} vs {test:13s}: " + "   ".join(
                f"{n} {np.mean(np.abs(v['sa'] - v['exp'])):.3f} ({np.max(np.abs(v['sa'] - v['exp'])):.3f})"
                for n, v in c.items()))
    g = sorted(res["grid"] + [r for r in sw if r["alpha"] == GRID_ALPHA],
               key=lambda r: r["cells"])
    if len(g) >= 2:
        lines += ["", f"grid study, alpha {GRID_ALPHA:g}:"]
        for r in g:
            lines.append(f"  {r['cells']:7d} cells: Cl {r['Cl']:.4f}  Cd {r['Cd']:.5f}  "
                         f"Cm {r['Cm']:+.4f}  {r['status']}")
        f = np.array([r["Cl"] for r in g]); d = np.array([r["Cd"] for r in g])
        lines.append("  vs SU2 grid-converged (compressible, M 0.17): " + "; ".join(
            f"{r['cells']} cells Cl x PG {r['Cl'] * PG:.4f} ({100 * (r['Cl'] * PG / SU2_A55['Cl'] - 1):+.2f} %), "
            f"Cd (Cd_p x PG) {r['Cd_p'] * PG + r['Cd_v']:.5f} "
            f"({100 * ((r['Cd_p'] * PG + r['Cd_v']) / SU2_A55['Cd'] - 1):+.1f} %)" for r in g))
        lines.append(f"  Cl change coarse->medium {100 * (f[1] / f[0] - 1):+.2f} %" +
                     (f", medium->fine {100 * (f[2] / f[1] - 1):+.2f} %" if len(g) > 2 else "") +
                     f";  Cd {100 * (d[1] / d[0] - 1):+.2f} %" +
                     (f", {100 * (d[2] / d[1] - 1):+.2f} %" if len(g) > 2 else ""))
    (VAL / "multi_30p30n_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Cp plots: one row per angle; whole airfoil, slat, flap
    el2, _ = me.load_multi(SPEC, level=2.0)
    fig, ax = plt.subplots(len(sw), 3, figsize=(20, 5.2 * len(sw)), squeeze=False,
                           gridspec_kw={"width_ratios": [2.2, 1, 1]})
    for row, r in zip(ax, sw):
        w = {k: np.asarray(v) for k, v in r["wall"].items()}
        pr = panel.solve([e["outline"] for e in el2], r["alpha"], names=[e["name"] for e in el2])
        for g_, (xl, sel) in zip(row, (((-0.1, 1.15), None), ((-0.09, 0.025), "slat"),
                                       ((0.86, 1.14), "flap"))):
            for k, e in enumerate(pr["elements"]):
                if sel and e["name"] != sel:
                    continue
                g_.plot(e["x"], e["Cp"] * PG, ":", color="0.45", lw=0.9,
                        label="inviscid (panel)" if k == 0 or sel else None)
                m = w["element"] == k
                g_.plot(w["x"][m], w["Cp"][m] * PG, ".", ms=2.2, color=f"C{k}",
                        label=f"SA {e['name']}")
            for test, mk in (("kevlar2017", "ko"), ("hardwall2013", "k^")):
                exp = _read_exp(r["alpha"], test)
                if exp is None:
                    continue
                m = np.ones(len(exp), bool) if sel is None else exp["element"] == sel
                g_.plot(exp["x"][m], exp["Cp"][m], mk, mfc="none", ms=5,
                        label=f"JAXA LWT2 {test}")
            # y range from SA + experiment: the panel method's Cp is singular at
            # the sharp corners (slat cusp, trailing edges) and would set the scale
            k_sel = [k for k, e in enumerate(pr["elements"]) if not sel or e["name"] == sel]
            cps = np.concatenate([w["Cp"][w["element"] == k] * PG for k in k_sel])
            g_.set_ylim(max(1.2, cps.max() + 0.3), cps.min() - 0.6)
            g_.set_xlim(*xl); g_.grid(alpha=.3); g_.set_xlabel("x/c")
            g_.set_ylabel("Cp")
        row[0].set_title(f"alpha {r['alpha']:g} deg: SA Cl {r['Cl'] * PG:.3f} (x PG), "
                         f"inviscid {pr['Cl'] * PG:.3f}")
        row[1].set_title("slat"); row[2].set_title("flap")
        row[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"30P30N (JAXA model), M {MACH:g}, Re {RE:.3g}: SA (this solver) vs JAXA "
                 "wind tunnel")
    fig.tight_layout(); fig.savefig(VAL / "multi_30p30n_cp.png", dpi=120); plt.close(fig)

    if len(g) >= 2:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        n = np.array([r["cells"] for r in g], float)
        for g_, k in zip(ax, ("Cl", "Cd")):
            g_.plot(n ** -0.5, [r[k] for r in g], "o-")
            g_.set_xlabel("N^-1/2"); g_.set_ylabel(k); g_.grid(alpha=.3); g_.set_xlim(left=0)
            g_.set_title(f"alpha {GRID_ALPHA:g}: {k} vs mesh size")
        fig.tight_layout(); fig.savefig(VAL / "multi_30p30n_grid.png", dpi=130); plt.close(fig)
    return cmp_all


if __name__ == "__main__":
    res = json.loads(OUT.read_text()) if "--plots" in sys.argv else run()
    report(res)
