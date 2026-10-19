"""
Wing sizing wizard: mission -> wing.

    python src/run.py design_wing_sizing --mass 750 --cruise-speed 60 --cruise-alt 2000 \\
        --stall-speed 25 --airfoil naca2412 [--span-max 11] [--sweep 0]

For every candidate planform (aspect ratio x taper x washout grid):
  1. wing area from the stall requirement, S = 2 W / (rho_0 V_s^2 C_L,max),
     with C_L,max of THAT planform from the quasi-3D wing tool at the
     stall-speed Reynolds numbers (iterated, the area changes the chords);
  2. cruise: C_L = W / (q S), alpha and drag from the wing polar at the
     cruise Reynolds / Mach numbers -> wing L/D;
  3. where the stall starts (first stalled strip, % semi-span): root-first
     stall (< ~50 %) keeps the ailerons working.
Ranked by cruise L/D with the span limit and the stall-pattern check.

Section polars: CFD polars from the cache (cases/wing_polars, made by
wing.py / the app); missing Reynolds numbers are computed (once).
Wing only: no fuselage, tail, trim or high-lift devices.
"""

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from aerolab.aircraft import wing as W

G = 9.80665


def eval_design(W_N, Vs, Vc, rho_c, mu_c, mach_c, polars, AR, taper, twist, sweep, S0,
                n_half=20, rho0=1.225, mu0=1.789e-5, iters=3):
    S = S0
    for _ in range(iters):
        b = math.sqrt(AR * S)
        cr = 2 * S / (b * (1 + taper))
        wg = W.Wing(b, cr, taper * cr, sweep=sweep, twist=twist, n_half=n_half)
        re_s = rho0 * Vs * wg.c / mu0
        res = W.sweep_alpha(wg, np.arange(0, 26, 1.0), polars, re_s, mach=0.0)
        CL = np.array([r["CL"] for r in res])
        i = int(np.argmax(CL))
        clmax = CL[i]
        S_new = 2 * W_N / (rho0 * Vs**2 * clmax)
        if abs(S_new - S) < 1e-3 * S:
            S = S_new
            break
        S = S_new
    # stall onset: first angle with a stalled strip
    eta_stall = None
    for r in res:
        if r["stall_fraction"] > 0:
            st = np.array(r["span"]["stalled"], bool)
            eta_stall = float(np.abs(np.array(r["span"]["eta"]))[st].min())
            break
    b = math.sqrt(AR * S)
    cr = 2 * S / (b * (1 + taper))
    wg = W.Wing(b, cr, taper * cr, sweep=sweep, twist=twist, n_half=n_half)
    q = 0.5 * rho_c * Vc**2
    CLc = W_N / (q * S)
    re_c = rho_c * Vc * wg.c / mu_c
    rc = W.sweep_alpha(wg, np.arange(-4, 14.1, 1.0), polars, re_c, mach=mach_c)
    al = np.array([r["alpha"] for r in rc])
    CLs = np.array([r["CL"] for r in rc])
    ok = np.argmax(CLs) if np.any(np.diff(CLs) < 0) else len(CLs) - 1
    if CLc > CLs[ok]:
        cruise = None
    else:
        a_c = float(np.interp(CLc, CLs[:ok + 1], al[:ok + 1]))
        r = W.solve(wg, a_c, polars, re_c, mach_c)
        cruise = {"alpha": a_c, "CL": r["CL"], "CD": r["CD"], "CDi": r["CDi"], "CDp": r["CDp"],
                  "L_D": r["CL"] / r["CD"], "e": r["e"], "Cm": r["Cm"]}
    return {"AR": AR, "taper": taper, "twist": twist, "sweep": sweep, "S": S, "span": b,
            "root_chord": cr, "tip_chord": taper * cr, "mac": wg.mac, "CLmax": float(clmax),
            "alpha_CLmax": float(res[i]["alpha"]), "stall_onset_eta": eta_stall,
            "CL_cruise": CLc, "cruise": cruise,
            "L_D": cruise["L_D"] if cruise else float("nan")}


def cached_polars(airfoil, polar_dir, mach):
    """every cached section polar of this airfoil (any Re; low-speed ones
    when mach < 0.3, else the nearest Mach)"""
    import re as _re
    from aerolab.airfoil import airfoil_geom as ag
    stem = ag.section_stem(airfoil)
    out = {}
    for f in Path(polar_dir).glob(f"{stem}_Re*_M*.json"):
        m = _re.match(r".+_Re([\d.e+-]+)_M([\d.]+)_", f.name)
        if not m:
            continue
        re, M = float(m.group(1)), float(m.group(2))
        if (mach < 0.3 and M < 0.3) or abs(M - mach) < 0.03:
            out[re] = W.SectionPolar.from_results(json.loads(f.read_text()), ar=10.0, label=f.stem)
    return sorted(out.items())


def run(mass, Vc, hc, Vs, airfoil, polar_dir, span_max=None, sweep=0.0,
        ARs=(6, 8, 10, 12, 14), tapers=(0.35, 0.5, 0.7, 1.0), twists=(0.0, -2.0, -4.0),
        polar_alpha=None, level=1.0, jobs=2, transition="free", log=print, use_cached=False):
    W_N = mass * G
    rho_c, mu_c, a_c, _, _ = W.isa(hc)
    mach_c = Vc / a_c
    rho0, mu0, *_ = W.isa(0.0)
    # rough chord range -> section polars at two Reynolds numbers
    S_est = 2 * W_N / (rho0 * Vs**2 * 1.3)
    c_lo = math.sqrt(S_est / max(ARs)) * 0.5
    c_hi = math.sqrt(S_est / min(ARs)) * 1.4
    re_lo = float(f"{rho0 * Vs * c_lo / mu0:.2g}")
    re_hi = float(f"{rho_c * Vc * c_hi / mu_c:.2g}")
    log(f"[sizing] W {W_N:.0f} N, cruise M {mach_c:.3f}; section polars at Re {re_lo:.3g} "
        f"and {re_hi:.3g}")
    if use_cached:
        table = cached_polars(airfoil, polar_dir, mach_c)
        if not table:
            raise SystemExit(f"no cached polars of {airfoil} in {polar_dir}")
        log("[sizing] using cached polars at Re " + ", ".join(f"{r:.3g}" for r, _ in table))
    else:
        table = W.cfd_polars(airfoil, sorted({re_lo, re_hi}), mach_c,
                             polar_alpha or W.parse_alpha("-6:22:2"), polar_dir, level, jobs, 10.0,
                             transition, log=log)
    polars = W.SpanPolars(table)
    clmax2d = max(p.clmax for _, p in table)
    S0 = 2 * W_N / (rho0 * Vs**2 * 0.9 * clmax2d)
    out = []
    t0 = time.time()
    for AR, lam, tw in itertools.product(ARs, tapers, twists):
        d = eval_design(W_N, Vs, Vc, rho_c, mu_c, mach_c, polars, AR, lam, tw, sweep, S0)
        d["span_ok"] = span_max is None or d["span"] <= span_max
        d["stall_ok"] = d["stall_onset_eta"] is not None and d["stall_onset_eta"] <= 0.6
        out.append(d)
        so = "-" if d["stall_onset_eta"] is None else f"{100 * d['stall_onset_eta']:.0f}%"
        log(f"  AR {AR:5.1f} taper {lam:4.2f} twist {tw:+4.1f}: S {d['S']:6.2f} m2 b {d['span']:5.2f} m "
            f"CLmax {d['CLmax']:.3f} L/D {d['L_D']:5.1f} stall onset {so}")
    ok = [d for d in out if d["span_ok"] and d["cruise"] is not None]
    ranked = sorted(ok, key=lambda d: (not d["stall_ok"], -d["L_D"]))
    log(f"[sizing] {len(out)} planforms in {time.time() - t0:.0f} s")
    return {"mission": {"mass_kg": mass, "weight_N": W_N, "cruise_speed": Vc, "cruise_alt": hc,
                        "stall_speed": Vs, "cruise_mach": mach_c, "span_max": span_max,
                        "sweep": sweep, "airfoil": str(airfoil)},
            "polars": [{"Re": re, "clmax": p.clmax, "label": p.label} for re, p in table],
            "designs": out, "ranked": ranked}, polars


def report(result, polars, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    D = result["designs"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for lam in sorted({d["taper"] for d in D}):
        for tw, ls in zip(sorted({d["twist"] for d in D}, reverse=True), ("-", "--", ":", "-.")):
            s = [d for d in D if d["taper"] == lam and d["twist"] == tw and d["cruise"]]
            if not s:
                continue
            axs[0].plot([d["AR"] for d in s], [d["L_D"] for d in s], ls, marker="o", ms=3,
                        label=f"taper {lam:g}, twist {tw:g}")
            axs[1].plot([d["AR"] for d in s], [d["span"] for d in s], ls, marker="o", ms=3)
            axs[2].plot([d["AR"] for d in s], [100 * (d["stall_onset_eta"] or 1) for d in s], ls,
                        marker="o", ms=3)
    axs[0].set_ylabel("cruise L/D (wing)"); axs[1].set_ylabel("span (m)")
    axs[2].set_ylabel("stall onset (% semi-span)"); axs[2].axhline(60, color="r", lw=0.6)
    if result["mission"]["span_max"]:
        axs[1].axhline(result["mission"]["span_max"], color="r", lw=0.6)
    for a in axs:
        a.set_xlabel("aspect ratio"); a.grid(alpha=.3)
    axs[0].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(out / "sizing.png", dpi=120); plt.close(fig)
    best = result["ranked"][0] if result["ranked"] else None
    if best:
        m = result["mission"]
        rho_c, mu_c, a_c, _, _ = W.isa(m["cruise_alt"])
        wg = W.Wing(best["span"], best["root_chord"], best["tip_chord"], sweep=best["sweep"],
                    twist=best["twist"], n_half=40)
        re_c = rho_c * m["cruise_speed"] * wg.c / mu_c
        res = W.sweep_alpha(wg, np.arange(-4, 22.1, 1.0), polars, re_c, m["cruise_mach"])
        W.report(wg, res, out / "best_wing",
                 f"Best wing: S {best['S']:.2f} m2, b {best['span']:.2f} m, AR {best['AR']:g}, "
                 f"taper {best['taper']:g}, twist {best['twist']:g} (cruise conditions)",
                 {"speed_m_s": m["cruise_speed"], "altitude_m": m["cruise_alt"],
                  "mach": m["cruise_mach"], "rho": rho_c, "re_mac": rho_c * m["cruise_speed"] * wg.mac / mu_c})
    js = json.loads(json.dumps(result, default=float))
    (out / "sizing.json").write_text(json.dumps(js, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mass", type=float, required=True, help="kg")
    ap.add_argument("--cruise-speed", type=float, required=True, help="m/s")
    ap.add_argument("--cruise-alt", type=float, default=0.0, help="m")
    ap.add_argument("--stall-speed", type=float, required=True, help="m/s at sea level")
    ap.add_argument("--airfoil", required=True)
    ap.add_argument("--span-max", type=float)
    ap.add_argument("--sweep", type=float, default=0.0)
    ap.add_argument("--ar", default="6,8,10,12,14")
    ap.add_argument("--taper", default="0.35,0.5,0.7,1.0")
    ap.add_argument("--twist", default="0,-2,-4")
    ap.add_argument("--polar-dir", default="../cases/wing_polars")
    ap.add_argument("--refine", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--transition", choices=["free", "off"], default="free")
    ap.add_argument("--use-cached", action="store_true",
                    help="use the cached section polars of this airfoil (any Re) instead of "
                         "computing new ones at the mission Reynolds numbers")
    ap.add_argument("--out", default="../cases/wing_sizing")
    a = ap.parse_args(argv)
    fl = lambda s: tuple(float(x) for x in s.split(","))
    res, polars = run(a.mass, a.cruise_speed, a.cruise_alt, a.stall_speed, a.airfoil, a.polar_dir,
                      a.span_max, a.sweep, fl(a.ar), fl(a.taper), fl(a.twist), level=a.refine,
                      jobs=a.jobs, transition=a.transition, use_cached=a.use_cached)
    report(res, polars, a.out)
    print("\nbest planforms (cruise L/D, span limit, root-first stall):")
    for d in res["ranked"][:5]:
        print(f"  AR {d['AR']:g} taper {d['taper']:g} twist {d['twist']:g}: S {d['S']:.2f} m2, "
              f"b {d['span']:.2f} m, c {d['root_chord']:.2f}/{d['tip_chord']:.2f} m, "
              f"L/D {d['L_D']:.1f}, CLmax {d['CLmax']:.3f}, stall onset "
              f"{100 * (d['stall_onset_eta'] or 1):.0f}% semi-span")
    print(f"-> {a.out}/ (sizing.png, sizing.json, best_wing/)")


if __name__ == "__main__":
    main()
