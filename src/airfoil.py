"""
Airfoil analysis tool (Phase 3, step E3): any airfoil -> polar, one command.

    python airfoil.py polar naca2412 --re 3e6 --alpha -4:16:2
    python airfoil.py polar myfoil.dat --re 1e6 --alpha 0:12:1 --out ../cases/myfoil
    python airfoil.py polar naca0012 --re 6e6 --alpha 0,5,10 --xfoil
    python airfoil.py show myfoil.dat                 geometry check + picture
    python airfoil.py polar 30p30n_jaxa --re 1.71e6 --alpha 5.5,9.5,14 --mach 0.17
    python airfoil.py polar naca23012+naca23012:0.3,0.98,-0.04,25 --re 3e6
    python airfoil.py show 30p30n --alpha 8          multi-element: geometry +
                                                      inviscid (panel) Cp

Multi-element airfoils (E5): the built-in 30p30n (original coordinates) or
30p30n_jaxa (JAXA wind-tunnel model, validated against its Cp), a JSON file listing the
elements, or elements joined by '+' (airfoil[:chord,x_le,y_le,deflection]);
see multi_element.py. Fully turbulent SA; per-element Cl/Cd/Cm in the
report and CSV; the inviscid panel-method lift is overlaid.

Airfoil: naca4412 / naca23012 (NACA 4- and 5-digit), tmr0012 (NASA TMR NACA
0012), or a coordinate file (.dat, Selig or Lednicer format). By default the
coordinates are only translated and scaled (angle of attack measured from the
file's x axis, as in XFOIL); --derotate puts the chord line on the x axis.

Flow model: Re >= 1e5 Spalart-Allmaras with free transition (e^N envelope
method, N_crit 9; --transition off = fully turbulent, --xtr = fixed trips),
Re < 1e5 laminar. Incompressible by default; --mach M (Phase 5) runs the
compressible solver (ideal gas, energy equation, shocks captured; free
transition only below M 0.4, fixed trips / fully turbulent above);
--mach M --prandtl-glauert keeps the incompressible solution and corrects it.

    python airfoil.py polar rae2822.dat --re 6.5e6 --mach 0.734 --alpha 0:4:1 --transition off

Output folder (default ../cases/<airfoil>_Re<re>):
    polar.csv            alpha, Cl, Cd, Cd_p, Cd_v, Cm, L/D, status, iterations ...
    polar.png            Cl-alpha, drag polar, Cm-alpha, L/D (+ XFOIL with --xfoil)
    distributions.png    Cp and Cf at every angle
    report.txt           settings, table, warnings
    geometry.png         the airfoil as meshed
    alpha_+04.00/ ...    full fields per angle (open with viewer.py)
    polar.json           everything above, machine-readable
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import airfoil_geom as ag
import multi_element as me


def parse_alpha(s):
    """'-4:16:2' (start:stop:step, stop included) or '0,5,10'."""
    if ":" in s:
        a, b, st = (float(v) for v in s.split(":"))
        n = int(round((b - a) / st)) + 1
        return [round(a + i * st, 6) for i in range(n)]
    return [float(v) for v in s.split(",")]


# ----------------------------------------------------------------------
def _xfoil_xtr(xf):
    """XFOIL's transition locations (module variable XOCTR, double precision)."""
    import ctypes
    try:
        v = (ctypes.c_double * 2).in_dll(xf._lib, "__i_xfoil_MOD_xoctr")
        return float(v[0]), float(v[1])
    except (ValueError, AttributeError):
        return float("nan"), float("nan")


def run_xfoil(spec, re, alphas, xtr=None, derotate=False, ncrit=9.0):
    """XFOIL on the same (normalised) coordinates. xtr=(top, bottom) forces
    transition (0.01 ~ fully turbulent, to compare with the SA solver);
    None = free transition (e^N, N = ncrit). Returns dict
    alpha -> (Cl, Cd, Cm, x_tr upper, x_tr lower)."""
    try:
        from xfoil import XFoil
        from xfoil.model import Airfoil
    except ImportError:
        print("[xfoil] not installed - skipping the XFOIL comparison")
        return {}
    loop, _ = ag.load_airfoil(spec, n_side=120, derotate=derotate, n_te=2)
    # XFOIL wants TE(upper) -> LE -> TE(lower), no base points
    up_te = int(np.argmax(np.where(loop[:, 0] > 0.999, loop[:, 1], -1)))
    lo_te = int(np.argmin(np.where(loop[:, 0] > 0.999, loop[:, 1], 1)))
    pts = np.roll(loop, -up_te, axis=0)
    k = (lo_te - up_te) % len(loop)
    pts = pts[:k + 1]
    xf = XFoil()
    xf.print = False
    xf.airfoil = Airfoil(pts[:, 0].copy(), pts[:, 1].copy())
    xf.repanel(n_nodes=240)
    xf.Re = re
    xf.max_iter = 200
    if xtr is not None:
        xf.xtr = xtr
    xf.n_crit = ncrit
    out = {}
    for a in sorted(alphas, key=abs):
        cl, cd, cm, cp = xf.a(a)
        if np.isfinite(cl):
            xt = _xfoil_xtr(xf)
            out[a] = (float(cl), float(cd), float(cm), xt[0], xt[1])
    return out


def apply_prandtl_glauert(res, mach):
    """Low-speed compressibility correction of the pressure-derived
    coefficients: C = C_incompressible / sqrt(1 - M^2) (Cl, Cm, Cd_p, Cp).
    Friction drag is left unchanged. Valid for M below ~0.3 (no shocks)."""
    if not 0 < mach < 0.7:
        raise ValueError("Prandtl-Glauert correction: need 0 < M < 0.7 (and it is only "
                         "accurate below ~0.3)")
    b = 1.0 / math.sqrt(1 - mach**2)
    for r in res:
        for k in ("Cl", "Cm", "Cd_p"):
            r[k + "_incompressible"] = r[k]
            r[k] = r[k] * b
        r["Cd"] = r["Cd_p"] + r["Cd_v"]
        for e in (r.get("elements") or {}).values():
            for k in ("Cl", "Cm", "Cd_p"):
                e[k] = e[k] * b
            e["Cd"] = e["Cd_p"] + e["Cd_v"]
        if "Cl_inviscid" in r:
            r["Cl_inviscid"] = r["Cl_inviscid"] * b
        r["wall"]["Cp"] = (np.asarray(r["wall"]["Cp"]) * b).round(7).tolist()
        r["mach_correction"] = f"Prandtl-Glauert M={mach:g}"
    return res


# ----------------------------------------------------------------------
def plot_polar(res, path, title, xf=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = np.array([r["alpha"] for r in res])
    cl = np.array([r["Cl"] for r in res])
    cd = np.array([r["Cd"] for r in res])
    cm = np.array([r["Cm"] for r in res])
    ok = np.array([r["status"].startswith("converged") for r in res])
    has_tr = any(r.get("x_tr_upper") is not None for r in res)
    fig, ax = plt.subplots(2, 3 if has_tr else 2, figsize=(19 if has_tr else 13, 9))
    P4 = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]
    lab0 = ("this solver (SA, " + res[0].get("transition", "fully turbulent") + ")"
            if res[0].get("turbulent", True) else "this solver (laminar)")
    for q, (x, y, xl, yl) in enumerate(((a, cl, "alpha (deg)", "Cl"), (cd, cl, "Cd", "Cl"),
                                        (a, cm, "alpha (deg)", "Cm (c/4)"),
                                        (a, cl / cd, "alpha (deg)", "L/D"))):
        g = P4[q]
        g.plot(x, y, "-", color="C3", lw=1.2)
        g.plot(x[ok], y[ok], "o", color="C3", ms=5, label=lab0)
        if (~ok).any():
            g.plot(x[~ok], y[~ok], "o", mfc="none", color="C3", ms=7,
                   label="not steady (mean of last iterations)")
        g.set_xlabel(xl); g.set_ylabel(yl); g.grid(alpha=.3)
    if has_tr:                                   # 2x3: Cl, polar, x_tr / Cm, L/D, (empty)
        g = ax[0, 2]
        xu = np.array([np.nan if r["x_tr_upper"] is None else r["x_tr_upper"] for r in res])
        xl = np.array([np.nan if r["x_tr_lower"] is None else r["x_tr_lower"] for r in res])
        g.plot(a, xu, "o-", color="C3", ms=5, label="this solver, upper")
        g.plot(a, xl, "s--", color="C3", ms=5, mfc="none", label="this solver, lower")
        g.set_xlabel("alpha (deg)"); g.set_ylabel("transition x/c"); g.set_ylim(-0.02, 1.05)
        g.set_title("Transition location"); g.grid(alpha=.3)
        ax[1, 2].axis("off")
    if any("Cl_inviscid" in r for r in res):
        P4[0].plot(a, [r.get("Cl_inviscid", np.nan) for r in res], "k:", marker=".",
                   label="inviscid (panel method)")
    if xf:
        for lab, d, st in xf:
            if not d:
                continue
            xa = np.array(sorted(d)); v = np.array([d[k] for k in xa])
            for g, (x, y) in zip(P4, ((xa, v[:, 0]), (v[:, 1], v[:, 0]), (xa, v[:, 2]),
                                      (xa, v[:, 0] / v[:, 1]))):
                g.plot(x, y, st, lw=1, label=lab)
            if has_tr and "free" in lab and v.shape[1] >= 5:
                ax[0, 2].plot(xa, v[:, 3], "k:", marker=".", label="XFOIL upper")
                ax[0, 2].plot(xa, v[:, 4], "k-.", marker=".", label="XFOIL lower")
    for g in ax.flat:
        if g.get_legend_handles_labels()[0]:
            g.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_distributions_multi(res, path, title):
    """Multi-element: Cp along every element surface (x/c of the assembly)
    and the element lift breakdown versus alpha."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = res[0]["element_names"]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2.2, 1]})
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(res):
        w = {k: np.asarray(v) for k, v in r["wall"].items()}
        c = cmap(i / max(len(res) - 1, 1))
        # points, not lines: coves, cusps and the cove step make any
        # upper/lower split by x zig-zag
        ax[0].plot(w["x"], w["Cp"], ".", ms=2.0, color=c, label=f"alpha {r['alpha']:g}")
    ax[0].invert_yaxis(); ax[0].grid(alpha=.3); ax[0].set_xlabel("x/c")
    ax[0].set_title("Cp on every element (wall faces)")
    ax[0].legend(fontsize=7, ncol=2)
    a = [r["alpha"] for r in res]
    for k, n in enumerate(names):
        ax[1].plot(a, [r["elements"][n]["Cl"] for r in res], "o-", ms=4, label=n)
    ax[1].plot(a, [r["Cl"] for r in res], "k-", lw=1.5, label="total")
    ax[1].set_xlabel("alpha (deg)"); ax[1].set_ylabel("Cl"); ax[1].grid(alpha=.3)
    ax[1].set_title("lift per element"); ax[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_multi_geometry(elements, info, path, panel_res=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = 2 if panel_res else 1
    fig, ax = plt.subplots(n, 1, figsize=(14, 4.5 * n), squeeze=False)
    g = ax[0, 0]
    for e in elements:
        L = np.vstack([e["loop"], e["loop"][:1]])
        g.plot(L[:, 0], L[:, 1], ".-", ms=2, lw=.8, label=f"{e['name']} ({len(e['loop'])} pts, "
               f"chord {e['chord']:.3f})")
    g.set_aspect("equal"); g.grid(alpha=.3); g.legend(fontsize=8)
    gaps = ", ".join(f"{k} {v:.4f}" for k, v in info["min_gaps"].items())
    g.set_title(f"{info['name']}: smallest gaps {gaps}")
    if panel_res:
        g = ax[1, 0]
        for e in panel_res["elements"]:
            g.plot(e["x"], e["Cp"], "-", lw=1, label=f"{e['name']}: Cl {e['Cl']:.3f}")
        g.invert_yaxis(); g.grid(alpha=.3); g.set_xlabel("x/c"); g.set_ylabel("Cp")
        g.legend(fontsize=8)
        g.set_title(f"inviscid panel method, alpha {panel_res['alpha']:g}: Cl {panel_res['Cl']:.3f}, "
                    f"Cm {panel_res['Cm']:+.3f}")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_distributions(res, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if res and "element_names" in res[0]:
        return plot_distributions_multi(res, path, title)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(res):
        w = {k: np.asarray(v) for k, v in r["wall"].items()}
        c = cmap(i / max(len(res) - 1, 1))
        for side, st in ((1, "-"), (-1, "--")):
            m = (w["side"] == side) & (w["x"] < 0.9999)
            ax[0].plot(w["x"][m], w["Cp"][m], st, color=c, lw=1,
                       label=f"alpha {r['alpha']:g}" if side == 1 else None)
            ax[1].plot(w["x"][m], w["Cf"][m], st, color=c, lw=1)
    ax[0].invert_yaxis(); ax[0].set_title("Cp (solid: upper, dashed: lower)")
    ax[1].set_title("Cf (negative = reversed flow)"); ax[1].axhline(0, color="k", lw=.6)
    lim = np.percentile(np.concatenate([np.asarray(r["wall"]["Cf"]) for r in res]), [1, 99])
    ax[1].set_ylim(min(lim[0], -0.002), lim[1] * 1.2)
    for g in ax:
        g.set_xlabel("x/c"); g.grid(alpha=.3)
    ax[0].legend(fontsize=7, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_geometry(loop, info, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [3, 1]})
    ax[0].plot(loop[:, 0], loop[:, 1], ".-", ms=2, lw=.8)
    ax[0].set_aspect("equal"); ax[0].grid(alpha=.3)
    ax[0].set_title(f"{info['name']}: {len(loop)} points, t/c = {info['max_thickness']:.4f} "
                    f"at x/c = {info['x_max_thickness']:.2f}, TE base {info['te_thickness']:.4f}")
    m = loop[:, 0] > 0.97
    ax[1].plot(loop[m, 0], loop[m, 1], ".-", ms=3); ax[1].set_aspect("equal")
    ax[1].set_title("trailing edge"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ----------------------------------------------------------------------
def cmd_show(a):
    if me.is_multi(a.airfoil):
        import panel
        elements, info = me.load_multi(a.airfoil)
        pr = panel.solve([e["outline"] for e in elements], a.alpha_show,
                         ref_chord=info["ref_chord"], names=[e["name"] for e in elements])
        out = Path(a.out or f"../cases/{_slug(a.airfoil)}")
        out.mkdir(parents=True, exist_ok=True)
        plot_multi_geometry(elements, info, out / "geometry.png", pr)
        for k, v in info.items():
            print(f"  {k:18s} {v}")
        print(f"  inviscid alpha {a.alpha_show:g}: Cl {pr['Cl']:.4f}  Cm {pr['Cm']:+.4f}  " +
              "  ".join(f"{e['name']} {e['Cl']:.4f}" for e in pr["elements"]))
        print(f"saved {out / 'geometry.png'}")
        return
    loop, info = ag.load_airfoil(a.airfoil, derotate=a.derotate)
    out = Path(a.out or f"../cases/{_slug(a.airfoil)}")
    out.mkdir(parents=True, exist_ok=True)
    plot_geometry(loop, info, out / "geometry.png")
    for k, v in info.items():
        print(f"  {k:18s} {v}")
    print(f"saved {out / 'geometry.png'}")


def _slug(spec):
    s = Path(spec).stem.lower().replace(" ", "_") if not "+" in str(spec) else \
        "_".join(Path(p.split(":")[0]).stem.lower() for p in str(spec).split("+"))
    return s


def cmd_polar(a):
    import polar as pl
    alphas = parse_alpha(a.alpha)
    out = Path(a.out or f"../cases/{_slug(a.airfoil)}_Re{a.re:.0e}".replace("+", ""))
    out.mkdir(parents=True, exist_ok=True)
    multi = me.is_multi(a.airfoil)
    if multi:
        import panel
        elements, info = me.load_multi(a.airfoil)
        plot_multi_geometry(elements, info, out / "geometry.png")
        if a.re < pl.LAMINAR_RE:
            raise SystemExit("multi-element airfoils need Re >= 1e5 (Spalart-Allmaras)")
        if a.xtr or a.transition == "free":
            print("[airfoil] multi-element: fully turbulent (e^N transition is single-element "
                  "only for now)")
        a.transition, a.xtr = "off", None
    else:
        loop, info = ag.load_airfoil(a.airfoil, derotate=a.derotate)
        plot_geometry(loop, info, out / "geometry.png")
    turb = a.re >= pl.LAMINAR_RE
    print(f"[airfoil] {info['name']}  Re {a.re:.3g}  {'SA' if turb else 'laminar'}"
          f"  alpha {alphas}  refine {a.refine}  jobs {a.jobs}")
    t0 = time.time()
    if a.xtr:
        xu, xl = (float(v) for v in a.xtr.split(","))
        trans = {"upper": xu, "lower": xl}
        tr_label = f"transition fixed at x/c {xu:g} (upper), {xl:g} (lower)"
    elif a.transition == "off":
        trans, tr_label = "off", "fully turbulent"
    else:
        trans, tr_label = "free", f"free transition (e^N, N_crit {a.ncrit:g})"
    pg = a.mach > 0 and a.prandtl_glauert
    comp = a.mach > 0 and not pg
    if comp and trans == "free" and a.mach >= 0.4:
        trans, tr_label = "off", "fully turbulent (no e^N transition at M >= 0.4)"
    res = pl.polar(a.airfoil, a.re, alphas, level=a.refine, jobs=a.jobs,
                   out=None if a.no_fields else out, max_iter=a.max_iter,
                   transition=trans, ncrit=a.ncrit, solver=a.solver,
                   mach=a.mach if comp else 0.0)
    for r in res:
        r["turbulent"] = turb
    if multi:                         # inviscid reference (same outlines, sharp TEs)
        for r in res:
            r["Cl_inviscid"] = panel.solve([e["outline"] for e in elements], r["alpha"],
                                           ref_chord=info["ref_chord"])["Cl"]
    if pg:
        apply_prandtl_glauert(res, a.mach)
    wall = time.time() - t0
    xf = []
    if a.xfoil:
        xf.append(("XFOIL, tripped at x/c 0.01 (valid at low alpha: once the stagnation point "
                   "moves aft the lower side stays laminar)",
                   run_xfoil(a.airfoil, a.re, alphas, xtr=(0.01, 0.01), derotate=a.derotate), "k--"))
        xf.append((f"XFOIL, free transition (N = {a.ncrit:g})",
                   run_xfoil(a.airfoil, a.re, alphas, derotate=a.derotate, ncrit=a.ncrit), "k:"))
    title = f"{info['name']}, Re = {a.re:.3g}" + (f", SA, {tr_label}" if turb else ", laminar") + \
        (f", M = {a.mach:g} (Prandtl-Glauert)" if pg else
         f", M = {a.mach:g} (compressible)" if comp else "")
    plot_polar(res, out / "polar.png", title, xf)
    plot_distributions(res, out / "distributions.png", title)
    cols = ["alpha", "Cl", "Cd", "Cd_p", "Cd_v", "Cm", "x_tr_upper", "x_tr_lower", "status",
            "iterations", "time_s", "yplus_max", "cells"]
    names = res[0].get("element_names", []) if multi else []
    ecols = [f"{k}_{n}" for n in names for k in ("Cl", "Cd", "Cm")] + \
        (["Cl_inviscid"] if multi else [])
    with open(out / "polar.csv", "w") as f:
        f.write(",".join(cols + ["L_D", "Cl_amp", "Cd_amp"] + ecols) + "\n")
        for r in res:
            ev = [f"{r['elements'][n][k]:.6f}" for n in names for k in ("Cl", "Cd", "Cm")] + \
                ([f"{r['Cl_inviscid']:.5f}"] if multi else [])
            f.write(",".join(str(round(r[c], 7)) if isinstance(r[c], float) else str(r[c]) for c in cols)
                    + f",{r['Cl'] / r['Cd']:.3f},{r.get('Cl_amp', 0):.5f},{r.get('Cd_amp', 0):.6f}"
                    + "".join("," + v for v in ev) + "\n")
    src = (f"elements: " + ", ".join(f"{e['name']} (chord {e['chord']:.4f}, {e['source']})"
                                     for e in info["elements"]) + "   smallest gaps: " +
           ", ".join(f"{k} {v:.4f}" for k, v in info["min_gaps"].items())) if multi else \
        (f"source: {info['source']}   t/c {info['max_thickness']:.4f}   "
         f"TE base {info['te_thickness']:.4f}   derotated: {info['derotated']}")
    lines = [f"{title}", src,
             f"mesh level {a.refine}, {res[0]['cells']} cells per angle; wall time {wall:.0f} s "
             f"({a.jobs} processes)", "",
             f"{'alpha':>6s} {'Cl':>8s} {'Cd':>9s} {'Cd_p':>9s} {'Cd_v':>9s} {'Cm':>8s} {'L/D':>7s} "
             f"{'xtr_up':>7s} {'xtr_lo':>7s}  status"]
    ft = lambda v: f"{v:7.3f}" if v is not None else f"{'-':>7s}"
    for r in res:
        lines.append(f"{r['alpha']:6.2f} {r['Cl']:8.4f} {r['Cd']:9.5f} {r['Cd_p']:9.5f} "
                     f"{r['Cd_v']:9.5f} {r['Cm']:+8.4f} {r['Cl'] / r['Cd']:7.1f} "
                     f"{ft(r.get('x_tr_upper'))} {ft(r.get('x_tr_lower'))}  {r['status']}"
                     + (" (x_tr averaged)" if (r.get("transition_converged_by") or "").startswith("averaged")
                        else "")
                     + (f" (Cl +-{r['Cl_amp']:.3f})" if "Cl_amp" in r else ""))
    if multi:
        lines += ["", "per element (Cl / Cd / Cm about the reference quarter chord):",
                  f"{'alpha':>6s} " + " ".join(f"{n:>26s}" for n in names) + f" {'Cl inviscid':>12s}"]
        for r in res:
            lines.append(f"{r['alpha']:6.2f} " + " ".join(
                f"{r['elements'][n]['Cl']:8.4f} {r['elements'][n]['Cd']:8.5f} "
                f"{r['elements'][n]['Cm']:+8.4f}" for n in names) + f" {r['Cl_inviscid']:12.4f}")
    if comp:
        lines += ["", f"compressible solver, M = {a.mach:g}; max local Mach per angle: " +
                  ", ".join(f"{r['alpha']:g}: {r.get('M_max', 0):.2f}" for r in res)]
    if pg:
        lines += ["", f"Cl, Cm, Cd_p and Cp corrected for M = {a.mach:g} with Prandtl-Glauert "
                  f"(factor {1 / math.sqrt(1 - a.mach**2):.4f}); uncorrected values in polar.json "
                  f"(*_incompressible)"]
    if xf:
        for lab, d, _ in xf:
            lines += ["", lab + ":"] + [f"{k:6.2f} {v[0]:8.4f} {v[1]:9.5f} {'':9s} {'':9s} {v[2]:+8.4f} "
                                         f"{v[0] / v[1]:7.1f} {v[3]:7.3f} {v[4]:7.3f}"
                                         for k, v in sorted(d.items())]
    warn = []
    if multi:
        warn.append("multi-element airfoil: fully turbulent SA (no transition model yet); "
                    "corners rounded (r <= 0.2 % element chord, coves 0.6 %) and sharp "
                    "trailing edges thickened to a 0.1 %-chord base for meshing")
    elif turb and trans == "off":
        warn.append("fully turbulent: at Re below ~1e7 real airfoils have laminar regions, so "
                    "Cd is typically overpredicted (use the default free transition)")
    if turb and trans == "free":
        warn.append(f"free transition: e^N envelope method with N_crit {a.ncrit:g} (clean, "
                    "low-turbulence flow; N_crit ~ -8.43 - 2.4 ln(Tu) for turbulence level Tu)")
    if any((r.get("transition_converged_by") or "").startswith("averaged") for r in res):
        warn.append("x_tr averaged: the transition point cycled around a leading-edge laminar "
                    "separation bubble; it was fixed at the mean of its last positions")
    if any(r["status"] == "transition not converged" for r in res):
        warn.append("at some angles the transition point was still moving when the iteration "
                    "limit was reached")
    if any(r["status"] != "converged" for r in res):
        warn.append("some angles did not reach a steady state (stall / separated flow): the "
                    "values are means over the last iterations; steady RANS is not reliable there")
    if warn:
        lines += ["", "notes:"] + [f"  - {w}" for w in warn]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    settings = {"mach": a.mach, "prandtl_glauert": bool(pg), "transition": tr_label,
                "refine": a.refine, "solver": a.solver, "alpha": a.alpha}
    (out / "polar.json").write_text(json.dumps({"info": info, "re": a.re, "settings": settings,
                                                "results": res,
                                                "xfoil": {lab: {str(k): v for k, v in d.items()}
                                                          for lab, d, _ in xf}}))
    print("\n".join(lines[4:]))
    print(f"[airfoil] -> {out}/  (polar.png, distributions.png, report.txt, polar.csv)")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("polar", help="angle-of-attack sweep")
    p.add_argument("airfoil")
    p.add_argument("--re", type=float, required=True)
    p.add_argument("--alpha", default="-4:16:2", help="start:stop:step or a,b,c (deg)")
    p.add_argument("--refine", type=float, default=1.0, help="mesh level (1 = validated)")
    p.add_argument("--jobs", type=int, default=2, help="parallel processes")
    p.add_argument("--max-iter", type=int, default=6000)
    p.add_argument("--out")
    p.add_argument("--xfoil", action="store_true", help="overlay XFOIL (tripped + free transition)")
    p.add_argument("--no-fields", action="store_true", help="skip per-angle field files")
    p.add_argument("--derotate", action="store_true", help="put the chord line on the x axis")
    p.add_argument("--transition", choices=["free", "off"], default="free",
                   help="free: e^N transition prediction (default); off: fully turbulent")
    p.add_argument("--solver", choices=["auto", "simple", "coupled"], default="auto",
                   help="SIMPLEC (default) or the coupled u-v-p solver (needs pypardiso; "
                        "auto = $CFD_SOLVER or simple)")
    p.add_argument("--ncrit", type=float, default=9.0,
                   help="critical amplification factor (9 = quiet wind tunnel)")
    p.add_argument("--xtr", help="fixed transition 'x_upper,x_lower' (e.g. trip strips)")
    p.add_argument("--mach", type=float, default=0.0,
                   help="free-stream Mach number: runs the compressible solver (shocks, "
                        "density and temperature changes)")
    p.add_argument("--prandtl-glauert", action="store_true",
                   help="with --mach: incompressible solution + Prandtl-Glauert correction "
                        "of Cl, Cm, Cd_p, Cp instead (old behaviour; valid below ~0.3)")
    s = sub.add_parser("show", help="load, check and plot the geometry")
    s.add_argument("airfoil")
    s.add_argument("--out")
    s.add_argument("--derotate", action="store_true")
    s.add_argument("--alpha", dest="alpha_show", type=float, default=0.0,
                   help="multi-element: angle for the inviscid panel-method Cp")
    argv = list(sys.argv[1:] if argv is None else argv)
    for i in range(len(argv) - 1):          # allow "--alpha -4:16:2" (negative start)
        if argv[i] == "--alpha" and argv[i + 1][:1] == "-":
            argv[i:i + 2] = [f"--alpha={argv[i + 1]}", ""]
    a = ap.parse_args([x for x in argv if x != ""])
    return cmd_polar(a) if a.cmd == "polar" else cmd_show(a)


if __name__ == "__main__":
    main()
