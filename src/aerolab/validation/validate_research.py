"""
Validation of the research tools: discrete adjoint and adjoint design,
goal-oriented error estimation, amplification-factor transport, swept-wing
transition, polar surrogate, field inversion / machine learning.

    python src/run.py validate_research adjoint     (~25 min)  -> validation/adjoint_report.md
    python src/run.py validate_research aft         (~6 min)   -> validation/aft_report.md
    python src/run.py validate_research swept       (~2 min)   -> validation/swept_transition_report.md
    python src/run.py validate_research surrogate   (~2 min)   -> validation/surrogate_report.md
    python src/run.py validate_research fiml                   -> validation/fiml_report.md
                                    (from cases/fiml_n0012: run.sh first, ~1 h)
    python src/run.py validate_research all
(bubble_ml.py writes validation/bubble_ml_report.md itself.)
"""

import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VAL = ROOT / "validation"
CASES = ROOT / "cases"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ----------------------------------------------------------------------
def adjoint_section(log=print):
    from aerolab.design import adjoint as A
    from aerolab.design import adjoint_error as E
    out = {}
    t0 = time.time()
    log("[adjoint] finite-difference check, NACA 0012, Re 3e6, alpha 2, fully turbulent")
    out["fd_turbulent"] = A.fd_check("naca0012", 3e6, 2.0, level=0.5, transition="off",
                                     shape_vars=(2, 6, 12), verbose=True)
    log("[adjoint] finite-difference check, NACA 2412, Re 3e6, alpha 4, fixed transition")
    out["fd_transition"] = A.fd_check("naca2412", 3e6, 4.0, level=0.5,
                                      transition={"upper": 0.25, "lower": 0.6},
                                      shape_vars=(4, 13), verbose=True)
    g = out["fd_turbulent"]["gradients"]
    out["timing"] = {k: g["timing"][k] for k in g["timing"]}
    out["size"] = {k: g[k] for k in ("cells", "unknowns", "jacobian_nnz", "colours")}
    log("[adjoint] adjoint-weighted residual on the nested mesh (with the fine solution)")
    e = E.estimate("naca0012", 3e6, 4.0, level=0.5, check=True, n_smooth=30, log=log)
    e0 = E.estimate("naca0012", 3e6, 4.0, level=0.5, check=False, n_smooth=0,
                    primal=e["_primal"], adjoint=e["_adjoint"], log=log)
    out["awr"] = {"smoothed_30": e["objectives"], "raw": e0["objectives"],
                  "cells": [e["cells_coarse"], e["cells_fine"]]}
    d = CASES / "design_adjoint_n0012" / "design.json"
    if d.exists():
        out["design"] = {k: v for k, v in json.loads(d.read_text()).items() if k != "history"}
        out["design"]["n_history"] = len(json.loads(d.read_text())["history"])
        if (d.parent / "design.png").exists():
            shutil.copy(d.parent / "design.png", VAL / "adjoint_design_naca0012.png")
    out["time_min"] = (time.time() - t0) / 60
    (VAL / "adjoint_results.json").write_text(json.dumps(out, indent=1, default=float))
    adjoint_report(out)
    return out


def adjoint_report(out=None):
    out = out or json.loads((VAL / "adjoint_results.json").read_text())
    plt = _plt()
    rows = out["fd_turbulent"]["rows"] + out["fd_transition"]["rows"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    lab = [f"d{r['objective']}/d{r['variable']}" for r in rows]
    adj = np.array([r["adjoint"] for r in rows])
    fd = np.array([r["fd"] for r in rows])
    sc = np.maximum(np.abs(fd), 1e-12)
    ax.bar(np.arange(len(rows)) - 0.2, adj / sc, 0.4, label="adjoint")
    ax.bar(np.arange(len(rows)) + 0.2, fd / sc, 0.4, label="finite differences of flow solutions")
    ax.set_xticks(np.arange(len(rows)), lab, rotation=60, fontsize=7)
    ax.set_ylabel("gradient / |FD gradient|"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Discrete adjoint vs central finite differences (each pair should be equal)")
    fig.tight_layout(); fig.savefig(VAL / "adjoint_fd_check.png", dpi=120); plt.close(fig)
    t = out["timing"]
    sz = out["size"]
    L = ["# Discrete adjoint (adjoint.py)", "",
         "Exact discrete adjoint of the incompressible RANS solver (SIMPLEC / coupled, "
         "Spalart-Allmaras, point-vortex far field). The converged state is the root of a "
         "residual R(X) that mirrors the solver term by term, with the quantities the solver "
         "lags between iterations (Rhie-Chow face fluxes, the skew-corrected Green-Gauss "
         "pressure gradient, outlet fluxes, the vortex circulation) as unknowns; dR/dX by "
         "coloured complex-step differences (exact to round-off), (dR/dX)^T psi = -dI/dX by "
         "sparse LU; design sensitivities (angle of attack, Reynolds number, CST shape "
         "coefficients through a mesh deformation that turns the boundary-layer columns with "
         "the wall) for one complex residual evaluation each.", "",
         "## Gradients vs finite differences of complete flow solutions", "",
         "| case | gradient | adjoint | finite differences (median of 3 steps) | FD spread | rel. difference |",
         "|---|---|---|---|---|---|"]
    for name, key in (("NACA 0012, α 2°, fully turbulent", "fd_turbulent"),
                      ("NACA 2412, α 4°, transition fixed 0.25 / 0.6", "fd_transition")):
        for r in out[key]["rows"]:
            L.append(f"| {name} | d{r['objective']}/d{r['variable']} | {r['adjoint']:+.5e} | "
                     f"{r['fd']:+.5e} | {r['fd_spread']:.1e} | {r['rel_diff']:.1e} |")
    L += ["", "Finite differences: central, at three step sizes (alpha 0.02 / 0.05 / 0.1 deg, "
          "CST 1e-4 / 2e-4 / 4e-4), each flow re-converged to residuals ~1e-10 on the same mesh "
          "topology (shape: the same deformation). The spread column is the scatter of the "
          "three: the discretisation has discrete switches (upwind direction, limiter, nearest "
          "wall face) that a perturbed solution may cross, which makes a single finite "
          "difference uncertain by up to a few per cent for some variables. Where the "
          "differences are clean, the adjoint agrees to 1e-5; where they scatter, the adjoint "
          "lies within their range. The adjoint itself is exact for the discrete equations: "
          "the coloured Jacobian equals a directional complex-step derivative to 1e-16 "
          "(tests/test_adjoint.py).", "",
          "![adjoint vs FD](adjoint_fd_check.png)", "",
          f"Cost (NACA 0012, level 0.5: {sz['cells']} cells, {sz['unknowns']} unknowns, "
          f"{sz['jacobian_nnz']} Jacobian entries, {sz['colours']} colours): flow "
          f"{t.get('primal', 0):.0f} s, Jacobian {t.get('jacobian', 0):.0f} s, LU "
          f"{t.get('factorisation', 0):.0f} s, three adjoint solves "
          f"{t.get('adjoint_solves', 0):.1f} s, 18 shape gradients {t.get('shape', 0):.0f} s. "
          "The LU dominates and grows faster than linearly (level 0.7: ~55 s). MKL PARDISO "
          "was tried and rejected: its static pivoting returned wrong solutions on this "
          "saddle-point system (pivots over 15 orders of magnitude) even with weighted "
          "matching.", ""]
    dz = out.get("design")
    if dz:
        sp, b = dz["start_point"], dz["best"]
        ver = dz.get("verification_level1", {})
        L += ["## Adjoint design: NACA 0012 -> minimum drag at Cl 0.5, Re 3e6", "",
              f"19 variables (all 18 CST coefficients + alpha), SLSQP inside sequential move "
              f"limits, every design a deformation of one mesh (level {dz['level']}); "
              f"{dz['n_history']} flow + adjoint solutions, {dz['time_min']:.0f} min.", "",
              f"| | Cd (design mesh) | Cd (re-meshed, level 1, Cl 0.5) |", "|---|---|---|",
              f"| start (CST fit of NACA 0012) | {sp['Cd']:.6f} | {ver.get('start', {}).get('Cd', float('nan')):.6f} |",
              f"| optimised | {b['Cd']:.6f} ({100 * (b['Cd'] / sp['Cd'] - 1):+.1f} %) | "
              f"{ver.get('best', {}).get('Cd', float('nan')):.6f} "
              f"({100 * (ver.get('best', {}).get('Cd', 1) / ver.get('start', {}).get('Cd', 1) - 1):+.1f} %) |",
              "", "The design converged after ~25 flow + adjoint solutions (see the history); "
              "the trial steps marked x came afterwards, on the plateau: the optimiser sharpens "
              "the nose (first CST coefficient: lower +0.025, upper -0.013) and on those shapes "
              "the SIMPLEC solver at mesh level 0.5 no longer converges from warm or cold starts, "
              "while the deformed mesh stays benign (near-wall lean < 0.7 deg, cell areas > 85 % "
              "of the original). Going further needs a more robust flow solver on sharp noses "
              "(or a nose-radius constraint), not a different gradient.",
              "", "Fully turbulent: the drag is mostly friction and the thickness is held "
              "(t/c >= 0.119), so the possible gain is small; half of the design-mesh gain "
              "survives re-meshing on the validated mesh. Re-meshing every design instead of "
              "deforming one mesh made Cd jump by ~3 % between neighbouring designs, which "
              "stalls a gradient method; plain inverse-distance deformation leaned the "
              "boundary-layer columns at the nose by 1.7° and the explicit non-orthogonal "
              "correction then diverged - the deformation now turns the columns with the wall.",
              "", "![design](adjoint_design_naca0012.png)", ""]
    a = out["awr"]
    L += ["## Goal-oriented error estimate (adjoint_error.py): mostly a negative result", "",
          f"Adjoint-weighted residual on the nested fine mesh (every cell split in four, "
          f"{a['cells'][0]} -> {a['cells'][1]} cells), NACA 0012, α 4°:", "",
          "| | coarse | fine (solved) | estimate, raw | estimate, 30 smoothing iterations |",
          "|---|---|---|---|---|"]
    for k in ("Cl", "Cd", "Cm"):
        s30, raw = a["smoothed_30"][k], a["raw"][k]
        L.append(f"| {k} | {s30['coarse']:.5f} | {s30.get('fine_solved', float('nan')):.5f} | "
                 f"{raw['estimate_fine']:.5f} | {s30['estimate_fine']:.5f} |")
    eff = {k: a["smoothed_30"][k].get("effectivity") for k in ("Cl", "Cd", "Cm")}
    L += ["", "Effectivity (estimated / actual error of the coarse value; 1 = exact): "
          + ", ".join(f"{k} {v:.2f}" for k, v in eff.items() if v is not None) + ". "
          "Without smoothing the correction is useless (the reconstruction leaves child-to-child "
          "residual jumps in the boundary layer that the coarse adjoint weights arbitrarily). "
          "After 30 fine-mesh iterations it gets the direction of the lift and moment errors "
          "right but overshoots about twice, and it does not estimate the drag error. Even the "
          "exact fine-mesh adjoint (240 000 unknowns) did not make the raw correction work "
          "(Cl 0.442 -> 0.453 against a fine value of 0.434): the reconstructed state is too far "
          "from the fine solution in the nonlinear sense (limiter and SA switches) and the "
          "adjoint fields reach 1e5-1e7 at the trailing-edge and wall singularities. This is "
          "in line with the experience that the adjoint-weighted residual works far less well "
          "for second-order finite-volume RANS than for DG. The cell-wise |psi . R| is kept "
          "as an adaptation indicator (adjoint_error.adapt; the mesher's triangle size field "
          "takes a multiplier, hybrid_mesher.build_mesh(size_scale=...)), not as a corrected "
          "value.", ""]
    (VAL / "adjoint_report.md").write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------------
def aft_section(log=print):
    from aerolab.airfoil import polar as pl
    from aerolab.physics import transition as T
    from aerolab.validation import validate_flatplate as vf
    from aerolab.physics import aft as AF
    from aerolab.core.simple_solver import SimpleSolver
    from aerolab.core.turbulence import wall_distance
    out = {}
    m = vf.plate_mesh(0.5)
    s = SimpleSolver(m, vf.BCS, 1.0 / 5e6, U0=(1.0, 0.0))
    s.solve(tol=1e-8, max_iter=8000, verbose=False)
    a = AF.AFT.__new__(AF.AFT)
    a.s, a.re, a.ncrit, a.closure = s, 5e6, 9.0, "thwaites"
    w = np.where(s.b_wall)[0]
    a._wall_faces = w
    a.d, a._nearest = wall_distance(m, w, return_index=True)
    a._before_te = m.xc[:, 0] < 1.99
    a.d_max = 0.1
    n = a.solve_n()
    st = a.state
    xs = np.linspace(1e-6, 2, 40000)
    th = 0.664 * np.sqrt(xs / 5e6)
    Ne, _ = T.integrate_N(xs, np.full_like(xs, 2.591), 5e6 * th, th)
    thx, Hx, rx, _ = T.laminar_bl(xs, np.ones_like(xs), 5e6)
    Nx, _ = T.integrate_N(xs, Hx, rx, thx)
    plate = []
    for xq in (0.2, 0.4, 0.6, 0.8, 1.0):
        xc = m.xc[np.argmin(np.abs(m.xc[:, 0] - xq)), 0]
        col = np.where((np.abs(m.xc[:, 0] - xc) < 1e-9) & st["inside"])[0]
        inner = col[st["phi"][col] < 0.6]
        plate.append({"x": float(xc), "n": float(n[col].max()), "N_blasius": float(np.interp(xc, xs, Ne)),
                      "N_xfoil_closure": float(np.interp(xc, xs, Nx)),
                      "Re_theta": float(np.median(st["Re_theta"][inner])),
                      "Re_theta_blasius": 0.664 * math.sqrt(5e6 * xc)})
    out["plate"] = plate
    log("[aft] NACA 0012, Re 3e6, alpha 0..6 with the transport model")
    res = pl.polar("naca0012", 3e6, [0.0, 2.0, 4.0, 6.0], level=1.0, jobs=4, transition="aft",
                   log=False)
    out["naca0012"] = [{k: r[k] for k in ("alpha", "Cl", "Cd", "x_tr_upper", "x_tr_lower", "status")}
                       for r in res]
    (VAL / "aft_results.json").write_text(json.dumps(out, indent=1))
    aft_report(out)
    return out


def aft_report(out=None):
    out = out or json.loads((VAL / "aft_results.json").read_text())
    ref = json.loads((VAL / "transition_tr_n0012_re3e6.json").read_text())
    en = {r["alpha"]: r for r in ref["results"]}
    xf = next(v for k, v in ref["xfoil"].items() if "free" in k)
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    A_ = np.array([r["alpha"] for r in out["naca0012"]])
    for side, k, mk in (("upper", "x_tr_upper", "o"), ("lower", "x_tr_lower", "s")):
        ax[0].plot(A_, [r[k] for r in out["naca0012"]], mk + "-", color="C3", label=f"transport, {side}")
        ax[0].plot(A_, [en[a][k] for a in A_], mk + "--", color="C0", mfc="none", label=f"e^N (marching), {side}")
        ax[0].plot(A_, [xf[f"{a:.1f}"][3 if side == "upper" else 4] for a in A_], mk + ":", color="k",
                   mfc="none", label=f"XFOIL, {side}")
    ax[0].set_xlabel("alpha (deg)"); ax[0].set_ylabel("transition x/c"); ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=7); ax[0].set_title("NACA 0012, Re 3e6, N_crit 9")
    p = out["plate"]
    ax[1].plot([q["x"] for q in p], [q["n"] for q in p], "o-", color="C3", label="transport (max across the layer)")
    ax[1].plot([q["x"] for q in p], [q["N_blasius"] for q in p], "k-", label="envelope, exact Blasius")
    ax[1].plot([q["x"] for q in p], [q["N_xfoil_closure"] for q in p], "k--", label="envelope, XFOIL closure")
    ax[1].axhline(9, color="gray", lw=.6)
    ax[1].set_xlabel("x (Re 5e6 per unit length)"); ax[1].set_ylabel("N"); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3); ax[1].set_title("Laminar flat plate")
    fig.tight_layout(); fig.savefig(VAL / "aft_validation.png", dpi=120); plt.close(fig)
    L = ["# Amplification-factor transport (aft.py)", "",
         "The e^N envelope method as two local transport equations on the RANS mesh: "
         "D n/Dt = |u| r(H, Re_theta, theta) and D I/Dt = |u| U_e^5 (Thwaites' integral, which "
         "carries the boundary-layer history), U_e from the local static pressure, H from "
         "Thwaites' lambda (Falkner-Skan anchored at Blasius' 2.591). No surface marching, no "
         "boundary-layer integrals. Run with `airfoil.py polar ... --transition aft`.", "",
         "## Laminar flat plate (Re 5e6 per unit length)", "",
         "| x | n (transport) | N exact Blasius | N XFOIL closure | Re_theta (transport) | Re_theta Blasius |",
         "|---|---|---|---|---|---|"]
    for q in p:
        L.append(f"| {q['x']:.3f} | {q['n']:.2f} | {q['N_blasius']:.2f} | {q['N_xfoil_closure']:.2f} | "
                 f"{q['Re_theta']:.0f} | {q['Re_theta_blasius']:.0f} |")
    L += ["", "The transported n follows the exact Blasius envelope (the classical e^9 result, "
          "Re_x ~ 3e6); XFOIL's laminar closure has H = 2.568 on a flat plate and runs later - "
          "the criterion is that sensitive to H (2.591 vs 2.61 alone moves e^9 by 18 %).", "",
          "## NACA 0012, Re 3e6", "",
          "| α | x_tr upper: transport / e^N / XFOIL | x_tr lower: transport / e^N / XFOIL | Cd: transport / e^N / XFOIL |",
          "|---|---|---|---|"]
    for r in out["naca0012"]:
        a = r["alpha"]
        x = xf[f"{a:.1f}"]
        L.append(f"| {a:g} | {r['x_tr_upper']:.3f} / {en[a]['x_tr_upper']:.3f} / {x[3]:.3f} | "
                 f"{r['x_tr_lower']:.3f} / {en[a]['x_tr_lower']:.3f} / {x[4]:.3f} | "
                 f"{r['Cd']:.5f} / {en[a]['Cd']:.5f} / {x[1]:.4f} |")
    L += ["", "![AFT validation](aft_validation.png)", "",
          "Development notes (kept because they matter for any local transition model): a "
          "purely local closure (each cell's Falkner-Skan state from u/U_e, Re_V and the local "
          "pressure gradient) recovers the Blasius layer from single cells but has no history "
          "and put transition 0.05-0.07 c early behind the suction peak; transporting "
          "Thwaites' integral fixed that. The boundary-layer detector needs three conditions: "
          "velocity deficit, rotation (d|Omega|/|u| > 0.2: the decelerating stagnation flow "
          "ahead of the nose otherwise looked like a separating layer) and a wall-distance "
          "bound (far-field cells with a numerical total-pressure loss otherwise tripped the "
          "leading edge). Molecular diffusion of n and I mixed the inner layer with "
          "entrained outer fluid (n -11 %); both are path integrals and are now purely "
          "advected. Laminar separation bubbles are outside the model (transition.py).", ""]
    (VAL / "aft_report.md").write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------------
def swept_section(log=print):
    from aerolab.physics import swept_transition as st
    from aerolab.aircraft import wing as W
    rows = []
    for sw in (0, 10, 20, 25, 30, 35, 40, 45, 50):
        r = st.analyse("naca0012", float(sw), 1e7, 2.0)
        rows.append({"sweep": sw, "Rbar": r["attachment_line"]["Rbar"],
                     "x_up": r["upper"]["x_tr"], "m_up": r["upper"]["mechanism"],
                     "x_lo": r["lower"]["x_tr"], "m_lo": r["lower"]["mechanism"],
                     "chi_up": r["upper"]["chi_max"]})
        log(f"  sweep {sw}: upper {rows[-1]['x_up']:.3f} ({rows[-1]['m_up']})")
    w = W.Wing(30, 6.0, 1.8, 30.0, 3, -2, 1)
    sol = W.solve(w, 3.0)
    t = st.wing_transition(w, sol, 1e7 / w.mac * w.c, "naca0012", n_eval=5)
    out = {"sweep_study": rows, "wing": t}
    (VAL / "swept_transition_results.json").write_text(json.dumps(out, indent=1))
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    s_ = [r["sweep"] for r in rows]
    ax[0].plot(s_, [r["x_up"] for r in rows], "o-", label="upper")
    ax[0].plot(s_, [r["x_lo"] for r in rows], "s--", label="lower")
    for r in rows:
        ax[0].annotate("CF" if "crossflow" in r["m_up"] else "TS", (r["sweep"], r["x_up"]),
                       fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[0].set_xlabel("leading-edge sweep (deg)"); ax[0].set_ylabel("transition x/c")
    ax[0].set_title("NACA 0012, Re 1e7, alpha 2 (TS: Tollmien-Schlichting, CF: crossflow)", fontsize=9)
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(s_, [r["Rbar"] for r in rows], "o-")
    ax[1].axhline(st.RBAR_CONTAMINATION, color="C1", ls="--", label="contamination (245)")
    ax[1].axhline(st.RBAR_INSTABILITY, color="C3", ls="--", label="instability (580)")
    ax[1].set_xlabel("leading-edge sweep (deg)"); ax[1].set_ylabel("attachment-line Rbar")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(VAL / "swept_transition.png", dpi=120); plt.close(fig)
    L = ["# Swept-wing transition (swept_transition.py, wing.py --transition-3d)", "",
         "Infinite swept wing, independence principle: attachment line (Poll's Rbar; "
         "contamination 245, instability 580), crossflow (Falkner-Skan-Cooke profile matched "
         "to the integral layer's shape factor; chi = w_cf,max delta_10 / nu, transition at "
         "200), Tollmien-Schlichting (e^N on the chordwise layer, N_crit from Tu by Mack). "
         "Checks (tests/test_swept_transition.py): Blasius / Hiemenz similarity values, "
         "no crossflow at zero pressure gradient (g = f' exactly), Rbar of a swept circular "
         "cylinder against the exact formula, the unswept wing reproduces the 2D e^N result.",
         "", "## Sweep study: NACA 0012, Re 1e7 (normal chord), alpha 2°", "",
         "| sweep | Rbar | upper | mechanism | lower | mechanism | max chi (upper) |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['sweep']} | {r['Rbar']:.0f} | {r['x_up']:.3f} | {r['m_up']} | {r['x_lo']:.3f} | "
                 f"{r['m_lo']} | {r['chi_up']:.0f} |")
    L += ["", "![swept](swept_transition.png)", "",
          "## Along a wing: 30° quarter-chord sweep, taper 0.3, Re_MAC 1e7, alpha 3°", "",
          "```", st.wing_table(t), "```", "",
          "The root attachment line sits at Rbar ~ 245, the contamination limit: with "
          "turbulence from a fuselage junction (`--contaminated`) the whole inboard wing "
          "is turbulent from the leading edge - the reason swept laminar wings need a "
          "Gaster bump or a Krueger flap at the root.", ""]
    (VAL / "swept_transition_report.md").write_text("\n".join(L) + "\n")
    return out


# ----------------------------------------------------------------------
def surrogate_section(log=print):
    from aerolab.design import surrogate as SG
    rows, sources = SG.collect()
    loo = SG.validate(log=log)
    sg = SG.Surrogate.train(rows, sources)
    cal = sg.calibrate(rows)
    sg.save(SG.MODEL)                       # the shipped model (data/)
    out = {"n": len(rows), "sources": sources, "loo": loo, "calibration": cal}
    (VAL / "surrogate_results.json").write_text(json.dumps(out, indent=1))
    L = ["# Gaussian-process polar surrogate (surrogate.py)", "",
         f"{len(rows)} converged points from {len(sources)} CFD polars; inputs: alpha, log Re, "
         "t/c and its position, max camber and its position, leading-edge parameter, flap "
         "chord x deflection, flap deflection, free / fully turbulent. One GP each for Cl, "
         "log Cd, Cm (ARD squared exponential + noise, maximum marginal likelihood).", "",
         "## Leave one polar out (uncalibrated sd)", "",
         "| left out | points (α max) | rms Cl, all / α ≤ 12° | rms Cd, all / α ≤ 12° | Cl within 2 sd | Cd within 2 sd |",
         "|---|---|---|---|---|---|"]
    for e in loo:
        L.append(f"| {e['polar']} | {e['n']} ({e['alpha_max']:g}°) | {e['rms_Cl']:.3f} / "
                 f"{e['rms_Cl_prestall']:.3f} | {100 * e['rms_Cd_rel']:.1f} % / "
                 f"{100 * e['rms_Cd_rel_prestall']:.1f} % | "
                 f"{100 * e['cover_Cl_2sd']:.0f} % | {100 * e['cover_Cd_2sd']:.0f} % |")
    L += ["", f"Calibration (sd multipliers so that 95 % of the left-out points fall in +-2 sd): "
          + ", ".join(f"{k} x{v:.2f}" for k, v in cal.items()) + ".", "",
          "Reading: below stall, for an airfoil family the library contains, the surrogate is "
          "useful (see the α ≤ 12° columns); past stall (the wing polars reach 20°+, where "
          "few polars have data), for a geometry class it has seen once (flapped sections) "
          "or for the only fully turbulent polar it is poor, and the uncalibrated GP is "
          "over-confident there - hence the calibration. The surrogate "
          "improves with every polar added to the library (`surrogate.py train`), and "
          "`surrogate.py suggest` names the angle where one more RANS run would teach it "
          "the most.", ""]
    (VAL / "surrogate_report.md").write_text("\n".join(L) + "\n")
    return out


# ----------------------------------------------------------------------
def fiml_section(log=print):
    d = CASES / "fiml_n0012"
    inv = []
    for f in sorted(d.glob("a*.npz")):
        z = np.load(f)
        h = json.loads(str(z["history"]))
        b, xc = z["beta"], z["xc"]
        dev = np.abs(b - 1) > 0.1
        bands = {}
        for lo, hi in ((0.0, 0.3), (0.3, 0.7), (0.7, 1.05)):
            m = dev & (xc[:, 0] >= lo) & (xc[:, 0] < hi)
            bands[f"{lo:g}-{hi:g}"] = float(b[m].mean()) if m.any() else None
        inv.append({"alpha": float(z["alpha"]), "Cl_target": float(z["cl_target"]),
                    "Cl_SA": h[0]["Cl"], "Cl_inverted": float(z["Cl"]), "iterations": len(h),
                    "beta_min": float(b.min()), "beta_max": float(b.max()),
                    "cells_corrected": int(dev.sum()), "cells": len(b),
                    "upper_share": float((dev & (xc[:, 1] > 0)).sum() / max(dev.sum(), 1)),
                    "mean_beta_x_bands": bands})
    pred = json.loads((d / "predictions.json").read_text()) if (d / "predictions.json").exists() else []
    out = {"inversions": inv, "predictions": pred}
    (VAL / "fiml_results.json").write_text(json.dumps(out, indent=1))
    plt = _plt()
    from aerolab.ml import fiml
    g = fiml.gregory_cl()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(g[:, 0], g[:, 1], "ks", mfc="none", label="Gregory & O'Reilly (exp.)")
    if pred:
        a = [p["alpha"] for p in pred]
        ax.plot(a, [p["Cl_SA"] for p in pred], "o--", label="SA")
        ax.plot(a, [p["Cl_FIML"] for p in pred], "o-", label="SA + learned beta (FIML)")
    for i in inv:
        ax.plot(i["alpha"], i["Cl_inverted"], "r*", ms=12, label="field inversion (training)"
                if i is inv[0] else None)
    ax.set_xlabel("alpha (deg)"); ax.set_ylabel("Cl"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("NACA 0012, Re 3e6: field inversion and machine learning", fontsize=9)
    fig.tight_layout(); fig.savefig(VAL / "fiml_naca0012.png", dpi=120); plt.close(fig)
    L = ["# Field inversion and machine learning (fiml.py)", "",
         "A multiplier beta(x) on the SA production term, found per training case by "
         "adjoint-based optimisation (one adjoint solve per iteration gives the gradient with "
         "respect to every cell's beta), then learned as a function of six local flow "
         "features by a small neural network and used inside SA for cases not seen in "
         "training. NACA 0012, Re 3e6 (mesh level 0.5), lift of Gregory & O'Reilly.", "",
         "## Inversions (training)", "",
         "| alpha | Cl SA | Cl target | Cl inverted | flow + adjoint solutions | beta range | cells with abs(beta - 1) > 0.1 |",
         "|---|---|---|---|---|---|---|"]
    for i in inv:
        L.append(f"| {i['alpha']:g} | {i['Cl_SA']:.4f} | {i['Cl_target']:.4f} | {i['Cl_inverted']:.4f} | "
                 f"{i['iterations']} | {i['beta_min']:.3f} - {i['beta_max']:.3f} | "
                 f"{i['cells_corrected']} of {i['cells']} ({100 * i['upper_share']:.0f} % upper surface) |")
    if inv:
        bd = inv[0]["mean_beta_x_bands"]
        L += ["", "Where the inversion acts (alpha " + f"{inv[0]['alpha']:g}): almost only in the "
              "upper-surface boundary layer - mean beta of the corrected cells " +
              ", ".join(f"x/c {k}: {v:.2f}" for k, v in bd.items() if v is not None) +
              ". Less production near the nose, more over the mid and aft upper surface: "
              "more mixing in the adverse pressure gradient, later trailing-edge separation, "
              "more lift near stall - the known deficiency of SA there, found from the lift "
              "data alone."]
    L += ["", "## Prediction with the learned model", "",
          "| alpha | in training | Cl SA | Cl SA + learned beta | Cl experiment | error SA | error learned |",
          "|---|---|---|---|---|---|---|"]
    tr = {i["alpha"] for i in inv}
    for p in pred:
        e0, e1 = p["Cl_SA"] / p["Cl_exp"] - 1, p["Cl_FIML"] / p["Cl_exp"] - 1
        L.append(f"| {p['alpha']:g} | {'yes' if p['alpha'] in tr else 'no'} | {p['Cl_SA']:.4f} | "
                 f"{p['Cl_FIML']:.4f} | {p['Cl_exp']:.4f} | {100 * e0:+.1f} % | {100 * e1:+.1f} % |")
    L += ["", "The model trained on 13° and 15° improves the lift at every angle, including the "
          "unseen 10° and 14° and the 16° beyond the training range, where baseline SA stalls "
          "early (-18 %) and the augmented model keeps the flow attached (-4 %). The training "
          "angles are not reproduced exactly (the network is a smooth fit of two beta fields)."]
    L += ["", "![FIML](fiml_naca0012.png)", "",
          "Limits: two training angles of one airfoil at one Reynolds number, a coarse mesh "
          "(the inversion also absorbs discretisation error), lift only. It demonstrates the "
          "pipeline; a transferable correction needs inversions over several airfoils and "
          "flow conditions (and pressure data: Inversion(cp=...) supports it).", ""]
    (VAL / "fiml_report.md").write_text("\n".join(L) + "\n")
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    what = argv[0] if argv else "all"
    secs = {"adjoint": adjoint_section, "aft": aft_section, "swept": swept_section,
            "surrogate": surrogate_section, "fiml": fiml_section}
    for k, f in secs.items():
        if what in (k, "all"):
            f()
    if what == "report-adjoint":
        adjoint_report()
    if what == "report-aft":
        aft_report()


if __name__ == "__main__":
    main()
