"""
Phase 5 validation 2: transonic airfoils with the compressible solver.

  A) NACA 0012 (TMR sharp-TE), inviscid (Euler), alpha 1.25 deg:
     M 0.5 (subsonic: drag must vanish) and M 0.8 (shocks on both sides).
     References: AGARD AR-211 (1985) Euler codes and Vassberg & Jameson,
     "In pursuit of grid convergence for two-dimensional Euler solutions",
     J. Aircraft 47 (2010): M 0.8 -> Cl ~0.35, Cd ~0.022, Cm ~ -0.04.
  B) RAE 2822 case 9 (Cook, McDonald & Firmin, AGARD AR-138, 1979),
     RANS (Spalart-Allmaras, fully turbulent), corrected wind-tunnel
     conditions M 0.734, alpha 2.79 deg, Re 6.5e6 (as in the NPARC / HiOCFD
     workshop test cases). Experiment: Cl 0.803, Cd 0.0168, Cm -0.099 and
     the surface pressures (SU2 TestCases, rans/rae2822).
     Mesh levels 0.7 / 1 / 1.4 for the grid dependence.

    python src/run.py validate_transonic [--quick]    -> ../validation/transonic_*.png, transonic_report.txt
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from aerolab.airfoil import airfoil_geom as ag
from aerolab.mesh import hybrid_mesher as hm
from aerolab.validation import validate_naca0012_sa as v
from aerolab.core.compressible import CompressibleSolver, CompressibleVortexFarfield
from aerolab.core.fvmesh import FVMesh
from aerolab.core.postproc import wall_data

OUT = Path(__file__).resolve().parents[3] / "validation"
REF = OUT / "reference"
RAE_DAT = REF / "rae2822.dat"
RAE_EXP = REF / "rae2822_case9_cp_experiment.plt"
RAE9 = {"mach": 0.734, "alpha": 2.79, "re": 6.5e6,
        "exp": {"Cl": 0.803, "Cd": 0.0168, "Cm": -0.099}}


def euler_mesh(wall, alpha, level=1.0):
    """inviscid mesh: a few thin layers (first 0.002 c) instead of the y+ ~ 1
    boundary layer (the 1e-6 c cells of a RANS mesh are useless without
    viscosity and make the inviscid pressure-velocity coupling stiff)"""
    r = level
    return hm.build_mesh(wall=wall, h1=0.002 / r, growth=1.2, n_layers=int(round(8 * r)),
                         bl_ar_cap=1.0, farfield_radius=100.0, n_farfield=int(round(48 * r)),
                         h_near=0.01 / r, grad=0.12 / r, wake_length=3.0, wake_h=0.02 / r,
                         wake_grad=0.08 / r, alpha_deg=alpha, verbose=False)


def fv_case(mesh, alpha):
    U = (math.cos(math.radians(alpha)), math.sin(math.radians(alpha)))
    P, ff = mesh["points"], mesh["patches"]["farfield"]
    t = P[ff[:, 1]] - P[ff[:, 0]]
    inflow = t[:, 1] * U[0] - t[:, 0] * U[1] < 0
    fvm = FVMesh(P, mesh["cells"], {"wall": mesh["patches"]["airfoil"], "inlet": ff[inflow],
                                    "outlet": ff[~inflow]})
    bcs = {"wall": {"type": "wall"}, "inlet": {"type": "inlet", "U": U},
           "outlet": {"type": "outlet"}}
    return fvm, bcs, U


def solve(fvm, bcs, U, alpha, mach, re=None, max_iter=4000, tol=1e-8, force_window=300,
          force_tol=2e-5, log=None):
    """run to residual tol, or until Cl, Cd, Cm change < force_tol over
    force_window iterations with residuals < 1e-4 (the limiter at shocks stalls them near 2e-5)"""
    t0 = time.time()
    inviscid = re is None
    s = CompressibleSolver(fvm, bcs, 0.0 if inviscid else 1.0 / re, mach, U0=U,
                           inviscid=inviscid, turbulence=None if inviscid else "sa")
    vf = CompressibleVortexFarfield(s, alpha, every=20)
    hist = []

    def cb(it, r):
        vf.update(it)
        if it % 10 == 0:
            c = wall_data(s, "wall", U)[1]
            hist.append((c["Cl"], c["Cd"], c["Cm"]))
            if log and it % 200 == 0:
                log(f"    it {it:5d} res {max(r.values()):.1e} Cl {c['Cl']:.4f} Cd {c['Cd']:.5f}"
                    f" ({time.time() - t0:.0f}s)")
            n = force_window // 10
            if len(hist) > n and max(r.values()) < 1e-4:
                h = np.array(hist[-n:])
                if np.all(np.ptp(h, axis=0) / np.maximum(np.abs(h[-1]), [0.1, 0.01, 0.01])
                          < force_tol):
                    return True
        return None
    it = s.solve(tol=tol, max_iter=max_iter, verbose=False, callback=cb)
    w, c = wall_data(s, "wall", U)
    res = {"mach": mach, "alpha": alpha, "re": re, "cells": int(fvm.ncells),
           "iterations": int(it), "converged": bool(s.converged),
           "time_s": round(time.time() - t0, 1), "Cl": float(c["Cl"]), "Cd": float(c["Cd"]),
           "Cm": float(c["Cm"]), "Cd_p": float(c["Cd_pressure"]),
           "Cd_v": float(c["Cd_viscous"]), "M_max": float(s.mach_field().max()),
           "supersonic_cells": s.stats["supersonic_cells"]}
    for side, nm in ((1, "upper"), (-1, "lower")):         # shock: steepest Cp rise
        k = w["side"] == side
        o = np.argsort(w["x"][k])
        x, cp = w["x"][k][o], w["Cp"][k][o]
        sel = (x > 0.1) & (x < 0.95)
        d = np.diff(cp[sel]) / np.diff(x[sel])
        j = int(np.argmax(d))
        res[f"x_shock_{nm}"] = float(0.5 * (x[sel][j] + x[sel][j + 1])) if d[j] > 5 else None
    return res, s, w


def main(quick=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    OUT.mkdir(exist_ok=True)
    lines = ["Transonic airfoil validation (compressible pressure-based solver, Phase 5)", ""]
    log = lambda s: (print(s, flush=True), lines.append(s))
    out = {}

    # ---- A) NACA 0012 Euler ---------------------------------------------------
    wall = v.tmr0012(200)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for j, M in enumerate((0.5, 0.8)):
        mesh = euler_mesh(wall, 1.25)
        fvm, bcs, U = fv_case(mesh, 1.25)
        log(f"NACA 0012 Euler M {M} alpha 1.25 ({fvm.ncells} cells)")
        r, s, w = solve(fvm, bcs, U, 1.25, M, log=log)
        out[f"naca0012_euler_M{M}"] = r
        log("  " + json.dumps(r))
        ax = axs[j]
        for sd, col in ((1, "b"), (-1, "r")):
            k = w["side"] == sd
            o = np.argsort(w["x"][k])
            ax.plot(w["x"][k][o], w["Cp"][k][o], col + "-", lw=1.2,
                    label="upper" if sd > 0 else "lower")
        cps = 2 / (1.4 * M**2) * (((1 + 0.2 * M**2) / 1.2) ** 3.5 - 1)
        ax.axhline(cps, color="gray", ls="--", lw=0.8, label="Cp* (sonic)")
        ax.invert_yaxis(); ax.grid(alpha=0.3); ax.legend()
        ax.set_title(f"NACA 0012 Euler M {M}, a 1.25: Cl {r['Cl']:.4f} Cd {r['Cd']:.5f} "
                     f"Cm {r['Cm']:.4f}", fontsize=9)
        ax.set_xlabel("x/c"); ax.set_ylabel("Cp")
    fig.tight_layout(); fig.savefig(OUT / "transonic_naca0012_euler_cp.png", dpi=130)
    plt.close(fig)

    # ---- B) RAE 2822 case 9 RANS ----------------------------------------------
    ex = np.loadtxt(RAE_EXP, skiprows=2, delimiter=",")
    levels = (1.0,) if quick else (0.7, 1.0, 1.4)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    best = None
    for lev in levels:
        loop, _ = ag.load_airfoil(str(RAE_DAT), n_side=int(round(160 * lev)))
        mesh = v.airfoil_mesh(lev, RAE9["alpha"], wall=loop, re=RAE9["re"])
        fvm, bcs, U = fv_case(mesh, RAE9["alpha"])
        log(f"RAE 2822 case 9 RANS level {lev} ({fvm.ncells} cells)")
        r, s, w = solve(fvm, bcs, U, RAE9["alpha"], RAE9["mach"], re=RAE9["re"], log=log)
        r["level"] = lev
        out[f"rae2822_case9_level{lev}"] = r
        log("  " + json.dumps(r))
        for sd, col in ((1, "b"), (-1, "r")):
            k = w["side"] == sd
            o = np.argsort(w["x"][k])
            ax.plot(w["x"][k][o], w["Cp"][k][o], col + "-",
                    lw=1.6 if lev == 1.0 else 0.8, alpha=1.0 if lev == 1.0 else 0.5,
                    label=(f"level {lev} " + ("upper" if sd > 0 else "lower")))
        if lev == 1.0:
            best = (r, s, fvm)
    ax.plot(ex[:, 0], ex[:, 1], "ko", ms=4, mfc="none", label="experiment (Cook et al.)")
    r = best[0]
    ax.set_title(f"RAE 2822 case 9 (M 0.734, a 2.79, Re 6.5e6), SA: Cl {r['Cl']:.3f} "
                 f"(exp 0.803), Cd {r['Cd']:.4f} (0.0168), Cm {r['Cm']:.3f} (-0.099)",
                 fontsize=9)
    ax.invert_yaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_xlabel("x/c"); ax.set_ylabel("Cp")
    fig.tight_layout(); fig.savefig(OUT / "transonic_rae2822_cp.png", dpi=130)
    plt.close(fig)
    # Mach field around the RAE 2822
    r, s, fvm = best
    fig, ax = plt.subplots(figsize=(10, 5))
    near = (np.abs(fvm.xc[:, 0] - 0.5) < 1.0) & (np.abs(fvm.xc[:, 1]) < 0.6)
    tri = mtri.Triangulation(fvm.xc[near, 0], fvm.xc[near, 1])
    cs = ax.tricontourf(tri, s.mach_field()[near], 40, cmap="jet")
    ax.tricontour(tri, s.mach_field()[near], [1.0], colors="k", linewidths=0.8)
    fig.colorbar(cs, ax=ax, label="Mach")
    ax.set_aspect("equal"); ax.set_xlim(-0.3, 1.5); ax.set_ylim(-0.4, 0.6)
    ax.set_title("RAE 2822 case 9: Mach number (black: sonic line)")
    fig.tight_layout(); fig.savefig(OUT / "transonic_rae2822_mach.png", dpi=130)
    plt.close(fig)

    (OUT / "transonic_results.json").write_text(json.dumps(out, indent=1))
    (OUT / "transonic_report.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
