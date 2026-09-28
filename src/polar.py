"""
Airfoil polars (Phase 3, step E1): angle-of-attack sweeps with warm starts.

For every angle: mesh (wake refinement follows alpha) -> solve -> Cl, Cd,
Cd_p, Cd_v, Cm (about c/4, nose-up positive), wall Cp/Cf.

  * warm start: each angle starts from the converged solution of the previous
    angle, interpolated to the new mesh (nearest cell) - far fewer iterations
    than a uniform start
  * `jobs` processes, each on a contiguous block of angles: it starts at the
    block's smallest |alpha|, sweeps up, then down again from its start
  * convergence: residuals < tol, OR the forces have settled (Cl, Cd, Cm
    change < force_tol over the last `window` iterations) with residuals
    < 1e-5 - engineering convergence, identical coefficients to ~4 digits
  * a point that does not converge (typically stall: separated flow that
    steady RANS cannot settle) is reported with the mean and the amplitude of
    the coefficients over the last iterations and flagged "unsteady"

Regimes: Re >= 1e5 -> Spalart-Allmaras, y+ < 1 mesh (validate_naca0012_sa);
Re < 1e5 -> laminar (validate_airfoil settings).
"""

import math
import time
from multiprocessing import get_context

import numpy as np

import airfoil_geom as ag
import hybrid_mesher as hm
from fvmesh import FVMesh
from postproc import wall_data
from simple_solver import SimpleSolver

LAMINAR_RE = 1e5


def make_case(spec, re, alpha, level=1.0, farfield=None):
    """Airfoil spec -> (FVMesh, mesh dict, bcs, U_inf, turbulence, info)."""
    turb = re >= LAMINAR_RE
    r = level
    loop, info = ag.load_airfoil(spec, n_side=int(round((160 if turb else 120) * r)))
    if turb:
        import validate_naca0012_sa as v
        mesh = v.airfoil_mesh(level, alpha, wall=loop, re=re, farfield=farfield or 100.0)
    else:
        from validate_airfoil import laminar_h1
        mesh = hm.build_mesh(wall=loop, h1=laminar_h1(re) / r, growth=1.1 ** (1 / r),
                             n_layers=int(round(17 * r)), bl_ar_cap=float("inf"),
                             farfield_radius=farfield or 40.0, n_farfield=int(round(64 * r)),
                             h_near=0.02 / r, grad=0.12 / r, wake_length=5.0, wake_h=0.01 / r,
                             wake_grad=0.03 / r, alpha_deg=alpha, label=info["name"],
                             verbose=False)
    U = (math.cos(math.radians(alpha)), math.sin(math.radians(alpha)))
    P = mesh["points"]
    ff = mesh["patches"]["farfield"]
    t = P[ff[:, 1]] - P[ff[:, 0]]
    inflow = t[:, 1] * U[0] - t[:, 0] * U[1] < 0
    patches = {"wall": mesh["patches"]["airfoil"], "inlet": ff[inflow], "outlet": ff[~inflow]}
    fvm = FVMesh(P, mesh["cells"], patches)
    bcs = {"wall": {"type": "wall"}, "inlet": {"type": "inlet", "U": U},
           "outlet": {"type": "outlet"}}
    return fvm, mesh, patches, bcs, U, ("sa" if turb else None), info


def run_point(spec, re, alpha, level=1.0, init=None, tol=1e-7, max_iter=6000,
              force_tol=2e-5, window=200, farfield=None, out=None):
    """One angle of attack. init: dict(xc, U, p, nuT) from a neighbouring
    angle (warm start). Returns (result dict, state dict for the next angle)."""
    t0 = time.time()
    fvm, mesh, patches, bcs, U, turb, info = make_case(spec, re, alpha, level, farfield)
    s = SimpleSolver(fvm, bcs, 1.0 / re, U0=U, turbulence=turb)
    if init is not None:
        from scipy.spatial import cKDTree
        _, j = cKDTree(init["xc"]).query(fvm.xc)
        # rotate the old velocity field by the change of angle: the far field
        # then matches the new free stream exactly
        da = math.radians(alpha - init["alpha"])
        R = np.array([[math.cos(da), -math.sin(da)], [math.sin(da), math.cos(da)]])
        s.set_initial(init["U"][j] @ R.T, init["p"][j],
                      init["nuT"][j] if init.get("nuT") is not None else None)
    vf = None
    if turb:
        import validate_naca0012_sa as v
        vf = v.VortexFarfield(s, alpha)
    hist = []

    def monitor(it, r):
        if vf is not None:
            vf.update(it)
        if it % 10 == 0:
            c = wall_data(s, "wall", U)[1]
            hist.append((it, c["Cl"], c["Cd"], c["Cm"]))
            n = window // 10
            if len(hist) > n and max(r.values()) < 1e-5:
                h = np.array(hist[-n - 1:])[:, 1:]
                scale = np.maximum(np.abs(h[-1]), [0.1, 0.01, 0.01])  # Cl, Cd, Cm scales
                if np.all(np.ptp(h, axis=0) / scale < force_tol):
                    return True
        return None

    it = s.solve(tol=tol, max_iter=max_iter, verbose=False, callback=monitor)
    w, c = wall_data(s, "wall", U)
    status = "converged" if s.converged else "unsteady"
    res = {"alpha": alpha, "Cl": float(c["Cl"]), "Cd": float(c["Cd"]), "Cm": float(c["Cm"]),
           "Cd_p": float(c["Cd_pressure"]), "Cd_v": float(c["Cd_viscous"]),
           "status": status, "iterations": int(it), "time_s": round(time.time() - t0, 1),
           "cells": int(fvm.ncells), "yplus_max": float(w["yplus"].max()),
           "warm_start": init is not None}
    if not s.converged and len(hist) >= 20:                 # mean + amplitude of the tail
        h = np.array(hist[len(hist) // 2:])
        for k, q in zip(("Cl", "Cd", "Cm"), (1, 2, 3)):
            res[k] = float(h[:, q].mean())
            res[k + "_amp"] = float(0.5 * np.ptp(h[:, q]))
    o = np.lexsort((w["x"], w["side"]))
    res["wall"] = {k: np.asarray(w[k])[o].round(7).tolist() for k in ("x", "y", "side", "Cp", "Cf")}
    if out is not None:                                    # full fields for the viewer
        import case as cs
        from pathlib import Path
        folder = Path(out) / f"alpha_{alpha:+06.2f}"
        meta = {"case": f"{info['name']} alpha {alpha:g}", "kind": "naca", "Re": re,
                "alpha_deg": alpha, "U_inf": list(U), "nu": 1.0 / re, "ref_length": 1.0,
                "turbulence": "Spalart-Allmaras (SA-noft2)" if turb else "laminar",
                "bcs": {"wall": "no-slip wall", "inlet": "U_inf" + (" + point vortex" if turb else ""),
                        "outlet": "zero-gradient U, p" + (" (Bernoulli, point vortex)" if turb else " = 0")}}
        q = cs.write_mesh(folder, fvm, mesh, patches, meta)
        cs.write_results(folder, s, fvm, mesh, patches, U, q)
        meta.update(iterations=int(it), converged=bool(s.converged),
                    coefficients={k: res[k] for k in ("Cl", "Cd", "Cm", "Cd_p", "Cd_v")})
        import json
        (folder / "case.json").write_text(json.dumps(meta, indent=2))
    state = {"xc": fvm.xc.copy(), "U": s.U.copy(), "p": s.p.copy(), "alpha": alpha,
             "nuT": s.turb.nuT.copy() if s.turb is not None else None}
    return res, state


def _chain(args):
    """A block of neighbouring angles: start at the smallest |alpha|, sweep
    upward, then downward from the start again (each step warm-started)."""
    spec, re, alphas, level, kw, log = args
    i0 = int(np.argmin(np.abs(alphas)))
    out = []
    start_state = None
    for seq in (alphas[i0:], alphas[:i0][::-1]):
        state = start_state
        for a in seq:
            r, st = run_point(spec, re, a, level, init=state, **kw)
            if log:
                print(f"  alpha {a:6.2f}: Cl {r['Cl']:8.4f}  Cd {r['Cd']:.5f}  Cm {r['Cm']:+.4f}  "
                      f"{r['status']:9s} {r['iterations']:5d} it  {r['time_s']:6.1f} s"
                      f"{'' if r['warm_start'] else '  (cold start)'}", flush=True)
            out.append(r)
            stalled = r["status"] != "converged" and r.get("Cl_amp", 0) > 0.05 * abs(r["Cl"])
            state = None if stalled else st          # never warm-start from a stalled field
            if a == alphas[i0]:
                start_state = state
    return out


def polar(spec, re, alphas, level=1.0, jobs=2, log=True, **kw):
    """Sweep alphas; `jobs` processes, each on a contiguous block of angles.
    Returns the results sorted by alpha."""
    alphas = sorted(float(a) for a in alphas)
    jobs = max(1, min(jobs, len(alphas)))
    blocks = [list(b) for b in np.array_split(alphas, jobs) if len(b)]
    if len(blocks) > 1:
        with get_context("fork").Pool(len(blocks)) as pool:
            parts = pool.map(_chain, [(spec, re, b, level, kw, log) for b in blocks])
    else:
        parts = [_chain((spec, re, blocks[0], level, kw, log))]
    return sorted([r for p in parts for r in p], key=lambda r: r["alpha"])
