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

Regimes: Re >= 1e5 -> Spalart-Allmaras, y+ < 1 mesh (validate_naca0012_sa),
with free transition (e^N envelope, transition.py) by default;
Re < 1e5 -> laminar (validate_airfoil settings).
"""

import math
import time
from multiprocessing import get_context

import numpy as np

import airfoil_geom as ag
import hybrid_mesher as hm
import multi_element as me
from fvmesh import FVMesh
from postproc import wall_data
from simple_solver import SimpleSolver

LAMINAR_RE = 1e5


def resolve_solver(solver, re):
    """"auto" -> the environment variable CFD_SOLVER if set ("simple" /
    "coupled"), else "simple". Phase 4b measurement (2-core machine, NACA
    0012 polar -4..16): coupled 470 s vs SIMPLEC 555 s, i.e. ~1.2x, with
    Cl up to 6e-4 lower at small angles (a second, nearly-degenerate
    state of the limited scheme) -> SIMPLEC stays the default until the
    coupled solver measures clearly faster on the target machine
    (python benchmark_coupled.py). The coupled settings were tuned on
    turbulent (SA) cases."""
    import os
    if solver == "auto":
        solver = os.environ.get("CFD_SOLVER", "simple")
    if solver not in ("simple", "coupled"):
        raise ValueError(f"solver must be simple, coupled or auto, not {solver!r}")
    if solver == "coupled":
        import pypardiso  # noqa: F401  (clear error if missing)
    return solver


def coefficients(solver, walls, U):
    """Total force coefficients over several wall patches (multi-element
    airfoil) + the per-patch ones. Returns (total dict, {patch: dict})."""
    per = {w: wall_data(solver, w, U)[1] for w in walls}
    tot = {k: float(sum(c[k] for c in per.values())) for k in next(iter(per.values()))}
    return tot, per


def make_case(spec, re, alpha, level=1.0, farfield=None):
    """Airfoil spec -> (FVMesh, mesh dict, patches, bcs, U_inf, turbulence, info).
    info["walls"]: the wall patch names (["wall"], or wall_<element> for a
    multi-element airfoil, see multi_element.py)."""
    turb = re >= LAMINAR_RE
    r = level
    U = (math.cos(math.radians(alpha)), math.sin(math.radians(alpha)))
    if me.is_multi(spec):
        if not turb:
            raise ValueError("multi-element airfoils: Re >= 1e5 (Spalart-Allmaras) only")
        elements, info = me.load_multi(spec, level=level)
        mesh = hm.multi_mesh(elements, re, level, alpha, farfield=farfield or 100.0,
                             label=info["name"])
        walls = [k for k in mesh["patches"] if k.startswith("wall_")]
        patches = {w: mesh["patches"][w] for w in walls}
    else:
        loop, info = ag.load_airfoil(spec, n_side=int(round((160 if turb else 120) * r)))
        mesh = _single_mesh(loop, info, re, alpha, level, farfield, turb)
        walls = ["wall"]
        patches = {"wall": mesh["patches"]["airfoil"]}
    P = mesh["points"]
    ff = mesh["patches"]["farfield"]
    t = P[ff[:, 1]] - P[ff[:, 0]]
    inflow = t[:, 1] * U[0] - t[:, 0] * U[1] < 0
    patches.update(inlet=ff[inflow], outlet=ff[~inflow])
    fvm = FVMesh(P, mesh["cells"], patches)
    bcs = {w: {"type": "wall"} for w in walls}
    bcs.update(inlet={"type": "inlet", "U": U}, outlet={"type": "outlet"})
    info["walls"] = walls
    return fvm, mesh, patches, bcs, U, ("sa" if turb else None), info


def _single_mesh(loop, info, re, alpha, level, farfield, turb):
    r = level
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
    return mesh


def run_point(spec, re, alpha, level=1.0, init=None, tol=1e-7, max_iter=6000,
              force_tol=2e-5, window=200, farfield=None, out=None, transition="free",
              ncrit=9.0, sequencing=False, solver="auto", mach=0.0):
    """One angle of attack. init: dict(xc, U, p, nuT, x_tr) from a neighbouring
    angle (warm start). transition (turbulent Re only): "free" (e^N, N_crit
    = ncrit), "off" (fully turbulent) or {"upper": x, "lower": x} (fixed trip).
    sequencing (Phase 4): a cold start (no init) at Re >= 1e5 is first solved
    on a 0.7x mesh (fully turbulent, residuals 1e-5) and the result
    interpolated to this mesh: SIMPLEC needs more iterations the finer the
    mesh (30P30N 622 / 845 / 1335 at levels 0.7 / 1 / 1.4), so the fine mesh
    only converges the last part (NACA 0012 SA: 710 -> 417 fine iterations).
    solver (Phase 4b): "coupled" (u, v, p solved together, MKL PARDISO;
    3-8x fewer iterations, each ~4x dearer), "simple" (SIMPLEC), or "auto"
    (= $CFD_SOLVER, default simple; see resolve_solver).
    mach (Phase 5): > 0 runs the compressible solver (compressible.py:
    ideal gas, energy equation, compressible SA; shocks are captured).
    Transition with mach > 0: fixed trips or fully turbulent; free (e^N,
    incompressible stability theory on the compressible edge velocity) only
    for M < 0.4.
    Returns (result dict, state dict for the next angle)."""
    t0 = time.time()
    compressible = mach > 0
    if compressible:
        solver = "compressible"
        # shocks + limiter: Cl, Cd creep at the 1e-4 level for a long time
        # (NACA 0012 Euler M 0.8: Cl 0.3575 -> 0.3577 over 800 iterations)
        force_tol = max(force_tol, 1e-4)
        if transition == "free" and mach >= 0.4:
            transition = "off"
    else:
        solver = resolve_solver(solver, re)
    # iteration-count based settings were tuned for SIMPLEC (~800 iterations
    # per point); the coupled solver needs ~10x fewer
    k_it = 0.1 if solver == "coupled" else 1.0
    if window == 200:
        window = int(round(200 * k_it))
    if sequencing and init is None and re >= LAMINAR_RE and level >= 0.85:
        try:
            _, init = run_point(spec, re, alpha, level=0.7 * level, tol=1e-5,
                                max_iter=max_iter, farfield=farfield, transition="off",
                                sequencing=False, force_tol=force_tol, window=window,
                                solver=solver if not compressible else "auto", mach=mach)
            init["sequencing"] = True
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            init = None                       # coarse start failed: cold start
    fvm, mesh, patches, bcs, U, turb, info = make_case(spec, re, alpha, level, farfield)
    multi = info.get("multi", False)
    # multi-element: the convection limiter acts on ALL cells during the
    # start (the thinned/redistributed quads in the coves and slots grew
    # wiggles from a uniform start), then only on the triangles as for a
    # single airfoil (30P30N alpha 8: Cl 3.2146 -> 3.2156, Cd -0.5 %)
    if compressible:
        from compressible import CompressibleSolver
        s = CompressibleSolver(fvm, bcs, 1.0 / re, mach, U0=U, turbulence=turb,
                               **({"limiter_cells": "all"} if multi else {}))
    elif solver == "coupled":
        from coupled import CoupledSolver
        s = CoupledSolver(fvm, bcs, 1.0 / re, U0=U, turbulence=turb,
                          **({"limiter_cells": "all"} if multi else {}))
    else:
        s = SimpleSolver(fvm, bcs, 1.0 / re, U0=U, turbulence=turb,
                         **({"limiter_cells": "all"} if multi else {}))
    switch = {"done": not multi}
    if init is not None and init.get("sequencing"):
        # coarse-mesh solution, same angle: linear interpolation (nearest
        # cell outside the convex hull of the coarse cell centres)
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import cKDTree
        cols = [init["U"][:, 0], init["U"][:, 1], init["p"]] + \
            ([init["nuT"]] if init.get("nuT") is not None else [])
        vals = LinearNDInterpolator(init["xc"], np.c_[tuple(cols)])(fvm.xc)
        bad = np.isnan(vals).any(axis=1)
        if bad.any():
            _, j = cKDTree(init["xc"]).query(fvm.xc[bad])
            vals[bad] = np.c_[tuple(cols)][j]
        s.set_initial(vals[:, :2], vals[:, 2],
                      np.maximum(vals[:, 3], 0.0) if len(cols) > 3 else None,
                      **_thermo_init(s, init, None, fvm))
    elif init is not None:
        from scipy.spatial import cKDTree
        _, j = cKDTree(init["xc"]).query(fvm.xc)
        # rotate the old velocity field by the change of angle: the far field
        # then matches the new free stream exactly
        da = math.radians(alpha - init["alpha"])
        R = np.array([[math.cos(da), -math.sin(da)], [math.sin(da), math.cos(da)]])
        s.set_initial(init["U"][j] @ R.T, init["p"][j],
                      init["nuT"][j] if init.get("nuT") is not None else None,
                      **_thermo_init(s, init, j, fvm))
    walls = info["walls"]
    vf = None
    if turb:
        import validate_naca0012_sa as v
        VF = v.VortexFarfield
        if compressible:
            from compressible import CompressibleVortexFarfield as VF
        vf = VF(s, alpha, wall=walls, center=(0.25 * info.get("ref_chord", 1.0), 0.0),
                every=max(1, int(round(50 * k_it))))
        if init is not None:
            vf.update(0)        # a warm / sequenced start already carries its lift
    if multi and transition != "off":
        transition = "off"        # e^N transition is single-element only (see report note)
    hist = []
    osc = {"flag": False}
    tr = None

    def monitor(it, r):
        if vf is not None:
            vf.update(it)
        if not switch["done"] and it >= 300 * k_it and max(r.values()) < 1e-4:
            s.limiter_mask = s.m.nverts == 3
            switch["done"] = True
        every = max(1, int(round(10 * k_it)))
        if it % every == 0:
            c = coefficients(s, walls, U)[0]
            hist.append((it, c["Cl"], c["Cd"], c["Cm"]))
            n = max(window // every, 3)
            # compressible: the limiter at shocks stalls the residuals near
            # 1e-5 (NACA 0012 M 0.8: 2e-5 while Cl/Cd are fixed to 4 digits)
            if len(hist) > n and max(r.values()) < (1e-4 if compressible else 1e-5):
                h = np.array(hist[-n - 1:])[:, 1:]
                scale = np.maximum(np.abs(h[-1]), [0.1, 0.01, 0.01])  # Cl, Cd, Cm scales
                if np.all(np.ptp(h, axis=0) / scale < force_tol):
                    return True
            # small persistent oscillation (e.g. laminar flow reaching a blunt
            # trailing edge sheds weakly): residuals stall, forces vary < 0.5 %
            # -> stop and report the mean
            total = len(s.history)
            k6, k1 = int(600 * k_it), int(100 * k_it)
            if total >= 1500 * k_it and len(hist) >= 60 and (tr is None or tr.converged):
                res_max = np.array([max(q.values()) for q in s.history[-k6:]])
                stalled = res_max[-k1:].min() > 0.5 * res_max[:k1].min()
                # a steadily GROWING residual is a slow transient still on its
                # way (coupled warm starts: x1.03/iteration for ~100
                # iterations, then converges), not a limit cycle
                q = len(res_max) // 4
                stalled = stalled and res_max[-q:].mean() < 1.3 * res_max[-2 * q:-q].mean()
                h = np.array(hist[-50:])[:, 1:]
                small = (np.ptp(h[:, 0]) < 0.005 * max(abs(h[:, 0].mean()), 0.1) and
                         np.ptp(h[:, 1]) < 0.01 * h[:, 1].mean())
                if stalled and small:
                    osc["flag"] = True
                    return True
        return None

    tr = None
    if turb and transition != "off":
        import transition as T
        if isinstance(transition, dict):
            tr = T.Transition(s, mesh, re, mode="fixed", x_tr=transition)
        else:
            # always start from an estimate on a (briefly) fully turbulent flow,
            # also on warm starts: carrying the previous angle's transition
            # points over can leave the laminar region past the new, further
            # upstream laminar separation -> the leading-edge bubble bursts and
            # the solution locks onto a stalled branch
            tr = T.Transition(s, mesh, re, mode="free", ncrit=ncrit, x_tr=None)
        it = T.solve_with_transition(s, tr, tol=tol, max_iter=max_iter, callback=monitor,
                                     scale=k_it)
    else:
        it = s.solve(tol=tol, max_iter=max_iter, verbose=False, callback=monitor)
    c, per = coefficients(s, walls, U)
    ws = [wall_data(s, wn, U)[0] for wn in walls]
    w = {k: np.concatenate([x[k] for x in ws]) for k in ("x", "y", "side", "Cp", "Cf", "yplus")}
    w["element"] = np.concatenate([np.full(len(x["x"]), k) for k, x in enumerate(ws)])
    status = "converged" if s.converged else "unsteady"
    if status == "converged" and tr is not None and not tr.converged:
        status = "transition not converged"
    if osc["flag"]:
        status = "converged (mean of a small oscillation)"
    res = {"alpha": alpha, "Cl": float(c["Cl"]), "Cd": float(c["Cd"]), "Cm": float(c["Cm"]),
           "Cd_p": float(c["Cd_pressure"]), "Cd_v": float(c["Cd_viscous"]),
           "status": status, "iterations": int(it), "time_s": round(time.time() - t0, 1),
           "cells": int(fvm.ncells), "yplus_max": float(w["yplus"].max()),
           "warm_start": init is not None and not init.get("sequencing"),
           "solver": solver, "mach": float(mach),
           "sequenced": bool(init is not None and init.get("sequencing")),
           "transition": ("fully turbulent" if tr is None else
                          (f"free, N_crit {ncrit:g}" if tr.mode == "free" else "fixed")),
           "x_tr_upper": None if tr is None else round(tr.x_tr["upper"], 4),
           "x_tr_lower": None if tr is None else round(tr.x_tr["lower"], 4),
           "transition_converged": None if tr is None else bool(tr.converged),
           "transition_converged_by": None if tr is None else tr.converged_by}
    if (osc["flag"] or not s.converged) and len(hist) >= 20:   # mean + amplitude of the tail
        h = np.array(hist[-50:] if osc["flag"] else hist[len(hist) // 2:])
        for k, q in zip(("Cl", "Cd", "Cm"), (1, 2, 3)):
            res[k] = float(h[:, q].mean())
            res[k + "_amp"] = float(0.5 * np.ptp(h[:, q]))
    if multi:
        res["elements"] = {wn[5:]: {k: float(per[wn][k2]) for k, k2 in
                                    (("Cl", "Cl"), ("Cd", "Cd"), ("Cm", "Cm"),
                                     ("Cd_p", "Cd_pressure"), ("Cd_v", "Cd_viscous"))}
                           for wn in walls}
        res["element_names"] = [wn[5:] for wn in walls]
    o = np.lexsort((w["x"], w["side"], w["element"]))
    res["wall"] = {k: np.asarray(w[k])[o].round(7).tolist()
                   for k in ("x", "y", "side", "Cp", "Cf", "element")}
    if out is not None:                                    # full fields for the viewer
        import case as cs
        from pathlib import Path
        folder = Path(out) / f"alpha_{alpha:+06.2f}"
        meta = {"case": f"{info['name']} alpha {alpha:g}", "kind": "naca", "Re": re,
                "mach": float(mach),
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
    if compressible:
        res["M_max"] = float(s.mach_field().max())
        res["supersonic_cells"] = int(s.stats["supersonic_cells"])
    state = {"xc": fvm.xc.copy(), "U": s.U.copy(), "p": s.p.copy(), "alpha": alpha,
             "T": s.T.copy() if compressible else None, "mach": mach,
             "x_tr": None if tr is None else dict(tr.x_tr),
             "nuT": s.turb.nuT.copy() if s.turb is not None else None}
    return res, state


def _thermo_init(s, init, j, fvm):
    """compressible warm / sequenced start: temperature from the previous
    state (same Mach), else from the total enthalpy and the new velocity"""
    if not hasattr(s, "mach"):
        return {}
    T = init.get("T")
    if T is None or init.get("mach") != s.mach:
        return {}
    if j is None:
        from scipy.interpolate import NearestNDInterpolator
        return {"T": NearestNDInterpolator(init["xc"], T)(fvm.xc)}
    return {"T": T[j]}


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
            try:
                r, st = run_point(spec, re, a, level, init=state, **kw)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as err:
                # a diverged point (NaN/inf in the solver) must not take the whole
                # sweep down; any other error (bad input) still stops it
                if not any(k in str(err).lower() for k in ("nan", "inf", "singular")):
                    raise
                nan = float("nan")
                r = {"alpha": a, "Cl": nan, "Cd": nan, "Cm": nan, "Cd_p": nan, "Cd_v": nan,
                     "status": f"diverged ({type(err).__name__})", "iterations": 0,
                     "time_s": 0.0, "cells": 0, "yplus_max": nan, "warm_start": state is not None,
                     "transition": "-", "x_tr_upper": None, "x_tr_lower": None,
                     "wall": {k: [] for k in ("x", "y", "side", "Cp", "Cf", "element")}}
                st = None
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
    import os
    os.environ["CFD_JOBS"] = str(jobs)          # PARDISO threads per worker (coupled.py)
    blocks = [list(b) for b in np.array_split(alphas, jobs) if len(b)]
    if len(blocks) > 1:
        with get_context("fork").Pool(len(blocks)) as pool:
            parts = pool.map(_chain, [(spec, re, b, level, kw, log) for b in blocks])
    else:
        parts = [_chain((spec, re, blocks[0], level, kw, log))]
    return sorted([r for p in parts for r in p], key=lambda r: r["alpha"])
