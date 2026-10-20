"""
Goal-oriented error estimation and mesh adaptation with the discrete adjoint.

    python src/run.py adjoint_error estimate naca0012 --re 3e6 --alpha 4 --level 0.5 --check
    python src/run.py adjoint_error adapt naca0012 --re 3e6 --alpha 4 --objective Cd --cycles 3

Adjoint-weighted residual (Pierce & Giles 2000; Venditti & Darmofal 2002):
the coarse-mesh solution X_H is reconstructed on the nested fine mesh h
(every cell split in four; X_H^h = parent value + parent gradient . offset);
the fine-mesh discrete residual of that state, weighted with the adjoint of
the objective, estimates the objective of the fine-mesh solution

    I_h(X_h) ~ I_h(X_H^h) - psi_h^T R_h(X_H^h)

with one residual evaluation and no fine-mesh flow solution. The auxiliary
unknowns of the residual (pressure gradient, face fluxes, vortex
circulation) are relaxed onto their own fine-mesh equations, so only the
momentum, continuity and SA rows carry residual and only those components of
the adjoint are needed; they are reconstructed from the coarse adjoint in the
same way. (Interpolating instead between the non-nested meshes the mesher
makes at two levels gave useless estimates: linear interpolation over the
stretched boundary-layer cells puts wrong wall gradients into the
reconstructed state.)

The cell contributions |psi_i . R_i| are the adaptation indicator: the mesher's
triangle size field is scaled down where they are large and up where they
are small (fixed-fraction marking), keeping the validated boundary-layer
mesh. The boundary-layer (quad) cells cannot be adapted by the size field;
their share of the estimated error is reported separately.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from aerolab.design import adjoint as A


def subdivide(points, cells, patches):
    """Uniform nested refinement: every quad -> 4 quads, every triangle -> 4
    triangles (edge midpoints, quad centre); boundary edges split in two, so
    the fine mesh has exactly the coarse (polygonal) geometry. Returns
    (points, cells, patches, parent) with parent[i] = coarse cell of fine cell i."""
    P = [tuple(p) for p in np.asarray(points, float)]
    mid = {}

    def m(a, b):
        k = (min(a, b), max(a, b))
        if k not in mid:
            mid[k] = len(P)
            P.append(tuple(0.5 * (np.asarray(P[a]) + np.asarray(P[b]))))
        return mid[k]
    out, parent = [], []
    for ci, c in enumerate(cells):
        k = len(c)
        ms = [m(c[i], c[(i + 1) % k]) for i in range(k)]
        if k == 3:
            a, b, cc = c
            mab, mbc, mca = ms
            out += [[a, mab, mca], [mab, b, mbc], [mca, mbc, cc], [mab, mbc, mca]]
        elif k == 4:
            ctr = len(P)
            P.append(tuple(np.mean([P[v] for v in c], axis=0)))
            for i in range(4):
                out.append([c[i], ms[i], ctr, ms[i - 1]])
        else:
            raise ValueError("subdivide: triangles and quads only")
        parent += [ci] * 4
    pat = {}
    for name, E in patches.items():
        e2 = []
        for a, b in np.asarray(E):
            mm = m(a, b)
            e2 += [(a, mm), (mm, b)]
        pat[name] = np.array(e2)
    return np.array(P), out, pat, np.array(parent)


def _reconstruct(mc, mf, parent, phi, grad):
    """piecewise-linear reconstruction on the children: phi_P + grad_P.(x - x_P)"""
    return phi[parent] + np.einsum("ij,ij->i", grad[parent], mf.xc - mc.xc[parent])


def transfer(pc, adj_c, re, alpha, n_smooth=0):
    """coarse solution + adjoint -> nested fine mesh: fine Residual, the
    reconstructed state X_H^h (auxiliary unknowns relaxed onto their fine
    equations) and the reconstructed adjoint (cell rows)"""
    from aerolab.core.fvmesh import FVMesh
    from aerolab.core.simple_solver import SimpleSolver
    sc = pc["solver"]
    mc = sc.m
    rc = adj_c.res
    walls = pc["walls"]
    pats = {n: np.c_[mc.ba[mc.patch_faces[n]], mc.bb[mc.patch_faces[n]]] for n in mc.patch_names}
    Pf, cf, patf, parent = subdivide(mc.points, mc.cells, pats)
    mf = FVMesh(Pf, cf, patf)
    U = (np.cos(np.radians(alpha)), np.sin(np.radians(alpha)))
    bcs = {w: {"type": "wall"} for w in walls}
    bcs.update(inlet={"type": "inlet", "U": U}, outlet={"type": "outlet"})
    turb = "sa" if sc.turb is not None else None
    s_f = SimpleSolver(mf, bcs, sc.nu, U0=U, turbulence=turb)
    Ub, _ = sc._boundary_values()
    flds = [_reconstruct(mc, mf, parent, sc.U[:, k], sc.gU[k]) for k in range(2)]
    pnew = _reconstruct(mc, mf, parent, sc.p, sc.gradp)
    nt = None
    if turb:
        g = mc.grad(sc.turb.nuT, sc.turb.boundary_nuT())
        nt = np.maximum(_reconstruct(mc, mf, parent, sc.turb.nuT, g), 0.0)
        if sc.turb.gamma is not None:
            s_f.turb.gamma = np.asarray(sc.turb.gamma)[parent]
        if sc.turb.seed is not None:
            idx, val = sc.turb.seed
            v = np.full(mc.ncells, np.nan)
            v[idx] = val
            sel = np.where(~np.isnan(v[parent]))[0]
            s_f.turb.seed = (sel, v[parent][sel]) if len(sel) else None
    s_f.set_initial(np.c_[flds[0], flds[1]], pnew, nt)
    s_f.limiter_mask = None if sc.limiter_mask is None else np.asarray(sc.limiter_mask)[parent]
    gamma_v = adj_c.X[rc.off["G"]] if rc.vortex else 0.0
    if n_smooth:
        # a few fine-mesh solver iterations remove the reconstruction's
        # high-frequency residual (child-to-child jumps in the boundary layer)
        from aerolab.validation import validate_naca0012_sa as v
        vf = None
        if pc["vortex"] is not None:
            vf = v.VortexFarfield(s_f, alpha, wall=walls, center=tuple(pc["vortex"].center),
                                  every=1)
            vf.gamma = gamma_v
        s_f.solve(tol=0.0, max_iter=n_smooth, verbose=False,
                  callback=(vf.update if vf is not None else None))
        if vf is not None:
            vf.update(0)
            gamma_v = vf.gamma
    res_f = A.Residual(s_f, alpha, pc["vortex"], walls)
    X = res_f.state(gamma_v)
    if not n_smooth:
        gp = sc.gradp[parent]             # starting guess; relaxed onto the fine equation
        X[res_f.off["gx"]:res_f.off["gx"] + res_f.n] = gp[:, 0]
        X[res_f.off["gy"]:res_f.off["gy"] + res_f.n] = gp[:, 1]
    aux = ["gx", "gy", "F", "Fo"] + (["G"] if res_f.vortex else [])
    ch = np.inf
    for it in range(600):
        out = res_f.evaluate(X, parts=True)
        ch = 0.0
        for k in aux:
            sl = slice(res_f.off[k], res_f.off[k] + np.size(out[k]))
            X[sl] -= out[k]
            ch = max(ch, float(np.abs(out[k]).max() / max(np.abs(X[sl]).max(), 1e-30)))
        if ch < 1e-12:
            break
    # adjoint (cell rows): reconstructed with its own least-squares gradient
    names = [k for k in res_f.fields if k in ("u", "v", "p", "nt")]
    psi_f = {}
    for obj, psi in adj_c.psi.items():
        d = {}
        for k in names:
            v = psi[rc.off[k]:rc.off[k] + rc.n]
            gv = mc.grad(v, v[mc.bcell])
            d[k] = _reconstruct(mc, mf, parent, v, gv)
        psi_f[obj] = d
    return res_f, X, psi_f, {"fvm": mf, "parent": parent, "aux_iterations": it + 1,
                             "aux_residual": ch, "solver": s_f}


def _solve_fine(s_f, alpha, walls, vortex, max_iter=30000):
    """converge the nested fine mesh (from the reconstructed state)"""
    from aerolab.validation import validate_naca0012_sa as v
    vf = None
    if vortex is not None:
        vf = v.VortexFarfield(s_f, alpha, wall=walls, center=tuple(vortex.center), every=1)
        vf.update(0)
    hist = []

    def cb(it, r):
        if vf is not None:
            vf.update(it)
        hist.append(max(r.values()))
        if len(hist) > 400 and min(hist[-200:]) > 0.7 * min(hist[-400:-200]):
            return True
        return None
    s_f.solve(tol=1e-10, max_iter=max_iter, verbose=False, callback=cb)
    return s_f


def estimate(spec, re, alpha, level=0.5, objectives=("Cl", "Cd", "Cm"), transition="off",
             size_scale=None, primal=None, adjoint=None, check=False, n_smooth=30, log=print):
    """Adjoint-weighted-residual estimate of the objective values on the
    nested fine mesh (every cell split in 4). check=True also solves the
    fine-mesh flow to measure the estimate (effectivity = estimated error /
    actual error of the coarse value)."""
    t0 = time.time()
    pc = primal or A.solve_primal(spec, re, alpha, level, transition, size_scale=size_scale)
    adj = adjoint
    if adj is None:
        adj = A.Adjoint(pc["solver"], alpha, pc["vortex"], pc["walls"])
        adj.solve(objectives)
    from aerolab.airfoil import polar as pl
    cH = pl.coefficients(pc["solver"], pc["walls"], pc["U"])[0]
    res_f, Xf, psi_f, extra = transfer(pc, adj, re, alpha, n_smooth)
    R = res_f.evaluate(Xf, parts=True)
    fo = res_f.forces(Xf)
    out = {"airfoil": spec, "re": re, "alpha": alpha, "level": level,
           "cells_coarse": int(pc["solver"].m.ncells), "cells_fine": int(res_f.n),
           "aux_iterations": extra["aux_iterations"], "aux_residual": extra["aux_residual"],
           "objectives": {}}
    quad = np.asarray(extra["fvm"].nverts) == 4
    eta = {}
    for k in objectives:
        contrib = sum(psi_f[k][f] * np.real(R[f]) for f in psi_f[k])
        dI = -float(contrib.sum())
        eta[k] = np.abs(contrib)
        out["objectives"][k] = {"coarse": float(cH[k]),
                                "fine_of_reconstructed": float(np.real(fo[k])),
                                "correction": dI,
                                "estimate_fine": float(np.real(fo[k])) + dI,
                                "indicator_sum": float(eta[k].sum()),
                                "indicator_share_bl_quads": float(eta[k][quad].sum() /
                                                                  max(eta[k].sum(), 1e-300))}
    if check:
        sf = _solve_fine(extra["solver"], alpha, pc["walls"], pc["vortex"])
        cf = pl.coefficients(sf, pc["walls"], pc["U"])[0]
        for k in objectives:
            o = out["objectives"][k]
            o["fine_solved"] = float(cf[k])
            o["error_coarse"] = float(cH[k] - cf[k])
            o["error_estimate"] = float(cH[k] - o["estimate_fine"])
            o["effectivity"] = (o["error_estimate"] / o["error_coarse"]
                                if cf[k] != cH[k] else None)
    out["time_s"] = time.time() - t0
    out["_eta"] = eta
    out["_fine"] = extra
    out["_primal"] = pc
    out["_adjoint"] = adj
    for k in objectives:
        o = out["objectives"][k]
        log(f"  {k}: coarse {o['coarse']:.6f}  fine(reconstructed) {o['fine_of_reconstructed']:.6f}"
            f"  correction {o['correction']:+.2e}  -> estimate {o['estimate_fine']:.6f}"
            + (f"  | fine solved {o['fine_solved']:.6f} (effectivity "
               f"{o['effectivity']:.2f})" if o.get("effectivity") is not None else ""))
    return out


# ----------------------------------------------------------------------
# adaptation
# ----------------------------------------------------------------------
class SizeScale:
    """piecewise-constant (cell-centred, smoothed over neighbours) size
    multiplier for the mesher"""

    def __init__(self, xy, factor, k=6):
        from scipy.spatial import cKDTree
        self.tree = cKDTree(xy)
        self.factor = np.asarray(factor, float)
        self.k = min(k, len(xy))

    def __call__(self, x, y):
        d, j = self.tree.query((x, y), k=self.k)
        w = 1.0 / np.maximum(d, 1e-12)
        return float((w * self.factor[j]).sum() / w.sum())


def adapt(spec, re, alpha, objective="Cd", cycles=3, level=0.5, fine_ratio=1.4, refine_frac=0.2,
          coarsen_frac=0.3, transition="off", out="../cases/adjoint_adapt", reference_level=1.4,
          log=print):
    """Goal-oriented adaptation of the triangle region; compares with uniform
    refinement (levels level * fine_ratio^k)."""
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    rows = []
    scale = None
    from aerolab.airfoil import polar as pl
    for cyc in range(cycles + 1):
        t0 = time.time()
        e = estimate(spec, re, alpha, level, (objective,), transition, size_scale=scale, log=log)
        o = e["objectives"][objective]
        pc = e["_primal"]
        m = pc["solver"].m
        rows.append({"kind": "adapted", "cycle": cyc, "cells": int(m.ncells),
                     objective: o["coarse"], "estimate": o["estimate_fine"],
                     "bl_share": o["indicator_share_bl_quads"], "time_s": time.time() - t0})
        log(f"[adapt] cycle {cyc}: {m.ncells} cells, {objective} {o['coarse']:.6f}, "
            f"corrected {o['estimate_fine']:.6f}, indicator share in BL quads "
            f"{100 * o['indicator_share_bl_quads']:.0f} %")
        if cyc == cycles:
            break
        # indicator on the coarse cells: fine contributions summed to the nearest coarse cell
        eta = np.bincount(e["_fine"]["parent"], e["_eta"][objective], m.ncells)
        tri = m.nverts == 3
        et = eta[tri]
        hi = np.quantile(et, 1 - refine_frac)
        lo = np.quantile(et, coarsen_frac)
        fac = np.ones(m.ncells)
        fac[tri & (eta >= hi)] = 0.5
        fac[tri & (eta <= lo)] = 1.4
        prev = np.ones(m.ncells) if scale is None else np.array([scale(*x) for x in m.xc])
        new = np.clip(prev * fac, 0.15, 3.0)
        scale = SizeScale(m.xc, new)
    # uniform refinement for comparison
    lv = level
    for k in range(cycles + 1):
        p = A.solve_primal(spec, re, alpha, lv, transition)
        c = pl.coefficients(p["solver"], p["walls"], p["U"])[0]
        rows.append({"kind": "uniform", "level": lv, "cells": int(p["solver"].m.ncells),
                     objective: float(c[objective])})
        log(f"[adapt] uniform level {lv:.2f}: {p['solver'].m.ncells} cells, {objective} {c[objective]:.6f}")
        lv *= fine_ratio ** 0.5
    ref = A.solve_primal(spec, re, alpha, reference_level, transition)
    cref = float(pl.coefficients(ref["solver"], ref["walls"], ref["U"])[0][objective])
    for r in rows:
        r["error"] = r[objective] - cref
    summ = {"airfoil": spec, "re": re, "alpha": alpha, "objective": objective,
            "reference_level": reference_level, "reference": cref,
            "reference_cells": int(ref["solver"].m.ncells), "rows": rows}
    (outp / "adapt.json").write_text(json.dumps(summ, indent=1, default=float))
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("estimate", "adapt"):
        p = sub.add_parser(name)
        p.add_argument("airfoil")
        p.add_argument("--re", type=float, required=True)
        p.add_argument("--alpha", type=float, default=2.0)
        p.add_argument("--level", type=float, default=0.5)
        p.add_argument("--transition", default="off")
        p.add_argument("--out")
    sub.choices["estimate"].add_argument("--check", action="store_true",
                                         help="also solve the fine mesh to measure the estimate")
    sub.choices["adapt"].add_argument("--objective", default="Cd")
    sub.choices["adapt"].add_argument("--cycles", type=int, default=3)
    a = ap.parse_args(argv)
    if a.cmd == "estimate":
        e = estimate(a.airfoil, a.re, a.alpha, a.level, transition=a.transition, check=a.check)
        if a.out:
            # the indicator map on the coarse cells (fine contributions summed per parent), for plots
            m = e["_primal"]["solver"].m
            par = e["_fine"]["parent"]
            np.savez_compressed(Path(a.out).with_name("indicator.npz"), xc=m.xc,
                                quad=np.asarray(m.nverts) == 4,
                                **{f"eta_{k}": np.bincount(par, v, m.ncells) for k, v in e["_eta"].items()})
            Path(a.out).write_text(json.dumps({k: v for k, v in e.items() if not k.startswith("_")},
                                              indent=1, default=float))
    else:
        adapt(a.airfoil, a.re, a.alpha, a.objective, a.cycles, a.level, transition=a.transition,
              out=a.out or "../cases/adjoint_adapt")


if __name__ == "__main__":
    main()
