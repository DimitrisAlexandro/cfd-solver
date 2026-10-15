"""
Gradient-based airfoil optimisation with adjoint gradients.

    python design_adjoint.py --start naca0012 --re 3e6 --cl 0.5 --t-min 0.12 --iters 12
    python design_adjoint.py --start naca2412 --re 3e6 --cl 0.6 --cm-min -0.06 --level 0.5

Design variables: all CST (Kulfan) coefficients of both surfaces (order 8:
18 shape variables) and the angle of attack. One flow solution + one adjoint
factorisation per design iteration gives the gradients of Cd, Cl and Cm with
respect to all 19 variables (adjoint.py), so the cost does not grow with the
number of shape variables - the derivative-free optimizer (design_optimize.py)
needs O(100) flow solutions for 8 variables.

Problem (SLSQP, Kraft 1988, inside sequential move limits):
    minimise    Cd
    subject to  Cl = Cl_target
                t/c >= t_min            (CST thickness, analytic)
                Cm >= Cm_min            (optional)
                upper surface above lower everywhere
                |dA| <= bound           (coefficient changes)

Mesh: the start shape (its CST fit) is meshed once at `level`; every design
is that mesh deformed (inverse distance weighting of the wall displacement,
adjoint.IDWDeformation). The deformation is linear in the CST coefficients,
so the objective is a smooth function of the design variables and the
adjoint shape gradient is exactly its derivative (re-meshing every design
made Cd jump by ~3 % between neighbouring designs: mesh noise that stalls a
gradient method). Transition locations are frozen in the adjoint
(--transition free: the gradient is then that of the frozen-transition
problem). The best design is re-meshed and re-run on the validated mesh
(level 1) at the target Cl.
Output: design.json (iteration history, gradients), best.dat, design.png.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import adjoint as adj_mod
import airfoil_geom as ag
import cst


class Problem:
    def __init__(self, start, re, cl, t_min=None, cm_min=None, level=0.5, transition="off",
                 order=8, bound=0.05, out="../cases/design_adjoint", log=print):
        self.re, self.cl, self.level, self.transition = re, cl, level, transition
        self.order = order
        self.au0, self.al0, self.fit_err = cst.fit(ag.sharp_outline(start), order)
        t0, *_ = cst.thickness(self.au0, self.al0)
        self.t0 = t0
        self.t_min = t0 * 0.995 if t_min is None else t_min
        self.cm_min = cm_min
        self.bound = bound
        self.k = order + 1
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.cache = {}
        self.history = []
        self.state = None
        self.alpha0 = cl / 0.11
        self.scale_v = 0.01            # design variable units: 0.01 per CST coefficient
        self.t_start = time.time()
        # baseline mesh of the start shape (CST fit) and the deformation modes
        self.start_file = self.out / "start_cst.dat"
        cst.write_dat(self.start_file, self.au0, self.al0, "start (CST fit)")
        s, fvm, mesh, U, turb, info = adj_mod.setup_case(str(self.start_file), re, self.alpha0,
                                                         level)
        self.base = {"mesh": mesh, "info": info}
        self.P0 = np.asarray(fvm.points, float)
        defo = adj_mod.IDWDeformation(fvm, info["walls"])
        self.dP = adj_mod.cst_wall_velocities(defo, order)
        self.topo = adj_mod.Topology(fvm)

    # x = [dAu (k), dAl (k), alpha / 1 deg] (CST changes in units of 0.01)
    def coeffs(self, x):
        d = np.asarray(x[:2 * self.k]) * self.scale_v
        return self.au0 + d[:self.k], self.al0 + d[self.k:]

    def evaluate(self, x):
        key = np.asarray(x, float).round(12).tobytes()
        if key in self.cache:
            return self.cache[key]
        au, al = self.coeffs(x)
        alpha = float(x[-1])
        t0 = time.time()
        P = self.P0 + sum(xi * self.scale_v * d for xi, d in zip(x[:2 * self.k], self.dP))
        # a valid deformed mesh (every cell keeps a positive area) and a
        # converged flow; otherwise a penalised value so that SLSQP's line
        # search steps back (warm start first, then a cold start)
        g = None
        if _areas_positive(self.topo, P):
            for init in (self.state, None):
                try:
                    g = adj_mod.gradients(str(self.start_file), self.re, alpha, self.level,
                                          self.transition, shape=True, order=self.order,
                                          init=init, dP=self.dP,
                                          base={**self.base, "points": P})
                    if np.isfinite(g["Cd"]) and g["primal_residual"] < 1e-6:
                        break
                except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError):
                    pass
                g = None
        if g is None:
            return self._failed(x, au, al, alpha, t0)
        pr = g["_primal"]
        s = pr["solver"]
        self.state = {"xc": s.m.xc.copy(), "U": s.U.copy(), "p": s.p.copy(),
                      "nuT": None if s.turb is None else s.turb.nuT.copy()}
        r = {"Cl": g["Cl"], "Cd": g["Cd"], "Cm": g["Cm"]}
        grad = {}
        for k in ("Cl", "Cd", "Cm"):
            gc = np.array(g["d_dcst"][k]) * self.scale_v
            grad[k] = np.r_[gc, g["d_dalpha"][k]]
        rec = {"iter": len(self.history), "x": list(map(float, x)), "alpha": alpha,
               **{k: float(v) for k, v in r.items()},
               "t": cst.thickness(au, al)[0], "time_s": round(time.time() - t0, 1),
               "grad": {k: v.tolist() for k, v in grad.items()},
               "primal_residual": pr["residual"]}
        self.history.append(rec)
        self.log(f"  eval {len(self.history):3d}: alpha {alpha:6.3f}  Cl {r['Cl']:.4f}  "
                 f"Cd {r['Cd']:.6f}  Cm {r['Cm']:+.4f}  t/c {rec['t']:.4f}  "
                 f"({rec['time_s']:.0f} s, total {(time.time() - self.t_start) / 60:.1f} min)",
                 flush=True)
        self.cache[key] = (r, grad)
        self._save()
        return r, grad

    def _failed(self, x, au, al, alpha, t0):
        """no valid flow solution: the last good point's values, Cd raised"""
        good = [h for h in self.history if h.get("ok", True)]
        last = good[-1] if good else None
        r = {"Cl": last["Cl"] if last else 0.0, "Cd": (last["Cd"] if last else 0.01) * 2.0,
             "Cm": last["Cm"] if last else 0.0}
        grad = {k: np.array(last["grad"][k]) if last else np.zeros(2 * self.k + 1)
                for k in ("Cl", "Cd", "Cm")}
        rec = {"iter": len(self.history), "x": list(map(float, x)), "alpha": alpha, "ok": False,
               **r, "t": cst.thickness(au, al)[0], "time_s": round(time.time() - t0, 1),
               "grad": {k: v.tolist() for k, v in grad.items()}}
        self.history.append(rec)
        self.log(f"  eval {len(self.history):3d}: alpha {alpha:6.3f}  no converged flow "
                 f"(diverged or inverted mesh) - penalised", flush=True)
        self.cache[np.asarray(x, float).round(12).tobytes()] = (r, grad)
        return r, grad

    # thickness constraints (cheap, finite differences on the CST outline)
    def thick(self, x):
        au, al = self.coeffs(x)
        t, _, tmin = cst.thickness(au, al)
        return np.array([t - self.t_min, tmin + 1e-4])

    def thick_jac(self, x, h=1e-6):
        J = np.zeros((2, len(x)))
        f0 = self.thick(x)
        for i in range(2 * self.k):
            xx = np.array(x, float)
            xx[i] += h
            J[:, i] = (self.thick(xx) - f0) / h
        return J

    def _save(self):
        (self.out / "design.json").write_text(json.dumps(
            {"re": self.re, "cl": self.cl, "t_min": self.t_min, "cm_min": self.cm_min,
             "level": self.level, "transition": self.transition, "order": self.order,
             "au0": self.au0.tolist(), "al0": self.al0.tolist(), "history": self.history},
            indent=1, default=float))


def optimise(start, re, cl, t_min=None, cm_min=None, iters=12, level=0.5, transition="off",
             order=8, bound=0.03, out="../cases/design_adjoint", verify=True, move=0.0015,
             log=print):
    from scipy.optimize import minimize
    pb = Problem(start, re, cl, t_min, cm_min, level, transition, order, bound, out, log)
    nx = 2 * pb.k + 1
    x0 = np.zeros(nx)
    x0[-1] = pb.alpha0
    log(f"[design] {start}: CST order {order} ({2 * pb.k} shape variables + alpha), "
        f"fit error {pb.fit_err:.1e}; Re {re:.3g}, Cl {cl}, t/c >= {pb.t_min:.4f}"
        + (f", Cm >= {cm_min}" if cm_min is not None else ""))
    # first: angle for the target lift on the start shape (Newton with dCl/dalpha)
    for _ in range(4):
        r, g = pb.evaluate(x0)
        if abs(r["Cl"] - cl) < 2e-4:
            break
        x0[-1] += (cl - r["Cl"]) / max(g["Cl"][-1], 0.02)
    start_rec = dict(pb.history[-1])
    sc = 1e4                                     # objective in drag counts
    cons = [{"type": "eq", "fun": lambda x: (pb.evaluate(x)[0]["Cl"] - cl) * 10,
             "jac": lambda x: pb.evaluate(x)[1]["Cl"] * 10},
            {"type": "ineq", "fun": lambda x: pb.thick(x) * 100,
             "jac": lambda x: pb.thick_jac(x) * 100}]
    if cm_min is not None:
        cons.append({"type": "ineq", "fun": lambda x: (pb.evaluate(x)[0]["Cm"] - cm_min) * 10,
                     "jac": lambda x: pb.evaluate(x)[1]["Cm"] * 10})
    # sequential move limits around the current design (SLSQP has no trust
    # region; its first quasi-Newton steps changed the nose radius by ~30 %):
    # each variable may move the wall by at most `move` (chord) per outer
    # iteration; the limit shrinks after a failed or worse step, grows after
    # a good one
    disp = np.array([np.abs(d).max() for d in pb.dP]) * pb.scale_v    # wall motion per unit
    lim_shape = bound / pb.scale_v
    x = x0.copy()
    cur = pb.evaluate(x)
    nit, msg = 0, "iteration limit"
    for outer in range(iters):
        dlt = np.r_[move / np.maximum(disp, 1e-12), 0.5 * move / 0.0015]
        lo = np.maximum(x - dlt, np.r_[np.full(2 * pb.k, -lim_shape), -5.0])
        hi = np.minimum(x + dlt, np.r_[np.full(2 * pb.k, lim_shape), 15.0])
        res = minimize(lambda z: pb.evaluate(z)[0]["Cd"] * sc, x,
                       jac=lambda z: pb.evaluate(z)[1]["Cd"] * sc,
                       method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                       options={"maxiter": 3, "ftol": 1e-6, "disp": False})
        nit += res.nit
        ok = [h for h in pb.history if h.get("ok", True) and abs(h["Cl"] - cl) < 2e-3
              and h["t"] >= pb.t_min - 1e-4 and (cm_min is None or h["Cm"] >= cm_min - 1e-3)]
        best_h = min(ok, key=lambda h: h["Cd"]) if ok else None
        failed = any(not h.get("ok", True) for h in pb.history[-4:])
        improved = best_h is not None and best_h["Cd"] < cur[0]["Cd"] - 1e-7
        if improved:
            x = np.array(best_h["x"])
            cur = pb.evaluate(x)
            move = move * (1.0 if failed else 1.5)
        else:
            move *= 0.5
        log(f"[design] outer {outer + 1}: best Cd {cur[0]['Cd']:.6f}, move limit {move:.1e} c"
            + (" (a step failed)" if failed else ""))
        if move < 1e-5:
            msg = "move limit below 1e-5 c (converged)"
            break
    res.message, res.nit, res.success = msg, nit, msg.endswith("(converged)")
    # best feasible design seen
    feas = [h for h in pb.history if h.get("ok", True) and abs(h["Cl"] - cl) < 3e-3 and h["t"] >= pb.t_min - 1e-4
            and (cm_min is None or h["Cm"] >= cm_min - 1e-3)]
    best = min(feas, key=lambda h: h["Cd"]) if feas else pb.history[-1]
    au, al = pb.coeffs(best["x"])
    cst.write_dat(pb.out / "best.dat", au, al, f"adjoint-optimised from {Path(str(start)).stem}")
    summ = {"start": str(start), "re": re, "cl": cl, "t_min": pb.t_min, "cm_min": cm_min,
            "level": level, "transition": transition, "order": order,
            "slsqp": {"success": bool(res.success), "message": str(res.message),
                      "nit": int(res.nit)}, "move_limit_final": move,
            "evaluations": len(pb.history), "start_point": start_rec, "best": best,
            "history": pb.history, "time_min": (time.time() - pb.t_start) / 60}
    log(f"[design] SLSQP: {res.message} ({res.nit} iterations, {len(pb.history)} flow + adjoint "
        f"solutions). Cd {start_rec['Cd']:.6f} -> {best['Cd']:.6f} "
        f"({100 * (best['Cd'] / start_rec['Cd'] - 1):+.1f} %) at Cl {best['Cl']:.4f}")
    if verify:
        import design_optimize as dopt
        log("[design] verification on mesh level 1 at the target Cl (start and best)")
        ver = {}
        for tag, f in (("start", pb.out / "start_cst.dat"), ("best", pb.out / "best.dat")):
            shp = _FileShape(f)
            r, _ = dopt.evaluate(shp, None, re, [cl], 1.0, transition, pb.out, None, cm_min,
                                 None, log)
            ver[tag] = r
            log(f"  {tag}: Cd {r.get('Cd', float('nan')):.6f}")
        summ["verification_level1"] = ver
    (pb.out / "design.json").write_text(json.dumps(summ, indent=1, default=float))
    plots(summ, pb, pb.out)
    return summ


def _areas_positive(topo, P):
    for k, (idx, C) in topo.groups.items():
        x, y = P[C, 0], P[C, 1]
        A = 0.5 * (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(1)
        if (A <= 0).any():
            return False
    return True


class _FileShape:
    """design_optimize.evaluate needs coeffs()/write(): wrap a fixed .dat"""

    def __init__(self, path):
        self.path = Path(path)
        self.au, self.al, _ = cst.fit(ag.sharp_outline(str(path)), 8)

    def coeffs(self, v):
        return self.au, self.al

    def write(self, v, path, name):
        return self.path


def plots(summ, pb, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H = summ["history"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    it = np.arange(1, len(H) + 1)
    ax[0].plot(it, [h["Cd"] * 1e4 for h in H], "o-", ms=4)
    ax[0].set_xlabel("flow + adjoint solution"); ax[0].set_ylabel("Cd (counts)")
    a2 = ax[0].twinx()
    a2.plot(it, [h["Cl"] for h in H], "s--", color="C1", ms=3)
    a2.axhline(summ["cl"], color="C1", lw=0.6)
    a2.set_ylabel("Cl", color="C1")
    ax[0].grid(alpha=.3)
    o0 = cst.outline(pb.au0, pb.al0)
    ob = cst.outline(*pb.coeffs(summ["best"]["x"]))
    ax[1].plot(o0[:, 0], o0[:, 1], "k--", label="start (CST fit)")
    ax[1].plot(ob[:, 0], ob[:, 1], "r-", label="optimised")
    ax[1].set_aspect("equal"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    v = summ.get("verification_level1", {})
    ax[1].set_title("  ".join(f"{k}: Cd {r.get('Cd', float('nan')):.5f}" for k, r in v.items())
                    or "shape", fontsize=9)
    g = np.array(H[0]["grad"]["Cd"][:-1]) / pb.scale_v
    k = pb.k
    ax[2].bar(np.arange(k) - 0.2, g[:k], 0.4, label="upper")
    ax[2].bar(np.arange(k) + 0.2, g[k:], 0.4, label="lower")
    ax[2].set_xlabel("CST coefficient"); ax[2].set_ylabel("dCd/dA (start shape)")
    ax[2].legend(); ax[2].grid(alpha=.3)
    ax[2].set_title("adjoint shape gradient")
    fig.tight_layout(); fig.savefig(Path(out) / "design.png", dpi=120); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="naca0012")
    ap.add_argument("--re", type=float, default=3e6)
    ap.add_argument("--cl", type=float, default=0.5)
    ap.add_argument("--t-min", type=float)
    ap.add_argument("--cm-min", type=float)
    ap.add_argument("--iters", type=int, default=12, help="outer (move-limit) iterations")
    ap.add_argument("--move", type=float, default=0.0015,
                    help="initial move limit: wall displacement per outer iteration (chord)")
    ap.add_argument("--level", type=float, default=0.5)
    ap.add_argument("--transition", default="off")
    ap.add_argument("--order", type=int, default=8)
    ap.add_argument("--bound", type=float, default=0.03)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--out", default="../cases/design_adjoint")
    a = ap.parse_args(argv)
    optimise(a.start, a.re, a.cl, a.t_min, a.cm_min, a.iters, a.level, a.transition, a.order,
             a.bound, a.out, verify=not a.no_verify, move=a.move)


if __name__ == "__main__":
    main()
