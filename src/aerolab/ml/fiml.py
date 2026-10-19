"""
Field inversion and machine learning (FIML) for the Spalart-Allmaras model
(Parish & Duraisamy, J. Comput. Phys. 305, 2016; Singh, Medida & Duraisamy,
AIAA J. 55, 2017), with the discrete adjoint of adjoint.py.

    python src/run.py fiml invert naca0012 --re 3e6 --alpha 15 --cl 1.514 --out ../cases/fiml_n0012
    python src/run.py fiml train ../cases/fiml_n0012/a15.npz ../cases/fiml_n0012/a12.npz --out model.json
    python src/run.py fiml predict naca0012 --re 3e6 --alpha 14 --model model.json

1. Inversion. A multiplier field beta(x) on the SA production term,
       P = beta cb1 S~ nuTilde,
   is found per case by minimising
       J(beta) = sum_k ((C_k(beta) - C_k,exp) / sigma_k)^2
                 + lambda sum_i (beta_i - 1)^2
   (lift and/or wall pressure; lambda keeps beta at 1 where the data say
   nothing; unweighted per cell - weighting with the cell volume left the
   tiny near-wall cells, where the correction acts, almost unregularised and
   beta ran to its bounds there). The gradient over all cells costs one
   adjoint solve: dR/dbeta is cell-local, so psi^T dR/dbeta is a single
   complex residual evaluation (Adjoint.gradient_beta). L-BFGS-B inside
   move limits (|delta beta| <= 0.25 per outer iteration, adapted), beta in
   [0.2, 3]; every trial flow starts from the last accepted flow field.
2. Learning. beta is a correction that should depend on the local flow,
   not on position: it is regressed on local, Galilean-invariant features
       q1 = log10(1 + nuTilde/nu)          (eddy-viscosity level)
       q2 = |S| / (|S| + |Omega|)          (strain vs rotation)
       q3 = log10(Re_d) = log10(d^2 |Omega| / nu)   (wall-distance Reynolds no.)
       q4 = P / (P + D) of SA               (production share)
       q5 = tanh(d (dp/ds) / (0.5 |u|^2 + 0.5 U_inf^2))
                                            (streamwise pressure gradient over the
                                             local dynamic pressure, Clauser-type)
       q6 = r of SA (min(r, 2))             (destruction length scale)
   by a small neural network (numpy, 2 hidden layers, tanh), trained on the
   cells of all inversion cases, weighted with the cell volume and with
   extra weight where beta departs from 1.
3. Prediction. The trained network sets beta(q(x)) inside the SA model
   (updated every `every` iterations, under-relaxed) for a case NOT used in
   the training; compare with the baseline SA and the data.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from aerolab.design import adjoint as A
from aerolab.core.turbulence import CB1, CW1, CW2, CW3, CV1, CV2, CV3, KAPPA

ROOT = Path(__file__).resolve().parents[3]
TMR = ROOT / "validation" / "reference" / "tmr"


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def gregory_cl():
    """NACA 0012, Re 3e6 (Gregory & O'Reilly 1970, digitised, NASA TMR)"""
    rows = []
    for line in (TMR / "CL_Gregory_expdata.dat").read_text().splitlines():
        p = line.split()
        if len(p) == 2:
            try:
                rows.append((float(p[0]), float(p[1])))
            except ValueError:
                pass
    return np.array(rows)


def cl_exp(alpha):
    d = gregory_cl()
    d = d[d[:, 0] < 16.5]                         # pre-stall branch
    return float(np.interp(alpha, d[:, 0], d[:, 1]))


# ----------------------------------------------------------------------
# features
# ----------------------------------------------------------------------
FEATURES = ["log_chi", "strain_ratio", "log_Re_d", "prod_share", "pgrad", "r_sa"]


def features(s):
    """local features q (ncells, 6) from a SimpleSolver with SA"""
    tb = s.turb
    nu = s.nu
    nt = np.maximum(tb.nuT, 0.0)
    gu, gv = s.gU
    Om = np.abs(gv[:, 0] - gu[:, 1])
    Sxy = 0.5 * (gu[:, 1] + gv[:, 0])
    S = np.sqrt(2 * (gu[:, 0] ** 2 + gv[:, 1] ** 2 + 2 * Sxy**2))
    d = tb.d
    chi = nt / nu
    fv1 = chi**3 / (chi**3 + CV1**3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)
    St = np.maximum(Om + nt * fv2 / (KAPPA**2 * d**2), 0.3 * Om)
    r = np.minimum(nt / np.maximum(St * KAPPA**2 * d**2, 1e-30), 10.0)
    g = r + CW2 * (r**6 - r)
    fw = g * ((1 + CW3**6) / (g**6 + CW3**6)) ** (1.0 / 6.0)
    P = CB1 * St * nt
    D = CW1 * fw * (nt / d) ** 2
    U = s.U
    um2 = (U**2).sum(1)
    dpds = np.einsum('ij,ij->i', s.gradp, U) / np.sqrt(np.maximum(um2, 1e-30))
    q = np.c_[np.log10(1.0 + chi),
              S / np.maximum(S + Om, 1e-30),
              np.log10(1.0 + d**2 * Om / nu),
              P / np.maximum(P + D, 1e-30),
              np.tanh(d * dpds / (0.5 * um2 + 0.5)),
              np.minimum(r, 2.0)]
    return q


# ----------------------------------------------------------------------
# inversion
# ----------------------------------------------------------------------
class Inversion:
    def __init__(self, spec, re, alpha, level=0.5, cl=None, cp=None, sigma_cl=0.005,
                 sigma_cp=0.02, lam=1e-2, log=print, transition="off", init=None):
        """cl: target lift; cp: (x, Cp, side) arrays of measured wall
        pressure (side +1 upper, -1 lower) - either or both.
        transition: as adjoint.solve_primal ("free": e^N transition of the
        baseline, then frozen - beta corrects on top of it).
        init: warm start of the baseline flow (polar state dict)"""
        self.spec, self.re, self.alpha, self.level = spec, re, alpha, level
        self.transition = transition
        self.cl, self.cp = cl, cp
        self.sig_cl, self.sig_cp, self.lam = sigma_cl, sigma_cp, lam
        self.log = log
        t0 = time.time()
        self.pr = A.solve_primal(spec, re, alpha, level, transition, init=init)
        s = self.pr["solver"]
        self.s = s
        self.vol = s.m.vol
        self.Vref = float(np.median(self.vol))
        self.beta = np.ones(s.m.ncells)
        self.hist = []
        self.lu = None
        log(f"[fiml] {spec} alpha {alpha}: baseline primal {time.time() - t0:.0f} s, "
            f"{s.m.ncells} cells")

    def _solve(self, beta):
        """flow with this beta, always from the last ACCEPTED flow field: near
        stall a trial beta can tip the flow into a stalled solution, and
        starting the next trial from that field kept it there (hysteresis)"""
        s, vf = self.s, self.pr["vortex"]
        if getattr(self, "_ref", None) is None:
            self._ref = (s.U.copy(), s.p.copy(), s.turb.nuT.copy(), vf.gamma)
        U, p, nt, gam = self._ref
        s.set_initial(U, p, nt)
        vf.gamma = gam
        s.turb.beta = beta.copy()
        hist = []

        def cb(it, r):
            vf.update(it)
            hist.append(max(r.values()))
            if len(hist) > 300 and min(hist[-150:]) > 0.7 * min(hist[-300:-150]):
                return True
            return None
        vf.every = 1
        # pseudo-time continuation with roll-back: a trial beta near stall
        # no longer ends the inversion with a diverged flow
        s.solve_robust(tol=1e-10, max_iter=6000, callback=cb,
                       extra_state=(lambda: vf.gamma, lambda g: setattr(vf, "gamma", g)))
        vf.update(0)
        if not np.isfinite(s.U).all():
            raise FloatingPointError("trial flow diverged")
        return hist[-1] if hist else 0.0

    def _cp_weights(self, adj, X):
        """misfit of a Cp objective and its weights per wall face: the CFD Cp
        is interpolated (linearly in x, on the measured side) to every
        measured point, J = sum_j (Cp(x_j) - Cp_j)^2 / n / sigma^2 - each
        measurement counts once (face-weighted misfits were dominated by
        the dense leading-edge faces)"""
        fo = adj.res.forces(X)
        c, per = fo["faces"]["wall"]
        m = self.s.m
        idx = m.patch_faces["wall"]
        x = m.bxf[idx, 0]
        side = np.where(-m.bSf[idx, 1] >= 0, 1, -1)
        cpv = np.real(per["Cp"])
        xe, ce, se = self.cp
        W = np.zeros((len(xe), len(x)))            # Cp(x_j) = W[j] @ Cp_faces
        for sd in (1, -1):
            f = np.where(side == sd)[0]
            if len(f) < 2:
                continue
            f = f[np.argsort(x[f])]
            xs = x[f]
            for jj in np.where(se == sd)[0]:
                if xe[jj] < xs[0] or xe[jj] > xs[-1]:
                    continue
                k = min(max(np.searchsorted(xs, xe[jj]) - 1, 0), len(xs) - 2)
                t = (xe[jj] - xs[k]) / max(xs[k + 1] - xs[k], 1e-30)
                W[jj, f[k]] += 1 - t
                W[jj, f[k + 1]] += t
        ok = W.sum(1) > 0
        n = max(ok.sum(), 1)
        mis = np.where(ok, W @ cpv - ce, 0.0)
        Jcp = float((mis**2).sum() / n / self.sig_cp**2)
        w = W.T @ (2 * mis / n / self.sig_cp**2)
        return Jcp, {"wall": w}

    def fun_grad(self, beta):
        res = self._solve(beta)
        adj = A.Adjoint(self.s, self.alpha, self.pr["vortex"], self.pr["walls"])
        adj.build()
        X = adj.X
        J, g = 0.0, np.zeros(len(beta))
        fo = adj.res.forces(X)
        cl = float(np.real(fo["Cl"]))
        rec = {"Cl": cl, "residual": res}
        if self.cl is not None:
            mis = (cl - self.cl) / self.sig_cl
            J += mis**2
            adj.solve(["Cl"])
            g += 2 * mis / self.sig_cl * adj.gradient_beta("Cl")
        if self.cp is not None:
            Jcp, w = self._cp_weights(adj, X)
            J += Jcp
            gx = adj.res.objective_gradient(X, "Cp", weights=w)
            adj.psi["Cp"] = adj.lu.solve(-gx)
            g += adj.gradient_beta("Cp")
            rec["J_cp"] = Jcp
        reg = self.lam * float(((beta - 1) ** 2).sum())
        J += reg
        g += 2 * self.lam * (beta - 1)
        rec.update(J=J, reg=reg, beta_min=float(beta.min()), beta_max=float(beta.max()))
        self.hist.append(rec)
        self.log(f"  [fiml] iter {len(self.hist):3d}: J {J:.4e} (reg {reg:.2e})  Cl {cl:.4f}"
                 + (f" (target {self.cl:.4f})" if self.cl is not None else "")
                 + (f"  J_cp {rec['J_cp']:.3e}" if "J_cp" in rec else "")
                 + f"  beta [{beta.min():.2f}, {beta.max():.2f}]", flush=True)
        return J, g

    def run(self, maxiter=20, move=0.25):
        """L-BFGS-B inside move limits: beta may change by at most `move`
        per outer iteration (a free first step put beta at its bounds and
        tipped the flow into stall); the limit grows after a better point and
        halves after a worse one. maxiter = flow + adjoint solutions."""
        from scipy.optimize import minimize
        cache = {}

        def f(b):
            k = b.tobytes()
            if k not in cache:
                s, vf = self.s, self.pr["vortex"]
                try:
                    cache[k] = self.fun_grad(b)
                except (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError):
                    # no converged flow for this trial beta: a large value
                    # sends the line search back; the next trial restarts
                    # from the accepted field (_solve)
                    self.log("  [fiml] trial flow diverged - rejected", flush=True)
                    J0 = cache[beta0][0] if beta0 in cache else 1.0
                    return (1e3 * (1.0 + abs(J0)), np.zeros(len(b)), self._ref)
                cache[k] = cache[k] + ((s.U.copy(), s.p.copy(), s.turb.nuT.copy(), vf.gamma),)
            return cache[k]
        beta = self.beta.copy()
        beta0 = beta.tobytes()
        Jb = f(beta)[0]
        self._ref = f(beta)[2]
        while len(self.hist) < maxiter and move > 1e-3:
            lo = np.maximum(beta - move, 0.2)
            hi = np.minimum(beta + move, 3.0)
            r = minimize(lambda b: f(b)[0], beta, jac=lambda b: f(b)[1], method="L-BFGS-B",
                         bounds=list(zip(lo, hi)),
                         options={"maxiter": 3, "maxfun": 4, "ftol": 1e-12})
            if r.fun < Jb:
                beta, Jb = r.x.copy(), float(r.fun)
                self._ref = f(beta)[2]
                move = min(1.5 * move, 1.0)
            else:
                move *= 0.5
            self.log(f"  [fiml] outer: J {Jb:.4e}, move limit {move:.3f}", flush=True)
        self.beta = beta
        self._solve(self.beta)
        return Jb

    def save(self, path):
        s = self.s
        from aerolab.airfoil import polar as pl
        c = pl.coefficients(s, self.pr["walls"], self.pr["U"])[0]
        np.savez_compressed(path, beta=self.beta, q=features(s), vol=self.vol, xc=s.m.xc,
                            alpha=self.alpha, re=self.re, Cl=c["Cl"], Cd=c["Cd"],
                            cl_target=np.nan if self.cl is None else self.cl,
                            history=json.dumps(self.hist))


# ----------------------------------------------------------------------
# learning: a small neural network in numpy
# ----------------------------------------------------------------------
class MLP:
    def __init__(self, n_in, hidden=(20, 20), seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, 1]
        self.W = [rng.standard_normal((a, b)) * math.sqrt(1.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.mu = np.zeros(n_in)
        self.sd = np.ones(n_in)

    def _fwd(self, X):
        A_ = [(X - self.mu) / self.sd]
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            z = A_[-1] @ W + b
            A_.append(np.tanh(z) if k < len(self.W) - 1 else z)
        return A_

    def predict(self, X):
        """beta = 1 + network output"""
        return 1.0 + self._fwd(np.asarray(X, float))[-1][:, 0]

    def fit(self, X, y, w=None, epochs=3000, lr=3e-3, l2=1e-5, batch=4096, seed=0, log=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float) - 1.0
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        w = w / w.mean()
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        rng = np.random.default_rng(seed)
        params = self.W + self.b
        mom = [np.zeros_like(p) for p in params]
        vel = [np.zeros_like(p) for p in params]
        t = 0
        for ep in range(epochs):
            idx = rng.choice(len(y), min(batch, len(y)), replace=False)
            A_ = self._fwd(X[idx])
            err = (A_[-1][:, 0] - y[idx]) * w[idx]
            delta = (2 * err / len(idx))[:, None]
            gW, gb = [None] * len(self.W), [None] * len(self.b)
            for k in range(len(self.W) - 1, -1, -1):
                gW[k] = A_[k].T @ delta + l2 * self.W[k]
                gb[k] = delta.sum(0)
                if k:
                    delta = (delta @ self.W[k].T) * (1 - A_[k] ** 2)
            t += 1
            for p, g, m_, v in zip(params, gW + gb, mom, vel):
                m_ *= 0.9; m_ += 0.1 * g
                v *= 0.999; v += 0.001 * g * g
                p -= lr * (m_ / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
            if log and ep % 500 == 0:
                pr = self.predict(X) - 1.0
                log(f"    epoch {ep:5d}: weighted MSE {float((w * (pr - y) ** 2).mean()):.3e}")
        return self

    def to_json(self):
        return {"W": [w.tolist() for w in self.W], "b": [b.tolist() for b in self.b],
                "mu": self.mu.tolist(), "sd": self.sd.tolist(), "features": FEATURES}

    @classmethod
    def from_json(cls, d):
        m = cls(len(d["mu"]), hidden=tuple(len(b) for b in d["b"][:-1]))
        m.W = [np.array(w) for w in d["W"]]
        m.b = [np.array(b) for b in d["b"]]
        m.mu, m.sd = np.array(d["mu"]), np.array(d["sd"])
        return m


def train(files, hidden=(20, 20), epochs=4000, log=print):
    Q, Y, Wt = [], [], []
    for f in files:
        d = np.load(f)
        q, b, v = d["q"], d["beta"], d["vol"]
        Q.append(q); Y.append(b)
        Wt.append(np.sqrt(v / np.median(v)) * (1.0 + 20.0 * np.abs(b - 1)))
    X, y, w = np.vstack(Q), np.concatenate(Y), np.concatenate(Wt)
    log(f"[fiml] training on {len(y)} cells from {len(files)} inversions, beta range "
        f"[{y.min():.2f}, {y.max():.2f}]")
    net = MLP(X.shape[1], hidden).fit(X, y, w, epochs=epochs, log=log)
    pr = net.predict(X)
    log(f"[fiml] fit: weighted RMS error {math.sqrt(float((w * (pr - y) ** 2).sum() / w.sum())):.4f}")
    return net


def solve_augmented(spec, re, alpha, net, level=0.5, every=20, relax=0.5, max_iter=8000,
                    init=None, transition="off", primal=None):
    """SA with beta = net(q(x)) updated every `every` iterations.
    primal: a converged baseline (solve_primal; it is modified in place)"""
    pr = primal or A.solve_primal(spec, re, alpha, level, transition, init=init)
    s, vf = pr["solver"], pr["vortex"]
    # the learned beta from the baseline flow, applied BEFORE solving (the
    # baseline is converged below the tolerance: a solve that only set beta
    # in its callback stopped at the first iteration with beta = 1)
    beta = np.ones(s.m.ncells)
    beta = (1 - relax) * beta + relax * np.clip(net.predict(features(s)), 0.05, 5.0)
    s.turb.beta = beta
    hist = []

    def cb(it, r):
        nonlocal beta
        vf.update(it)
        if it % every == 0:
            bnew = np.clip(net.predict(features(s)), 0.05, 5.0)
            beta = (1 - relax) * beta + relax * bnew
            s.turb.beta = beta
        hist.append(max(r.values()))
        if it > 2000 and min(hist[-300:]) > 0.7 * min(hist[-600:-300]):
            return True
        return None
    vf.every = 1
    ex = (lambda: vf.gamma, lambda g: setattr(vf, "gamma", g))
    s.solve_robust(tol=0.0, max_iter=min(1000, max_iter), callback=cb, extra_state=ex)  # beta settles
    s.solve_robust(tol=1e-9, max_iter=max_iter, callback=cb, extra_state=ex)
    from aerolab.airfoil import polar as pl
    c = pl.coefficients(s, pr["walls"], pr["U"])[0]
    return c, pr, hist


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("invert")
    p.add_argument("airfoil")
    p.add_argument("--re", type=float, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--cl", type=float, help="target Cl (default: Gregory's data for NACA 0012)")
    p.add_argument("--level", type=float, default=0.5)
    p.add_argument("--lam", type=float, default=1e-2)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("files", nargs="+")
    t.add_argument("--out", required=True)
    q = sub.add_parser("predict")
    q.add_argument("airfoil")
    q.add_argument("--re", type=float, required=True)
    q.add_argument("--alpha", type=float, required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--level", type=float, default=0.5)
    a = ap.parse_args(argv)
    if a.cmd == "invert":
        cl = a.cl if a.cl is not None else cl_exp(a.alpha)
        inv = Inversion(a.airfoil, a.re, a.alpha, a.level, cl=cl, lam=a.lam)
        inv.run(a.iters)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        inv.save(out / f"a{a.alpha:g}.npz")
    elif a.cmd == "train":
        net = train(a.files)
        Path(a.out).write_text(json.dumps(net.to_json()))
    else:
        net = MLP.from_json(json.loads(Path(a.model).read_text()))
        from aerolab.airfoil import polar as pl
        base = A.solve_primal(a.airfoil, a.re, a.alpha, a.level)
        cb = pl.coefficients(base["solver"], base["walls"], base["U"])[0]
        c, _, _ = solve_augmented(a.airfoil, a.re, a.alpha, net, a.level)
        rec = {"alpha": a.alpha, "re": a.re, "Cl_SA": cb["Cl"], "Cd_SA": cb["Cd"],
               "Cl_FIML": c["Cl"], "Cd_FIML": c["Cd"], "Cl_exp": cl_exp(a.alpha)}
        print(f"alpha {a.alpha:g}: SA Cl {cb['Cl']:.4f}, augmented SA Cl {c['Cl']:.4f} "
              f"Cd {c['Cd']:.5f}  (data {rec['Cl_exp']:.4f})")
        f = Path(a.model).parent / "predictions.json"
        old = json.loads(f.read_text()) if f.exists() else []
        old = [r for r in old if r["alpha"] != a.alpha] + [rec]
        f.write_text(json.dumps(sorted(old, key=lambda r: r["alpha"]), indent=1))


if __name__ == "__main__":
    main()
