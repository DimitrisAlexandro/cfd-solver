"""
Gaussian-process surrogate of the section polars, with uncertainty: every
RANS polar computed so far becomes training data for an instant estimate of
Cl, Cd and Cm (mean and standard deviation) of any airfoil / Re / angle, and
the surrogate says where its next RANS run would teach it the most.

    python surrogate.py train                         (polar library -> model)
    python surrogate.py predict naca2412 --re 3e6 --alpha 0:12:2
    python surrogate.py suggest naca4415 --re 2e6      (where to run RANS next)
    python surrogate.py validate                      (leave-one-polar-out)

Inputs (scaled to unit variance):
    alpha, log10 Re, t/c, x/c of max thickness, max camber, x/c of max
    camber, leading-edge radius parameter (first CST coefficients), flap
    chord fraction x deflection, flap deflection, free transition (1/0)
Outputs, each its own GP: Cl, log Cd, Cm.
Kernel: ARD squared exponential + white noise; hyperparameters by maximum
marginal likelihood (L-BFGS-B, several restarts) - Rasmussen & Williams
(2006), algorithm 2.1. The prediction variance includes the noise term, so
the +-2 sigma band is a prediction interval for a new RANS result; the
standard deviations are then calibrated by leave-one-polar-out validation
(a GP fitted to a handful of polars is over-confident away from them).
Training data: data/wing_polars/*.json, cases/**/polar.json and the saved
validation polars (validation/transition_*.json); only converged points.
Use: a fast first look at an airfoil before running RANS, a cheap objective
for design studies, and active learning (suggest = the angle / Re of the
largest predictive standard deviation).
"""

import argparse
import json
import math
import re as _re
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

import airfoil_geom as ag
import cst

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data" / "surrogate_model.json"
FEATURES = ["alpha", "log10_re", "t", "x_t", "camber", "x_camber", "le", "flap_cf_x_delta",
            "flap_delta", "free_transition"]


# ----------------------------------------------------------------------
# geometry features
# ----------------------------------------------------------------------
_GEO = {}


def geometry(spec):
    base, flap = ag.parse_flap(spec)
    if base not in _GEO:
        o = ag.sharp_outline(base)
        au, al, _ = cst.fit(o, 8)
        x = cst.x_cosine(400)
        yu, yl = cst.surface(x, au), cst.surface(x, al)
        t, c = yu - yl, 0.5 * (yu + yl)
        i, j = int(np.argmax(t)), int(np.argmax(np.abs(c)))
        _GEO[base] = [float(t[i]), float(x[i]), float(c[j]), float(x[j]),
                      float(au[0] - al[0])]
    cf, d = flap if flap else (0.0, 0.0)
    return _GEO[base] + [cf * d / 10.0, d / 10.0]


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def _rows(spec, re, results, free):
    out = []
    for r in results:
        if not str(r.get("status", "converged")).startswith("converged"):
            continue
        if not all(np.isfinite([r["Cl"], r["Cd"], r["Cm"]])) or r["Cd"] <= 0:
            continue
        out.append({"spec": spec, "re": float(re), "alpha": float(r["alpha"]), "free": free,
                    "Cl": float(r["Cl"]), "Cd": float(r["Cd"]), "Cm": float(r["Cm"])})
    return out


def collect():
    rows, sources = [], []
    for f in sorted((ROOT / "data" / "wing_polars").glob("*.json")):
        m = _re.match(r"(.+)_Re([0-9.e+]+)_M([0-9.]+)_L([0-9.]+)_(\w+)\.json", f.name)
        if not m or float(m.group(3)) > 0.3:
            continue
        rows += _rows(m.group(1), float(m.group(2)), json.loads(f.read_text()),
                      1.0 if m.group(5) == "free" else 0.0)
        sources.append(f.name)
    for f in sorted(list((ROOT / "cases").rglob("polar.json")) +
                    list((ROOT / "validation").glob("transition_*.json"))):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        st = d.get("settings", {})
        if st.get("mach", 0.0) and st["mach"] > 0.3 or d.get("info", {}).get("multi"):
            continue
        spec = st.get("airfoil") or d.get("info", {}).get("source")
        if not spec:
            continue
        free = 0.0 if (st.get("transition_mode") == "off" or
                       "fully turbulent" in str(d["results"][0].get("transition", ""))) else 1.0
        try:
            geometry(spec)
        except Exception:
            continue
        rows += _rows(spec, d["re"], d["results"], free)
        sources.append(str(f.relative_to(ROOT)))
    return rows, sources


def design_matrix(rows):
    X = np.array([[r["alpha"], math.log10(r["re"]), *geometry(r["spec"]), r["free"]] for r in rows])
    Y = {"Cl": np.array([r["Cl"] for r in rows]), "logCd": np.log(np.array([r["Cd"] for r in rows])),
         "Cm": np.array([r["Cm"] for r in rows])}
    return X, Y


# ----------------------------------------------------------------------
# Gaussian process (ARD squared exponential + noise)
# ----------------------------------------------------------------------
class GP:
    def __init__(self):
        self.theta = None

    @staticmethod
    def _k(A, B, ell, sf2):
        d = (A[:, None, :] - B[None, :, :]) / ell
        return sf2 * np.exp(-0.5 * (d**2).sum(2))

    def _nll(self, th, X, y):
        D = X.shape[1]
        ell, sf2, sn2 = np.exp(th[:D]), np.exp(th[D]), np.exp(th[D + 1])
        K = self._k(X, X, ell, sf2) + (sn2 + 1e-8) * np.eye(len(y))
        try:
            L = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        a = cho_solve(L, y)
        return 0.5 * y @ a + np.log(np.diag(L[0])).sum() + 0.5 * len(y) * math.log(2 * math.pi)

    def fit(self, X, y, restarts=4, seed=0):
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-9
        self.my, self.sy = y.mean(), y.std() + 1e-12
        Xs, ys = (X - self.mx) / self.sx, (y - self.my) / self.sy
        D = X.shape[1]
        rng = np.random.default_rng(seed)
        best = None
        for k in range(restarts):
            th0 = np.r_[np.log(rng.uniform(0.5, 3.0, D)), 0.0, math.log(1e-3 if k else 1e-2)]
            r = minimize(self._nll, th0, args=(Xs, ys), method="L-BFGS-B",
                         bounds=[(-3, 5)] * D + [(-4, 4), (-14, 1)])
            if best is None or r.fun < best.fun:
                best = r
        self.theta = best.x
        self.nll = float(best.fun)
        self.Xs, self.ys = Xs, ys
        ell, sf2, sn2 = self._hyp()
        K = self._k(Xs, Xs, ell, sf2) + (sn2 + 1e-8) * np.eye(len(ys))
        self.L = cho_factor(K, lower=True)
        self.alpha_ = cho_solve(self.L, ys)
        return self

    def _hyp(self):
        D = self.Xs.shape[1]
        return np.exp(self.theta[:D]), math.exp(self.theta[D]), math.exp(self.theta[D + 1])

    def predict(self, X):
        Xs = (np.atleast_2d(X) - self.mx) / self.sx
        ell, sf2, sn2 = self._hyp()
        Ks = self._k(Xs, self.Xs, ell, sf2)
        mu = Ks @ self.alpha_
        v = cho_solve(self.L, Ks.T)
        var = np.maximum(sf2 + sn2 - np.einsum('ij,ji->i', Ks, v), 1e-16)
        return mu * self.sy + self.my, np.sqrt(var) * self.sy

    def to_json(self):
        return {"theta": self.theta.tolist(), "mx": self.mx.tolist(), "sx": self.sx.tolist(),
                "my": self.my, "sy": self.sy, "Xs": self.Xs.tolist(), "ys": self.ys.tolist()}

    @classmethod
    def from_json(cls, d):
        g = cls()
        g.theta = np.array(d["theta"]); g.mx, g.sx = np.array(d["mx"]), np.array(d["sx"])
        g.my, g.sy = d["my"], d["sy"]
        g.Xs, g.ys = np.array(d["Xs"]), np.array(d["ys"])
        ell, sf2, sn2 = g._hyp()
        g.L = cho_factor(g._k(g.Xs, g.Xs, ell, sf2) + (sn2 + 1e-8) * np.eye(len(g.ys)), lower=True)
        g.alpha_ = cho_solve(g.L, g.ys)
        return g


class Surrogate:
    def __init__(self, gps=None, sources=None, n=0, calib=None):
        self.gps, self.sources, self.n = gps or {}, sources or [], n
        # sd multipliers from leave-one-polar-out validation (calibrate())
        self.calib = calib or {"Cl": 1.0, "logCd": 1.0, "Cm": 1.0}

    @classmethod
    def train(cls, rows=None, sources=None):
        if rows is None:
            rows, sources = collect()
        X, Y = design_matrix(rows)
        return cls({k: GP().fit(X, y) for k, y in Y.items()}, sources, len(rows))

    def predict(self, spec, re, alphas, free=1.0):
        X = np.array([[a, math.log10(re), *geometry(spec), free] for a in np.atleast_1d(alphas)])
        out = {}
        for k, g in self.gps.items():
            mu, sd = g.predict(X)
            sd = sd * self.calib.get(k, 1.0)
            if k == "logCd":
                out["Cd"], out["Cd_sd"] = np.exp(mu), np.exp(mu) * sd       # lognormal, first order
            else:
                out[k], out[k + "_sd"] = mu, sd
        out["alpha"] = np.atleast_1d(alphas)
        return out

    def suggest(self, spec, re, alphas=np.arange(-4, 16.5, 1.0), free=1.0):
        """the angle where the prediction is least certain (relative to the
        quantity's scale): the most informative next RANS run"""
        p = self.predict(spec, re, alphas, free)
        score = p["Cl_sd"] / 0.05 + p["Cd_sd"] / np.maximum(0.1 * p["Cd"], 1e-4)
        i = int(np.argmax(score))
        return float(alphas[i]), p, score

    def save(self, path=MODEL):
        Path(path).write_text(json.dumps({"n": self.n, "sources": self.sources, "features": FEATURES,
                                          "calibration": self.calib,
                                          "gps": {k: g.to_json() for k, g in self.gps.items()}}))

    @classmethod
    def load(cls, path=MODEL):
        d = json.loads(Path(path).read_text())
        return cls({k: GP.from_json(v) for k, v in d["gps"].items()}, d["sources"], d["n"],
                   d.get("calibration"))

    def calibrate(self, rows, target=0.95):
        """sd multipliers such that, leaving each polar out in turn, the
        +-2 sd band covers `target` of its points (a GP fitted to a few
        polars is over-confident outside them: uncalibrated coverage was
        36-100 %)"""
        keys = sorted({(r["spec"], r["re"], r["free"]) for r in rows})
        z = {"Cl": [], "logCd": [], "Cm": []}
        for key in keys:
            tr = [r for r in rows if (r["spec"], r["re"], r["free"]) != key]
            te = [r for r in rows if (r["spec"], r["re"], r["free"]) == key]
            if len(te) < 3 or len(tr) < 10:
                continue
            Xt, Yt = design_matrix(tr)
            Xe, Ye = design_matrix(te)
            for k in z:
                mu, sd = GP().fit(Xt, Yt[k], restarts=2).predict(Xe)
                z[k] += list(np.abs(Ye[k] - mu) / sd)
        for k, v in z.items():
            if v:
                # the factor that brings the target quantile of |error|/sd to 2
                self.calib[k] = max(1.0, float(np.quantile(v, target)) / 2.0)
        return self.calib


def validate(log=print):
    """leave one polar (airfoil, Re, transition) out: errors and coverage of
    the +-2 sigma band"""
    rows, sources = collect()
    keys = sorted({(r["spec"], r["re"], r["free"]) for r in rows})
    res = []
    for key in keys:
        train_rows = [r for r in rows if (r["spec"], r["re"], r["free"]) != key]
        test = [r for r in rows if (r["spec"], r["re"], r["free"]) == key]
        if len(test) < 3 or len(train_rows) < 10:
            continue
        sg = Surrogate.train(train_rows, [])
        p = sg.predict(key[0], key[1], [r["alpha"] for r in test], key[2])
        cl, cd = np.array([r["Cl"] for r in test]), np.array([r["Cd"] for r in test])
        al = np.array([r["alpha"] for r in test])
        pre = al <= 12.0
        e = {"polar": f"{key[0]} Re {key[1]:.3g} {'free' if key[2] else 'turbulent'}",
             "n": len(test), "rms_Cl": float(np.sqrt(np.mean((p["Cl"] - cl) ** 2))),
             "rms_Cl_prestall": float(np.sqrt(np.mean((p["Cl"][pre] - cl[pre]) ** 2))) if pre.any() else None,
             "rms_Cd_rel_prestall": float(np.sqrt(np.mean(((p["Cd"][pre] - cd[pre]) / cd[pre]) ** 2)))
             if pre.any() else None,
             "alpha_max": float(al.max()),
             "rms_Cd_rel": float(np.sqrt(np.mean(((p["Cd"] - cd) / cd) ** 2))),
             "cover_Cl_2sd": float(np.mean(np.abs(p["Cl"] - cl) <= 2 * p["Cl_sd"])),
             "cover_Cd_2sd": float(np.mean(np.abs(p["Cd"] - cd) <= 2 * p["Cd_sd"]))}
        res.append(e)
        log(f"  {e['polar']:45s} n {e['n']:3d}  rms Cl {e['rms_Cl']:.3f} (alpha <= 12: "
            f"{e['rms_Cl_prestall']:.3f})  rms Cd {100 * e['rms_Cd_rel']:5.1f} %"
            f"  within 2 sd: Cl {100 * e['cover_Cl_2sd']:.0f} %, Cd {100 * e['cover_Cd_2sd']:.0f} %")
    return res


def main(argv=None):
    from airfoil import parse_alpha
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    sub.add_parser("validate")
    for name in ("predict", "suggest"):
        p = sub.add_parser(name)
        p.add_argument("airfoil")
        p.add_argument("--re", type=float, required=True)
        p.add_argument("--alpha", default="-4:14:2")
        p.add_argument("--turbulent", action="store_true", help="fully turbulent instead of free")
    argv = list(argv if argv is not None else __import__("sys").argv[1:])
    for i in range(len(argv) - 1):
        if argv[i] == "--alpha" and argv[i + 1][:1] == "-":
            argv[i:i + 2] = [f"--alpha={argv[i + 1]}", ""]
    a = ap.parse_args([x for x in argv if x != ""])
    if a.cmd == "train":
        rows, sources = collect()
        sg = Surrogate.train(rows, sources)
        cal = sg.calibrate(rows)
        sg.save()
        print(f"[surrogate] {sg.n} points from {len(sg.sources)} polars -> {MODEL}; "
              f"sd calibration " + ", ".join(f"{k} x{v:.2f}" for k, v in cal.items()))
        return
    if a.cmd == "validate":
        validate()
        return
    sg = Surrogate.load() if MODEL.exists() else Surrogate.train()
    free = 0.0 if a.turbulent else 1.0
    if a.cmd == "predict":
        p = sg.predict(a.airfoil, a.re, parse_alpha(a.alpha), free)
        for i, al in enumerate(p["alpha"]):
            print(f"  alpha {al:6.2f}: Cl {p['Cl'][i]:.4f} +- {2 * p['Cl_sd'][i]:.4f}   "
                  f"Cd {p['Cd'][i]:.5f} +- {2 * p['Cd_sd'][i]:.5f}   Cm {p['Cm'][i]:+.4f} +- "
                  f"{2 * p['Cm_sd'][i]:.4f}   (+- 2 sd)")
    else:
        al, p, sc = sg.suggest(a.airfoil, a.re, free=free)
        print(f"  most informative next RANS run: alpha {al:g} (Cl +- {2 * p['Cl_sd'].max():.3f} "
              f"at worst)")


if __name__ == "__main__":
    main()
