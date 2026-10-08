"""
Inverse airfoil design: target pressure distribution -> airfoil shape.

    python design_inverse.py --target naca2412 --alpha 3 --start naca0012
    python design_inverse.py --target-csv mycp.csv --alpha 2 --start naca2412 --out ../cases/inv1
        (mycp.csv: columns x, Cp, side  with side = 1 upper, -1 lower)

Method: the shape is a CST airfoil (Kulfan; 8 Bernstein coefficients per
side). Its inviscid pressure distribution (vortex panel method, panel.py)
is matched to the target in the least-squares sense with Levenberg-
Marquardt (scipy.optimize.least_squares) at the given angle of attack,
with a small smoothness penalty on the coefficients and a minimum
thickness. Seconds per design. The designed airfoil is then a normal
.dat file for the CFD analysis (viscous check).

Validation: the NACA 2412 pressure distribution at alpha 3 is given as the
target, the design starts from a NACA 0012 -> the NACA 2412 shape is
recovered (validate in tests/test_design.py).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import cst
import panel


def panel_cp(au, al, alpha, n=100):
    o = cst.outline(au, al, n)
    r = panel.solve([o], alpha)
    e = r["elements"][0]
    side = np.where(np.arange(len(e["x"])) < n - 1, 1, -1)       # upper panels first
    return e["x"], e["Cp"], side, r


def target_from_airfoil(spec, alpha, n=100):
    """Cp of an existing airfoil (CST fit of it, panel method)"""
    import airfoil_geom as ag
    au, al, _ = cst.fit(ag.sharp_outline(spec), 8)
    x, cp, side, r = panel_cp(au, al, alpha, n)
    return x, cp, side, (au, al)


def design(x_t, cp_t, side_t, alpha, start=(None, None), n_coef=8, t_min=0.0, smooth=1e-3,
           n=100, log=None):
    """returns dict(au, al, cp fit, history)"""
    au0, al0 = start
    if au0 is None:
        au0 = np.full(n_coef + 1, 0.17)
        al0 = -au0
    au0, al0 = np.asarray(au0, float), np.asarray(al0, float)
    k = len(au0)
    hist = []

    def unpack(v):
        return v[:k], v[k:]

    def res(v):
        au, al = unpack(v)
        x, cp, side, _ = panel_cp(au, al, alpha, n)
        r = []
        for sd in (1, -1):
            s = side == sd
            o = np.argsort(x[s])
            st = side_t == sd
            r.append(np.interp(x_t[st], x[s][o], cp[s][o]) - cp_t[st])
        t, _, tmin_x = cst.thickness(au, al)
        pen = [smooth * np.diff(au, 2), smooth * np.diff(al, 2),
               [10.0 * max(0.0, t_min - t), 10.0 * max(0.0, -tmin_x)]]
        out = np.concatenate(r + [np.ravel(p) for p in pen])
        hist.append(float(np.sqrt(np.mean(np.concatenate(r) ** 2))))
        return out
    sol = least_squares(res, np.r_[au0, al0], method="lm" if t_min == 0 else "trf",
                        x_scale=0.05, max_nfev=400)
    au, al = unpack(sol.x)
    x, cp, side, r = panel_cp(au, al, alpha, n)
    t, xt, _ = cst.thickness(au, al)
    return {"au": au, "al": al, "x": x, "cp": cp, "side": side, "Cl": r["Cl"], "Cm": r["Cm"],
            "rms": hist[-1] if hist else None, "history": hist, "thickness": t, "x_thickness": xt,
            "evaluations": int(sol.nfev)}


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import airfoil_geom as ag
    raw_outline = ag.sharp_outline
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", help="airfoil whose inviscid Cp is the target")
    ap.add_argument("--target-csv", help="x, Cp, side (1 upper / -1 lower)")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--start", default="naca0012")
    ap.add_argument("--t-min", type=float, default=0.0, help="minimum thickness / chord")
    ap.add_argument("--out", default="../cases/inverse_design")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.target_csv:
        d = np.genfromtxt(a.target_csv, delimiter=",", names=True)
        x_t, cp_t, side_t = d["x"], d["Cp"], d["side"].astype(int)
        tshape = None
    else:
        x_t, cp_t, side_t, tshape = target_from_airfoil(a.target, a.alpha)
    s0 = cst.fit(raw_outline(a.start), 8)[:2]
    r = design(x_t, cp_t, side_t, a.alpha, s0, t_min=a.t_min)
    cst.write_dat(out / "designed.dat", r["au"], r["al"], "inverse design")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for sd, c in ((1, "b"), (-1, "r")):
        s = side_t == sd
        ax[0].plot(x_t[s], cp_t[s], c + "o", ms=3, mfc="none", label="target" if sd > 0 else None)
        s2 = r["side"] == sd
        o = np.argsort(r["x"][s2])
        ax[0].plot(r["x"][s2][o], r["cp"][s2][o], c + "-", label="designed" if sd > 0 else None)
    ax[0].invert_yaxis(); ax[0].legend(); ax[0].set_xlabel("x/c"); ax[0].set_ylabel("Cp")
    ax[0].set_title(f"alpha {a.alpha:g}: rms dCp {r['rms']:.4f}, Cl {r['Cl']:.3f}")
    o = cst.outline(r["au"], r["al"])
    ax[1].plot(o[:, 0], o[:, 1], "k-", label="designed")
    o0 = cst.outline(*s0)
    ax[1].plot(o0[:, 0], o0[:, 1], "g:", label=f"start ({a.start})")
    if tshape:
        ot = cst.outline(*tshape)
        ax[1].plot(ot[:, 0], ot[:, 1], "m--", lw=0.8, label=f"target airfoil ({a.target})")
    ax[1].set_aspect("equal"); ax[1].legend(); ax[1].set_title(f"t/c {r['thickness']:.4f}")
    fig.tight_layout(); fig.savefig(out / "inverse.png", dpi=120)
    (out / "inverse.json").write_text(json.dumps({
        "alpha": a.alpha, "au": r["au"].tolist(), "al": r["al"].tolist(), "rms": r["rms"],
        "Cl": r["Cl"], "Cm": r["Cm"], "thickness": r["thickness"], "evaluations": r["evaluations"],
        "target": {"x": x_t.tolist(), "Cp": cp_t.tolist(), "side": side_t.tolist()},
        "designed": {"x": r["x"].tolist(), "Cp": r["cp"].tolist(), "side": r["side"].tolist()},
        "shape_error_vs_target_airfoil": (float(np.abs(cst.outline(r["au"], r["al"]) -
                                                      cst.outline(*tshape)).max()) if tshape else None)},
        indent=1))
    print(f"rms dCp {r['rms']:.5f}, Cl {r['Cl']:.4f}, t/c {r['thickness']:.4f}, "
          f"{r['evaluations']} evaluations -> {out}/designed.dat")


if __name__ == "__main__":
    main()
