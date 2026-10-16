"""Field inversion over several airfoils and a learned SA correction tested
on airfoils left out of training (leave-one-airfoil-out).

Experimental data (all measured, see validation/reference):
  NACA 0012  Re 3e6     Gregory & O'Reilly (lift and upper-surface Cp)
  NACA 0012  Re 6e6     Ladson, 80 grit (tripped; lift)
  NACA 4412  Re 1.52e6  Coles & Wadcock, alpha 13.87 (Cp, trailing-edge separation)
  S809       Re 0.75e6  Ohio State, clean (lift; free transition)
  E387       Re 2e5/3e5 NASA Langley LTPT (lift; laminar separation bubble)
NACA 2412 is not included: no measured data could be obtained in this
environment (the Abbott & von Doenhoff tables are only available as
scanned reports), and none is invented.

Every case: Inversion (fiml.py) of the SA production multiplier beta
against the case's lift and/or Cp, saved as <out>/inv/<case>.npz. Then for
each airfoil family F a model is trained on the inversions of the OTHER
families and run (augmented SA) on F's test points; the error against the
data is compared with baseline SA.

  python src/fiml_multi.py invert [--cases ...] [--jobs 3]
  python src/fiml_multi.py loao [--jobs 3]
  python src/fiml_multi.py report
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "validation" / "reference"
TMR = REF / "tmr"
OUT = ROOT / "cases" / "fiml_multi"

S809 = str(REF / "fiml" / "s809.dat")
E387 = str(REF / "e387.dat")


# ----------------------------------------------------------------------
# experimental data
# ----------------------------------------------------------------------
def _zones(path):
    import validate_naca0012_sa as v
    return v.tmr_zones(path)


def gregory_cp(alpha):
    """(x, Cp, side) upper surface, Gregory & O'Reilly (alpha 0, 10, 15)"""
    z = _zones(TMR / "CP_Gregory_expdata.dat")[f"alpha={alpha:g}"]
    return z[:, 0], z[:, 1], np.ones(len(z), int)


def gregory_cl(alpha):
    import fiml
    return fiml.cl_exp(alpha)


def ladson_cl(alpha, zone="80 grit"):
    z = _zones(TMR / "CLCD_Ladson_expdata.dat")[zone]
    o = np.argsort(z[:, 0])
    return float(np.interp(alpha, z[o, 0], z[o, 1]))


def coles_wadcock_cp():
    d = np.loadtxt(REF / "fiml" / "naca4412_cp_coles_wadcock.dat", comments=("#", "variables"))
    x, y, cp = d.T
    # side from the surface ordinate (the camber line of a 4412 at x/c > 0.9 is
    # above zero, so use the ordinate relative to the camber line)
    yc = np.where(x < 0.4, 0.04 / 0.16 * (0.8 * x - x**2),
                  0.04 / 0.36 * (0.2 + 0.8 * x - x**2))
    return x, cp, np.where(y >= yc, 1, -1)


def s809_osu(alpha, k=1):
    d = np.loadtxt(REF / "fiml" / "s809_osu_re750k_clean.dat")
    return float(np.interp(alpha, d[:, 0], d[:, k]))


def e387_ltpt(re, alpha):
    import validate_lowre as v
    return v.ltpt_at(re, alpha, 1)


# ----------------------------------------------------------------------
# cases
# ----------------------------------------------------------------------
def _case(name, family, airfoil, re, alpha, transition="off", cl=None, cp=None):
    return {"name": name, "family": family, "airfoil": airfoil, "re": re, "alpha": alpha,
            "transition": transition, "cl": cl, "cp": cp}


def training_cases():
    """the inversions (cl: callable or None; cp: loader name or None)"""
    return [
        _case("n0012_re3e6_a10", "NACA 0012", "naca0012", 3e6, 10.0, cl="gregory", cp="gregory"),
        _case("n0012_re3e6_a15", "NACA 0012", "naca0012", 3e6, 15.0, cl="gregory", cp="gregory"),
        _case("n0012_re6e6_a15.26", "NACA 0012", "naca0012", 6e6, 15.26, cl="ladson"),
        _case("n4412_re1.52e6_a13.87", "NACA 4412", "naca4412", 1.52e6, 13.87, cp="coles"),
        _case("s809_re7.5e5_a8.1", "S809", S809, 7.5e5, 8.1, "free", cl="osu"),
        _case("s809_re7.5e5_a14.2", "S809", S809, 7.5e5, 14.2, "free", cl="osu"),
        _case("e387_re2e5_a8", "E387", E387, 2e5, 8.03, "free", cl="ltpt"),
        _case("e387_re3e5_a10", "E387", E387, 3e5, 10.02, "free", cl="ltpt"),
    ]


def test_points():
    """per family: the points the held-out model is run on"""
    return {
        "NACA 0012": [_case(f"n0012_re3e6_a{a:g}", "NACA 0012", "naca0012", 3e6, a,
                            cl="gregory", cp="gregory" if a in (10, 15) else None)
                      for a in (10.0, 13.0, 15.0, 16.0)],
        "NACA 4412": [_case("n4412_re1.52e6_a13.87", "NACA 4412", "naca4412", 1.52e6, 13.87,
                            cp="coles")],
        "S809": [_case(f"s809_re7.5e5_a{a:g}", "S809", S809, 7.5e5, a, "free", cl="osu")
                 for a in (6.2, 10.2, 14.2, 16.3)],
        "E387": [_case(f"e387_re{re:.0e}_a{a:g}", "E387", E387, re, a, "free", cl="ltpt")
                 for re, a in ((2e5, 6.04), (2e5, 10.02), (3e5, 8.02), (3e5, 12.01))],
    }


def target_cl(c):
    k = c["cl"]
    if k is None:
        return None
    if k == "gregory":
        return gregory_cl(c["alpha"])
    if k == "ladson":
        return ladson_cl(c["alpha"])
    if k == "osu":
        return s809_osu(c["alpha"])
    if k == "ltpt":
        return e387_ltpt(c["re"], c["alpha"])
    raise ValueError(k)


def target_cp(c):
    k = c["cp"]
    if k is None:
        return None
    if k == "gregory":
        return gregory_cp(c["alpha"])
    if k == "coles":
        return coles_wadcock_cp()
    raise ValueError(k)


def find_case(name):
    for c in training_cases() + [p for v in test_points().values() for p in v]:
        if c["name"] == name:
            return c
    raise KeyError(name)


# ----------------------------------------------------------------------
# comparison with the data
# ----------------------------------------------------------------------
def cp_error(s, walls, U, cp):
    """RMS of (Cp_CFD - Cp_exp) at the measured points (CFD interpolated
    along x on the measured side)"""
    import postproc
    w = postproc.wall_data(s, walls[0], U)[0]
    xe, ce, se = cp
    err = []
    for sd in (1, -1):
        k = se == sd
        if not k.any():
            continue
        sel = w["side"] == sd
        o = np.argsort(w["x"][sel])
        xc, cc = w["x"][sel][o], w["Cp"][sel][o]
        ok = (xe[k] >= xc.min()) & (xe[k] <= xc.max())
        err.append(np.interp(xe[k][ok], xc, cc) - ce[k][ok])
    e = np.concatenate(err)
    return float(np.sqrt((e**2).mean()))


def wall_cp(s, walls, U):
    import postproc
    w = postproc.wall_data(s, walls[0], U)[0]
    return {"x": w["x"].tolist(), "Cp": w["Cp"].tolist(), "side": w["side"].tolist()}


# ----------------------------------------------------------------------
# steps
# ----------------------------------------------------------------------
def invert(name, out=OUT, iters=20, level=0.5, log=print):
    import fiml
    import polar as pl
    c = find_case(name)
    cl, cp = target_cl(c), target_cp(c)
    t0 = time.time()
    inv = fiml.Inversion(c["airfoil"], c["re"], c["alpha"], level, cl=cl, cp=cp, log=log,
                         transition=c["transition"])
    s, pr = inv.s, inv.pr
    c0 = pl.coefficients(s, pr["walls"], pr["U"])[0]
    e0 = cp_error(s, pr["walls"], pr["U"], cp) if cp is not None else None
    cp0 = wall_cp(s, pr["walls"], pr["U"])
    inv.run(iters)
    c1 = pl.coefficients(s, pr["walls"], pr["U"])[0]
    e1 = cp_error(s, pr["walls"], pr["U"], cp) if cp is not None else None
    d = Path(out) / "inv"
    d.mkdir(parents=True, exist_ok=True)
    inv.save(d / f"{name}.npz")
    rec = {**{k: v for k, v in c.items() if k not in ("cl", "cp")}, "data": {"cl": c["cl"], "cp": c["cp"]},
           "cl_target": cl, "Cl_SA": c0["Cl"], "Cd_SA": c0["Cd"], "Cl_inv": c1["Cl"],
           "Cd_inv": c1["Cd"], "cp_rms_SA": e0, "cp_rms_inv": e1,
           "beta_min": float(inv.beta.min()), "beta_max": float(inv.beta.max()),
           "cells_changed": int((np.abs(inv.beta - 1) > 0.1).sum()), "cells": int(len(inv.beta)),
           "solutions": len(inv.hist), "time_s": round(time.time() - t0),
           "wall_SA": cp0, "wall_inv": wall_cp(s, pr["walls"], pr["U"])}
    (d / f"{name}.json").write_text(json.dumps(rec, indent=1))
    log(f"[fiml_multi] {name}: Cl SA {c0['Cl']:.4f} -> {c1['Cl']:.4f} (target {cl}), "
        f"Cp rms {e0} -> {e1}  ({rec['time_s']} s)")
    return rec


def predict(name, model, out_json, level=0.5, log=print):
    """baseline SA and SA + learned beta at one test point"""
    import adjoint as A
    import fiml
    import polar as pl
    c = find_case(name)
    net = fiml.MLP.from_json(json.loads(Path(model).read_text()))
    cl, cp = target_cl(c), target_cp(c)
    t0 = time.time()
    base = A.solve_primal(c["airfoil"], c["re"], c["alpha"], level, c["transition"])
    s = base["solver"]
    cb = pl.coefficients(s, base["walls"], base["U"])[0]
    rec = {"name": name, "family": c["family"], "alpha": c["alpha"], "re": c["re"],
           "cl_exp": cl, "Cl_SA": cb["Cl"], "Cd_SA": cb["Cd"], "SA_residual": base["residual"]}
    if cp is not None:
        rec["cp_rms_SA"] = cp_error(s, base["walls"], base["U"], cp)
        rec["wall_SA"] = wall_cp(s, base["walls"], base["U"])
    state = {"xc": s.m.xc.copy(), "U": s.U.copy(), "p": s.p.copy(), "nuT": s.turb.nuT.copy()}
    c1, pr, hist = fiml.solve_augmented(c["airfoil"], c["re"], c["alpha"], net, level,
                                        transition=c["transition"], primal=base)
    s = pr["solver"]
    rec.update(Cl_FIML=c1["Cl"], Cd_FIML=c1["Cd"], FIML_residual=hist[-1] if hist else None,
               beta_min=float(s.turb.beta.min()), beta_max=float(s.turb.beta.max()),
               time_s=round(time.time() - t0))
    if cp is not None:
        rec["cp_rms_FIML"] = cp_error(s, pr["walls"], pr["U"], cp)
        rec["wall_FIML"] = wall_cp(s, pr["walls"], pr["U"])
    del state
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(rec, indent=1))
    log(f"[fiml_multi] predict {name} ({Path(model).stem}): Cl SA {cb['Cl']:.4f}, "
        f"FIML {c1['Cl']:.4f}, exp {cl}")
    return rec


def train_fold(held_out, out=OUT, log=print):
    """model from the inversions of every family except `held_out`
    (None: all families)"""
    import fiml
    files = []
    for c in training_cases():
        f = Path(out) / "inv" / f"{c['name']}.npz"
        if c["family"] != held_out and f.exists():
            files.append(f)
    net = fiml.train(files, log=log)
    tag = "all" if held_out is None else "without_" + held_out.replace(" ", "")
    p = Path(out) / "models" / f"{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**net.to_json(), "trained_on": [f.stem for f in files]}))
    return p


def _parallel(cmds, jobs, log=print):
    """run CLI commands, `jobs` at a time"""
    running = []
    for cmd in cmds:
        while len(running) >= jobs:
            running = [p for p in running if p.poll() is None]
            time.sleep(2)
        log("  $ " + " ".join(cmd[2:]))
        running.append(subprocess.Popen(cmd))
    for p in running:
        p.wait()


def run_inversions(names=None, jobs=3, iters=20):
    names = names or [c["name"] for c in training_cases()]
    todo = [n for n in names if not (OUT / "inv" / f"{n}.json").exists()]
    _parallel([[sys.executable, __file__, "invert-one", n, "--iters", str(iters)] for n in todo],
              jobs)


def run_loao(jobs=3):
    cmds = []
    for fam, pts in test_points().items():
        model = train_fold(fam)
        for c in pts:
            o = OUT / "pred" / model.stem / f"{c['name']}.json"
            if not o.exists():
                cmds.append([sys.executable, __file__, "predict-one", c["name"], str(model),
                             str(o)])
    _parallel(cmds, jobs)


def summarise(out=OUT):
    inv = [json.loads(p.read_text()) for p in sorted((Path(out) / "inv").glob("*.json"))]
    pred = [json.loads(p.read_text()) for p in sorted((Path(out) / "pred").glob("*/*.json"))]
    s = {"inversions": [{k: v for k, v in r.items() if not k.startswith("wall")} for r in inv],
         "predictions": [{k: v for k, v in r.items() if not k.startswith("wall")} for r in pred]}
    for fam in {r["family"] for r in pred}:
        rs = [r for r in pred if r["family"] == fam and r.get("cl_exp") is not None]
        if rs:
            s.setdefault("cl_error_by_family", {})[fam] = {
                "SA_mean_abs": float(np.mean([abs(r["Cl_SA"] - r["cl_exp"]) for r in rs])),
                "FIML_mean_abs": float(np.mean([abs(r["Cl_FIML"] - r["cl_exp"]) for r in rs])),
                "points": len(rs)}
    (Path(out) / "summary.json").write_text(json.dumps(s, indent=1))
    return s


def plot(out=OUT, path=ROOT / "validation" / "fiml_multi.png"):
    """lift error per held-out family (SA vs SA + learned beta) and the Cp
    of the held-out NACA 4412"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pred = [json.loads(p.read_text()) for p in sorted((Path(out) / "pred").glob("*/*.json"))]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    fams = ["NACA 0012", "S809", "E387"]
    mk = {"NACA 0012": "o", "S809": "s", "E387": "^", "NACA 4412": "D"}
    for f in fams:
        rs = sorted([r for r in pred if r["family"] == f and r.get("cl_exp") is not None],
                    key=lambda r: (r["re"], r["alpha"]))
        if not rs:
            continue
        a = [r["alpha"] for r in rs]
        ax[0].plot(a, [r["Cl_SA"] - r["cl_exp"] for r in rs], mk[f], mfc="none", color="gray",
                   label=f"{f}: SA")
        ax[0].plot(a, [r["Cl_FIML"] - r["cl_exp"] for r in rs], mk[f], color="C3",
                   label=f"{f}: SA + learned beta (family not in training)")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("alpha (deg)")
    ax[0].set_ylabel("Cl - Cl experiment")
    ax[0].legend(fontsize=7)
    ax[0].set_title("lift error, leave-one-airfoil-out")
    r4 = [r for r in pred if r["family"] == "NACA 4412" and "wall_FIML" in r]
    if r4:
        r = r4[0]
        xe, ce, se = coles_wadcock_cp()
        for k, st, lab in (("wall_SA", "--", "SA"), ("wall_FIML", "-", "SA + learned beta (4412 unseen)")):
            w = r[k]
            x, cp, sd = map(np.asarray, (w["x"], w["Cp"], w["side"]))
            for side in (1, -1):
                o = np.argsort(x[sd == side])
                ax[1].plot(x[sd == side][o], cp[sd == side][o], st, color="C0" if k == "wall_SA" else "C3",
                           label=lab if side == 1 else None)
        ax[1].plot(xe, ce, "ko", mfc="none", ms=4, label="Coles & Wadcock")
        inv = Path(out) / "inv" / "n4412_re1.52e6_a13.87.json"
        if inv.exists():
            w = json.loads(inv.read_text())["wall_inv"]
            x, cp, sd = map(np.asarray, (w["x"], w["Cp"], w["side"]))
            o = np.argsort(x[sd == 1])
            ax[1].plot(x[sd == 1][o], cp[sd == 1][o], ":", color="C2", label="field inversion (4412 data)")
        ax[1].invert_yaxis()
        ax[1].set_xlabel("x/c")
        ax[1].set_ylabel("Cp")
        ax[1].set_title("NACA 4412, alpha 13.87, Re 1.52e6")
        ax[1].legend(fontsize=7)
    inv = [json.loads(p.read_text()) for p in sorted((Path(out) / "inv").glob("*.json"))]
    names = [r["name"] for r in inv]
    y = np.arange(len(inv))
    e0 = [abs(r["Cl_SA"] - r["cl_target"]) if r["cl_target"] is not None else np.nan for r in inv]
    e1 = [abs(r["Cl_inv"] - r["cl_target"]) if r["cl_target"] is not None else np.nan for r in inv]
    ax[2].barh(y + 0.2, e0, 0.4, color="gray", label="SA")
    ax[2].barh(y - 0.2, e1, 0.4, color="C3", label="field inversion")
    ax[2].set_yticks(y, names, fontsize=7)
    ax[2].set_xlabel("|Cl - Cl experiment| (training cases)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("invert")
    p.add_argument("--cases", nargs="*")
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p = sub.add_parser("invert-one")
    p.add_argument("name")
    p.add_argument("--iters", type=int, default=20)
    p = sub.add_parser("predict-one")
    p.add_argument("name")
    p.add_argument("model")
    p.add_argument("out")
    p = sub.add_parser("loao")
    p.add_argument("--jobs", type=int, default=3)
    sub.add_parser("summary")
    sub.add_parser("plot")
    a = ap.parse_args(argv)
    if a.cmd == "invert":
        run_inversions(a.cases, a.jobs, a.iters)
    elif a.cmd == "invert-one":
        invert(a.name, iters=a.iters)
    elif a.cmd == "predict-one":
        predict(a.name, a.model, a.out)
    elif a.cmd == "loao":
        run_loao(a.jobs)
    elif a.cmd == "plot":
        print(plot())
    else:
        print(json.dumps(summarise(), indent=1)[:4000])


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
