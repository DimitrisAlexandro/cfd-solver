"""3D-2 validation: RANS (SA) wings against the vortex lattice.

    python -m fv3d.validate_wing3d                 # all cases (hours on 2 cores)
    python -m fv3d.validate_wing3d --quick         # rectangular wing, alpha 4 only

Cases
  rect   rectangular NACA 0012 wing, AR 6, Re 3e6: alpha 0, 4, 8
  taper  NACA 0012, taper 0.5, quarter-chord sweep 20 deg, 2 deg washout, AR 8, alpha 4
  grid   the rectangular wing at alpha 4 on a finer mesh

Reference: AeroLab's vortex lattice (wing.py, 6 chordwise panels) with sections of
the same 2D lift slope as our validated 2D RANS (NACA 0012, SA): the comparison
then isolates the three-dimensional part - downwash, tip loss, sweep, taper.
Outputs: validation/wing3d_validation.json, wing3d_report.md, wing3d_validation.png
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fv3d import wing3d  # noqa: E402

CASES = {
    "rect": dict(semi_span=3.0, taper=1.0, sweep_c4=0.0, twist=0.0, alphas=(0.0, 4.0, 8.0), level=0.4),
    "taper": dict(semi_span=3.0, taper=0.5, sweep_c4=20.0, twist=-2.0, alphas=(4.0,), level=0.4),
    "grid": dict(semi_span=3.0, taper=1.0, sweep_c4=0.0, twist=0.0, alphas=(4.0,), level=0.55),
}


def section_polar(out_dir, re=3e6, alphas=(-4, -2, 0, 2, 4, 6, 8, 10, 12), level=0.7):
    """2D SA polar of NACA 0012 at the wing's Reynolds number from our validated 2D
    solver (TMR-type mesh, far field 100 chords), cached in out_dir"""
    f = Path(out_dir) / "wing3d_section_polar.json"
    if f.exists():
        d = json.loads(f.read_text())
        if d.get("re") == re and d.get("level") == level:
            return d
    import polar
    from simple_solver import SimpleSolver
    rows = []
    for a in alphas:
        fvm, mesh, patches, bcs, U, turb, info = polar.make_case("naca0012", re, a, level)
        s2 = SimpleSolver(fvm, bcs, 1.0 / re, turbulence="sa", U0=U)
        s2.solve(tol=1e-8, max_iter=4000, verbose=False)
        Fp, Fv = s2.force("wall")
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        F = Fp + Fv
        rows.append({"alpha": a, "Cl": float(-F[0] * sa + F[1] * ca) / 0.5,
                     "Cd": float(F[0] * ca + F[1] * sa) / 0.5, "Cm": 0.0})
    d = {"re": re, "level": level, "rows": rows}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d, indent=1))
    return d


def vlm(case, alpha, sec):
    """vortex lattice with the 2D RANS section polar (nonlinear lift curve)"""
    import wing as W
    b = 2 * case["semi_span"]
    wg = W.Wing(b, 1.0, case["taper"], sweep=case["sweep_c4"], twist=case["twist"], n_half=40)
    rows = sec["rows"]
    sp = W.SectionPolar([r["alpha"] for r in rows], [r["Cl"] for r in rows], [r["Cd"] for r in rows],
                        [r["Cm"] for r in rows], ar=b * b / (b * (1 + case["taper"]) / 2))
    pol = W.SpanPolars([(sec["re"], sp)])
    r = W.solve(wg, alpha, polars=pol, re_strip=np.full(len(wg.c), sec["re"]))
    spn = r["span"]
    half = spn["eta"] > 0
    return {"CL": float(r["CL"]), "CDi": float(r["CDi"]), "CD": float(r["CD"]), "e": float(r["e"]),
            "eta": spn["eta"][half].tolist(), "cl": spn["cl"][half].tolist(), "clc": spn["cl_c"][half].tolist()}


def run(names, out_dir, max_iter=1500, force_tol=3e-3, log=print):
    f = Path(out_dir) / "wing3d_validation.json"
    out = json.loads(f.read_text()) if f.exists() else {"cases": {}}     # add to earlier cases
    for nm in names:
        c = CASES[nm]
        t0 = time.time()
        log(f"== {nm}: {c}")
        d = wing3d.run(semi_span=c["semi_span"], taper=c["taper"], sweep_c4=c["sweep_c4"], twist=c["twist"],
                       alphas=c["alphas"], level=c["level"], max_iter=max_iter, force_tol=force_tol, log=log)
        rows = []
        for r in d["results"]:
            last = [h for h in r["history"] if h["it"] > r["iterations"] - 200]
            var = {k: (max(h[k] for h in last) - min(h[k] for h in last)) if last else None for k in ("CL", "CD")}
            rows.append({"alpha": r["alpha"], "CL": r["CL"], "CD": r["CD"], "CDp": r["CDp"], "CDv": r["CDv"],
                         "Cm": r["Cm"], "iterations": r["iterations"], "status": r["status"],
                         "residual_cont": r["history"][-1]["cont"] if r["history"] else None,
                         "force_variation_last_200": var, "span": r["span"]})
        out["cases"][nm] = {"setup": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()},
                            "mesh": d["mesh"], "info": d["info"], "results": rows,
                            "time_s": round(time.time() - t0, 1)}
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "wing3d_validation.json").write_text(json.dumps(out, indent=1))
    return out


def add_reference(d, out_dir):
    """vortex lattice + 2D RANS section polar for every RANS result"""
    sec = section_polar(out_dir)
    d["reference"] = "AeroLab vortex lattice (6 chordwise panels) with the 2D SA polar of the section"
    d["section_polar"] = sec
    for nm, c in d["cases"].items():
        setup = dict(c["setup"])
        for r in c["results"]:
            r["vlm"] = vlm(setup, r["alpha"], sec)
    (Path(out_dir) / "wing3d_validation.json").write_text(json.dumps(d, indent=1))
    return d


def report(d, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    L = ["# 3D RANS wing validation (3D-2)", "",
         f"Reference: {d['reference']} (NACA 0012, Re {d['section_polar']['re']:.3g}, "
         f"level {d['section_polar']['level']}).", "",
         "| case | cells | alpha | CL (RANS) | CL (VLM) | dCL | CD (RANS) | CD (VLM: CDi + section) | iterations | continuity |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for nm, c in d["cases"].items():
        for r in c["results"]:
            v = r["vlm"]
            dcl = (r["CL"] - v["CL"]) / v["CL"] * 100 if abs(v["CL"]) > 1e-3 else float("nan")
            L.append(f"| {nm} | {c['mesh']['cells']} | {r['alpha']:g} | {r['CL']:.4f} | {v['CL']:.4f} | "
                     f"{dcl:+.1f}% | {r['CD']:.5f} | {v['CD']:.5f} | {r['iterations']} | "
                     f"{r['residual_cont']:.1e} |")
    (out_dir / "wing3d_report.md").write_text("\n".join(L) + "\n")
    cases = [(nm, c) for nm, c in d["cases"].items()]
    fig, ax = plt.subplots(1, max(len(cases), 1), figsize=(5 * max(len(cases), 1), 4), squeeze=False)
    for k, (nm, c) in enumerate(cases):
        a = ax[0, k]
        for r in c["results"]:
            if abs(r["alpha"]) < 1e-9:
                continue
            a.plot(r["span"]["eta"], r["span"]["clc"], "o-", ms=3, label=f"RANS a={r['alpha']:g}")
            a.plot(r["vlm"]["eta"], r["vlm"]["clc"], "--", label=f"VLM a={r['alpha']:g}")
        a.set_title(nm)
        a.set_xlabel("eta = 2y/b")
        a.set_ylabel("c_l c / c_root")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "wing3d_validation.png", dpi=120)
    return out_dir / "wing3d_report.md"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cases", default="rect,taper,grid")
    ap.add_argument("--out", default=str(ROOT / "validation"))
    ap.add_argument("--max-iter", type=int, default=1500)
    ap.add_argument("--report-only", action="store_true", help="rebuild the comparison from the saved JSON")
    a = ap.parse_args(argv)
    if a.report_only:
        d = json.loads((Path(a.out) / "wing3d_validation.json").read_text())
        d = add_reference(d, a.out)
        print("report:", report(d, a.out))
        return
    if a.quick:
        CASES["rect"] = dict(CASES["rect"], alphas=(4.0,))
        names = ["rect"]
    else:
        names = a.cases.split(",")
    d = run(names, a.out, a.max_iter)
    d = add_reference(d, a.out)
    print("report:", report(d, a.out))


if __name__ == "__main__":
    main()
