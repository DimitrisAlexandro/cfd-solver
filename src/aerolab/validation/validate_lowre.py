"""
Low-Reynolds-number mode: Eppler 387 against the NASA Langley LTPT
measurements (McGhee, Walker & Millard, NASA TM-4062, 1988) and XFOIL.

    python src/run.py validate_lowre                  # Re 1e5, 2e5, 3e5; ~1.5 h on 2 cores
    python src/run.py validate_lowre --re 2e5 --alpha 0,4,8

Writes validation/lowre_e387.png, lowre_report.md, lowre_results.json.

Reference data: LTPT section characteristics at R = 100 000, 200 000 and
300 000 (Mach 0.03-0.06), as listed with the pressure distributions of the
report (Fig. 17), coordinates from its Table I (design, thickened trailing
edge: validation/reference/e387.dat). UIUC (Selig & McGranahan 2004) found
the LTPT and UIUC data agree well at Re 2e5, less well at 3e5 and least at
1e5; oil flow at Re 3e5, alpha 5: laminar to ~0.40 c, reattachment ~0.58 c.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

VAL = Path(__file__).resolve().parents[3] / "validation"
E387 = str(VAL / "reference" / "e387.dat")

# NASA TM-4062, Fig. 17 listings: alpha, cl, cd, cm (cd None where not measured)
LTPT = {
    1e5: [(-2.98, .102, .0217, -.1040), (-1.99, .210, .0162, -.1069), (0.00, .390, .0167, -.0978),
          (1.99, .587, .0211, -.0981), (4.00, .778, .0230, -.0920), (6.00, .974, .0224, -.0829),
          (8.00, 1.172, .0210, -.0752), (10.06, 1.200, .0413, -.0599), (11.00, 1.201, .0525, -.0550),
          (12.01, 1.189, None, -.0517), (13.01, 1.160, None, -.0541), (14.00, .842, None, -.1378)],
    2e5: [(-2.84, .066, .0163, -.0813), (-1.99, .156, .0133, -.0814), (0.01, .352, .0105, -.0782),
          (2.04, .574, .0118, -.0794), (3.99, .785, .0133, -.0803), (6.04, .999, .0144, -.0790),
          (8.03, 1.182, .0174, -.0751), (10.02, 1.231, .0357, -.0643), (11.04, 1.214, .0570, -.0606),
          (12.09, 1.174, None, -.0593), (14.01, 1.155, None, -.0727), (16.09, .851, None, -.1463)],
    3e5: [(-2.93, .051, .0140, -.0768), (-2.01, .146, .0118, -.0788), (0.01, .352, .0089, -.0767),
          (2.00, .573, .0099, -.0785), (4.00, .792, .0109, -.0794), (6.01, 1.009, .0118, -.0799),
          (8.02, 1.180, .0168, -.0756), (10.02, 1.251, .0316, -.0661), (11.02, 1.241, .0621, -.0670),
          (12.01, 1.215, None, -.0690), (14.01, 1.179, None, -.0751), (16.00, .999, None, -.1693)],
}


def ltpt_at(re, alpha, k):
    """LTPT value (1 = cl, 2 = cd, 3 = cm) interpolated to alpha"""
    rows = [r for r in LTPT[re] if r[k] is not None]
    a = [r[0] for r in rows]
    if alpha < a[0] - 0.5 or alpha > a[-1] + 0.5:
        return None
    return float(np.interp(alpha, a, [r[k] for r in rows]))


def run(res=(1e5, 2e5, 3e5), alphas=(-2, 0, 2, 4, 6, 8, 10), jobs=2, log=print):
    from aerolab.airfoil import airfoil as AF
    from aerolab.airfoil import polar as pl
    out = {}
    for re in res:
        log(f"[lowre] E387 Re {re:.0e}: RANS + e^N (low-Re mode)")
        r = pl.polar(E387, re, list(alphas), jobs=jobs, transition="free", ncrit=9.0, log=False)
        xf = AF.run_xfoil(E387, re, list(alphas))
        out[re] = {"rans": [{k: v for k, v in p.items() if k not in ("wall",)} for p in r],
                   "xfoil": {str(a): v for a, v in xf.items()}}
        for p in r:
            ref = (ltpt_at(re, p["alpha"], 1), ltpt_at(re, p["alpha"], 2))
            log(f"  alpha {p['alpha']:5.1f}: Cl {p['Cl']:.3f} (LTPT {ref[0]}), Cd {p['Cd']:.5f} "
                f"(LTPT {ref[1]}), x_tr {p['x_tr_upper']}  {p['status']}")
    return out


def errors(out):
    rows = []
    for re, d in out.items():
        for p in d["rans"]:
            a = p["alpha"]
            cl_e, cd_e = ltpt_at(float(re), a, 1), ltpt_at(float(re), a, 2)
            xf = d["xfoil"].get(str(a)) or d["xfoil"].get(str(float(a))) or d["xfoil"].get(str(int(a)))
            rows.append({"Re": float(re), "alpha": a, "Cl": p["Cl"], "Cl_LTPT": cl_e, "Cd": p["Cd"], "Cd_LTPT": cd_e,
                         "Cl_xfoil": xf[0] if xf else None, "Cd_xfoil": xf[1] if xf else None,
                         "x_tr": p.get("x_tr_upper"), "x_tr_xfoil": xf[3] if xf else None,
                         "status": p["status"],
                         "dCd_pct": 100 * (p["Cd"] / cd_e - 1) if cd_e else None,
                         "dCl": p["Cl"] - cl_e if cl_e is not None else None})
    return rows


def figure(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = sorted(out, key=float)
    fig, ax = plt.subplots(3, len(res), figsize=(5.2 * len(res), 12), squeeze=False)
    for j, re in enumerate(res):
        d = out[re]
        r = d["rans"]
        e = LTPT[float(re)]
        a = [p["alpha"] for p in r]
        xa = sorted(float(k) for k in d["xfoil"])
        xv = [d["xfoil"][k] for k in sorted(d["xfoil"], key=float)]
        ax[0, j].plot([x[0] for x in e], [x[1] for x in e], "ks", ms=5, mfc="none", label="NASA LTPT")
        ax[0, j].plot(xa, [v[0] for v in xv], "k:", label="XFOIL (N 9)")
        ax[0, j].plot(a, [p["Cl"] for p in r], "o-", color="#1D4E89", label="this solver")
        ax[0, j].set_xlabel("α (deg)"); ax[0, j].set_ylabel("Cl"); ax[0, j].set_title(f"Eppler 387, Re {float(re):.0e}")
        ee = [x for x in e if x[2] is not None]
        ax[1, j].plot([x[2] for x in ee], [x[1] for x in ee], "ks", ms=5, mfc="none", label="NASA LTPT")
        ax[1, j].plot([v[1] for v in xv], [v[0] for v in xv], "k:", label="XFOIL")
        ax[1, j].plot([p["Cd"] for p in r], [p["Cl"] for p in r], "o-", color="#1D4E89", label="this solver")
        ax[1, j].set_xlabel("Cd"); ax[1, j].set_ylabel("Cl"); ax[1, j].set_xlim(0, 0.04)
        ax[2, j].plot(a, [p.get("x_tr_upper") for p in r], "o-", color="#1D4E89", label="upper, this solver")
        ax[2, j].plot(xa, [v[3] for v in xv], "k:", label="upper, XFOIL")
        if abs(float(re) - 3e5) < 1:
            ax[2, j].errorbar([5.0], [0.49], yerr=[[0.09], [0.09]], fmt="none", ecolor="#C2571A", capsize=4,
                              label="UIUC oil flow: bubble 0.40–0.58 c")
        ax[2, j].set_xlabel("α (deg)"); ax[2, j].set_ylabel("transition x/c"); ax[2, j].set_ylim(0, 1)
        for k in range(3):
            ax[k, j].grid(alpha=0.3)
            ax[k, j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def report(out):
    rows = errors(out)
    L = ["# Low-Reynolds-number mode: Eppler 387", "",
         "RANS (Spalart–Allmaras) with e^N transition (N_crit 9) and separation-induced transition: "
         "transition at laminar separation + Mayle's long-bubble length Re_l = 1000 Re_θs^0.7; leading-edge "
         "bubbles (separation ahead of 0.15 c): not shorter than Horton's Re_l = 4×10⁴ and no trip; mid-chord "
         "bubbles: a turbulence trip behind the transition point (ν̃ ≥ 100 ν over 0.02 c × 0.05 c) so the "
         "shear layer reattaches. Every angle from a cold start. "
         "Reference: NASA Langley LTPT (TM-4062) and XFOIL. "
         "Reproduce with `python src/run.py validate_lowre`; `--report-only` rebuilds this page from the saved "
         "results.", "",
         "| Re | α | Cl | LTPT | XFOIL | Cd | LTPT | XFOIL | ΔCd vs LTPT | x_tr upper | XFOIL x_tr | status |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, n: "–" if v is None else f"{v:.{n}f}"
    for r in rows:
        dcd = "–" if r["dCd_pct"] is None else f"{r['dCd_pct']:+.0f} %"
        L.append(f"| {r['Re']:.0e} | {r['alpha']:g} | {r['Cl']:.3f} | {f(r['Cl_LTPT'], 3)} | {f(r['Cl_xfoil'], 3)} | "
                 f"{r['Cd']:.5f} | {f(r['Cd_LTPT'], 4)} | {f(r['Cd_xfoil'], 4)} | {dcd} | "
                 f"{f(r['x_tr'], 2)} | {f(r['x_tr_xfoil'], 2)} | {r['status']} |")
    for re in sorted({r["Re"] for r in rows}):
        sel = [r for r in rows if r["Re"] == re and r["dCd_pct"] is not None and r["alpha"] <= 8.5]
        if sel:
            L.append("")
            L.append(f"Re {re:.0e}, α −2…8: mean |ΔCd| {np.mean([abs(r['dCd_pct']) for r in sel]):.0f} %, "
                     f"mean |ΔCl| {np.mean([abs(r['dCl']) for r in sel]):.3f} vs LTPT.")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--re", default="1e5,2e5,3e5")
    ap.add_argument("--alpha", default="-2,0,2,4,6,8,10")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--report-only", action="store_true", help="rebuild the figure and report from the JSON")
    a = ap.parse_args(argv)
    js = VAL / "lowre_results.json"
    old = json.loads(js.read_text()) if js.exists() else {}
    if not a.report_only:
        res = [float(v) for v in a.re.split(",")]
        alphas = [float(v) for v in a.alpha.split(",")]
        out = run(res, alphas, a.jobs)
        old.update({f"{float(k):.0e}": v for k, v in out.items()})
        js.write_text(json.dumps(old, indent=1, default=float))
    full = {float(k): v for k, v in old.items()}
    figure(full, VAL / "lowre_e387.png")
    (VAL / "lowre_report.md").write_text(report(full))
    print(f"-> {VAL}/lowre_e387.png, lowre_report.md")


if __name__ == "__main__":
    main()
