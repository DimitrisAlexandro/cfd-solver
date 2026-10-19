"""
Phase 6 validation: the quasi-3D wing tool (wing.py).

  1. Elliptic planform, thin-airfoil sections: span efficiency e = 1 (exact
     for elliptic loading), lift slope -> lifting-line 2 pi/(1 + 2/AR) at
     large AR; spanwise / chordwise grid convergence.
  2. Rectangular and swept-tapered flat wings: lift-curve slope vs the
     DATCOM / Helmbold formula (USAF DATCOM 1978, sec. 4.1.3.2),
         CL_a = 2 pi A / (2 + sqrt(A^2 beta^2 (1 + tan^2 L_c/2 / beta^2) + 4))
     (accuracy quoted by DATCOM: ~ +-5 %).
  3. Compressibility (Goethert): the same formula with beta = sqrt(1 - M^2).
  4. Nonlinear sections, 2D limit: a very high aspect-ratio wing with a CFD
     section polar reproduces the polar (alpha shifted by the induced angle).

    python src/run.py validate_wing      -> ../validation/wing_*.png, wing_report.txt
"""

import json
import math
from pathlib import Path

import numpy as np

from aerolab.aircraft import wing as W

OUT = Path(__file__).resolve().parents[3] / "validation"


def datcom(AR, sweep_c4, taper, mach=0.0):
    beta = math.sqrt(1 - mach**2)
    th = math.tan(math.radians(sweep_c4)) - 4 / AR * 0.25 * (1 - taper) / (1 + taper)
    return 2 * math.pi * AR / (2 + math.sqrt(AR**2 * beta**2 * (1 + th**2 / beta**2) + 4))


def trapezoid(AR, taper, sweep, n_half=40):
    cr = 1.0
    b = AR * cr * (1 + taper) / 2
    return W.Wing(b, cr, taper * cr, sweep=sweep, n_half=n_half)


def elliptic(AR, n_half=40, b=10.0):
    c0 = 4 * b / (math.pi * AR)
    return W.Wing(b, c0, chord_fn=lambda e: c0 * np.sqrt(np.maximum(1 - e**2, 0)), n_half=n_half)


def cla(wing, mach=0.0, nc=4):
    r = W.solve(wing, 4.0, mach=mach, nc=nc)
    return r["CL"] / math.radians(4.0), r


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out, lines = {}, ["Quasi-3D wing tool (vortex lattice + section polars): validation", ""]
    log = lambda s: (print(s), lines.append(s))

    # 1. elliptic wing
    log("1. elliptic planform, thin-airfoil sections (e must be 1)")
    ell = []
    for AR in (4, 6, 10, 20, 40):
        a, r = cla(elliptic(AR))
        ell.append((AR, a, r["e"]))
        log(f"   AR {AR:3d}: CL_a {a:.4f}/rad  lifting line {2 * math.pi / (1 + 2 / AR):.4f}  "
            f"e {r['e']:.4f}")
    conv = []
    for n in (10, 20, 40, 80):
        for nc in (1, 4, 8):
            a, r = cla(elliptic(6, n_half=n), nc=nc)
            conv.append((n, nc, a, r["e"]))
    log("   grid convergence AR 6 (strips/half, chordwise): " +
        ", ".join(f"({n},{nc}) {a:.4f} e {e:.4f}" for n, nc, a, e in conv))
    out["elliptic"] = ell

    # 2. flat wings vs DATCOM
    log("2. flat wings vs DATCOM lift slope")
    rect = []
    for AR in (1, 2, 4, 6, 10, 20):
        a, r = cla(trapezoid(AR, 1.0, 0.0))
        d = datcom(AR, 0.0, 1.0)
        rect.append((AR, a, d, r["e"]))
        log(f"   rectangular AR {AR:3d}: {a:.4f}  DATCOM {d:.4f}  ({100 * (a / d - 1):+.1f} %)  "
            f"e {r['e']:.3f}")
    sw = []
    for s in (0, 15, 30, 45, 60):
        a, _ = cla(trapezoid(5, 0.5, s))
        d = datcom(5, s, 0.5)
        sw.append((s, a, d))
        log(f"   AR 5, taper 0.5, sweep {s:2d}: {a:.4f}  DATCOM {d:.4f}  ({100 * (a / d - 1):+.1f} %)")
    out["rectangular"], out["swept"] = rect, sw

    # 3. compressibility
    log("3. compressibility (AR 6, taper 0.5, sweep 30)")
    cm = []
    for M in (0.0, 0.3, 0.5, 0.7, 0.8):
        a, _ = cla(trapezoid(6, 0.5, 30), mach=M)
        d = datcom(6, 30, 0.5, M)
        cm.append((M, a, d))
        log(f"   M {M:.1f}: {a:.4f}  DATCOM {d:.4f}  ({100 * (a / d - 1):+.1f} %)")
    out["mach"] = cm

    # 4. 2D limit with a CFD section polar (NACA 0012, Re 6e6, our solver)
    log("4. nonlinear sections: AR 100 rectangular wing with the CFD NACA 0012 polar")
    pol = json.loads((OUT / "phase3_n0012_polar.json").read_text())
    sp = W.SectionPolar.from_results(pol["results"], ar=100)
    polars = W.SpanPolars([(pol["re"], sp)])
    wg = W.Wing(100.0, 1.0, n_half=40)
    res = W.sweep_alpha(wg, np.arange(0, 17, 2.0), polars, np.full(len(wg.c), pol["re"]))
    lim = []
    for r in res:
        a_i = r["CL"] / (math.pi * wg.AR * 0.97)            # induced angle (rad), ~elliptic
        cl2 = sp(np.array([r["alpha"] - math.degrees(a_i)]))[0][0]
        lim.append((r["alpha"], r["CL"], float(cl2)))
        rel = f"({100 * (r['CL'] / cl2 - 1):+.1f} %)" if abs(cl2) > 0.05 else ""
        log(f"   alpha {r['alpha']:5.1f}: wing CL {r['CL']:.4f}  section Cl at alpha - alpha_i "
            f"{cl2:.4f}  {rel}  CDp {r['CDp']:.5f}")
    out["2d_limit"] = lim

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axs[0]
    ARs = np.array([x[0] for x in rect])
    ax.plot(ARs, [x[1] for x in rect], "bo-", label="VLM, rectangular")
    ax.plot(ARs, [x[2] for x in rect], "b--", label="DATCOM, rectangular")
    ax.plot([x[0] for x in ell], [x[1] for x in ell], "gs-", label="VLM, elliptic")
    A = np.linspace(1, 40, 100)
    ax.plot(A, 2 * np.pi / (1 + 2 / A), "g:", label="lifting line 2pi/(1+2/AR)")
    ax.set_xscale("log"); ax.set_xlabel("aspect ratio"); ax.set_ylabel("CL_alpha (1/rad)")
    ax.legend(); ax.grid(alpha=.3)
    ax = axs[1]
    ax.plot([x[0] for x in sw], [x[1] for x in sw], "bo-", label="VLM (AR 5, taper 0.5)")
    ax.plot([x[0] for x in sw], [x[2] for x in sw], "k--", label="DATCOM")
    ax.set_xlabel("quarter-chord sweep (deg)"); ax.set_ylabel("CL_alpha (1/rad)")
    ax.legend(); ax.grid(alpha=.3)
    ax = axs[2]
    ax.plot([x[0] for x in cm], [x[1] for x in cm], "ro-", label="VLM + Goethert (AR 6, 30 deg)")
    ax.plot([x[0] for x in cm], [x[2] for x in cm], "k--", label="DATCOM")
    ax.set_xlabel("Mach"); ax.set_ylabel("CL_alpha (1/rad)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "wing_validation.png", dpi=120); plt.close(fig)
    (OUT / "wing_report.txt").write_text("\n".join(lines) + "\n")
    (OUT / "wing_validation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
