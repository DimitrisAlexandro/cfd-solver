"""
Phase 3 validation (E1-E3): complete polars made with the airfoil tool.

  1. NACA 0012 (NASA TMR shape), Re 6e6, alpha -4..16, SA fully turbulent,
     incompressible; NASA's CFL3D reference and Ladson's experiment are at
     M = 0.15, so Cl/Cm are also compared after the Prandtl-Glauert factor
     1/sqrt(1 - 0.15^2) = 1.0114 (this explains the ~1.3 % lift offset);
     vs NASA CFL3D SA (alpha 0, 10, 15), NASA grid-converged values (alpha 10,
     incl. Cm) and Ladson's tripped wind-tunnel data (NASA TM 4074).
  2. NACA 2412, Re 3e6, alpha -4..16, SA fully turbulent, vs XFOIL tripped at
     x/c = 0.01 (independent method; comparable only while both surfaces are
     turbulent from the leading edge, i.e. low alpha) - checks camber, Cm and
     the .dat/NACA geometry path.

    python src/run.py validate_phase3          run both polars (~8 min each) + plots
    python src/run.py validate_phase3 --plot   plots/report from the saved polars
"""

import json
import shutil
import subprocess

import aerolab
import sys
from pathlib import Path

import numpy as np

from aerolab.validation.validate_flatplate import tmr_zones

ROOT = Path(__file__).resolve().parents[3]
VAL = ROOT / "validation"
TMR = VAL / "reference" / "tmr"
CASES = {"n0012": ("tmr0012", 6e6, ROOT / "cases" / "polar_n0012_re6e6"),
         "n2412": ("naca2412", 3e6, ROOT / "cases" / "polar_n2412_re3e6")}
NASA_A10 = {"Cl": 1.0912, "Cd": 0.01221, "Cm": 0.00678}     # grid-converged SA, alpha 10


def run():
    for k, (spec, re, out) in CASES.items():
        subprocess.run([sys.executable, *aerolab.script_argv(["airfoil.py"]), "polar", spec, "--re", str(re),
                        "--alpha=-4:16:2", "--xfoil", "--transition", "off",   # NASA: fully turbulent
                        "--out", str(out)], check=True, cwd=aerolab.SRC)
        shutil.copy(out / "polar.json", VAL / f"phase3_{k}_polar.json")


def load(k):
    d = json.loads((VAL / f"phase3_{k}_polar.json").read_text())
    r = d["results"]
    a = np.array([x["alpha"] for x in r])
    return d, a, {q: np.array([x[q] for x in r]) for q in ("Cl", "Cd", "Cm", "Cd_p", "Cd_v")}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "--plot" not in sys.argv:
        run()
    lines = []
    # ---- NACA 0012 -------------------------------------------------------------
    d, a, c = load("n0012")
    cfl = next(iter(tmr_zones(TMR / "n0012clcd_cfl3d_sa.dat").values()))
    lad = tmr_zones(TMR / "CLCD_Ladson_expdata.dat")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for (name, z), mk in zip(lad.items(), ("o", "s", "^")):
        ax[0].plot(z[:, 0], z[:, 1], mk, mfc="none", color="gray", ms=5, label=f"Ladson, {name} (exp.)")
        ax[1].plot(z[:, 2], z[:, 1], mk, mfc="none", color="gray", ms=5)
    ax[0].plot(cfl[:, 0], cfl[:, 1], "k*", ms=12, label="NASA CFL3D SA (897x257)")
    ax[1].plot(cfl[:, 2], cfl[:, 1], "k*", ms=12)
    PG = 1 / np.sqrt(1 - 0.15**2)
    ax[0].plot(a, c["Cl"], "rD-", ms=5, label="this tool, incompressible")
    ax[0].plot(a, c["Cl"] * PG, "b.--", ms=5, label="this tool, Prandtl-Glauert M 0.15")
    ax[1].plot(c["Cd"], c["Cl"], "rD-", ms=5)
    ax[1].plot(c["Cd_p"] * PG + c["Cd_v"], c["Cl"] * PG, "b.--", ms=5)
    ax[2].plot(a, c["Cm"], "rD-", ms=5, label="this tool, incompressible")
    ax[2].plot(a, c["Cm"] * PG, "b.--", ms=5, label="this tool, Prandtl-Glauert M 0.15")
    ax[2].plot(10, NASA_A10["Cm"], "k*", ms=14, label="NASA grid-converged SA, alpha 10")
    ax[0].set(xlabel="alpha (deg)", ylabel="Cl", xlim=(-5, 18))
    ax[1].set(xlabel="Cd", ylabel="Cl", xlim=(0.005, 0.03))
    ax[2].set(xlabel="alpha (deg)", ylabel="Cm about c/4 (nose-up +)")
    for g in ax:
        g.grid(alpha=.3)
    ax[0].legend(fontsize=7); ax[2].legend(fontsize=7)
    fig.suptitle("NACA 0012, Re = 6e6: complete polar from the airfoil tool vs NASA and experiment")
    fig.tight_layout(); fig.savefig(VAL / "phase3_naca0012_polar.png", dpi=130); plt.close(fig)

    lines += [f"NACA 0012 (TMR), Re 6e6, SA fully turbulent - {len(a)} angles, "
              f"{d['results'][0]['cells']} cells each; references at M = 0.15, "
              f"PG = Prandtl-Glauert corrected (x{PG:.4f})", "",
              "  vs NASA CFL3D SA (897x257):"]
    for al, cl0, cd0 in cfl:
        if al in a:
            i = int(np.where(a == al)[0][0])
            lines.append(f"    alpha {al:4.0f}: Cl {c['Cl'][i]:8.4f} (PG {c['Cl'][i] * PG:.4f}) vs {cl0:8.4f}"
                         f"   Cd {c['Cd'][i]:.5f} vs {cd0:.5f} ({100 * (c['Cd'][i] / cd0 - 1):+.1f} %)")
    i = int(np.where(a == 10)[0][0])
    lines.append(f"  vs NASA grid-converged SA, alpha 10: Cl {c['Cl'][i]:.4f} "
                 f"({100 * (c['Cl'][i] / NASA_A10['Cl'] - 1):+.1f} %; PG {c['Cl'][i] * PG:.4f}, "
                 f"{100 * (c['Cl'][i] * PG / NASA_A10['Cl'] - 1):+.1f} %) vs {NASA_A10['Cl']};  "
                 f"Cd {c['Cd'][i]:.5f} vs {NASA_A10['Cd']} ({100 * (c['Cd'][i] / NASA_A10['Cd'] - 1):+.1f} %);  "
                 f"Cm {c['Cm'][i]:+.5f} (PG {c['Cm'][i] * PG:+.5f}) vs {NASA_A10['Cm']:+.5f}")
    lo = (a >= -4) & (a <= 8)
    slope = np.polyfit(np.radians(a[lo]), c["Cl"][lo], 1)[0]
    z = lad["80 grit"]
    zs = np.polyfit(np.radians(z[(z[:, 0] >= -4) & (z[:, 0] <= 8), 0]),
                    z[(z[:, 0] >= -4) & (z[:, 0] <= 8), 1], 1)[0]
    lines.append(f"  lift slope (-4..8 deg): {slope:.3f} /rad (PG {slope * PG:.3f}) vs Ladson 80 grit "
                 f"{zs:.3f} /rad ({100 * (slope * PG / zs - 1):+.1f} % after PG)")

    # ---- NACA 2412 ---------------------------------------------------------------
    d, a, c = load("n2412")
    xf = {float(k): v for k, v in next((v for lab, v in d["xfoil"].items() if "tripped" in lab), {}).items()}
    xfree = {float(k): v for k, v in next((v for lab, v in d["xfoil"].items() if "free" in lab), {}).items()}
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for dd, st, lab in ((xf, "k--", "XFOIL tripped x/c 0.01"), (xfree, "k:", "XFOIL free transition")):
        xa = np.array(sorted(dd)); v = np.array([dd[k] for k in xa])
        if len(xa):
            ax[0].plot(xa, v[:, 0], st, label=lab); ax[1].plot(v[:, 1], v[:, 0], st)
            ax[2].plot(xa, v[:, 2], st)
    ax[0].plot(a, c["Cl"], "rD-", ms=5, label="this tool (SA, fully turbulent)")
    ax[1].plot(c["Cd"], c["Cl"], "rD-", ms=5); ax[2].plot(a, c["Cm"], "rD-", ms=5)
    ax[0].set(xlabel="alpha (deg)", ylabel="Cl"); ax[1].set(xlabel="Cd", ylabel="Cl")
    ax[2].set(xlabel="alpha (deg)", ylabel="Cm about c/4")
    for g in ax:
        g.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("NACA 2412, Re = 3e6 (cambered): airfoil tool vs XFOIL")
    fig.tight_layout(); fig.savefig(VAL / "phase3_naca2412_polar.png", dpi=130); plt.close(fig)
    lines += ["", "NACA 2412, Re 3e6, SA fully turbulent vs XFOIL tripped at x/c 0.01 "
              "(both turbulent from the LE: compare at low alpha):"]
    for al in (-4, -2, 0, 2, 4):
        if al in xf and al in a:
            i = int(np.where(a == al)[0][0])
            v = xf[al]
            lines.append(f"    alpha {al:3d}: Cl {c['Cl'][i]:7.4f} vs {v[0]:7.4f}   Cd {c['Cd'][i]:.5f} vs "
                         f"{v[1]:.5f} ({100 * (c['Cd'][i] / v[1] - 1):+.1f} %)   Cm {c['Cm'][i]:+.4f} vs {v[2]:+.4f}")
    i0 = np.argmin(np.abs(c["Cl"]))
    a0 = np.interp(0.0, c["Cl"], a)
    lines.append(f"  zero-lift angle {a0:.2f} deg (thin-airfoil theory for NACA 2412: -2.08 deg)")
    (VAL / "phase3_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved", VAL / "phase3_naca0012_polar.png", VAL / "phase3_naca2412_polar.png")


if __name__ == "__main__":
    main()
