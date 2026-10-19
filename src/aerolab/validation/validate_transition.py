"""
Transition validation (Phase 3, step E4): free-transition polars from the
airfoil tool vs XFOIL (same e^N criterion, N_crit 9, but a completely
different flow model: panel method + integral boundary layer).

  NACA 0012, Re 3e6    alpha -2..12
  NACA 2412, Re 3e6    alpha -2..10   (+ the fully turbulent polar of E3)
  NACA 0012, Re 1e6    alpha 0..8     (longer laminar runs)

Unit-level checks are in tests/test_transition.py (Blasius e^9, Hiemenz,
Howarth, and the transition points XFOIL gets from its own pressure
distribution reproduced to ~0.01 c).

    python src/run.py validate_transition          run the three polars (~45 min) + plots
    python src/run.py validate_transition --plot   plots/report from the saved polars
"""

import json
import shutil
import subprocess

import aerolab
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VAL = ROOT / "validation"
CASES = [("tr_n0012_re3e6", "naca0012", 3e6, "-2:12:2", "NACA 0012, Re 3e6"),
         ("tr_n2412_re3e6", "naca2412", 3e6, "-2:10:2", "NACA 2412, Re 3e6"),
         ("tr_n0012_re1e6", "naca0012", 1e6, "0:8:2", "NACA 0012, Re 1e6")]


def run():
    for folder, spec, re, al, _ in CASES:
        out = ROOT / "cases" / folder
        subprocess.run([sys.executable, *aerolab.script_argv(["airfoil.py"]), "polar", spec, "--re", str(re),
                        f"--alpha={al}", "--xfoil", "--out", str(out)], check=True, cwd=aerolab.SRC)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "--plot" not in sys.argv:
        run()
    for folder, *_ in CASES:
        shutil.copy(ROOT / "cases" / folder / "polar.json", VAL / f"transition_{folder}.json")
    fig, ax = plt.subplots(3, 3, figsize=(18, 14))
    lines = ["Free transition (e^N, N_crit 9) vs XFOIL (free transition, N_crit 9)", ""]
    for row, (folder, spec, re, al, title) in enumerate(CASES):
        d = json.loads((VAL / f"transition_{folder}.json").read_text())
        r = d["results"]
        a = np.array([x["alpha"] for x in r])
        cl = np.array([x["Cl"] for x in r]); cd = np.array([x["Cd"] for x in r])
        xu = np.array([x["x_tr_upper"] for x in r], float); xl = np.array([x["x_tr_lower"] for x in r], float)
        ok = np.array([x["status"].startswith("converged") for x in r])
        xf = next((v for k, v in d["xfoil"].items() if "free" in k), {})
        keys = sorted(xf, key=float)
        xa = np.array([float(k) for k in keys])
        xv = np.array([xf[k] for k in keys], float) if keys else np.zeros((0, 5))
        g = ax[row, 0]
        g.plot(a, cl, "rD-", label="this tool, free transition")
        if len(xa):
            g.plot(xa, xv[:, 0], "k:o", ms=4, mfc="none", label="XFOIL, free transition")
        g.set(xlabel="alpha (deg)", ylabel="Cl", title=title)
        g = ax[row, 1]
        g.plot(cd, cl, "rD-", label="this tool, free transition")
        if not ok.all():
            g.plot(cd[~ok], cl[~ok], "rs", ms=10, mfc="none", label="not fully steady")
        if len(xa):
            g.plot(xv[:, 1], xv[:, 0], "k:o", ms=4, mfc="none", label="XFOIL, free transition")
        if folder == "tr_n2412_re3e6" and (VAL / "phase3_n2412_polar.json").exists():
            t = json.loads((VAL / "phase3_n2412_polar.json").read_text())["results"]
            ta = np.array([x["alpha"] for x in t]); m = ta <= 10
            g.plot(np.array([x["Cd"] for x in t])[m], np.array([x["Cl"] for x in t])[m], "b.--",
                   label="this tool, fully turbulent (E3)")
        g.set(xlabel="Cd", ylabel="Cl", title="drag polar")
        g = ax[row, 2]
        g.plot(a, xu, "rD-", label="this tool, upper")
        g.plot(a, xl, "rs--", mfc="none", label="this tool, lower")
        if len(xa):
            g.plot(xa, xv[:, 3], "k:o", ms=4, label="XFOIL, upper")
            g.plot(xa, xv[:, 4], "k-.^", ms=4, mfc="none", label="XFOIL, lower")
        g.set(xlabel="alpha (deg)", ylabel="transition x/c", ylim=(-0.03, 1.05), title="transition location")
        for q in ax[row]:
            q.grid(alpha=.3); q.legend(fontsize=7)
        lines += [title, f"  {'alpha':>5s} {'Cl':>7s} {'XFOIL':>7s} {'Cd':>8s} {'XFOIL':>8s} {'dCd':>6s}   "
                  f"{'xtr up':>6s} {'XFOIL':>6s}  {'xtr lo':>6s} {'XFOIL':>6s}  status"]
        for i, al_ in enumerate(a):
            j = np.where(np.isclose(xa, al_))[0]
            if len(j):
                v = xv[j[0]]
                lines.append(f"  {al_:5.1f} {cl[i]:7.4f} {v[0]:7.4f} {cd[i]:8.5f} {v[1]:8.5f} "
                             f"{100 * (cd[i] / v[1] - 1):+5.1f}%   {xu[i]:6.3f} {v[3]:6.3f}  {xl[i]:6.3f} "
                             f"{v[4]:6.3f}  {r[i]['status']}")
        lines.append("")
    fig.suptitle("Transition prediction: airfoil tool (RANS SA + e^N) vs XFOIL")
    fig.tight_layout(); fig.savefig(VAL / "transition_polars.png", dpi=110); plt.close(fig)
    (VAL / "transition_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved", VAL / "transition_polars.png")


if __name__ == "__main__":
    main()
