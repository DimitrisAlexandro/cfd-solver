"""
Independent reference data from XFOIL (M. Drela, MIT) for the airfoil
validation: Cl, Cd vs alpha and surface Cp.

XFOIL uses a completely different model (panel method + integral boundary
layer) from our finite-volume Navier-Stokes solver, so agreement between the
two is a meaningful cross-check. It gets exactly the same airfoil
coordinates as the mesher (blunt trailing edge).

The output is saved in ../validation/reference/, so XFOIL is only needed to
REGENERATE it. Install (Ubuntu's packaged xfoil crashes on floating-point
traps; this builds a working one from source):
    sudo apt install gfortran cmake
    pip install "git+https://github.com/DARcorporation/xfoil-python.git"

    python src/run.py xfoil_ref
"""

from pathlib import Path

import numpy as np

from aerolab.mesh.geometry import naca4

OUT = Path(__file__).resolve().parents[3] / "validation" / "reference"


def surface(code="0012", n_side=160):
    """Upper TE -> LE -> lower TE (XFOIL closes a blunt TE itself)."""
    loop = naca4(code, n_side=n_side, n_te=2)
    return loop[:2 * n_side - 1]


def run(code="0012", re=1000.0, alphas=range(0, 8), n_crit=14.0):
    from xfoil import XFoil
    from xfoil.model import Airfoil

    xf = XFoil()
    xf.print = False
    s = surface(code)
    xf.airfoil = Airfoil(s[:, 0].copy(), s[:, 1].copy())
    xf.repanel(n_nodes=240)
    xf.Re = re
    xf.n_crit = n_crit            # stays laminar at these Re anyway
    xf.max_iter = 300
    rows, cps = [], {}
    for a in alphas:
        cl, cd, cm, _ = xf.a(float(a))
        if np.isfinite(cl) and np.isfinite(cd):
            rows.append([a, cl, cd, cm])
            d = xf.get_cp_distribution()          # (x, y, cp) or (x, cp)
            cps[a] = np.c_[d[0], d[-1]]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"naca{code}_re{int(re)}"
    np.savetxt(OUT / f"xfoil_{tag}_polar.txt", np.array(rows), header="alpha CL CD CM",
               fmt="%.6f")
    for a, d in cps.items():
        np.savetxt(OUT / f"xfoil_{tag}_cp_a{a:g}.txt", d, header="x Cp", fmt="%.6f")
    return np.array(rows)


if __name__ == "__main__":
    for re in (1000.0, 2000.0, 5000.0):
        p = run(re=re)
        print(f"XFOIL NACA 0012, Re = {re:g}")
        for r in p:
            print(f"  alpha {r[0]:4.1f}   Cl {r[1]:8.4f}   Cd {r[2]:8.5f}")
