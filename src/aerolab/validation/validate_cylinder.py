"""
Validation 3: steady flow past a circular cylinder, Re = 20 and 40,
on a HYBRID mesh from our own mesher (quad boundary layer + triangles) -
the same pipeline that produces the airfoil meshes.

Quantities: drag coefficient Cd, recirculation length L/D (from the rear
of the cylinder), separation angle measured from the rear stagnation point.

Reference values (as commonly cited; verify against the originals):
  Dennis & Chang (1970), J. Fluid Mech. 42      numerical
  Fornberg (1980), J. Fluid Mech. 98            numerical
  Coutanceau & Bouard (1977), J. Fluid Mech. 79 experiment (L/D, angle)
  Tritton (1959), J. Fluid Mech. 6              experiment (Cd)

    python src/run.py validate_cylinder      -> ../validation/cylinder_*.png
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from aerolab.mesh import hybrid_mesher as hm
from aerolab.core.fvmesh import FVMesh
from aerolab.mesh.geometry import circle
from aerolab.core.simple_solver import SimpleSolver

REFERENCE = {
    20: {"Cd": {"Dennis & Chang 1970": 2.045, "Fornberg 1980": 2.00, "Tritton 1959 (exp)": 2.22},
         "L/D": {"Dennis & Chang 1970": 0.94, "Fornberg 1980": 0.91, "Coutanceau & Bouard 1977 (exp)": 0.93},
         "theta": {"Dennis & Chang 1970": 43.7, "Coutanceau & Bouard 1977 (exp)": 44.8}},
    40: {"Cd": {"Dennis & Chang 1970": 1.522, "Fornberg 1980": 1.498, "Tritton 1959 (exp)": 1.59},
         "L/D": {"Dennis & Chang 1970": 2.35, "Fornberg 1980": 2.24, "Coutanceau & Bouard 1977 (exp)": 2.13},
         "theta": {"Dennis & Chang 1970": 53.8, "Coutanceau & Bouard 1977 (exp)": 53.5}},
}


def cylinder_mesh(n_wall=200, h1=0.004, farfield=60.0, refine=1.0, verbose=False):
    """Cylinder D=1 centred at (0.5, 0) so it spans x in [0,1] like a chord.
    refine > 1 makes every length scale finer by that factor."""
    r = refine
    w = circle((0.5, 0.0), 0.5, int(round(n_wall * r)))
    mesh = hm.build_mesh(wall=w, h1=h1 / r, growth=1.1 ** (1 / r),
                         farfield_radius=farfield, n_farfield=int(80 * r),
                         h_near=0.03 / r, grad=0.1 / r, wake_length=4.0,
                         wake_h=0.04 / r, wake_grad=0.03 / r,
                         label=f"cylinder (refine x{r:g})", verbose=verbose)
    P = mesh["points"]
    # split the farfield into inlet (flow enters) and outlet by the normal
    ff = mesh["patches"]["farfield"]
    t = P[ff[:, 1]] - P[ff[:, 0]]
    n_x = t[:, 1]                       # outward normal = (ty, -tx)
    patches = {"cylinder": mesh["patches"]["airfoil"],
               "inlet": ff[n_x < 0], "outlet": ff[n_x >= 0]}
    return FVMesh(P, mesh["cells"], patches), mesh


def analyse(s):
    m = s.m
    Fp, Fv = s.force("cylinder")
    Cd_p, Cd_v = 2 * Fp[0], 2 * Fv[0]                 # 0.5 rho U^2 D = 0.5
    # recirculation length on the wake centreline
    Ub = np.where(s.b_fixedU[:, None], s.Ub, s.U[m.bcell])
    near = np.hypot(m.xc[:, 0] - 2, m.xc[:, 1]) < 4
    nb = np.hypot(m.bxf[:, 0] - 2, m.bxf[:, 1]) < 4
    f = LinearNDInterpolator(np.vstack([m.xc[near], m.bxf[nb]]), np.r_[s.U[near, 0], Ub[nb, 0]])
    x = np.linspace(1.0, 5.0, 4001)
    u = f(x, np.zeros_like(x))
    i = np.where((u[:-1] < 0) & (u[1:] >= 0))[0]
    L = (x[i[0]] - u[i[0]] * (x[i[0] + 1] - x[i[0]]) / (u[i[0] + 1] - u[i[0]]) - 1.0) if len(i) else 0.0
    # separation angle from the rear stagnation point (upper half)
    idx, pb, tau, S, Sm = s.wall_values("cylinder")
    xf = m.bxf[idx]
    th = np.degrees(np.arctan2(xf[:, 1], xf[:, 0] - 0.5))       # 0 at the rear
    up = th > 0
    order = np.argsort(th[up])
    thu = th[up][order]
    tdir = np.c_[-np.sin(np.radians(thu)), np.cos(np.radians(thu))]  # towards the front
    ts = np.einsum('ij,ij->i', tau[up][order], tdir)                 # >0: reversed flow
    j = np.where((ts[:-1] > 0) & (ts[1:] <= 0))[0]
    theta = thu[j[0]] - ts[j[0]] * (thu[j[0] + 1] - thu[j[0]]) / (ts[j[0] + 1] - ts[j[0]]) if len(j) else 0.0
    return {"Cd": Cd_p + Cd_v, "Cd_p": Cd_p, "Cd_v": Cd_v, "L/D": L, "theta": theta,
            "Cl": 2 * (Fp[1] + Fv[1])}


def run(Re, refine=1.0, verbose=False):
    fvm, mesh = cylinder_mesh(refine=refine)
    bcs = {"cylinder": {"type": "wall"}, "inlet": {"type": "inlet", "U": (1.0, 0.0)},
           "outlet": {"type": "outlet"}}
    s = SimpleSolver(fvm, bcs, 1.0 / Re, U0=(1.0, 0.0))
    t0 = time.time()
    it = s.solve(tol=1e-7, max_iter=20000, verbose=verbose, every=500)
    res = analyse(s)
    res.update(cells=fvm.ncells, iters=it, time=time.time() - t0)
    return res, s, mesh


def main(out="../validation", quick=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(out).mkdir(parents=True, exist_ok=True)
    refines = (0.7, 1.0) if quick else (0.7, 1.0, 1.4)
    results, solvers = {}, {}
    for Re in (20, 40):
        for r in refines:
            res, s, mesh = run(Re, r)
            results[(Re, r)] = res
            solvers[(Re, r)] = (s, mesh)
            print(f"Re={Re} refine x{r}: {res['cells']} cells, {res['iters']} iters, "
                  f"{res['time']:.0f}s -> Cd={res['Cd']:.4f} (p {res['Cd_p']:.3f} + v {res['Cd_v']:.3f}), "
                  f"L/D={res['L/D']:.3f}, theta={res['theta']:.1f} deg, Cl={res['Cl']:.1e}")

    print("\nFinest mesh vs literature:")
    for Re in (20, 40):
        r = results[(Re, refines[-1])]
        print(f" Re = {Re}")
        for q, lab in (("Cd", "Cd"), ("L/D", "L/D"), ("theta", "separation angle (deg)")):
            refs = ", ".join(f"{k} {v}" for k, v in REFERENCE[Re][q].items())
            print(f"   {lab:24s} this solver {r[q]:.3f}   | {refs}")

    # plots: streamlines + mesh for Re=40 finest
    s, mesh = solvers[(40, refines[-1])]
    m = s.m
    near = np.hypot(m.xc[:, 0] - 1.5, m.xc[:, 1]) < 4
    X, Y = np.meshgrid(np.linspace(-1, 4, 400), np.linspace(-1.5, 1.5, 240))
    fu = LinearNDInterpolator(m.xc[near], s.U[near, 0])(X, Y)
    fv = LinearNDInterpolator(m.xc[near], s.U[near, 1])(X, Y)
    inside = np.hypot(X - 0.5, Y) < 0.5
    fu[inside] = np.nan; fv[inside] = np.nan
    fig, ax = plt.subplots(2, 1, figsize=(10, 9))
    cs = ax[0].contourf(X, Y, fu, levels=np.linspace(-0.3, 1.3, 33), cmap="RdBu_r", extend="both")
    ax[0].streamplot(X[0], Y[:, 0], np.nan_to_num(fu), np.nan_to_num(fv), density=1.6,
                     color="k", linewidth=0.5, arrowsize=0.6)
    ax[0].add_patch(plt.Circle((0.5, 0), 0.5, color="#444444"))
    fig.colorbar(cs, ax=ax[0], label="u / U")
    r40 = results[(40, refines[-1])]
    ax[0].axvline(1 + r40["L/D"], color="lime", lw=1.5, ls="--")
    ax[0].set_title(f"Re = 40: u-velocity and streamlines (recirculation length L/D = {r40['L/D']:.2f})")
    ax[0].set_aspect("equal"); ax[0].set_xlim(-1, 4); ax[0].set_ylim(-1.5, 1.5)
    from matplotlib.collections import PolyCollection
    P = mesh["points"]
    ax[1].add_collection(PolyCollection([P[c] for c in mesh["cells"][mesh["n_bl"]:]],
                                        facecolor="white", edgecolor="#4a6fa5", lw=0.3))
    ax[1].add_collection(PolyCollection([P[c] for c in mesh["cells"][:mesh["n_bl"]]],
                                        facecolor="#dce9f5", edgecolor="#c0504d", lw=0.3))
    ax[1].set_aspect("equal"); ax[1].set_xlim(-1, 4); ax[1].set_ylim(-1.5, 1.5)
    ax[1].set_title(f"Hybrid mesh from our mesher ({m.ncells} cells)")
    fig.tight_layout(); fig.savefig(f"{out}/cylinder_flow.png", dpi=140)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax_, q, lab in ((axes[0], "Cd", "Cd"), (axes[1], "L/D", "L/D")):
        for Re, mk in ((20, "o-"), (40, "s-")):
            n = [results[(Re, r)]["cells"] for r in refines]
            ax_.plot(n, [results[(Re, r)][q] for r in refines], mk, label=f"this solver, Re={Re}")
            for k, v in REFERENCE[Re][q].items():
                ax_.axhline(v, lw=0.8, ls=":", color="C0" if Re == 20 else "C1")
        ax_.set_xscale("log"); ax_.set_xlabel("cells"); ax_.set_ylabel(lab)
        ax_.set_title(f"{lab} vs mesh size (dotted: literature)"); ax_.grid(alpha=.3)
        ax_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/cylinder_convergence.png", dpi=140)
    return results


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
