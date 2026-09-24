"""
Validation 2: lid-driven cavity (unit square, lid u = 1, Re = 1/nu).

References
  Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387-411
      centreline velocity tables (129x129 multigrid, 2nd order) - Re 100, 1000
  Botella & Peyret (1998), Computers & Fluids 27, 421-433
      spectral benchmark (to ~6 digits) - Re 1000 centreline extrema
  (Values below as commonly tabulated from these papers; check the
   originals before quoting them in the thesis.)

What it proves: correct convection-diffusion-pressure coupling in a
recirculating flow, grid convergence toward a high-accuracy reference,
and that unstructured triangles give the same answer.

    python validate_cavity.py        -> ../validation/cavity_*.png
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from fvmesh import FVMesh, rect_quad, rect_tri
from simple_solver import SimpleSolver

# --- Ghia et al. 1982 -------------------------------------------------------
GHIA_Y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
                   0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
GHIA_U = {100: [1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641,
                -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192,
                -0.03717, 0.00000],
          1000: [1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702,
                 -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196,
                 -0.18109, 0.00000]}
GHIA_X = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
                   0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
GHIA_V = {100: [0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445,
                -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091,
                0.09233, 0.00000],
          1000: [0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665,
                 -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012,
                 0.27485, 0.00000]}
# --- Botella & Peyret 1998, Re = 1000 ----------------------------------------
BOTELLA_1000 = {"u_min": -0.388569, "y_umin": 0.1717,
                "v_max": 0.376944, "x_vmax": 0.1578,
                "v_min": -0.527077, "x_vmin": 0.9092}

BCS = {"top": {"type": "wall", "U": (1.0, 0.0)}, "bottom": {"type": "wall"},
       "left": {"type": "wall"}, "right": {"type": "wall"}}


def solve(Re, mesh, tol=1e-7, verbose=False):
    s = SimpleSolver(mesh, BCS, 1.0 / Re)
    t0 = time.time()
    it = s.solve(tol=tol, max_iter=20000, verbose=verbose, every=500)
    return s, it, time.time() - t0


def interpolator(s, k):
    m = s.m
    Ub = np.where(s.b_fixedU[:, None], s.Ub, s.U[m.bcell])
    pts = np.vstack([m.xc, m.bxf, [[0, 0], [1, 0], [0, 1], [1, 1]]])
    corner = [0, 0, 0, 0] if k == 1 else [0, 0, 1, 1]      # lid corners: u = 1
    val = np.r_[s.U[:, k], Ub[:, k], corner]
    return LinearNDInterpolator(pts, val)


def centerlines(s, n=2001):
    t = np.linspace(0, 1, n)
    u = interpolator(s, 0)(np.full(n, 0.5), t)     # u(y) on x = 0.5
    v = interpolator(s, 1)(t, np.full(n, 0.5))     # v(x) on y = 0.5
    return t, u, v


def extremum(t, f, kind):
    """Extremum of a sampled profile, refined with a local quadratic fit."""
    i = np.nanargmin(f) if kind == "min" else np.nanargmax(f)
    k = max(10, len(t) // 200)
    sl = slice(max(i - k, 0), min(i + k + 1, len(t)))
    c = np.polyfit(t[sl], f[sl], 2)
    tx = -c[1] / (2 * c[0])
    return np.polyval(c, tx), tx


def botella_errors(s):
    t, u, v = centerlines(s)
    umin, yu = extremum(t, u, "min")
    vmax, xv = extremum(t, v, "max")
    vmin, xw = extremum(t, v, "min")
    B = BOTELLA_1000
    return {"u_min": umin, "v_max": vmax, "v_min": vmin, "y_umin": yu, "x_vmax": xv, "x_vmin": xw,
            "err_u_min": abs(umin / B["u_min"] - 1) * 100,
            "err_v_max": abs(vmax / B["v_max"] - 1) * 100,
            "err_v_min": abs(vmin / B["v_min"] - 1) * 100}


def main(out="../validation", quick=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(out).mkdir(parents=True, exist_ok=True)
    grids = (17, 33, 65) if quick else (33, 65, 129)
    fine = grids[-1]

    # 1) profiles vs Ghia on the finest quad grid ---------------------------
    runs = {}
    for Re in (100, 1000):
        s, it, dt = solve(Re, FVMesh(*rect_quad(fine, fine)))
        runs[Re] = s
        print(f"Re={Re:5d} quad {fine}x{fine}: {it} iterations, {dt:.0f} s")

    # 2) Re=1000 grid convergence vs Botella & Peyret ----------------------
    conv = []
    for n in grids:
        s = runs[1000] if n == fine else solve(1000, FVMesh(*rect_quad(n, n)))[0]
        conv.append((n, botella_errors(s)))

    # 3) unstructured triangles --------------------------------------------
    htri = 1 / 32 if quick else 1 / 64
    mt = FVMesh(*rect_tri(htri))
    st, it, dt = solve(1000, mt)
    tri_err = botella_errors(st)
    print(f"Re= 1000 triangles ({mt.ncells} cells): {it} iterations, {dt:.0f} s")
    st100, _, _ = solve(100, mt)

    print("\nRe = 1000 centreline extrema vs Botella & Peyret (1998):")
    B = BOTELLA_1000
    print(f"{'grid':>16s} {'u_min':>9s} {'err%':>6s} {'v_max':>9s} {'err%':>6s} "
          f"{'v_min':>9s} {'err%':>6s}")
    for n, e in conv:
        print(f"{f'quad {n}x{n}':>16s} {e['u_min']:9.5f} {e['err_u_min']:6.2f} {e['v_max']:9.5f} "
              f"{e['err_v_max']:6.2f} {e['v_min']:9.5f} {e['err_v_min']:6.2f}")
    e = tri_err
    print(f"{f'tri {mt.ncells}':>16s} {e['u_min']:9.5f} {e['err_u_min']:6.2f} {e['v_max']:9.5f} "
          f"{e['err_v_max']:6.2f} {e['v_min']:9.5f} {e['err_v_min']:6.2f}")
    print(f"{'Botella (ref)':>16s} {B['u_min']:9.5f} {'':6s} {B['v_max']:9.5f} {'':6s} "
          f"{B['v_min']:9.5f}")
    e1, e2 = conv[-2][1], conv[-1][1]
    orders = {k: np.log(e1[f"err_{k}"] / e2[f"err_{k}"]) / np.log(2)
              for k in ("u_min", "v_max", "v_min")}
    print("observed order (last two grids): " +
          ", ".join(f"{k} {v:.2f}" for k, v in orders.items()))

    # Ghia comparison numbers
    for Re, s in runs.items():
        iu, iv = interpolator(s, 0), interpolator(s, 1)
        du = np.abs(iu(np.full_like(GHIA_Y, 0.5), GHIA_Y) - GHIA_U[Re]).max()
        dv = np.abs(iv(GHIA_X, np.full_like(GHIA_X, 0.5)) - GHIA_V[Re]).max()
        print(f"Re={Re}: max |u - Ghia| = {du:.4f}, max |v - Ghia| = {dv:.4f} (lid velocity = 1)")

    # plots ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, Re in zip(axes, (100, 1000)):
        s = runs[Re]
        stri = st100 if Re == 100 else st
        t, u, v = centerlines(s)
        _, ut, vt = centerlines(stri)
        ax.plot(u, t, "b-", label=f"u(y) at x=0.5, quads {fine}²")
        ax.plot(ut, t, "b:", label=f"u(y), triangles ({mt.ncells} cells)")
        ax.plot(GHIA_U[Re], GHIA_Y, "bo", mfc="none", label="Ghia et al. 1982")
        ax.plot(t, v, "r-", label="v(x) at y=0.5, quads")
        ax.plot(t, vt, "r:", label="v(x), triangles")
        ax.plot(GHIA_X, GHIA_V[Re], "rs", mfc="none", label="Ghia et al. 1982")
        ax.set_title(f"Lid-driven cavity, Re = {Re}")
        ax.set_xlabel("u  /  x"); ax.set_ylabel("y  /  v"); ax.grid(alpha=.3)
        ax.legend(fontsize=7.5, loc="center right" if Re == 100 else "upper left")
    fig.tight_layout(); fig.savefig(f"{out}/cavity_profiles.png", dpi=140)

    fig, ax = plt.subplots(figsize=(6, 4.8))
    hs = np.array([1 / n for n, _ in conv])
    for k, mk in (("u_min", "o-"), ("v_max", "s-"), ("v_min", "^-")):
        ax.loglog(hs, [e[f"err_{k}"] for _, e in conv], mk, label=k)
    ax.loglog(hs, conv[0][1]["err_u_min"] * (hs / hs[0]) ** 2, "k--", label="slope 2")
    ax.set_xlabel("cell size h"); ax.set_ylabel("% error vs Botella & Peyret")
    ax.set_title("Re = 1000 grid convergence"); ax.grid(alpha=.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out}/cavity_convergence.png", dpi=140)

    # streamlines Re=1000
    s = runs[1000]
    g = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(g, g)
    U = interpolator(s, 0)(X, Y); V = interpolator(s, 1)(X, Y)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.streamplot(g, g, U, V, density=2.2, linewidth=0.6, color=np.hypot(U, V), cmap="viridis")
    ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Re = 1000 streamlines (quads {fine}²)")
    fig.tight_layout(); fig.savefig(f"{out}/cavity_streamlines.png", dpi=140)
    return conv, tri_err


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
