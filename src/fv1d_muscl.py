"""
1D Finite Volume solver — second-order MUSCL reconstruction with a slope
limiter, applied to linear advection and inviscid Burgers' equation.

This extends fv1d_advection.py (first-order upwind) with:
  - piecewise-linear reconstruction inside each cell (instead of piecewise
    constant), using a limited slope so no new oscillations are created
    near discontinuities (this is the "TVD" property)
  - a choice of limiters: minmod, van Leer, superbee

Why this matters for the mesher/solver roadmap: at a 2D/3D face, the flux
is computed from *reconstructed* left/right states at the face, not raw
cell averages. The reconstruction step here (get uL, uR at a face from
neighboring cell data + limited slopes) is exactly the same operation
you'll do per-face once cells have real neighbors from the mesh.

Author: Dimitris
"""

import numpy as np
import matplotlib.pyplot as plt

from fv1d_advection import (
    Grid1D, ic_square_wave, ic_gaussian, ic_sine_hump,
    flux_burgers, apply_bc,
)


# --------------------------------------------------------------------------
# Slope limiters
# --------------------------------------------------------------------------
# Each limiter takes two consecutive differences (a = u_i - u_{i-1},
# b = u_{i+1} - u_i) and returns a limited slope estimate for cell i.
# All reduce to 0 at extrema (no new min/max created) and to the
# unlimited central slope in smooth regions -> TVD, second order in smooth
# regions, first order (upwind-like) near discontinuities.

def minmod(a, b):
    """Most diffusive limiter — safest, most robust."""
    s = np.sign(a)
    return np.where(s == np.sign(b), s * np.minimum(np.abs(a), np.abs(b)), 0.0)


def van_leer(a, b):
    """Smoother than minmod, less diffusive, still TVD."""
    ab = a * b
    return np.where(ab > 0, 2 * ab / (a + b + 1e-14), 0.0)


def superbee(a, b):
    """Least diffusive (sharpest), can be too aggressive on smooth data."""
    s1 = minmod(a, 2 * b)
    s2 = minmod(2 * a, b)
    same_sign = np.sign(a) == np.sign(b)
    out = np.where(np.abs(s1) > np.abs(s2), s1, s2)
    return np.where(same_sign, out, 0.0)


LIMITERS = {"minmod": minmod, "van_leer": van_leer, "superbee": superbee}


# --------------------------------------------------------------------------
# MUSCL reconstruction
# --------------------------------------------------------------------------

def reconstruct_faces(u_ext, limiter_fn):
    """
    Given cell averages with 2 ghost cells on each side (length n+4),
    compute limited slopes for each real cell, then the left/right
    reconstructed states at every interior face.

    Returns (uL, uR): the left-state and right-state arrays at each of
    the n+1 interfaces (i.e. uL[i], uR[i] are the states on the two
    sides of face i-1/2, for i = 0..n... indices arranged so that
    uL, uR line up face-by-face, length n+1).
    """
    # u_ext has 2 ghost cells each side: indices 0,1 = left ghosts,
    # -2,-1 = right ghosts, 2:-2 = real cells.
    n_ext = len(u_ext)

    # backward/forward differences for every cell that has both neighbors
    a = u_ext[1:-1] - u_ext[:-2]   # u_i - u_{i-1}, length n_ext-2
    b = u_ext[2:] - u_ext[1:-1]    # u_{i+1} - u_i, length n_ext-2

    slope = limiter_fn(a, b)  # limited slope per cell, length n_ext-2
    # slope[k] corresponds to u_ext[k+1]

    # We need, for each interior face between u_ext[i] and u_ext[i+1]
    # (i = 1 .. n_ext-3, i.e. faces strictly inside the ghost-padded array
    # that correspond to the n+1 real faces):
    #   uL at face = u_ext[i]   + 0.5*slope[i-1]   (extrapolate right edge of left cell)
    #   uR at face = u_ext[i+1] - 0.5*slope[i]     (extrapolate left edge of right cell)
    uL = u_ext[1:-2] + 0.5 * slope[:-1]
    uR = u_ext[2:-1] - 0.5 * slope[1:]

    return uL, uR


# --------------------------------------------------------------------------
# Riemann solvers acting on reconstructed (uL, uR) face states
# --------------------------------------------------------------------------

def riemann_linear(uL, uR, a):
    """Upwind Riemann solver for linear advection given face states."""
    return a * uL if a >= 0 else a * uR


def riemann_burgers_godunov(uL, uR):
    """Exact Godunov flux for Burgers' equation given reconstructed face states."""
    f = np.zeros_like(uL)
    shock = uL > uR
    s = 0.5 * (uL + uR)
    f_shock = np.where(s > 0, flux_burgers(uL), flux_burgers(uR))
    f_rare = np.where(
        uL >= 0, flux_burgers(uL),
        np.where(uR <= 0, flux_burgers(uR), 0.0)
    )
    return np.where(shock, f_shock, f_rare)


# --------------------------------------------------------------------------
# Time stepping (SSP-RK2 / Heun's method — second order in time to match
# second-order space; first-order Euler in time would throw away the
# spatial accuracy gained from MUSCL)
# --------------------------------------------------------------------------

def rhs_linear(u, dx, a, bc_type, limiter_fn):
    u_ext = apply_bc(u, bc_type=bc_type, n_ghost=2)
    uL, uR = reconstruct_faces(u_ext, limiter_fn)
    F = riemann_linear(uL, uR, a)
    return -(F[1:] - F[:-1]) / dx


def rhs_burgers(u, dx, bc_type, limiter_fn):
    u_ext = apply_bc(u, bc_type=bc_type, n_ghost=2)
    uL, uR = reconstruct_faces(u_ext, limiter_fn)
    F = riemann_burgers_godunov(uL, uR)
    return -(F[1:] - F[:-1]) / dx


def ssp_rk2_step(u, dt, rhs_fn):
    """Strong-stability-preserving RK2 (Heun's method)."""
    k1 = rhs_fn(u)
    u1 = u + dt * k1
    k2 = rhs_fn(u1)
    return 0.5 * u + 0.5 * (u1 + dt * k2)


# --------------------------------------------------------------------------
# Solver drivers
# --------------------------------------------------------------------------

def solve_linear_advection_muscl(n=200, x_max=1.0, a=1.0, t_final=0.5, cfl=0.4,
                                  bc_type="periodic", ic_fn=ic_square_wave,
                                  limiter="minmod"):
    grid = Grid1D(n, 0.0, x_max)
    u = ic_fn(grid.x)
    limiter_fn = LIMITERS[limiter]
    t = 0.0

    while t < t_final:
        dt = cfl * grid.dx / abs(a)
        if t + dt > t_final:
            dt = t_final - t
        u = ssp_rk2_step(u, dt, lambda uu: rhs_linear(uu, grid.dx, a, bc_type, limiter_fn))
        t += dt

    return grid, u


def solve_burgers_muscl(n=200, x_max=1.0, t_final=0.3, cfl=0.4,
                         bc_type="outflow", ic_fn=ic_sine_hump,
                         limiter="minmod"):
    grid = Grid1D(n, 0.0, x_max)
    u = ic_fn(grid.x)
    limiter_fn = LIMITERS[limiter]
    t = 0.0

    while t < t_final:
        smax = np.max(np.abs(u)) if np.max(np.abs(u)) > 0 else 1e-6
        dt = cfl * grid.dx / smax
        if t + dt > t_final:
            dt = t_final - t
        u = ssp_rk2_step(u, dt, lambda uu: rhs_burgers(uu, grid.dx, bc_type, limiter_fn))
        t += dt

    return grid, u


# --------------------------------------------------------------------------
# Demo: compare first-order upwind vs MUSCL (minmod / van Leer / superbee)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from fv1d_advection import solve_linear_advection, solve_burgers

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Linear advection: square wave, compare smearing across schemes ---
    grid_ref = Grid1D(200, 0.0, 1.0)
    u_init = ic_square_wave(grid_ref.x)

    _, u_upwind = solve_linear_advection(n=200, a=1.0, t_final=0.5, cfl=0.5,
                                          bc_type="periodic", ic_fn=ic_square_wave)
    _, u_minmod = solve_linear_advection_muscl(n=200, a=1.0, t_final=0.5, cfl=0.4,
                                                bc_type="periodic", ic_fn=ic_square_wave,
                                                limiter="minmod")
    _, u_vanleer = solve_linear_advection_muscl(n=200, a=1.0, t_final=0.5, cfl=0.4,
                                                 bc_type="periodic", ic_fn=ic_square_wave,
                                                 limiter="van_leer")

    axes[0].plot(grid_ref.x, u_init, "k--", lw=1, label="initial")
    axes[0].plot(grid_ref.x, u_upwind, "b-", lw=1, label="1st-order upwind")
    axes[0].plot(grid_ref.x, u_minmod, "g-", lw=1.5, label="MUSCL (minmod)")
    axes[0].plot(grid_ref.x, u_vanleer, "r-", lw=1.5, label="MUSCL (van Leer)")
    axes[0].set_title("Square-wave advection: sharpness comparison\n"
                       "(square wave should stay square — less smearing = better)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # --- Burgers: shock formation, check no oscillations near the shock ---
    grid_b = Grid1D(200, 0.0, 1.0)
    u_b0 = ic_sine_hump(grid_b.x)
    _, u_b_upwind = solve_burgers(n=200, t_final=0.3, cfl=0.5,
                                   bc_type="outflow", ic_fn=ic_sine_hump)
    _, u_b_minmod = solve_burgers_muscl(n=200, t_final=0.3, cfl=0.4,
                                         bc_type="outflow", ic_fn=ic_sine_hump,
                                         limiter="minmod")

    axes[1].plot(grid_b.x, u_b0, "k--", lw=1, label="initial")
    axes[1].plot(grid_b.x, u_b_upwind, "b-", lw=1, label="1st-order upwind")
    axes[1].plot(grid_b.x, u_b_minmod, "g-", lw=1.5, label="MUSCL (minmod)")
    axes[1].set_title("Burgers' shock formation\n"
                       "(MUSCL should give a sharper shock, still no oscillations)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = "fv1d_muscl_demo.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    # --- Sanity checks ---
    assert np.isfinite(u_minmod).all() and np.isfinite(u_vanleer).all()
    assert np.isfinite(u_b_minmod).all()

    # TVD check: total variation should not increase over time for minmod
    def total_variation(u):
        return np.sum(np.abs(np.diff(u)))

    tv0 = total_variation(u_init)
    tv_minmod = total_variation(u_minmod)
    print(f"Total variation check (advection, minmod): initial={tv0:.4f}, "
          f"final={tv_minmod:.4f}  (final should be <= initial for TVD)")
    assert tv_minmod <= tv0 + 1e-10, "TVD violated for minmod limiter!"

    # Check MUSCL is sharper than upwind (less numerical diffusion) for the
    # square wave: max value should be closer to 1.0 (less smeared peak)
    print(f"Peak value after advection: upwind={u_upwind.max():.3f}, "
          f"minmod={u_minmod.max():.3f}, van_leer={u_vanleer.max():.3f} "
          f"(higher = sharper, closer to exact value of 1.0)")
    assert u_minmod.max() >= u_upwind.max() - 1e-6, \
        "MUSCL should not be more diffusive than 1st-order upwind"

    print("All sanity checks passed.")
