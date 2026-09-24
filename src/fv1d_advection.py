"""
1D Finite Volume solver — linear advection and inviscid Burgers' equation.

Warm-up step before building a full mesher + Navier-Stokes solver:
validates flux schemes, boundary handling, and explicit time integration
on the simplest possible PDE.

Equations supported:
    Linear advection:  du/dt + a * du/dx = 0
    Inviscid Burgers:   du/dt + u * du/dx = 0   (flux f(u) = u^2 / 2)

Author: Dimitris
"""

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------

class Grid1D:
    """Uniform 1D finite-volume grid with `n` cells on [x_min, x_max]."""

    def __init__(self, n, x_min=0.0, x_max=1.0):
        self.n = n
        self.x_min = x_min
        self.x_max = x_max
        self.dx = (x_max - x_min) / n
        # cell centers
        self.x = x_min + (np.arange(n) + 0.5) * self.dx


# --------------------------------------------------------------------------
# Initial conditions
# --------------------------------------------------------------------------

def ic_square_wave(x, lo=0.2, hi=0.4, low_val=0.0, high_val=1.0):
    u = np.full_like(x, low_val, dtype=float)
    u[(x >= lo) & (x <= hi)] = high_val
    return u


def ic_gaussian(x, x0=0.3, sigma=0.05, amplitude=1.0):
    return amplitude * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def ic_sine_hump(x, lo=0.1, hi=0.6):
    u = np.zeros_like(x)
    mask = (x >= lo) & (x <= hi)
    u[mask] = np.sin(np.pi * (x[mask] - lo) / (hi - lo)) ** 2
    return u


# --------------------------------------------------------------------------
# Flux functions
# --------------------------------------------------------------------------

def flux_linear(u, a):
    """f(u) = a*u for linear advection."""
    return a * u


def flux_burgers(u):
    """f(u) = u^2/2 for inviscid Burgers' equation."""
    return 0.5 * u ** 2


# --------------------------------------------------------------------------
# Boundary conditions (apply via ghost cells)
# --------------------------------------------------------------------------

def apply_bc(u, bc_type="periodic", n_ghost=1):
    """
    Return array padded with n_ghost ghost cells on each side.
    bc_type: "periodic" or "outflow" (zero-gradient)
    """
    if bc_type == "periodic":
        return np.concatenate([u[-n_ghost:], u, u[:n_ghost]])
    elif bc_type == "outflow":
        left = np.repeat(u[0], n_ghost)
        right = np.repeat(u[-1], n_ghost)
        return np.concatenate([left, u, right])
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")


# --------------------------------------------------------------------------
# Riemann / interface flux schemes
# --------------------------------------------------------------------------

def upwind_flux_linear(u_ext, a):
    """
    First-order upwind flux at each of the n+1 interfaces for linear
    advection with wave speed a (constant sign assumed).
    u_ext has 1 ghost cell on each side: length n+2.
    Returns array of length n+1 (face fluxes F_{i-1/2} for i=0..n).
    """
    if a >= 0:
        # flux comes from the left cell
        return a * u_ext[:-1]
    else:
        # flux comes from the right cell
        return a * u_ext[1:]


def upwind_flux_burgers(u_ext):
    """
    Godunov flux for inviscid Burgers' equation at each interface.
    u_ext has 1 ghost cell on each side: length n+2.
    Returns array of length n+1.
    """
    uL = u_ext[:-1]
    uR = u_ext[1:]
    f = np.zeros_like(uL)

    # Godunov's exact Riemann solver for f(u) = u^2/2
    shock = uL > uR
    rarefaction = ~shock

    # Shock case: speed = (uL+uR)/2
    s = 0.5 * (uL + uR)
    f_shock = np.where(s > 0, flux_burgers(uL), flux_burgers(uR))

    # Rarefaction case
    f_rare = np.where(
        uL >= 0, flux_burgers(uL),
        np.where(uR <= 0, flux_burgers(uR), 0.0)  # sonic point: f=0
    )

    f = np.where(shock, f_shock, f_rare)
    return f


# --------------------------------------------------------------------------
# Time stepping
# --------------------------------------------------------------------------

def compute_dt(u, dx, cfl, wave_speed_fn):
    """wave_speed_fn(u) -> max |characteristic speed| over the domain."""
    smax = wave_speed_fn(u)
    if smax < 1e-14:
        return 1e-6
    return cfl * dx / smax


def step_explicit_euler(u, dx, dt, flux_at_faces):
    """
    Explicit (forward) Euler update using precomputed face fluxes.
    flux_at_faces: array of length n+1, F[i] = flux at interface i-1/2
                   for i = 0..n (so F[0] = left boundary flux of cell 0,
                   F[n] = right boundary flux of cell n-1).
    """
    return u - (dt / dx) * (flux_at_faces[1:] - flux_at_faces[:-1])


# --------------------------------------------------------------------------
# Solver drivers
# --------------------------------------------------------------------------

def solve_linear_advection(n=200, x_max=1.0, a=1.0, t_final=0.5, cfl=0.5,
                            bc_type="periodic", ic_fn=ic_square_wave):
    grid = Grid1D(n, 0.0, x_max)
    u = ic_fn(grid.x)
    t = 0.0

    while t < t_final:
        dt = compute_dt(u, grid.dx, cfl, wave_speed_fn=lambda _u: abs(a))
        if t + dt > t_final:
            dt = t_final - t

        u_ext = apply_bc(u, bc_type=bc_type, n_ghost=1)
        F = upwind_flux_linear(u_ext, a)
        u = step_explicit_euler(u, grid.dx, dt, F)

        t += dt

    return grid, u


def solve_burgers(n=200, x_max=1.0, t_final=0.3, cfl=0.5,
                   bc_type="outflow", ic_fn=ic_sine_hump):
    grid = Grid1D(n, 0.0, x_max)
    u = ic_fn(grid.x)
    t = 0.0

    while t < t_final:
        smax = np.max(np.abs(u)) if np.max(np.abs(u)) > 0 else 1e-6
        dt = cfl * grid.dx / smax
        if t + dt > t_final:
            dt = t_final - t

        u_ext = apply_bc(u, bc_type=bc_type, n_ghost=1)
        F = upwind_flux_burgers(u_ext)
        u = step_explicit_euler(u, grid.dx, dt, F)

        t += dt

    return grid, u


# --------------------------------------------------------------------------
# Demo / validation script
# --------------------------------------------------------------------------

if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Linear advection: square wave should translate rigidly ---
    grid0, u0 = Grid1D(200, 0.0, 1.0), ic_square_wave(Grid1D(200, 0.0, 1.0).x)
    grid1, u1 = solve_linear_advection(
        n=200, x_max=1.0, a=1.0, t_final=0.5, cfl=0.5,
        bc_type="periodic", ic_fn=ic_square_wave
    )
    axes[0].plot(grid0.x, u0, "k--", label="initial (t=0)")
    axes[0].plot(grid1.x, u1, "b-", label="upwind (t=0.5)")
    axes[0].set_title("Linear advection: du/dt + a du/dx = 0\n(a=1, upwind, periodic BC)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Inviscid Burgers: smooth hump should steepen into a shock ---
    grid2, u2 = Grid1D(200, 0.0, 1.0), ic_sine_hump(Grid1D(200, 0.0, 1.0).x)
    grid3, u3 = solve_burgers(
        n=200, x_max=1.0, t_final=0.3, cfl=0.5,
        bc_type="outflow", ic_fn=ic_sine_hump
    )
    axes[1].plot(grid2.x, u2, "k--", label="initial (t=0)")
    axes[1].plot(grid3.x, u3, "r-", label="Godunov (t=0.3)")
    axes[1].set_title("Inviscid Burgers: du/dt + u du/dx = 0\n(Godunov flux, outflow BC)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = "fv1d_demo.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    # Basic sanity checks
    assert np.isfinite(u1).all(), "advection solution has NaN/Inf"
    assert np.isfinite(u3).all(), "burgers solution has NaN/Inf"
    # mass conservation check for periodic advection (should be exact up to fp error)
    mass0 = np.sum(u0) * grid0.dx
    mass1 = np.sum(u1) * grid1.dx
    print(f"Mass check (advection): initial={mass0:.6f}, final={mass1:.6f}, "
          f"diff={abs(mass0 - mass1):.2e}")
    print("All sanity checks passed.")