"""
Airfoil geometry and wall-spacing helpers.
"""

import numpy as np


def naca4(code="0012", n_side=120, te_blend=0.85, n_te=2):
    """
    Closed, counter-clockwise NACA 4-digit loop with a BLUNT trailing edge
    (the standard -0.1015 coefficient, TE thickness ~0.25 % chord for 0012).

    A blunt TE is used on purpose: a sharp TE is a ~360 deg convex corner
    where boundary-layer normals are undefined; a blunt TE turns it into
    two ~90 deg corners that extrude cleanly.

    Point distribution: blend of full-cosine (clusters at LE and TE) and
    half-cosine (clusters at LE only). te_blend=1 -> strong TE clustering.

    Loop order: upper TE -> LE (along upper surface) -> lower TE -> TE base.
    Returns (M, 2) array without repeating the first point.
    """
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0

    beta = np.linspace(0.0, np.pi, n_side)
    x = te_blend * 0.5 * (1 - np.cos(beta)) + (1 - te_blend) * (1 - np.cos(beta / 2))

    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1015 * x**4)
    if m > 0 and p > 0:
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2),
                      m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p)**2 * (p - x))
    else:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)

    upper = np.c_[xu, yu][::-1]            # TE -> LE
    lower = np.c_[xl, yl][1:]              # LE -> TE (skip duplicated LE)
    s = np.linspace(0, 1, n_te + 1)[1:-1]  # interior points of the TE base
    base = np.c_[xl[-1] + s * (xu[-1] - xl[-1]), yl[-1] + s * (yu[-1] - yl[-1])]
    loop = np.vstack([upper, lower, base])
    if signed_area(loop) < 0:
        loop = loop[::-1]
    return loop


def signed_area(loop):
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def first_cell_height(reynolds, yplus=1.0, chord=1.0):
    """First layer height (chord units) for a target y+, from the turbulent
    flat-plate estimate Cf = 0.026 / Re^(1/7). An estimate only: check the
    actual y+ from the solution and adjust."""
    cf = 0.026 / reynolds ** (1.0 / 7.0)
    u_tau = np.sqrt(0.5 * cf)          # in units of U_inf
    nu = chord / reynolds              # U_inf = 1
    return yplus * nu / u_tau


def circle(center, radius, n):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)]
