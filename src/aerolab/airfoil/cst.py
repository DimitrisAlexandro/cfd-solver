"""
CST (Kulfan class-shape transformation) airfoil parameterisation, used by
the design tools (inverse design, optimizer).

    y(x) = C(x) S(x) + x dz_te,  C = sqrt(x) (1 - x),  S = sum_i A_i B_i,n(x)
upper and lower surfaces with their own Bernstein coefficients A (lower
coefficients negative for a normal airfoil). Kulfan, J. Aircraft 45 (2008).
"""

import math
from pathlib import Path

import numpy as np


def bernstein(x, n):
    k = np.arange(n + 1)
    comb = np.array([math.comb(n, i) for i in k], float)
    return comb[None] * x[:, None] ** k[None] * (1 - x[:, None]) ** (n - k[None])


def surface(x, A, dz_te=0.0):
    A = np.asarray(A, float)
    return np.sqrt(x) * (1 - x) * (bernstein(x, len(A) - 1) @ A) + x * dz_te


def x_cosine(n=120):
    return 0.5 * (1 - np.cos(np.linspace(0, math.pi, n)))


def outline(au, al, n=120, te_gap=0.0):
    """sharp (or gap) open outline TE -> upper -> LE -> lower -> TE,
    counter-clockwise (panel.solve / airfoil_geom input)"""
    x = x_cosine(n)
    yu = surface(x, au, te_gap / 2)
    yl = surface(x, al, -te_gap / 2)
    return np.r_[np.c_[x[::-1], yu[::-1]], np.c_[x[1:], yl[1:]]]


def fit(pts, n=8):
    """least-squares CST fit of an outline (TE -> around -> TE): (au, al, err)"""
    pts = np.asarray(pts, float)
    i = int(np.argmin(pts[:, 0]))
    up, lo = pts[:i + 1][::-1], pts[i:]                   # LE -> TE each
    if up[:, 1].mean() < lo[:, 1].mean():
        up, lo = lo, up
    out, err = [], 0.0
    for s in (up, lo):
        x, y = np.clip(s[:, 0], 0, 1), s[:, 1]
        k = (x > 1e-4)
        dz = y[np.argmax(x)]
        M = np.sqrt(x[k])[:, None] * (1 - x[k])[:, None] * bernstein(x[k], n)
        A = np.linalg.lstsq(M, y[k] - x[k] * dz, rcond=None)[0]
        out.append(A)
        err = max(err, np.abs(M @ A + x[k] * dz - y[k]).max())
    return out[0], out[1], err


def thickness(au, al, n=200):
    x = x_cosine(n)
    t = surface(x, au) - surface(x, al)
    i = int(np.argmax(t))
    return float(t[i]), float(x[i]), float(t.min())


def write_dat(path, au, al, name="cst airfoil", n=160):
    o = outline(au, al, n)
    Path(path).write_text(name + "\n" + "\n".join(f"{x:.7f} {y:.7f}" for x, y in o) + "\n")
    return path
