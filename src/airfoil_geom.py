"""
Airfoil geometry input (Phase 3, step E2).

Any airfoil -> a clean, closed, counter-clockwise loop for the mesher:

    load_airfoil("naca2412")            NACA 4-digit
    load_airfoil("naca23012")           NACA 5-digit (standard, non-reflexed)
    load_airfoil("tmr0012")             NASA TMR NACA 0012 (sharp trailing edge)
    load_airfoil("path/to/foil.dat")    coordinate file, Selig or Lednicer format

Pipeline (same for every source):
  1. raw surface points, ordered upper TE -> LE -> lower TE
  2. normalise: leading edge at (0, 0), trailing-edge midpoint at (1, 0)
     (translate, rotate - the chord line becomes the x axis - and scale)
  3. cubic spline through the points versus arc length
  4. repanel: n_side points per side with leading- and trailing-edge
     clustering (same distribution as the validated NACA meshes)
  5. trailing edge: a sharp (or thinner than te_thick) trailing edge is cut
     where the thickness equals te_thick and x is rescaled to chord 1: a small
     blunt base extrudes cleanly (a sharp TE is a ~360 deg corner). A thicker,
     truly blunt TE is kept.
  6. checks: no self-intersection, upper above lower, sensible thickness.

Formats
  Selig:     name line, then x y from the upper TE over the LE to the lower TE
  Lednicer:  name line, "n_upper n_lower" line, upper LE -> TE, lower LE -> TE
"""

import math
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

from geometry import signed_area

TE_THICK = 0.001          # minimum trailing-edge base thickness (chord units)


# ----------------------------------------------------------------------
# sources
# ----------------------------------------------------------------------
def _cosine(n):
    b = np.linspace(0.0, math.pi, n)
    return 0.5 * (1 - np.cos(b))


def naca_raw(code, n=400, sharp=False):
    """Raw NACA 4- or 5-digit surface, upper TE -> LE -> lower TE.
    sharp=True uses the closed-TE thickness coefficient (-0.1036)."""
    code = code.strip()
    x = _cosine(n)
    if len(code) == 4:
        m, p, t = int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:]) / 100.0
        if m > 0 and p > 0:
            yc = np.where(x < p, m / p**2 * (2 * p * x - x**2),
                          m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
            dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p)**2 * (p - x))
        else:
            yc, dyc = np.zeros_like(x), np.zeros_like(x)
    elif len(code) == 5:
        L, P, Q = int(code[0]), int(code[1]), int(code[2])
        t = int(code[3:]) / 100.0
        if Q != 0:
            raise ValueError("reflexed NACA 5-digit airfoils (third digit 1) are not supported")
        table = {1: (0.0580, 361.400), 2: (0.1260, 51.640), 3: (0.2025, 15.957),
                 4: (0.2900, 6.643), 5: (0.3910, 3.230)}
        if P not in table:
            raise ValueError(f"NACA 5-digit: second digit must be 1..5, got {P}")
        m, k1 = table[P]
        k1 = k1 * (0.15 * L) / 0.3                  # table values are for design Cl 0.3
        yc = np.where(x < m, k1 / 6 * (x**3 - 3 * m * x**2 + m**2 * (3 - m) * x),
                      k1 * m**3 / 6 * (1 - x))
        dyc = np.where(x < m, k1 / 6 * (3 * x**2 - 6 * m * x + m**2 * (3 - m)),
                       -k1 * m**3 / 6)
    else:
        raise ValueError(f"NACA code must have 4 or 5 digits, got '{code}'")
    a4 = -0.1036 if sharp else -0.1015
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + a4 * x**4)
    th = np.arctan(dyc)
    up = np.c_[x - yt * np.sin(th), yc + yt * np.cos(th)]
    lo = np.c_[x + yt * np.sin(th), yc - yt * np.cos(th)]
    return np.vstack([up[::-1], lo[1:]])


def tmr0012_raw(n=400):
    x = _cosine(n)
    y = 0.594689181 * (0.298222773 * np.sqrt(x) - 0.127125232 * x - 0.357907906 * x**2
                       + 0.291984971 * x**3 - 0.105174606 * x**4)
    y[-1] = 0.0
    return np.vstack([np.c_[x, y][::-1], np.c_[x, -y][1:]])


def read_dat(path):
    """Coordinate file (Selig or Lednicer) -> raw points upper TE -> LE -> lower TE."""
    rows, name = [], None
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = line.replace(",", " ").split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            vals = None
        if vals is None or len(vals) < 2:
            if name is None and line.strip():
                name = line.strip()
            continue
        rows.append(vals[:2])
    if len(rows) < 10:
        raise ValueError(f"{path}: fewer than 10 coordinate lines")
    a = np.array(rows, float)
    if a[0, 0] > 1.5 and a[0, 1] > 1.5 and float(a[0, 0]).is_integer():
        nu, nl = int(a[0, 0]), int(a[0, 1])              # Lednicer: counts line
        up, lo = a[1:1 + nu], a[1 + nu:1 + nu + nl]
        if len(up) != nu or len(lo) != nl:
            raise ValueError(f"{path}: Lednicer header says {nu}+{nl} points, file has fewer")
        if up[0, 0] > up[-1, 0]:
            up = up[::-1]
        if lo[0, 0] > lo[-1, 0]:
            lo = lo[::-1]
        pts = np.vstack([up[::-1], lo[1:] if np.allclose(lo[0], up[0]) else lo])
    else:
        pts = a                                           # Selig
    # drop exact duplicates (repeated LE / TE points)
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-12]
    pts = pts[keep]
    return pts, (name or Path(path).stem)


def write_dat(path, loop, name="airfoil", fmt="selig"):
    """Write a loop (upper TE -> LE -> lower TE order) as Selig or Lednicer."""
    loop = np.asarray(loop)
    i_le = int(np.argmin(loop[:, 0]))
    with open(path, "w") as f:
        f.write(name + "\n")
        if fmt == "selig":
            for x, y in loop:
                f.write(f"{x:.8f} {y:.8f}\n")
        else:
            up, lo = loop[:i_le + 1][::-1], loop[i_le:]
            f.write(f"{len(up)}. {len(lo)}.\n\n")
            for x, y in up:
                f.write(f"{x:.8f} {y:.8f}\n")
            f.write("\n")
            for x, y in lo:
                f.write(f"{x:.8f} {y:.8f}\n")


# ----------------------------------------------------------------------
# processing
# ----------------------------------------------------------------------
def _orient(pts):
    """Order upper TE -> LE -> lower TE (= counter-clockwise)."""
    if signed_area(pts) < 0:                              # clockwise -> reverse
        pts = pts[::-1]
    return pts


def normalise(pts, derotate=False):
    """Leading edge at (0, 0), chord 1. The LE is the surface point with the
    smallest x (on the spline) and the TE the midpoint of the end points.
    By default only translate + scale: the angle of attack stays measured from
    the file's x axis (as in XFOIL and published data). derotate=True also
    rotates the LE-TE chord line onto the x axis (for twisted/rotated files)."""
    pts = _orient(np.asarray(pts, float))
    te = 0.5 * (pts[0] + pts[-1])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    sx, sy = CubicSpline(s, pts[:, 0]), CubicSpline(s, pts[:, 1])
    if derotate:                        # XFOIL definition: farthest point from the TE
        i0 = int(np.argmax(np.linalg.norm(pts - te, axis=1)))
        f = lambda q: -math.hypot(sx(q) - te[0], sy(q) - te[1])
    else:
        i0 = int(np.argmin(pts[:, 0]))
        f = sx
    r = minimize_scalar(f, bounds=(s[max(i0 - 2, 0)], s[min(i0 + 2, len(s) - 1)]),
                        method="bounded", options={"xatol": 1e-12})
    le = np.array([sx(r.x), sy(r.x)])
    c = te - le
    ang = math.atan2(c[1], c[0]) if derotate else 0.0
    chord = float(np.hypot(*c)) if derotate else float(c[0])
    R = np.array([[math.cos(-ang), -math.sin(-ang)], [math.sin(-ang), math.cos(-ang)]])
    out = (pts - le) @ R.T / chord
    return out, {"chord_input": chord, "chord_angle_deg": math.degrees(math.atan2(c[1], c[0])),
                 "derotated": bool(derotate)}


def _sides(pts):
    """Split a normalised loop at the LE into upper / lower splines x -> y."""
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    sx, sy = CubicSpline(s, pts[:, 0]), CubicSpline(s, pts[:, 1])
    i0 = int(np.argmin(pts[:, 0]))
    r = minimize_scalar(sx, bounds=(s[max(i0 - 2, 0)], s[min(i0 + 2, len(s) - 1)]),
                        method="bounded", options={"xatol": 1e-13})
    s_le = r.x
    dense_u = np.linspace(s_le, 0.0, 6000)
    dense_l = np.linspace(s_le, s[-1], 6000)
    return (sx, sy), (s_le, s[-1]), dense_u, dense_l


def repanel(pts, n_side=160, te_blend=0.85, n_te=3, te_thick=TE_THICK, rescale=True):
    """Normalised raw loop -> mesher loop (CCW), n_side points per side.
    rescale: stretch back to x_TE = 1 after cutting a sharp trailing edge
    (off for flapped sections: their chord reference is the unflapped one)."""
    (sx, sy), (s_le, s_end), du, dl = _sides(pts)
    xu, yu = sx(du), sy(du)
    xl, yl = sx(dl), sy(dl)
    for xs, nm in ((xu, "upper"), (xl, "lower")):
        if np.any(np.diff(xs) < -1e-9):
            raise ValueError(f"{nm} surface is not monotonic in x (re-entrant shape?) - "
                             "check the coordinate file")
    yup = lambda x: np.interp(x, xu, yu)
    ylo = lambda x: np.interp(x, xl, yl)
    x_end = min(xu[-1], xl[-1])
    t_end = yup(x_end) - ylo(x_end)
    if t_end < te_thick:                        # sharp or too thin: cut at te_thick
        x_cut = brentq(lambda x: yup(x) - ylo(x) - te_thick, 0.8 * x_end, x_end)
    else:
        x_cut = x_end
    b = np.linspace(0.0, np.pi, n_side)
    xd = (te_blend * 0.5 * (1 - np.cos(b)) + (1 - te_blend) * (1 - np.cos(b / 2))) * x_cut
    # exact spline points: parameter where x(s) = xd on each side
    su = np.interp(xd, xu, du)
    sl = np.interp(xd, xl, dl)
    up = np.c_[sx(su), sy(su)]
    lo = np.c_[sx(sl), sy(sl)]
    lo[0] = up[0]                                          # common LE point
    scale = 1.0 / up[-1, 0] if (x_cut < x_end - 1e-12 and rescale) else 1.0
    up, lo = up * scale, lo * scale
    t = np.linspace(0, 1, n_te + 1)[1:-1]
    base = lo[-1] + t[:, None] * (up[-1] - lo[-1])
    loop = np.vstack([up[::-1], lo[1:], base])
    if signed_area(loop) < 0:
        loop = loop[::-1]
    return loop


def check(loop):
    """Geometry sanity checks -> dict (raises on invalid geometry)."""
    n = len(loop)
    a = loop
    b = np.roll(loop, -1, axis=0)
    # segment-segment intersection (non-adjacent), vectorised per segment
    def cross(o, p, q):
        return (p[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1]) - (p[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0])
    for i in range(n):
        j = np.arange(n)
        mask = (j != i) & (j != (i + 1) % n) & ((j + 1) % n != i)
        c, d = a[j[mask]], b[j[mask]]
        d1, d2 = cross(a[i], b[i], c), cross(a[i], b[i], d)
        d3, d4 = cross(c, d, a[i][None]), cross(c, d, b[i][None])
        if np.any((d1 * d2 < 0) & (d3 * d4 < 0)):
            raise ValueError("airfoil outline intersects itself")
    i_le = int(np.argmin(loop[:, 0]))
    x = np.linspace(0.01, 0.99, 197) * min(1.0, float(loop[:, 0].max()))   # flapped: TE ahead of x = 1
    # CCW loop: from the TE along the lower... find both sides by y at each x
    th = []
    for xx in x:
        ys = []
        for p, q in zip(a, b):
            if (p[0] - xx) * (q[0] - xx) <= 0 and p[0] != q[0]:
                ys.append(p[1] + (xx - p[0]) / (q[0] - p[0]) * (q[1] - p[1]))
        th.append(max(ys) - min(ys) if len(ys) >= 2 else 0.0)
    th = np.array(th)
    if th.min() <= 0:
        raise ValueError("zero or negative thickness inside the chord")
    return {"points": n, "max_thickness": float(th.max()),
            "x_max_thickness": float(x[np.argmax(th)]),
            "te_thickness": float(np.ptp(loop[np.isclose(loop[:, 0], loop[:, 0].max())][:, 1])),
            "le": loop[i_le].tolist()}


FLAP_TAG = "@flap="


def parse_flap(spec):
    """'naca2412@flap=25c20' -> ('naca2412', (0.25, 20.0)): flap chord in % of
    the chord, deflection in deg (+ = trailing edge down, 'p' = decimal
    point, so the tag has no dots and file names stay unambiguous); no tag
    -> (spec, None)"""
    s = str(spec)
    if FLAP_TAG not in s:
        return s, None
    base, f = s.split(FLAP_TAG, 1)
    c, d = f.split("c", 1)
    cf, delta = float(c.replace("p", ".")) / 100.0, float(d.replace("p", "."))
    if not (0.05 <= cf <= 0.6):
        raise ValueError(f"flap chord fraction {cf} outside 0.05 .. 0.6")
    return base, (cf, delta)


def flap_spec(spec, cf, delta):
    """section spec with a plain flap of cf (fraction of chord) at delta deg"""
    if abs(delta) < 1e-9:
        return str(spec)
    return f"{spec}{FLAP_TAG}" + f"{round(cf * 100):d}c{delta:g}".replace(".", "p")


def section_stem(spec):
    """file-name stem of a section spec (flap tag kept; Path.stem would cut a
    .dat file's tag off)"""
    base, flap = parse_flap(spec)
    stem = Path(str(base)).stem
    if flap:
        stem += FLAP_TAG + f"{round(flap[0] * 100):d}c{flap[1]:g}".replace(".", "p")
    return stem


def deflect_flap(pts, cf, delta_deg, blend=0.04, n=240):
    """Plain (sealed) trailing-edge flap on a normalised outline: the camber
    line aft of the hinge (x = 1 - cf) is rotated by delta (+ = trailing
    edge down) with the angle blended smoothly over `blend` chord around
    the hinge (a rounded, sealed hinge: no gap, no kink for the mesher);
    the thickness distribution is carried along normal to the rotated
    reference direction. Chord reference unchanged (1). Returns the outline
    upper TE -> LE -> lower TE (counter-clockwise)."""
    p = np.asarray(pts, float)
    (sx, sy), (s_le, s_end), du, dl = _sides(p)
    xu, yu, xl, yl = sx(du), sy(du), sx(dl), sy(dl)
    b = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(b)) * min(xu[-1], xl[-1])
    yU, yL = np.interp(x, xu, yu), np.interp(x, xl, yl)
    yc, th = 0.5 * (yU + yL), 0.5 * (yU - yL)
    xh = 1.0 - cf
    # above 20 deg the blend widens with the deflection (same turn radius as
    # at 20 deg): a tighter concave lower-surface corner squeezed the
    # boundary-layer cells (30 deg: 58 deg non-orthogonality) and the RANS
    # solution diverged at every angle
    blend = blend * max(1.0, abs(delta_deg) / 20.0)
    u = np.clip((x - (xh - 0.5 * blend)) / blend, 0, 1)
    phi = -math.radians(delta_deg) * u * u * (3 - 2 * u)            # smoothstep, clockwise
    C = np.zeros((n, 2))
    C[0] = (x[0], yc[0])
    for i in range(n - 1):
        f = 0.5 * (phi[i] + phi[i + 1])
        d = np.array([x[i + 1] - x[i], yc[i + 1] - yc[i]])
        R = np.array([[math.cos(f), -math.sin(f)], [math.sin(f), math.cos(f)]])
        C[i + 1] = C[i] + R @ d
    nrm = np.c_[-np.sin(phi), np.cos(phi)]                            # rotated "up"
    up = C + th[:, None] * nrm
    lo = C - th[:, None] * nrm
    return np.vstack([up[::-1], lo[1:]])


def load_airfoil(spec, n_side=160, te_thick=TE_THICK, n_te=3, derotate=False):
    """spec: 'naca2412', 'naca23012', '0012', 'tmr0012' or a .dat path.
    Returns (loop, info). The loop goes to hybrid_mesher.build_mesh(wall=loop)."""
    s, flap = parse_flap(str(spec).strip())
    low = s.lower()
    if low in ("tmr", "tmr0012", "naca0012tmr"):
        raw, name = tmr0012_raw(), "NACA 0012 (NASA TMR, sharp TE)"
    elif low.startswith("naca") or (low.isdigit() and len(low) in (4, 5)):
        code = low.replace("naca", "").replace(" ", "")
        raw, name = naca_raw(code), f"NACA {code}"
    else:
        p = Path(s)
        if not p.exists():
            raise FileNotFoundError(f"airfoil '{s}': not a NACA code and no such file")
        raw, name = read_dat(p)
    norm, info = normalise(raw, derotate=derotate)
    if flap:
        norm = deflect_flap(norm, *flap)
        name += f", flap {flap[0]:g} c at {flap[1]:g}°"
    loop = repanel(norm, n_side=n_side, te_thick=te_thick, n_te=n_te, rescale=flap is None)
    info.update(check(loop), name=name, source=str(spec), flap=flap)
    return loop, info


def sharp_outline(spec):
    """normalised raw outline TE -> around -> TE, counter-clockwise (no
    repanelling, sharp trailing edge as given): input of the panel method
    and of the CST fit (design tools)"""
    s, flap = parse_flap(str(spec).strip())
    low = s.lower()
    if low in ("tmr", "tmr0012", "naca0012tmr"):
        raw = tmr0012_raw()
    elif low.startswith("naca") or (low.isdigit() and len(low) in (4, 5)):
        raw = naca_raw(low.replace("naca", "").replace(" ", ""))
    else:
        raw = read_dat(Path(s))[0]
    n, _ = normalise(raw)
    if flap:
        n = deflect_flap(n, *flap)
    x, y = n[:, 0], n[:, 1]
    if 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        n = n[::-1]
    return n
