"""
Multi-element airfoil geometry (Phase 3, step E5).

A multi-element airfoil (slat + main + flap, airfoil + flap, ...) -> one
clean counter-clockwise loop per element for the mesher, plus the sharp
(open) outline for the panel method.

Specs (load_multi):
    "30p30n"                         built-in McDonnell Douglas 30P30N
                                     (slat 30 deg, flap 30 deg, stowed chord 1)
    "case.json"                      elements from a JSON file (format below)
    "naca23012+naca23012:0.3,0.98,-0.04,25"
                                     elements joined by '+'; each one is
                                     airfoil[:chord,x_le,y_le,deflection]

JSON format
    {"name": "NACA 23012 + 30 % flap", "ref_chord": 1.0,
     "moment_ref": [0.25, 0.0],
     "elements": [
        {"name": "main", "airfoil": "naca23012"},
        {"name": "flap", "airfoil": "flap.dat", "chord": 0.3,
         "x": 0.98, "y": -0.04, "deflection": 25},
        {"name": "slat", "file": "slat.dat", "frame": "global"}]}
  "airfoil" elements are normalised (LE at 0, chord 1), scaled by `chord`,
  rotated by `deflection` (deg, positive = trailing edge down) about their
  leading edge, and the leading edge moved to (x, y).
  "file" elements with "frame": "global" are used as given (coordinates
  already in the assembly frame, e.g. the 30P30N files). Relative paths are
  relative to the JSON file.

Per element:
  1. raw outline from the trailing edge around to the trailing edge
     (Selig order; an open trailing edge = finite base)
  2. corners (turn > 30 deg) are kept as nodes; between them a spline in arc
     length is resampled with a spacing set by curvature, distance to the
     corners and the trailing edge, and a maximum (all scaled by `level`)
  3. sharp trailing edge (thinner than te_thick = 0.1 % of the element
     chord) -> thickened to a small blunt base at the same position for the
     mesher (keeps the slot gaps: 30P30N slat 2.94 %, flap 1.23 % vs the
     nominal 2.95 / 1.27 %); the panel method keeps the sharp outline
  4. checks: no self-intersection, no element overlaps another; the gaps
     between elements are reported
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

import airfoil_geom as ag
from geometry import signed_area

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "validation" / "reference"

BUILTIN = {
    "30p30n": {
        "name": "30P30N (slat 30 deg, flap 30 deg)", "ref_chord": 1.0,
        "moment_ref": [0.25, 0.0],
        "elements": [{"name": n, "file": str(REF / "30p30n" / f"{n}.dat"), "frame": "global"}
                     for n in ("slat", "main", "flap")]},
}

CORNER_DEG = 30.0          # a vertex turning more than this is a corner


# ----------------------------------------------------------------------
# spec -> raw outlines
# ----------------------------------------------------------------------
def is_multi(spec):
    s = str(spec).strip().lower()
    return s in BUILTIN or s.endswith(".json") or "+" in s


def parse_spec(spec):
    """spec -> config dict (see module docstring)."""
    s = str(spec).strip()
    if s.lower() in BUILTIN:
        return json.loads(json.dumps(BUILTIN[s.lower()]))
    if s.lower().endswith(".json"):
        p = Path(s)
        cfg = json.loads(p.read_text())
        for e in cfg["elements"]:
            for key in ("file", "airfoil"):
                v = e.get(key)
                if v and not v.lower().startswith("naca") and not Path(v).is_absolute() \
                        and (p.parent / v).exists():
                    e[key] = str(p.parent / v)
        cfg.setdefault("name", p.stem)
        return cfg
    elements = []
    for k, part in enumerate(s.split("+")):
        name_part, _, args = part.partition(":")
        e = {"name": ("main" if k == 0 else f"element{k + 1}"), "airfoil": name_part.strip()}
        if args:
            v = [float(a) for a in args.split(",")]
            for key, val in zip(("chord", "x", "y", "deflection"), v):
                e[key] = val
        elements.append(e)
    if len(elements) == 2:
        elements[1]["name"] = "flap"
    return {"name": s, "elements": elements}


def raw_outline(e):
    """Element config -> raw points (N,2), trailing edge -> around -> trailing
    edge, counter-clockwise, in the assembly frame."""
    if e.get("frame") == "global" or ("file" in e and "airfoil" not in e):
        pts, _ = ag.read_dat(e["file"])
        pts = np.asarray(pts, float)
        if signed_area(pts) < 0:
            pts = pts[::-1]
        return pts
    spec = e["airfoil"]
    low = spec.lower()
    if low.startswith("naca") or (low.isdigit() and len(low) in (4, 5)):
        raw = ag.naca_raw(low.replace("naca", ""), sharp=True)
    elif low in ("tmr", "tmr0012"):
        raw = ag.tmr0012_raw()
    else:
        raw, _ = ag.read_dat(spec)
    pts, _ = ag.normalise(raw)
    c = float(e.get("chord", 1.0))
    d = math.radians(float(e.get("deflection", 0.0)))
    R = np.array([[math.cos(-d), -math.sin(-d)], [math.sin(-d), math.cos(-d)]])
    pts = (pts * c) @ R.T + np.array([e.get("x", 0.0), e.get("y", 0.0)])
    return pts


# ----------------------------------------------------------------------
# resampling
# ----------------------------------------------------------------------
def _turn(P, closed=False):
    """Turning angle (rad, + = left/convex for a CCW loop) at interior vertices."""
    a = P[1:-1] - P[:-2]
    b = P[2:] - P[1:-1]
    return np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], np.einsum('ij,ij->i', a, b))


def _spacing_field(xy, kappa, s, s_fixed, h_max, h_min, beta, grade):
    """Target spacing on a dense sample: curvature, corners, grading."""
    h = np.full(len(s), h_max)
    h = np.minimum(h, beta / np.maximum(np.abs(kappa), 1e-12))
    for sf in s_fixed:
        h = np.minimum(h, h_min + grade * np.abs(s - sf))
    # grading: neighbours differ by at most `grade` x distance
    for i in range(1, len(s)):
        h[i] = min(h[i], h[i - 1] + grade * (s[i] - s[i - 1]))
    for i in range(len(s) - 2, -1, -1):
        h[i] = min(h[i], h[i + 1] + grade * (s[i + 1] - s[i]))
    return np.maximum(h, 0.5 * h_min)


def resample(raw, chord_ref=1.0, level=1.0, h_max=0.010, h_min_rel=0.0015, beta=0.045,
             grade=0.12, corner_deg=CORNER_DEG):
    """Open outline (TE -> TE) -> resampled open outline with the same end
    points and corners. Spacings in units of chord_ref (h_min_rel relative to
    the element's own chord), all divided by level."""
    P = np.asarray(raw, float)
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-12]
    P = P[keep]
    ce = element_chord(P)
    turn = _turn(P)
    corner = np.r_[0, 1 + np.where(np.abs(turn) > math.radians(corner_deg))[0], len(P) - 1]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    # dense sample piece by piece (splines do not cross corners)
    dense_s, dense_xy, dense_k = [], [], []
    for a, b in zip(corner[:-1], corner[1:]):
        seg = slice(a, b + 1)
        ss, pp = s[seg], P[seg]
        if len(ss) >= 4:
            sx, sy = CubicSpline(ss, pp[:, 0]), CubicSpline(ss, pp[:, 1])
            t = np.linspace(ss[0], ss[-1], max(50, 40 * (b - a)))
            dx, dy, ddx, ddy = sx(t, 1), sy(t, 1), sx(t, 2), sy(t, 2)
            k = (dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-30)
            xy = np.c_[sx(t), sy(t)]
        else:                                             # straight piece
            t = np.linspace(ss[0], ss[-1], 20)
            xy = np.c_[np.interp(t, ss, pp[:, 0]), np.interp(t, ss, pp[:, 1])]
            k = np.zeros_like(t)
        dense_s.append(t); dense_xy.append(xy); dense_k.append(k)
    s_fixed = s[corner]
    out = [P[0]]
    for t, xy, k in zip(dense_s, dense_xy, dense_k):
        h = _spacing_field(xy, k, t, s_fixed, h_max * chord_ref / level,
                           h_min_rel * ce / level, beta / level, grade)
        # equidistribute 1/h, n panels with exact end points
        w = np.r_[0.0, np.cumsum(0.5 * (1 / h[1:] + 1 / h[:-1]) * np.diff(t))]
        n = max(2, int(math.ceil(w[-1])))
        tt = np.interp(np.linspace(0, w[-1], n + 1), w, t)
        seg = np.c_[np.interp(tt, t, xy[:, 0]), np.interp(tt, t, xy[:, 1])]
        seg[0], seg[-1] = xy[0], xy[-1]
        out.extend(seg[1:])
    return np.array(out)


def _short(path):
    p = Path(str(path))
    try:
        return str(p.resolve().relative_to(ROOT))
    except (ValueError, OSError):
        return str(path)


def element_chord(P):
    """Largest distance from the trailing edge (midpoint of the end points)."""
    te = 0.5 * (P[0] + P[-1])
    return float(np.linalg.norm(P - te, axis=1).max())


def blunt(open_loop, te_thick, n_te=3, blend=0.03):
    """Open outline (TE -> TE) -> closed CCW mesher loop with a blunt base:
    a sharp TE (end gap < te_thick) is thickened to a te_thick base at its
    original position; an existing base (gap >= te_thick) is kept."""
    P = np.asarray(open_loop, float)
    gap = np.linalg.norm(P[0] - P[-1])
    if gap < te_thick:
        # thicken the last part of both sides (offset along the outward
        # normal, (1 - d/L)^2 blend over L = blend x element chord) so the
        # trailing edge opens to a te_thick base at its original position.
        # Cutting the wedge instead (as airfoil_geom does for a single
        # airfoil, then rescaling) would move a thin main-element TE
        # upstream by ~0.5 % c and widen the slot to the flap.
        P = P.copy()
        add = 0.5 * (te_thick - gap)
        e = np.diff(P, axis=0)
        e /= np.linalg.norm(e, axis=1)[:, None]
        t = np.r_[[e[0]], 0.5 * (e[1:] + e[:-1]), [e[-1]]]
        t /= np.linalg.norm(t, axis=1)[:, None]
        nrm = np.c_[t[:, 1], -t[:, 0]]                    # outward for a CCW outline
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        L = blend * element_chord(P)
        d = np.minimum(s, s[-1] - s)                      # arc distance to the TE
        w = np.where(d < L, (1 - d / L) ** 2, 0.0)
        P += (add * w)[:, None] * nrm
    t = np.linspace(0, 1, n_te + 1)[1:-1]
    base = P[-1] + t[:, None] * (P[0] - P[-1])
    loop = np.vstack([P, base])
    if signed_area(loop) < 0:
        raise ValueError("element outline is clockwise after blunting")
    return loop


def round_corners(loop, r_max, min_turn_deg=30.0, step_deg=15.0, fill=0.45, closed=True,
                  which="both"):
    """Replace every corner of a closed loop (turn > min_turn_deg, convex or
    concave) by a circular arc, points every <= step_deg. Radius r_max,
    reduced so the arc uses at most `fill` of each adjacent edge.
    A sharp corner would need a fan of first-layer cells around one wall
    node (1e-12 c^2 cells next to 1e-6-thin quads: the SA run diverged
    there at the first iterations); an arc of a few points is extruded like
    any smooth wall. The radii are ~1e-4 c: no aerodynamic effect."""
    P = np.asarray(loop, float)
    n = len(P)
    out = []
    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            out.append(P[i])
            continue
        A, V, B = P[i - 1], P[i], P[(i + 1) % n]
        a, b = V - A, B - V
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        a, b = a / la, b / lb
        th = math.atan2(a[0] * b[1] - a[1] * b[0], a @ b)     # + left (convex, CCW)
        if abs(th) < math.radians(min_turn_deg) or (which == "concave" and th > 0) or \
                (which == "convex" and th < 0):
            out.append(V)
            continue
        tl = fill * min(la, lb)
        r = min(r_max, tl / math.tan(abs(th) / 2))
        tl = r * math.tan(abs(th) / 2)
        S = V - a * tl
        left = np.array([-a[1], a[0]])
        C = S + (left if th > 0 else -left) * r
        m = int(math.ceil(abs(th) / math.radians(step_deg)))
        phi0 = math.atan2(S[1] - C[1], S[0] - C[0])
        for k in range(m + 1):
            ph = phi0 + th * k / m
            out.append(C + r * np.array([math.cos(ph), math.sin(ph)]))
    return np.array(out)


# ----------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------
def _seg_intersect(a, b, c, d):
    """All pairs: segments a[i]-b[i] vs c[j]-d[j] properly intersect?"""
    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - \
               (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    A, B = a[:, None], b[:, None]
    C, D = c[None], d[None]
    o1, o2 = orient(A, B, C), orient(A, B, D)
    o3, o4 = orient(C, D, A), orient(C, D, B)
    return (o1 * o2 < 0) & (o3 * o4 < 0)


def loops_intersect(L1, L2):
    a, b = L1, np.roll(L1, -1, axis=0)
    c, d = L2, np.roll(L2, -1, axis=0)
    return bool(_seg_intersect(a, b, c, d).any())


def point_in_loop(p, L):
    x, y = p
    xs, ys = L[:, 0], L[:, 1]
    xn, yn = np.roll(xs, -1), np.roll(ys, -1)
    c = ((ys > y) != (yn > y)) & (x < (xn - xs) * (y - ys) / (yn - ys + 1e-300) + xs)
    return bool(np.sum(c) % 2)


def min_distance(L1, L2):
    """Smallest distance between two closed polylines (points to segments)."""
    def pts_to_segs(P, L):
        a, b = L, np.roll(L, -1, axis=0)
        ab = b - a
        t = np.clip(np.einsum('pij,ij->pi', P[:, None] - a[None], ab) /
                    np.maximum((ab**2).sum(1), 1e-300)[None], 0, 1)
        q = a[None] + t[..., None] * ab[None]
        return np.linalg.norm(P[:, None] - q, axis=2).min()
    return float(min(pts_to_segs(L1, L2), pts_to_segs(L2, L1)))


# ----------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------
def load_multi(spec, level=1.0, te_thick_rel=0.001, n_te=3, r_corner_rel=0.002,
               r_concave_rel=0.006):
    """spec -> (elements, info).
    elements[k]: name, loop (closed CCW mesher loop, corners rounded with
    radius <= r_corner_rel x element chord), outline (open TE -> TE,
    sharp, for the panel method), chord, te (trailing-edge midpoint)."""
    cfg = parse_spec(spec)
    ref_chord = float(cfg.get("ref_chord", 1.0))
    elements = []
    for e in cfg["elements"]:
        raw = raw_outline(e)
        # concave corners (coves) get a real fillet BEFORE resampling: the
        # quad layers of the two faces would otherwise be squeezed into the
        # corner and leave a pocket of slivers (the SA run diverged there).
        # Radius 0.6 % of the element chord, inside a separated cove.
        raw = round_corners(raw, r_concave_rel * element_chord(raw), closed=False,
                            which="concave", step_deg=5.0)
        out = resample(raw, chord_ref=ref_chord, level=level)
        ce = element_chord(out)
        loop = round_corners(blunt(out, te_thick_rel * ce, n_te=n_te), r_corner_rel * ce)
        elements.append({"name": e.get("name", f"element{len(elements) + 1}"), "loop": loop,
                         "outline": out, "chord": ce, "te": 0.5 * (out[0] + out[-1]),
                         "source": _short(e.get("file") or e.get("airfoil"))})
    names = [e["name"] for e in elements]
    if len(set(names)) != len(names):
        raise ValueError(f"element names must be unique: {names}")
    for e in elements:
        a = e["loop"]
        b = np.roll(a, -1, axis=0)
        hits = _seg_intersect(a, b, a, b)
        n = len(a)
        i, j = np.nonzero(hits)
        bad = [(p, q) for p, q in zip(i, j) if abs(p - q) > 1 and abs(p - q) != n - 1]
        if bad:
            raise ValueError(f"element '{e['name']}' outline intersects itself")
    gaps = {}
    for p in range(len(elements)):
        for q in range(p + 1, len(elements)):
            A, B = elements[p]["loop"], elements[q]["loop"]
            if loops_intersect(A, B) or point_in_loop(A[0], B) or point_in_loop(B[0], A):
                raise ValueError(f"elements '{elements[p]['name']}' and "
                                 f"'{elements[q]['name']}' overlap")
            gaps[f"{elements[p]['name']}-{elements[q]['name']}"] = min_distance(A, B)
    info = {"name": cfg.get("name", str(spec)), "ref_chord": ref_chord,
            "moment_ref": list(cfg.get("moment_ref", [0.25 * ref_chord, 0.0])),
            "elements": [{"name": e["name"], "chord": round(e["chord"], 5),
                          "points": int(len(e["loop"])), "source": e["source"]} for e in elements],
            "min_gaps": {k: round(v, 5) for k, v in gaps.items()}, "multi": True}
    return elements, info
