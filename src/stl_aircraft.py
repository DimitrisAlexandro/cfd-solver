"""STL <-> aircraft model.

Import: an aircraft surface mesh (binary or ASCII STL) is cut into slices.
- y = const slices: airfoil-shaped loops are the wing, horizontal tail and canard.
- z = const slices: airfoil-shaped loops near the plane of symmetry are fins.
- x = const slices: the round loop on the axis is the fuselage.

Loops are tracked from slice to slice into surfaces. Each surface is fitted
by least squares with the model's straight-tapered panel (chord, quarter-chord
line, dihedral and incidence linear along the span). Sections are taken from
the mesh: the nearest NACA 4-digit section (default) or the cut coordinates.

Export: the model as a closed surface mesh (lofted sections, body of
revolution) for CAD, meshing or a check of the import.

    tris = read_stl(open("plane.stl", "rb").read())        # (n, 3, 3)
    ac, warnings, info = to_aircraft(tris, units="mm")
    write_stl(aircraft_mesh(ac), "plane_model.stl")
"""
import math
import re
import struct
from pathlib import Path

import numpy as np

import aircraft as A

UNITS = {"m": 1.0, "dm": 0.1, "cm": 1e-2, "mm": 1e-3, "in": 0.0254, "ft": 0.3048}
AXES = {"x": 0, "y": 1, "z": 2}


# ----------------------------------------------------------------------
# files
# ----------------------------------------------------------------------
def read_stl(data):
    """triangles (n, 3, 3) from STL bytes (binary or ASCII)"""
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data) and n > 0:
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            return np.frombuffer(data, dtype=dt, count=n, offset=84)["v"].astype(float)
    txt = data.decode("ascii", "replace")
    v = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", txt)
    if not v or len(v) % 3:
        raise ValueError("not an STL file (no triangles found)")
    return np.array(v, float).reshape(-1, 3, 3)


def stl_bytes(tris, name="aircraft"):
    tris = np.asarray(tris, np.float32)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec = np.zeros(len(tris), dt)
    rec["n"], rec["v"] = n, tris
    head = f"AeroLab {name}"[:80].encode("ascii", "replace").ljust(80, b" ")
    return head + struct.pack("<I", len(tris)) + rec.tobytes()


def write_stl(tris, path, name="aircraft"):
    Path(path).write_bytes(stl_bytes(tris, name))


def orient(tris, length_axis="x", nose="min", up_axis="+z", units="m"):
    """aircraft axes: x aft from the nose, y to the right (span), z up; metres.
    The mesh is centred in y; x starts at 0 at the foremost point."""
    li = AXES[length_axis]
    ui = AXES[up_axis[-1]]
    if li == ui:
        raise ValueError("the length axis and the up axis must differ")
    si = 3 - li - ui
    P = tris.reshape(-1, 3) * UNITS[units]
    X = P[:, li] * (1 if nose == "min" else -1)
    Z = P[:, ui] * (-1 if up_axis.startswith("-") else 1)
    Y = P[:, si]
    X = X - X.min()
    Y = Y - 0.5 * (Y.min() + Y.max())
    return np.stack([X, Y, Z], 1).reshape(-1, 3, 3)


def indexed(tris):
    V, inv = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    T = inv.reshape(-1, 3)
    ok = (T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])
    return V, T[ok]


# ----------------------------------------------------------------------
# slicing
# ----------------------------------------------------------------------
class Slicer:
    def __init__(self, V, T):
        self.V, self.T = V, T

    def loops(self, axis, c):
        """closed intersection loops of the plane coord[axis] = c (ordered 3-D points)"""
        V, T = self.V, self.T
        d = V[:, axis] - c
        side = d > 0
        s = side[T]
        cross = s.any(1) & ~s.all(1)
        if not cross.any():
            return []
        Tc, sc = T[cross], s[cross]
        e = np.stack([Tc[:, [0, 1]], Tc[:, [1, 2]], Tc[:, [2, 0]]], 1)
        ec = np.stack([sc[:, 0] != sc[:, 1], sc[:, 1] != sc[:, 2], sc[:, 2] != sc[:, 0]], 1)
        sel = np.sort(e, axis=2)[ec].reshape(-1, 2, 2)
        keys, inv = np.unique(sel.reshape(-1, 2), axis=0, return_inverse=True)
        inv = inv.reshape(-1, 2)
        a, b = keys[:, 0], keys[:, 1]
        t = d[a] / (d[a] - d[b])
        P = V[a] + t[:, None] * (V[b] - V[a])
        nn = len(keys)
        src = np.concatenate([inv[:, 0], inv[:, 1]])
        dst = np.concatenate([inv[:, 1], inv[:, 0]])
        o = np.argsort(src, kind="stable")
        src, dst = src[o], dst[o]
        cnt = np.bincount(src, minlength=nn)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        adj0 = np.where(cnt >= 1, dst[np.minimum(start, len(dst) - 1)], -1)
        adj1 = np.where(cnt >= 2, dst[np.minimum(start + 1, len(dst) - 1)], -1)
        seen = np.zeros(nn, bool)
        out = []
        for st in range(nn):
            if seen[st]:
                continue
            path, prev, cur = [], -1, st
            while True:
                seen[cur] = True
                path.append(cur)
                nxt = adj0[cur] if adj0[cur] != prev else adj1[cur]
                if nxt < 0 or seen[nxt]:
                    break
                prev, cur = cur, nxt
            closed = len(path) > 2 and path[0] in (adj0[cur], adj1[cur])
            if len(path) >= 2:
                out.append((P[path], closed))
        return self._join([(p, c) for p, c in out], axis)

    @staticmethod
    def _join(chains, axis):
        """open chains (meshes that are not closed: separate left/right or upper/lower
        sheets) are joined end to end where their ends nearly meet"""
        closed = [p for p, c in chains if c and len(p) >= 4]
        opn = [p for p, c in chains if not c]
        if not opn:
            return closed

        def size(p):
            return float(np.linalg.norm(np.ptp(p, axis=0))) + 1e-12
        while True:
            best = None
            for i in range(len(opn)):
                for j in range(i + 1, len(opn)):
                    a, b = opn[i], opn[j]
                    tol = 0.05 * min(size(a), size(b))
                    for ea, eb in ((-1, 0), (-1, -1), (0, 0), (0, -1)):
                        d = float(np.linalg.norm(a[ea] - b[eb]))
                        if d < tol and (best is None or d < best[0]):
                            best = (d, i, j, ea, eb)
            if best is None:
                break
            _, i, j, ea, eb = best
            a, b = opn[i], opn[j]
            a = a if ea == -1 else a[::-1]           # a ends at the junction
            b = b if eb == 0 else b[::-1]            # b starts there
            opn = [q for k, q in enumerate(opn) if k not in (i, j)] + [np.vstack([a, b])]
        for p in opn:
            if len(p) < 4:
                continue
            if axis == 0:                 # body cross-sections: order by angle about the centre
                cols = [k for k in range(3) if k != axis]
                c = p[:, cols].mean(0)
                p = p[np.argsort(np.arctan2(p[:, cols[1]] - c[1], p[:, cols[0]] - c[0]))]
            closed.append(p)
        return closed


def _area(P2):
    x, y = P2[:, 0], P2[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def section(P2, n=60):
    """airfoil properties of a closed loop in (chordwise, normal) coordinates"""
    iTE = int(np.argmax(P2[:, 0]))
    dist = np.linalg.norm(P2 - P2[iTE], axis=1)
    iLE = int(np.argmax(dist))
    for _ in range(3):                  # leading edge: foremost point along the chord line
        ex = (P2[iTE] - P2[iLE]) / np.linalg.norm(P2[iTE] - P2[iLE])
        iLE = int(np.argmin((P2 - P2[iTE]) @ ex))
    LE, TE = P2[iLE], P2[iTE]
    c = float(np.linalg.norm(TE - LE))
    ex = (TE - LE) / c
    ez = np.array([-ex[1], ex[0]])
    loc = np.stack([(P2 - LE) @ ex, (P2 - LE) @ ez], 1) / c
    k = len(P2)
    i, j = iLE, iTE
    p1 = loc[[(i + q) % k for q in range((j - i) % k + 1)]]
    p2 = loc[[(i - q) % k for q in range((i - j) % k + 1)]]
    xg = 0.5 * (1 - np.cos(np.linspace(0, math.pi, n)))

    def curve(p):
        o = np.argsort(p[:, 0], kind="stable")
        x, y = p[o, 0], p[o, 1]
        x, idx = np.unique(x, return_index=True)
        return np.interp(xg, x, y[idx])
    y1, y2 = curve(p1), curve(p2)
    yu, yl = (y1, y2) if y1.mean() >= y2.mean() else (y2, y1)
    yu[0] = yl[0] = 0.0
    yu[-1] = yl[-1] = 0.5 * (yu[-1] + yl[-1])
    return {"le": LE, "te": TE, "chord": c, "qc": LE + 0.25 * (TE - LE),
            "inc": math.degrees(math.atan2(LE[1] - TE[1], TE[0] - LE[0])), "xg": xg, "yu": yu, "yl": yl,
            "t": float((yu - yl).max())}


def naca_fit(sec):
    zc = 0.5 * (sec["yu"] + sec["yl"])
    i = int(np.argmax(zc))
    m, p, t = max(zc[i], 0.0), sec["xg"][i], sec["t"]
    M = int(min(max(round(100 * m), 0), 9))
    P = int(min(max(round(10 * p), 1), 9)) if M else 0
    T = int(min(max(round(100 * t), 4), 40))
    return f"naca{M}{P}{T:02d}"


def section_dat(sec, path, name):
    xg = sec["xg"]
    up = np.stack([xg[::-1], sec["yu"][::-1]], 1)
    lo = np.stack([xg[1:], sec["yl"][1:]], 1)
    pts = np.vstack([up, lo])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(name + "\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in pts) + "\n")
    return str(path)


# ----------------------------------------------------------------------
# lifting surfaces
# ----------------------------------------------------------------------
def _tracks(sl, axis, wcol, stations, L, thin=0.25):
    """airfoil loops on planes coord[axis] = s, linked from plane to plane"""
    tracks, ds = [], stations[1] - stations[0]
    for k, s in enumerate(stations):
        secs = []
        for P in sl.loops(axis, s):
            P2 = P[:, [0, wcol]]
            ex, ew = np.ptp(P2[:, 0]), np.ptp(P2[:, 1])
            if ex <= 1e-6 or ex > 0.6 * L or ew / ex > thin:
                continue
            q = section(P2)
            q["s"], q["k"] = float(s), k
            q["xr"] = (float(P2[:, 0].min()), float(P2[:, 0].max()))
            q["wc"] = float(P2[:, 1].mean())
            secs.append(q)
        used = set()
        cand = []
        for ti, tr in enumerate(tracks):
            last = tr[-1]
            if k - last["k"] > 2:
                continue
            for si, q in enumerate(secs):
                ov = min(last["xr"][1], q["xr"][1]) - max(last["xr"][0], q["xr"][0])
                cmin = min(last["chord"], q["chord"])
                if ov > 0.4 * cmin and abs(q["wc"] - last["wc"]) < 0.5 * max(last["chord"], q["chord"]) + 2 * ds \
                        and 0.75 < q["chord"] / last["chord"] < 1.33:
                    cand.append((ov / cmin, ti, si))
        took_t = set()
        for _, ti, si in sorted(cand, reverse=True):
            if ti in took_t or si in used:
                continue
            tracks[ti].append(secs[si])
            took_t.add(ti)
            used.add(si)
        for si, q in enumerate(secs):
            if si not in used:
                tracks.append([q])
    return tracks


def _segments(Y, tol=0.03, max_seg=4):
    """split a sampled curve (n x channels, normalised) into straight pieces: greedy
    Douglas-Peucker on the largest deviation, at most max_seg pieces"""
    def dev(i0, i1):
        if i1 - i0 < 2:
            return 0.0, i0
        t = np.linspace(0, 1, i1 - i0 + 1)[:, None]
        d = np.abs(Y[i0:i1 + 1] - (Y[i0] + t * (Y[i1] - Y[i0]))).max(1)
        k = int(np.argmax(d[1:-1])) + 1
        return float(d[k]), i0 + k
    br = [0, len(Y) - 1]
    while len(br) - 1 < max_seg:
        worst = max(((dev(a, b), a, b) for a, b in zip(br[:-1], br[1:])), key=lambda v: v[0][0])
        (d, k), a, b = worst
        if d <= tol or k - a < 2 or b - k < 2:
            break
        br = sorted(br + [k])
    return br


def _fit(tr, V, axis, wcol, s_root):
    """piecewise-linear fit of a surface track: straight panels between kinks.
    Returns the root -> tip definition, the kinks and the fit quality."""
    s = np.array([q["s"] for q in tr])
    c = np.array([q["chord"] for q in tr])
    xq = np.array([q["qc"][0] for q in tr])
    wq = np.array([q["qc"][1] for q in tr])
    inc = np.array([q["inc"] for q in tr])
    keep = s <= s[0] + 0.92 * (s[-1] - s[0])
    if keep.sum() < 3:
        keep[:] = True
    for _ in range(2):                  # drop outliers (fairings, junctions): 3 x median deviation
        f = {k: np.polyfit(s[keep], v[keep], 1) for k, v in (("c", c), ("x", xq), ("w", wq), ("i", inc))}
        res = np.abs(c - np.polyval(f["c"], s)) + np.abs(xq - np.polyval(f["x"], s))
        lim = 3 * np.median(res[keep]) + 1e-3 * c.mean()
        new = keep & (res <= lim)
        if new.sum() < 3 or new.sum() == keep.sum():
            break
        keep = new
    # straight pieces (kinks where the planform, dihedral or twist line breaks)
    sk, ck, xk, wk, ik = s[keep], c[keep], xq[keep], wq[keep], inc[keep]
    cm = float(ck.mean())
    Y = np.stack([ck / cm, xk / cm, wk / cm, ik / 25.0], 1)
    br = _segments(Y) if len(sk) >= 7 else [0, len(sk) - 1]
    fits = []
    for a, b in zip(br[:-1], br[1:]):
        sl = slice(a, b + 1)
        fits.append({k: np.polyfit(sk[sl], v[sl], 1) for k, v in (("c", ck), ("x", xk), ("w", wk), ("i", ik))})
    kinks_s = []
    for f1, f2, b in zip(fits[:-1], fits[1:], br[1:-1]):
        # break position: where the lines of the channel with the largest slope change meet
        dsl = {k: abs(f1[k][0] - f2[k][0]) / (cm if k != "i" else 25.0) for k in f1}
        k = max(dsl, key=dsl.get)
        den = f1[k][0] - f2[k][0]
        y = (f2[k][1] - f1[k][1]) / den if abs(den) > 1e-12 else sk[b]
        kinks_s.append(float(np.clip(y, sk[max(b - 1, 0)], sk[min(b + 1, len(sk) - 1)])))
    # tip: outermost mesh point in the box of the last section
    q = tr[-1]
    m = ((V[:, 0] > q["xr"][0] - 0.5 * q["chord"]) & (V[:, 0] < q["xr"][1] + 0.5 * q["chord"])
         & (np.abs(V[:, wcol] - q["qc"][1]) < 0.6 * q["chord"]) & (V[:, axis] >= s[-1]))
    ds = (s[-1] - s[0]) / max(len(s) - 1, 1)
    s_tip = float(V[m, axis].max()) if m.any() else s[-1] + 0.5 * ds
    s_tip = min(s_tip, s[-1] + 1.5 * ds)
    if s_root is None:
        s_root = s[0] - 0.5 * ds

    def ev(f, k, x):
        return float(np.polyval(f[k], x))
    st = [(s_root, *(ev(fits[0], k, s_root) for k in "cxwi"))]
    for y, f1, f2 in zip(kinks_s, fits[:-1], fits[1:]):
        st.append((y, *(0.5 * (ev(f1, k, y) + ev(f2, k, y)) for k in "cxwi")))
    st.append((s_tip, *(ev(fits[-1], k, s_tip) for k in "cxwi")))
    st = np.array(st)                                   # s, chord, x_qc, w_qc, incidence
    st[:, 1] = np.maximum(st[:, 1], 0.02 * st[0, 1])
    sr, cr, xr, wr, ir = st[0]
    stp, ct, xt, wt, it = st[-1]
    semi = stp - sr
    x_le_root = xr - 0.25 * cr
    kinks = [{"y": round(float(y - sr), 5), "chord": round(float(cc), 5),
              "dx": round(float(xx - 0.25 * cc - x_le_root), 5), "dz": round(float(ww - wr), 5),
              "twist": round(float(ii - ir), 4)} for y, cc, xx, ww, ii in st[1:-1]]
    a_mesh = float(np.trapezoid(c, s)) + 0.5 * ds * (c[0] + c[-1])
    if s_root < s[0] - ds:              # root extended into the fuselage
        a_mesh += (s[0] - s_root) * c[0]
    return {"s": s, "s_root": sr, "s_tip": stp, "semi": semi, "cr": cr, "ct": ct,
            "x_le": x_le_root, "w": wr,
            "sweep": math.degrees(math.atan2(xt - xr, semi)), "slope_w": math.degrees(math.atan2(wt - wr, semi)),
            "inc": ir, "twist": it - ir, "kinks": kinks,
            "area_fit": float(np.trapezoid(st[:, 1], st[:, 0])), "area_mesh": a_mesh, "n": len(tr),
            "c_mean": float(c.mean())}


def _pick_sections(tr, fit, mode, af_dir, tag):
    """root and tip sections: the cuts at 20 % and 85 % of the track, extrapolated
    linearly to the root and tip stations -> airfoil specs"""
    n = len(tr)
    qa, qb = tr[min(int(0.2 * n), n - 1)], tr[min(int(0.85 * n), n - 1)]
    sa, sb = qa["s"], qb["s"]
    out = []
    for st, where in ((fit["s_root"], "root"), (fit["s_tip"], "tip")):
        w = 0.0 if sb - sa < 1e-9 else float(np.clip((st - sa) / (sb - sa), -0.5, 1.5))
        q = dict(qa)
        q["yu"] = (1 - w) * qa["yu"] + w * qb["yu"]
        q["yl"] = np.minimum((1 - w) * qa["yl"] + w * qb["yl"], q["yu"])
        q["t"] = float((q["yu"] - q["yl"]).max())
        if mode == "cut":
            out.append(section_dat(q, Path(af_dir) / f"{tag}_{where}.dat", f"{tag} {where} (cut from STL)"))
        else:
            out.append(naca_fit(q))
    return out[0], (out[1] if out[1] != out[0] else None)


def _fuselage(sl, V, L, n=60):
    xs = np.linspace(0, L, n + 2)[1:-1]
    rows = []
    for x in xs:
        best = None
        for P in sl.loops(0, x):
            P2 = P[:, [1, 2]]
            y0, y1 = P2[:, 0].min(), P2[:, 0].max()
            w, h = y1 - y0, np.ptp(P2[:, 1])
            if not (y0 < 1e-6 < y1) or h <= 0 or not 0.4 < w / h < 2.5:
                continue
            a = _area(P2)
            if best is None or a > best[0]:
                best = (a, 2 * math.sqrt(a / math.pi), float(0.5 * (P2[:, 1].min() + P2[:, 1].max())))
        if best:
            rows.append((x, best[1], best[2]))
    if len(rows) < 5:
        return None
    R = np.array(rows)
    i = int(np.argmax(R[:, 1]))
    d, zc = R[i, 1], R[i, 2]
    m = (np.abs(V[:, 1]) < 0.5 * d) & (np.abs(V[:, 2] - zc) < 0.5 * d)
    x0, x1 = (float(V[m, 0].min()), float(V[m, 0].max())) if m.any() else (R[0, 0], R[-1, 0])
    length = x1 - x0
    return {"x": x0, "length": length, "diameter": float(d), "z": float(zc),
            "x_max": float(np.clip((R[i, 0] - x0) / length, 0.05, 0.9)),
            "tail_ratio": float(np.clip(R[-1, 1] / d, 0.0, 1.0)), "stations": R}


def _inside(hfit, htr, x, y, z):
    """is the point within horizontal surface (track) htr?"""
    s = np.array([q["s"] for q in htr])
    if not s[0] - 1e-9 <= abs(y) <= s[-1] + 1e-9:
        return False
    k = int(np.argmin(np.abs(s - abs(y))))
    q = htr[k]
    return q["xr"][0] - 0.05 * q["chord"] <= x <= q["xr"][1] and abs(z - q["qc"][1]) < 0.35 * q["chord"]


def to_aircraft(tris, length_axis="x", nose="min", up_axis="+z", units="m", sections="naca",
                af_dir=".", mass=750.0, cg_mac=0.25, add_controls=True, name="Imported from STL", n_slices=48):
    """aircraft model from a surface mesh; returns (aircraft, warnings, info)"""
    warns = []
    tri = orient(tris, length_axis, nose, up_axis, units)
    V, T = indexed(tri)
    if len(T) < 20:
        raise ValueError("too few triangles")
    L, B, H = np.ptp(V, axis=0)
    if abs(V[:, 1].max() + V[:, 1].min()) > 0.02 * B:
        warns.append("The mesh is not symmetric in y; the right half (+y) is used and mirrored.")
    sl = Slicer(V, T)
    fz = _fuselage(sl, V, L)
    if fz:
        V = V - [0, 0, fz["z"]]
        sl = Slicer(V, T)
        fz["z"] = 0.0
    else:
        warns.append("No fuselage found (no round cross-section on the axis); enter it on the Edit tab.")

    # horizontal surfaces: y slices
    ymax = V[:, 1].max()
    ys = (np.arange(n_slices) + 0.5) / n_slices * ymax
    htr = [t for t in _tracks(sl, 1, 2, ys, L) if len(t) >= 4]
    hfits = [(t, _fit(t, V, 1, 2, 0.0)) for t in htr]
    hfits = [(t, f) for t, f in hfits if abs(f["slope_w"]) < 60]
    if not hfits:
        raise ValueError("no wing found: no airfoil-shaped sections in the y = const cuts. Check the axes "
                         "(span must be the remaining axis) and the units.")
    big = max(f["area_fit"] for _, f in hfits)
    cm = max(f["c_mean"] for _, f in hfits if f["area_fit"] == big)
    keep = []
    for t, f in hfits:
        if f["area_fit"] < 0.03 * big or f["c_mean"] < 0.12 * cm:
            warns.append(f"A small horizontal element at x ≈ {f['x_le']:.2f} m (area "
                         f"{f['area_fit']:.3f} m², strut or fairing?) is ignored.")
        else:
            keep.append((t, f))
    keep.sort(key=lambda tf: -tf[1]["area_fit"])

    # vertical surfaces: z slices
    zs = np.linspace(V[:, 2].min(), V[:, 2].max(), n_slices + 2)[1:-1]
    vtr = [t for t in _tracks(sl, 2, 1, zs, L) if len(t) >= 4]
    vkeep = []
    for t in vtr:
        f = _fit(t, V, 2, 1, None)
        if f["semi"] < 0.04 * 2 * ymax or f["semi"] < 0.3 * f["c_mean"]:
            continue
        hits = sum(any(_inside(hf, ht, q["qc"][0], q["qc"][1], q["s"]) for ht, hf in keep) for q in t)
        if hits > 0.5 * len(t):
            continue
        vkeep.append((t, f))

    surfs, info = [], {"surfaces": []}
    wing_x = keep[0][1]["x_le"]
    names = {}
    for idx, (t, f) in enumerate(keep):
        nm = "wing" if idx == 0 else ("horizontal tail" if f["x_le"] > wing_x else "canard")
        names[nm] = names.get(nm, 0) + 1
        nm = nm if names[nm] == 1 else f"{nm} {names[nm]}"
        af, af_t = _pick_sections(t, f, sections, af_dir, re.sub(r"\W+", "_", f"{name}_{nm}"))
        s = A.Surface(nm, round(2 * f["semi"], 4), round(f["cr"], 4), round(f["ct"], 4), round(f["x_le"], 4),
                      0.0, round(f["w"], 4), round(f["sweep"], 2), round(f["slope_w"], 2), round(f["twist"], 2),
                      round(f["inc"], 2), af, af_t, True, False, [], 14 if idx == 0 else 8, 6 if idx == 0 else 5,
                      kinks=f["kinks"])
        surfs.append(s)
        info["surfaces"].append((nm, f))
    for t, f in vkeep:
        yc = float(np.mean([q["qc"][1] for q in t]))
        nm = "fin" if abs(yc) < 0.1 * ymax else ("winglet" if abs(yc) > 0.8 * ymax else "fin")
        names[nm] = names.get(nm, 0) + 1
        nm = nm if names[nm] == 1 else f"{nm} {names[nm]}"
        if abs(f["inc"]) > 1.0:
            warns.append(f"'{nm}' is set at {f['inc']:.1f}° to the x axis in the mesh; the model uses 0°.")
        af, af_t = _pick_sections(t, f, sections, af_dir, re.sub(r"\W+", "_", f"{name}_{nm}"))
        s = A.Surface(nm, round(f["semi"], 4), round(f["cr"], 4), round(f["ct"], 4), round(f["x_le"], 4),
                      round(yc, 4), round(f["s_root"], 4), round(f["sweep"], 2), 0.0, 0.0, 0.0, af, af_t,
                      False, True, [], 8, 5, kinks=[{**k, "dz": 0.0} for k in f["kinks"]])
        surfs.append(s)
        info["surfaces"].append((nm, f))
    if not vkeep:
        warns.append("No fin found in the z = const cuts.")
    for nm, f in info["surfaces"]:
        e = f["area_fit"] / max(f["area_mesh"], 1e-12) - 1
        if abs(e) > 0.04:
            warns.append(f"'{nm}': the tapered-panel fit has {100 * e:+.0f} % of the mesh's planform area "
                         "(cranked or elliptic planform); adjust the chords on the Edit tab if needed.")

    fus = A.Fuselage()
    if fz:
        fus = A.Fuselage(round(fz["length"], 4), round(fz["diameter"], 4), round(fz["x"], 4), 0.0,
                         round(fz["x_max"], 3), round(fz["tail_ratio"], 3))
    ac = A.Aircraft(name, surfs, fus, mass=float(mass))
    if add_controls:
        r_f = 0.5 * fus.diameter / max(ac.wing.semi, 1e-9) if fz else 0.1
        ac.wing.controls = [A.Control("aileron", 0.58, 0.95, 0.25, "antisymmetric"),
                            A.Control("flap", round(min(r_f + 0.03, 0.5), 3), 0.56, 0.25, "symmetric")]
        for s in surfs[1:]:
            if s.name.startswith("horizontal tail") and not any(c.name == "elevator" for x in surfs for c in x.controls):
                s.controls = [A.Control("elevator", 0.05, 1.0, 0.35, "symmetric")]
            if s.name == "fin":
                s.controls = [A.Control("rudder", 0.08, 0.95, 0.35, "symmetric")]
        warns.append("Typical control surfaces were added (aileron, flap, elevator, rudder); set their "
                     "real spans and chords on the Edit tab.")
    xm, _ = ac.wing.mac_position()
    ac.cg = (round(xm + cg_mac * ac.wing.mac, 4), 0.0, 0.0)
    info["fuselage"] = fz
    info["mesh"] = {"triangles": int(len(T)), "length": float(L), "span": float(B), "height": float(H)}
    info["V"], info["T"] = V, T
    return ac, warns, info


# ----------------------------------------------------------------------
# export
# ----------------------------------------------------------------------
def _section_ring(af, n=40):
    import airfoil_geom as ag
    o = ag.sharp_outline(af)
    i = int(np.argmin(o[:, 0]))
    up, lo = o[:i + 1][::-1], o[i:]
    xg = 0.5 * (1 - np.cos(np.linspace(0, math.pi, n)))

    def f(p):
        x, idx = np.unique(p[:, 0], return_index=True)
        return np.interp(xg, x, p[idx, 1])
    yu, yl = f(up), f(lo)
    te = 0.5 * (yu[-1] + yl[-1])          # chord line through the trailing-edge midpoint
    yu, yl = yu - te * xg, yl - te * xg
    yu[-1] = yl[-1] = 0.0
    yu[0] = yl[0] = 0.0
    ring = np.vstack([np.stack([xg[::-1], yu[::-1]], 1), np.stack([xg[1:-1], yl[1:-1]], 1)])
    k = int(np.argmin(np.abs(xg - 0.3)))
    return ring, np.array([0.3, 0.5 * (yu[k] + yl[k])])


def _loft(rings, caps):
    """triangles between consecutive rings (m, k, 3) + fan caps at both ends"""
    tri = []
    m, k = rings.shape[:2]
    for a in range(m - 1):
        for j in range(k):
            p, q = rings[a, j], rings[a, (j + 1) % k]
            r, s = rings[a + 1, j], rings[a + 1, (j + 1) % k]
            tri += [(p, q, s), (p, s, r)]
    for ring, c in zip((rings[0], rings[-1]), caps):
        for j in range(k):
            tri.append((c, ring[(j + 1) % k], ring[j]))
    return tri


def surface_mesh(s, n_span=12, n_af=40):
    r_root, c_root = _section_ring(s.airfoil, n_af)
    r_tip, c_tip = _section_ring(s.tip_airfoil, n_af) if s.tip_airfoil else (r_root, c_root)
    half = np.unique(np.r_[np.linspace(0, 1, n_span + 1), s.stations()[0]])   # kinks are stations
    etas = np.r_[-half[::-1], half[1:]] if s.symmetric else half

    def place(e, pts):
        ae = abs(e)
        c = float(s.chord(ae))
        i = math.radians(s.incidence + float(s.twist_at(ae)))
        dx = (pts[:, 0] - 0.25) * c
        yt = pts[:, 1] * c
        xq = float(s.x_le(ae)) + 0.25 * c
        if s.vertical:
            return np.stack([xq + dx * math.cos(i) + yt * math.sin(i),
                             s.y + yt * math.cos(i) - dx * math.sin(i),
                             np.full(len(pts), s.z + ae * s.semi)], 1)
        return np.stack([xq + dx * math.cos(i) + yt * math.sin(i),
                         np.full(len(pts), s.y + e * s.semi),
                         s.z + float(s.rise(ae)) - dx * math.sin(i) + yt * math.cos(i)], 1)
    rings, caps = [], []
    for e in etas:
        w = abs(e)
        rings.append(place(e, (1 - w) * r_root + w * r_tip))
    for e in (etas[0], etas[-1]):
        w = abs(e)
        caps.append(place(e, ((1 - w) * c_root + w * c_tip)[None])[0])
    return _loft(np.array(rings), caps)


def fuselage_mesh(f, n_x=60, n_t=32):
    if f.length <= 0 or f.diameter <= 0:
        return []
    xs = f.x + f.length * 0.5 * (1 - np.cos(np.linspace(0, math.pi, n_x)))
    r = f.radius(xs)
    r[0] = 0.0
    th = np.linspace(0, 2 * math.pi, n_t, endpoint=False)
    rings = np.stack([np.stack([np.full(n_t, x), rr * np.cos(th), f.z + rr * np.sin(th)], 1)
                      for x, rr in zip(xs, r)])
    return _loft(rings, (rings[0, 0], np.array([xs[-1], 0.0, f.z])))


def aircraft_mesh(ac, n_span=12, n_af=40):
    tri = []
    for s in ac.surfaces:
        tri += surface_mesh(s, n_span, n_af)
    tri += fuselage_mesh(ac.fuselage)
    return np.array(tri, float)


def guess_orientation(tris, units="m"):
    """axes of a mesh: up = smallest extent; length axis and nose side and the
    sign of 'up' from trial imports (fuselage found, largest wing, tail aft, fin up)"""
    ext = np.ptp(tris.reshape(-1, 3), axis=0)
    up = "xyz"[int(np.argmin(ext))]
    best = None
    for la in "xyz":
        if la == up:
            continue
        try:
            ac, _, info = to_aircraft(tris, la, "min", "+" + up, units, add_controls=False, n_slices=24)
        except Exception:                       # noqa: BLE001
            continue
        score = ac.wing.area * (2.0 if info["fuselage"] else 1.0) * (ac.wing.span / max(info["mesh"]["span"], 1e-9))
        if best is None or score > best[0]:
            best = (score, la, ac)
    if best is None:
        return "x", "min", "+z"
    _, la, ac = best
    names = [s.name for s in ac.surfaces]
    nose = "max" if ("canard" in names and not any(n.startswith("horizontal tail") for n in names)) else "min"
    fins = [s for s in ac.surfaces if s.vertical]
    sign = "-" if fins and np.mean([s.z + 0.5 * s.span for s in fins]) < 0 else "+"
    return la, nose, sign + up


def guess_units(tris):
    return "mm" if np.ptp(tris.reshape(-1, 3), axis=0).max() > 100 else "m"
