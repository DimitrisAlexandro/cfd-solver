"""Wing meshes for the 3D solver (half wing on a symmetry plane).

Method
  1. 2D hybrid airfoil mesh (the validated mesher: quad boundary layer at y+ < 1,
     triangles outside, wake refinement), far field `farfield` root chords away.
  2. Flat tip cap: the airfoil interior is triangulated (our CDT, cells sized like
     the wall panels) and used beyond the tip only. Beyond the tip the wall-resolving
     rows are re-spaced over `tip_relax` spanwise levels (default 2) so that their
     first row matches the fill cells across the wall line: y+ ~ 1 rows next to much
     larger fill cells decouple pressure and velocity (the linear face weight -> 1
     toward the thin cell) and the run stalls or diverges at the tip. One level is too
     abrupt, eight leave thin rows next to the fill too long; two is stable to 8 deg+.
  3. The 2D mesh is swept through spanwise levels. At each level the section is
     scaled to the local chord, twisted about its quarter chord, and offset for sweep
     and dihedral; the transform is applied fully near the wing and blended smoothly
     to zero at the far field, so the outer boundary stays a fixed cylinder.
  4. Spanwise spacing: geometric clustering at the tip from both sides (first layer
     `tip_first` x tip chord), uniform-ish inboard, growing outboard to the far field.

Axes: x downstream (chordwise), y up (lift), z spanwise (root at z = 0).
Lengths in root chords; Re on the root chord. Patches: wing, symmetry, farfield.
"""
import math

import numpy as np

from . import meshgen3d as G


def _spacing_to(L, first, dmax, q):
    """node coordinates on [0, L], finest (first) at L, growing by q up to dmax toward 0"""
    steps = []
    s, tot = first, 0.0
    while tot + s < L and s < dmax:
        steps.append(s)
        tot += s
        s *= q
    rest = L - tot
    if rest > 0:
        nu = max(1, int(math.ceil(rest / dmax)))
        steps += [rest / nu] * nu
    z = L - np.r_[0.0, np.cumsum(steps)]
    z[-1] = 0.0
    return np.sort(z)


def _spacing_from(z0, L, first, dmax, q):
    """node coordinates from z0 to z0 + L, finest (first) at z0, growing by q to dmax"""
    steps = []
    s, tot = first, 0.0
    while tot + s < L:
        steps.append(s)
        tot += s
        s = min(s * q, dmax)
    steps[-1] += L - tot
    return z0 + np.r_[0.0, np.cumsum(steps)]


def _wall_loop(wall_edges):
    nxt = {}
    for a, b in wall_edges:
        nxt[int(b)] = int(a)                       # edges are (a, b) with b -> a along the loop
    start = int(wall_edges[0][1])
    loop = [start]
    while True:
        k = nxt[loop[-1]]
        if k == start:
            break
        loop.append(k)
    return loop


def _fill_airfoil(P2, wall_edges, h):
    """triangles inside the airfoil loop (wall nodes kept; our CDT), size h.
    Returns (new points, triangles in global node numbering)."""
    from cdt2d import CDT
    loop = _wall_loop(wall_edges)
    P = P2[loop]
    x0, y0 = P.min(0) - 0.5
    x1, y1 = P.max(0) + 0.5
    cdt = CDT(x0, y0, x1, y1)
    ids = [cdt.insert(x, y) for x, y in P]
    for i in range(len(ids)):
        cdt.insert_segment(ids[i], ids[(i + 1) % len(ids)], "wall")
    cdt.mark_regions()
    cdt.refine(lambda x, y: h, min_angle_deg=22.0, splittable=())
    gmap = {cid: loop[i] for i, cid in enumerate(ids)}
    new, newpts, out = {}, [], []
    for t in cdt.triangles(region=1):
        tt = []
        for v in t:
            if v not in gmap and v not in new:
                new[v] = len(P2) + len(newpts)
                newpts.append((cdt.x[v], cdt.y[v]))
            tt.append(gmap[v] if v in gmap else new[v])
        out.append(tt)
    return np.array(newpts).reshape(-1, 2), out


def _geom_dist(h, n, L):
    """n+1 cumulative positions 0..L (n cells), geometric with first cell h (uniform
    when n h >= L)"""
    if n * h >= L:
        return np.linspace(0.0, L, n + 1)
    lo, hi = 1.0 + 1e-12, 4.0
    for _ in range(80):
        g = 0.5 * (lo + hi)
        if h * (g ** n - 1) / (g - 1) > L:
            hi = g
        else:
            lo = g
    st = h * g ** np.arange(n)
    st *= L / st.sum()
    return np.r_[0.0, np.cumsum(st)]


def relax_map(P2, cells, n_bl, M, wall_edges, match=0.6):
    """Beyond the flat tip cap the wall-resolving rows are not needed, and y+ ~ 1 rows
    next to the much larger fill cells decouple pressure and velocity (the linear
    face weight -> 1 toward the thin cell; the run stalls or diverges at the tip).
    R(beta, P) gives the 2D node positions with every boundary-layer column re-spaced
    (beta = 1) to geometric rows whose first row matches the fill cells across the
    wall line (`match` x the local wall-panel length: the fill triangles there are
    about that size), by a geometric blend of the arc-length positions, so the row
    thicknesses change by a bounded factor from one spanwise level to the next.
    BL cell k*M + i = [node(k, i), node(k+1, i), node(k+1, i+1), ...]."""
    K = n_bl // M
    T = np.empty((K + 1, M), np.int64)
    T[0] = [cells[i][0] for i in range(M)]
    for k in range(K):
        T[k + 1] = [cells[k * M + i][1] for i in range(M)]
    X = P2[T]
    s0 = np.r_[np.zeros((1, M)), np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=2), axis=0)]
    # local wall-panel length at each wall node (mean of its two panels)
    plen = {}
    for a, b in np.asarray(wall_edges):
        L = float(np.linalg.norm(P2[a] - P2[b]))
        for v in (int(a), int(b)):
            plen.setdefault(v, []).append(L)
    h_t = np.array([match * np.mean(plen.get(int(T[0, c]), [s0[-1, c] / K])) for c in range(M)])
    lh = np.log(h_t)
    for _ in range(20):                          # smooth along the layer: neighbouring
        lh = 0.5 * lh + 0.25 * (np.roll(lh, 1) + np.roll(lh, -1))   # columns keep their rows level
    h_t = np.exp(lh)
    s1 = np.array([_geom_dist(h_t[c], K, s0[-1, c]) for c in range(M)]).T
    inner = T[1:K]

    def R(beta, P):
        out = P.copy()
        if beta <= 0:
            return out
        sb = np.exp((1 - beta) * np.log(np.maximum(s0[1:K], 1e-300)) + beta * np.log(s1[1:K]))
        for i in range(M):
            out[inner[:, i], 0] = np.interp(sb[:, i], s0[:, i], X[:, i, 0])
            out[inner[:, i], 1] = np.interp(sb[:, i], s0[:, i], X[:, i, 1])
        return out
    return R


def wing_mesh(airfoil="naca0012", semi_span=3.0, taper=1.0, sweep_c4=0.0, dihedral=0.0, twist=0.0,
              tip_airfoil=None, re=3e6, level=0.5, farfield=12.0, tip_first=0.01, growth=1.3,
              blend=(0.5, None), tip_relax=2, verbose=False):
    """polymesh dict of a half wing + info dict (reference area, MAC, span stations).
    twist: tip incidence relative to the root (deg, nose up +, linear); sweep_c4 and
    dihedral in degrees. tip_airfoil: not yet (same section along the span)."""
    if tip_airfoil not in (None, airfoil):
        raise NotImplementedError("one section along the span for now")
    import airfoil_geom as ag
    import validate_naca0012_sa as v
    wall, _ = ag.load_airfoil(airfoil, n_side=int(round(160 * level)))
    m2 = v.airfoil_mesh(level, 0.0, wall=wall, re=re, farfield=farfield)
    P2 = np.asarray(m2["points"], float)
    cells = [list(c) for c in m2["cells"]]
    nouter = len(cells)
    # tip cap closure: the airfoil interior is triangulated (cells used beyond the tip
    # only), sized like the wall panels
    we = np.asarray(m2["patches"]["airfoil"])
    h_fill = 0.8 * float(np.median(np.linalg.norm(P2[we[:, 0]] - P2[we[:, 1]], axis=1)))
    fillP, fill = _fill_airfoil(P2, we, h_fill)
    tipQ = relax_map(P2, cells, m2["n_bl"], np.asarray(m2["layers"]).shape[1], we)
    P2 = np.vstack([P2, fillP])
    cells += fill
    edge_patch = {(min(a, b), max(a, b)): "farfield" for a, b in m2["patches"]["farfield"]}

    # spanwise levels
    ct = taper
    dmax_in = max(semi_span / max(int(round(16 * level / 0.5)), 4), 3 * tip_first)
    z_in = _spacing_to(semi_span, tip_first * ct, dmax_in, growth)
    z_out = _spacing_from(semi_span, farfield, tip_first * ct, max(farfield / 8, dmax_in), growth)
    z = np.r_[z_in, z_out[1:]]
    itip = len(z_in) - 1                                   # level index of the tip plane
    K = len(z) - 1
    active = np.ones((K, len(cells)), bool)
    active[:itip, nouter:] = False                         # fill cells only beyond the tip

    # blending weight: 1 near the wing, smooth to 0 at the far field
    from scipy.spatial import cKDTree
    wpts = P2[np.unique(np.asarray(m2["patches"]["airfoil"]).ravel())]
    r, _ = cKDTree(wpts).query(P2)
    r1 = blend[0]
    r2 = blend[1] if blend[1] else 0.6 * farfield
    s = np.clip((r - r1) / (r2 - r1), 0, 1)
    wgt = 1 - s * s * (3 - 2 * s)                          # smoothstep

    tanL, tanG = math.tan(math.radians(sweep_c4)), math.tan(math.radians(dihedral))

    def section(eta):
        c = 1.0 + (ct - 1.0) * eta
        zz = eta * semi_span
        xq = 0.25 + zz * tanL                              # quarter-chord x
        yq = zz * tanG
        th = math.radians(twist * eta)
        return c, xq, yq, th

    def xyz(l):
        zl = z[l]
        eta = min(zl / semi_span, 1.0)
        c, xq, yq, th = section(eta)
        Q = tipQ(min(max(l - itip, 0) / tip_relax, 1.0), P2)
        dx, dy = Q[:, 0] - 0.25, Q[:, 1]
        ca, sa = math.cos(th), math.sin(th)
        X = xq + c * (ca * dx + sa * dy)                    # nose-up twist rotates LE upward
        Y = yq + c * (-sa * dx + ca * dy)
        Xb = Q[:, 0] + wgt * (X - Q[:, 0])
        Yb = Q[:, 1] + wgt * (Y - Q[:, 1])
        return np.c_[Xb, Yb, np.full(len(P2), zl)]

    pm = G.extrude(P2, cells, edge_patch, K + 1, xyz=xyz, active=active, cap_lower="symmetry",
                   cap_upper="farfield", side_inactive="wing", cap_inactive="wing")
    cell_l, cell_c2 = np.nonzero(active)                   # 3D cell -> (layer, 2D cell), as extrude numbers them
    S_half = semi_span * (1 + ct) / 2
    mac = 2.0 / 3.0 * (1 + ct + ct * ct) / (1 + ct)
    info = {"airfoil": airfoil, "semi_span": semi_span, "taper": ct, "sweep_c4": sweep_c4, "dihedral": dihedral,
            "twist": twist, "re_root": re, "level": level, "farfield": farfield, "S_half": S_half, "mac": mac,
            "aspect_ratio": (2 * semi_span) ** 2 / (2 * S_half), "z_levels": z.tolist(), "itip": itip,
            "cells_2d": len(cells), "layers": K, "x_ref": 0.25, "chord": lambda zz: 1.0 + (ct - 1.0) * np.clip(zz / semi_span, 0, 1),
            "section": section,
            # the 2D section mesh (to start the 3D run from a 2D solution) and the map
            "section_mesh": {"points": np.asarray(m2["points"], float), "cells": [list(c) for c in m2["cells"]],
                             "airfoil": np.asarray(m2["patches"]["airfoil"]),
                             "farfield": np.asarray(m2["patches"]["farfield"])},
            "cell_layer": cell_l, "cell_2d": cell_c2, "n_section_cells": nouter,
            "z_cell": 0.5 * (z[cell_l] + z[cell_l + 1])}
    if verbose:
        print(f"[wing mesh] 2D {nouter} + {len(fill)} fill cells, {K} layers ({itip} to the tip), "
              f"{pm['ncells']} cells")
    return pm, info
