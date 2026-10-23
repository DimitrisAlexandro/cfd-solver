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
     to zero at the far field, so the outer boundary stays a fixed cylinder. With a
     tip section the section shape itself also changes linearly along the span
     (tip_morph).
  4. Spanwise spacing: geometric clustering at the tip from both sides (first layer
     `tip_first` x tip chord), uniform-ish inboard, growing outboard to the far field.

Axes: x downstream (chordwise), y up (lift), z spanwise (root at z = 0).
Lengths in root chords; Re on the root chord. Patches: wing, symmetry, farfield.
"""
import math

import numpy as np

from . import meshgen3d as G


def _spacing_to(L, first, dmax, q, band=0.0, dband=None):
    """node coordinates on [0, L], finest (first) at L, growing by q up to dmax toward 0.
    Within `band` of L the spacing stays at most dband (the tip vortex)."""
    steps = []
    s, tot = first, 0.0
    while tot + s < L and s < dmax:
        if dband and tot < band and s > dband:
            s = dband
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


def _spacing_from(z0, L, first, dmax, q, band=0.0, dband=None):
    """node coordinates from z0 to z0 + L, finest (first) at z0, growing by q to dmax.
    Within `band` of z0 the spacing stays at most dband (the tip vortex)."""
    steps = []
    s, tot = first, 0.0
    while tot + s < L:
        if dband and tot < band and s > dband:
            s = dband
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


def _te_strip(P2, loop, n_side, h):
    """Quads across the thin trailing-edge end of the airfoil interior: the upper and
    lower wall points are repanelled at the same x, so wall points k of both sides
    face each other. Triangles there are slivers (panels much longer than the
    thickness: up to ~80 deg non-orthogonality beyond the tip, and a thin or cusped
    trailing edge diverged there); quads between the facing points are orthogonal.
    Returns (polygons, rest of the loop to triangulate; it closes across the strip's
    inner end), or None when the loop is not two n_side sides plus a base."""
    N = len(loop)
    i_le = int(np.argmin(P2[loop, 0]))
    kte = n_side - 1
    if N < 2 * n_side - 1 or n_side < 4:
        return None
    A = [loop[(i_le + k) % N] for k in range(n_side)]
    B = [loop[(i_le - k) % N] for k in range(n_side)]
    t = np.linalg.norm(P2[A] - P2[B], axis=1)
    if t[kte - 1] >= h:                                # thick enough: triangles are fine
        return None
    k_in = kte - 1
    while k_in > n_side // 2 and t[k_in - 1] < h:
        k_in -= 1
    # trailing-edge cell: pair kte-1, the two corners and the base between them
    te = [loop[(i_le + k) % N] for k in range(kte - 1, N - kte + 2)]
    polys = [te] + [[A[k - 1], A[k], B[k], B[k - 1]] for k in range(kte - 1, k_in, -1)]
    rest = [loop[(i_le + k) % N] for k in range(-k_in, k_in + 1)]
    return polys, rest


def _fill_airfoil(P2, wall_edges, h, n_side=None):
    """cells inside the airfoil loop (wall nodes kept), size h: quads across a thin
    trailing edge (_te_strip, with the wall's points per side n_side), our CDT
    triangles elsewhere. Returns (new points, cells in global node numbering)."""
    from aerolab.mesh.cdt2d import CDT
    loop = _wall_loop(wall_edges)
    strip = _te_strip(P2, loop, n_side, h) if n_side else None
    quads, loop = strip if strip else ([], loop)
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
    return np.array(newpts).reshape(-1, 2), out + quads


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
    T = _columns(cells, n_bl, M)
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


def _columns(cells, n_bl, M):
    """node ids of the boundary-layer columns, (K + 1, M): row 0 on the wall"""
    K = n_bl // M
    T = np.empty((K + 1, M), np.int64)
    T[0] = [cells[i][0] for i in range(M)]
    for k in range(K):
        T[k + 1] = [cells[k * M + i][1] for i in range(M)]
    return T


def signed_areas(P, cells):
    """signed area of every 2D cell (polygons of any size)"""
    out = np.empty(len(cells))
    by_n = {}
    for i, c in enumerate(cells):
        by_n.setdefault(len(c), []).append(i)
    for n, idx in by_n.items():
        C = np.array([cells[i] for i in idx])
        x, y = P[C, 0], P[C, 1]
        out[idx] = 0.5 * np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1)
    return out


def tip_morph(P2, cells, n_bl, M, wall_root, wall_tip, farfield_edges):
    """node displacements (n, 2) that turn the root section mesh into the tip section's.
    The two wall loops are repanelled alike (same points per side, same x spacing), so
    wall point i of the root goes to wall point i of the tip. Every boundary-layer
    column moves with its wall node and turns with the wall (by the change of the
    wall's tangent angle there), so it stays normal to the tip wall and the y+ ~ 1 rows
    keep their heights: columns moved without turning leave the first-row faces up to
    ~6 deg off the cell-centre line at the leading edge, and the run diverged there.
    The rest of the mesh, the tip-cap fill included, follows by a graph-Laplace
    interpolation that is zero on the far field."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    from scipy.spatial import cKDTree
    if len(wall_root) != len(wall_tip):
        raise ValueError("root and tip sections are not repanelled alike")
    n = len(P2)
    T = _columns(cells, n_bl, M)
    dist, j = cKDTree(wall_root).query(P2[T[0]])
    if dist.max() > 1e-8:
        raise ValueError("the section mesh's wall does not lie on the root section")
    def tangent_angle(L):
        d = np.roll(L, -1, 0) - np.roll(L, 1, 0)
        return np.arctan2(d[:, 1], d[:, 0])
    phi = np.angle(np.exp(1j * (tangent_angle(wall_tip) - tangent_angle(wall_root))))[j]
    ca, sa = np.cos(phi), np.sin(phi)
    D = np.zeros((n, 2))
    fixed = np.zeros(n, bool)
    for k in range(T.shape[0]):
        r = P2[T[k]] - wall_root[j]                        # offset from the wall node, turned by phi
        D[T[k]] = wall_tip[j] + np.c_[ca * r[:, 0] - sa * r[:, 1], sa * r[:, 0] + ca * r[:, 1]] - P2[T[k]]
        fixed[T[k]] = True
    fixed[np.unique(np.asarray(farfield_edges).ravel())] = True    # D = 0 there
    a, b = [], []
    for c in cells:
        for p, q in zip(c, c[1:] + c[:1]):
            a.append(p)
            b.append(q)
    a, b = np.array(a), np.array(b)
    A = coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    A.data[:] = 1.0                                        # edges shared by two cells count once
    L = (coo_matrix((np.asarray(A.sum(1)).ravel(), (np.arange(n), np.arange(n))), shape=(n, n)) - A).tocsr()
    free = np.where(~fixed)[0]
    if len(free):
        Lff = L[free][:, free].tocsc()
        rhs = -(L[free][:, fixed] @ D[fixed])
        D[free] = np.c_[spsolve(Lff, rhs[:, 0]), spsolve(Lff, rhs[:, 1])]
    return D


def wing_mesh(airfoil="naca0012", semi_span=3.0, taper=1.0, sweep_c4=0.0, dihedral=0.0, twist=0.0,
              tip_airfoil=None, re=3e6, level=0.5, farfield=12.0, tip_first=0.01, growth=1.3,
              blend=(0.5, None), tip_relax=2, verbose=False, wake_alphas=(0.0, 8.0), vortex_band=0.3, tip_smooth=0.5):
    """polymesh dict of a half wing + info dict (reference area, MAC, span stations).
    twist: tip incidence relative to the root (deg, nose up +, linear); sweep_c4 and
    dihedral in degrees. tip_airfoil: the tip section (None = the root's); the
    section shape changes linearly from root to tip (tip_morph)."""
    from aerolab.airfoil import airfoil_geom as ag
    from aerolab.validation import validate_naca0012_sa as v
    if tip_airfoil in ("", airfoil):
        tip_airfoil = None
    n_side = int(round(160 * level))
    wall, _ = ag.load_airfoil(airfoil, n_side=n_side)
    # the section mesh carries a wake band covering the run's angles (one mesh for all)
    m2 = v.airfoil_mesh(level, 0.0, wall=wall, re=re, farfield=farfield, wake_alphas=list(wake_alphas),
                        wake_growth=0.012)
    P2 = np.asarray(m2["points"], float)
    cells = [list(c) for c in m2["cells"]]
    nouter = len(cells)
    # tip cap closure: the airfoil interior is triangulated (cells used beyond the tip
    # only), sized like the wall panels
    we = np.asarray(m2["patches"]["airfoil"])
    h_fill = 0.8 * float(np.median(np.linalg.norm(P2[we[:, 0]] - P2[we[:, 1]], axis=1)))
    fillP, fill = _fill_airfoil(P2, we, h_fill, n_side)
    M = np.asarray(m2["layers"]).shape[1]
    tipQ = relax_map(P2, cells, m2["n_bl"], M, we)
    P2 = np.vstack([P2, fillP])
    cells += fill
    edge_patch = {(min(a, b), max(a, b)): "farfield" for a, b in m2["patches"]["farfield"]}
    D = None
    if tip_airfoil is not None:
        wall_t, _ = ag.load_airfoil(tip_airfoil, n_side=n_side)
        D = tip_morph(P2, cells, m2["n_bl"], M, wall, wall_t, m2["patches"]["farfield"])
        for Q in (P2, tipQ(1.0, P2)):                        # the tip section, plain and relaxed
            if np.any(signed_areas(Q + D, cells) * np.sign(signed_areas(Q, cells)) <= 0):
                raise ValueError(f"the tip section {tip_airfoil} is too different from the root section "
                                 f"{airfoil} to morph the mesh between them")

    # spanwise levels
    ct = taper
    dmax_in = max(semi_span / max(int(round(16 * level / 0.5)), 4), 3 * tip_first)
    # the tip vortex lives within ~0.15 tip chords of the tip: no coarse layers near it
    vb, dvb = vortex_band * ct, 0.02 / level * ct
    z_in = _spacing_to(semi_span, tip_first * ct, dmax_in, growth, vb, dvb)
    z_out = _spacing_from(semi_span, farfield, tip_first * ct, max(farfield / 8, dmax_in), growth, vb, dvb)
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
        if zl <= semi_span or not tip_smooth:
            eta = min(zl / semi_span, 1.0)
        else:
            # beyond the tip the planform change (taper, sweep, twist) eases off over
            # tip_smooth tip chords instead of stopping dead at the tip plane: a kink in
            # the shear of the thin tip-plane layers kept a mode alive there (continuity
            # stalled at ~5e-3 on tapered and swept wings)
            L = tip_smooth * ct
            eta = 1.0 + (L / semi_span) * (1.0 - math.exp(-(zl - semi_span) / L))
        c, xq, yq, th = section(eta)
        Q = tipQ(min(max(l - itip, 0) / tip_relax, 1.0), P2)
        if D is not None:
            Q = Q + min(eta, 1.0) * D                       # root -> tip section shape
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
    info = {"airfoil": airfoil, "tip_airfoil": tip_airfoil or airfoil, "semi_span": semi_span, "taper": ct, "sweep_c4": sweep_c4, "dihedral": dihedral,
            "twist": twist, "re_root": re, "level": level, "farfield": farfield, "S_half": S_half, "mac": mac,
            "aspect_ratio": (2 * semi_span) ** 2 / (2 * S_half), "z_levels": z.tolist(), "itip": itip,
            "cells_2d": len(cells), "layers": K, "x_ref": 0.25, "chord": lambda zz: 1.0 + (ct - 1.0) * np.clip(zz / semi_span, 0, 1),
            "section": section,
            # the 2D section mesh (to start the 3D run from a 2D solution) and the map
            "section_mesh": {"points": np.asarray(m2["points"], float), "cells": [list(c) for c in m2["cells"]],
                             "airfoil": np.asarray(m2["patches"]["airfoil"]),
                             "farfield": np.asarray(m2["patches"]["farfield"])},
            "cell_layer": cell_l, "cell_2d": cell_c2, "n_section_cells": nouter,
            "cells2": cells, "n2": len(P2),
            "z_cell": 0.5 * (z[cell_l] + z[cell_l + 1])}
    if verbose:
        print(f"[wing mesh] 2D {nouter} + {len(fill)} fill cells, {K} layers ({itip} to the tip), "
              f"{pm['ncells']} cells")
    return pm, info
