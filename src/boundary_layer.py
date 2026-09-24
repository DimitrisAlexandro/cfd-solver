"""
Advancing-layer extrusion of quad (prism-in-3D) boundary-layer cells.

Starting from the wall loop, each new layer is marched along vertex
normals that are RECOMPUTED from the current front and smoothed more
strongly the further we are from the wall (wall-orthogonal near the
surface, smooth further out).

Convex corners (the blunt trailing edge) get a NORMAL FAN: the corner node
is duplicated so that several normals leave it, spaced at most `fan_step`
degrees apart. The first layer there is a fan of triangles (collapsed
quads); from layer 2 on everything is ordinary quads. Without a fan, one
averaged normal has to cover a ~90 deg turn, the front stretches into a few
long edges and the cells around the TE become huge kites.

Layer height at node i for layer k:
    h = min(h1 * growth**k,  ar_cap * local tangential spacing)
The cap makes the outermost cells roughly square, so the triangles that
start at the BL edge have a matching size instead of a 20:1 jump.

Every layer is validated (all cells convex and counter-clockwise, front
not self-intersecting). If a layer fails, growth stops there and the
valid layers are kept.
"""

import numpy as np


def _edge_normals(P):
    e = np.roll(P, -1, axis=0) - P                       # edge i: P_i -> P_i+1
    le = np.linalg.norm(e, axis=1)
    return np.c_[e[:, 1], -e[:, 0]] / le[:, None], le  # outward (right of tangent)


def _vertex_normals(P):
    en, le = _edge_normals(P)
    n = en + np.roll(en, 1, axis=0)                      # edges i-1 and i
    n /= np.linalg.norm(n, axis=1)[:, None]
    cos_half = np.clip(np.einsum('ij,ij->i', n, en), 0.5, 1.0)
    ds = 0.5 * (le + np.roll(le, 1))
    return n, 1.0 / cos_half, ds


def add_corner_fans(wall, threshold_deg=35.0, step_deg=20.0):
    """Duplicate convex corner nodes of a CCW loop into a fan.
    Returns (points, normals, scale, ds, canon, fan_ds):
      canon[i]  index of the first copy of node i (equal for fan copies)
      fan_ds    corner wall spacing for fan copies, 0 elsewhere."""
    P = np.asarray(wall, float)
    n_v, scale_v, ds_v = _vertex_normals(P)
    en, _ = _edge_normals(P)
    n_in, n_out = np.roll(en, 1, axis=0), en
    pts, nrm, scl, ds, canon, fan_ds = [], [], [], [], [], []
    for i in range(len(P)):
        turn = np.arctan2(n_in[i, 0] * n_out[i, 1] - n_in[i, 1] * n_out[i, 0],
                          n_in[i] @ n_out[i])            # > 0: convex corner
        if turn > np.radians(threshold_deg):
            k = int(np.ceil(turn / np.radians(step_deg)))
            a0 = np.arctan2(n_in[i, 1], n_in[i, 0])
            first = len(pts)
            for j in range(k + 1):
                a = a0 + turn * j / k
                pts.append(P[i]); nrm.append([np.cos(a), np.sin(a)])
                scl.append(1.0); ds.append(ds_v[i]); canon.append(first)
                fan_ds.append(ds_v[i])
        else:
            canon.append(len(pts))
            pts.append(P[i]); nrm.append(n_v[i]); scl.append(scale_v[i])
            ds.append(ds_v[i]); fan_ds.append(0.0)
    return (np.array(pts), np.array(nrm), np.array(scl), np.array(ds),
            np.array(canon), np.array(fan_ds))


def _cells_ok(P, Q, collapsed):
    """Cells [P_i, Q_i, Q_i+1, P_i+1] strictly convex and CCW; where the wall
    edge is collapsed (fan), check the triangle [P_i, Q_i, Q_i+1]."""
    A, B, C, D = P, Q, np.roll(Q, -1, axis=0), np.roll(P, -1, axis=0)
    def cross(o, a, b):
        return (a[:, 0] - o[:, 0]) * (b[:, 1] - o[:, 1]) - (a[:, 1] - o[:, 1]) * (b[:, 0] - o[:, 0])
    quad = (cross(A, B, C) > 0) & (cross(B, C, D) > 0) & (cross(C, D, A) > 0) & (cross(D, A, B) > 0)
    tri = cross(A, B, C) > 0
    return np.where(collapsed, tri, quad)


def _self_intersects(Q):
    a, b = Q, np.roll(Q, -1, axis=0)
    n = len(Q)
    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    for i in range(n):
        j = np.arange(i + 2, n)
        if i == 0:
            j = j[j != n - 1]
        if len(j) == 0:
            continue
        o1 = orient(a[i], b[i], a[j]); o2 = orient(a[i], b[i], b[j])
        o3 = orient(a[j], b[j], a[i]); o4 = orient(a[j], b[j], b[i])
        if np.any((o1 * o2 < 0) & (o3 * o4 < 0)):
            return True
    return False


def extrude(wall, h1, growth=1.15, n_layers=60, ar_cap=1.0, smooth=0.3,
            fan_threshold_deg=35.0, fan_step_deg=20.0, stop_capped=0.95,
            cap_window=10, verbose=True):
    """wall: (M,2) CCW closed loop.
    n_layers is a MAXIMUM: growth stops once `stop_capped` of the nodes have
    reached square outer cells (h >= ar_cap * spacing) - more layers than that
    only adds square cells the triangles could have covered. So the BL depth
    adapts to Re: high Re (tiny h1) automatically gets more layers.
    Returns (layers (K+1, M', 2), canon (M',)). Layer 0 contains the fan
    duplicates; canon maps them to one wall node."""
    P, n0, s0, ds0, canon, fan_ds = add_corner_fans(wall, fan_threshold_deg, fan_step_deg)
    collapsed0 = canon == canon[(np.arange(len(P)) + 1) % len(P)]
    layers = [P]
    for k in range(n_layers):
        if k == 0:
            n, scale, ds = n0, s0, ds0
        else:
            n, scale, ds = _vertex_normals(P)
            ds = np.maximum(ds, fan_ds)   # young fan nodes are close together
        w = smooth * min(1.0, k / 8.0)   # no smoothing at the wall
        for _ in range(3):
            if w == 0:
                break
            n = n + w * (np.roll(n, 1, axis=0) + np.roll(n, -1, axis=0))
            n /= np.linalg.norm(n, axis=1)[:, None]
        h_geom = h1 * growth ** k
        # cap on spacing averaged over +-cap_window nodes: a single squeezed
        # node would otherwise cap itself, lag, get squeezed more (notch)
        if cap_window > 0:
            w = 2 * cap_window + 1
            ds = np.convolve(np.r_[ds[-cap_window:], ds, ds[:cap_window]],
                             np.ones(w) / w, mode='valid')
        h = np.minimum(h_geom, ar_cap * ds)
        # smooth capped heights along the front: an isolated low node would
        # otherwise lag behind and carve a notch into the BL edge
        for _ in range(5):
            h = np.minimum(h_geom, 0.5 * h + 0.25 * (np.roll(h, 1) + np.roll(h, -1)))
        if k > 0 and np.mean(h_geom >= ar_cap * ds) >= stop_capped:
            break
        Q = P + (h * scale)[:, None] * n
        ok = _cells_ok(P, Q, collapsed0 if k == 0 else np.zeros(len(P), bool))
        if not ok.all() or _self_intersects(Q):
            if verbose:
                print(f"  boundary layer: stopped at {k} layers "
                      f"({(~ok).sum()} invalid cells in layer {k + 1})")
            break
        layers.append(Q)
        P = Q
    return np.array(layers), canon


def cells(n_layers_done, M, canon):
    """Cell connectivity (CCW), global point index k*M + i.
    First-layer cells at fans are triangles."""
    out = []
    for k in range(n_layers_done):
        for i in range(M):
            j = (i + 1) % M
            a = canon[i] if k == 0 else k * M + i
            b = canon[j] if k == 0 else k * M + j
            if a == b:
                out.append([a, (k + 1) * M + i, (k + 1) * M + j])
            else:
                out.append([a, (k + 1) * M + i, (k + 1) * M + j, b])
    return out
