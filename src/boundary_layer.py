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

import math

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


def _self_intersects(Q, all_hits=False):
    """True if the closed front crosses itself (all_hits: the list of the
    crossing segment indices instead)."""
    a, b = Q, np.roll(Q, -1, axis=0)
    n = len(Q)
    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    hits = []
    for i in range(n):
        j = np.arange(i + 2, n)
        if i == 0:
            j = j[j != n - 1]
        if len(j) == 0:
            continue
        o1 = orient(a[i], b[i], a[j]); o2 = orient(a[i], b[i], b[j])
        o3 = orient(a[j], b[j], a[i]); o4 = orient(a[j], b[j], b[i])
        x = (o1 * o2 < 0) & (o3 * o4 < 0)
        if np.any(x):
            if not all_hits:
                return True
            hits += [i] + j[x].tolist()
    return sorted(set(hits)) if all_hits else False


def _redistribute(P, Q, min_ratio=0.5, half=8, passes=3, fixed=None):
    """Concave corners squeeze the front: neighbouring front nodes converge
    and finally coincide. Where a front edge has shrunk below min_ratio of
    the edge below it, slide the nodes of a +-half window along the new
    front so that their spacing follows the spacing of the layer below
    (window end points fixed)."""
    M = len(P)
    Q = Q.copy()
    lp = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    for _ in range(passes):
        lq = np.linalg.norm(np.roll(Q, -1, axis=0) - Q, axis=1)
        tight = np.where((lp > 0) & (lq < min_ratio * lp))[0]
        if len(tight) == 0:
            break
        done = set()
        for i in tight:
            if i in done:
                continue
            idx = (np.arange(i - half, i + half + 2)) % M
            if np.any(lp[idx[:-1]] == 0) or (fixed is not None and fixed[idx].any()):
                continue                     # corner fans: leave alone
            seg = Q[idx]
            sq = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
            sp = np.r_[0.0, np.cumsum(lp[idx[:-1]])]
            target = sp / sp[-1] * sq[-1]
            Q[idx[1:-1], 0] = np.interp(target[1:-1], sq, seg[:, 0])
            Q[idx[1:-1], 1] = np.interp(target[1:-1], sq, seg[:, 1])
            done.update(idx.tolist())
    return Q


def extrude(wall, h1, growth=1.15, n_layers=60, ar_cap=1.0, smooth=0.3,
            fan_threshold_deg=35.0, fan_step_deg=20.0, stop_capped=0.95,
            cap_window=10, verbose=True, max_thickness=None, repair=False):
    """wall: (M,2) CCW closed loop.
    n_layers is a MAXIMUM: growth stops once `stop_capped` of the nodes have
    reached square outer cells (h >= ar_cap * spacing) - more layers than that
    only adds square cells the triangles could have covered. So the BL depth
    adapts to Re: high Re (tiny h1) automatically gets more layers.
    max_thickness: optional callable (points (N,2) -> (N,) thickness limit)
    - e.g. a fraction of the distance to another element of a multi-element
    airfoil. Where the free extrusion would be thicker, the layers there are
    regrown with a lower local growth ratio (same number of layers, same
    first height where possible), so the layer stays structured and the
    cells keep a smooth stretching instead of piling up flat cells.
    repair: when a layer has invalid cells (concave corners, coves), lower
    the layer height locally (halved around the bad cells, recovering by
    25 % per layer) and retry, instead of stopping the whole layer there.
    Returns (layers (K+1, M', 2), canon (M',)). Layer 0 contains the fan
    duplicates; canon maps them to one wall node."""
    if max_thickness is None:
        return _extrude(wall, h1, growth, n_layers, ar_cap, smooth, fan_threshold_deg,
                        fan_step_deg, stop_capped, cap_window, verbose, repair)
    # limited layer: the number of layers is the one that reaches the
    # largest allowed thickness at the nominal growth; every node gets its
    # own growth to end at its own (smoothed) limit
    P0 = add_corner_fans(wall, fan_threshold_deg, fan_step_deg)[0]
    Lim = np.asarray(max_thickness(P0), float)
    K = int(min(n_layers, math.ceil(math.log(1 + Lim.max() * (growth - 1) / h1) /
                                    math.log(growth))))
    T = np.full(len(P0), h1 * (growth**K - 1) / (growth - 1))
    # target thickness: the free thickness capped by the limit, then its
    # peaks eroded along the front (min with the neighbour average, never
    # raised): an isolated thick node (a fan at a corner, a node just past
    # a cove) would otherwise stick out of the layer edge as a spike
    target = np.minimum(T, Lim)
    for _ in range(30):
        target = np.minimum(target, 0.5 * target + 0.25 * (np.roll(target, 1) +
                                                            np.roll(target, -1)))
    lim = target < T * (1 - 1e-6)
    def growth_for(target, lim):
        """per-node growth g (and if needed a smaller first height) so that
        the K layers end at the target: h1 (g^K - 1)/(g - 1) = target"""
        g = np.full(len(T), growth, float)
        h1n = np.full(len(T), float(h1))
        for i in np.where(lim)[0]:
            if K * h1 >= target[i]:
                g[i], h1n[i] = 1.0, target[i] / K
                continue
            lo, hi = 1.0 + 1e-9, growth
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if h1 * (mid**K - 1) / (mid - 1) > target[i]:
                    hi = mid
                else:
                    lo = mid
            g[i] = lo
        return g, h1n

    best = None
    for attempt in range(12):
        g, h1n = growth_for(target, lim)
        layers2, canon2, bad = _extrude(wall, h1n, g, K, ar_cap, smooth, fan_threshold_deg,
                                        fan_step_deg, 2.0, cap_window, False, repair,
                                        return_bad=True)
        if best is None or len(layers2) > len(best[0]):
            best = (layers2, canon2)
        if not bad:
            break
        # the march got stuck (typically next to a corner fan whose
        # neighbour is much thinner): thin the layer around the trouble
        # spot and regrow
        near = np.unique((np.array(bad)[:, None] + np.arange(-8, 10)[None]) % len(T))
        target[near] *= 0.5
        for _ in range(10):
            target = np.minimum(target, 0.5 * target + 0.25 * (np.roll(target, 1) +
                                                                np.roll(target, -1)))
        lim = target < T * (1 - 1e-6)
    if verbose:
        print(f"  boundary layer: {lim.sum()} of {len(T)} nodes thickness-limited "
              f"(growth down to {g.min():.3f}, thickness >= {target[lim].min():.2e}), "
              f"{len(best[0]) - 1} of {K} layers"
              + (f" after {attempt} local thinnings" if attempt else ""))
    return best


def _extrude(wall, h1, growth, n_layers, ar_cap, smooth, fan_threshold_deg, fan_step_deg,
             stop_capped, cap_window, verbose, repair=False, return_bad=False, damp_tries=1):
    """The layer march. h1, growth: scalars or per-node arrays (fan points
    included, i.e. matching layer 0)."""
    P, n0, s0, ds0, canon, fan_ds = add_corner_fans(wall, fan_threshold_deg, fan_step_deg)
    collapsed0 = canon == canon[(np.arange(len(P)) + 1) % len(P)]
    layers = [P]
    damp = np.ones(len(P))
    n_repaired = 0
    bad = []
    # fan nodes at convex corners and two nodes either side: their front
    # spacing changes by design (the fan opens), not by a squeeze
    fan_zone = np.zeros(len(P), bool)
    for sh in range(-2, 3):
        fan_zone |= np.roll(fan_ds > 0, sh)
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
        h_geom = np.broadcast_to(h1 * np.power(growth, k), (len(P),))
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
        coll = collapsed0 if k == 0 else np.zeros(len(P), bool)
        if repair:
            damp = np.minimum(1.0, damp * 1.25)
        # local height damping retries are kept for experiments (damp_tries);
        # by default a failed layer ends the march and extrude() regrows
        # with a smoothly thinned target there instead (damping leaves
        # notches in the layer edge)
        for attempt in range(damp_tries if repair else 1):
            Q = P + (h * damp * scale)[:, None] * n
            if repair and k > 0:
                Q = _redistribute(P, Q, fixed=fan_zone)
            ok = _cells_ok(P, Q, coll)
            bad = np.where(~ok)[0].tolist()
            if ok.all():
                hits = _self_intersects(Q, all_hits=repair)
                bad = hits if repair else ([0] if hits else [])
            if not bad or not repair:
                break
            near = np.unique((np.array(bad)[:, None] + np.arange(-3, 5)[None]) % len(P))
            damp[near] *= 0.5
            n_repaired += 1
        if bad:
            if verbose:
                print(f"  boundary layer: stopped at {k} layers "
                      f"({len(bad)} invalid cells in layer {k + 1})")
            break
        layers.append(Q)
        P = Q
    if verbose and n_repaired:
        print(f"  boundary layer: {n_repaired} local height repairs, "
              f"{len(layers) - 1} layers")
    if return_bad:
        return np.array(layers), canon, (bad if len(layers) - 1 < n_layers else [])
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
