"""3D mesh generators -> polymesh dicts for FVMesh3D.

    from_hexes(points, hexes, patch_fn)        any hexahedral mesh
    extrude(P2, cells2, edge_patch, ...)       2D mesh (tri/quad) swept through levels:
                                               prisms/hexes, optional per-layer active
                                               cells (used by the wing tip cap)
    box(nx, ny, nz, ...)                       structured box (cavity, duct)
    duct(...)                                  rectangular duct, inlet/outlet
    sphere(n, nr, r0, r1, ...)                 cubed-sphere O-grid around a sphere
"""
import numpy as np

HEX_FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])


def from_hexes(points, hexes, patch_fn):
    """points (N,3), hexes (nc,8) VTK ordering; patch_fn(face_centres (m,3)) -> names (m,)"""
    hexes = np.asarray(hexes, np.int64)
    nc = len(hexes)
    F = hexes[:, HEX_FACES].reshape(-1, 4)                  # (6 nc, 4)
    cell = np.repeat(np.arange(nc), 6)
    key = np.sort(F, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    if (cnt > 2).any():
        raise ValueError("a face is shared by more than two cells")
    order = np.argsort(inv, kind="stable")
    inv_s = inv[order]
    first = np.r_[True, inv_s[1:] != inv_s[:-1]]
    fidx = order[first]                                     # one face per unique key
    owner = cell[fidx]
    nei = np.full(len(fidx), -1)
    second = order[~first]
    nei_pos = np.searchsorted(inv_s[first], inv_s[~first])
    nei[nei_pos] = cell[second]
    faces = F[fidx]
    patch = np.full(len(faces), "", dtype=object)
    b = nei < 0
    xf = np.asarray(points)[faces[b]].mean(1)
    patch[b] = patch_fn(xf)
    return {"points": np.asarray(points, float), "faces": faces, "owner": owner, "neighbour": nei,
            "patch": patch, "ncells": nc}


def extrude(P2, cells2, edge_patch, nlev, xyz=None, active=None, cap_lower="zmin", cap_upper="zmax",
            side_inactive="wall", cap_inactive="wall"):
    """Sweep a 2D mesh through nlev node levels (nlev-1 layers of cells).
    P2 (n2,2); cells2: list of polygons; edge_patch: {(min,max): name} of the 2D boundary
    edges; xyz(l) -> (n2,3) coordinates of level l (default (x, y, l));
    active (nlev-1, nc2) bool: which 2D cells exist in each layer. Faces between an
    active and an inactive cell get side_inactive (lateral) / cap_inactive (caps)."""
    P2 = np.asarray(P2, float)
    n2, nc2 = len(P2), len(cells2)
    K = nlev - 1
    W = max(len(c) for c in cells2)
    C2 = np.full((nc2, W), -1, np.int64)
    for i, c in enumerate(cells2):
        C2[i, :len(c)] = c
    act = np.ones((K, nc2), bool) if active is None else np.asarray(active, bool)
    cid = np.full((K, nc2), -1, np.int64)
    cid[act] = np.arange(act.sum())
    ncells = int(act.sum())
    # 2D cell orientation (+1 counter-clockwise): faces are written outward from
    # their owner, so the 3D mesh needs no geometric orientation test (which slivers
    # - a collapsed wing tip - can fool)
    xs, ys = P2[:, 0], P2[:, 1]
    sgn = np.empty(nc2)
    for ci, c in enumerate(cells2):
        cc = np.asarray(c)
        sgn[ci] = 1.0 if np.sum(xs[cc] * ys[np.roll(cc, -1)] - xs[np.roll(cc, -1)] * ys[cc]) > 0 else -1.0
    # 2D edges
    emap, edir = {}, {}
    for ci, c in enumerate(cells2):
        m = len(c)
        for k in range(m):
            a, b = c[k], c[(k + 1) % m]
            emap.setdefault((min(a, b), max(a, b)), []).append(ci)
            edir[(min(a, b), max(a, b), ci)] = a < b              # listed as min -> max
    ea = np.array([k[0] for k in emap], np.int64)
    eb = np.array([k[1] for k in emap], np.int64)
    ec1 = np.array([v[0] for v in emap.values()], np.int64)
    ec2 = np.array([v[1] if len(v) == 2 else -1 for v in emap.values()], np.int64)
    if any(len(v) > 2 for v in emap.values()):
        raise ValueError("2D edge shared by more than two cells")
    epatch = np.array([edge_patch.get(k, "") for k in emap], dtype=object)
    # does the quad [ea, eb, eb', ea'] point out of cell ec1 / ec2 ?
    out1 = np.array([edir[(k[0], k[1], v[0])] == (sgn[v[0]] > 0) for k, v in emap.items()])
    out2 = np.array([len(v) == 2 and edir[(k[0], k[1], v[1])] == (sgn[v[1]] > 0) for k, v in emap.items()])
    if ((ec2 < 0) & (epatch == "")).any():
        raise ValueError("2D boundary edge without a patch")
    faces, own, nei, pat = [], [], [], []
    for l in range(K):
        c1 = cid[l][ec1]
        c2 = np.where(ec2 >= 0, cid[l][np.maximum(ec2, 0)], -1)
        nodes = np.c_[ea + l * n2, eb + l * n2, eb + (l + 1) * n2, ea + (l + 1) * n2]
        rev = nodes[:, ::-1]
        both = (c1 >= 0) & (c2 >= 0)
        one = (c1 >= 0) ^ (c2 >= 0)
        faces.append(np.where(out1[:, None], nodes, rev)[both]); own.append(c1[both]); nei.append(c2[both])
        pat.append(np.full(both.sum(), "", dtype=object))
        o = np.where(c1 >= 0, c1, c2)
        outo = np.where(c1 >= 0, out1, out2)
        faces.append(np.where(outo[:, None], nodes, rev)[one]); own.append(o[one]); nei.append(np.full(one.sum(), -1))
        pat.append(np.where(ec2[one] < 0, epatch[one], side_inactive))
    for l in range(nlev):
        below = cid[l - 1] if l > 0 else np.full(nc2, -1)
        above = cid[l] if l < K else np.full(nc2, -1)
        nodes = np.where(C2 >= 0, C2 + l * n2, -1)
        # reversed polygons (padding kept at the end)
        cnt = (C2 >= 0).sum(1)
        ar = np.arange(W)[None, :]
        ridx = np.where(ar < cnt[:, None], cnt[:, None] - 1 - ar, ar)
        rnodes = np.take_along_axis(nodes, ridx, axis=1)
        up = sgn > 0                                          # CCW polygon: normal +z
        both = (below >= 0) & (above >= 0)
        one = (below >= 0) ^ (above >= 0)
        faces.append(np.where(up[:, None], nodes, rnodes)[both]); own.append(below[both]); nei.append(above[both])
        pat.append(np.full(both.sum(), "", dtype=object))
        o = np.where(below >= 0, below, above)
        outz = np.where(below >= 0, up, ~up)
        name = cap_lower if l == 0 else cap_upper if l == K else cap_inactive
        faces.append(np.where(outz[:, None], nodes, rnodes)[one]); own.append(o[one]); nei.append(np.full(one.sum(), -1))
        pat.append(np.full(one.sum(), name, dtype=object))
    pts = np.vstack([xyz(l) if xyz else np.c_[P2, np.full(n2, float(l))] for l in range(nlev)])
    # cell centres: centroid of the mid-layer section (the polygon through the mid
    # points of the cell's spanwise edges). For cells that taper between levels (a collapsing wing tip) the
    # volume centroid drifts toward the larger end; tiny neighbours tapering at
    # different rates then get centres at different heights and ~90 deg non-orthogonal
    # faces. The mid-section point keeps a layer's centres level.
    cnt = (C2 >= 0).sum(1)
    Ci = np.where(C2 >= 0, C2, C2[:, :1])
    nxt = np.take_along_axis(Ci, (np.arange(W)[None, :] + 1) % cnt[:, None], axis=1)
    valid = np.arange(W)[None, :] < cnt[:, None]

    def centroids(Pl):
        a, b = Pl[Ci], Pl[nxt]                              # (nc2, W, 3)
        o = a[:, :1]
        cr = np.cross(a - o, b - o) * valid[..., None]
        ar = np.linalg.norm(cr.sum(1), axis=1)
        nrm = cr.sum(1) / np.maximum(ar, 1e-300)[:, None]
        wt = np.einsum("ckj,cj->ck", cr, nrm)
        g = (o + a + b) / 3.0
        cen = (wt[..., None] * g).sum(1) / np.maximum(wt.sum(1), 1e-300)[:, None]
        mean = (Pl[Ci] * valid[..., None]).sum(1) / cnt[:, None]
        return np.where((ar > 0)[:, None], cen, mean)
    centres = np.zeros((ncells, 3))
    for l in range(K):
        mid = 0.5 * (pts[l * n2:(l + 1) * n2] + pts[(l + 1) * n2:(l + 2) * n2])
        centres[cid[l][act[l]]] = centroids(mid)[act[l]]
    Wf = max(f.shape[1] for f in faces)
    faces = [np.pad(f, ((0, 0), (0, Wf - f.shape[1])), constant_values=-1) for f in faces]
    return {"points": pts, "faces": np.vstack(faces), "owner": np.concatenate(own),
            "neighbour": np.concatenate(nei), "patch": np.concatenate(pat), "ncells": ncells,
            "oriented": True, "centres": centres}


def agglomerate(pm, group):
    """merge cells: group[c] = new cell index of old cell c (any integers; renumbered
    densely). Faces inside a group are dropped; several faces left between the same
    two new cells are merged by FVMesh3D (summed area vectors)."""
    g = np.asarray(group, np.int64)
    _, new = np.unique(g, return_inverse=True)
    own = new[np.asarray(pm["owner"], np.int64)]
    nb = np.asarray(pm["neighbour"], np.int64)
    nei = np.where(nb >= 0, new[np.maximum(nb, 0)], -1)
    keep = own != nei
    return {"points": pm["points"], "faces": np.asarray(pm["faces"])[keep], "owner": own[keep],
            "neighbour": nei[keep], "patch": np.asarray(pm["patch"], dtype=object)[keep],
            "ncells": int(new.max()) + 1, "oriented": pm.get("oriented", False)}


def box(nx, ny, nz, lx=1.0, ly=1.0, lz=1.0, names=None, grading=None):
    """structured hexahedra on [0,lx]x[0,ly]x[0,lz]; names: dict side -> patch for
    sides xmin, xmax, ymin, ymax, zmin, zmax (default: the side names);
    grading: optional (fx, fy, fz) node-coordinate maps on [0, 1]"""
    names = {s: s for s in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")} | dict(names or {})
    g = grading or (None, None, None)
    ax = [np.linspace(0, 1, n + 1) if f is None else f(np.linspace(0, 1, n + 1)) for n, f in zip((nx, ny, nz), g)]
    xs, ys, zs = ax[0] * lx, ax[1] * ly, ax[2] * lz
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    idx = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
    I, J, Kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    I, J, Kk = I.ravel(), J.ravel(), Kk.ravel()
    hexes = np.c_[idx(I, J, Kk), idx(I + 1, J, Kk), idx(I + 1, J + 1, Kk), idx(I, J + 1, Kk),
                  idx(I, J, Kk + 1), idx(I + 1, J, Kk + 1), idx(I + 1, J + 1, Kk + 1), idx(I, J + 1, Kk + 1)]
    tol = 1e-9 * max(lx, ly, lz)

    def patch_fn(xf):
        out = np.empty(len(xf), dtype=object)
        for s, (k, v) in {"xmin": (0, 0.0), "xmax": (0, lx), "ymin": (1, 0.0), "ymax": (1, ly),
                          "zmin": (2, 0.0), "zmax": (2, lz)}.items():
            out[np.abs(xf[:, k] - v) < tol] = names[s]
        return out
    return from_hexes(pts, hexes, patch_fn)


def tanh_cluster(beta):
    """node map on [0,1] clustered at both ends (beta ~ 2-3)"""
    return lambda s: 0.5 * (1 + np.tanh(beta * (2 * s - 1)) / np.tanh(beta))


def duct(nx, ny, nz, length, width=1.0, height=1.0, beta=1.5):
    """rectangular duct along x: inlet (x=0), outlet (x=length), walls elsewhere"""
    names = {"xmin": "inlet", "xmax": "outlet", "ymin": "wall", "ymax": "wall", "zmin": "wall", "zmax": "wall"}
    return box(nx, ny, nz, length, width, height, names,
               grading=(None, tanh_cluster(beta), tanh_cluster(beta)))


def sphere(n=16, nr=40, r0=0.5, r1=15.0, first=None):
    """cubed-sphere O-grid: 6 x n x n cells per shell, nr shells from r0 to r1 with
    geometric growth (first layer thickness `first`, default from a uniform log map).
    Patches: sphere, farfield."""
    t = np.tan(np.linspace(-np.pi / 4, np.pi / 4, n + 1))   # equi-angular cube coordinates
    A, B = np.meshgrid(t, t, indexing="ij")
    one = np.ones_like(A)
    faces = [np.stack([one, A, B], -1), np.stack([-one, B, A], -1), np.stack([B, one, A], -1),
             np.stack([A, -one, B], -1), np.stack([A, B, one], -1), np.stack([B, A, -one], -1)]
    dirs = np.concatenate([f.reshape(-1, 3) for f in faces])
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    key = np.round(dirs * 1e9).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    ndir = len(uniq)
    D = np.zeros((ndir, 3))
    D[inv] = dirs
    if first is None:
        r = r0 * (r1 / r0) ** np.linspace(0, 1, nr + 1)
    else:                                                   # geometric with given first thickness
        from scipy.optimize import brentq
        q = brentq(lambda q: first * (q ** nr - 1) / (q - 1) - (r1 - r0), 1.0000001, 3.0)
        r = r0 + np.r_[0.0, np.cumsum(first * q ** np.arange(nr))]
    pts = (r[:, None, None] * D[None]).reshape(-1, 3)
    hexes = []
    m = n + 1
    for f in range(6):
        g = inv[f * m * m:(f + 1) * m * m].reshape(m, m)
        q = np.c_[g[:-1, :-1].ravel(), g[1:, :-1].ravel(), g[1:, 1:].ravel(), g[:-1, 1:].ravel()]
        for k in range(nr):
            hexes.append(np.c_[q + k * ndir, q + (k + 1) * ndir])
    hexes = np.vstack(hexes)

    def patch_fn(xf):
        rr = np.linalg.norm(xf, axis=1)
        return np.where(rr < 0.5 * (r0 + r[1]), "sphere", "farfield").astype(object)
    return from_hexes(pts, hexes, patch_fn)
