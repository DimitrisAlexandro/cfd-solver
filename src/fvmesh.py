"""
Finite-volume mesh: turns (points, cells, patches) into the face-based data
a cell-centred FV solver needs. Works for any polygons (triangles, quads,
mixed), so the same code runs on cavity grids and on the hybrid airfoil mesh.

Conventions
    internal face f: owner own[f], neighbour nei[f]; Sf points OUT of owner
    boundary face b: cell bcell[b], patch name bpatch[b]; bSf points OUT of domain
    |Sf| = face length (2D: area per unit depth)
"""

import numpy as np


class FVMesh:
    def __init__(self, points, cells, patches):
        P = np.asarray(points, float)
        self.points = P
        self.ncells = len(cells)
        self.nverts = np.array([len(c) for c in cells])       # 3 = triangle, 4 = quad

        # cell areas / centroids (polygon formula)
        self.vol = np.empty(self.ncells)
        self.xc = np.empty((self.ncells, 2))
        for i, c in enumerate(cells):
            x, y = P[c, 0], P[c, 1]
            xn, yn = np.roll(x, -1), np.roll(y, -1)
            cr = x * yn - xn * y
            A = 0.5 * cr.sum()
            if A <= 0:
                raise ValueError(f"cell {i} is not counter-clockwise / has zero area")
            self.vol[i] = A
            self.xc[i] = [((x + xn) * cr).sum() / (6 * A), ((y + yn) * cr).sum() / (6 * A)]

        # faces from cell edges
        edges = {}
        for ci, c in enumerate(cells):
            n = len(c)
            for k in range(n):
                a, b = c[k], c[(k + 1) % n]
                edges.setdefault((min(a, b), max(a, b)), []).append((ci, a, b))
        patch_of = {}
        for name, E in patches.items():
            for a, b in np.asarray(E):
                patch_of[(min(a, b), max(a, b))] = name

        own, nei, fa, fb = [], [], [], []
        bcell, bpatch, ba, bb = [], [], [], []
        for key, lst in edges.items():
            if len(lst) == 2:
                (c1, a, b), (c2, _, _) = lst
                own.append(c1); nei.append(c2); fa.append(a); fb.append(b)
            elif len(lst) == 1:
                c1, a, b = lst[0]
                if key not in patch_of:
                    raise ValueError(f"boundary edge {key} is not in any patch")
                bcell.append(c1); bpatch.append(patch_of[key]); ba.append(a); bb.append(b)
            else:
                raise ValueError(f"edge {key} shared by {len(lst)} cells")

        self.own, self.nei = np.array(own), np.array(nei)
        t = P[fb] - P[fa]                        # owner traverses a -> b (CCW)
        self.Sf = np.c_[t[:, 1], -t[:, 0]]       # right of a->b = out of owner
        self.xf = 0.5 * (P[fa] + P[fb])
        self.bcell = np.array(bcell)
        self.bpatch = np.array(bpatch)
        tb = P[bb] - P[ba]
        self.bSf = np.c_[tb[:, 1], -tb[:, 0]]
        self.bxf = 0.5 * (P[ba] + P[bb])
        self.patch_names = list(patches.keys())
        self.patch_faces = {n: np.where(self.bpatch == n)[0] for n in self.patch_names}

        # geometry factors
        d = self.xc[self.nei] - self.xc[self.own]
        self.d = d
        dS = np.einsum('ij,ij->i', d, self.Sf)
        S2 = np.einsum('ij,ij->i', self.Sf, self.Sf)
        self.coef = S2 / dS                           # |S|^2 / (d.S) (over-relaxed)
        self.Delta = d * self.coef[:, None]           # orthogonal part of S
        # linear interpolation weight of the OWNER
        self.w = np.einsum('ij,ij->i', self.xc[self.nei] - self.xf, self.Sf) / dS
        # skewness vector: linear interpolation gives the value where the
        # centroid line crosses the face, not at the face centre. On
        # triangles that offset is O(h) -> first-order error unless corrected.
        self.rskew = self.xf - (self.xc[self.own] + (1 - self.w)[:, None] * d)

        db = self.bxf - self.xc[self.bcell]
        self.db = db
        dbS = np.einsum('ij,ij->i', db, self.bSf)
        self.bcoef = np.einsum('ij,ij->i', self.bSf, self.bSf) / dbS
        self.bDelta = db * self.bcoef[:, None]
        nb_ = self.bSf / np.linalg.norm(self.bSf, axis=1)[:, None]
        # tangential part of (face centre - cell centre): used to extrapolate
        # "zero normal gradient" boundary values correctly on skewed cells
        self.db_t = db - np.einsum('ij,ij->i', db, nb_)[:, None] * nb_

        self._setup_lsq()

    # ------------------------------------------------------------------
    # least-squares gradient (inverse-distance weighted)
    # ------------------------------------------------------------------
    def _setup_lsq(self):
        n = self.ncells
        M = np.zeros((n, 2, 2))
        wi = 1.0 / np.einsum('ij,ij->i', self.d, self.d)
        dd = wi[:, None, None] * np.einsum('ij,ik->ijk', self.d, self.d)
        np.add.at(M, self.own, dd)
        np.add.at(M, self.nei, dd)
        wb = 1.0 / np.einsum('ij,ij->i', self.db, self.db)
        np.add.at(M, self.bcell, wb[:, None, None] * np.einsum('ij,ik->ijk', self.db, self.db))
        self.Minv = np.linalg.inv(M)
        self.lsq_wi, self.lsq_wb = wi, wb

    def grad(self, phi, phi_b):
        """Cell gradients (n,2). phi: cell values, phi_b: boundary-face values."""
        r = np.zeros((self.ncells, 2))
        diff = (phi[self.nei] - phi[self.own]) * self.lsq_wi
        c = self.d * diff[:, None]
        for k in range(2):
            r[:, k] += np.bincount(self.own, c[:, k], self.ncells)
            r[:, k] += np.bincount(self.nei, c[:, k], self.ncells)
            r[:, k] += np.bincount(self.bcell, self.db[:, k] * ((phi_b - phi[self.bcell]) * self.lsq_wb),
                                   self.ncells)
        return np.einsum('ijk,ik->ij', self.Minv, r)

    def grad_gg(self, phi, phi_b, grad_skew=None):
        """Green-Gauss gradient: (1/V) sum_f phi_f S_f. CONSERVATIVE: summed
        over cells (times V) it telescopes to the boundary integral, so a
        pressure gradient built this way gives exact discrete momentum
        balance. grad_skew (optional, lagged) adds skewness correction."""
        phif = self.interp(phi) if grad_skew is None else self.interp_skew(phi, grad_skew)
        g = np.empty((self.ncells, 2))
        for k in range(2):
            g[:, k] = self.sum_to_cells(phif * self.Sf[:, k], phi_b * self.bSf[:, k])
        return g / self.vol[:, None]

    def interp(self, phi):
        """Linear interpolation of cell values (n,) or (n,k) to internal faces."""
        w = self.w if phi.ndim == 1 else self.w[:, None]
        return w * phi[self.own] + (1 - w) * phi[self.nei]

    def interp_skew(self, phi, grad):
        """Face-centre value with skewness correction; phi (n,), grad (n,2)."""
        return self.interp(phi) + np.einsum('ij,ij->i', self.interp(grad), self.rskew)

    def sum_to_cells(self, fint, fbnd=None):
        """Sum face quantities to cells: +owner, -neighbour (divergence form)."""
        s = np.bincount(self.own, fint, self.ncells) - np.bincount(self.nei, fint, self.ncells)
        if fbnd is not None:
            s += np.bincount(self.bcell, fbnd, self.ncells)
        return s


# ----------------------------------------------------------------------
# simple mesh builders for validation cases
# ----------------------------------------------------------------------
def rect_quad(nx, ny, x0=0.0, x1=1.0, y0=0.0, y1=1.0, names=None):
    """Structured quads on a rectangle. names = patch names for
    (bottom, right, top, left)."""
    names = names or ("bottom", "right", "top", "left")
    xs, ys = np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = np.c_[X.ravel(), Y.ravel()]
    idx = lambda i, j: i * (ny + 1) + j
    cells = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
             for i in range(nx) for j in range(ny)]
    patches = {n: [] for n in names}
    for i in range(nx):
        patches[names[0]].append((idx(i, 0), idx(i + 1, 0)))
        patches[names[2]].append((idx(i + 1, ny), idx(i, ny)))
    for j in range(ny):
        patches[names[1]].append((idx(nx, j), idx(nx, j + 1)))
        patches[names[3]].append((idx(0, j + 1), idx(0, j)))
    return pts, cells, {k: np.array(v) for k, v in patches.items()}


def rect_tri(h, x0=0.0, x1=1.0, y0=0.0, y1=1.0, names=None, min_angle=28.0):
    """Unstructured triangles on a rectangle (our own CDT + refinement)."""
    from cdt2d import CDT
    names = names or ("bottom", "right", "top", "left")
    nx, ny = max(2, round((x1 - x0) / h)), max(2, round((y1 - y0) / h))
    sides = [np.c_[np.linspace(x0, x1, nx + 1)[:-1], np.full(nx, y0)],
             np.c_[np.full(ny, x1), np.linspace(y0, y1, ny + 1)[:-1]],
             np.c_[np.linspace(x1, x0, nx + 1)[:-1], np.full(nx, y1)],
             np.c_[np.full(ny, x0), np.linspace(y1, y0, ny + 1)[:-1]]]
    loop = np.vstack(sides)
    side_of = np.concatenate([[k] * len(s) for k, s in enumerate(sides)])
    cdt = CDT(x0, y0, x1, y1)
    ids = [cdt.insert(x, y) for x, y in loop]
    for i in range(len(ids)):
        cdt.insert_segment(ids[i], ids[(i + 1) % len(ids)], 'wall')
    cdt.mark_regions()
    cdt.refine(lambda x, y: h, min_angle_deg=min_angle, splittable=())
    tris = cdt.triangles(region=1)
    used = sorted({v for t in tris for v in t})
    ren = {v: i for i, v in enumerate(used)}
    pts = np.array([[cdt.x[v], cdt.y[v]] for v in used])
    cells = [[ren[v] for v in t] for t in tris]
    patches = {n: [] for n in names}
    for i in range(len(ids)):
        patches[names[side_of[i]]].append((ren[ids[i]], ren[ids[(i + 1) % len(ids)]]))
    return pts, cells, {k: np.array(v) for k, v in patches.items()}
