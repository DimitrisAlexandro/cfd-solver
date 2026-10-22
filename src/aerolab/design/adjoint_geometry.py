"""Mesh geometry for the discrete adjoint (adjoint.py): complex-safe elementwise helpers
(branches follow the real part) and the geometry recomputed from moved mesh points
(areas, centroids, face vectors, interpolation weights, least-squares gradient operators),
written so that complex-step derivatives pass through it."""
import math

import numpy as np
import scipy.sparse as sp

COS80 = math.cos(math.radians(80.0))


# ----------------------------------------------------------------------
# complex-safe elementwise helpers: branches follow the real part
# ----------------------------------------------------------------------
def cmax(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.where(np.real(a) >= np.real(b), a, b)


def cmin(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.where(np.real(a) <= np.real(b), a, b)


def cpos(x):
    return np.where(np.real(x) > 0, x, 0.0)


def cneg(x):
    return np.where(np.real(x) < 0, x, 0.0)


def cabs(x):
    return np.where(np.real(x) < 0, -x, x)


def cclip(x, lo, hi):
    r = np.real(x)
    return np.where(r < lo, lo, np.where(r > hi, hi, x))


def _dot(a, b):
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]


# ----------------------------------------------------------------------
# geometry from points (fixed topology), complex-safe
# ----------------------------------------------------------------------
class Topology:
    """Connectivity of an FVMesh (does not change when the points move) and
    the topological scatter matrices (face -> cell sums)."""

    def __init__(self, fvm):
        self.n = fvm.ncells
        self.own, self.nei, self.bcell = fvm.own, fvm.nei, fvm.bcell
        self.fa, self.fb, self.ba, self.bb = fvm.fa, fvm.fb, fvm.ba, fvm.bb
        self.nf, self.nb = len(self.own), len(self.bcell)
        self.groups = {}
        for k in sorted(set(len(c) for c in fvm.cells)):
            idx = np.array([i for i, c in enumerate(fvm.cells) if len(c) == k])
            self.groups[k] = (idx, np.array([fvm.cells[i] for i in idx]))
        n, nf, nb = self.n, self.nf, self.nb
        fi = np.arange(nf)
        self.Sown = sp.csr_matrix((np.ones(nf), (self.own, fi)), (n, nf))
        self.Snei = sp.csr_matrix((np.ones(nf), (self.nei, fi)), (n, nf))
        self.Sdiv = (self.Sown - self.Snei).tocsr()
        self.Sb = sp.csr_matrix((np.ones(nb), (self.bcell, np.arange(nb))), (n, nb))
        self.patch_faces = fvm.patch_faces
        self.npoints = len(fvm.points)


class Geometry:
    """Everything FVMesh computes from the points, vectorised and valid for
    complex points (complex-step shape derivatives). Identical formulas to
    fvmesh.FVMesh (checked in the tests to round-off)."""

    def __init__(self, topo, P, wall_faces, wall_dist_index=None):
        self.t = t = topo
        P = np.asarray(P)
        self.points = P
        n = t.n
        dt = np.result_type(P.dtype, float)
        vol = np.zeros(n, dt)
        xc = np.zeros((n, 2), dt)
        for k, (idx, C) in t.groups.items():
            x, y = P[C, 0], P[C, 1]
            xn, yn = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
            cr = x * yn - xn * y
            A = 0.5 * cr.sum(1)
            vol[idx] = A
            xc[idx, 0] = ((x + xn) * cr).sum(1) / (6 * A)
            xc[idx, 1] = ((y + yn) * cr).sum(1) / (6 * A)
        self.vol, self.xc = vol, xc
        own, nei, bc = t.own, t.nei, t.bcell
        tt = P[t.fb] - P[t.fa]
        self.Sf = np.c_[tt[:, 1], -tt[:, 0]]
        self.xf = 0.5 * (P[t.fa] + P[t.fb])
        tb = P[t.bb] - P[t.ba]
        self.bSf = np.c_[tb[:, 1], -tb[:, 0]]
        self.bxf = 0.5 * (P[t.ba] + P[t.bb])
        d = xc[nei] - xc[own]
        self.d = d
        dS = _dot(d, self.Sf)
        S2 = _dot(self.Sf, self.Sf)
        dS = cmax(dS, COS80 * np.sqrt(_dot(d, d) * S2))
        self.coef = S2 / dS
        self.Delta = d * self.coef[:, None]
        self.w = cclip(_dot(xc[nei] - self.xf, self.Sf) / dS, 0.0, 1.0)
        self.rskew = self.xf - (xc[own] + (1 - self.w)[:, None] * d)
        db = self.bxf - xc[bc]
        self.db = db
        dbS = _dot(db, self.bSf)
        self.bcoef = _dot(self.bSf, self.bSf) / dbS
        self.bDelta = db * self.bcoef[:, None]
        nb_ = self.bSf / np.sqrt(_dot(self.bSf, self.bSf))[:, None]
        self.db_t = db - _dot(db, nb_)[:, None] * nb_
        # least-squares gradient: grad = Minv r, r_k = sum c_k (phi_N - phi_P)
        wi = 1.0 / _dot(d, d)
        wb = 1.0 / _dot(db, db)
        M = np.zeros((n, 2, 2), dt)
        for a in range(2):
            for b in range(2):
                q = wi * d[:, a] * d[:, b]
                M[:, a, b] = t.Sown @ q + t.Snei @ q + t.Sb @ (wb * db[:, a] * db[:, b])
        det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
        Mi = np.empty_like(M)
        Mi[:, 0, 0], Mi[:, 1, 1] = M[:, 1, 1] / det, M[:, 0, 0] / det
        Mi[:, 0, 1], Mi[:, 1, 0] = -M[:, 0, 1] / det, -M[:, 1, 0] / det
        self.Minv = Mi
        self.cw = d * wi[:, None]               # (nf, 2)
        self.cwb = db * wb[:, None]             # (nb, 2)
        # wall distance: exact point-to-segment distance to the nearest wall
        # face (the face chosen on the real geometry, then frozen)
        self.wall_faces = wall_faces
        if wall_dist_index is None:
            from aerolab.core.turbulence import wall_distance
            Preal = np.real(P)
            if np.iscomplexobj(P):
                raise ValueError("pass wall_dist_index for complex points")
            _, j = wall_distance(_RealMesh(Preal, topo, xc.real), wall_faces, return_index=True)
            wall_dist_index = j
        self.wall_dist_index = wall_dist_index
        fw = wall_faces[wall_dist_index]
        a_, b_ = P[t.ba[fw]], P[t.bb[fw]]
        AB = b_ - a_
        tpar = _dot(xc - a_, AB) / _dot(AB, AB)
        tpar = cclip(tpar, 0.0, 1.0)
        q = xc - (a_ + tpar[:, None] * AB)
        self.wall_d = cmax(np.sqrt(_dot(q, q)), 1e-30)

    # operators -----------------------------------------------------------
    def interp(self, phi):
        w = self.w if phi.ndim == 1 else self.w[:, None]
        return w * phi[self.t.own] + (1 - w) * phi[self.t.nei]

    def interp_skew(self, phi, grad):
        return self.interp(phi) + _dot(self.interp(grad), self.rskew)

    def sum_to_cells(self, fint, fbnd=None):
        s = self.t.Sdiv @ fint
        if fbnd is not None:
            s = s + self.t.Sb @ fbnd
        return s

    def grad(self, phi, phib):
        t = self.t
        dphi = phi[t.nei] - phi[t.own]
        dphib = phib - phi[t.bcell]
        r = np.empty((t.n, 2), np.result_type(phi.dtype, self.cw.dtype))
        for k in range(2):
            q = self.cw[:, k] * dphi
            r[:, k] = t.Sown @ q + t.Snei @ q + t.Sb @ (self.cwb[:, k] * dphib)
        Mi = self.Minv
        return np.c_[Mi[:, 0, 0] * r[:, 0] + Mi[:, 0, 1] * r[:, 1],
                     Mi[:, 1, 0] * r[:, 0] + Mi[:, 1, 1] * r[:, 1]]

    def grad_gg(self, phi, phib, grad_skew=None):
        phif = self.interp(phi) if grad_skew is None else self.interp_skew(phi, grad_skew)
        g = np.c_[self.sum_to_cells(phif * self.Sf[:, 0], phib * self.bSf[:, 0]),
                  self.sum_to_cells(phif * self.Sf[:, 1], phib * self.bSf[:, 1])]
        return g / self.vol[:, None]


class _RealMesh:
    """the attributes turbulence.wall_distance needs"""

    def __init__(self, P, topo, xc):
        tb = P[topo.bb] - P[topo.ba]
        self.bSf = np.c_[tb[:, 1], -tb[:, 0]]
        self.bxf = 0.5 * (P[topo.ba] + P[topo.bb])
        self.xc = xc
