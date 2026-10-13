"""3D finite-volume mesh: polyhedral cells from faces (OpenFOAM polyMesh style).

Input: a "polymesh" dict
    points     (N, 3)
    faces      (Nf, K) int, node indices, padded with -1 (K = max nodes per face)
    owner      (Nf,)   cell on one side
    neighbour  (Nf,)   cell on the other side, -1 for a boundary face
    patch      (Nf,)   patch name of boundary faces ("" for internal faces)
    ncells     int
Face orientation is fixed here (internal: out of the owner; boundary: out of the
domain), so generators need not care.

The class exposes the same interface as the 2D FVMesh (fvmesh.py) in three
components: own, nei, Sf, xf, xc, vol, d, coef, Delta, w, rskew, bcell, bSf, bxf,
db, bcoef, bDelta, db_t, patch_faces, grad (weighted least squares), grad_gg
(Green-Gauss, conservative), interp, interp_skew, sum_to_cells.
"""
import numpy as np
import scipy.sparse as sp

from . import native as N

NONORTH_CLAMP_DEG = 80.0
# optional bound on the linear interpolation weights (0: none; bounding them
# destabilises trailing-edge fans, where extreme weights are the consistent choice)
W_CLIP = 0.0


def face_geometry(P, F, warp=False):
    """area vectors and centroids of polygons F (padded with -1) by triangle fans
    about the vertex average. warp=True also returns R (nf, 3, 3) = sum over the fan
    triangles of a_t (g_t - xf)^T: for a linear field phi, the exact face integral is
    phi(xf) S + R grad(phi) (R = 0 for a planar face)."""
    K = F.shape[1]
    valid = F >= 0
    cnt = valid.sum(1)
    Fi = np.where(valid, F, F[:, :1])                  # pad with the first node
    X = P[Fi]                                          # (nf, K, 3)
    c0 = (X * valid[..., None]).sum(1) / cnt[:, None]
    # close each polygon: next node of the last valid one is the first one
    nxt = np.empty_like(Fi)
    rows = np.arange(len(F))
    for k in range(K):
        nk = np.where(valid[:, k + 1], k + 1, 0) if k + 1 < K else np.zeros(len(F), np.int64)
        nxt[:, k] = Fi[rows, nk]
    Xn = P[nxt]
    a = 0.5 * np.cross(X - c0[:, None], Xn - c0[:, None])           # (nf, K, 3)
    a = a * valid[..., None]
    S = a.sum(1)
    g = (c0[:, None] + X + Xn) / 3.0
    Smag = np.linalg.norm(S, axis=1)
    nhat = S / np.maximum(Smag, 1e-300)[:, None]
    wa = np.einsum("fkj,fj->fk", a, nhat)
    xf = (wa[..., None] * g).sum(1) / np.maximum(wa.sum(1), 1e-300)[:, None]
    if not warp:
        return S, xf
    R = np.einsum("fki,fkj->fij", a, g - xf[:, None])
    return S, xf, R


class FVMesh3D:
    def __init__(self, pm, check=True):
        P = np.asarray(pm["points"], float)
        F = np.asarray(pm["faces"], np.int64).copy()
        own = np.asarray(pm["owner"], np.int64)
        nei = np.asarray(pm["neighbour"], np.int64)
        patch = np.asarray(pm["patch"], dtype=object)
        n = int(pm["ncells"])
        self.points, self.ncells = P, n

        S, xf, R = face_geometry(P, F, warp=True)
        # approximate cell centres (mean of their face centres) to orient faces
        cnt = np.bincount(own, minlength=n) + np.bincount(nei[nei >= 0], minlength=n)
        if (cnt < 4).any():
            raise ValueError(f"{int((cnt < 4).sum())} cells have fewer than 4 faces")
        c_app = np.zeros((n, 3))
        for k in range(3):
            c_app[:, k] = (np.bincount(own, xf[:, k], n) + np.bincount(nei[nei >= 0], xf[nei >= 0, k], n)) / cnt
        internal = nei >= 0
        ref = np.where(internal[:, None], c_app[np.where(internal, nei, 0)], xf)
        if pm.get("oriented", False):         # generator wrote faces outward from the owner
            flip = np.zeros(len(F), bool)
        else:
            flip = np.einsum("ij,ij->i", ref - c_app[own], S) < 0
        if flip.any():
            Ff = F[flip]
            valid = Ff >= 0
            rev = np.full_like(Ff, -1)
            for i, (row, v) in enumerate(zip(Ff, valid)):
                r = row[v][::-1]
                rev[i, :len(r)] = r
            F[flip] = rev
            S[flip] *= -1.0
            R[flip] *= -1.0
        self.faces_all = F

        # cell volumes and centroids: pyramids from the approximate centre
        vol = np.zeros(n)
        mom = np.zeros((n, 3))
        for side, cells, sgn in (("own", own, 1.0), ("nei", nei, -1.0)):
            m = cells >= 0
            c = cells[m]
            # exact volume of the triangulated faces: (xf - c).S + tr(R)
            v = sgn * (np.einsum("ij,ij->i", xf[m] - c_app[c], S[m]) + np.trace(R[m], axis1=1, axis2=2)) / 3.0
            vol += np.bincount(c, v, n)
            cen = 0.75 * xf[m] + 0.25 * c_app[c]
            for k in range(3):
                mom[:, k] += np.bincount(c, v * cen[:, k], n)
        if check and (vol <= 0).any():
            raise ValueError(f"{int((vol <= 0).sum())} cells with non-positive volume")
        self.vol = vol
        self.xc = mom / vol[:, None]
        if pm.get("centres") is not None:      # generator-supplied cell centres (see extrude)
            self.xc = np.asarray(pm["centres"], float)

        # internal / boundary split
        ii = np.where(internal)[0]
        bb = np.where(~internal)[0]
        o_i, n_i, S_i, x_i, R_i = own[ii], nei[ii], S[ii], xf[ii], R[ii]
        # agglomerated cells can share several faces: merge them into one (summed area
        # vector, area-weighted centre); the polygon of the first one is kept for output
        lo, hi = np.minimum(o_i, n_i), np.maximum(o_i, n_i)
        key = lo * n + hi
        uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
        if len(uk) < len(key):
            sg = np.where(o_i == lo, 1.0, -1.0)[:, None]
            a = np.linalg.norm(S_i, axis=1)
            Sm = np.zeros((len(uk), 3))
            xm = np.zeros((len(uk), 3))
            np.add.at(Sm, inv, sg * S_i)
            np.add.at(xm, inv, a[:, None] * x_i)
            xm /= np.bincount(inv, a, len(uk))[:, None]
            # merged face: R gains the offsets of the sub-face centres from the new centre
            Rs = sg[:, :, None] * (R_i + np.einsum("fi,fj->fij", S_i, x_i - xm[inv]))
            Rm = np.zeros((len(uk), 3, 3))
            np.add.at(Rm, inv, Rs)
            o_i, n_i, S_i, x_i, R_i = lo[first], hi[first], Sm, xm, Rm
            ii = ii[first]
            self.n_merged_faces = int(len(key) - len(uk))
        else:
            self.n_merged_faces = 0
        self.own, self.nei = o_i, n_i
        self.Sf, self.xf = S_i, x_i
        self.faces = F[ii]
        # warped (or merged) faces: kept sparsely for the Green-Gauss warp correction
        Rb = R[bb]
        rel = lambda Rx, Sx: np.abs(Rx).reshape(len(Rx), -1).max(1) / np.maximum(
            np.einsum("ij,ij->i", Sx, Sx), 1e-300) ** 0.75
        self.warp_i = np.where(rel(R_i, S_i) > 1e-9)[0]
        self.warp_R = R_i[self.warp_i]
        self.warp_b = np.where(rel(Rb, S[bb]) > 1e-9)[0]
        self.warp_Rb = Rb[self.warp_b]
        self.bcell, self.bSf, self.bxf = own[bb], S[bb], xf[bb]
        self.bfaces = F[bb]
        self.bpatch = patch[bb].astype(str)
        self.patch_names = list(dict.fromkeys(self.bpatch.tolist()))
        self.patch_faces = {nm: np.where(self.bpatch == nm)[0] for nm in self.patch_names}
        if check:
            closure = np.zeros((n, 3))
            for k in range(3):
                closure[:, k] = np.bincount(self.own, self.Sf[:, k], n) - np.bincount(self.nei, self.Sf[:, k], n) \
                    + np.bincount(self.bcell, self.bSf[:, k], n)
            scale = vol ** (2.0 / 3.0)
            bad = np.linalg.norm(closure, axis=1) > 1e-8 * scale
            if bad.any():
                raise ValueError(f"{int(bad.sum())} cells are not closed")

        # geometry factors (as the 2D mesh)
        d = self.xc[self.nei] - self.xc[self.own]
        self.d = d
        dS = np.einsum("ij,ij->i", d, self.Sf)
        S2 = np.einsum("ij,ij->i", self.Sf, self.Sf)
        dmag_S = np.sqrt((d ** 2).sum(1) * S2)
        cos_no = dS / np.maximum(dmag_S, 1e-300)
        self.nonortho = np.degrees(np.arccos(np.clip(cos_no, -1, 1)))
        lim = np.cos(np.radians(NONORTH_CLAMP_DEG)) * dmag_S
        self.n_clamped = int((dS < lim).sum())
        dS = np.maximum(dS, lim)
        self.coef = S2 / dS
        self.Delta = d * self.coef[:, None]
        self.w_raw = np.einsum("ij,ij->i", self.xc[self.nei] - self.xf, self.Sf) / dS
        self.w = np.clip(self.w_raw, W_CLIP, 1.0 - W_CLIP)
        self.rskew = self.xf - (self.xc[self.own] + (1 - self.w)[:, None] * d)
        db = self.bxf - self.xc[self.bcell]
        self.db = db
        dbS = np.einsum("ij,ij->i", db, self.bSf)
        bS2 = np.einsum("ij,ij->i", self.bSf, self.bSf)
        dbS = np.maximum(dbS, np.cos(np.radians(NONORTH_CLAMP_DEG)) * np.sqrt((db ** 2).sum(1) * bS2))
        self.bcoef = bS2 / dbS
        self.bDelta = db * self.bcoef[:, None]
        nb_ = self.bSf / np.linalg.norm(self.bSf, axis=1)[:, None]
        self.db_t = db - np.einsum("ij,ij->i", db, nb_)[:, None] * nb_
        self.nverts = np.full(n, 0)
        self._setup_lsq()

    # ------------------------------------------------------------------
    def _setup_lsq(self):
        n = self.ncells
        own, nei, bc = self.own, self.nei, self.bcell
        M = np.zeros((n, 3, 3))
        wi = 1.0 / np.einsum("ij,ij->i", self.d, self.d)
        dd = wi[:, None, None] * np.einsum("ij,ik->ijk", self.d, self.d)
        np.add.at(M, own, dd)
        np.add.at(M, nei, dd)
        wb = 1.0 / np.einsum("ij,ij->i", self.db, self.db)
        np.add.at(M, bc, wb[:, None, None] * np.einsum("ij,ik->ijk", self.db, self.db))
        # one-cell-thick extrusions are rank-deficient in the normal direction:
        # regularise (the missing component of the gradient is then zero)
        tr = np.trace(M, axis1=1, axis2=2)
        M += (1e-12 * tr)[:, None, None] * np.eye(3)[None]
        self.Minv = np.linalg.inv(M)
        nf, nb = len(own), len(bc)
        fi = np.arange(nf)
        self._W = sp.csr_matrix((np.r_[self.w, 1 - self.w], (np.r_[fi, fi], np.r_[own, nei])), (nf, n))
        self._S = sp.csr_matrix((np.r_[np.ones(nf), -np.ones(nf)], (np.r_[own, nei], np.r_[fi, fi])), (n, nf))
        self._Sb = sp.csr_matrix((np.ones(nb), (bc, np.arange(nb))), (n, nb))
        R, Rb = [], []
        for k in range(3):
            c = self.d[:, k] * wi
            cb = self.db[:, k] * wb
            R.append(sp.csr_matrix((np.r_[c, -c, c, -c, -cb],
                                    (np.r_[own, own, nei, nei, bc], np.r_[nei, own, nei, own, bc])), (n, n)))
            Rb.append(sp.csr_matrix((cb, (bc, np.arange(nb))), (n, nb)))
        Mi = self.Minv
        D = sp.diags
        G = [sp.hstack([sum(D(Mi[:, k, j]) @ R[j] for j in range(3)),
                        sum(D(Mi[:, k, j]) @ Rb[j] for j in range(3))]) for k in range(3)]
        self._G = sp.vstack(G).tocsr()

    def grad(self, phi, phi_b):
        """weighted least-squares cell gradients (n, 3)"""
        return N.matvec(self._G, np.concatenate([phi, phi_b])).reshape(3, -1).T

    def grad_gg(self, phi, phi_b, grad_skew=None):
        """Green-Gauss gradient (conservative). With grad_skew (a lagged gradient) the
        face values get the skewness correction and warped faces their face-integral
        correction R grad: exact for linear fields on any polyhedra."""
        phif = self.interp(phi) if grad_skew is None else self.interp_skew(phi, grad_skew)
        g = np.empty((self.ncells, 3))
        for k in range(3):
            g[:, k] = self.sum_to_cells(phif * self.Sf[:, k], phi_b * self.bSf[:, k])
        if grad_skew is not None and (len(self.warp_i) or len(self.warp_b)):
            fi = self.warp_i
            o, nb = self.own[fi], self.nei[fi]
            w = self.w[fi][:, None]
            gf = w * grad_skew[o] + (1 - w) * grad_skew[nb]
            c = np.einsum("fij,fj->fi", self.warp_R, gf)
            bi = self.warp_b
            cb = np.einsum("fij,fj->fi", self.warp_Rb, grad_skew[self.bcell[bi]])
            for k in range(3):
                g[:, k] += np.bincount(o, c[:, k], self.ncells) - np.bincount(nb, c[:, k], self.ncells) \
                    + np.bincount(self.bcell[bi], cb[:, k], self.ncells)
        return g / self.vol[:, None]

    def interp(self, phi):
        return N.matvec(self._W, phi)

    def interp_skew(self, phi, grad):
        return self.interp(phi) + np.einsum("ij,ij->i", self.interp(grad), self.rskew)

    def sum_to_cells(self, fint, fbnd=None):
        s = N.matvec(self._S, fint)
        if fbnd is not None:
            s += N.matvec(self._Sb, fbnd)
        return s

    # ------------------------------------------------------------------
    def quality(self):
        """summary for pre-run checks and reports"""
        return {"cells": int(self.ncells), "faces": int(len(self.own) + len(self.bcell)),
                "nonortho_max": float(self.nonortho.max()) if len(self.nonortho) else 0.0,
                "nonortho_p99": float(np.percentile(self.nonortho, 99)) if len(self.nonortho) else 0.0,
                "clamped": self.n_clamped, "vol_min": float(self.vol.min()),
                "vol_ratio_max": float((np.maximum(self.vol[self.own], self.vol[self.nei]) /
                                        np.minimum(self.vol[self.own], self.vol[self.nei])).max())}
