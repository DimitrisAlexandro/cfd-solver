"""Shape variables for the discrete adjoint (adjoint.py): the linear inverse-distance-weighted
mesh deformation (Luke, Collins & Blades 2012) and the wall displacement per CST coefficient."""
import numpy as np


# ----------------------------------------------------------------------
# mesh deformation (inverse distance weighting)
# ----------------------------------------------------------------------
class IDWDeformation:
    """Linear map from the displacement of the wall points to the
    displacement of every mesh point (Luke, Collins & Blades 2012):
        dx(p) = sum_i w_i(p) dx_i / sum_i w_i(p),
        w_i = A_i [ (L_def / r)^3 + (alpha_def L_def / r)^5 ]
    over the wall points (moving) and the far-field points (fixed). The
    weights are computed once on the undeformed mesh, so the map is linear
    (exact for complex-step derivatives)."""

    def __init__(self, fvm, wall_patches=("wall",), fixed_patches=("inlet", "outlet"),
                 a=3.0, b=5.0, alpha_def=5.0, follow=(0.02, 0.08)):
        """follow=(d1, d2): points closer than d1 to the wall move with the
        nearest wall node AND turn with its normal (linearised rotation
        dtheta = d(dw.n)/ds of the offset), blended into the IDW field up to
        d2. Plain IDW translates the boundary-layer columns without turning
        them: where a vertical CST displacement turns the wall (the nose) the
        columns lean off-normal, and on cells with aspect ratios ~1e3 the
        explicit non-orthogonal diffusion correction of the solver then
        diverged (NACA 0012, 1.7 deg lean at x = 0.002)."""
        P = np.asarray(fvm.points, float)
        self.P = P
        wf = np.concatenate([fvm.patch_faces[w] for w in wall_patches])
        self.wall_pts = np.unique(np.r_[fvm.ba[wf], fvm.bb[wf]])
        self._setup_follow(fvm, wf, follow)
        ff = np.concatenate([fvm.patch_faces[w] for w in fixed_patches if w in fvm.patch_faces])
        fixed_pts = np.unique(np.r_[fvm.ba[ff], fvm.bb[ff]])
        src = np.r_[self.wall_pts, fixed_pts]
        self.nwall = len(self.wall_pts)
        X = P[src]
        L = float(np.ptp(P[self.wall_pts], axis=0).max())
        # area weights: half the length of the adjacent boundary faces
        ar = np.zeros(len(P))
        for faces in (wf, ff):
            le = np.linalg.norm(P[fvm.bb[faces]] - P[fvm.ba[faces]], axis=1)
            np.add.at(ar, fvm.ba[faces], 0.5 * le)
            np.add.at(ar, fvm.bb[faces], 0.5 * le)
        A = ar[src]
        moving = np.ones(len(P), bool)
        moving[src] = False
        idx = np.where(moving)[0]
        W = np.zeros((len(idx), self.nwall))
        for k0 in range(0, len(idx), 4000):
            sl = idx[k0:k0 + 4000]
            r = np.maximum(np.linalg.norm(P[sl, None, :] - X[None, :, :], axis=2), 1e-12)
            w = A[None, :] * ((L / r) ** a + (alpha_def * L / r) ** b)
            W[k0:k0 + len(sl)] = w[:, :self.nwall] / w.sum(1)[:, None]
        self.idx, self.W = idx, W

    def _setup_follow(self, fvm, wf, follow):
        from scipy.spatial import cKDTree
        self.follow = follow
        if follow is None:
            return
        P = self.P
        pos = {p: k for k, p in enumerate(self.wall_pts)}
        # wall-node neighbours along the loop (prev, next) from the wall faces
        nb = {k: [] for k in range(len(self.wall_pts))}
        for a, b in zip(fvm.ba[wf], fvm.bb[wf]):
            nb[pos[a]].append(pos[b])
            nb[pos[b]].append(pos[a])
        W = P[self.wall_pts]
        prv = np.array([v[0] if v else k for k, v in nb.items()])
        nxt = np.array([v[1] if len(v) > 1 else k for k, v in nb.items()])
        chord = W[nxt] - W[prv]
        L = np.maximum(np.linalg.norm(chord, axis=1), 1e-300)
        self._t = chord / L[:, None]
        self._prv, self._nxt, self._L = prv, nxt, L
        d, j = cKDTree(W).query(P)
        d1, d2 = follow
        self._blend = np.clip((d2 - d) / (d2 - d1), 0.0, 1.0)
        self._near = j
        self._r = P - W[j]

    def field(self, d_wall):
        """d_wall (nwall, 2) displacement of self.wall_pts -> (npoints, 2)"""
        d_wall = np.asarray(d_wall)
        out = np.zeros_like(self.P, dtype=np.result_type(d_wall.dtype, float))
        out[self.idx] = self.W @ d_wall
        if self.follow is not None:
            # rotation of the wall tangent: dtheta = (d(dw)/ds) . n, n = (-t_y, t_x)
            dd = (d_wall[self._nxt] - d_wall[self._prv]) / self._L[:, None]
            nrm = np.c_[-self._t[:, 1], self._t[:, 0]]
            dth = (dd * nrm).sum(1)
            j, r = self._near, self._r
            fol = d_wall[j] + dth[j][:, None] * np.c_[-r[:, 1], r[:, 0]]
            b = self._blend[:, None]
            out = b * fol + (1 - b) * out
        out[self.wall_pts] = d_wall
        return out


# ----------------------------------------------------------------------
# CST shape variables on a mesh
# ----------------------------------------------------------------------
def cst_wall_velocities(defo, order=8, te_split=None):
    """Wall-point displacement per CST coefficient (upper then lower;
    d y = C(x) S_j(x), x fixed): the mesh wall points are classified upper /
    lower by the loop position relative to the leading edge (min x)."""
    from aerolab.airfoil import cst
    P = defo.P[defo.wall_pts]
    x0 = P[:, 0].min()
    chord = P[:, 0].max() - x0
    x = np.clip((P[:, 0] - x0) / chord, 0.0, 1.0)
    i_le = int(np.argmin(P[:, 0]))
    # upper / lower: above / below the chord line through LE and TE
    y_le = P[i_le, 1]
    te = P[np.argmax(P[:, 0]), 1]
    yc = y_le + (te - y_le) * x
    upper = P[:, 1] >= yc
    C = np.sqrt(x) * (1 - x)
    B = cst.bernstein(x, order)                       # (npts, order+1)
    out = []
    for side in (True, False):
        for j in range(order + 1):
            d = np.zeros((len(P), 2))
            sel = upper == side
            d[sel, 1] = chord * C[sel] * B[sel, j]
            out.append(defo.field(d))
    return out
