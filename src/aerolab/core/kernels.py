"""
Compiled kernels (Phase 4): the per-iteration face loops of the SIMPLE solver
as numba functions. Each kernel computes exactly what the numpy code it
replaces computed (same formulas, same order of the face sums up to
round-off); simple_solver uses them when numba is installed and falls back
to the numpy versions otherwise (USE_NUMBA = False forces the fallback).

    venkat(...)            Venkatakrishnan limiter psi per cell
    CsrPattern             fixed sparsity pattern of the face-based matrices:
                           [diag | owner row, neighbour col | neighbour row,
                           owner col] -> CSR values filled in place each
                           iteration (no COO -> CSR conversion and sort)
    momentum_coeffs(...)   diagonal, off-diagonals and sources of the momentum
                           equations in one pass over the faces
"""

import numpy as np

try:
    import numba
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                       # pragma: no cover - fallback path
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco(a[0]) if a and callable(a[0]) else deco

USE_NUMBA = HAVE_NUMBA


# ----------------------------------------------------------------------
# Venkatakrishnan limiter
# ----------------------------------------------------------------------
@njit(cache=True)
def _venkat(phi, gx, gy, own, nei, bc, xc, xf, bxf, phib, eps2, mask, use_mask):
    n = phi.shape[0]
    pmax = phi.copy()
    pmin = phi.copy()
    for f in range(own.shape[0]):
        a, b = own[f], nei[f]
        if phi[b] > pmax[a]:
            pmax[a] = phi[b]
        if phi[b] < pmin[a]:
            pmin[a] = phi[b]
        if phi[a] > pmax[b]:
            pmax[b] = phi[a]
        if phi[a] < pmin[b]:
            pmin[b] = phi[a]
    for f in range(bc.shape[0]):
        a = bc[f]
        if phib[f] > pmax[a]:
            pmax[a] = phib[f]
        if phib[f] < pmin[a]:
            pmin[a] = phib[f]
    psi = np.ones(n)

    def lim(c, x, y):
        d2 = gx[c] * (x - xc[c, 0]) + gy[c] * (y - xc[c, 1])
        if abs(d2) <= 1e-30:
            return 1.0
        if use_mask and not mask[c]:
            return 1.0
        d1 = pmax[c] - phi[c] if d2 > 0 else pmin[c] - phi[c]
        e = eps2
        num = (d1 * d1 + e) * d2 + 2.0 * d2 * d2 * d1
        den = d2 * (d1 * d1 + 2.0 * d2 * d2 + d1 * d2 + e)
        v = num / den
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        return v

    for f in range(own.shape[0]):
        for c in (own[f], nei[f]):
            v = lim(c, xf[f, 0], xf[f, 1])
            if v < psi[c]:
                psi[c] = v
    for f in range(bc.shape[0]):
        c = bc[f]
        v = lim(c, bxf[f, 0], bxf[f, 1])
        if v < psi[c]:
            psi[c] = v
    return psi


def venkat(m, phi, g, phib, eps2, mask):
    """psi in [0, 1] per cell (see SimpleSolver._venkat). mask: cells where
    the limiter acts (None = all)."""
    use = mask is not None
    mk = mask if use else np.zeros(1, np.bool_)
    return _venkat(np.ascontiguousarray(phi), np.ascontiguousarray(g[:, 0]),
                   np.ascontiguousarray(g[:, 1]), m.own, m.nei, m.bcell, m.xc, m.xf, m.bxf,
                   np.ascontiguousarray(phib), float(eps2), mk, use)


# ----------------------------------------------------------------------
# fixed CSR pattern for [diag, owner-row/neighbour-col, neighbour-row/owner-col]
# ----------------------------------------------------------------------
class CsrPattern:
    """The face-based matrices (momentum, pressure correction, SA) all have
    the pattern diag + one entry per face in the owner row and one in the
    neighbour row. The CSR structure is built once; pos_diag, pos_on, pos_no
    give where each value goes in the CSR data array."""

    def __init__(self, n, own, nei):
        import scipy.sparse as sp
        nf = len(own)
        rows = np.r_[np.arange(n), own, nei]
        cols = np.r_[np.arange(n), nei, own]
        tag = np.arange(n + 2 * nf, dtype=np.float64) + 1.0   # unique ids
        A = sp.csr_matrix((tag, (rows, cols)), (n, n))
        A.sum_duplicates()
        A.sort_indices()
        if A.nnz != n + 2 * nf:
            raise ValueError("duplicate faces between the same two cells")
        where = np.empty(n + 2 * nf, np.int64)
        where[(A.data - 1).astype(np.int64)] = np.arange(A.nnz)
        self.indptr, self.indices = A.indptr.copy(), A.indices.copy()
        self.pos_diag = where[:n]
        self.pos_on = where[n:n + nf]
        self.pos_no = where[n + nf:]
        self.n, self.nnz = n, A.nnz

    def matrix(self, diag, a_on, a_no):
        import scipy.sparse as sp
        data = _fill(self.nnz, self.pos_diag, self.pos_on, self.pos_no,
                     np.ascontiguousarray(diag, dtype=np.float64),
                     np.ascontiguousarray(a_on, dtype=np.float64),
                     np.ascontiguousarray(a_no, dtype=np.float64))
        A = sp.csr_matrix((data, self.indices, self.indptr), (self.n, self.n), copy=False)
        A.has_sorted_indices = True
        return A


@njit(cache=True)
def _fill(nnz, pd, pon, pno, diag, a_on, a_no):
    data = np.empty(nnz)
    for i in range(pd.shape[0]):
        data[pd[i]] = diag[i]
    for f in range(pon.shape[0]):
        data[pon[f]] = a_on[f]
        data[pno[f]] = a_no[f]
    return data


# ----------------------------------------------------------------------
# momentum coefficients and sources in one pass over the faces
# ----------------------------------------------------------------------
@njit(cache=True)
def momentum_coeffs(n, own, nei, F, nu_f, coef, w, Sf, Delta, xc, xf,
                    gU0, gU1, gk0, gk1, second_order):
    """Internal-face part of SimpleSolver.iterate's momentum assembly.
    gU0/gU1: gradients of U_x / U_y (n,2); gk0/gk1: the (limited) gradients
    used in the deferred correction. Returns diag (n,), a_on, a_no (nf,),
    src (n,2) = non-orthogonal diffusion + viscous transpose + deferred
    correction (the transpose terms are added by the caller only for the
    turbulent case via the flag-free formula: nu_f * (grad U)^T . S)."""
    nf = own.shape[0]
    diag = np.zeros(n)
    a_on = np.empty(nf)
    a_no = np.empty(nf)
    src = np.zeros((n, 2))
    tr = np.zeros((n, 2))
    for f in range(nf):
        a, b = own[f], nei[f]
        D = nu_f[f] * coef[f]
        Fp = F[f] if F[f] > 0.0 else 0.0
        Fm = -F[f] if F[f] < 0.0 else 0.0
        a_on[f] = -D - Fm
        a_no[f] = -D - Fp
        diag[a] += D + Fp
        diag[b] += D + Fm
        wf = w[f]
        sx = Sf[f, 0] - Delta[f, 0]
        sy = Sf[f, 1] - Delta[f, 1]
        # interpolated gradients of both components at the face
        g00 = wf * gU0[a, 0] + (1 - wf) * gU0[b, 0]
        g01 = wf * gU0[a, 1] + (1 - wf) * gU0[b, 1]
        g10 = wf * gU1[a, 0] + (1 - wf) * gU1[b, 0]
        g11 = wf * gU1[a, 1] + (1 - wf) * gU1[b, 1]
        # explicit non-orthogonal diffusion correction
        c0 = nu_f[f] * (g00 * sx + g01 * sy)
        c1 = nu_f[f] * (g10 * sx + g11 * sy)
        # transpose of the viscous stress: (grad U)^T . S
        t0 = nu_f[f] * (g00 * Sf[f, 0] + g10 * Sf[f, 1])
        t1 = nu_f[f] * (g01 * Sf[f, 0] + g11 * Sf[f, 1])
        src[a, 0] += c0
        src[a, 1] += c1
        src[b, 0] -= c0
        src[b, 1] -= c1
        tr[a, 0] += t0
        tr[a, 1] += t1
        tr[b, 0] -= t0
        tr[b, 1] -= t1
        if second_order:
            u = a if F[f] >= 0.0 else b
            dx = xf[f, 0] - xc[u, 0]
            dy = xf[f, 1] - xc[u, 1]
            src[a, 0] -= F[f] * (gk0[u, 0] * dx + gk0[u, 1] * dy)
            src[a, 1] -= F[f] * (gk1[u, 0] * dx + gk1[u, 1] * dy)
            src[b, 0] += F[f] * (gk0[u, 0] * dx + gk0[u, 1] * dy)
            src[b, 1] += F[f] * (gk1[u, 0] * dx + gk1[u, 1] * dy)
    return diag, a_on, a_no, src, tr


# ----------------------------------------------------------------------
# symmetric Gauss-Seidel for one matrix and several right-hand sides
# ----------------------------------------------------------------------
@njit(cache=True)
def _resnorm(indptr, indices, data, X, B, out):
    n, k = X.shape
    for c in range(k):
        out[c] = 0.0
    for i in range(n):
        for c in range(k):
            s = B[i, c]
            for jj in range(indptr[i], indptr[i + 1]):
                s -= data[jj] * X[indices[jj], c]
            out[c] += s * s
    for c in range(k):
        out[c] = np.sqrt(out[c])


@njit(cache=True)
def sgs(indptr, indices, data, X, B, rtol, max_sweeps):
    """Symmetric Gauss-Seidel on A X = B (CSR, sorted indices, all columns at
    once: the matrix is read once per sweep for every right-hand side), in
    place on X, until every column's residual 2-norm has dropped by rtol
    (checked after sweep 1 and every 2nd sweep, as LinearSolver.scalar).
    Returns the number of sweeps."""
    n, k = X.shape
    diag = np.empty(n)
    for i in range(n):
        diag[i] = 0.0
        for jj in range(indptr[i], indptr[i + 1]):
            if indices[jj] == i:
                diag[i] = data[jj]
    r0 = np.empty(k)
    r = np.empty(k)
    _resnorm(indptr, indices, data, X, B, r0)
    done = True
    for c in range(k):
        if r0[c] > 0.0:
            done = False
    if done:
        return 0
    sweeps = 0
    for it in range(1, max_sweeps + 1):
        for i in range(n):                                   # forward
            for c in range(k):
                s = B[i, c]
                for jj in range(indptr[i], indptr[i + 1]):
                    j = indices[jj]
                    if j != i:
                        s -= data[jj] * X[j, c]
                X[i, c] = s / diag[i]
        for i in range(n - 1, -1, -1):                       # backward
            for c in range(k):
                s = B[i, c]
                for jj in range(indptr[i], indptr[i + 1]):
                    j = indices[jj]
                    if j != i:
                        s -= data[jj] * X[j, c]
                X[i, c] = s / diag[i]
        sweeps = it
        if it % 2 == 0 or it == 1:
            _resnorm(indptr, indices, data, X, B, r)
            ok = True
            for c in range(k):
                if r[c] > rtol * r0[c]:
                    ok = False
            if ok:
                break
    return sweeps


# ----------------------------------------------------------------------
# Spalart-Allmaras source and its cell-local derivative
# ----------------------------------------------------------------------
@njit(cache=True)
def _sa_S(nT, Om, d2, nu, g, has_g, CB1, CW1, CW2, CW3, CV1, CV2, CV3, KAPPA):
    chi = nT / nu
    fv1 = chi**3 / (chi**3 + CV1**3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)
    Sbar = nT * fv2 / (KAPPA**2 * d2)
    if Sbar >= -CV2 * Om:
        St = Om + Sbar
    else:
        St = Om + Om * (CV2**2 * Om + CV3 * Sbar) / ((CV3 - 2 * CV2) * Om - Sbar)
    if St > 0:
        r = nT / (St * KAPPA**2 * d2)
        if not np.isfinite(r):
            r = 10.0
    else:
        r = 10.0
    if r > 10.0:
        r = 10.0
    gg = r + CW2 * (r**6 - r)
    fw = gg * ((1 + CW3**6) / (gg**6 + CW3**6)) ** (1.0 / 6.0)
    prod = CB1 * St * nT
    if has_g:
        prod *= g
    return prod - CW1 * fw * nT**2 / d2


@njit(cache=True)
def sa_source(nuT, Omega, d, nu, gamma, has_gamma, CB1, CW1, CW2, CW3, CV1, CV2, CV3, KAPPA):
    """S0 = SA source (per unit volume) and J = dS/dnuT by the same one-sided
    difference as SpalartAllmaras.solve (eps = 1e-6 max(nuT, nu))."""
    n = nuT.shape[0]
    S0 = np.empty(n)
    J = np.empty(n)
    for i in range(n):
        d2 = d[i] * d[i]
        g = gamma[i] if has_gamma else 1.0
        s0 = _sa_S(nuT[i], Omega[i], d2, nu, g, has_gamma, CB1, CW1, CW2, CW3, CV1, CV2, CV3,
                   KAPPA)
        eps = 1e-6 * max(nuT[i], nu)
        s1 = _sa_S(nuT[i] + eps, Omega[i], d2, nu, g, has_gamma, CB1, CW1, CW2, CW3, CV1, CV2,
                   CV3, KAPPA)
        S0[i] = s0
        J[i] = (s1 - s0) / eps
    return S0, J


# ----------------------------------------------------------------------
# coupled (u, v, p) block system: matrix-free product
# ----------------------------------------------------------------------
@njit(cache=True)
def coupled_matvec(n, own, nei, bc, diag, a_on, a_no, w, Sf, bSf, pb_impl, cf, cb_p, outlet,
                   x, want_mom, want_cont):
    """y = K x for x = [u (n), v (n), p (n)]:
       momentum k: diag u + sum a_nb u_nb + sum_f (w p_P + (1-w) p_N) S_f,k
                   + boundary pb_impl p_P bS_k
       continuity: sum_f +-[(w u_P + (1-w) u_N).S_f - cf (p_N - p_P)]
                   + outlet faces (U_P . bS + cb_p p_P)
    want_mom / want_cont skip a block (used for the D and G products)."""
    y = np.zeros(3 * n)
    u = x[:n]
    v = x[n:2 * n]
    p = x[2 * n:]
    if want_mom:
        for i in range(n):
            y[i] = diag[i] * u[i]
            y[n + i] = diag[i] * v[i]
    for f in range(own.shape[0]):
        a, b = own[f], nei[f]
        wf = w[f]
        sx, sy = Sf[f, 0], Sf[f, 1]
        if want_mom:
            y[a] += a_on[f] * u[b]
            y[b] += a_no[f] * u[a]
            y[n + a] += a_on[f] * v[b]
            y[n + b] += a_no[f] * v[a]
            pf = wf * p[a] + (1.0 - wf) * p[b]
            y[a] += pf * sx
            y[n + a] += pf * sy
            y[b] -= pf * sx
            y[n + b] -= pf * sy
        if want_cont:
            Fi = (wf * u[a] + (1.0 - wf) * u[b]) * sx + (wf * v[a] + (1.0 - wf) * v[b]) * sy \
                - cf[f] * (p[b] - p[a])
            y[2 * n + a] += Fi
            y[2 * n + b] -= Fi
    for f in range(bc.shape[0]):
        c = bc[f]
        if want_mom:
            y[c] += pb_impl[f] * bSf[f, 0] * p[c]
            y[n + c] += pb_impl[f] * bSf[f, 1] * p[c]
        if want_cont and outlet[f]:
            y[2 * n + c] += u[c] * bSf[f, 0] + v[c] * bSf[f, 1] + cb_p[f] * p[c]
    return y
