"""Bindings to the C++ kernels (native/aero3d.cpp) with numpy fallbacks.

The library is compiled on first use with g++ (-O3 -march=native -fopenmp) and cached
by source hash in native/build/. Without a compiler everything still runs through the
numpy / pyamg fallbacks (slower). Environment:
    AEROLAB_NATIVE=0      force the fallbacks
    AEROLAB_THREADS=n     OpenMP threads (default: all cores)
    AEROLAB_NATIVE_MARCH  g++ -march target (default "native": tuned to this CPU; a Docker
                          image built for other machines can use e.g. "x86-64-v2")
"""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent / "native"
SRC = HERE / "aero3d.cpp"
_lib = None
_err = None

I64 = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
I8 = np.ctypeslib.ndpointer(np.int8, flags="C_CONTIGUOUS")
U8 = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
F64 = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
c_i = ctypes.c_int64
c_d = ctypes.c_double
VP = ctypes.c_void_p


def _build():
    march = os.environ.get("AEROLAB_NATIVE_MARCH", "native").strip() or "native"
    h = hashlib.sha256(SRC.read_bytes() + march.encode()).hexdigest()[:10]
    out = HERE / "build" / f"libaero3d_{h}.so"
    if not out.exists():
        out.parent.mkdir(exist_ok=True)
        cmd = ["g++", "-O3", f"-march={march}", "-fopenmp", "-shared", "-fPIC", "-std=c++17", str(SRC), "-o",
               str(out) + ".tmp"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:                 # retry without -march (portable build)
            cmd.remove(f"-march={march}")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-2000:])
        os.replace(str(out) + ".tmp", out)
    return out


def lib():
    """the loaded library or None (then the fallbacks run)"""
    global _lib, _err
    if _lib is not None or _err is not None:
        return _lib
    if os.environ.get("AEROLAB_NATIVE", "1") == "0":
        _err = "disabled by AEROLAB_NATIVE=0"
        return None
    try:
        L = ctypes.CDLL(str(_build()))
    except Exception as e:                                   # noqa: BLE001
        _err = f"native kernels unavailable ({e}); using numpy fallbacks"
        return None
    L.aero_threads.argtypes = [ctypes.c_int]
    L.aero_threads.restype = ctypes.c_int
    L.mom_assemble.argtypes = [c_i, c_i, I64, I64, I64, I64, I8, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64,
                               ctypes.c_int, ctypes.c_int, F64, F64, F64, F64, F64]
    L.venkat.argtypes = [c_i, I64, I64, I64, I64, I8, I64, I64, F64, F64, F64, F64, F64, F64, c_d, VP, F64]
    L.csr_fill.argtypes = [c_i, c_i, I64, I64, I64, F64, F64, F64, F64]
    L.spmv.argtypes = [c_i, I64, I64, F64, F64, F64]
    L.sgs.argtypes = [c_i, c_i, I64, I64, F64, F64, F64, c_d, c_i]
    L.sgs.restype = c_i
    L.amg_new.restype = VP
    L.amg_free.argtypes = [VP]
    L.amg_add_level.argtypes = [VP, c_i, I64, I64, F64, c_i, I64, I64, F64, I64, I64, F64]
    L.amg_set_coarse.argtypes = [VP, c_i, F64]
    L.amg_pcg.argtypes = [VP, c_i, I64, I64, F64, F64, F64, c_d, c_i]
    L.amg_pcg.restype = c_i
    L.sa_source.argtypes = [c_i, F64, F64, F64, c_d, VP, F64, F64, F64]
    nt = int(os.environ.get("AEROLAB_THREADS", "0") or 0)
    L.aero_threads(nt)
    _lib = L
    return L


def status():
    L = lib()
    return {"native": L is not None, "threads": L.aero_threads(0) if L else 1, "note": _err}


def _c(a, dt=np.float64):
    return np.ascontiguousarray(a, dtype=dt)


# ----------------------------------------------------------------------
class Adjacency:
    """internal and boundary faces of every cell (CSR-like), for cell-parallel gathers"""

    def __init__(self, m):
        n, nf = m.ncells, len(m.own)
        cells = np.r_[m.own, m.nei]
        faces = np.r_[np.arange(nf), np.arange(nf)]
        sign = np.r_[np.ones(nf, np.int8), -np.ones(nf, np.int8)]
        o = np.argsort(cells, kind="stable")
        self.cface, self.csign = _c(faces[o], np.int64), _c(sign[o], np.int8)
        self.cptr = _c(np.r_[0, np.cumsum(np.bincount(cells, minlength=n))], np.int64)
        ob = np.argsort(m.bcell, kind="stable")
        self.bface = _c(ob, np.int64)
        self.bptr = _c(np.r_[0, np.cumsum(np.bincount(m.bcell, minlength=n))], np.int64)


class CsrPattern:
    """fixed CSR pattern of the face-based matrices (diag + owner/neighbour entries)"""

    def __init__(self, n, own, nei):
        nf = len(own)
        rows = np.r_[np.arange(n), own, nei]
        cols = np.r_[np.arange(n), nei, own]
        tag = np.arange(n + 2 * nf, dtype=np.float64) + 1.0
        A = sp.csr_matrix((tag, (rows, cols)), (n, n))
        A.sum_duplicates()
        A.sort_indices()
        if A.nnz != n + 2 * nf:
            raise ValueError("duplicate faces between the same two cells")
        where = np.empty(n + 2 * nf, np.int64)
        where[(A.data - 1).astype(np.int64)] = np.arange(A.nnz)
        self.indptr, self.indices = _c(A.indptr, np.int64), _c(A.indices, np.int64)
        self.pos_diag, self.pos_on, self.pos_no = _c(where[:n], np.int64), _c(where[n:n + nf], np.int64), \
            _c(where[n + nf:], np.int64)
        self.n, self.nf, self.nnz = n, nf, A.nnz

    def matrix(self, diag, a_on, a_no):
        L = lib()
        data = np.empty(self.nnz)
        if L:
            L.csr_fill(self.n, self.nf, self.pos_diag, self.pos_on, self.pos_no, _c(diag), _c(a_on), _c(a_no), data)
        else:
            data[self.pos_diag] = diag
            data[self.pos_on] = a_on
            data[self.pos_no] = a_no
        A = sp.csr_matrix((data, self.indices, self.indptr), (self.n, self.n), copy=False)
        A.has_sorted_indices = True
        return A


# ----------------------------------------------------------------------
def mom_assemble(m, adj, F, nu_f, gU, gk, second_order, transpose):
    """gU, gk: (3, n, 3) gradients of the 3 velocity components (gk: limited).
    Returns diag (n,), a_on, a_no (nf,), src (n,3) (internal faces only)."""
    n, nf = m.ncells, len(m.own)
    L = lib()
    nu_f = np.broadcast_to(np.asarray(nu_f, float), (nf,))
    if L:
        diag, a_on, a_no, src = np.empty(n), np.empty(nf), np.empty(nf), np.empty((n, 3))
        L.mom_assemble(n, nf, m.own, m.nei, adj.cptr, adj.cface, adj.csign, _c(F), _c(nu_f), m.coef, m.w,
                       m.Sf, m.Delta, m.xc, m.xf, _c(gU), _c(gk), int(second_order), int(transpose),
                       diag, a_on, a_no, src, np.empty(nf * 5))
        return diag, a_on, a_no, src
    own, nei = m.own, m.nei
    D = nu_f * m.coef
    Fp, Fm = np.maximum(F, 0.0), np.maximum(-F, 0.0)
    a_on, a_no = -D - Fm, -D - Fp
    diag = np.bincount(own, D + Fp, n) + np.bincount(nei, D + Fm, n)
    w = m.w[:, None]
    gf = np.stack([w * gU[k][own] + (1 - w) * gU[k][nei] for k in range(3)])     # (3, nf, 3)
    s = m.Sf - m.Delta
    up = np.where(F >= 0, own, nei)
    dx = m.xf - m.xc[up]
    src = np.zeros((n, 3))
    for k in range(3):
        c = nu_f * np.einsum("ij,ij->i", gf[k], s)
        if transpose:
            c = c + nu_f * np.einsum("jf,fj->f", gf[:, :, k], m.Sf)
        if second_order:
            c = c - F * np.einsum("ij,ij->i", gk[k][up], dx)
        src[:, k] = m.sum_to_cells(c)
    return diag, a_on, a_no, src


def venkat(m, adj, phi, g, phib, eps2, mask=None):
    L = lib()
    n = m.ncells
    if L:
        psi = np.empty(n)
        mk = _c(mask, np.uint8) if mask is not None else None
        L.venkat(n, m.own, m.nei, adj.cptr, adj.cface, adj.csign, adj.bptr, adj.bface, m.xc, m.xf, m.bxf,
                 _c(phi), _c(g), _c(phib), float(eps2), mk.ctypes.data if mk is not None else None, psi)
        return psi
    own, nei, bc = m.own, m.nei, m.bcell
    pmax, pmin = phi.copy(), phi.copy()
    np.maximum.at(pmax, own, phi[nei]); np.maximum.at(pmax, nei, phi[own])
    np.minimum.at(pmin, own, phi[nei]); np.minimum.at(pmin, nei, phi[own])
    np.maximum.at(pmax, bc, phib); np.minimum.at(pmin, bc, phib)
    psi = np.ones(n)
    for c, x in ((own, m.xf), (nei, m.xf), (bc, m.bxf)):
        d2 = np.einsum("ij,ij->i", g[c], x - m.xc[c])
        d1 = np.where(d2 > 0, pmax[c] - phi[c], pmin[c] - phi[c])
        num = (d1 ** 2 + eps2) * d2 + 2 * d2 ** 2 * d1
        den = d2 * (d1 ** 2 + 2 * d2 ** 2 + d1 * d2 + eps2)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(np.abs(d2) > 1e-30, num / den, 1.0)
        if mask is not None:
            f = np.where(mask[c], f, 1.0)
        np.minimum.at(psi, c, np.clip(f, 0.0, 1.0))
    return psi


def sgs(A, X, B, rtol=0.05, max_sweeps=100):
    """symmetric Gauss-Seidel on A X = B in place (X, B: (n, k)); returns sweeps"""
    L = lib()
    if L:
        Xc = _c(X)
        s = L.sgs(A.shape[0], X.shape[1], _c(A.indptr, np.int64), _c(A.indices, np.int64), _c(A.data), Xc,
                  _c(B), float(rtol), int(max_sweeps))
        X[...] = Xc
        return int(s)
    from pyamg.relaxation.relaxation import gauss_seidel
    sw = 0
    for c in range(X.shape[1]):
        x, b = X[:, c].copy(), B[:, c]
        r0 = np.linalg.norm(b - A @ x)
        if r0 == 0:
            continue
        for k in range(1, max_sweeps + 1):
            gauss_seidel(A, x, b, iterations=1, sweep="symmetric")
            if (k % 2 == 0 or k == 1) and np.linalg.norm(b - A @ x) <= rtol * r0:
                break
        X[:, c] = x
        sw = max(sw, k)
    return sw


def _coarse_inverse(Ac):
    """pseudo-inverse of the coarsest AMG level (dense, symmetric)"""
    if not np.isfinite(Ac).all():
        raise FloatingPointError("AMG: non-finite pressure matrix (the solution diverged)")
    Ac = 0.5 * (Ac + Ac.T)
    try:
        return np.linalg.pinv(Ac, hermitian=True)
    except np.linalg.LinAlgError:
        # the divide-and-conquer SVD can fail on badly scaled matrices; gesvd is slower but robust
        import scipy.linalg
        u, s, vt = scipy.linalg.svd(Ac, lapack_driver="gesvd")
        cut = s > 1e-15 * s.max()
        return (vt[cut].T / s[cut]) @ u[:, cut].T


class AMG:
    """algebraic multigrid hierarchy (pyamg Ruge-Stuben) + V-cycle preconditioned CG in C++"""

    def __init__(self, A, max_coarse=400):
        import pyamg
        self.ml = pyamg.ruge_stuben_solver(A.tocsr(), max_coarse=max_coarse)
        self.h = None
        L = lib()
        if L:
            self.h = L.amg_new()
            lv = self.ml.levels
            for i in range(len(lv) - 1):
                Al, P, R = lv[i].A.tocsr(), lv[i].P.tocsr(), lv[i].R.tocsr()
                for M in (Al, P, R):
                    M.sort_indices()
                L.amg_add_level(self.h, Al.shape[0], _c(Al.indptr, np.int64), _c(Al.indices, np.int64), _c(Al.data),
                                P.shape[1], _c(P.indptr, np.int64), _c(P.indices, np.int64), _c(P.data),
                                _c(R.indptr, np.int64), _c(R.indices, np.int64), _c(R.data))
            Ac = lv[-1].A.toarray()
            L.amg_set_coarse(self.h, Ac.shape[0], _c(_coarse_inverse(Ac)))

    def solve(self, A, b, x0=None, rtol=0.1, maxit=200):
        """returns (x, iterations); iterations < 0: not converged"""
        L = lib()
        x = np.zeros(len(b)) if x0 is None else _c(x0).copy()
        if L and self.h:
            it = L.amg_pcg(self.h, A.shape[0], _c(A.indptr, np.int64), _c(A.indices, np.int64), _c(A.data), _c(b),
                           x, float(rtol), int(maxit))
            return x, int(it)
        from scipy.sparse.linalg import cg
        n_it = [0]
        x, info = cg(A, b, x0=x, rtol=rtol, atol=0.0, M=self.ml.aspreconditioner(cycle="V"), maxiter=maxit,
                     callback=lambda _: n_it.__setitem__(0, n_it[0] + 1))
        return x, (n_it[0] if info == 0 else -n_it[0])

    def __del__(self):
        L = _lib
        if L is not None and getattr(self, "h", None):
            L.amg_free(self.h)
            self.h = None


SA_C = None


def sa_source(nuT, Omega, d, nu, gamma, consts):
    """(S0, dS/dnuT) per cell; consts = (CB1, CW1, CW2, CW3, CV1, CV2, CV3, KAPPA)"""
    L = lib()
    n = len(nuT)
    if L:
        S0, J = np.empty(n), np.empty(n)
        g = _c(gamma) if gamma is not None else None
        L.sa_source(n, _c(nuT), _c(Omega), _c(d), float(nu), g.ctypes.data if g is not None else None,
                    _c(consts), S0, J)
        return S0, J
    from aerolab.core import kernels
    return kernels.sa_source(_c(nuT), _c(Omega), _c(d), float(nu),
                             np.ones(1) if gamma is None else _c(gamma), gamma is not None, *consts)


def matvec(M, x):
    """CSR matrix times vector (n,) or matrix (n, k), multithreaded in C++"""
    L = lib()
    if L is None:
        return M @ x
    ip, ix, a = M._aero_csr if hasattr(M, "_aero_csr") else _cache_csr(M)
    x = np.asarray(x, float)
    if x.ndim == 1:
        y = np.empty(M.shape[0])
        L.spmv(M.shape[0], ip, ix, a, _c(x), y)
        return y
    y = np.empty((M.shape[0], x.shape[1]))
    for k in range(x.shape[1]):
        yk = np.empty(M.shape[0])
        L.spmv(M.shape[0], ip, ix, a, _c(x[:, k]), yk)
        y[:, k] = yk
    return y


def _cache_csr(M):
    M._aero_csr = (_c(M.indptr, np.int64), _c(M.indices, np.int64), _c(M.data))
    return M._aero_csr
