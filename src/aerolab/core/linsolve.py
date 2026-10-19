"""
Linear solvers for the segregated (SIMPLE) solver, step D.

Inside SIMPLE every linear system only needs an approximate solution: the
outer iterations correct the rest. Solving exactly (sparse LU) was 82 % of
the run time.

  kind = "direct"     exact sparse LU (SuperLU) - the reference
  kind = "iterative"  (default)
      momentum, turbulence  symmetric Gauss-Seidel sweeps from the previous
                            solution until the residual drops by rtol (0.05),
                            like OpenFOAM's smoothSolver; these matrices are
                            diagonally dominant (upwind + under-relaxation)
      pressure correction   conjugate gradients preconditioned with algebraic
                            multigrid (pyamg Ruge-Stuben); the multigrid
                            hierarchy is rebuilt only every `amg_rebuild`
                            iterations (the matrix changes slowly), which
                            cuts the setup cost ~10x. Scales ~linearly with the
                            number of cells (LU does not - essential for 3D).

Benchmarks on the real 29k-cell NACA 0012 SA matrices: momentum LU 232 ms ->
GS 8 ms, SA 155 -> 7 ms, pressure 127 -> ~10-50 ms.
"""

import numpy as np

from aerolab.core import kernels
import scipy.sparse as sp

# BLAS threading only hurts at CFD vector sizes: with 2 threads and a busy
# second core a 29k-long dot product took 411 us instead of 2.9 us (the CG
# pressure solve became slower than the direct LU). Pin BLAS to one thread.
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1, user_api="blas")
except ImportError:
    pass
from scipy.sparse.linalg import splu

try:
    import pyamg
    from pyamg.relaxation.relaxation import gauss_seidel
except ImportError:                     # iterative mode needs pyamg
    pyamg = None


class LinearSolver:
    def __init__(self, kind="iterative", rtol_scalar=0.05, rtol_p=0.1, max_sweeps=100,
                 amg_rebuild=5):
        if kind == "iterative" and pyamg is None:
            print("[linsolve] pyamg not installed (pip install pyamg) -> direct solver")
            kind = "direct"
        self.kind = kind
        self.rtol_scalar, self.rtol_p = rtol_scalar, rtol_p
        self.max_sweeps, self.amg_rebuild = max_sweeps, amg_rebuild
        self._ml, self._ml_age = None, 0
        self.stats = {"sweeps": 0, "p_iters": 0, "amg_setups": 0}

    # ------------------------------------------------------------------
    def scalar(self, A, b, x0):
        """Diagonally dominant transport equation (momentum component, nuTilde).
        A: CSR matrix. Returns the (approximate) solution."""
        if self.kind == "direct":
            return splu(A.tocsc()).solve(b)
        if _fast(A):
            X = np.array(x0, dtype=float, copy=True).reshape(-1, 1)
            self.stats["sweeps"] += kernels.sgs(A.indptr, A.indices, A.data, X,
                                                np.ascontiguousarray(b, float).reshape(-1, 1),
                                                self.rtol_scalar, self.max_sweeps)
            return X[:, 0]
        x = np.array(x0, dtype=float, copy=True)
        r0 = np.linalg.norm(b - A @ x)
        if r0 == 0.0:
            return x
        for k in range(1, self.max_sweeps + 1):
            gauss_seidel(A, x, b, iterations=1, sweep="symmetric")
            if k % 2 == 0 or k == 1:                  # residual check is ~1 sweep
                if np.linalg.norm(b - A @ x) <= self.rtol_scalar * r0:
                    break
        self.stats["sweeps"] += k
        return x

    def scalars(self, A, bs, x0s):
        """Several right-hand sides with the same matrix (velocity components)."""
        if self.kind == "direct":
            lu = splu(A.tocsc())
            return [lu.solve(b) for b in bs]
        if _fast(A):              # compiled: both components in one sweep
            X = np.ascontiguousarray(np.c_[tuple(x0s)], dtype=float)
            self.stats["sweeps"] += kernels.sgs(A.indptr, A.indices, A.data, X,
                                                np.ascontiguousarray(np.c_[tuple(bs)], float),
                                                self.rtol_scalar, self.max_sweeps)
            return [X[:, c].copy() for c in range(X.shape[1])]
        return [self.scalar(A, b, x0) for b, x0 in zip(bs, x0s)]

    # ------------------------------------------------------------------
    def pressure(self, A, b):
        """Symmetric positive (semi-)definite pressure-correction system."""
        if self.kind == "direct":
            return splu(A.tocsc()).solve(b)
        nb = np.linalg.norm(b)
        if nb == 0.0:
            return np.zeros_like(b)
        if self._ml is None or self._ml_age >= self.amg_rebuild:
            self._ml = pyamg.ruge_stuben_solver(A.tocsr())
            self._ml_age = 0
            self.stats["amg_setups"] += 1
        self._ml_age += 1
        M = self._ml.aspreconditioner(cycle="V")
        it = [0]

        def cb(xk):
            it[0] += 1
        x, info = _cg(A, b, M, self.rtol_p, 200, cb)
        if info != 0:                     # stale hierarchy: rebuild once and retry
            self._ml = pyamg.ruge_stuben_solver(A.tocsr())
            self._ml_age = 1
            self.stats["amg_setups"] += 1
            x, info = _cg(A, b, self._ml.aspreconditioner(cycle="V"), self.rtol_p, 200, cb)
        self.stats["p_iters"] += it[0]
        return x


def _fast(A):
    """compiled Gauss-Seidel: numba available and a CSR matrix with sorted indices"""
    return kernels.USE_NUMBA and sp.issparse(A) and A.format == "csr" and A.has_sorted_indices


def _cg(A, b, M, rtol, maxiter, cb):
    from scipy.sparse.linalg import cg
    return cg(A, b, rtol=rtol, atol=0.0, M=M, maxiter=maxiter, callback=cb)


def csr(vals, rows, cols, n):
    """Assemble a CSR matrix (duplicates summed, sorted indices for Gauss-Seidel)."""
    A = sp.csr_matrix((vals, (rows, cols)), (n, n))
    A.sum_duplicates()
    A.sort_indices()
    return A
