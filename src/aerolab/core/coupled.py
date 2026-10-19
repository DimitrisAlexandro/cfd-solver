"""
Pressure-based COUPLED solver (Phase 4): momentum and continuity solved as
one linear system per outer iteration, instead of SIMPLEC's momentum
predictor + pressure correction.

Why: on the stretched boundary-layer cells of an airfoil mesh SIMPLEC
converges geometrically and slowly (NACA 0012 SA: residuals x0.986 per
iteration, the same with exact linear solves, frozen eddy viscosity,
first-order convection, any under-relaxation >= 0.9 and non-orthogonal
pressure correctors): its pressure equation approximates the velocity-
pressure coupling of those cells badly. Solved together (Darwish, Sraj &
Moukalled 2009), the coupling is exact within each outer iteration.

Discretisation: identical to SimpleSolver (same momentum matrix, Green-Gauss
pressure gradient with p_f = w p_P + (1-w) p_N + lagged skew correction,
Rhie-Chow face flux with the same explicit parts), so the converged
solution is the same; only the iteration differs.

Unknowns [u (n), v (n), p (n)]. Rows:
  momentum k:  a_P u_P + sum a_nb u_nb + sum_f p_f S_f,k          = b_k
  continuity:  sum_f [ (w u_P + (1-w) u_N).S_f - Df coef (p_N - p_P) ]
               + boundary fluxes                                  = - explicit parts
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from aerolab.core import kernels
from aerolab.core.simple_solver import SimpleSolver


class CoupledSolver(SimpleSolver):
    """Same constructor as SimpleSolver; alpha_u is the momentum under-
    relaxation of the coupled system (can be close to 1)."""

    def __init__(self, *a, linear="pardiso", gmres_rtol=0.05, sa_subits=3, **k):
        k.setdefault("alpha_u", 0.99)
        super().__init__(*a, **k)
        if not self.has_outlet:
            raise NotImplementedError("coupled solver: needs an outlet (pressure level)")
        self.coupled_linear = linear
        self.gmres_rtol = gmres_rtol
        self._ml, self._ml_age = None, 0
        self.lin_stats = {"gmres_its": 0, "amg_setups": 0, "factorizations": 0}
        self._lu = None
        self._lu_age = 0
        self.refactor_every = 1           # outer iterations a factorization is reused
        self.sa_subits = sa_subits        # SA iterations per outer iteration
        self.refactor_its = 8             # ... or refactor when GMRES needs more
        # adaptive relaxation: with alpha_u ~ 1 the lagged explicit parts
        # (deferred-correction convection, Rhie-Chow / skew terms) can leave
        # one slowly GROWING mode near convergence (NACA 0012 warm starts:
        # x1.03 per iteration from 1e-6). When the residual has risen to
        # adapt_rise x its running minimum, alpha_u steps down (0.99 -> 0.95
        # -> 0.9). The converged answer does not depend on alpha_u (Majumdar
        # terms on interior AND outlet faces), only the path does.
        self.adapt_rise = 4.0
        self.alpha_min = 0.9
        self._res_min = np.inf
        if self._pattern is None:
            self._pattern = kernels.CsrPattern(self.m.ncells, self.m.own, self.m.nei)

    # ------------------------------------------------------------------
    def iterate(self):
        m, n = self.m, self.m.ncells
        own, nei, bc = m.own, m.nei, m.bcell
        nf, nb = len(own), len(bc)
        Ub, pb = self._boundary_values()
        gU = [m.grad(self.U[:, k], Ub[:, k]) for k in range(2)]
        self.gU = gU
        self.gradp = m.grad_gg(self.p, pb, grad_skew=self.gradp)

        diag, a_on, a_no, src, fast, nu_f, nu_b, Db = self._momentum(Ub, gU)
        src = src + self.gradp * m.vol[:, None]          # pressure term now implicit
        diag = diag / self.au
        src = src + ((1 - self.au) * diag)[:, None] * self.U

        # ---- pressure in the momentum rows: sum_f p_f S_f ----------------
        w = m.w
        Sx, Sy = m.Sf[:, 0], m.Sf[:, 1]
        # lagged skew correction of the face pressure (as grad_gg(grad_skew=...))
        pskew = np.einsum('ij,ij->i', m.interp(self.gradp), m.rskew)
        for k, S in enumerate((Sx, Sy)):
            src[:, k] -= m.sum_to_cells(pskew * S)
        # boundary faces: outlet p fixed, else p_b = p_P + grad p_P . db (lagged part explicit)
        o = self.b_outlet
        bS = m.bSf
        pb_expl = np.where(o, self.pb_out, np.einsum('ij,ij->i', self.gradp[bc], m.db))
        pb_impl = np.where(o, 0.0, 1.0)                  # coefficient of p_P
        for k in range(2):
            src[:, k] -= np.bincount(bc, pb_expl * bS[:, k], n)

        # ---- continuity: Rhie-Chow with the relaxed diagonal ---------------
        Dp = m.vol / diag
        Df = m.interp(Dp)
        gp_f = m.interp(self.gradp)
        uskew = np.einsum('ij,ij->i', m.interp(gU[0]), m.rskew) * Sx + \
            np.einsum('ij,ij->i', m.interp(gU[1]), m.rskew) * Sy
        Fexpl = uskew + Df * np.einsum('ij,ij->i', gp_f, m.Delta)
        UbarS_old = self.Ubar_S_old
        Fexpl = Fexpl + (1 - self.au) * (self.F - UbarS_old)        # Majumdar
        cf = Df * m.coef
        # boundary fluxes
        fixed = self.b_fixedU
        Fb_expl = np.where(fixed, np.einsum('ij,ij->i', Ub, bS), 0.0)
        # outlet: Fb = (U_P + gU.db_t).S - Dp_P ((pb - p_P) bcoef - gradp_P . bDelta)
        Fb_expl += np.where(o, np.einsum('ij,ij->i', np.c_[np.einsum('ij,ij->i', gU[0][bc], m.db_t),
                                                           np.einsum('ij,ij->i', gU[1][bc], m.db_t)], bS)
                            - Dp[bc] * (self.pb_out * m.bcoef
                                        - np.einsum('ij,ij->i', self.gradp[bc], m.bDelta)), 0.0)
        gUdb = np.c_[np.einsum('ij,ij->i', gU[0][bc], m.db_t),
                     np.einsum('ij,ij->i', gU[1][bc], m.db_t)]
        Fb_expl += np.where(o, (1 - self.au) * (self.Fb - self.Ubar_Sb_old), 0.0)  # Majumdar
        cb_p = np.where(o, Dp[bc] * m.bcoef, 0.0)        # coefficient of p_P in Fb

        rhs = np.concatenate([src[:, 0], src[:, 1],
                              -(m.sum_to_cells(Fexpl) + np.bincount(bc, Fb_expl, n))])
        x0 = np.concatenate([self.U[:, 0], self.U[:, 1], self.p])
        ops = dict(n=n, diag=diag, a_on=a_on, a_no=a_no, pb_impl=pb_impl, cf=cf, cb_p=cb_p)
        K = lambda xx, mom=True, cont=True: kernels.coupled_matvec(
            n, own, nei, bc, diag, a_on, a_no, w, m.Sf, bS, pb_impl, cf, cb_p, o,
            np.ascontiguousarray(xx), mom, cont)
        r_before = rhs - K(x0)
        scale_u = max((diag * np.linalg.norm(self.U, axis=1)).sum(), 1e-30)
        Fref = max(np.abs(self.Fb).sum(), 1e-30)
        res = {"cont": np.abs(r_before[2 * n:]).sum() / Fref,
               "Ux": np.abs(r_before[:n]).sum() / scale_u,
               "Uy": np.abs(r_before[n:2 * n]).sum() / scale_u}
        if self.coupled_linear == "direct":
            x = spsolve(self._assemble(ops, w, Sx, Sy, bS, o).tocsc(), rhs)
        elif self.coupled_linear == "pardiso":
            x = self._pardiso(K, self._assemble(ops, w, Sx, Sy, bS, o), rhs, x0)
        else:
            x = self._gmres(K, rhs, x0, diag, a_on, a_no, cf, cb_p, Dp)
        self.U = np.c_[x[:n], x[n:2 * n]]
        self.p = x[2 * n:]
        # consistent face fluxes from the new fields
        UbarS = m.interp(self.U[:, 0]) * Sx + m.interp(self.U[:, 1]) * Sy + uskew
        self.F = UbarS - cf * (self.p[nei] - self.p[own]) + (Fexpl - uskew)
        self.Ubar_S_old = UbarS
        self.Fb = Fb_expl + np.where(o, np.einsum('ij,ij->i', self.U[bc], bS) + cb_p * self.p[bc],
                                     0.0)
        self.Ubar_Sb_old = np.where(o, np.einsum('ij,ij->i', self.U[bc] + gUdb, bS),
                                    self.Ubar_Sb_old)
        if self.turb is not None:
            self.gU = [m.grad(self.U[:, k], self._boundary_values()[0][:, k]) for k in range(2)]
            res["nuT"] = self.turb.solve()
            for _ in range(self.sa_subits - 1):  # the flow solve costs far more:
                self.turb.solve()                # let SA catch up with it
        self.history.append(res)
        self._adapt(max(res.values()))
        return res

    def _adapt(self, r):
        if r < self._res_min:
            self._res_min = r
        elif self._res_min < 1e-4 and r > self.adapt_rise * self._res_min and \
                self.au > self.alpha_min + 1e-9:
            self.au = max(self.alpha_min, 0.95 if self.au > 0.95 else self.au - 0.05)
            self._res_min = r                # watch again from here

    # ------------------------------------------------------------------
    def _gmres(self, K, rhs, x0, diag, a_on, a_no, cf, cb_p, Dp):
        """GMRES on the block system, preconditioned by one linearised SIMPLE
        step (block LU with the SIMPLE Schur approximation):
           u1 = Auu^-1 r_u            (2 symmetric Gauss-Seidel sweeps)
           p  = S^-1 (r_p - D u1)     (one AMG V-cycle; S = compact Laplacian,
                                       coefficient cf + interp(V/a_P) |S|^2/d.S)
           u  = u1 - diag^-1 G p"""
        import pyamg
        from scipy.sparse.linalg import LinearOperator, gmres
        m, n = self.m, self.m.ncells
        pat = self._pattern
        Auu = pat.matrix(diag, a_on, a_no)
        # Schur approximation: Rhie-Chow part + velocity part of the coupling
        cs = cf + m.interp(Dp) * m.coef
        sdiag = np.bincount(m.own, cs, n) + np.bincount(m.nei, cs, n) + \
            np.bincount(m.bcell, cb_p + np.where(self.b_outlet, Dp[m.bcell] * m.bcoef, 0.0), n)
        Sp = pat.matrix(sdiag, -cs, -cs)
        if self._ml is None or self._ml_age >= 5:
            self._ml = pyamg.ruge_stuben_solver(Sp)
            self._ml_age = 0
            self.lin_stats["amg_setups"] += 1
        self._ml_age += 1
        ml = self._ml
        zero_n = np.zeros(n)

        def prec(r):
            r = np.asarray(r).ravel()
            X = np.zeros((n, 2))
            kernels.sgs(Auu.indptr, Auu.indices, Auu.data, X,
                        np.ascontiguousarray(np.c_[r[:n], r[n:2 * n]]), 0.0, 2)
            u1 = np.concatenate([X[:, 0], X[:, 1], zero_n])
            rp = r[2 * n:] - K(u1, False, True)[2 * n:]
            p = ml.solve(rp, maxiter=1, cycle="V", tol=1e-30)
            Gp = K(np.concatenate([zero_n, zero_n, p]), True, False)
            return np.concatenate([X[:, 0] - Gp[:n] / diag, X[:, 1] - Gp[n:2 * n] / diag, p])

        N = 3 * n
        A_op = LinearOperator((N, N), matvec=lambda xx: K(xx), dtype=float)
        M_op = LinearOperator((N, N), matvec=prec, dtype=float)
        its = [0]

        def cb(_):
            its[0] += 1
        # tolerance relative to the INITIAL residual (|rhs| is dominated by
        # the relaxation term (1-alpha) a_P U and would stop GMRES at once)
        r0 = np.linalg.norm(rhs - K(x0))
        x, info = gmres(A_op, rhs, x0=x0, rtol=0.0, atol=self.gmres_rtol * r0, restart=40,
                        maxiter=4, M=M_op, callback=cb, callback_type="pr_norm")
        self.lin_stats["gmres_its"] += its[0]
        return x

    def _assemble(self, ops, w, Sx, Sy, bS, o):
        """The block matrix explicitly (direct solves, checks)."""
        m = self.m
        n = ops["n"]
        own, nei, bc = m.own, m.nei, m.bcell
        I = np.arange(n)
        R, C, V = [], [], []

        def add(r, c, v):
            R.append(r); C.append(c); V.append(np.broadcast_to(v, np.shape(r)))
        for k, S in enumerate((Sx, Sy)):
            ro = k * n
            add(ro + I, ro + I, ops["diag"])
            add(ro + own, ro + nei, ops["a_on"])
            add(ro + nei, ro + own, ops["a_no"])
            add(ro + own, 2 * n + own, w * S)
            add(ro + own, 2 * n + nei, (1 - w) * S)
            add(ro + nei, 2 * n + own, -w * S)
            add(ro + nei, 2 * n + nei, -(1 - w) * S)
            add(ro + bc, 2 * n + bc, ops["pb_impl"] * bS[:, k])
            add(2 * n + own, ro + own, w * S)
            add(2 * n + own, ro + nei, (1 - w) * S)
            add(2 * n + nei, ro + own, -w * S)
            add(2 * n + nei, ro + nei, -(1 - w) * S)
            add(2 * n + bc, ro + bc, np.where(o, bS[:, k], 0.0))
        cf = ops["cf"]
        add(2 * n + own, 2 * n + own, cf)
        add(2 * n + own, 2 * n + nei, -cf)
        add(2 * n + nei, 2 * n + nei, cf)
        add(2 * n + nei, 2 * n + own, -cf)
        add(2 * n + bc, 2 * n + bc, ops["cb_p"])
        rows, cols, vals = np.concatenate(R), np.concatenate(C), np.concatenate(V)
        N = 3 * n
        if getattr(self, "_bpat", None) is None or self._bpat[0] != len(rows):
            # sparsity pattern of the block matrix: fixed for a mesh, built once.
            # pos[i] = where COO entry i lands in the CSR data (duplicates summed)
            key = rows.astype(np.int64) * N + cols
            ukey, pos = np.unique(key, return_inverse=True)
            indptr = np.searchsorted(ukey // N, np.arange(N + 1)).astype(np.int64)
            self._bpat = (len(rows), pos, indptr, (ukey % N).astype(np.int64), len(ukey))
        _, pos, indptr, indices, nnz = self._bpat
        A = sp.csr_matrix((np.bincount(pos, vals, nnz), indices, indptr), (N, N))
        A.has_sorted_indices = True
        return A

    # ------------------------------------------------------------------
    def _pardiso(self, K, A, rhs, x0):
        if self.refactor_every <= 1:              # direct solve every iteration
            if self._lu is None:
                self._lu = PardisoLU()
            self._lu.factor(A)
            self.lin_stats["factorizations"] += 1
            return self._lu.solve(rhs)
        return self._pardiso_reuse(K, A, rhs, x0)

    def _pardiso_reuse(self, K, A, rhs, x0):
        """Direct LU (MKL PARDISO) of the block matrix, reused as the GMRES
        preconditioner for the following outer iterations: the matrix
        changes slowly, so an old factorization still solves the new system
        in a few Krylov iterations. Refactor after refactor_every iterations
        or when GMRES needed more than refactor_its iterations."""
        import pypardiso
        from scipy.sparse.linalg import LinearOperator, gmres
        A = A.tocsr()
        A.sum_duplicates()
        if self._lu is None or self._lu_age >= self.refactor_every:
            if self._lu is None:
                self._lu = pypardiso.PyPardisoSolver()
            else:
                self._lu.free_memory(everything=True)   # PARDISO keeps the old factors
            self._lu_A = A
            self._lu.factorize(A)
            self._lu_age = 0
            self.lin_stats["factorizations"] += 1
            fresh = True
        else:
            fresh = False
        self._lu_age += 1
        lu, A0 = self._lu, self._lu_A
        if fresh:
            x = lu.solve(A0, rhs)
            return x
        N = A.shape[0]
        M = LinearOperator((N, N), matvec=lambda r: lu.solve(A0, np.asarray(r).ravel()),
                           dtype=float)
        r0 = np.linalg.norm(rhs - K(x0))
        its = [0]
        x, info = gmres(LinearOperator((N, N), matvec=K, dtype=float), rhs, x0=x0, rtol=0.0,
                        atol=self.gmres_rtol * r0, restart=30, maxiter=3, M=M,
                        callback=lambda _: its.__setitem__(0, its[0] + 1),
                        callback_type="pr_norm")
        self.lin_stats["gmres_its"] += its[0]
        if its[0] > self.refactor_its:
            self._lu_age = self.refactor_every   # refactor next time
        return x


class PardisoLU:
    """MKL PARDISO for a matrix whose sparsity pattern never changes: the
    symbolic analysis (fill-reducing reordering, phase 11) runs once, every
    new set of values only needs the numeric factorization (phase 22)."""

    def __init__(self):
        import os
        import pypardiso
        from threadpoolctl import threadpool_limits
        # threads: all cores for one run; cores / jobs when the polar tool
        # runs several angles in parallel (CFD_JOBS), or CFD_PARDISO_THREADS
        n = int(os.environ.get("CFD_PARDISO_THREADS", "0")) or \
            max(1, (os.cpu_count() or 1) // max(1, int(os.environ.get("CFD_JOBS", "1"))))
        threadpool_limits(limits={"mkl": n})
        self.threads = n
        self.ps = pypardiso.PyPardisoSolver()
        self.pattern = None
        self.A = None

    def factor(self, A):
        ps = self.ps
        ps._check_A(A)
        key = (A.shape[0], A.nnz)
        z = np.zeros((A.shape[0], 1))
        if self.pattern != key:
            if self.pattern is not None:
                ps.free_memory(everything=True)
            ps.set_phase(11)
            ps._call_pardiso(A, z)
            self.pattern = key
        ps.set_phase(22)
        ps._call_pardiso(A, z)
        self.A = A

    def solve(self, b):
        self.ps.set_phase(33)
        return self.ps._call_pardiso(self.A, np.asfortranarray(b, dtype=float))

    def __del__(self):
        try:
            self.ps.free_memory(everything=True)
        except Exception:
            pass
