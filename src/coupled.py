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

import kernels
from simple_solver import SimpleSolver


class CoupledSolver(SimpleSolver):
    """Same constructor as SimpleSolver; alpha_u is the momentum under-
    relaxation of the coupled system (can be close to 1)."""

    def __init__(self, *a, linear="gmres", gmres_rtol=0.05, **k):
        k.setdefault("alpha_u", 0.95)
        super().__init__(*a, **k)
        if not self.has_outlet:
            raise NotImplementedError("coupled solver: needs an outlet (pressure level)")
        self.coupled_linear = linear
        self.gmres_rtol = gmres_rtol
        self._ml, self._ml_age = None, 0
        self.lin_stats = {"gmres_its": 0, "amg_setups": 0}
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
        if self.turb is not None:
            self.gU = [m.grad(self.U[:, k], self._boundary_values()[0][:, k]) for k in range(2)]
            res["nuT"] = self.turb.solve()
        self.history.append(res)
        return res

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
        return sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                             (3 * n, 3 * n))
