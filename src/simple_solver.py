"""
Steady incompressible Navier-Stokes solver: cell-centred finite volume,
collocated variables, SIMPLE pressure-velocity coupling.

    div(U U) - div(nu grad U) = -grad p        (density = 1)
    div(U) = 0

Discretisation
  * convection : upwind implicit + deferred correction to linear-upwind
                 (second order)
  * diffusion  : over-relaxed orthogonal part implicit, non-orthogonal
                 correction explicit
  * gradients  : weighted least squares (fvmesh.py)
  * face fluxes: Rhie-Chow interpolation (+ Majumdar under-relaxation fix,
                 so the converged answer does not depend on alpha_u)

Boundary conditions, per patch:
  {'type': 'wall',   'U': (0, 0)}        fixed velocity, zero-gradient p
  {'type': 'inlet',  'U': (1, 0)}        same as wall numerically; U may be
                                         a function f(x, y) -> (n,2)
  {'type': 'outlet'}                     zero-gradient U, fixed p = 0
If there is no outlet, pressure is pinned in one cell (cavity).

Linear systems are solved with SciPy's sparse direct solver (SuperLU):
writing a competitive sparse linear solver is its own project.
"""

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


class SimpleSolver:
    def __init__(self, mesh, bcs, nu, alpha_u=0.8, alpha_p=None, second_order=True,
                 U0=(0.0, 0.0), simplec=True):
        self.m, self.nu = mesh, nu
        # SIMPLEC needs no pressure under-relaxation; SIMPLE uses 1 - alpha_u
        self.simplec = simplec
        self.au = alpha_u
        self.ap = alpha_p if alpha_p is not None else (1.0 if simplec else 1.0 - alpha_u)
        self.second_order = second_order
        m = mesh
        n = m.ncells

        # --- boundary classification -----------------------------------
        nb = len(m.bcell)
        self.b_fixedU = np.zeros(nb, bool)
        self.b_outlet = np.zeros(nb, bool)
        self.Ub = np.zeros((nb, 2))
        for name, bc in bcs.items():
            idx = m.patch_faces[name]
            if bc['type'] in ('wall', 'inlet'):
                self.b_fixedU[idx] = True
                U = bc.get('U', (0.0, 0.0))
                self.Ub[idx] = U(m.bxf[idx, 0], m.bxf[idx, 1]) if callable(U) else U
            elif bc['type'] == 'outlet':
                self.b_outlet[idx] = True
            else:
                raise ValueError(bc['type'])
        missing = set(m.patch_names) - set(bcs)
        if missing:
            raise ValueError(f"no boundary condition for patches {missing}")
        self.has_outlet = self.b_outlet.any()

        # --- fields ---------------------------------------------------------
        self.U = np.tile(np.asarray(U0, float), (n, 1))
        self.p = np.zeros(n)
        self.F = np.einsum('ij,ij->i', m.interp(self.U), m.Sf)          # internal
        self.Fb = np.einsum('ij,ij->i', np.where(self.b_fixedU[:, None], self.Ub,
                                                 self.U[m.bcell]), m.bSf)
        self.Ubar_S_old = self.F.copy()
        self.gradp = np.zeros((n, 2))
        self.gU = [np.zeros((n, 2)), np.zeros((n, 2))]
        self.history = []

    # ------------------------------------------------------------------
    def _boundary_values(self):
        m = self.m
        # outlets: zero NORMAL gradient -> keep the variation along the face
        c = m.bcell
        Uext = self.U[c] + np.c_[np.einsum('ij,ij->i', self.gU[0][c], m.db_t),
                                 np.einsum('ij,ij->i', self.gU[1][c], m.db_t)]
        Ub = np.where(self.b_fixedU[:, None], self.Ub, Uext)
        # pressure: extrapolated (lagged gradient) except fixed at outlets
        pb = self.p[m.bcell] + np.einsum('ij,ij->i', self.gradp[m.bcell], m.db)
        pb = np.where(self.b_outlet, 0.0, pb)
        return Ub, pb

    def iterate(self):
        m, nu, n = self.m, self.nu, self.m.ncells
        own, nei, bc = m.own, m.nei, m.bcell
        Ub, pb = self._boundary_values()
        gU = [m.grad(self.U[:, k], Ub[:, k]) for k in range(2)]
        self.gU = gU
        # conservative (Green-Gauss) pressure gradient, skew-corrected with
        # the previous iteration's gradient
        self.gradp = m.grad_gg(self.p, pb, grad_skew=self.gradp)

        # ---- momentum matrix ------------------------------------------------
        F, Fb = self.F, self.Fb
        D = nu * m.coef
        a_on = -D - np.maximum(-F, 0.0)        # coefficient of neighbour in owner row
        a_no = -D - np.maximum(F, 0.0)         # coefficient of owner in neighbour row
        diag = np.bincount(own, D + np.maximum(F, 0.0), n) + \
            np.bincount(nei, D + np.maximum(-F, 0.0), n)
        Db = nu * m.bcoef
        diag += np.bincount(bc, np.where(self.b_fixedU, Db, 0.0) +
                            np.where(self.b_outlet, np.maximum(Fb, 0.0), 0.0), n)

        src = np.zeros((n, 2))
        src -= self.gradp * m.vol[:, None]
        up = F >= 0
        U_cell = np.where(up, own, nei)
        for k in range(2):
            gf = m.interp(gU[k])
            # explicit non-orthogonal diffusion correction
            c = nu * np.einsum('ij,ij->i', gf, m.Sf - m.Delta)
            s = m.sum_to_cells(c)
            # boundary: fixed value (diffusion + inflow/outflow convection)
            bsrc = np.where(self.b_fixedU, Db * Ub[:, k] - Fb * Ub[:, k] +
                            nu * np.einsum('ij,ij->i', gU[k][bc], m.bSf - m.bDelta), 0.0)
            # outlet backflow (Fb<0) treated explicitly
            # outlet: phi_b = phi_P + grad(phi_P).db_t ; phi_P part implicit
            bsrc += np.where(self.b_outlet,
                             -np.maximum(Fb, 0.0) * np.einsum('ij,ij->i', gU[k][bc], m.db_t)
                             - np.minimum(Fb, 0.0) * Ub[:, k], 0.0)
            s += np.bincount(bc, bsrc, n)
            if self.second_order:
                # deferred correction: F * (phi_linear_upwind - phi_upwind)
                xU = m.xc[U_cell]
                corr = F * np.einsum('ij,ij->i', gU[k][U_cell], m.xf - xU)
                s -= m.sum_to_cells(corr)
            src[:, k] += s

        a0 = diag.copy()
        diag = diag / self.au
        src += ((1 - self.au) * diag)[:, None] * self.U
        A = sp.csc_matrix((np.r_[diag, a_on, a_no], (np.r_[np.arange(n), own, nei],
                                                     np.r_[np.arange(n), nei, own])), (n, n))
        lu = splu(A)
        # momentum residual (before solving), normalised
        # (both components normalised by |U|: a component that is ~0
        #  everywhere would otherwise look unconverged forever)
        scale = max((diag * np.linalg.norm(self.U, axis=1)).sum(), 1e-30)
        resU = [np.abs(A @ self.U[:, k] - src[:, k]).sum() / scale for k in range(2)]
        Ustar = np.c_[lu.solve(src[:, 0]), lu.solve(src[:, 1])]

        # ---- Rhie-Chow face fluxes -------------------------------------------
        Dp = m.vol / diag                                   # V / a_P (relaxed)
        Df = m.interp(Dp)
        gp_f = m.interp(self.gradp)
        # face-centre velocity (skewness-corrected, lagged gradients)
        Uf = np.c_[m.interp_skew(Ustar[:, 0], gU[0]), m.interp_skew(Ustar[:, 1], gU[1])]
        UbarS = np.einsum('ij,ij->i', Uf, m.Sf)
        Fstar = UbarS - Df * ((self.p[nei] - self.p[own]) * m.coef -
                              np.einsum('ij,ij->i', gp_f, m.Delta))
        Fstar += (1 - self.au) * (self.F - self.Ubar_S_old)  # Majumdar correction
        self.Ubar_S_old = UbarS

        Fbstar = np.einsum('ij,ij->i', Ub, m.bSf)
        # pressure-correction coefficient: SIMPLE uses V/a_P, SIMPLEC uses
        # V/(a_P - sum|a_nb|), i.e. it also accounts for the neighbours'
        # velocity corrections -> no pressure under-relaxation needed
        if self.simplec:
            nbsum = np.bincount(own, -a_on, n) + np.bincount(nei, -a_no, n)
            Dc = m.vol / np.maximum(diag - nbsum, 1e-12 * diag)
        else:
            Dc = Dp
        Dcf = m.interp(Dc)
        cb = Dc[bc] * m.bcoef
        if self.has_outlet:
            o = self.b_outlet
            Ustar_b = Ustar[bc[o]] + np.c_[np.einsum('ij,ij->i', gU[0][bc[o]], m.db_t[o]),
                                           np.einsum('ij,ij->i', gU[1][bc[o]], m.db_t[o])]
            Fbstar[o] = np.einsum('ij,ij->i', Ustar_b, m.bSf[o]) - Dp[bc[o]] * (
                (0.0 - self.p[bc[o]]) * m.bcoef[o] -
                np.einsum('ij,ij->i', self.gradp[bc[o]], m.bDelta[o]))

        # ---- pressure correction ---------------------------------------------
        cf = Dcf * m.coef
        pdiag = np.bincount(own, cf, n) + np.bincount(nei, cf, n) + \
            np.bincount(bc, np.where(self.b_outlet, cb, 0.0), n)
        imbalance = m.sum_to_cells(Fstar, Fbstar)
        rhs = -imbalance
        rows = np.r_[np.arange(n), own, nei]
        cols = np.r_[np.arange(n), nei, own]
        vals = np.r_[pdiag, -cf, -cf]
        if not self.has_outlet:          # pin p' in cell 0
            keep = ~(((rows == 0) | (cols == 0)) & (rows != cols))
            rows, cols, vals = rows[keep], cols[keep], vals[keep]
            rhs[0] = 0.0
        Ap = sp.csc_matrix((vals, (rows, cols)), (n, n))
        pc = splu(Ap).solve(rhs)

        # ---- corrections -------------------------------------------------------
        self.F = Fstar - cf * (pc[nei] - pc[own])
        Fb_new = Fbstar.copy()
        o = self.b_outlet
        Fb_new[o] = Fbstar[o] - cb[o] * (0.0 - pc[bc[o]])
        self.Fb = Fb_new
        pcb = np.where(self.b_outlet, 0.0, pc[bc])
        gpc = m.grad_gg(pc, pcb)          # same operator as the momentum eq.
        self.U = Ustar - Dc[:, None] * gpc
        self.p = self.p + self.ap * pc
        if not self.has_outlet:
            self.p -= self.p[0]

        # continuity residual relative to the throughflow (or, in a closed
        # box, to a typical face flux times sqrt(N) ~ U*L)
        Fref = np.abs(self.Fb).sum() if np.abs(self.Fb).sum() > 0 else \
            np.abs(self.F).sum() / np.sqrt(n)
        Fref = max(Fref, 1e-30)
        res = {"cont": np.abs(imbalance).sum() / Fref, "Ux": resU[0], "Uy": resU[1]}
        self.history.append(res)
        return res

    def solve(self, tol=1e-6, max_iter=5000, verbose=True, every=100, callback=None):
        """callback(it, residuals) is called after every iteration (monitoring,
        live history files). Sets self.converged."""
        t0 = time.time()
        self.converged = False
        for it in range(1, max_iter + 1):
            r = self.iterate()
            if callback is not None:
                callback(it, r)
            if verbose and (it % every == 0 or it == 1):
                print(f"  it {it:5d}  cont {r['cont']:.2e}  Ux {r['Ux']:.2e}  "
                      f"Uy {r['Uy']:.2e}  ({time.time() - t0:.1f}s)")
            if max(r.values()) < tol:
                if verbose:
                    print(f"  converged in {it} iterations ({time.time() - t0:.1f}s), "
                          f"residuals < {tol:g}")
                self.converged = True
                return it
        if verbose:
            print(f"  NOT converged after {max_iter} iterations: {r}")
        return max_iter

    # ------------------------------------------------------------------
    # post-processing
    # ------------------------------------------------------------------
    def wall_values(self, patch):
        """Pressure and wall shear on a wall patch."""
        m = self.m
        idx = m.patch_faces[patch]
        c = m.bcell[idx]
        pb = self.p[c] + np.einsum('ij,ij->i', self.gradp[c], m.db[idx])
        S = m.bSf[idx]
        Smag = np.linalg.norm(S, axis=1)
        n_out = S / Smag[:, None]                      # points out of fluid (into body)
        dn = -np.einsum('ij,ij->i', m.db[idx], n_out)  # wall distance of cell centre
        du = self.U[c] - self.Ub[idx]
        du_t = du - np.einsum('ij,ij->i', du, n_out)[:, None] * n_out
        tau = self.nu * du_t / np.abs(dn)[:, None]     # shear stress vector on wall (x|S|)
        return idx, pb, tau, S, Smag

    def force(self, patch):
        """Force of the fluid on the patch (per unit depth): (pressure, viscous)."""
        _, pb, tau, S, Smag = self.wall_values(patch)
        Fp = (pb[:, None] * S).sum(0)
        Fv = (tau * Smag[:, None]).sum(0)
        return Fp, Fv
