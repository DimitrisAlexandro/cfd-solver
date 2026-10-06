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
  {'type': 'symmetry'}                   slip wall: no normal flow, no shear
  {'type': 'inlet',  'U': (1, 0)}        same as wall numerically; U may be
                                         a function f(x, y) -> (n,2)
  {'type': 'outlet'}                     zero-gradient U, fixed p = 0
If there is no outlet, pressure is pinned in one cell (cavity).

Turbulence: turbulence="sa" adds the Spalart-Allmaras model (turbulence.py);
the momentum equation then uses nu + nu_t (incl. the transpose stress term).

Time-accurate (unsteady) mode: call advance(dt) instead of solve().
  * BDF2 (second order, implicit); the very first step is backward Euler
  * PIMPLE-style: each time step runs outer SIMPLEC iterations until the
    residuals drop below tol, so the solution is time-accurate at every step
  * transient Rhie-Chow correction (Choi 1999): the answer does not depend
    on the time step through the pressure-smoothing term
  * time-dependent boundary values: 'U': f(x, y, t) -> (n,2)

Linear systems are solved with SciPy's sparse direct solver (SuperLU):
writing a competitive sparse linear solver is its own project.
"""

import inspect

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from linsolve import LinearSolver, csr
import kernels

# "iterative" (Gauss-Seidel + algebraic multigrid, step D) or "direct" (sparse LU)
DEFAULT_LINEAR_SOLVER = "iterative"


class SimpleSolver:
    def __init__(self, mesh, bcs, nu, alpha_u=0.9, alpha_p=None, second_order=True,
                 U0=(0.0, 0.0), simplec=True, turbulence=None, nut_inf_ratio=3.0,
                 alpha_turb=0.9, limiter="auto", limiter_eps=0.1, limiter_cells="tri",
                 linear_solver=None):
        """limiter: gradient limiter of the linear-upwind convection
        (deferred correction): None or "venkat" (Venkatakrishnan). "auto" =
        "venkat" for turbulent runs (high Re: no physical viscosity outside
        the boundary layer damps overshoots), None for laminar runs.
        limiter_cells: "tri" limits only triangles (the unstructured region,
        where the wiggles appear; limiting the structured boundary-layer
        quads adds needless dissipation: NACA 0012 alpha 10, Cd +6 %), "all".
        limiter_eps: Venkatakrishnan threshold as a fraction of the velocity
        scale, eps = limiter_eps * U_ref (U_ref = max |U| on fixed-velocity
        boundaries). Overshoots well below eps are not limited, so smooth
        extrema (suction peak) keep second order; the standard (K h)^3
        threshold shrinks with the cell size and clipped the suction peak on
        fine grids (NACA 0012 alpha 10: Cd rose with refinement)."""
        self.m, self.nu = mesh, nu
        self.lin = LinearSolver(linear_solver or DEFAULT_LINEAR_SOLVER)
        self._pattern = None                  # fixed CSR pattern (kernels.CsrPattern)
        self.limiter = ("venkat" if turbulence else None) if limiter == "auto" else limiter
        self.limiter_eps = limiter_eps
        # cells where the limiter acts (None = all)
        self.limiter_mask = (mesh.nverts == 3) if limiter_cells == "tri" else None
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
        self.pb_out = np.zeros(nb)            # outlet static pressure per face
        self.b_wall = np.zeros(nb, bool)
        self.b_inlet = np.zeros(nb, bool)
        self.b_sym = np.zeros(nb, bool)
        self.Ub = np.zeros((nb, 2))
        self._time_bcs = []                    # (face indices, f(x, y, t))
        for name, bc in bcs.items():
            idx = m.patch_faces[name]
            if bc['type'] == 'symmetry':
                # slip wall: no normal velocity, no shear, zero-gradient p
                self.b_fixedU[idx] = True       # Dirichlet with Ub = tangential part of U_P
                self.b_sym[idx] = True
                continue
            if bc['type'] in ('wall', 'inlet'):
                self.b_fixedU[idx] = True
                (self.b_wall if bc['type'] == 'wall' else self.b_inlet)[idx] = True
                U = bc.get('U', (0.0, 0.0))
                if callable(U) and len(inspect.signature(U).parameters) == 3:
                    self._time_bcs.append((idx, U))
                    self.Ub[idx] = U(m.bxf[idx, 0], m.bxf[idx, 1], 0.0)
                else:
                    self.Ub[idx] = U(m.bxf[idx, 0], m.bxf[idx, 1]) if callable(U) else U
            elif bc['type'] == 'outlet':
                self.b_outlet[idx] = True
                self.pb_out[idx] = bc.get('p', 0.0)
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
        # transient state (inactive until advance() is called)
        self.dt = None
        self.time = 0.0
        self.step = 0
        self.U_n = self.U_nm1 = None           # velocity at time levels n, n-1
        self.F_n = self.F_nm1 = None           # face fluxes at n, n-1
        self.UfS_n = self.UfS_nm1 = None       # interpolated face velocity . S at n, n-1
        # turbulence model
        self.turb = None
        if turbulence == "sa":
            from turbulence import SpalartAllmaras
            if not self.b_wall.any():
                raise ValueError("the SA model needs at least one wall patch")
            self.turb = SpalartAllmaras(self, nut_inf_ratio=nut_inf_ratio, alpha=alpha_turb)
        elif turbulence not in (None, "laminar"):
            raise ValueError(f"unknown turbulence model {turbulence}")

    # ------------------------------------------------------------------
    def _boundary_values(self):
        m = self.m
        # outlets: zero NORMAL gradient -> keep the variation along the face
        c = m.bcell
        Uext = self.U[c] + np.c_[np.einsum('ij,ij->i', self.gU[0][c], m.db_t),
                                 np.einsum('ij,ij->i', self.gU[1][c], m.db_t)]
        if self.b_sym.any():                   # symmetry: tangential part of the cell value
            sy = self.b_sym
            nrm = m.bSf[sy] / np.linalg.norm(m.bSf[sy], axis=1)[:, None]
            Uc = self.U[c[sy]]
            self.Ub[sy] = Uc - np.einsum('ij,ij->i', Uc, nrm)[:, None] * nrm
        Ub = np.where(self.b_fixedU[:, None], self.Ub, Uext)
        # pressure: extrapolated (lagged gradient) except fixed at outlets
        pb = self.p[m.bcell] + np.einsum('ij,ij->i', self.gradp[m.bcell], m.db)
        pb = np.where(self.b_outlet, self.pb_out, pb)
        return Ub, pb

    def _momentum(self, Ub, gU):
        """Momentum equations (both components, before under-relaxation):
        diag, off-diagonals a_on (owner row) / a_no (neighbour row), source
        incl. the pressure gradient -grad(p) V, time terms. Also returns
        whether the compiled kernels were used and the face viscosities."""
        m, nu, n = self.m, self.nu, self.m.ncells
        own, nei, bc = m.own, m.nei, m.bcell
        # ---- momentum matrix ------------------------------------------------
        F, Fb = self.F, self.Fb
        if self.turb is not None:              # molecular + eddy viscosity on faces
            nu_f, nu_b = self.turb.face_viscosity()
        else:
            nu_f, nu_b = nu, np.full(len(bc), nu)
        fast = kernels.USE_NUMBA
        if fast and self._pattern is None:
            self._pattern = kernels.CsrPattern(n, own, nei)
        Db = nu_b * m.bcoef
        if fast:
            nu_fa = np.broadcast_to(np.asarray(nu_f, float), F.shape)
            if self.second_order and self.limiter is not None:
                gk = [gU[k] * self._venkat(self.U[:, k], gU[k], Ub[:, k])[:, None]
                      for k in range(2)]
            else:
                gk = gU
            diag, a_on, a_no, src, tr = kernels.momentum_coeffs(
                n, own, nei, F, np.ascontiguousarray(nu_fa), m.coef, m.w, m.Sf, m.Delta,
                m.xc, m.xf, gU[0], gU[1], gk[0], gk[1], bool(self.second_order))
            if self.turb is not None:
                src += tr
            diag += np.bincount(bc, np.where(self.b_fixedU, Db, 0.0) +
                                np.where(self.b_outlet, np.maximum(Fb, 0.0), 0.0), n)
            src -= self.gradp * m.vol[:, None]
            for k in range(2):
                bsrc = np.where(self.b_fixedU, Db * Ub[:, k] - Fb * Ub[:, k] +
                                nu_b * np.einsum('ij,ij->i', gU[k][bc], m.bSf - m.bDelta), 0.0)
                bsrc += np.where(self.b_outlet,
                                 -np.maximum(Fb, 0.0) * np.einsum('ij,ij->i', gU[k][bc], m.db_t)
                                 - np.minimum(Fb, 0.0) * Ub[:, k], 0.0)
                if self.turb is not None:
                    gTb = np.c_[gU[0][bc, k], gU[1][bc, k]]
                    bsrc += np.where(self.b_outlet, 0.0, nu_b) * np.einsum('ij,ij->i', gTb, m.bSf)
                src[:, k] += np.bincount(bc, bsrc, n)
        else:
            D = nu_f * m.coef
            a_on = -D - np.maximum(-F, 0.0)        # coefficient of neighbour in owner row
            a_no = -D - np.maximum(F, 0.0)         # coefficient of owner in neighbour row
            diag = np.bincount(own, D + np.maximum(F, 0.0), n) + \
                np.bincount(nei, D + np.maximum(-F, 0.0), n)
            diag += np.bincount(bc, np.where(self.b_fixedU, Db, 0.0) +
                                np.where(self.b_outlet, np.maximum(Fb, 0.0), 0.0), n)
            src = np.zeros((n, 2))
            src -= self.gradp * m.vol[:, None]
            up = F >= 0
            U_cell = np.where(up, own, nei)
            for k in range(2):
                gf = m.interp(gU[k])
                # explicit non-orthogonal diffusion correction
                c = nu_f * np.einsum('ij,ij->i', gf, m.Sf - m.Delta)
                s = m.sum_to_cells(c)
                # boundary: fixed value (diffusion + inflow/outflow convection)
                bsrc = np.where(self.b_fixedU, Db * Ub[:, k] - Fb * Ub[:, k] +
                                nu_b * np.einsum('ij,ij->i', gU[k][bc], m.bSf - m.bDelta), 0.0)
                # outlet backflow (Fb<0) treated explicitly
                # outlet: phi_b = phi_P + grad(phi_P).db_t ; phi_P part implicit
                bsrc += np.where(self.b_outlet,
                                 -np.maximum(Fb, 0.0) * np.einsum('ij,ij->i', gU[k][bc], m.db_t)
                                 - np.minimum(Fb, 0.0) * Ub[:, k], 0.0)
                s += np.bincount(bc, bsrc, n)
                if self.turb is not None:
                    # transpose part of the viscous stress, div(nu_eff (grad U)^T):
                    # zero for constant viscosity, not for a varying eddy viscosity
                    gTf = np.c_[m.interp(gU[0][:, k]), m.interp(gU[1][:, k])]
                    s += m.sum_to_cells(nu_f * np.einsum('ij,ij->i', gTf, m.Sf))
                    gTb = np.c_[gU[0][bc, k], gU[1][bc, k]]
                    s += np.bincount(bc, np.where(self.b_outlet, 0.0, nu_b) *
                                     np.einsum('ij,ij->i', gTb, m.bSf), n)
                if self.second_order:
                    # deferred correction: F * (phi_linear_upwind - phi_upwind)
                    xU = m.xc[U_cell]
                    gk = gU[k] if self.limiter is None else \
                        gU[k] * self._venkat(self.U[:, k], gU[k], Ub[:, k])[:, None]
                    corr = F * np.einsum('ij,ij->i', gk[U_cell], m.xf - xU)
                    s -= m.sum_to_cells(corr)
                src[:, k] += s

        if self.dt is not None:                # BDF2 (Euler on the first step)
            c1, c2, c3 = (1.0, 1.0, 0.0) if self.U_nm1 is None else (1.5, 2.0, 0.5)
            diag += c1 * m.vol / self.dt
            Uold = c2 * self.U_n - (c3 * self.U_nm1 if c3 else 0.0)
            src += (m.vol / self.dt)[:, None] * Uold

        return diag, a_on, a_no, src, fast, nu_f, nu_b, Db

    def iterate(self):
        m, nu, n = self.m, self.nu, self.m.ncells
        own, nei, bc = m.own, m.nei, m.bcell
        Ub, pb = self._boundary_values()
        gU = [m.grad(self.U[:, k], Ub[:, k]) for k in range(2)]
        self.gU = gU
        # conservative (Green-Gauss) pressure gradient, skew-corrected with
        # the previous iteration's gradient
        self.gradp = m.grad_gg(self.p, pb, grad_skew=self.gradp)

        diag, a_on, a_no, src, fast, nu_f, nu_b, Db = self._momentum(Ub, gU)
        F, Fb = self.F, self.Fb

        diag = diag / self.au
        src += ((1 - self.au) * diag)[:, None] * self.U
        if fast:
            A = self._pattern.matrix(diag, a_on, a_no)
        else:
            A = csr(np.r_[diag, a_on, a_no], np.r_[np.arange(n), own, nei],
                    np.r_[np.arange(n), nei, own], n)
        # momentum residual (before solving), normalised
        # (both components normalised by |U|: a component that is ~0
        #  everywhere would otherwise look unconverged forever)
        scale = max((diag * np.linalg.norm(self.U, axis=1)).sum(), 1e-30)
        resU = [np.abs(A @ self.U[:, k] - src[:, k]).sum() / scale for k in range(2)]
        Ustar = np.c_[tuple(self.lin.scalars(A, [src[:, 0], src[:, 1]],
                                             [self.U[:, 0], self.U[:, 1]]))]

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
        if self.dt is not None:
            # transient Rhie-Chow correction: old-time contributions use the old
            # FACE fluxes, not interpolated cell velocities (Choi 1999)
            if self.U_nm1 is None:
                old = self.F_n - self.UfS_n
            else:
                old = (2.0 * self.F_n - 0.5 * self.F_nm1) - (2.0 * self.UfS_n - 0.5 * self.UfS_nm1)
            Fstar += Df / self.dt * old

        Fbstar = np.einsum('ij,ij->i', Ub, m.bSf)
        # pressure-correction coefficient: SIMPLE uses V/a_P, SIMPLEC uses
        # V/(a_P - sum|a_nb|), i.e. it also accounts for the neighbours'
        # velocity corrections -> no pressure under-relaxation needed
        if self.simplec:
            # a_P - sum|a_nb| = (1 - alpha_u) a_P + (net outflow + boundary terms);
            # cells with a net convective INFLOW (not yet mass-conserving, e.g.
            # wall cells of a uniform initial field) would make it ~0 and the
            # velocity correction explode -> floor at the ideal (1 - alpha_u) a_P.
            # (The converged solution does not depend on Dc; Rhie-Chow uses V/a_P.)
            nbsum = np.bincount(own, -a_on, n) + np.bincount(nei, -a_no, n)
            Dc = m.vol / np.maximum(diag - nbsum, (1.0 - self.au) * diag)
        else:
            Dc = Dp
        Dcf = m.interp(Dc)
        cb = Dc[bc] * m.bcoef
        if self.has_outlet:
            o = self.b_outlet
            Ustar_b = Ustar[bc[o]] + np.c_[np.einsum('ij,ij->i', gU[0][bc[o]], m.db_t[o]),
                                           np.einsum('ij,ij->i', gU[1][bc[o]], m.db_t[o])]
            Fbstar[o] = np.einsum('ij,ij->i', Ustar_b, m.bSf[o]) - Dp[bc[o]] * (
                (self.pb_out[o] - self.p[bc[o]]) * m.bcoef[o] -
                np.einsum('ij,ij->i', self.gradp[bc[o]], m.bDelta[o]))

        # ---- pressure correction ---------------------------------------------
        cf = Dcf * m.coef
        pdiag = np.bincount(own, cf, n) + np.bincount(nei, cf, n) + \
            np.bincount(bc, np.where(self.b_outlet, cb, 0.0), n)
        imbalance = m.sum_to_cells(Fstar, Fbstar)
        rhs = -imbalance
        if fast and self.has_outlet:
            Ap = self._pattern.matrix(pdiag, -cf, -cf)
        else:
            rows = np.r_[np.arange(n), own, nei]
            cols = np.r_[np.arange(n), nei, own]
            vals = np.r_[pdiag, -cf, -cf]
            if not self.has_outlet:          # pin p' in cell 0
                keep = ~(((rows == 0) | (cols == 0)) & (rows != cols))
                rows, cols, vals = rows[keep], cols[keep], vals[keep]
                rhs[0] = 0.0
            Ap = csr(vals, rows, cols, n)
        pc = self.lin.pressure(Ap, rhs)

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
        if self.turb is not None:
            self.gU = [m.grad(self.U[:, k], self._boundary_values()[0][:, k]) for k in range(2)]
            res["nuT"] = self.turb.solve()
        self.history.append(res)
        return res

    def _venkat(self, phi, g, phib):
        """Venkatakrishnan (1995) limiter psi in [0, 1] per cell for the
        reconstruction phi_P + g.(x_f - x_P) on every face of the cell."""
        m = self.m
        if kernels.USE_NUMBA:
            uref = max(np.abs(self.Ub[self.b_fixedU]).max() if self.b_fixedU.any() else 0.0,
                       1e-30)
            return kernels.venkat(m, phi, g, phib, (self.limiter_eps * uref) ** 2,
                                  self.limiter_mask)
        own, nei, bc = m.own, m.nei, m.bcell
        n = m.ncells
        pmax = phi.copy(); pmin = phi.copy()
        np.maximum.at(pmax, own, phi[nei]); np.maximum.at(pmax, nei, phi[own])
        np.minimum.at(pmin, own, phi[nei]); np.minimum.at(pmin, nei, phi[own])
        np.maximum.at(pmax, bc, phib); np.minimum.at(pmin, bc, phib)
        uref = max(np.abs(self.Ub[self.b_fixedU]).max() if self.b_fixedU.any() else 0.0, 1e-30)
        eps2 = np.full(n, (self.limiter_eps * uref) ** 2)
        psi = np.ones(n)
        for c, x in ((own, m.xf), (nei, m.xf), (bc, m.bxf)):
            d2 = np.einsum('ij,ij->i', g[c], x - m.xc[c])
            d1 = np.where(d2 > 0, pmax[c] - phi[c], pmin[c] - phi[c])
            e = eps2[c]
            num = (d1**2 + e) * d2 + 2 * d2**2 * d1
            den = d2 * (d1**2 + 2 * d2**2 + d1 * d2 + e)
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where(np.abs(d2) > 1e-30, num / den, 1.0)
            if self.limiter_mask is not None:            # psi = 1 outside the mask
                f = np.where(self.limiter_mask[c], f, 1.0)
            np.minimum.at(psi, c, np.clip(f, 0.0, 1.0))
        return psi

    def solve(self, tol=1e-6, max_iter=5000, verbose=True, every=100, callback=None):
        """callback(it, residuals) is called after every iteration (monitoring,
        live history files); if it returns True the run counts as converged
        and stops (e.g. force-based convergence). Sets self.converged."""
        t0 = time.time()
        self.converged = False
        for it in range(1, max_iter + 1):
            r = self.iterate()
            if callback is not None and callback(it, r) is True:
                if verbose:
                    print(f"  converged (callback criterion) in {it} iterations "
                          f"({time.time() - t0:.1f}s)")
                self.converged = True
                return it
            if verbose and (it % every == 0 or it == 1):
                extra = f"  nuT {r['nuT']:.2e}" if "nuT" in r else ""
                print(f"  it {it:5d}  cont {r['cont']:.2e}  Ux {r['Ux']:.2e}  "
                      f"Uy {r['Uy']:.2e}{extra}  ({time.time() - t0:.1f}s)")
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
    # unsteady time stepping
    # ------------------------------------------------------------------
    def set_time_bcs(self, t):
        m = self.m
        for idx, f in self._time_bcs:
            self.Ub[idx] = f(m.bxf[idx, 0], m.bxf[idx, 1], t)

    def advance(self, dt, max_outer=30, tol=1e-6):
        """One implicit time step of size dt. Returns (outer iterations, residuals)."""
        m = self.m
        if self.dt is not None and abs(dt - self.dt) > 1e-12 * dt:
            self.U_nm1 = self.F_nm1 = self.UfS_nm1 = None   # BDF2 needs a constant dt
        self.dt = dt
        UfS = np.einsum('ij,ij->i', np.c_[m.interp_skew(self.U[:, 0], self.gU[0]),
                                          m.interp_skew(self.U[:, 1], self.gU[1])], m.Sf)
        if self.U_n is not None:
            self.U_nm1, self.F_nm1, self.UfS_nm1 = self.U_n, self.F_n, self.UfS_n
        if self.turb is not None:
            self.turb.time_levels()
        self.U_n, self.F_n, self.UfS_n = self.U.copy(), self.F.copy(), UfS
        self.time += dt
        self.set_time_bcs(self.time)
        for k in range(1, max_outer + 1):
            r = self.iterate()
            if max(r.values()) < tol:
                break
        self.step += 1
        return k, r

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

    def set_initial(self, U, p=None, nuTilde=None):
        """Start from given cell fields (e.g. the solution at a neighbouring
        angle of attack, interpolated): sets U, p (and SA nuTilde) and the
        face fluxes consistent with them."""
        m = self.m
        self.U = np.array(U, float)
        if p is not None:
            self.p = np.array(p, float)
        self.F = np.einsum('ij,ij->i', m.interp(self.U), m.Sf)
        self.Fb = np.einsum('ij,ij->i', np.where(self.b_fixedU[:, None], self.Ub,
                                                 self.U[m.bcell]), m.bSf)
        self.Ubar_S_old = self.F.copy()
        if nuTilde is not None and self.turb is not None:
            self.turb.nuT = np.maximum(np.array(nuTilde, float), 0.0)

    def moment(self, patch, ref=(0.25, 0.0)):
        """z-moment (counter-clockwise positive) of the fluid force on the
        patch about the point ref, per unit depth."""
        idx, pb, tau, S, Smag = self.wall_values(patch)
        F = pb[:, None] * S + tau * Smag[:, None]
        r = self.m.bxf[idx] - np.asarray(ref, float)
        return float((r[:, 0] * F[:, 1] - r[:, 1] * F[:, 0]).sum())

    def force(self, patch):
        """Force of the fluid on the patch (per unit depth): (pressure, viscous)."""
        _, pb, tau, S, Smag = self.wall_values(patch)
        Fp = (pb[:, None] * S).sum(0)
        Fv = (tau * Smag[:, None]).sum(0)
        return Fp, Fv

