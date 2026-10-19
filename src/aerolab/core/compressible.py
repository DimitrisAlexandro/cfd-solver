"""
Phase 5: compressible (all-speed) pressure-based solver.

Steady compressible Navier-Stokes / RANS (ideal gas), same mesh, same
discretisation family and boundary handling as the incompressible
SimpleSolver:

    div(rho U)                                   = 0
    div(rho U U) - div(tau)                      = -grad p
    div(rho U H) - div(Gamma grad H
                       + (mu_eff - Gamma) grad K) = 0        H = cp T + K, K = |U|^2/2
    p_abs = rho R T,  mu = Sutherland(T)

Non-dimensional: rho_inf = 1, |U_inf| = 1, T_inf = 1, chord = 1, so
p_inf,abs = P0 = 1/(gamma M^2), R = P0, cp = 1/((gamma-1) M^2), mu_inf = 1/Re.
The solver works with the GAUGE pressure p = p_abs - P0 (as the
incompressible solver), so Cp = p / (1/2) unchanged.

Pressure-velocity-density coupling (SIMPLEC, compressible form,
Demirdzic, Lilek & Peric 1993; Moukalled, Mangani & Darwish 2016 ch. 16):
  mass flux  m_f = rho_f [ U*_f.S - D_f (grad p)_f-correction ]   (Rhie-Chow +
                   Majumdar, as incompressible)
  correction m'_f = rho_f U'_f.S + rho'_f U*_f.S ,  U' = -D grad p',
             rho' = psi p',  psi = d rho / d p = 1 / (R T)
  -> the p' equation has a diffusion part (incompressible) AND a convection
     part psi p' (upwind): elliptic at low Mach, hyperbolic in supersonic
     regions - shocks are captured.
  rho_f: linear upwind with Venkatakrishnan limiter on ALL cells (shock
  capturing; a central rho_f would oscillate at shocks).

Energy: total enthalpy H (exactly conserved across shocks and constant in
inviscid flow from a uniform free stream), adiabatic walls, viscous work
in thin-layer form (mu_eff - Gamma) grad K, Gamma = mu/Pr + mu_t/Pr_t.

Turbulence: compressible Spalart-Allmaras (rho nuTilde form, NASA TMR).

Inviscid (Euler) runs: inviscid=True -> walls become slip walls, mu = 0.

Boundary conditions as incompressible: 'inlet' (fixed U, H = H_inf),
'outlet' (fixed p), 'wall', 'symmetry'; the free-stream vortex correction
is CompressibleVortexFarfield (Prandtl-Glauert-scaled point vortex,
isentropic outlet pressure).
"""

import math

import numpy as np
from scipy.sparse.linalg import splu

from aerolab.core import kernels
from aerolab.core.linsolve import csr
from aerolab.core.simple_solver import SimpleSolver

GAMMA = 1.4


class CompressibleSolver(SimpleSolver):
    def __init__(self, mesh, bcs, mu, mach, alpha_u=0.8, alpha_p=0.5, alpha_rho=1.0,
                 alpha_h=0.95, gamma=GAMMA, Pr=0.72, Prt=0.9, sutherland=110.4 / 288.15,
                 inviscid=False, U0=(1.0, 0.0), turbulence=None, limiter="venkat",
                 rho_limiter_eps=0.05, **kw):
        """mu: free-stream dynamic viscosity (= 1/Re); mach: free-stream Mach
        number (> 0). sutherland: Sutherland constant / T_inf (None: constant mu).
        alpha_p: pressure under-relaxation (compressible SIMPLEC needs < 1)."""
        if mach <= 0:
            raise ValueError("mach must be > 0")
        if inviscid:
            bcs = {k: ({"type": "symmetry"} if v["type"] == "wall" else v) for k, v in bcs.items()}
            mu, turbulence = 0.0, None
        self.inviscid = inviscid
        self.mach, self.gamma, self.Pr, self.Prt = mach, gamma, Pr, Prt
        self.P0 = 1.0 / (gamma * mach**2)                 # free-stream absolute pressure
        self.R = self.P0                                  # rho_inf = T_inf = 1
        self.cp = gamma * self.R / (gamma - 1.0)
        self.H_inf = self.cp + 0.5
        self.mu_inf = mu
        self.S_suth = sutherland
        self.alpha_rho, self.alpha_h = alpha_rho, alpha_h
        self.rho_limiter_eps = rho_limiter_eps
        n = mesh.ncells
        self.rho = np.ones(n)
        self.T = np.ones(n)
        self.H = np.full(n, self.H_inf)
        self.mu = np.full(n, mu)
        # SimpleSolver uses nu for the (laminar) diffusion and the SA model
        # Euler: limit the convection on all cells (the thin wall layers are
        # quads too and showed wiggles in the supersonic zone); RANS: as the
        # incompressible solver (triangles only: limiting the boundary-layer
        # quads adds needless dissipation)
        kw.setdefault("limiter_cells", "all" if inviscid else "tri")
        # Euler: Venkatakrishnan threshold 1 % of U_inf instead of 10 %: the
        # 10 % threshold left visible Cp wiggles in the supersonic zone
        # (NACA 0012 M 0.8: Cl 0.3578 -> 0.3566, Cd 0.0231 -> 0.0233)
        if inviscid:
            kw.setdefault("limiter_eps", 0.01)
        super().__init__(mesh, bcs, max(mu, 1e-300), alpha_u=alpha_u, alpha_p=alpha_p,
                         U0=U0, turbulence=None, limiter=limiter, **kw)
        # total-pressure inlets {'type': 'inlet', 'p0': gauge total pressure,
        # 'T0': total temperature, 'dir': flow direction}: velocity and density
        # follow from the static pressure (isentropic); velocity inlets keep
        # U fixed and take the isentropic free-stream density for |U_b|
        nb_ = len(mesh.bcell)
        self.b_pt = np.zeros(nb_, bool)
        self.p0b, self.T0b = np.zeros(nb_), np.ones(nb_)
        self.dirb = np.zeros((nb_, 2))
        for name, bc_ in bcs.items():
            if bc_["type"] == "inlet" and "p0" in bc_:
                idx = mesh.patch_faces[name]
                self.b_pt[idx] = True
                self.p0b[idx] = bc_["p0"]
                self.T0b[idx] = bc_.get("T0", 1.0)
                d = np.asarray(bc_.get("dir", (1.0, 0.0)), float)
                self.dirb[idx] = d / np.linalg.norm(d)
        self.k_pt = np.zeros(nb_)
        self.ap = alpha_p
        m = self.m
        self.Fv = self.F.copy()                    # volumetric face flux U.S (Majumdar)
        self.Fbv = self.Fb.copy()
        self.Hb = np.full(len(m.bcell), self.H_inf)
        self.turb = None
        if turbulence == "sa":
            if not self.b_wall.any():
                raise ValueError("the SA model needs at least one wall patch")
            self.turb = CompressibleSA(self, alpha=kw.get("alpha_turb", 0.9))
        elif turbulence not in (None, "laminar"):
            raise ValueError(turbulence)
        self._lu_p = None
        self.stats = {"supersonic_cells": 0}
        # start-up: the first iterations run with frozen density (incompressible
        # SIMPLEC, psi = 0): a cold start's first pressure corrections are O(1)
        # and would change rho by tens of percent
        self.incompressible_start = 30
        self.psi_scale = 1.0

    # ------------------------------------------------------------------
    # thermodynamics
    # ------------------------------------------------------------------
    def sutherland(self, T):
        if self.S_suth is None:
            return np.full_like(T, self.mu_inf)
        S = self.S_suth
        return self.mu_inf * T**1.5 * (1.0 + S) / (T + S)

    def mach_field(self):
        a2 = self.gamma * self.R * self.T
        return np.linalg.norm(self.U, axis=1) / np.sqrt(a2)

    def p_abs(self):
        return self.P0 + self.p

    def _boundary_thermo(self, Ub, pb):
        """boundary-face T and rho. Velocity inlets: T from H_inf, rho
        isentropic from the free stream (fixed: no density correction there -
        an inlet density from the extrapolated pressure fed the global
        pressure level back into the inflow and diverged). Total-pressure
        inlets: isentropic from (p0, T0) and the static pressure. Elsewhere
        the adjacent cell temperature (adiabatic / zero gradient) and the
        boundary pressure."""
        m, g = self.m, self.gamma
        Tb = self.T[m.bcell].copy()
        vin = self.b_inlet & ~self.b_pt
        Tb[vin] = np.maximum((self.H_inf - 0.5 * (Ub[vin] ** 2).sum(1)) / self.cp, 1e-3)
        rhob = (self.P0 + pb) / (self.R * Tb)
        rhob[vin] = Tb[vin] ** (1.0 / (g - 1.0))
        if self.b_pt.any():
            pt = self.b_pt
            Tb[pt] = self._pt_T
            rhob[pt] = (self.P0 + pb[pt]) / (self.R * Tb[pt])
        return Tb, rhob

    def _update_total_inlets(self):
        """velocity on total-pressure inlets from the extrapolated static
        pressure; also the linearisation d(mass flux)/dp = (1 - M^2)/|U|."""
        if not self.b_pt.any():
            return
        m, g = self.m, self.gamma
        pt = self.b_pt
        c = m.bcell[pt]
        pb = self.p[c] + np.einsum('ij,ij->i', self.gradp[c], m.db[pt])
        P = self.P0 + pb
        P0t = self.P0 + self.p0b[pt]
        ratio = np.clip(P / P0t, 0.3, 1.0)
        T = self.T0b[pt] * ratio ** ((g - 1) / g)
        V = np.sqrt(np.maximum(2 * self.cp * (self.T0b[pt] - T), 0.0))
        self._pt_T = T
        self.Ub[pt] = V[:, None] * self.dirb[pt]
        M2 = V**2 / (g * self.R * T)
        Vs = np.maximum(V, 1e-3)
        Sn = -np.einsum('ij,ij->i', self.dirb[pt], m.bSf[pt])          # > 0 for inflow
        self.k_pt[pt] = np.maximum(1 - M2, 0.0) / Vs * Sn

    # ------------------------------------------------------------------
    # hooks used by SimpleSolver._momentum
    # ------------------------------------------------------------------
    def _face_viscosity(self):
        m = self.m
        mu = self.mu
        if self.turb is not None:
            mut = self.rho * self.turb.nu_t()
            mu_f = m.interp(mu) + np.maximum(m.interp(mut), 0.0)
            nb = self.turb.boundary_nuT()
            mub = mu[m.bcell]
            nub = mub / self.rho[m.bcell]
            mu_b = mub + self.rho[m.bcell] * nb * self.turb.fv1_of(nb, nub)
            return mu_f, mu_b
        return m.interp(mu), mu[m.bcell].copy()

    def _transpose_stress(self):
        return not self.inviscid

    # ------------------------------------------------------------------
    def _limited_face(self, phi, phib, F, mask=None, eps=None):
        """linear-upwind face value (upwind by the sign of F), Venkatakrishnan-
        limited on the cells in mask (None = all)."""
        m = self.m
        g = m.grad(phi, phib)
        scale = max(np.abs(phi).max(), 1e-30)
        e2 = ((self.rho_limiter_eps if eps is None else eps) * scale) ** 2
        if kernels.USE_NUMBA:
            psi = kernels.venkat(m, phi, g, phib, e2, mask)
        else:
            sv = self.limiter_mask, self.limiter_eps
            self.limiter_mask = mask
            psi = SimpleSolver._venkat(self, phi, g, phib)
            self.limiter_mask = sv[0]
        up = np.where(F >= 0, m.own, m.nei)
        return phi[up] + psi[up] * np.einsum('ij,ij->i', g[up], m.xf - m.xc[up])

    def _extra_momentum_source(self, mu_f, mu_b, gU):
        """-2/3 mu div(U) I part of the viscous stress (compressible)."""
        m = self.m
        divU = gU[0][:, 0] + gU[1][:, 1]
        q_f = (2.0 / 3.0) * mu_f * m.interp(divU)
        q_b = (2.0 / 3.0) * mu_b * divU[m.bcell]
        return -np.c_[m.sum_to_cells(q_f * m.Sf[:, 0], q_b * m.bSf[:, 0]),
                      m.sum_to_cells(q_f * m.Sf[:, 1], q_b * m.bSf[:, 1])]

    # ------------------------------------------------------------------
    def iterate(self):
        m, n = self.m, self.m.ncells
        own, nei, bc = m.own, m.nei, m.bcell
        self._update_total_inlets()
        Ub, pb = self._boundary_values()
        gU = [m.grad(self.U[:, k], Ub[:, k]) for k in range(2)]
        self.gU = gU
        self.gradp = m.grad_gg(self.p, pb, grad_skew=self.gradp)
        Tb, rhob = self._boundary_thermo(Ub, pb)
        frozen = len(self.history) < self.incompressible_start
        if frozen:
            rhob = self.rho[bc].copy()

        # ---- momentum (F = MASS flux, viscosity = dynamic) --------------------
        diag, a_on, a_no, src, fast, mu_f, mu_b, Db = self._momentum(Ub, gU)
        if not self.inviscid:
            src += self._extra_momentum_source(mu_f, mu_b, gU)
        # non-conservative form div(m U) - U div(m) while the mass fluxes are
        # not yet conservative (exact at convergence): keeps U bounded
        imb = m.sum_to_cells(self.F, self.Fb)
        diag = diag - imb
        extra = np.maximum(1e-3 * np.median(diag) - diag, 0.0)    # stagnation cells
        diag = diag + extra
        src += extra[:, None] * self.U
        diag = diag / self.au
        src += ((1 - self.au) * diag)[:, None] * self.U
        A = self._pattern.matrix(diag, a_on, a_no) if fast else \
            csr(np.r_[diag, a_on, a_no], np.r_[np.arange(n), own, nei],
                np.r_[np.arange(n), nei, own], n)
        scale = max((diag * np.linalg.norm(self.U, axis=1)).sum(), 1e-30)
        resU = [np.abs(A @ self.U[:, k] - src[:, k]).sum() / scale for k in range(2)]
        Ustar = np.c_[tuple(self.lin.scalars(A, [src[:, 0], src[:, 1]],
                                             [self.U[:, 0], self.U[:, 1]]))]

        # ---- face density: limited linear upwind (shock capturing) ------------
        rho_f = self._limited_face(self.rho, rhob, self.F)
        rho_f = np.maximum(rho_f, 0.05 * np.minimum(self.rho[own], self.rho[nei]))

        # ---- Rhie-Chow volumetric flux, then mass flux -------------------------
        Dp = m.vol / diag
        Df = m.interp(Dp)
        gp_f = m.interp(self.gradp)
        Uf = np.c_[m.interp_skew(Ustar[:, 0], gU[0]), m.interp_skew(Ustar[:, 1], gU[1])]
        UbarS = np.einsum('ij,ij->i', Uf, m.Sf)
        Vstar = UbarS - Df * ((self.p[nei] - self.p[own]) * m.coef -
                              np.einsum('ij,ij->i', gp_f, m.Delta))
        Vstar += (1 - self.au) * (self.Fv - self.Ubar_S_old)        # Majumdar
        self.Ubar_S_old = UbarS
        Fstar = rho_f * Vstar

        Vbstar = np.einsum('ij,ij->i', Ub, m.bSf)
        o = self.b_outlet
        if o.any():
            Ustar_b = Ustar[bc[o]] + np.c_[np.einsum('ij,ij->i', gU[0][bc[o]], m.db_t[o]),
                                           np.einsum('ij,ij->i', gU[1][bc[o]], m.db_t[o])]
            UbSb = np.einsum('ij,ij->i', Ustar_b, m.bSf[o])
            Vbstar[o] = UbSb - Dp[bc[o]] * ((self.pb_out[o] - self.p[bc[o]]) * m.bcoef[o] -
                                             np.einsum('ij,ij->i', self.gradp[bc[o]], m.bDelta[o]))
            Vbstar[o] += (1 - self.au) * (self.Fbv[o] - self.Ubar_Sb_old[o])
            self.Ubar_Sb_old[o] = UbSb
        # outlet backflow: density of the cell (extrapolated) is fine
        Fbstar = rhob * Vbstar

        # ---- pressure correction with compressibility ------------------------
        nbsum = np.bincount(own, -a_on, n) + np.bincount(nei, -a_no, n)
        Dc = m.vol / np.maximum(diag - nbsum, (1.0 - self.au) * diag)
        Dcf = m.interp(Dc)
        cf = rho_f * Dcf * m.coef                         # diffusion part
        cb = rhob * Dc[bc] * m.bcoef
        # d rho / d p along the flow: with the total enthalpy fixed the
        # temperature drops with p (dT = dp/(rho cp)) -> the isentropic
        # 1/a^2 = 1/(gamma R T), not the isothermal 1/(R T); the isothermal
        # value overshoots the density correction by gamma and diverges at
        # M >= 0.5 (channel test). Only the convergence path depends on psi.
        psi = 1.0 / (self.gamma * self.R * self.T)
        psib = 1.0 / (self.gamma * self.R * Tb)
        if frozen:
            psi, psib = 0.0 * psi, 0.0 * psib
        psi, psib = self.psi_scale * psi, self.psi_scale * psib
        Fp = np.maximum(Vstar, 0.0)                       # volumetric, owner upwind
        Fm = np.minimum(Vstar, 0.0)                       # neighbour upwind
        pdiag = np.bincount(own, cf + Fp * psi[own], n) + np.bincount(nei, cf - Fm * psi[nei], n)
        a_pn = -cf + Fm * psi[nei]                        # row own, col nei
        a_np = -cf - Fp * psi[own]                        # row nei, col own
        # boundaries: outlet (p' = 0 at the face, rho' upwind = cell for outflow);
        # inlet with fixed U: rho_b' = psi_b p'_P (p extrapolated)
        bdiag = np.where(o, cb + np.maximum(Vbstar, 0.0) * psi[bc], 0.0)
        bdiag += np.where(self.b_pt, self.k_pt, 0.0)
        pdiag += np.bincount(bc, bdiag, n)
        imbalance = m.sum_to_cells(Fstar, Fbstar)
        Ap = self._pattern.matrix(pdiag, a_pn, a_np) if fast else \
            csr(np.r_[pdiag, a_pn, a_np], np.r_[np.arange(n), own, nei],
                np.r_[np.arange(n), nei, own], n)
        pc = self._solve_p(Ap, -imbalance)

        # ---- corrections --------------------------------------------------------
        pc_up = np.where(Vstar >= 0, pc[own] * psi[own], pc[nei] * psi[nei])
        self.F = Fstar - cf * (pc[nei] - pc[own]) + Vstar * pc_up
        Fb = Fbstar.copy()
        Fb[o] = Fbstar[o] + cb[o] * pc[bc[o]] + np.maximum(Vbstar[o], 0.0) * psi[bc[o]] * pc[bc[o]]
        pt = self.b_pt
        Fb[pt] = Fbstar[pt] + self.k_pt[pt] * pc[bc[pt]]
        self.Fb = Fb
        # volumetric fluxes for the next Majumdar term
        # (velocity part only: the density correction rho' V* must not enter
        # the Majumdar term as a velocity - with it the iteration diverged
        # at M 0.5, faster the stronger the momentum under-relaxation)
        self.Fv = Vstar - Dcf * m.coef * (pc[nei] - pc[own])
        Fbv = Vbstar.copy()
        Fbv[o] = Vbstar[o] + Dc[bc[o]] * m.bcoef[o] * pc[bc[o]]
        self.Fbv = Fbv
        pcb = np.where(o, 0.0, pc[bc])
        gpc = m.grad_gg(pc, pcb)
        self.U = Ustar - Dc[:, None] * gpc
        self.p = self.p + (1.0 if frozen else self.ap) * pc
        # keep the absolute pressure positive
        self.p = np.maximum(self.p, -0.98 * self.P0)
        rho_corr = self.rho + self.ap * psi * pc

        Fref = max(np.abs(self.Fb).sum(), 1e-30)
        res = {"cont": np.abs(imbalance).sum() / Fref, "Ux": resU[0], "Uy": resU[1]}

        # ---- energy (total enthalpy) -------------------------------------------
        res["H"] = self._energy(mu_f, mu_b)
        K = 0.5 * (self.U ** 2).sum(1)
        self.T = np.maximum((self.H - K) / self.cp, 0.05)
        rho_eos = (self.P0 + self.p) / (self.R * self.T)
        # density: EOS with the new p, T (rho_corr is the linearised value the
        # pressure correction assumed; they agree at convergence)
        if not frozen:
            self.rho = rho_corr + self.alpha_rho * (rho_eos - rho_corr)
            self.rho = np.maximum(self.rho, 1e-3)
        self.mu = self.sutherland(self.T) if not self.inviscid else np.zeros(n)
        self.stats["supersonic_cells"] = int((self.mach_field() > 1.0).sum())

        if self.turb is not None:
            self.gU = [m.grad(self.U[:, k], self._boundary_values()[0][:, k]) for k in range(2)]
            res["nuT"] = self.turb.solve()
        self.history.append(res)
        return res

    def _solve_p(self, A, b):
        """non-symmetric pressure-correction system: sparse LU (robust; the
        pattern is fixed). Iterative alternative: BiCGSTAB + AMG."""
        if self.lin.kind == "iterative":
            try:
                import pyamg
                from scipy.sparse.linalg import bicgstab
                if self._lu_p is None or self._lu_p[1] >= 5:
                    self._lu_p = [pyamg.smoothed_aggregation_solver(
                        A.tocsr(), symmetry="nonsymmetric", max_coarse=200), 0]
                self._lu_p[1] += 1
                M = self._lu_p[0].aspreconditioner(cycle="V")
                x, info = bicgstab(A, b, rtol=0.05, atol=0.0, M=M, maxiter=100)
                if info == 0 and np.all(np.isfinite(x)):
                    return x
                self._lu_p = None
            except Exception:
                self._lu_p = None
        return splu(A.tocsc()).solve(b)

    def _energy(self, mu_f, mu_b):
        """total enthalpy transport; returns the normalised residual"""
        m, n = self.m, self.m.ncells
        own, nei, bc = m.own, m.nei, m.bcell
        F, Fb = self.F, self.Fb
        H = self.H
        if self.inviscid:
            G_f = np.zeros(len(own))
            G_b = np.zeros(len(bc))
        else:
            mu_l = m.interp(self.mu)
            mut_f = np.maximum(mu_f - mu_l, 0.0)
            G_f = mu_l / self.Pr + mut_f / self.Prt
            mub_l = self.mu[bc]
            G_b = mub_l / self.Pr + np.maximum(mu_b - mub_l, 0.0) / self.Prt
        Dh = G_f * m.coef
        a_on = -Dh - np.maximum(-F, 0.0)
        a_no = -Dh - np.maximum(F, 0.0)
        diag = np.bincount(own, Dh + np.maximum(F, 0.0), n) + \
            np.bincount(nei, Dh + np.maximum(-F, 0.0), n)
        src = np.zeros(n)
        inl, o = self.b_inlet, self.b_outlet
        H_in = np.where(self.b_pt, self.cp * self.T0b, self.H_inf)
        Hb = np.where(inl, H_in, H[bc])
        gH = m.grad(H, Hb)
        # boundaries: inlet Dirichlet (convection + diffusion), outlet zero
        # gradient (outflow implicit, backflow at H_inf), walls adiabatic
        Dbh = G_b * m.bcoef
        diag += np.bincount(bc, np.where(inl, Dbh, 0.0) + np.where(o, np.maximum(Fb, 0.0), 0.0), n)
        src += np.bincount(bc, np.where(inl, Dbh * H_in - Fb * H_in, 0.0) +
                           np.where(o, -np.minimum(Fb, 0.0) * self.H_inf, 0.0), n)
        # second-order deferred correction (limited like the momentum)
        if self.second_order:
            Hf_lin = self._limited_face(H, Hb, F, mask=self.limiter_mask, eps=0.02)
            up = np.where(F >= 0, own, nei)
            src -= m.sum_to_cells(F * (Hf_lin - H[up]))
        if not self.inviscid:
            # non-orthogonal diffusion correction + viscous work (thin layer)
            src += m.sum_to_cells(G_f * np.einsum('ij,ij->i', m.interp(gH), m.Sf - m.Delta))
            K = 0.5 * (self.U ** 2).sum(1)
            Ub, _ = self._boundary_values()
            Kb = 0.5 * (Ub ** 2).sum(1)
            gK = m.grad(K, Kb)
            wv_f = (mu_f - G_f) * np.einsum('ij,ij->i', m.interp(gK), m.Sf)
            wv_b = np.where(self.b_wall, (mu_b - G_b) * (Kb - K[bc]) * m.bcoef, 0.0)
            src += m.sum_to_cells(wv_f, wv_b)
        diag = diag - m.sum_to_cells(F, Fb)     # div(m H) - H div(m), see iterate
        extra = np.maximum(1e-3 * np.median(diag) - diag, 0.0)    # stagnation cells
        diag = diag + extra
        src = src + extra * H
        diag_r = diag / self.alpha_h
        src = src + (1 - self.alpha_h) * diag_r * H
        A = self._pattern.matrix(diag_r, a_on, a_no) if self._pattern is not None else \
            csr(np.r_[diag_r, a_on, a_no], np.r_[np.arange(n), own, nei],
                np.r_[np.arange(n), nei, own], n)
        res = np.abs(A @ H - src).sum() / max(np.abs(diag_r * H).sum(), 1e-30)
        self.H = self.lin.scalar(A, src, H)
        return res

    # ------------------------------------------------------------------
    def set_initial(self, U, p=None, nuTilde=None, rho=None, T=None):
        super().set_initial(U, p, nuTilde)
        if T is not None:
            self.T = np.array(T, float)
        K = 0.5 * (self.U ** 2).sum(1)
        self.H = self.cp * self.T + K
        self.rho = np.array(rho, float) if rho is not None else \
            (self.P0 + self.p) / (self.R * self.T)
        self.mu = self.sutherland(self.T) if not self.inviscid else np.zeros(self.m.ncells)
        rf = self.m.interp(self.rho)
        self.Fv = self.F.copy()
        self.F = self.F * rf
        self.Fbv = self.Fb.copy()
        self.Fb = self.Fb * self.rho[self.m.bcell]

    def wall_values(self, patch):
        """pressure and wall shear (dynamic viscosity of the wall cell)"""
        m = self.m
        idx = m.patch_faces[patch]
        c = m.bcell[idx]
        pb = self.p[c] + np.einsum('ij,ij->i', self.gradp[c], m.db[idx])
        S = m.bSf[idx]
        Smag = np.linalg.norm(S, axis=1)
        n_out = S / Smag[:, None]
        dn = -np.einsum('ij,ij->i', m.db[idx], n_out)
        if self.inviscid:
            tau = np.zeros((len(idx), 2))
        else:
            du = self.U[c] - self.Ub[idx]
            du_t = du - np.einsum('ij,ij->i', du, n_out)[:, None] * n_out
            tau = self.mu[c][:, None] * du_t / np.abs(dn)[:, None]
        return idx, pb, tau, S, Smag


# ----------------------------------------------------------------------
class CompressibleSA:
    """Spalart-Allmaras (SA-noft2) in the compressible form of the NASA TMR:
        div(rho U nuT) = rho (P - D) + (1/sigma)[div((mu + rho nuT) grad nuT)
                                                 + cb2 rho |grad nuT|^2]
    with nu = mu/rho per cell (Sutherland). Same constants and source as
    turbulence.SpalartAllmaras (whose source() is reused)."""

    def __init__(self, solver, nut_inf_ratio=3.0, alpha=0.9):
        from aerolab.core.turbulence import SpalartAllmaras, wall_distance
        self.s = solver
        m = solver.m
        self.alpha = alpha
        self.nu_inf = solver.mu_inf                   # rho_inf = 1
        self.nut_inf = nut_inf_ratio * self.nu_inf
        self.d = np.maximum(wall_distance(m, np.where(solver.b_wall)[0]), 1e-30)
        self.nuT = np.full(m.ncells, self.nut_inf)
        self.gamma = None
        self._sa = SpalartAllmaras.__new__(SpalartAllmaras)   # for source()
        self._sa.d, self._sa.gamma = self.d, None

    @property
    def nu(self):
        return self.s.mu / self.s.rho

    def fv1_of(self, nuT, nu):
        from aerolab.core.turbulence import CV1
        chi = nuT / nu
        return chi**3 / (chi**3 + CV1**3)

    def fv1(self, nuT=None):
        return self.fv1_of(self.nuT if nuT is None else nuT, self.nu)

    def nu_t(self):
        nut = self.nuT * self.fv1()
        return nut if self.gamma is None else nut * self.gamma

    def boundary_nuT(self):
        s, m = self.s, self.s.m
        nb = np.where(s.b_wall, 0.0, self.nuT[m.bcell])
        return np.where(s.b_inlet, self.nut_inf, nb)

    def face_viscosity(self):
        return self.s._face_viscosity()

    def solve(self):
        from aerolab.core.turbulence import CB2, SIGMA
        s, m = self.s, self.s.m
        n = m.ncells
        own, nei, bc = m.own, m.nei, m.bcell
        rho, nu = s.rho, self.nu
        nuT = self.nuT
        nb = self.boundary_nuT()
        g = m.grad(nuT, nb)
        gu, gv = s.gU
        Omega = np.abs(gv[:, 0] - gu[:, 1])
        self._sa.nu, self._sa.gamma = nu, self.gamma
        S0 = self._sa.source(nuT, Omega)
        eps = 1e-6 * np.maximum(nuT, nu)
        J = (self._sa.source(nuT + eps, Omega) - S0) / eps
        Jn = np.maximum(-J, 0.0)
        diag = Jn * rho * m.vol
        src = (S0 + Jn * nuT) * rho * m.vol
        # cb2 term explicit (>= 0)
        src += CB2 / SIGMA * rho * (g ** 2).sum(1) * m.vol
        # convection (mass flux, upwind) + diffusion (mu + rho nuT)/sigma
        F, Fb = s.F, s.Fb
        kf = (m.interp(s.mu) + np.maximum(m.interp(rho * nuT), 0.0)) / SIGMA * m.coef
        a_on = -kf - np.maximum(-F, 0.0)
        a_no = -kf - np.maximum(F, 0.0)
        diag += np.bincount(own, kf + np.maximum(F, 0.0), n) + \
            np.bincount(nei, kf + np.maximum(-F, 0.0), n)
        src += m.sum_to_cells(kf / m.coef * np.einsum('ij,ij->i', m.interp(g), m.Sf - m.Delta))
        dirich = s.b_wall | s.b_inlet
        Dbk = (s.mu[bc] + rho[bc] * nb) / SIGMA * m.bcoef
        diag += np.bincount(bc, np.where(dirich, Dbk, 0.0) +
                            np.where(~dirich, np.maximum(Fb, 0.0), 0.0), n)
        src += np.bincount(bc, np.where(dirich, Dbk * nb - Fb * nb, 0.0) +
                           np.where(~dirich, -np.minimum(Fb, 0.0) * nb, 0.0), n)
        diag = diag - m.sum_to_cells(F, Fb)       # non-conservative form (as momentum)
        diag = np.maximum(diag, 1e-3 * np.median(diag))
        diag_r = diag / self.alpha
        src = src + (1 - self.alpha) * diag_r * nuT
        pat = s._pattern
        A = pat.matrix(diag_r, a_on, a_no) if pat is not None else \
            csr(np.r_[diag_r, a_on, a_no], np.r_[np.arange(n), own, nei],
                np.r_[np.arange(n), nei, own], n)
        res = np.abs(A @ nuT - src).sum() / max(np.abs(diag_r * nuT).sum(), 1e-30)
        self.nuT = np.maximum(s.lin.scalar(A, src, nuT), 0.0)
        return res


# ----------------------------------------------------------------------
class CompressibleVortexFarfield:
    """Free-stream + point vortex with Prandtl-Glauert scaling (compressible
    far field of a lifting airfoil):
        phi = -(Gamma/2pi) atan(beta y'/x'),  beta = sqrt(1 - M^2)
    in wind axes (x' along U_inf). Inlet: velocity; outlet: isentropic
    pressure p/p_inf = (1 + (g-1)/2 M^2 (1 - |u|^2))^(g/(g-1))."""

    def __init__(self, solver, alpha, wall="wall", patch="inlet", outlet="outlet", every=50,
                 center=(0.25, 0.0)):
        self.s, self.alpha, self.every = solver, alpha, every
        self.walls = [wall] if isinstance(wall, str) else list(wall)
        self.center = center
        self.idx = solver.m.patch_faces[patch]
        self.out = solver.m.patch_faces.get(outlet, np.zeros(0, int))
        self.U = (math.cos(math.radians(alpha)), math.sin(math.radians(alpha)))
        self.gamma = 0.0

    def velocity(self, xy):
        a = math.radians(self.alpha)
        M = self.s.mach
        beta = math.sqrt(max(1.0 - M**2, 0.05))
        dx, dy = xy[:, 0] - self.center[0], xy[:, 1] - self.center[1]
        xw = dx * math.cos(a) + dy * math.sin(a)          # wind axes
        yw = -dx * math.sin(a) + dy * math.cos(a)
        den = 2 * math.pi * (xw**2 + beta**2 * yw**2)
        uw = 1.0 + self.gamma * beta * yw / den
        vw = -self.gamma * beta * xw / den
        return np.c_[uw * math.cos(a) - vw * math.sin(a), uw * math.sin(a) + vw * math.cos(a)]

    def update(self, it, res=None):
        from aerolab.core.postproc import wall_data
        if it % self.every:
            return
        self.gamma = 0.5 * sum(wall_data(self.s, w, self.U)[1]["Cl"] for w in self.walls)
        s = self.s
        s.Ub[self.idx] = self.velocity(s.m.bxf[self.idx])
        uo = self.velocity(s.m.bxf[self.out])
        g, M = s.gamma, s.mach
        ratio = np.maximum(1 + 0.5 * (g - 1) * M**2 * (1 - (uo**2).sum(1)), 1e-3) ** (g / (g - 1))
        s.pb_out[self.out] = s.P0 * (ratio - 1.0)
