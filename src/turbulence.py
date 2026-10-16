"""
Spalart-Allmaras one-equation turbulence model (SA-noft2), exactly as defined
by the NASA Turbulence Modeling Resource (tmbwg.github.io/turbmodels/spalart.html):

  D nuT/Dt = cb1 S~ nuT - cw1 fw (nuT/d)^2
             + (1/sigma) [ div((nu + nuT) grad nuT) + cb2 |grad nuT|^2 ]
  nu_t = nuT fv1,  fv1 = chi^3/(chi^3 + cv1^3),  chi = nuT/nu
  S~   = Omega + Sbar,  Sbar = nuT fv2/(kappa^2 d^2),  fv2 = 1 - chi/(1 + chi fv1)
         with the Allmaras et al. (2012) limiter when Sbar < -cv2 Omega:
         S~ = Omega + Omega (cv2^2 Omega + cv3 Sbar) / ((cv3 - 2 cv2) Omega - Sbar)
  r = min(nuT/(S~ kappa^2 d^2), 10),  g = r + cw2 (r^6 - r),
  fw = g ((1 + cw3^6)/(g^6 + cw3^6))^(1/6)
  noft2: the ft2 laminar-suppression term is omitted (ct3 = 0).
  BCs: nuT = 0 on walls, nuT = 3 nu in the freestream.

Discretisation (finite volume, same mesh / face fluxes as the flow solver):
  convection first-order upwind (implicit); the diffusion and cb2 terms
  combined as in CFL3D/FUN3D,
      (1/sigma)[div((nu+nuT) grad nuT) + cb2 |grad nuT|^2]
    = (1/sigma)[(1+cb2) div((nu+nuT) grad nuT) - cb2 (nu+nuT) lap nuT],
  i.e. an implicit face coefficient (1/sigma)[(1+cb2)(nu+nuT_f) - cb2(nu+nuT_P)]
  (clipped >= 0) with explicit non-orthogonal correction; net source P - D
  linearised per cell (the stabilising part -dS/dnuT > 0 implicit), under-relaxed,
  clipped at nuT >= 0.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu  # noqa: F401  (kept for reference)

from linsolve import csr
import kernels

CB1, CB2, SIGMA, KAPPA = 0.1355, 0.622, 2.0 / 3.0, 0.41
CV1, CW2, CW3 = 7.1, 0.3, 2.0
CW1 = CB1 / KAPPA**2 + (1.0 + CB2) / SIGMA
CV2, CV3 = 0.7, 0.9


def wall_distance(mesh, wall_faces, k=8, return_index=False):
    """Distance from every cell centre to the nearest wall face (exact
    point-to-segment distance over the k nearest face midpoints).
    return_index: also the position (in wall_faces) of the nearest face."""
    from scipy.spatial import cKDTree
    S = mesh.bSf[wall_faces]
    mid = mesh.bxf[wall_faces]
    edge = np.c_[-S[:, 1], S[:, 0]]              # edge vector (bSf = right normal of edge)
    a, b = mid - 0.5 * edge, mid + 0.5 * edge
    k = min(k, len(wall_faces))
    _, idx = cKDTree(mid).query(mesh.xc, k=k)
    idx = idx.reshape(len(mesh.xc), k)
    P = mesh.xc[:, None, :]
    A, B = a[idx], b[idx]
    AB = B - A
    t = np.clip(np.einsum('ijk,ijk->ij', P - A, AB) / np.einsum('ijk,ijk->ij', AB, AB), 0, 1)
    d = np.linalg.norm(P - (A + t[..., None] * AB), axis=2)
    if return_index:
        j = d.argmin(axis=1)
        return d[np.arange(len(d)), j], idx[np.arange(len(d)), j]
    return d.min(axis=1)


class SpalartAllmaras:
    def __init__(self, solver, nut_inf_ratio=3.0, alpha=0.8):
        self.s = solver
        m = solver.m
        self.nu = solver.nu
        self.alpha = alpha
        self.nut_inf = nut_inf_ratio * self.nu
        self.d = np.maximum(wall_distance(m, np.where(solver.b_wall)[0]), 1e-30)
        self.nuT = np.full(m.ncells, self.nut_inf)
        self.grad = np.zeros((m.ncells, 2))
        self.nuT_n = self.nuT_nm1 = None
        # intermittency: 1 = turbulent (default), 0 = laminar region (eddy
        # viscosity and production off) - set by transition.Transition
        self.gamma = None
        # separation-induced transition (transition.Transition): cells of the
        # separated shear layer just behind the transition point, where nuT is
        # held at least at a turbulent level - a trip, as SA's own trip term
        # (ft1) does; None = off
        self.seed = None
        # production multiplier field beta(x) (field inversion / learned
        # corrections, fiml.py): S_prod -> beta S_prod; None = 1
        self.beta = None

    def _prod_factor(self):
        """intermittency x beta (multiplies the production only)"""
        g, b = getattr(self, "gamma", None), getattr(self, "beta", None)
        if b is None:
            return g
        return b if g is None else np.asarray(g) * np.asarray(b)

    # ------------------------------------------------------------------
    def fv1(self, nuT=None):
        chi = (self.nuT if nuT is None else nuT) / self.nu
        return chi**3 / (chi**3 + CV1**3)

    def nu_t(self):
        """Eddy viscosity (times the intermittency: 0 in laminar regions)."""
        nut = self.nuT * self.fv1()
        g = getattr(self, "gamma", None)
        return nut if g is None else nut * g

    def boundary_nuT(self):
        s, m = self.s, self.s.m
        nb = np.where(s.b_wall, 0.0, self.nuT[m.bcell])       # outlet/symmetry: zero-gradient
        nb = np.where(s.b_inlet, self.nut_inf, nb)
        return nb

    def face_viscosity(self):
        """Effective (molecular + eddy) viscosity on internal and boundary faces."""
        m = self.s.m
        nut = self.nu_t()
        nu_f = self.nu + np.maximum(m.interp(nut), 0.0)
        nb = self.boundary_nuT()
        nu_b = self.nu + nb * self.fv1(nb)
        return nu_f, nu_b

    def time_levels(self):
        self.nuT_nm1 = self.nuT_n
        self.nuT_n = self.nuT.copy()

    def source(self, nuT, Omega):
        """Net SA source per unit volume: cb1 S~ nuT - cw1 fw (nuT/d)^2."""
        d2 = self.d**2
        chi = nuT / self.nu
        fv1 = chi**3 / (chi**3 + CV1**3)
        fv2 = 1.0 - chi / (1.0 + chi * fv1)
        Sbar = nuT * fv2 / (KAPPA**2 * d2)
        with np.errstate(divide="ignore", invalid="ignore"):
            lim = Omega + Omega * (CV2**2 * Omega + CV3 * Sbar) / ((CV3 - 2 * CV2) * Omega - Sbar)
        St = np.where(Sbar >= -CV2 * Omega, Omega + Sbar, lim)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(St > 0, nuT / (St * KAPPA**2 * d2), 10.0)
        r = np.minimum(np.nan_to_num(r, nan=10.0, posinf=10.0), 10.0)
        gg = r + CW2 * (r**6 - r)
        fw = gg * ((1 + CW3**6) / (gg**6 + CW3**6)) ** (1.0 / 6.0)
        g = self._prod_factor()
        prod = CB1 * St * nuT if g is None else CB1 * St * nuT * g
        return prod - CW1 * fw * nuT**2 / d2

    def _lap_nonorth(self, gf, g):
        """Non-orthogonal part of the Laplacian (sum over faces of grad_f.(Sf - Delta))."""
        m = self.s.m
        return m.sum_to_cells(np.einsum('ij,ij->i', gf, m.Sf - m.Delta))

    # ------------------------------------------------------------------
    def solve(self):
        s, m = self.s, self.s.m
        n = m.ncells
        own, nei, bc = m.own, m.nei, m.bcell
        nuT = self.nuT
        nb = self.boundary_nuT()
        g = m.grad(nuT, nb)
        self.grad = g

        # ---- source terms: net S = P - D, linearised in nuT -------------------
        gu, gv = s.gU
        Omega = np.abs(gv[:, 0] - gu[:, 1])                      # |vorticity| (2D)
        if kernels.USE_NUMBA:
            gam = self._prod_factor()
            S0, J = kernels.sa_source(np.ascontiguousarray(nuT), np.ascontiguousarray(Omega),
                                      self.d, float(self.nu),
                                      np.ones(1) if gam is None else np.ascontiguousarray(gam, float),
                                      gam is not None, CB1, CW1, CW2, CW3, CV1, CV2, CV3, KAPPA)
        else:
            S0 = self.source(nuT, Omega)
            eps = 1e-6 * np.maximum(nuT, self.nu)
            J = (self.source(nuT + eps, Omega) - S0) / eps        # dS/dnuT (cell-local)
        Jn = np.maximum(-J, 0.0)                                  # only the stabilising part
        diag_d = Jn * m.vol
        src_d = (S0 + Jn * nuT) * m.vol

        # ---- matrix: convection (upwind) + diffusion incl. cb2 ---------------
        F, Fb = s.F, s.Fb
        nuTf = np.maximum(m.interp(nuT), 0.0)
        kf = (1.0 + CB2) * (self.nu + nuTf)
        k_own = np.maximum(kf - CB2 * (self.nu + nuT[own]), 0.0) / SIGMA * m.coef
        k_nei = np.maximum(kf - CB2 * (self.nu + nuT[nei]), 0.0) / SIGMA * m.coef
        # row of the owner uses k_own, row of the neighbour uses k_nei
        a_on = -k_own - np.maximum(-F, 0.0)          # A[own, nei]
        a_no = -k_nei - np.maximum(F, 0.0)           # A[nei, own]
        diag = np.bincount(own, k_own + np.maximum(F, 0.0), n) + \
            np.bincount(nei, k_nei + np.maximum(-F, 0.0), n)
        src = src_d
        diag += diag_d
        # explicit non-orthogonal diffusion correction (conservative part only)
        gf = m.interp(g)
        cno = kf / SIGMA * np.einsum('ij,ij->i', gf, m.Sf - m.Delta)
        src += m.sum_to_cells(cno)
        src -= CB2 / SIGMA * (self.nu + nuT) * self._lap_nonorth(gf, g)

        # boundaries
        dirich = s.b_wall | s.b_inlet
        Dbk = np.maximum((1.0 + CB2) * (self.nu + nb) - CB2 * (self.nu + nuT[bc]), 0.0) \
            / SIGMA * m.bcoef
        diag += np.bincount(bc, np.where(dirich, Dbk, 0.0) +
                            np.where(~dirich, np.maximum(Fb, 0.0), 0.0), n)
        src += np.bincount(bc, np.where(dirich, Dbk * nb - Fb * nb, 0.0) +
                           np.where(~dirich, -np.minimum(Fb, 0.0) * nb, 0.0), n)

        # time term (BDF2 / Euler, same as momentum)
        if s.dt is not None and self.nuT_n is not None:
            if self.nuT_nm1 is None:
                c1, old = 1.0, self.nuT_n
            else:
                c1, old = 1.5, 2.0 * self.nuT_n - 0.5 * self.nuT_nm1
            diag += c1 * m.vol / s.dt
            src += m.vol / s.dt * old

        # under-relaxation
        diag_r = diag / self.alpha
        src = src + (1 - self.alpha) * diag_r * nuT
        norm = max(np.abs(diag_r * nuT).sum(), 1e-30)
        if getattr(s, "cfl", None) is not None:      # pseudo-time (see SimpleSolver)
            pt = diag / s.cfl
            diag_r = diag_r + pt
            src = src + pt * nuT
        if self.seed is not None:
            # trip: nuT >= the seed level, imposed implicitly (penalty) so that the
            # residual measures the converged state (a floor applied after the
            # solve kept the residual from falling: every low-Re point ran to
            # the iteration limit)
            idx, val = self.seed
            tgt = np.maximum(nuT[idx], val)
            pen = 1e6 * diag_r[idx]
            diag_r = diag_r.copy()
            diag_r[idx] += pen
            src = src.copy()
            src[idx] += pen * tgt
        pat = getattr(s, "_pattern", None)
        if pat is not None:                     # fixed CSR pattern (kernels)
            A = pat.matrix(diag_r, a_on, a_no)
        else:
            A = csr(np.r_[diag_r, a_on, a_no], np.r_[np.arange(n), own, nei],
                    np.r_[np.arange(n), nei, own], n)
        res = np.abs(A @ nuT - src).sum() / norm
        self.nuT = np.maximum(s.lin.scalar(A, src, nuT), 0.0)
        return res
