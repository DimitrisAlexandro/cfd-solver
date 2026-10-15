"""
Discrete adjoint of the incompressible RANS airfoil solver (SIMPLEC / coupled,
Spalart-Allmaras): the gradient of a force coefficient (Cl, Cd, Cm) with
respect to every flow and shape parameter for the cost of ONE extra linear
solve, whatever the number of design variables.

    python adjoint.py gradient naca0012 --re 3e6 --alpha 4 --objective cd,cl,cm
    python adjoint.py check naca0012 --re 3e6 --alpha 4 --level 0.5   # vs finite differences

Method
------
The converged solver state is a root of a discrete residual R(X; D) = 0, with
the design D (angle of attack, Reynolds number, airfoil shape). For an
objective I(X; D) the adjoint psi solves

    (dR/dX)^T psi = -(dI/dX)^T          then   dI/dD = dI/dD|_X + psi^T dR/dD|_X.

1. The residual (Residual.evaluate) re-implements the discrete equations the
   solver converges to, term by term (same convection, deferred correction,
   Venkatakrishnan limiter, diffusion with non-orthogonal and transpose
   terms, Rhie-Chow fluxes with the Majumdar term, Green-Gauss pressure
   gradient with its skew correction, SA with the CFL3D cb2 split, the
   point-vortex far field). The solver carries several quantities from one
   iteration to the next that are implicit at convergence; they are unknowns
   here, each with its own equation, so the stencil stays compact:

       X = [u, v, p, nuTilde, dp/dx, dp/dy (cells) | F (internal faces) |
            F (outlet faces) | Gamma (vortex circulation = Cl/2)]

   R(X*) = 0 at the converged solution to the solver's own round-off
   (`residual_check`).
2. dR/dX: complex-step derivatives (Squire & Trapp 1998: f'(x) = Im f(x + ih)/h,
   no subtraction error, exact to machine precision) of that residual, all
   columns of one colour at once (Curtis-Powell-Reid colouring of the
   radius-2 cell stencil: ~100-300 residual evaluations for the whole
   Jacobian). The residual is written complex-safe: branches (upwinding,
   limiter, max/min, clipping) follow the real part.
3. (dR/dX)^T psi = -dI/dX: sparse direct LU (one factorisation, any number
   of objectives).
4. dR/dD and dI/dD at fixed X: complex step again, one residual evaluation
   per design variable. Shape variables move the mesh with a linear
   inverse-distance-weighted deformation (Luke, Collins & Blades 2012) of
   the wall displacement, and the geometry (areas, centroids, face vectors,
   interpolation weights, least-squares gradient operators, wall distance)
   is recomputed from the moved points (Geometry).

Frozen (treated as constants of the linearisation): the transition location
(intermittency field) and the separation-bubble trip cells - the usual
"frozen transition" adjoint; the limiter's active-cell mask; the wall
distance's nearest-face choice. Everything else is differentiated.

Validation: `python adjoint.py check` compares dI/dalpha and the CST shape
gradients with central finite differences of complete, tightly converged
flow solutions (validation/adjoint_report.md).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

import kernels
from turbulence import CB1, CB2, SIGMA, KAPPA, CV1, CW1, CW2, CW3, CV2, CV3

H_CS = 1e-30          # complex step
COS80 = math.cos(math.radians(80.0))


# ----------------------------------------------------------------------
# complex-safe elementwise helpers: branches follow the real part
# ----------------------------------------------------------------------
def cmax(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.where(np.real(a) >= np.real(b), a, b)


def cmin(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.where(np.real(a) <= np.real(b), a, b)


def cpos(x):
    return np.where(np.real(x) > 0, x, 0.0)


def cneg(x):
    return np.where(np.real(x) < 0, x, 0.0)


def cabs(x):
    return np.where(np.real(x) < 0, -x, x)


def cclip(x, lo, hi):
    r = np.real(x)
    return np.where(r < lo, lo, np.where(r > hi, hi, x))


def _dot(a, b):
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]


# ----------------------------------------------------------------------
# geometry from points (fixed topology), complex-safe
# ----------------------------------------------------------------------
class Topology:
    """Connectivity of an FVMesh (does not change when the points move) and
    the topological scatter matrices (face -> cell sums)."""

    def __init__(self, fvm):
        self.n = fvm.ncells
        self.own, self.nei, self.bcell = fvm.own, fvm.nei, fvm.bcell
        self.fa, self.fb, self.ba, self.bb = fvm.fa, fvm.fb, fvm.ba, fvm.bb
        self.nf, self.nb = len(self.own), len(self.bcell)
        self.groups = {}
        for k in sorted(set(len(c) for c in fvm.cells)):
            idx = np.array([i for i, c in enumerate(fvm.cells) if len(c) == k])
            self.groups[k] = (idx, np.array([fvm.cells[i] for i in idx]))
        n, nf, nb = self.n, self.nf, self.nb
        fi = np.arange(nf)
        self.Sown = sp.csr_matrix((np.ones(nf), (self.own, fi)), (n, nf))
        self.Snei = sp.csr_matrix((np.ones(nf), (self.nei, fi)), (n, nf))
        self.Sdiv = (self.Sown - self.Snei).tocsr()
        self.Sb = sp.csr_matrix((np.ones(nb), (self.bcell, np.arange(nb))), (n, nb))
        self.patch_faces = fvm.patch_faces
        self.npoints = len(fvm.points)


class Geometry:
    """Everything FVMesh computes from the points, vectorised and valid for
    complex points (complex-step shape derivatives). Identical formulas to
    fvmesh.FVMesh (checked in the tests to round-off)."""

    def __init__(self, topo, P, wall_faces, wall_dist_index=None):
        self.t = t = topo
        P = np.asarray(P)
        self.points = P
        n = t.n
        dt = np.result_type(P.dtype, float)
        vol = np.zeros(n, dt)
        xc = np.zeros((n, 2), dt)
        for k, (idx, C) in t.groups.items():
            x, y = P[C, 0], P[C, 1]
            xn, yn = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
            cr = x * yn - xn * y
            A = 0.5 * cr.sum(1)
            vol[idx] = A
            xc[idx, 0] = ((x + xn) * cr).sum(1) / (6 * A)
            xc[idx, 1] = ((y + yn) * cr).sum(1) / (6 * A)
        self.vol, self.xc = vol, xc
        own, nei, bc = t.own, t.nei, t.bcell
        tt = P[t.fb] - P[t.fa]
        self.Sf = np.c_[tt[:, 1], -tt[:, 0]]
        self.xf = 0.5 * (P[t.fa] + P[t.fb])
        tb = P[t.bb] - P[t.ba]
        self.bSf = np.c_[tb[:, 1], -tb[:, 0]]
        self.bxf = 0.5 * (P[t.ba] + P[t.bb])
        d = xc[nei] - xc[own]
        self.d = d
        dS = _dot(d, self.Sf)
        S2 = _dot(self.Sf, self.Sf)
        dS = cmax(dS, COS80 * np.sqrt(_dot(d, d) * S2))
        self.coef = S2 / dS
        self.Delta = d * self.coef[:, None]
        self.w = cclip(_dot(xc[nei] - self.xf, self.Sf) / dS, 0.0, 1.0)
        self.rskew = self.xf - (xc[own] + (1 - self.w)[:, None] * d)
        db = self.bxf - xc[bc]
        self.db = db
        dbS = _dot(db, self.bSf)
        self.bcoef = _dot(self.bSf, self.bSf) / dbS
        self.bDelta = db * self.bcoef[:, None]
        nb_ = self.bSf / np.sqrt(_dot(self.bSf, self.bSf))[:, None]
        self.db_t = db - _dot(db, nb_)[:, None] * nb_
        # least-squares gradient: grad = Minv r, r_k = sum c_k (phi_N - phi_P)
        wi = 1.0 / _dot(d, d)
        wb = 1.0 / _dot(db, db)
        M = np.zeros((n, 2, 2), dt)
        for a in range(2):
            for b in range(2):
                q = wi * d[:, a] * d[:, b]
                M[:, a, b] = t.Sown @ q + t.Snei @ q + t.Sb @ (wb * db[:, a] * db[:, b])
        det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
        Mi = np.empty_like(M)
        Mi[:, 0, 0], Mi[:, 1, 1] = M[:, 1, 1] / det, M[:, 0, 0] / det
        Mi[:, 0, 1], Mi[:, 1, 0] = -M[:, 0, 1] / det, -M[:, 1, 0] / det
        self.Minv = Mi
        self.cw = d * wi[:, None]               # (nf, 2)
        self.cwb = db * wb[:, None]             # (nb, 2)
        # wall distance: exact point-to-segment distance to the nearest wall
        # face (the face chosen on the real geometry, then frozen)
        self.wall_faces = wall_faces
        if wall_dist_index is None:
            from turbulence import wall_distance
            Preal = np.real(P)
            if np.iscomplexobj(P):
                raise ValueError("pass wall_dist_index for complex points")
            _, j = wall_distance(_RealMesh(Preal, topo, xc.real), wall_faces, return_index=True)
            wall_dist_index = j
        self.wall_dist_index = wall_dist_index
        fw = wall_faces[wall_dist_index]
        a_, b_ = P[t.ba[fw]], P[t.bb[fw]]
        AB = b_ - a_
        tpar = _dot(xc - a_, AB) / _dot(AB, AB)
        tpar = cclip(tpar, 0.0, 1.0)
        q = xc - (a_ + tpar[:, None] * AB)
        self.wall_d = cmax(np.sqrt(_dot(q, q)), 1e-30)

    # operators -----------------------------------------------------------
    def interp(self, phi):
        w = self.w if phi.ndim == 1 else self.w[:, None]
        return w * phi[self.t.own] + (1 - w) * phi[self.t.nei]

    def interp_skew(self, phi, grad):
        return self.interp(phi) + _dot(self.interp(grad), self.rskew)

    def sum_to_cells(self, fint, fbnd=None):
        s = self.t.Sdiv @ fint
        if fbnd is not None:
            s = s + self.t.Sb @ fbnd
        return s

    def grad(self, phi, phib):
        t = self.t
        dphi = phi[t.nei] - phi[t.own]
        dphib = phib - phi[t.bcell]
        r = np.empty((t.n, 2), np.result_type(phi.dtype, self.cw.dtype))
        for k in range(2):
            q = self.cw[:, k] * dphi
            r[:, k] = t.Sown @ q + t.Snei @ q + t.Sb @ (self.cwb[:, k] * dphib)
        Mi = self.Minv
        return np.c_[Mi[:, 0, 0] * r[:, 0] + Mi[:, 0, 1] * r[:, 1],
                     Mi[:, 1, 0] * r[:, 0] + Mi[:, 1, 1] * r[:, 1]]

    def grad_gg(self, phi, phib, grad_skew=None):
        phif = self.interp(phi) if grad_skew is None else self.interp_skew(phi, grad_skew)
        g = np.c_[self.sum_to_cells(phif * self.Sf[:, 0], phib * self.bSf[:, 0]),
                  self.sum_to_cells(phif * self.Sf[:, 1], phib * self.bSf[:, 1])]
        return g / self.vol[:, None]


class _RealMesh:
    """the attributes turbulence.wall_distance needs"""

    def __init__(self, P, topo, xc):
        tb = P[topo.bb] - P[topo.ba]
        self.bSf = np.c_[tb[:, 1], -tb[:, 0]]
        self.bxf = 0.5 * (P[topo.ba] + P[topo.bb])
        self.xc = xc


# ----------------------------------------------------------------------
# the discrete residual
# ----------------------------------------------------------------------
class Residual:
    """Discrete steady equations of a converged SimpleSolver / CoupledSolver
    (incompressible; laminar or SA; inlet / outlet / wall patches; optional
    point-vortex far field). Built from the solver object; evaluate(X) is
    complex-safe."""

    def __init__(self, solver, alpha, vortex=None, walls=("wall",), center=(0.25, 0.0),
                 ref_length=1.0):
        s = solver
        m = s.m
        if s.b_sym.any():
            raise NotImplementedError("adjoint: symmetry patches are not supported")
        if getattr(s, "dt", None) is not None:
            raise NotImplementedError("adjoint: steady solutions only")
        if hasattr(s, "mach"):
            raise NotImplementedError("adjoint: incompressible solver only")
        self.s = s
        self.topo = Topology(m)
        self.alpha = float(alpha)
        self.nu = float(s.nu)
        self.walls = list(walls)
        self.center = np.asarray(center, float)
        self.ref_length = ref_length
        self.vortex = vortex is not None
        t = self.topo
        self.n, self.nf, self.nb = t.n, t.nf, t.nb
        self.b_fixedU, self.b_outlet = s.b_fixedU.copy(), s.b_outlet.copy()
        self.b_wall, self.b_inlet = s.b_wall.copy(), s.b_inlet.copy()
        self.inlet_faces = np.where(self.b_inlet)[0]
        self.outlet_faces = np.where(self.b_outlet)[0]
        self.no = len(self.outlet_faces)
        self.Ub_wall = s.Ub.copy()                  # wall velocities (0) - inlet set from Gamma
        self.pb_out0 = s.pb_out.copy()
        self.second_order = s.second_order
        self.limiter = s.limiter
        self.limiter_eps = s.limiter_eps
        self.limiter_mask = None if s.limiter_mask is None else s.limiter_mask.copy()
        self.turb = s.turb is not None
        if self.turb:
            tb = s.turb
            self.nut_inf = tb.nut_inf
            self.gamma = None if tb.gamma is None else np.asarray(tb.gamma, float).copy()
            self.seed = None if tb.seed is None else (tb.seed[0].copy(), tb.seed[1].copy())
            self.beta_field = None if getattr(tb, "beta", None) is None else \
                np.asarray(tb.beta, float).copy()
            self.alpha_turb = tb.alpha
        self.wall_faces = np.where(self.b_wall)[0]
        # layout of X
        n, nf = self.n, self.nf
        k = 6 if self.turb else 5
        self.fields = ["u", "v", "p"] + (["nt"] if self.turb else []) + ["gx", "gy"]
        self.off = {f: i * n for i, f in enumerate(self.fields)}
        self.off["F"] = k * n
        self.off["Fo"] = k * n + nf
        self.off["G"] = k * n + nf + self.no
        self.N = self.off["G"] + (1 if self.vortex else 0)
        self.geo0 = Geometry(self.topo, m.points, self.wall_faces)

    # ------------------------------------------------------------------
    def state(self, gamma_vortex=0.0):
        """X from the solver's current (converged) fields."""
        s = self.s
        X = np.zeros(self.N)
        n = self.n
        X[self.off["u"]:self.off["u"] + n] = s.U[:, 0]
        X[self.off["v"]:self.off["v"] + n] = s.U[:, 1]
        X[self.off["p"]:self.off["p"] + n] = s.p
        if self.turb:
            X[self.off["nt"]:self.off["nt"] + n] = s.turb.nuT
        X[self.off["gx"]:self.off["gx"] + n] = s.gradp[:, 0]
        X[self.off["gy"]:self.off["gy"] + n] = s.gradp[:, 1]
        X[self.off["F"]:self.off["F"] + self.nf] = s.F
        X[self.off["Fo"]:self.off["Fo"] + self.no] = s.Fb[self.outlet_faces]
        if self.vortex:
            X[self.off["G"]] = gamma_vortex
        return X

    def unpack(self, X):
        n = self.n
        f = {k: X[self.off[k]:self.off[k] + n] for k in self.fields}
        f["F"] = X[self.off["F"]:self.off["F"] + self.nf]
        f["Fo"] = X[self.off["Fo"]:self.off["Fo"] + self.no]
        f["G"] = X[self.off["G"]] if self.vortex else 0.0
        return f

    # ------------------------------------------------------------------
    def boundary(self, geo, G, alpha):
        """inlet velocity (U_inf + point vortex), outlet pressure (Bernoulli)"""
        a = alpha * math.pi / 180.0
        Ub = np.array(self.Ub_wall, dtype=np.result_type(geo.bxf.dtype, np.asarray(G).dtype,
                                                         np.asarray(alpha).dtype, float))
        pb_out = np.array(self.pb_out0, dtype=Ub.dtype)
        ca, sa = np.cos(a), np.sin(a)
        for faces, inlet in ((self.inlet_faces, True), (self.outlet_faces, False)):
            xy = geo.bxf[faces]
            if self.vortex:
                dx, dy = xy[:, 0] - self.center[0], xy[:, 1] - self.center[1]
                k = G / (2 * math.pi * (dx**2 + dy**2))
                uv = np.c_[ca + k * dy, sa - k * dx]
            else:
                uv = np.c_[np.broadcast_to(ca, len(faces)), np.broadcast_to(sa, len(faces))]
            if inlet:
                Ub[faces] = uv
            elif self.vortex:
                pb_out[faces] = 0.5 * (1.0 - (uv**2).sum(1))
        return Ub, pb_out

    def _venkat(self, geo, phi, g, phib, eps2):
        """Venkatakrishnan limiter, as kernels._venkat (branches on real parts)"""
        t = geo.t
        own, nei, bc, n = t.own, t.nei, t.bcell, t.n
        # neighbour extrema (incl. the cell itself)
        cell = np.r_[np.arange(n), own, nei, bc]
        val = np.r_[phi, phi[nei], phi[own], phib]
        pmax = _group_select(cell, val, n, "max")
        pmin = _group_select(cell, val, n, "min")
        cl = np.r_[own, nei, bc]
        xx = np.r_[geo.xf, geo.xf, geo.bxf]
        d2 = _dot(g[cl], xx - geo.xc[cl])
        d1 = np.where(np.real(d2) > 0, pmax[cl] - phi[cl], pmin[cl] - phi[cl])
        num = (d1 * d1 + eps2) * d2 + 2.0 * d2 * d2 * d1
        den = d2 * (d1 * d1 + 2.0 * d2 * d2 + d1 * d2 + eps2)
        small = np.abs(np.real(d2)) <= 1e-30
        v = cclip(num / np.where(small, 1.0, den), 0.0, 1.0)
        v = np.where(small, 1.0, v)
        if self.limiter_mask is not None:
            v = np.where(self.limiter_mask[cl], v, 1.0)
        return _group_select(np.r_[np.arange(n), cl],
                             np.r_[np.ones(n, dtype=v.dtype), v], n, "min")

    # ------------------------------------------------------------------
    def evaluate(self, X, geo=None, alpha=None, nu=None, beta=None, parts=False):
        """R(X). geo: Geometry (moved / complex points), alpha (deg), nu:
        design parameters (complex for complex-step derivatives). beta:
        optional multiplier field on the SA production (field inversion)."""
        geo = self.geo0 if geo is None else geo
        alpha = self.alpha if alpha is None else alpha
        nu = self.nu if nu is None else nu
        t = geo.t
        n, own, nei, bc = t.n, t.own, t.nei, t.bcell
        f = self.unpack(X)
        U = np.c_[f["u"], f["v"]]
        p = f["p"]
        gradp = np.c_[f["gx"], f["gy"]]
        F, G = f["F"], f["G"]
        Ub, pb_out = self.boundary(geo, G, alpha)
        dt = np.result_type(Ub.dtype, X.dtype, geo.coef.dtype, np.asarray(nu).dtype)
        Ub, pb_out = Ub.astype(dt), pb_out.astype(dt)
        o, fixed = self.b_outlet, self.b_fixedU
        ofc = self.outlet_faces
        # boundary face fluxes: fixed faces Ub.S, outlet faces unknowns
        Fb = _dot(Ub, geo.bSf)
        Fb = Fb.astype(np.result_type(Fb.dtype, X.dtype))
        Fb[ofc] = f["Fo"]

        # velocity gradients with the outlet extrapolation Ub = U_P + gU.db_t
        # (fixed point at outlet cells, as the solver at convergence)
        gU = [geo.grad(U[:, k], Ub[:, k]) for k in range(2)]
        for it in range(60):
            Ube = Ub.copy()
            for k in range(2):
                Ube[ofc, k] = U[bc[ofc], k] + _dot(gU[k][bc[ofc]], geo.db_t[ofc])
            gn = [geo.grad(U[:, k], Ube[:, k]) for k in range(2)]
            ch = max(np.abs(gn[k] - gU[k]).max() for k in range(2))
            gU = gn
            if ch < 1e-15 * max(np.abs(gn[0]).max(), 1.0):
                break
        Ub = Ube
        gu, gv = gU

        # ---- viscosity --------------------------------------------------
        if self.turb:
            nt = f["nt"]
            chi = nt / nu
            fv1 = chi**3 / (chi**3 + CV1**3)
            nut = nt * fv1 if self.gamma is None else nt * fv1 * self.gamma
            nu_f = nu + cmax(geo.interp(nut), 0.0)
            nbv = np.where(self.b_wall, 0.0, nt[bc])
            nbv = np.where(self.b_inlet, self.nut_inf * nu / self.nu, nbv)
            chib = nbv / nu
            nu_b = nu + nbv * chib**3 / (chib**3 + CV1**3)
        else:
            nu_f = nu + 0.0 * geo.coef
            nu_b = nu + 0.0 * geo.bcoef

        # ---- momentum ---------------------------------------------------
        D = nu_f * geo.coef
        Fp, Fm = cpos(F), cpos(-F)
        a_on = -D - Fm
        a_no = -D - Fp
        diag = t.Sown @ (D + Fp) + t.Snei @ (D + Fm)
        Db = nu_b * geo.bcoef
        diag = diag + t.Sb @ (np.where(fixed, Db, 0.0) + np.where(o, cpos(Fb), 0.0))
        gf = [geo.interp(gU[k]) for k in range(2)]           # (nf,2) each
        SmD = geo.Sf - geo.Delta
        src = []
        if self.second_order and self.limiter is not None:
            uref = _abs_max(Ub[fixed])
            eps2 = (self.limiter_eps * cmax(uref, 1e-30)) ** 2
            gk = [gU[k] * self._venkat(geo, U[:, k], gU[k], Ub[:, k], eps2)[:, None]
                  for k in range(2)]
        else:
            gk = gU
        up = np.where(np.real(F) >= 0, own, nei)
        dxu = geo.xf - geo.xc[up]
        for k in range(2):
            fl = nu_f * _dot(gf[k], SmD)                     # non-orthogonal diffusion
            if self.turb:                                    # transpose stress
                fl = fl + nu_f * (gf[0][:, k] * geo.Sf[:, 0] + gf[1][:, k] * geo.Sf[:, 1])
            if self.second_order:
                fl = fl - F * _dot(gk[k][up], dxu)            # deferred correction
            bsrc = np.where(fixed, Db * Ub[:, k] - Fb * Ub[:, k] +
                            nu_b * _dot(gU[k][bc], geo.bSf - geo.bDelta), 0.0)
            bsrc = bsrc + np.where(o, -cpos(Fb) * _dot(gU[k][bc], geo.db_t)
                                   - cneg(Fb) * Ub[:, k], 0.0)
            if self.turb:
                bsrc = bsrc + nu_b * (gU[0][bc, k] * geo.bSf[:, 0] + gU[1][bc, k] * geo.bSf[:, 1])
            src.append(geo.sum_to_cells(fl, bsrc) - gradp[:, k] * geo.vol)
        RU = [diag * U[:, k] + t.Sown @ (a_on * U[nei, k]) + t.Snei @ (a_no * U[own, k]) - src[k]
              for k in range(2)]

        # ---- Rhie-Chow face fluxes (converged form, independent of alpha_u)
        Dp = geo.vol / diag
        Df = geo.interp(Dp)
        gp_f = geo.interp(gradp)
        UbarS = geo.interp_skew(U[:, 0], gu) * geo.Sf[:, 0] + geo.interp_skew(U[:, 1], gv) * geo.Sf[:, 1]
        RF = F - (UbarS - Df * ((p[nei] - p[own]) * geo.coef - _dot(gp_f, geo.Delta)))
        c = bc[ofc]
        Ubo = U[c] + np.c_[_dot(gu[c], geo.db_t[ofc]), _dot(gv[c], geo.db_t[ofc])]
        RFo = f["Fo"] - (_dot(Ubo, geo.bSf[ofc]) -
                         Dp[c] * ((pb_out[ofc] - p[c]) * geo.bcoef[ofc] -
                                  _dot(gradp[c], geo.bDelta[ofc])))
        # ---- continuity -----------------------------------------------------
        Rc = geo.sum_to_cells(F, Fb)
        # ---- pressure gradient (Green-Gauss, skew-corrected with itself) ----
        pb = p[bc] + _dot(gradp[bc], geo.db)
        pb = np.where(o, pb_out, pb)
        Rg = gradp - geo.grad_gg(p, pb, grad_skew=gradp)

        out = {"u": RU[0], "v": RU[1], "p": Rc, "gx": Rg[:, 0], "gy": Rg[:, 1], "F": RF,
               "Fo": RFo}
        # ---- Spalart-Allmaras ----------------------------------------------
        if self.turb:
            out["nt"] = self._sa(geo, nt, nbv, gU, F, Fb, nu, beta)
        if self.vortex:
            out["G"] = G - 0.5 * sum(self.forces(X, geo, alpha, nu, Ub=Ub)["Cl_w"])
        if parts:
            return out
        R = np.zeros(self.N, dtype=np.result_type(*[np.asarray(v).dtype for v in out.values()]))
        for k, v in out.items():
            R[self.off[k]:self.off[k] + np.size(v)] = v
        return R

    def _sa(self, geo, nt, nbv, gU, F, Fb, nu, beta=None):
        t = geo.t
        own, nei, bc, n = t.own, t.nei, t.bcell, t.n
        gu, gv = gU
        g = geo.grad(nt, nbv)
        Omega = cabs(gv[:, 0] - gu[:, 1])
        d2 = geo.wall_d**2
        chi = nt / nu
        fv1 = chi**3 / (chi**3 + CV1**3)
        fv2 = 1.0 - chi / (1.0 + chi * fv1)
        Sbar = nt * fv2 / (KAPPA**2 * d2)
        den = (CV3 - 2 * CV2) * Omega - Sbar
        lim = Omega + Omega * (CV2**2 * Omega + CV3 * Sbar) / np.where(np.real(den) == 0, 1.0, den)
        St = np.where(np.real(Sbar) >= -CV2 * np.real(Omega), Omega + Sbar, lim)
        rr = np.where(np.real(St) > 0, nt / np.where(np.real(St) > 0, St, 1.0) / (KAPPA**2 * d2), 10.0)
        rr = np.where(np.isfinite(np.real(rr)), rr, 10.0)
        rr = cmin(rr, 10.0)
        gg = rr + CW2 * (rr**6 - rr)
        fw = gg * ((1 + CW3**6) / (gg**6 + CW3**6)) ** (1.0 / 6.0)
        prod = CB1 * St * nt
        if self.gamma is not None:
            prod = prod * self.gamma
        if beta is None:
            beta = self.beta_field
        if beta is not None:
            prod = prod * beta
        S0 = prod - CW1 * fw * nt**2 / d2
        nuTf = cmax(geo.interp(nt), 0.0)
        kf = (1.0 + CB2) * (nu + nuTf)
        k_own = cmax(kf - CB2 * (nu + nt[own]), 0.0) / SIGMA * geo.coef
        k_nei = cmax(kf - CB2 * (nu + nt[nei]), 0.0) / SIGMA * geo.coef
        Fp, Fm = cpos(F), cpos(-F)
        a_on = -k_own - Fm
        a_no = -k_nei - Fp
        diag = t.Sown @ (k_own + Fp) + t.Snei @ (k_nei + Fm)
        gf = geo.interp(g)
        SmD = geo.Sf - geo.Delta
        cno = kf / SIGMA * _dot(gf, SmD)
        lap_no = geo.sum_to_cells(_dot(gf, SmD))
        src = S0 * geo.vol + geo.sum_to_cells(cno) - CB2 / SIGMA * (nu + nt) * lap_no
        dirich = self.b_wall | self.b_inlet
        Dbk = cmax((1.0 + CB2) * (nu + nbv) - CB2 * (nu + nt[bc]), 0.0) / SIGMA * geo.bcoef
        diag = diag + t.Sb @ (np.where(dirich, Dbk, 0.0) + np.where(~dirich, cpos(Fb), 0.0))
        src = src + t.Sb @ (np.where(dirich, Dbk * nbv - Fb * nbv, 0.0) +
                            np.where(~dirich, -cneg(Fb) * nbv, 0.0))
        R = diag * nt + t.Sown @ (a_on * nt[nei]) + t.Snei @ (a_no * nt[own]) - src
        if self.seed is not None:
            idx, val = self.seed
            pen = 1e6 * diag[idx] / self.alpha_turb
            R = R.copy()
            R[idx] = R[idx] + pen * (nt[idx] - cmax(nt[idx], val))
        return R

    # ------------------------------------------------------------------
    def forces(self, X, geo=None, alpha=None, nu=None, Ub=None, ref=None):
        """Cl, Cd, Cm (and per wall patch) from X, as postproc.wall_data:
        wall pressure p_P + grad p_P . db, shear nu (U_P - U_wall)_t / |dn|."""
        geo = self.geo0 if geo is None else geo
        alpha = self.alpha if alpha is None else alpha
        nu = self.nu if nu is None else nu
        f = self.unpack(X)
        U = np.c_[f["u"], f["v"]]
        gradp = np.c_[f["gx"], f["gy"]]
        a = alpha * math.pi / 180.0
        drag = np.array([np.cos(a), np.sin(a)])
        lift = np.array([-np.sin(a), np.cos(a)])
        ref = (0.25 * self.ref_length, 0.0) if ref is None else ref
        q = 0.5
        L = self.ref_length
        tot = {"Cl": 0.0, "Cd": 0.0, "Cm": 0.0, "Cd_pressure": 0.0, "Cd_viscous": 0.0,
               "Cl_w": [], "faces": {}}
        for w in self.walls:
            idx = self.topo.patch_faces[w]
            c = self.topo.bcell[idx]
            pb = f["p"][c] + _dot(gradp[c], geo.db[idx])
            S = geo.bSf[idx]
            Smag = np.sqrt(_dot(S, S))
            n_out = S / Smag[:, None]
            dn = -_dot(geo.db[idx], n_out)
            du = U[c] - self.Ub_wall[idx]
            du_t = du - _dot(du, n_out)[:, None] * n_out
            tau = nu * du_t / cabs(dn)[:, None]
            Fpf = pb[:, None] * S
            Fvf = tau * Smag[:, None]
            Ff = Fpf + Fvf
            r = geo.bxf[idx] - np.asarray(ref)
            Mz = r[:, 0] * Ff[:, 1] - r[:, 1] * Ff[:, 0]
            per = {"Cl": (Ff @ lift) / (q * L), "Cd": (Ff @ drag) / (q * L),
                   "Cm": -Mz / (q * L**2), "Cd_pressure": (Fpf @ drag) / (q * L),
                   "Cd_viscous": (Fvf @ drag) / (q * L), "Cp": pb / q}
            tot["faces"][w] = (c, per)
            for k in ("Cl", "Cd", "Cm", "Cd_pressure", "Cd_viscous"):
                tot[k] = tot[k] + per[k].sum()
            tot["Cl_w"].append(per["Cl"].sum())
        return tot

    # ------------------------------------------------------------------
    def objective_gradient(self, X, name, geo=None, weights=None):
        """dI/dX (dense vector, nonzero on the wall cells only): every wall
        face's contribution depends on its own cell's u, v, p, dp/dx, dp/dy.
        weights: {wall patch: per-face weights} -> d(sum_f w_f q_f)/dX for a
        per-face quantity q (e.g. "Cp" for a pressure-matching objective)."""
        g = np.zeros(self.N)
        for fld in ("u", "v", "p", "gx", "gy"):
            Xc = X.astype(complex)
            sl = slice(self.off[fld], self.off[fld] + self.n)
            Xc[sl] += 1j * H_CS
            fo = self.forces(Xc, geo)
            for w, (c, per) in fo["faces"].items():
                d = np.imag(per[name]) / H_CS
                if weights is not None:
                    d = d * weights[w]
                np.add.at(g[sl], c, d)
        return g

    def objective(self, X, name, geo=None, alpha=None, nu=None):
        return self.forces(X, geo, alpha, nu)[name]


def _abs_max(Ub):
    """max |component| over the fixed-velocity faces (the limiter's U_ref),
    complex-safe: the entry with the largest |real part|"""
    v = Ub.ravel()
    i = int(np.argmax(np.abs(np.real(v))))
    return cabs(v[i])


def _group_select(cell, val, n, how):
    """per-cell max / min of val over the entries of each cell, returning the
    (possibly complex) selected entry"""
    r = np.real(val)
    order = np.lexsort((r if how == "max" else -r, cell))
    # last entry of each cell group = max (or min with the negated key)
    cs = cell[order]
    last = np.r_[cs[1:] != cs[:-1], True]
    sel = order[last]
    out = np.empty(n, dtype=val.dtype)
    out[cell[sel]] = val[sel]
    return out


# ----------------------------------------------------------------------
# sparsity, colouring, Jacobian
# ----------------------------------------------------------------------
@kernels.njit(cache=True)
def _greedy_colour(c_ptr, c_idx, r_ptr, r_idx, ncols):
    """Column colouring: columns sharing a row get different colours
    (Curtis-Powell-Reid). c_*: rows of each column, r_*: columns of each row."""
    colour = -np.ones(ncols, np.int64)
    mark = -np.ones(ncols + 1, np.int64)
    for j in range(ncols):
        for a in range(c_ptr[j], c_ptr[j + 1]):
            r = c_idx[a]
            for b in range(r_ptr[r], r_ptr[r + 1]):
                c = colour[r_idx[b]]
                if c >= 0:
                    mark[c] = j
        c = 0
        while mark[c] == j:
            c += 1
        colour[j] = c
    return colour


class Jacobian:
    """dR/dX by coloured complex-step differences."""

    def __init__(self, res, radius=2):
        self.res = res
        t = res.topo
        n, nf, nb = t.n, t.nf, t.nb
        # cell adjacency (with the cells themselves), radius `radius`
        A = sp.csr_matrix((np.ones(2 * nf + n), (np.r_[t.own, t.nei, np.arange(n)],
                                                  np.r_[t.nei, t.own, np.arange(n)])), (n, n))
        Ar = A.copy()
        for _ in range(radius - 1):
            Ar = Ar @ A
        Ar.data[:] = 1.0
        # home cells of every unknown / residual (faces: owner and neighbour)
        rows, cols = [], []
        for fld in res.fields:
            o = res.off[fld]
            rows.append(o + np.arange(n)); cols.append(np.arange(n))
        o = res.off["F"]
        rows += [o + np.arange(nf), o + np.arange(nf)]; cols += [t.own, t.nei]
        o = res.off["Fo"]
        rows.append(o + np.arange(res.no)); cols.append(t.bcell[res.outlet_faces])
        Nl = res.off["G"]                         # local unknowns (Gamma: separately)
        H = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows),
                                                                 np.concatenate(cols))), (Nl, n))
        P = (H @ Ar @ H.T).tocsr()
        P.data[:] = 1.0
        self.P = P
        Pc = P.tocsc()
        self.colour = _greedy_colour(Pc.indptr.astype(np.int64), Pc.indices.astype(np.int64),
                                     P.indptr.astype(np.int64), P.indices.astype(np.int64), Nl)
        self.ncolours = int(self.colour.max()) + 1
        self.Nl = Nl

    def compute(self, X, geo=None, verbose=False):
        res = self.res
        t0 = time.time()
        P = self.P
        rr = np.repeat(np.arange(self.Nl), np.diff(P.indptr))
        cc = P.indices
        col_of = self.colour[cc]
        order = np.argsort(col_of, kind="stable")
        bounds = np.searchsorted(col_of[order], np.arange(self.ncolours + 1))
        vals = np.zeros(len(cc))
        for c in range(self.ncolours):
            seed = (self.colour == c).astype(float)
            Xc = X.astype(complex)
            Xc[:self.Nl] += 1j * H_CS * seed
            d = np.imag(res.evaluate(Xc, geo)) / H_CS
            e = order[bounds[c]:bounds[c + 1]]
            vals[e] = d[rr[e]]
            if verbose and c % 50 == 0:
                print(f"    colour {c}/{self.ncolours} ({time.time() - t0:.1f}s)", flush=True)
        N = res.N
        J = sp.csr_matrix((vals, (rr, cc)), (N, N))
        if res.vortex:
            g = res.off["G"]
            Xc = X.astype(complex)
            Xc[g] += 1j * H_CS
            dcol = np.imag(res.evaluate(Xc, geo)) / H_CS        # dR/dGamma (dense column)
            nz = np.nonzero(dcol)[0]
            # row of the Gamma equation: dGamma - 0.5 dCl/dX
            grow = -0.5 * res.objective_gradient(X, "Cl", geo)
            nzr = np.nonzero(grow[:g])[0]
            extra = sp.csr_matrix((np.r_[dcol[nz], grow[nzr]],
                                   (np.r_[nz, np.full(len(nzr), g)], np.r_[np.full(len(nz), g), nzr])),
                                  (N, N))
            J = J + extra
        J.eliminate_zeros()
        self.time = time.time() - t0
        return J

    def check(self, J, X, geo=None, seed=0):
        """relative difference between J v and a direct complex-step
        directional derivative for a random v (catches a too-small stencil)"""
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.res.N)
        Xc = X.astype(complex) + 1j * H_CS * v
        jv = np.imag(self.res.evaluate(Xc, geo)) / H_CS
        return float(np.abs(J @ v - jv).max() / max(np.abs(jv).max(), 1e-300))


# ----------------------------------------------------------------------
# adjoint solve and gradients
# ----------------------------------------------------------------------
class Adjoint:
    """Adjoint of one converged flow solution.

        adj = Adjoint(solver, alpha, vortex=vf, walls=["wall"])
        adj.solve(["Cl", "Cd", "Cm"])
        adj.gradient_alpha()            -> {"Cl": dCl/dalpha (per degree), ...}
        adj.gradient_shape(dP)          -> {"Cl": [...], ...} for mesh-point
                                           displacement fields dP[k] (npoints, 2)
    """

    def __init__(self, solver, alpha, vortex=None, walls=("wall",), center=(0.25, 0.0),
                 verbose=False):
        self.res = Residual(solver, alpha, vortex, walls, center)
        self.X = self.res.state(vortex.gamma if vortex is not None else 0.0)
        self.verbose = verbose
        self.psi = {}
        self.timing = {}

    def residual_check(self):
        """|R(X*)| per block relative to the size of the terms (should be at
        the level of the solver's convergence)"""
        out = self.res.evaluate(self.X, parts=True)
        n = {}
        for k, v in out.items():
            n[k] = float(np.abs(v).max())
        return n

    def build(self, jacobian=None, factorise=True):
        """dR/dX (coloured complex step) and its transposed LU. jacobian: a
        Jacobian object of the same mesh (reuses the colouring).

        Linear solver: SuperLU (partial pivoting, COLAMD ordering). The
        matrix is a saddle-point system (continuity rows have no diagonal)
        with pivots over ~15 orders of magnitude (eddy-viscosity rows scale
        with nu); MKL PARDISO's static pivoting returned wrong solutions
        here even with weighted matching, so it is not used."""
        t0 = time.time()
        self.jac = jacobian if jacobian is not None else Jacobian(self.res)
        self.timing["colouring"] = time.time() - t0
        self.J = self.jac.compute(self.X, verbose=self.verbose)
        self.timing["jacobian"] = self.jac.time
        if factorise:
            t0 = time.time()
            self.lu = splu(self.J.T.tocsc(), permc_spec="COLAMD")
            self.timing["factorisation"] = time.time() - t0
        return self

    def solve(self, objectives=("Cl", "Cd", "Cm"), precond=None, rtol=1e-10):
        """psi for each objective. precond: the `lu` of an earlier Adjoint on
        the SAME mesh (e.g. the previous field-inversion iteration): GMRES
        on (dR/dX)^T preconditioned by the old factorisation, which saves
        the factorisation while the Jacobian changes little; falls back to
        a fresh LU when GMRES does not converge."""
        from scipy.sparse.linalg import LinearOperator, gmres
        if not hasattr(self, "J"):
            self.build(factorise=precond is None)
        t0 = time.time()
        self.gmres_its = 0
        for name in objectives:
            g = self.res.objective_gradient(self.X, name)
            if precond is not None and not hasattr(self, "lu"):
                A = self.J.T.tocsr()
                N = A.shape[0]
                its = [0]
                x, info = gmres(A, -g, M=LinearOperator((N, N), matvec=precond.solve), rtol=rtol,
                                atol=0.0, restart=60, maxiter=3,
                                callback=lambda r: its.__setitem__(0, its[0] + 1),
                                callback_type="pr_norm")
                self.gmres_its += its[0]
                if info == 0:
                    self.psi[name] = x
                    continue
                t1 = time.time()
                self.lu = splu(self.J.T.tocsc(), permc_spec="COLAMD")
                self.timing["factorisation"] = time.time() - t1
            if not hasattr(self, "lu"):
                t1 = time.time()
                self.lu = splu(self.J.T.tocsc(), permc_spec="COLAMD")
                self.timing["factorisation"] = time.time() - t1
            self.psi[name] = self.lu.solve(-g)
        self.timing["adjoint_solves"] = time.time() - t0
        return self.psi

    def solve_beta(self, name, g):
        """adjoint for an arbitrary objective gradient g = dI/dX"""
        if not hasattr(self, "lu"):
            self.build()
        self.psi[name] = self.lu.solve(-g)
        return self.psi[name]

    def _partials(self, geo=None, alpha=None, nu=None, beta=None):
        """dR/dD and dI/dD at fixed X for ONE complex-perturbed design"""
        R = self.res.evaluate(self.X.astype(complex), geo, alpha, nu, beta)
        fo = self.res.forces(self.X.astype(complex), geo, alpha, nu)
        return np.imag(R) / H_CS, {k: float(np.imag(fo[k]) / H_CS) for k in ("Cl", "Cd", "Cm")}

    def total(self, dR, dI):
        return {k: dI[k] + float(self.psi[k] @ dR) for k in self.psi}

    def gradient_alpha(self):
        """d/dalpha (per degree)"""
        dR, dI = self._partials(alpha=self.res.alpha + 1j * H_CS)
        return self.total(dR, dI)

    def gradient_nu(self):
        """d/d(nu) - for d/d(Re) multiply by -nu/Re"""
        dR, dI = self._partials(nu=self.res.nu + 1j * H_CS)
        return self.total(dR, dI)

    def gradient_shape(self, dP_list):
        """dP_list: mesh-point velocity fields dP_k (npoints, 2), one per
        design variable (from a mesh deformation). Returns {name: array}."""
        out = {k: [] for k in self.psi}
        P0 = np.real(self.res.geo0.points)
        jidx = self.res.geo0.wall_dist_index
        for dP in dP_list:
            geo = Geometry(self.res.topo, P0 + 1j * H_CS * np.asarray(dP), self.res.wall_faces,
                           jidx)
            dR, dI = self._partials(geo=geo)
            for k, v in self.total(dR, dI).items():
                out[k].append(v)
        return {k: np.array(v) for k, v in out.items()}

    def gradient_beta(self, name):
        """dI/dbeta_i for the SA production multiplier field beta (solver's
        turb.beta, 1 if unset):
        dR_nt,i/dbeta_i is cell-local, so the whole field gradient is one
        complex residual evaluation"""
        b0 = self.res.beta_field if self.res.beta_field is not None else np.ones(self.res.n)
        beta = b0 + 1j * H_CS
        R = self.res.evaluate(self.X.astype(complex), beta=beta, parts=True)
        dRnt = np.imag(R["nt"]) / H_CS
        o = self.res.off["nt"]
        return self.psi[name][o:o + self.res.n] * dRnt


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
    import cst
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


# ----------------------------------------------------------------------
# primal solution for the adjoint
# ----------------------------------------------------------------------
def setup_case(spec, re, alpha, level=1.0, solver="simple", regime="auto", fvm_override=None,
               size_scale=None, base=None):
    """mesh + solver object (not solved) as polar.run_point builds them.
    base: reuse a mesh (polar.make_case)"""
    import polar as pl
    from simple_solver import SimpleSolver
    fvm, mesh, patches, bcs, U, turb, info = pl.make_case(spec, re, alpha, level, regime=regime,
                                                          size_scale=size_scale, base=base)
    if fvm_override is not None:
        from fvmesh import FVMesh
        fvm = FVMesh(fvm_override, mesh["cells"], patches)
    if solver == "coupled":
        from coupled import CoupledSolver
        s = CoupledSolver(fvm, bcs, 1.0 / re, U0=U, turbulence=turb)
    else:
        s = SimpleSolver(fvm, bcs, 1.0 / re, U0=U, turbulence=turb)
    return s, fvm, mesh, U, turb, info


def solve_primal(spec, re, alpha, level=1.0, transition="off", ncrit=9.0, tol=1e-10,
                 max_iter=20000, init=None, solver="simple", verbose=False, regime="auto",
                 fvm_override=None, size_scale=None, base=None):
    """Steady flow, converged tightly (the adjoint linearises about it):
    as polar.run_point, then the point-vortex far field is updated every
    iteration and the residuals are driven to `tol` (or until they stall).
    transition: "off", "free" (e^N; frozen for the adjoint) or {"upper": x,
    "lower": x}. Returns dict(solver, vortex, transition, info, fvm, mesh)."""
    s, fvm, mesh, U, turb, info = setup_case(spec, re, alpha, level, solver, regime,
                                             fvm_override, size_scale, base)
    walls = info["walls"]
    if init is not None:
        if len(init["U"]) == fvm.ncells:
            s.set_initial(init["U"], init["p"], init.get("nuT"))
        else:
            from scipy.spatial import cKDTree
            _, j = cKDTree(init["xc"]).query(fvm.xc)
            s.set_initial(init["U"][j], init["p"][j],
                          None if init.get("nuT") is None else init["nuT"][j])
    vf = None
    if turb:
        import validate_naca0012_sa as v
        vf = v.VortexFarfield(s, alpha, wall=walls,
                              center=(0.25 * info.get("ref_chord", 1.0), 0.0), every=50)
        if init is not None:
            vf.update(0)
    tr = None
    if turb and transition != "off":
        import transition as T
        if isinstance(transition, dict):
            tr = T.Transition(s, mesh, re, mode="fixed", x_tr=transition)
            s.solve(tol=1e-6, max_iter=max_iter, verbose=False, callback=vf.update)
        else:
            tr = T.Transition(s, mesh, re, mode="free", ncrit=ncrit, x_tr=None)
            T.solve_with_transition(s, tr, tol=1e-6, max_iter=max_iter,
                                    callback=lambda it, r: vf.update(it))
    else:
        s.solve(tol=1e-6, max_iter=max_iter, verbose=False,
                callback=(vf.update if vf is not None else None))
    # tight convergence with the far field updated every iteration
    if vf is not None:
        vf.every = 1
    hist = []

    def cb(it, r):
        if vf is not None:
            vf.update(it)
        hist.append(max(r.values()))
        # stop when the residual has stalled (round-off / limiter floor)
        if len(hist) > 400 and min(hist[-200:]) > 0.7 * min(hist[-400:-200]):
            return True
        return None
    s.solve(tol=tol, max_iter=max_iter, verbose=verbose, callback=cb)
    if vf is not None:
        vf.update(0)
    return {"solver": s, "vortex": vf, "transition": tr, "info": info, "fvm": fvm, "mesh": mesh,
            "walls": walls, "U": U, "residual": hist[-1] if hist else None,
            "iterations": len(s.history)}


def gradients(spec, re, alpha, level=1.0, transition="off", objectives=("Cl", "Cd", "Cm"),
              shape=True, order=8, verbose=False, primal=None, init=None, dP=None, base=None):
    """Flow + adjoint: dI/dalpha, dI/dRe and (shape=True) dI/d(CST
    coefficient) for every objective. init: warm start (polar state dict).
    Returns a dict (keys starting with '_' hold the solver objects)."""
    t0 = time.time()
    pr = primal or solve_primal(spec, re, alpha, level, transition, init=init, base=base)
    s = pr["solver"]
    t_primal = time.time() - t0
    adj = Adjoint(s, alpha, pr["vortex"], pr["walls"], verbose=verbose)
    rc = adj.residual_check()
    adj.solve(objectives)
    import polar as pl
    c = pl.coefficients(s, pr["walls"], pr["U"])[0]
    out = {"airfoil": spec, "re": re, "alpha": alpha, "level": level,
           "cells": int(s.m.ncells), "unknowns": int(adj.res.N),
           "jacobian_nnz": int(adj.J.nnz), "colours": int(adj.jac.ncolours),
           "Cl": c["Cl"], "Cd": c["Cd"], "Cm": c["Cm"],
           "residual_check": rc, "primal_residual": pr["residual"],
           "d_dalpha": adj.gradient_alpha()}
    gnu = adj.gradient_nu()
    out["d_dRe"] = {k: -v * adj.res.nu / re for k, v in gnu.items()}
    if shape:
        t1 = time.time()
        if dP is None:
            defo = IDWDeformation(s.m, pr["walls"])
            dP = cst_wall_velocities(defo, order)
        g = adj.gradient_shape(dP)
        out["d_dcst"] = {k: v.tolist() for k, v in g.items()}
        out["cst_order"] = order
        adj.timing["shape"] = time.time() - t1
    out["timing"] = {"primal": t_primal, **adj.timing, "total": time.time() - t0}
    out["_adjoint"] = adj
    out["_primal"] = pr
    return out


# ----------------------------------------------------------------------
# finite-difference check
# ----------------------------------------------------------------------
def fd_check(spec, re, alpha, level=0.5, transition="off", d_alpha=0.05, shape_vars=(2, 12),
             order=8, eps_shape=2e-4, verbose=True):
    """Adjoint gradients against central finite differences of complete
    flow solutions (same mesh topology: the shape perturbation deforms the
    mesh with the same IDW map, transition locations frozen)."""
    log = print if verbose else (lambda *a, **k: None)
    tr_frozen = transition
    g = gradients(spec, re, alpha, level, transition, shape=True, order=order)
    pr = g["_primal"]
    adj = g["_adjoint"]
    if isinstance(transition, str) and transition == "free":
        tr_frozen = dict(pr["transition"].x_tr)       # FD with the transition frozen
    state = {"xc": pr["fvm"].xc, "U": pr["solver"].U, "p": pr["solver"].p,
             "nuT": None if pr["solver"].turb is None else pr["solver"].turb.nuT}
    log(f"[check] {spec} Re {re:.3g} alpha {alpha}: Cl {g['Cl']:.6f} Cd {g['Cd']:.6f} "
        f"Cm {g['Cm']:+.6f}; residual check {g['residual_check']}")
    rows = []

    def coeffs(**kw):
        p = solve_primal(spec, re, kw.get("alpha", alpha), level, tr_frozen, init=state,
                         fvm_override=kw.get("points"))
        import polar as pl
        return pl.coefficients(p["solver"], p["walls"], p["U"])[0]
    # angle of attack
    cp_ = coeffs(alpha=alpha + d_alpha)
    cm_ = coeffs(alpha=alpha - d_alpha)
    for k in ("Cl", "Cd", "Cm"):
        fd = (cp_[k] - cm_[k]) / (2 * d_alpha)
        rows.append({"variable": "alpha", "objective": k, "adjoint": g["d_dalpha"][k], "fd": fd})
    # shape (CST coefficients)
    defo = IDWDeformation(pr["solver"].m, pr["walls"])
    dP = cst_wall_velocities(defo, order)
    for j in shape_vars:
        P0 = np.asarray(pr["fvm"].points)
        cp_ = coeffs(points=P0 + eps_shape * dP[j])
        cm_ = coeffs(points=P0 - eps_shape * dP[j])
        for k in ("Cl", "Cd", "Cm"):
            fd = (cp_[k] - cm_[k]) / (2 * eps_shape)
            rows.append({"variable": f"cst[{j}]", "objective": k,
                         "adjoint": g["d_dcst"][k][j], "fd": fd})
    for r in rows:
        r["rel_diff"] = abs(r["adjoint"] - r["fd"]) / max(abs(r["fd"]), 1e-12)
        log(f"  d{r['objective']}/d{r['variable']:8s} adjoint {r['adjoint']:+.6e}  "
            f"FD {r['fd']:+.6e}  rel. diff {r['rel_diff']:.1e}")
    return {"gradients": {k: v for k, v in g.items() if not k.startswith("_")}, "rows": rows}


# ----------------------------------------------------------------------
def _jsonable(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gradient", "check"):
        p = sub.add_parser(name)
        p.add_argument("airfoil")
        p.add_argument("--re", type=float, required=True)
        p.add_argument("--alpha", type=float, default=2.0)
        p.add_argument("--level", type=float, default=0.7 if name == "gradient" else 0.5)
        p.add_argument("--transition", default="off",
                       help="off (fully turbulent), free (e^N, frozen in the adjoint) or "
                            "'x_upper,x_lower' (fixed)")
        p.add_argument("--objective", default="Cl,Cd,Cm")
        p.add_argument("--out")
    a = ap.parse_args(argv)
    tr = a.transition
    if "," in tr:
        xu, xl = (float(v) for v in tr.split(","))
        tr = {"upper": xu, "lower": xl}
    if a.cmd == "gradient":
        g = gradients(a.airfoil, a.re, a.alpha, a.level, tr,
                      objectives=a.objective.split(","), verbose=True)
        print(f"Cl {g['Cl']:.5f}  Cd {g['Cd']:.6f}  Cm {g['Cm']:+.5f}   "
              f"({g['cells']} cells, {g['unknowns']} unknowns, {g['colours']} colours)")
        for k, v in g["d_dalpha"].items():
            print(f"  d{k}/dalpha = {v:+.6e} per degree   d{k}/dRe = {g['d_dRe'][k]:+.3e}")
        print("  timing: " + ", ".join(f"{k} {v:.1f}s" for k, v in g["timing"].items()))
        if a.out:
            Path(a.out).write_text(json.dumps(_jsonable(g), indent=1))
    else:
        r = fd_check(a.airfoil, a.re, a.alpha, a.level, tr)
        if a.out:
            Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
