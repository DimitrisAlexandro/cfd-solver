"""Steady incompressible RANS in 3D: cell-centred finite volume on polyhedral meshes,
collocated SIMPLEC, the same discretisation as the validated 2D solver
(simple_solver.py), with the heavy loops in C++ (native.py):

  convection   upwind implicit + deferred correction to linear upwind, optional
               Venkatakrishnan limiter (turbulent runs: on non-hexahedral cells);
               bounded form div(F U) - U div(F) (the momentum diagonal stays
               dominant in cells with a net inflow during the transient)
  diffusion    over-relaxed orthogonal part implicit, non-orthogonal part explicit;
               full viscous stress incl. the transpose term with eddy viscosity
  gradients    weighted least squares; Green-Gauss (conservative) for pressure
  face fluxes  Rhie-Chow + Majumdar (converged answer independent of alpha_u)
  pressure     SIMPLEC correction, CG + algebraic multigrid (V-cycle in C++) on the
               diagonally scaled system (every cell's mass balance weighs alike)
  turbulence   Spalart-Allmaras (SA-noft2, negative-S~ limiter), |vorticity| in 3D

Boundary conditions, per patch:
  {'type': 'wall', 'U': (0,0,0)}      no slip
  {'type': 'symmetry'}                no normal flow, no shear
  {'type': 'inlet', 'U': (1,0,0)}     fixed velocity (nuTilde = 3 nu)
  {'type': 'outlet', 'p': 0}          fixed pressure, zero-gradient velocity
  {'type': 'farfield', 'U': U_inf}    inlet where U_inf . n < 0, outlet elsewhere

Note: steady RANS on blunt trailing edges can have more than one discrete steady
state; the defaults (bounded convection, no first-order start, plain p' gradient)
follow the 2D solver's path and land on its validated state.
"""
import time

import numpy as np

from . import native as N

CB1, CB2, SIGMA, KAPPA = 0.1355, 0.622, 2.0 / 3.0, 0.41
CV1, CW2, CW3 = 7.1, 0.3, 2.0
CW1 = CB1 / KAPPA ** 2 + (1.0 + CB2) / SIGMA
CV2, CV3 = 0.7, 0.9
SA_CONSTS = np.array([CB1, CW1, CW2, CW3, CV1, CV2, CV3, KAPPA])


class Solver3D:
    def __init__(self, mesh, bcs, nu, alpha_u=0.9, second_order=True, U0=(0.0, 0.0, 0.0), turbulence=None,
                 nut_inf_ratio=3.0, alpha_turb=0.9, limiter="auto", limiter_eps=0.1, limiter_cells="nonhex",
                 rtol_scalar=0.05, rtol_p=0.01, amg_rebuild=50, first_order_start=0):
        """first_order_start: iterations of first-order upwind convection at a cold start
        before the second-order scheme is switched on (coarse airfoil meshes diverge from a
        uniform start otherwise; set_initial() skips it for warm starts)"""
        self.m, self.nu = mesh, nu
        self.simplec = True
        # skewness-correction sweeps of the p' gradient in the velocity update (0: as 2D)
        self.gpc_skew_iters = 0
        self.bounded = True
        self.d_interp = "linear"        # Rhie-Chow face coefficient: "linear" (as 2D) or "cv"
        self.p_scale = True
        self.first_order_start = first_order_start if second_order else 0
        m = mesh
        n = m.ncells
        self.au = alpha_u
        self.second_order = second_order
        self.limiter = ("venkat" if turbulence else None) if limiter == "auto" else limiter
        self.limiter_eps = limiter_eps
        if limiter_cells == "nonhex":
            nfaces = np.bincount(m.own, minlength=n) + np.bincount(m.nei, minlength=n) + \
                np.bincount(m.bcell, minlength=n)
            self.limiter_mask = nfaces != 6
        else:
            self.limiter_mask = None
        self.rtol_scalar, self.rtol_p, self.amg_rebuild = rtol_scalar, rtol_p, amg_rebuild
        self.adj = N.Adjacency(m)
        self.pat = N.CsrPattern(n, m.own, m.nei)
        self._amg, self._amg_age, self._amg_it0 = None, 0, None
        self._p_lsq = None
        self.amg_rebuild_max = amg_rebuild
        self.converged = self.diverged = False
        self.stats = {"sweeps": 0, "p_iters": 0, "amg_setups": 0, "t_mom": 0.0, "t_p": 0.0, "t_turb": 0.0,
                      "t_grad": 0.0}

        nb = len(m.bcell)
        self.b_fixedU = np.zeros(nb, bool)
        self.b_outlet = np.zeros(nb, bool)
        self.b_wall = np.zeros(nb, bool)
        self.b_inlet = np.zeros(nb, bool)
        self.b_sym = np.zeros(nb, bool)
        self.pb_out = np.zeros(nb)
        self.Ub = np.zeros((nb, 3))
        for name, bc in bcs.items():
            idx = m.patch_faces.get(name)
            if idx is None:
                raise ValueError(f"no patch {name!r} in the mesh (patches: {m.patch_names})")
            t = bc["type"]
            if t == "symmetry":
                self.b_fixedU[idx] = True
                self.b_sym[idx] = True
            elif t in ("wall", "inlet"):
                self.b_fixedU[idx] = True
                (self.b_wall if t == "wall" else self.b_inlet)[idx] = True
                U = bc.get("U", (0.0, 0.0, 0.0))
                self.Ub[idx] = U(m.bxf[idx]) if callable(U) else np.asarray(U, float)
            elif t == "outlet":
                self.b_outlet[idx] = True
                self.pb_out[idx] = bc.get("p", 0.0)
            elif t == "farfield":
                Uinf = np.asarray(bc["U"], float)
                inflow = m.bSf[idx] @ Uinf < 0
                ii, oo = idx[inflow], idx[~inflow]
                self.b_fixedU[ii] = True
                self.b_inlet[ii] = True
                self.Ub[ii] = Uinf
                self.b_outlet[oo] = True
                self.pb_out[oo] = bc.get("p", 0.0)
            else:
                raise ValueError(t)
        missing = set(m.patch_names) - set(bcs)
        if missing:
            raise ValueError(f"no boundary condition for patches {missing}")
        self.has_outlet = self.b_outlet.any()

        self.U = np.tile(np.asarray(U0, float), (n, 1))
        self.p = np.zeros(n)
        self.F = np.einsum("ij,ij->i", m.interp(self.U), m.Sf)
        self.Fb = np.einsum("ij,ij->i", np.where(self.b_fixedU[:, None], self.Ub, self.U[m.bcell]), m.bSf)
        self.Ubar_S_old = self.F.copy()
        self.Ubar_Sb_old = self.Fb.copy()
        self.gradp = np.zeros((n, 3))
        self.gU = np.zeros((3, n, 3))
        self.history = []
        self.turb = None
        if turbulence == "sa":
            if not self.b_wall.any():
                raise ValueError("the SA model needs a wall patch")
            self.turb = SA3D(self, nut_inf_ratio, alpha_turb)
        elif turbulence not in (None, "laminar"):
            raise ValueError(turbulence)

    # ------------------------------------------------------------------
    def _boundary_values(self):
        m = self.m
        c = m.bcell
        Uext = self.U[c] + np.einsum("kij,ij->ik", self.gU[:, c, :], m.db_t)
        if self.b_sym.any():
            sy = self.b_sym
            nrm = m.bSf[sy] / np.linalg.norm(m.bSf[sy], axis=1)[:, None]
            Uc = self.U[c[sy]]
            self.Ub[sy] = Uc - np.einsum("ij,ij->i", Uc, nrm)[:, None] * nrm
        Ub = np.where(self.b_fixedU[:, None], self.Ub, Uext)
        pb = self.p[c] + np.einsum("ij,ij->i", self.gradp[c], m.db)
        pb = np.where(self.b_outlet, self.pb_out, pb)
        return Ub, pb

    def _grads(self, Ub):
        m = self.m
        return np.stack([m.grad(self.U[:, k], Ub[:, k]) for k in range(3)])

    def iterate(self):
        m, n = self.m, self.m.ncells
        own, nei, bc = m.own, m.nei, m.bcell
        t0 = time.time()
        second = self.second_order and len(self.history) >= self.first_order_start
        Ub, pb = self._boundary_values()
        gU = self._grads(Ub)
        self.gU = gU
        if self._p_lsq is None:                    # once per solver (the mesh and the outlets are fixed)
            self._p_lsq = m.lagged_grad_unstable(~self.b_outlet)
            self.stats["p_lsq_cells"] = int(self._p_lsq.sum())
        glag = self.gradp
        if self._p_lsq.any():                      # cells where the lagged correction would amplify itself
            glag = glag.copy()
            glag[self._p_lsq] = m.grad(self.p, pb)[self._p_lsq]
        self.gradp = m.grad_gg(self.p, pb, grad_skew=glag)
        if second and self.limiter is not None:
            fixed = self.b_fixedU & ~self.b_sym           # walls and inlets (symmetry: local velocity)
            uref = max(np.abs(self.Ub[fixed]).max() if fixed.any() else 0.0, 1e-30)
            eps2 = (self.limiter_eps * uref) ** 2
            gk = np.stack([gU[k] * N.venkat(m, self.adj, self.U[:, k], gU[k], Ub[:, k], eps2,
                                            self.limiter_mask)[:, None] for k in range(3)])
        else:
            gk = gU
        t1 = time.time()
        self.stats["t_grad"] += t1 - t0

        # ---- momentum ------------------------------------------------------
        F, Fb = self.F, self.Fb
        nu_f, nu_b = self.turb.face_viscosity() if self.turb else (np.full(len(own), self.nu),
                                                                   np.full(len(bc), self.nu))
        transpose = self.turb is not None
        diag, a_on, a_no, src = N.mom_assemble(m, self.adj, F, nu_f, gU, gk, second, transpose)
        # symmetry planes carry no shear: no diffusion term there (zero normal flux via Fb = 0)
        Db = np.where(self.b_sym, 0.0, nu_b * m.bcoef)
        diag += np.bincount(bc, np.where(self.b_fixedU, Db, 0.0) + np.where(self.b_outlet, np.maximum(Fb, 0.0), 0.0), n)
        if self.bounded is True or (self.bounded and len(self.history) < self.bounded):
            # bounded convection, div(F U) - U div(F): the continuity error of the
            # current fluxes leaves the diagonal (no effect once mass is conserved).
            # Cells with a net inflow (wall cells of the uniform initial field, thin
            # cells at a stagnation line) otherwise lose diagonal dominance and the
            # momentum predictor returns velocities of 50x the free stream.
            netF = m.sum_to_cells(F, Fb)
            dg = diag - netF
            diag = np.maximum(dg, 1e-3 * diag)
        src -= self.gradp * m.vol[:, None]
        for k in range(3):
            bsrc = np.where(self.b_fixedU & ~self.b_sym, Db * Ub[:, k] - Fb * Ub[:, k] +
                            nu_b * np.einsum("ij,ij->i", gU[k][bc], m.bSf - m.bDelta), 0.0)
            bsrc += np.where(self.b_outlet, -np.maximum(Fb, 0.0) * np.einsum("ij,ij->i", gU[k][bc], m.db_t)
                             - np.minimum(Fb, 0.0) * Ub[:, k], 0.0)
            if transpose:
                bsrc += np.where(self.b_sym, 0.0, nu_b * np.einsum("ij,ij->i", gU[:, bc, k].T, m.bSf))
            src[:, k] += np.bincount(bc, bsrc, n)
        au = np.full(n, self.au)
        diag = diag / au
        src += ((1 - au) * diag)[:, None] * self.U
        A = self.pat.matrix(diag, a_on, a_no)
        scale = max((diag * np.linalg.norm(self.U, axis=1)).sum(), 1e-30)
        R = A @ self.U - src
        resU = np.abs(R).sum(0) / scale
        Ustar = self.U.copy()
        self.stats["sweeps"] += N.sgs(A, Ustar, src, self.rtol_scalar, 100)
        t2 = time.time()
        self.stats["t_mom"] += t2 - t1

        # ---- Rhie-Chow face fluxes ---------------------------------------------
        Dp = m.vol / diag
        if self.d_interp == "cv":
            # face coefficient of a control volume straddling the face: (V_o + V_n) /
            # (a_o + a_n). Next to a much thinner cell the linear weight -> 1 hands the
            # face the thin cell's tiny V/a, the face flux stops feeling the pressure
            # difference and the large cell decouples (wing tip, beyond the cap).
            Df = (m.vol[own] + m.vol[nei]) / (diag[own] + diag[nei])
        else:
            Df = m.interp(Dp)
        gp_f = m.interp(self.gradp)
        Uf = np.stack([m.interp_skew(Ustar[:, k], gU[k]) for k in range(3)], 1)
        UbarS = np.einsum("ij,ij->i", Uf, m.Sf)
        Fstar = UbarS - Df * ((self.p[nei] - self.p[own]) * m.coef - np.einsum("ij,ij->i", gp_f, m.Delta))
        au_f = m.interp(au)
        Fstar += (1 - au_f) * (self.F - self.Ubar_S_old)
        self.Ubar_S_old = UbarS
        Fbstar = np.einsum("ij,ij->i", Ub, m.bSf)
        if self.simplec:
            nbsum = np.bincount(own, -a_on, n) + np.bincount(nei, -a_no, n)
            Dc = m.vol / np.maximum(diag - nbsum, (1.0 - au) * diag)
        else:
            Dc = Dp
        if self.d_interp == "cv":
            ac = m.vol / Dc
            Dcf = (m.vol[own] + m.vol[nei]) / (ac[own] + ac[nei])
        else:
            Dcf = m.interp(Dc)
        cb = Dc[bc] * m.bcoef
        if self.has_outlet:
            o = self.b_outlet
            Ustar_b = Ustar[bc[o]] + np.einsum("kij,ij->ik", gU[:, bc[o], :], m.db_t[o])
            UbSb = np.einsum("ij,ij->i", Ustar_b, m.bSf[o])
            Fbstar[o] = UbSb - Dp[bc[o]] * ((self.pb_out[o] - self.p[bc[o]]) * m.bcoef[o] -
                                            np.einsum("ij,ij->i", self.gradp[bc[o]], m.bDelta[o]))
            Fbstar[o] += (1 - au[bc[o]]) * (self.Fb[o] - self.Ubar_Sb_old[o])
            self.Ubar_Sb_old[o] = UbSb

        # ---- pressure correction -------------------------------------------------
        cf = Dcf * m.coef
        pdiag = np.bincount(own, cf, n) + np.bincount(nei, cf, n) + np.bincount(bc, np.where(self.b_outlet, cb, 0.0), n)
        imbalance = m.sum_to_cells(Fstar, Fbstar)
        rhs = -imbalance
        if not self.has_outlet:                    # closed box: pin p' in cell 0
            pdiag = pdiag.copy()
            pdiag[0] += 1e3 * pdiag[0]
            rhs[0] = 0.0
        # symmetric diagonal scaling: the CG stopping test then weighs every cell's mass
        # balance alike (tiny cells - trailing-edge fans, the wing tip - otherwise keep
        # large local residuals under a small global norm)
        dh = 1.0 / np.sqrt(pdiag) if self.p_scale else np.ones(n)
        Ap = self.pat.matrix(pdiag * dh * dh, -cf * dh[own] * dh[nei], -cf * dh[nei] * dh[own])
        rhs = rhs * dh
        if not (np.isfinite(Ap.data).all() and np.isfinite(rhs).all()):
            # the solution blew up: report it (solve() stops on a non-finite residual)
            # instead of failing inside the multigrid setup
            self.diverged = True
            res = {"cont": np.nan, "Ux": resU[0], "Uy": resU[1], "Uz": resU[2]}
            self.history.append(res)
            return res
        # the multigrid hierarchy is reused while it still works: rebuilt when CG needs
        # clearly more iterations than right after the last setup (the matrix changes
        # slowly), at the latest every amg_rebuild_max iterations
        if self._amg is None or self._amg_age >= self.amg_rebuild_max:
            self._new_amg(Ap)
        self._amg_age += 1
        pc, it = self._amg.solve(Ap, rhs, rtol=self.rtol_p, maxit=200)
        if self._amg_it0 is None and it > 0:
            self._amg_it0 = it
        if it < 0 or (self._amg_it0 is not None and it > max(2 * self._amg_it0, self._amg_it0 + 4)):
            self._new_amg(Ap)
            if it < 0:
                pc, it = self._amg.solve(Ap, rhs, rtol=self.rtol_p, maxit=400)
        self.stats["p_iters"] += abs(it)
        pc = pc * dh
        self.F = Fstar - cf * (pc[nei] - pc[own])
        Fb_new = Fbstar.copy()
        o = self.b_outlet
        Fb_new[o] = Fbstar[o] - cb[o] * (0.0 - pc[bc[o]])
        self.Fb = Fb_new
        pcb = np.where(self.b_outlet, 0.0, pc[bc])
        gpc = m.grad_gg(pc, pcb)
        for _ in range(self.gpc_skew_iters):        # skewness correction of the p' gradient
            gpc = m.grad_gg(pc, pcb, grad_skew=gpc)
        self.U = Ustar - Dc[:, None] * gpc
        self.p = self.p + (1.0 if self.simplec else 1.0 - self.au) * pc
        if not self.has_outlet:
            self.p -= self.p[0]
        t3 = time.time()
        self.stats["t_p"] += t3 - t2
        Fref = np.abs(self.Fb).sum() if np.abs(self.Fb).sum() > 0 else np.abs(self.F).sum() / np.sqrt(n)
        res = {"cont": np.abs(imbalance).sum() / max(Fref, 1e-30), "Ux": resU[0], "Uy": resU[1], "Uz": resU[2]}
        if self.turb is not None:
            self.gU = self._grads(self._boundary_values()[0])
            res["nuT"] = self.turb.solve()
            self.stats["t_turb"] += time.time() - t3
        self.history.append(res)
        return res

    def solve(self, tol=1e-6, max_iter=5000, verbose=True, every=100, callback=None):
        t0 = time.time()
        self.converged = self.diverged = False
        r = {}
        for it in range(1, max_iter + 1):
            r = self.iterate()
            if callback is not None and callback(it, r) is True:
                self.converged = True
                if verbose:
                    print(f"  converged (callback) in {it} iterations ({time.time() - t0:.1f}s)", flush=True)
                return it
            if verbose and (it % every == 0 or it == 1):
                print(f"  it {it:5d}  " + "  ".join(f"{k} {v:.2e}" for k, v in r.items()) +
                      f"  ({time.time() - t0:.1f}s)", flush=True)
            if not np.isfinite(max(r.values())):
                self.diverged = True
                if verbose:
                    print(f"  DIVERGED at iteration {it}", flush=True)
                return it
            if max(r.values()) < tol:
                self.converged = True
                if verbose:
                    print(f"  converged in {it} iterations ({time.time() - t0:.1f}s)", flush=True)
                return it
        if verbose:
            print(f"  NOT converged after {max_iter} iterations", flush=True)
        return max_iter

    def _new_amg(self, Ap):
        self._amg = N.AMG(Ap)
        self._amg_age, self._amg_it0 = 0, None
        self.stats["amg_setups"] += 1

    # ------------------------------------------------------------------
    def set_initial(self, U, p=None, nuTilde=None):
        m = self.m
        self.U = np.array(U, float)
        if p is not None:
            self.p = np.array(p, float)
        self.F = np.einsum("ij,ij->i", m.interp(self.U), m.Sf)
        self.Fb = np.einsum("ij,ij->i", np.where(self.b_fixedU[:, None], self.Ub, self.U[m.bcell]), m.bSf)
        self.Ubar_S_old = self.F.copy()
        self.Ubar_Sb_old = self.Fb.copy()
        if nuTilde is not None and self.turb is not None:
            self.turb.nuT = np.maximum(np.array(nuTilde, float), 0.0)
        self.first_order_start = 0                     # warm start: second order from the start

    def wall_values(self, patch):
        m = self.m
        idx = m.patch_faces[patch]
        c = m.bcell[idx]
        pb = self.p[c] + np.einsum("ij,ij->i", self.gradp[c], m.db[idx])
        S = m.bSf[idx]
        Smag = np.linalg.norm(S, axis=1)
        nrm = S / Smag[:, None]
        dn = np.abs(np.einsum("ij,ij->i", m.db[idx], nrm))
        du = self.U[c] - self.Ub[idx]
        du_t = du - np.einsum("ij,ij->i", du, nrm)[:, None] * nrm
        tau = self.nu * du_t / dn[:, None]
        return idx, pb, tau, S, Smag

    def force(self, patch):
        """(pressure force, viscous force) of the fluid on the patch (3-vectors)"""
        _, pb, tau, S, Smag = self.wall_values(patch)
        return (pb[:, None] * S).sum(0), (tau * Smag[:, None]).sum(0)

    def moment(self, patch, ref=(0.25, 0.0, 0.0)):
        idx, pb, tau, S, Smag = self.wall_values(patch)
        Fv = pb[:, None] * S + tau * Smag[:, None]
        r = self.m.bxf[idx] - np.asarray(ref, float)
        return np.cross(r, Fv).sum(0)


# ----------------------------------------------------------------------
def point_triangle_distance(P, A, B, C):
    """distances from points P (m,3) to triangles (A,B,C) (m,3) each (vectorised)"""
    AB, AC = B - A, C - A
    nrm = np.cross(AB, AC)
    nn = np.einsum("ij,ij->i", nrm, nrm)
    AP = P - A
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.einsum("ij,ij->i", AP, nrm) / nn
    Q = P - t[:, None] * nrm                      # projection on the plane
    # barycentric of Q
    v0, v1, v2 = AB, AC, Q - A
    d00 = np.einsum("ij,ij->i", v0, v0); d01 = np.einsum("ij,ij->i", v0, v1); d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0); d21 = np.einsum("ij,ij->i", v2, v1)
    den = d00 * d11 - d01 * d01
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
    inside = (v >= 0) & (w >= 0) & (v + w <= 1) & (nn > 0)
    dplane = np.sqrt(np.einsum("ij,ij->i", P - Q, P - Q))

    def seg(X, Y):
        XY = Y - X
        L2 = np.maximum(np.einsum("ij,ij->i", XY, XY), 1e-300)
        s = np.clip(np.einsum("ij,ij->i", P - X, XY) / L2, 0, 1)
        D = P - (X + s[:, None] * XY)
        return np.sqrt(np.einsum("ij,ij->i", D, D))
    dedge = np.minimum(np.minimum(seg(A, B), seg(B, C)), seg(C, A))
    return np.where(inside, dplane, dedge)


def wall_distance(m, wall_faces, k=8, chunk=200000):
    """distance from every cell centre to the nearest wall face (exact point-polygon
    distance over the k nearest face centroids)"""
    from scipy.spatial import cKDTree
    F = m.bfaces[wall_faces]
    P = m.points
    k = min(k, len(wall_faces))
    tree = cKDTree(m.bxf[wall_faces])
    out = np.empty(m.ncells)
    for s in range(0, m.ncells, chunk):
        X = m.xc[s:s + chunk]
        _, idx = tree.query(X, k=k)
        idx = idx.reshape(len(X), k)
        best = np.full(len(X), np.inf)
        for j in range(k):
            f = F[idx[:, j]]
            n_ = (f >= 0).sum(1)
            for t in range(1, f.shape[1] - 1):           # fan triangles (0, t, t+1)
                ok = t + 1 < n_
                if not ok.any():
                    continue
                d = point_triangle_distance(X, P[f[:, 0]], P[f[:, t]], P[np.where(ok, f[:, t + 1], f[:, t])])
                best = np.minimum(best, np.where(ok, d, np.inf))
        out[s:s + chunk] = best
    return out


class SA3D:
    """Spalart-Allmaras (noft2), identical discretisation to turbulence.SpalartAllmaras"""

    def __init__(self, s, nut_inf_ratio=3.0, alpha=0.9):
        self.s = s
        m = s.m
        self.nu = s.nu
        self.alpha = alpha
        self.nut_inf = nut_inf_ratio * self.nu
        self.d = np.maximum(wall_distance(m, np.where(s.b_wall)[0]), 1e-30)
        self.nuT = np.full(m.ncells, self.nut_inf)
        self.gamma = None

    def fv1(self, nuT):
        chi = nuT / self.nu
        return chi ** 3 / (chi ** 3 + CV1 ** 3)

    def nu_t(self):
        nut = self.nuT * self.fv1(self.nuT)
        return nut if self.gamma is None else nut * self.gamma

    def boundary_nuT(self):
        s, m = self.s, self.s.m
        nb = np.where(s.b_wall, 0.0, self.nuT[m.bcell])
        return np.where(s.b_inlet, self.nut_inf, nb)

    def face_viscosity(self):
        m = self.s.m
        nu_f = self.nu + np.maximum(m.interp(self.nu_t()), 0.0)
        nb = self.boundary_nuT()
        return nu_f, self.nu + nb * self.fv1(nb)

    def solve(self):
        s, m = self.s, self.s.m
        n = m.ncells
        own, nei, bc = m.own, m.nei, m.bcell
        nuT = self.nuT
        nb = self.boundary_nuT()
        g = m.grad(nuT, nb)
        gU = s.gU
        wx = gU[2][:, 1] - gU[1][:, 2]
        wy = gU[0][:, 2] - gU[2][:, 0]
        wz = gU[1][:, 0] - gU[0][:, 1]
        Omega = np.sqrt(wx * wx + wy * wy + wz * wz)
        S0, J = N.sa_source(nuT, Omega, self.d, self.nu, self.gamma, SA_CONSTS)
        Jn = np.maximum(-J, 0.0)
        diag = Jn * m.vol
        src = (S0 + Jn * nuT) * m.vol
        F, Fb = s.F, s.Fb
        nuTf = np.maximum(m.interp(nuT), 0.0)
        kf = (1.0 + CB2) * (self.nu + nuTf)
        k_own = np.maximum(kf - CB2 * (self.nu + nuT[own]), 0.0) / SIGMA * m.coef
        k_nei = np.maximum(kf - CB2 * (self.nu + nuT[nei]), 0.0) / SIGMA * m.coef
        a_on = -k_own - np.maximum(-F, 0.0)
        a_no = -k_nei - np.maximum(F, 0.0)
        diag = diag + np.bincount(own, k_own + np.maximum(F, 0.0), n) + np.bincount(nei, k_nei + np.maximum(-F, 0.0), n)
        gf = m.interp(g)
        nonorth = np.einsum("ij,ij->i", gf, m.Sf - m.Delta)
        src += m.sum_to_cells(kf / SIGMA * nonorth)
        src -= CB2 / SIGMA * (self.nu + nuT) * m.sum_to_cells(nonorth)
        dirich = s.b_wall | s.b_inlet
        Dbk = np.maximum((1.0 + CB2) * (self.nu + nb) - CB2 * (self.nu + nuT[bc]), 0.0) / SIGMA * m.bcoef
        diag += np.bincount(bc, np.where(dirich, Dbk, 0.0) + np.where(~dirich, np.maximum(Fb, 0.0), 0.0), n)
        src += np.bincount(bc, np.where(dirich, Dbk * nb - Fb * nb, 0.0) +
                           np.where(~dirich, -np.minimum(Fb, 0.0) * nb, 0.0), n)
        diag_r = diag / self.alpha
        src = src + (1 - self.alpha) * diag_r * nuT
        A = s.pat.matrix(diag_r, a_on, a_no)
        res = np.abs(A @ nuT - src).sum() / max(np.abs(diag_r * nuT).sum(), 1e-30)
        X = nuT.reshape(-1, 1).copy()
        s.stats["sweeps"] += N.sgs(A, X, src.reshape(-1, 1), s.rtol_scalar, 100)
        self.nuT = np.maximum(X[:, 0], 0.0)
        return res
