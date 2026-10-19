"""
Local amplification-factor transport (AFT-type) transition model: the e^N
envelope method as field equations solved on the RANS mesh, in the spirit of
Coder & Maughmer's amplification factor transport (AIAA J. 52, 2014) but with
its own closure (below).

    D n / Dt = |u| r(H, Re_theta, theta)                             (1)
    D I / Dt = |u| U_e^5                                             (2)

r is the envelope growth rate per unit length of Drela & Giles (XFOIL's
DAMPL, transition.amplification_rate):
    r = dN/dRe_theta(H) * AF(H) / theta   where Re_theta > Re_theta,crit(H)
Because the production in (1) is |u| times a rate per unit LENGTH, every
fluid layer of the boundary layer integrates dn/ds = r along its own path:
the field n equals the envelope amplification factor N(s) at every height,
given the boundary-layer state at that point. Equation (2) carries Thwaites'
integral I = int U_e^5 ds the same way, which gives that state with its
upstream history, from local data only:
    U_e      = sqrt(1 - 2p)                 (Bernoulli; p is constant across
                                             the layer)
    theta^2  = 0.45 nu I / U_e^6            (Thwaites 1949)
    lambda   = theta^2 (dU_e/ds) / nu,      dU_e/ds = -(grad p . t) / U_e
    H        = H(lambda): Falkner-Skan for lambda >= 0, Cebeci & Bradshaw's
               fit of Thwaites' data shifted to Blasius' 2.591 at 0 (thwaites_H)
    Re_theta = U_e theta / nu
No surface marching, no boundary-layer integrals, no search for the edge:
both equations are local transport equations, purely advective (n and I are
path integrals; molecular diffusion mixed the inner layer with entrained
outer fluid: on the flat plate theta -5 % and n -11 %), linear (one sparse
factorisation, two solves per update). Production only inside the layer: |u|/U_e < 0.99, rotational
(d |Omega| / |u| > 0.2), wall distance below 5 % of the wall length, ahead
of the trailing edge.

Two closures were compared (closure=...):
  "thwaites" (default): above.
  "similarity": the state from the cell's own data alone - the vorticity
     Reynolds number Re_V = d^2 Omega / nu, u/U_e and the local pressure
     gradient - inverted through the Falkner-Skan family (fsc.py):
         phi = f'(eta),  Re_V / Re_theta = eta^2 f''/theta_eta,
         lambda_theta = beta theta_eta^2 = G Re_V^2 / q^2
     It recovers the Blasius layer from single cells (Re_theta within 1 %,
     tests/test_aft.py) but has no history: behind a suction peak H came out
     ~0.09 too high (NACA 0012, Re 3e6, x = 0.3: 2.82 vs 2.73 of the integral
     boundary layer), transition 0.05-0.07 c early.

Coupling with SA: a wall face becomes turbulent when the largest n among the
cells whose nearest wall face it is reaches N_crit; the intermittency of the
cells then follows the faces as in transition.Transition (gamma = 0 over
laminar faces, 1 elsewhere, wake turbulent). Laminar separation is not
modelled (Thwaites' lambda is clipped at -0.09): separation-induced
transition and bubbles stay with transition.py.

Validation: tests/test_aft.py, validation/aft_report.md (flat plate vs the
envelope e^9 result; NACA 0012 vs the marching e^N method and XFOIL).
"""

import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from aerolab.physics import fsc
from aerolab.physics import transition as T

PHI_EDGE = 0.99


class _Tables:
    """per-member tables of the Falkner-Skan family"""

    def __init__(self):
        t = fsc.table()
        self.beta = t["beta"]
        self.H = t["H"]
        self.theta = t["theta"]
        self.lam = self.beta * self.theta**2
        self.prof = []
        for b in self.beta:
            eta, fp, fpp, _ = fsc.profile(round(float(b), 6))
            # monotone part of f' (up to the edge)
            k = int(np.argmax(fp >= 0.9995)) or len(fp) - 1
            self.prof.append((eta[:k + 1], fp[:k + 1], fpp[:k + 1]))

    def q(self, j, phi):
        """Re_V / Re_theta at u/U_e = phi on member j"""
        eta, fp, fpp = self.prof[j]
        e = np.interp(phi, fp, eta)
        return e**2 * np.interp(e, eta, fpp) / self.theta[j]


_TAB = None


def tables():
    global _TAB
    if _TAB is None:
        _TAB = _Tables()
    return _TAB


def local_state(phi, re_v, G):
    """vectorised solution of (2): beta, H, Re_theta for every cell (arrays)"""
    tb = tables()
    nb = len(tb.beta)
    phi = np.clip(phi, 1e-4, PHI_EDGE)
    Q = np.empty((nb, len(phi)))
    for j in range(nb):
        Q[j] = tb.q(j, phi)
    Q = np.maximum(Q, 1e-12)
    F = tb.lam[:, None] - G[None, :] * re_v[None, :] ** 2 / Q**2
    # F(beta) increases with beta for the attached family: first sign change
    s = F > 0
    idx = np.argmax(s, axis=0)                      # first beta with F > 0
    allpos, allneg = s.all(axis=0), ~s.any(axis=0)
    j1 = np.clip(idx, 1, nb - 1)
    j0 = j1 - 1
    cols = np.arange(len(phi))
    f0, f1 = F[j0, cols], F[j1, cols]
    w = np.clip(f0 / np.where(f0 != f1, f0 - f1, 1.0), 0.0, 1.0)
    beta = tb.beta[j0] + w * (tb.beta[j1] - tb.beta[j0])
    H = tb.H[j0] + w * (tb.H[j1] - tb.H[j0])
    q = Q[j0, cols] + w * (Q[j1, cols] - Q[j0, cols])
    beta = np.where(allpos, tb.beta[0], np.where(allneg, tb.beta[-1], beta))
    H = np.where(allpos, tb.H[0], np.where(allneg, tb.H[-1], H))
    q = np.where(allpos, Q[0], np.where(allneg, Q[-1], q))
    return beta, H, re_v / q


def thwaites_H(lam):
    """shape factor from Thwaites' lambda: the Falkner-Skan family for
    lambda >= 0 (exact Blasius 2.591 at 0) and Cebeci & Bradshaw's fit to
    Thwaites' adverse-gradient data, shifted by -0.019 to be continuous
    there. The e^N onset is so sensitive to H that the unshifted fits'
    H(0) = 2.61 moved the e^9 point on a flat plate from x = 0.61 to 0.50."""
    lam = np.asarray(lam, float)
    tb = tables()
    o = np.argsort(tb.lam)
    fav = np.interp(np.maximum(lam, 0.0), tb.lam[o], tb.H[o])
    adv = 2.088 + 0.0731 / (np.minimum(lam, 0.0) + 0.14) - 0.0191
    return np.where(lam >= 0, fav, adv)


class AFT:
    """amplification-factor transport on a SimpleSolver with SA"""

    def __init__(self, solver, re, ncrit=T.NCRIT, walls=("wall",), ramp_n=40,
                 closure="thwaites"):
        s = self.s = solver
        self.closure = closure
        m = s.m
        self.re, self.ncrit = re, ncrit
        self.n = np.zeros(m.ncells)
        wall = np.where(s.b_wall)[0]
        self._wall_faces = wall
        from aerolab.core.turbulence import wall_distance
        self.d, self._nearest = wall_distance(m, wall, return_index=True)
        x_te = float(m.bxf[wall][:, 0].max())
        self._before_te = m.xc[:, 0] < x_te - 0.003 * float(np.ptp(m.bxf[wall][:, 0]))
        self.d_max = 0.05 * float(np.ptp(m.bxf[wall][:, 0]))
        self.ramp_n = ramp_n
        self._ramp = None
        self.history = []
        self.face_turb = np.zeros(len(wall), bool)
        self.converged = False

    # ------------------------------------------------------------------
    def _operator(self, diffusion=True):
        """steady convection (upwind, the solver's face fluxes) + molecular
        diffusion (optional); zero-gradient walls / outlet; inflow brings 0"""
        s, m = self.s, self.s.m
        n = m.ncells
        own, nei, bc = m.own, m.nei, m.bcell
        F, Fb = s.F, s.Fb
        D = s.nu * m.coef * (1.0 if diffusion else 0.0)
        a_on = -D - np.maximum(-F, 0.0)
        a_no = -D - np.maximum(F, 0.0)
        diag = np.bincount(own, D + np.maximum(F, 0.0), n) + np.bincount(nei, D + np.maximum(-F, 0.0), n)
        diag += np.bincount(bc, np.maximum(Fb, 0.0), n)
        A = sp.csr_matrix((np.r_[diag + 1e-14 * m.vol, a_on, a_no],
                           (np.r_[np.arange(n), own, nei], np.r_[np.arange(n), nei, own])), (n, n))
        from scipy.sparse.linalg import splu
        return splu(A.tocsc())

    def _edge(self):
        s = self.s
        um = np.linalg.norm(s.U, axis=1)
        Ue = np.sqrt(np.maximum(1.0 - 2.0 * s.p, 1e-12))
        t = s.U / np.maximum(um, 1e-30)[:, None]
        dUe = -np.einsum('ij,ij->i', s.gradp, t) / Ue          # dU_e/ds along the flow
        # the laminar layer: velocity deficit AND near the wall (a laminar
        # layer is far thinner than 5 % of the wall length; without the bound
        # far-field cells with a small numerical total-pressure loss, 8 chords
        # away, counted as "inside", accumulated n and tripped the leading
        # edge through the nearest-face map)
        # ... AND rotational: sigma = d |Omega| / |u| is ~1 in a boundary layer
        # (exactly 1 in the linear sublayer, ~0.5 at u/U_e = 0.9 for Blasius)
        # and ~0 in irrotational flow. Without it the decelerating stagnation
        # flow ahead of the nose (d ~ 0.03 c, u/U_e 0.98 from a numerical
        # total-pressure loss) looked like a separating layer (lambda -0.09)
        # and produced n ~ 5 that convected along the lower surface (NACA 0012
        # alpha 4: transition 0.48 instead of ~0.87)
        gu, gv = s.gU
        sigma = self.d * np.abs(gv[:, 0] - gu[:, 1]) / np.maximum(um, 1e-30)
        inside = ((um / Ue < PHI_EDGE) & (um > 1e-6) & self._before_te & (self.d < self.d_max)
                  & (sigma > 0.2))
        return um, Ue, dUe, inside

    def production(self):
        """|u| r per cell (per unit time) and the local state.
        closure "thwaites": theta from the transported Thwaites integral
        I = int U_e^5 ds (D I / Dt = |u| U_e^5 in the laminar layer),
            theta^2 = 0.45 nu I / U_e^6,  lambda = theta^2 (dU_e/ds) / nu,
            H = thwaites_H(lambda)
        closure "similarity": local Falkner-Skan inversion (local_state):
        no history - in the adverse gradient behind a suction peak H came out
        ~0.09 too high (NACA 0012, Re 3e6, x = 0.3: 2.82 vs 2.73 of the integral
        boundary layer) and transition ~0.07 c early."""
        s, m = self.s, self.s.m
        nu = s.nu
        um, Ue, dUe, inside = self._edge()
        st = {"phi": um / Ue, "inside": inside, "H": np.full(m.ncells, np.nan),
              "Re_theta": np.full(m.ncells, np.nan), "lambda": np.full(m.ncells, np.nan)}
        if self.closure == "similarity":
            gu, gv = s.gU
            Om = np.abs(gv[:, 0] - gu[:, 1])
            re_v = self.d**2 * Om / nu
            G = nu * dUe / Ue**2
            beta, H, ret = local_state(st["phi"][inside], re_v[inside], G[inside])
            theta = ret * nu / Ue[inside]
        else:
            lu = self._lu
            PI = np.zeros(m.ncells)
            PI[inside] = um[inside] * Ue[inside] ** 5
            self.I = np.maximum(lu.solve(PI * m.vol), 0.0)
            th2 = 0.45 * nu * self.I[inside] / Ue[inside] ** 6
            lam = np.clip(th2 * dUe[inside] / nu, -0.09, 0.25)
            H = thwaites_H(lam)
            theta = np.sqrt(th2)
            ret = Ue[inside] * theta / nu
            st["lambda"][inside] = lam
        r = T.amplification_rate(H, ret, theta)
        P = np.zeros(m.ncells)
        P[inside] = um[inside] * r
        st["H"][inside], st["Re_theta"][inside] = H, ret
        return P, st

    def solve_n(self):
        """steady transport (1), (2): pure advection, one factorisation"""
        self._lu = self._operator(diffusion=False)
        P, st = self.production()
        self.n = np.maximum(self._lu.solve(P * self.s.m.vol), 0.0)
        self.state = st
        return self.n

    # ------------------------------------------------------------------
    def update(self):
        """n from the current flow, then the laminar / turbulent wall faces
        and the intermittency. Returns the number of faces that changed."""
        self.solve_n()
        nf = len(self._wall_faces)
        nmax = np.zeros(nf)
        sel = self.state["inside"]
        np.maximum.at(nmax, self._nearest[sel], self.n[sel])
        turb = nmax >= self.ncrit
        changed = int((turb != self.face_turb).sum())
        self.face_turb = turb
        self.nmax = nmax
        gamma = np.where(~turb[self._nearest] & self._before_te, 0.0, 1.0)
        cur = self.s.turb.gamma
        if cur is None:
            cur = np.ones_like(gamma)
        if self.ramp_n > 0 and np.any(cur != gamma):
            self._ramp = [np.asarray(cur, float).copy(), gamma, 0]
            self.s.turb.gamma = np.asarray(cur, float).copy()
        else:
            self.s.turb.gamma = gamma
            self._ramp = None
        self.history.append({"changed": changed, "turbulent_faces": int(turb.sum())})
        return changed

    def step(self):
        if self._ramp is None:
            return
        g0, g1, k = self._ramp
        k += 1
        w = min(k / self.ramp_n, 1.0)
        self.s.turb.gamma = g0 + w * (g1 - g0)
        self._ramp = None if w >= 1.0 else [g0, g1, k]

    def transition_points(self):
        """x of the first turbulent wall face on each surface (upper: wall
        normal pointing up), measured from the stagnation region"""
        m = self.s.m
        wf = self._wall_faces
        x = m.bxf[wf, 0]
        nup = -m.bSf[wf, 1] > 0                      # outward body normal up
        out = {}
        for side, sel in (("upper", nup), ("lower", ~nup)):
            xs = x[sel & self.face_turb]
            out[side] = float(xs.min()) if len(xs) else 1.0
        return out


def solve_with_aft(solver, aft, tol=1e-7, max_iter=12000, every=100, settle=300, callback=None,
                   max_updates=40):
    """fully turbulent start for `settle` iterations (a laminar start at
    Re 3e6 separated at the trailing edge and diverged), then alternate
    `every` flow iterations and an n / intermittency update (blended in over
    aft.ramp_n iterations) until no more than one wall face changes in 3
    successive updates; then converge the flow. The first estimate, on the
    fuller turbulent profiles, puts transition too far aft; the laminar
    profiles that follow correct it. Returns the number of iterations."""
    user_cb = callback

    def cb(it, r):
        aft.step()
        return user_cb(it, r) if user_cb is not None else None
    solver.turb.gamma = None
    total = solver.solve(tol=0.0, max_iter=min(settle, max_iter), verbose=False, callback=cb)
    quiet = 0
    for k in range(max_updates):
        ch = aft.update()
        quiet = quiet + 1 if ch <= 1 else 0
        if quiet >= 3 or total >= max_iter:
            break
        total += solver.solve(tol=0.0, max_iter=min(every, max_iter - total), verbose=False,
                              callback=cb)
    aft.converged = quiet >= 3
    total += solver.solve(tol=tol, max_iter=max(max_iter - total, 1), verbose=False, callback=cb)
    return total
