"""
Laminar-turbulent transition (Phase 3, step E4): e^N envelope method coupled
to the RANS (Spalart-Allmaras) solution - the transition prediction of XFOIL
and MSES (Drela & Giles 1987), in the way DLR's TAU couples it to RANS:

  1. edge velocity U_e(s) along each surface from the RANS wall pressure
     (Bernoulli: U_e = sqrt(1 - 2 p_wall), U_inf = 1, p_inf = 0), split at the
     stagnation point (sign change of the edge tangential velocity)
  2. laminar integral boundary layer on that U_e(s): momentum and kinetic-
     energy integral equations with XFOIL's laminar closure (HSL, CFL, DIL),
     marched from the stagnation point over the whole surface
     (checked: Blasius, Hiemenz stagnation flow, Howarth separation)
  3. amplification factor N(s): XFOIL's envelope rate (DAMPL)
        dN/ds = AF(H) dN/dRe_theta(H) / theta   where Re_theta > Re_theta,crit(H)
  4. transition where N = N_crit (9 = quiet wind tunnel; Mack: N = -8.43 -
     2.4 ln(Tu)), or at laminar separation + Horton's bubble transition
     length 4e4 nu/U_e if the laminar layer separates first
  5. laminar regions (stagnation point -> transition): eddy viscosity and SA
     production off (intermittency 0), as CFL3D's prescribed laminar regions
  6. the points are updated from the flow (under-relaxed) until they agree
     with the estimate to one mesh column; a cold start first runs fully
     turbulent and takes the estimate from that flow

Why the integral boundary layer and not the RANS velocity profiles: the
envelope criterion is extremely sensitive to the shape factor H (the critical
Re_theta changes steeply around H = 2.6); H integrated from the RANS profiles
varied by +-2 % between meshes (transition +-0.07 c), U_e does not. On XFOIL's
own pressure distribution this module reproduces XFOIL's transition points
to ~0.01 c. extract() still provides the RANS-profile quantities (diagnostics,
flat-plate check).
"""

import math

import numpy as np

NCRIT = 9.0


# ----------------------------------------------------------------------
# envelope amplification rate (XFOIL DAMPL, Drela & Giles 1987)
# ----------------------------------------------------------------------
def log10_retheta_crit(H):
    hmi = 1.0 / (np.asarray(H, float) - 1.0)
    return 2.492 * hmi**0.43 + 0.7 * (np.tanh(14.0 * hmi - 9.24) + 1.0)


def amplification_rate(H, Re_theta, theta):
    """dN/ds (per unit length) of the envelope method."""
    H = np.maximum(np.asarray(H, float), 1.05)
    Re_theta = np.maximum(np.asarray(Re_theta, float), 1e-30)
    hmi = 1.0 / (H - 1.0)
    grcrit = log10_retheta_crit(H)
    gr = np.log10(Re_theta)
    dgr = 0.08
    rn = np.clip((gr - (grcrit - dgr)) / (2 * dgr), 0.0, 1.0)
    rfac = 3 * rn**2 - 2 * rn**3                              # smooth onset
    dadr = 0.028 * (H - 1.0) - 0.0345 * np.exp(-(3.87 * hmi - 2.52) ** 2)
    af = -0.05 + 2.7 * hmi - 5.5 * hmi**2 + 3.0 * hmi**3
    rate = np.where(rfac > 0, af * dadr / np.maximum(theta, 1e-30) * rfac, 0.0)
    # disturbances are never damped in the envelope method (N is monotone);
    # stations without a meaningful laminar profile contribute nothing
    valid = (np.asarray(theta) > 1e-9) & (H > 1.1) & (H < 20.0)
    return np.where(valid, np.maximum(rate, 0.0), 0.0)


def integrate_N(s, H, Re_theta, theta):
    """N(s) along a surface (trapezoid)."""
    a = amplification_rate(H, Re_theta, theta)
    return np.r_[0.0, np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(s))], a


def ncrit_from_turbulence(tu_percent):
    """Mack's correlation (Tu in %)."""
    return -8.43 - 2.4 * math.log(tu_percent / 100.0)


# ----------------------------------------------------------------------
# laminar integral boundary layer (Drela's XFOIL laminar closure: HSL, CFL, DIL)
# ----------------------------------------------------------------------
def _hstar(H):
    """Laminar H* (XFOIL HSL, Falkner-Skan fit)."""
    H = np.asarray(H, float)
    t = H - 4.35
    return np.where(H < 4.35,
                    0.0111 * t**2 / (H + 1) - 0.0278 * t**3 / (H + 1) + 1.528 - 0.0002 * (t * H) ** 2,
                    0.015 * t**2 / H + 1.528)


def _dhstar(H, eps=1e-6):
    return (_hstar(np.asarray(H) + eps) - _hstar(np.asarray(H) - eps)) / (2 * eps)


def _cf_half_ret(H):
    """Re_theta Cf / 2, laminar (XFOIL CFL, Falkner-Skan fit)."""
    H = np.asarray(H, float)
    cf_ret = np.where(H < 5.5, 0.0727 * (5.5 - H) ** 3 / (H + 1) - 0.07,
                      0.015 * (1 - 1 / (H - 4.5)) ** 2 - 0.07)
    return 0.5 * cf_ret


def _cd2_ret_over_hs(H):        # 2 Re_theta CD / H*
    return np.where(H < 4.0, 0.207 + 0.00205 * np.abs(4 - H) ** 5.5,
                    0.207 - 0.0016 * (H - 4) ** 2 / (1 + 0.02 * (H - 4) ** 2))


def laminar_bl(s, Ue, re, H_sep=3.8):
    """Laminar integral boundary layer on a given edge-velocity distribution
    (momentum + kinetic-energy integral equations, Drela's laminar closure):
        d theta/ds  = Cf/2 - (2 + H) theta/Ue dUe/ds
        theta dH*/ds = 2 CD - H* Cf/2 - H* (1 - H) theta/Ue dUe/ds
    marched from the stagnation point (Hiemenz start) with RK4 sub-steps.
    Stops at laminar separation (H >= H_sep: the direct method is singular
    there). Returns theta, H, Re_theta (nan after separation) and the index of
    the separation station (or None)."""
    from scipy.interpolate import PchipInterpolator
    s = np.asarray(s, float)
    Ue = np.maximum(np.asarray(Ue, float), 1e-6)
    ue = PchipInterpolator(s, Ue)
    due = ue.derivative()
    nu = 1.0 / re
    n = len(s)
    th = np.full(n, np.nan)
    H = np.full(n, np.nan)
    # start at station 1 from Thwaites' solution (covers the stagnation point,
    # theta^2 = 0.075 nu / (dUe/ds), and a flat-plate leading edge alike)
    ss = np.linspace(s[0], s[1], 50)
    I = np.trapezoid(ue(ss) ** 5, ss)
    t0 = math.sqrt(max(0.45 * nu * I / max(float(ue(s[1])), 1e-6) ** 6, 1e-30))
    lam = t0**2 / nu * float(due(s[1]))
    h0 = 2.61 - 3.75 * lam + 5.24 * lam**2 if lam >= 0 else 2.088 + 0.0731 / (lam + 0.14)
    y = np.array([t0, min(max(h0, 2.2), 3.0)])
    th[1], H[1] = y

    def rhs(x, y):
        t, h = max(float(y[0]), 1e-12), min(max(float(y[1]), 1.02), 7.0)
        u = max(float(ue(x)), 1e-6)
        du = float(due(x))
        ret = max(u * t / nu, 1e-3)
        cf2 = float(_cf_half_ret(h)) / ret
        hs = float(_hstar(h))
        cd = float(_cd2_ret_over_hs(h)) * hs / (2 * ret)
        dt = cf2 - (2 + h) * t / u * du
        dhs = (2 * cd - hs * cf2 - hs * (1 - h) * t / u * du) / t
        return np.array([dt, dhs / float(_dhstar(h))])

    sep = None
    for i in range(2, n):
        a, b = s[i - 1], s[i]
        # sub-steps: resolve theta-scale changes and, near the stagnation
        # point, the local length scale s itself
        m = max(2, int(math.ceil((b - a) / max(4.0 * float(y[0]), 1e-5))),
                int(math.ceil((b - a) / max(0.05 * (a - s[0]), 1e-9))))
        m = min(m, 2000)
        hh = (b - a) / m
        x = a
        for _ in range(m):
            k1 = rhs(x, y)
            k2 = rhs(x + hh / 2, y + hh / 2 * k1)
            k3 = rhs(x + hh / 2, y + hh / 2 * k2)
            k4 = rhs(x + hh, y + hh * k3)
            y = y + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x += hh
            if y[1] >= H_sep or not np.all(np.isfinite(y)):
                break
        if y[1] >= H_sep or not np.all(np.isfinite(y)):
            sep = i
            break
        th[i], H[i] = y
    ret = Ue * th * re
    return th, H, ret, sep


# ----------------------------------------------------------------------
# boundary-layer columns
# ----------------------------------------------------------------------
def airfoil_columns(mesh):
    """Wall-normal cell columns of the hybrid mesh's boundary layer.
    Cell index = k*M + i (layer k, wall column i). Columns at degenerate
    (fan) wall edges and on the trailing-edge base are skipped."""
    L = mesh["layers"]
    K, M = L.shape[0] - 1, L.shape[1]
    a, b = L[0], np.roll(L[0], -1, axis=0)
    edge = b - a
    le = np.linalg.norm(edge, axis=1)
    ok = le > 1e-12 * max(le.max(), 1e-30)
    t = edge / np.maximum(le, 1e-300)[:, None]
    n = np.c_[t[:, 1], -t[:, 0]]                    # CCW loop: the fluid is on the right
    mid = 0.5 * (a + b)
    ok &= np.abs(t[:, 0]) > 0.2                      # not the (vertical) TE base
    idx = np.where(ok)[0]
    cells = np.arange(K)[:, None] * M + idx[None, :]
    return {"cells": cells, "xw": mid[idx], "t": t[idx], "n": n[idx], "col": idx}


def structured_columns(xw, t, n, cells):
    """Columns for structured meshes (e.g. the flat plate)."""
    return {"cells": np.asarray(cells), "xw": np.asarray(xw), "t": np.asarray(t),
            "n": np.asarray(n), "col": np.arange(len(xw))}


def extract(solver, cols, re):
    """Boundary-layer integrals for every column. Returns dict of arrays."""
    m = solver.m
    C = cols["cells"]                                            # (K, ncol)
    U = solver.U[C]                                               # (K, ncol, 2)
    y = np.einsum('kcj,cj->kc', m.xc[C] - cols["xw"][None], cols["n"])
    ut = np.einsum('kcj,cj->kc', U, cols["t"])                    # signed, along the loop
    p = solver.p[C]
    p0def = 0.5 - (p + 0.5 * (U**2).sum(2))                       # total-pressure deficit
    pw = p[0]
    Ue = np.sqrt(np.maximum(1.0 - 2.0 * pw, 1e-12))
    K, nc = ut.shape
    inside = p0def > 0.005                                        # viscous layer
    # edge: first cell from the wall where the deficit is < 1 % q_inf (after >= 2 cells)
    first_out = np.where(inside[2:].any(0), np.argmax(~inside[2:], axis=0) + 2, K - 1)
    first_out[~inside[2:].any(0)] = 2
    # flow direction from the OUTER flow (the first cell reverses in separated flow)
    u_edge = ut[first_out, np.arange(nc)]
    sign = np.sign(u_edge)
    sign[sign == 0] = 1.0
    u = ut * sign[None, :] / Ue[None, :]                          # u/U_e, positive downstream
    kk = np.arange(K)[:, None]
    w = (kk <= first_out[None, :]).astype(float)
    yy = np.vstack([np.zeros(nc), y])                             # wall point u = 0
    uu = np.vstack([np.zeros(nc), np.clip(u, -1.0, 1.5)])
    ww = np.vstack([np.ones(nc), w])
    dy = np.diff(yy, axis=0)
    f1 = (1 - uu) * ww
    f2 = uu * (1 - uu) * ww
    mw = np.minimum(ww[1:], ww[:-1])
    dstar = (0.5 * (f1[1:] + f1[:-1]) * dy * mw).sum(0)
    theta = (0.5 * (f2[1:] + f2[:-1]) * dy * mw).sum(0)
    theta = np.maximum(theta, 1e-12)
    return {"x": cols["xw"][:, 0], "y": cols["xw"][:, 1], "Ue": Ue, "dstar": dstar,
            "theta": theta, "H": dstar / theta, "Re_theta": Ue * theta * re,
            "u1": u_edge, "cf_sign": np.sign(ut[0]) * sign, "edge_k": first_out,
            "col": cols["col"]}


def surfaces(bl, cols):
    """Split the columns at the stagnation point (sign change of the edge
    tangential velocity near the leading edge) into the upper and lower
    surface, each ordered from the stagnation point downstream with arc
    length s."""
    x = bl["x"]
    u1 = bl["u1"]
    order = np.arange(len(x))                                     # CCW order: upper TE -> LE -> lower TE
    i_le = int(np.argmin(x))
    # stagnation: the sign change of u1 closest to the leading edge
    ch = np.where(np.sign(u1[:-1]) != np.sign(u1[1:]))[0]
    i_st = int(ch[np.argmin(np.abs(ch - i_le))]) if len(ch) else i_le
    out = {}
    for name, sel in (("upper", order[:i_st + 1][::-1]), ("lower", order[i_st + 1:])):
        pts = np.c_[bl["x"][sel], bl["y"][sel]]
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        out[name] = {"i": sel, "s": s, **{k: bl[k][sel] for k in
                                           ("x", "y", "Ue", "dstar", "theta", "H", "Re_theta")}}
    out["i_stag"] = i_st
    return out


# ----------------------------------------------------------------------
# coupling
# ----------------------------------------------------------------------
class Transition:
    """Free (e^N) or fixed transition for an airfoil solved with SA.

    x_tr: chordwise transition positions {"upper": x, "lower": x}; with
    mode "free" they are only the starting guess and are updated by update().
    """

    def __init__(self, solver, mesh, re, mode="free", ncrit=NCRIT, x_tr=None,
                 max_step=0.1, relax=0.8, tol=0.003):
        self.s, self.re, self.mode, self.ncrit = solver, re, mode, ncrit
        self.cols = airfoil_columns(mesh)
        # x_tr None: guess it from a short fully turbulent start (see
        # solve_with_transition); the estimate only needs the edge velocity
        self.auto_start = x_tr is None and mode == "free"
        self.x_tr = dict(x_tr or {"upper": 0.3, "lower": 0.5})
        self.max_step, self.relax, self.tol = max_step, relax, tol
        self.history = []
        self.converged = mode != "free"
        self.converged_by = "estimate" if mode == "free" else "fixed"
        self._relax = {"upper": relax, "lower": relax}
        # smallest laminar-separation point seen so far (+ bubble transition
        # length): the laminar region never extends past it again
        self._sep_cap = {"upper": 1.0, "lower": 1.0}
        self._sep_hist = {"upper": [], "lower": []}      # last 3 settled estimates
        self._last_dir = {"upper": 0, "lower": 0}
        self.last = None
        m = solver.m
        wall = np.where(solver.b_wall)[0]
        # nearest wall face of every cell, and wall face -> column
        from scipy.spatial import cKDTree
        self._wall_faces = wall
        from turbulence import wall_distance
        _, self._nearest = wall_distance(m, wall, return_index=True)
        self._face_col = cKDTree(self.cols["xw"]).query(m.bxf[wall])[1]
        if self.auto_start:
            solver.turb.gamma = None               # fully turbulent until the first estimate
        else:
            self.apply()

    def initial_guess(self):
        """Transition points estimated directly from the current (fully
        turbulent) flow's edge velocity; then the laminar regions are applied."""
        bl = extract(self.s, self.cols, self.re)
        est = self.estimate(surfaces(bl, self.cols))
        for side in ("upper", "lower"):
            self.x_tr[side] = float(est[side]["x_est"])
        self.history.append({"x_tr": dict(self.x_tr), "settled": True, "initial_guess": True,
                             **{sd: {"x_est": est[sd]["x_est"], "how": est[sd]["how"],
                                     "N_end": est[sd]["N_end"], "x_sep": est[sd]["x_sep"]}
                                for sd in ("upper", "lower")}})
        self.auto_start = False
        self.apply()

    # gamma = 0 in the laminar parts, 1 elsewhere
    def apply(self):
        bl = self.last or extract(self.s, self.cols, self.re)
        surf = surfaces(bl, self.cols)
        lam_col = np.zeros(len(self.cols["xw"]), bool)
        for side in ("upper", "lower"):
            sd = surf[side]
            lam = sd["x"] <= self.x_tr[side]
            # laminar only up to the first point past x_tr (monotone in s)
            stop = np.argmax(~lam) if (~lam).any() else len(lam)
            lam_col[sd["i"][:stop]] = True
        lam_col[surf["i_stag"]] = True
        face_lam = lam_col[self._face_col]
        gamma = np.where(face_lam[self._nearest], 0.0, 1.0)
        self.s.turb.gamma = gamma
        self.surf = surf
        return gamma

    def estimate(self, surf):
        """Transition estimate per surface from the laminar integral boundary
        layer on the RANS edge velocity (marched over the whole surface, so
        the N_crit point is found directly). Laminar separation before N_crit
        -> transition at separation + Horton's bubble transition length
        (4e4 nu / U_e)."""
        out = {}
        for side in ("upper", "lower"):
            sd = surf[side]
            th, H, ret, sep = laminar_bl(sd["s"], sd["Ue"], self.re)
            end = sep if sep is not None else len(th)
            end = max(end, 2)
            N, rate = integrate_N(sd["s"][:end], H[:end], ret[:end], th[:end])
            hit = np.where(N >= self.ncrit)[0]
            if len(hit):
                j = hit[0]
                f = (self.ncrit - N[j - 1]) / max(N[j] - N[j - 1], 1e-12)
                x_est, how = sd["x"][j - 1] + f * (sd["x"][j] - sd["x"][j - 1]), "N_crit"
            elif sep is not None:
                x_est = min(sd["x"][sep] + 4e4 / (self.re * max(sd["Ue"][sep], 0.1)), 1.0)
                how = "laminar separation"
            else:
                x_est, how = 1.0, "laminar to the trailing edge"
            out[side] = {"x_est": float(x_est), "how": how, "N_end": float(N[end - 1]),
                         "x_sep": None if sep is None else float(sd["x"][sep]),
                         "N": N, "x": sd["x"][:end], "H": H[:end], "theta": th[:end]}
        return out

    def update(self, settled=True):
        """New transition points from the current flow. Returns the largest
        relative movement (0 when converged)."""
        if self.mode != "free":
            return 0.0
        bl = extract(self.s, self.cols, self.re)
        self.last = bl
        surf = surfaces(bl, self.cols)
        est = self.estimate(surf)
        moves, info = {}, {}
        for side in ("upper", "lower"):
            e = est[side]
            # a laminar separation seen in one of the last 3 settled flows caps
            # the laminar region (a leading-edge bubble otherwise feeds back on
            # the edge velocity and the estimate oscillates between separation
            # and no separation); a one-off spurious separation expires
            if settled:
                self._sep_hist[side].append(e["x_est"] if e["how"] == "laminar separation" else 1.0)
                self._sep_hist[side] = self._sep_hist[side][-3:]
                self._sep_cap[side] = min(self._sep_hist[side])
            x_target = min(e["x_est"], self._sep_cap[side])
            step = x_target - self.x_tr[side]
            d = int(np.sign(step)) if abs(step) > 1e-9 else 0
            if d and self._last_dir[side] and d != self._last_dir[side]:
                self._relax[side] = max(0.5 * self._relax[side], 0.2)     # oscillating: damp
            if d:
                self._last_dir[side] = d
            relax = self._relax[side] if settled else 0.5 * self._relax[side]
            if not settled:
                # a changing flow (usually: laminar separation somewhere) may
                # only move transition upstream - moving it downstream then
                # can grow a leading-edge bubble into stall that never settles
                step = min(step, 0.0)
            x_new = float(np.clip(self.x_tr[side] + relax * np.clip(step, -0.3, 0.3), 0.0, 1.0))
            xs = np.sort(surf[side]["x"])
            k = np.searchsorted(xs, self.x_tr[side])
            h = np.diff(xs)[min(max(k - 1, 0), len(xs) - 2)] if len(xs) > 2 else 0.0
            moves[side] = abs(step) / max(self.tol, h) * self.tol
            info[side] = {k2: v for k2, v in e.items() if k2 not in ("N", "x", "H", "theta")}
            info[side]["x_tr_old"] = self.x_tr[side]
            info[side]["x_est"] = float(x_target)
            self.x_tr[side] = x_new
        self.est = est
        self.history.append({"x_tr": dict(self.x_tr), "settled": settled, **info})
        self.apply()
        move = max(moves.values())
        self.converged = settled and move < self.tol
        return move


def solve_with_transition(solver, tr, tol=1e-7, max_iter=12000, chunk=400, settle=60,
                          callback=None, max_updates=60, verbose=False, scale=1.0):
    """Alternate: let the flow respond (at least `settle` iterations, then until
    the residuals are < 1e-5 or `chunk` iterations), update the transition
    points; when they no longer move, converge the flow to `tol`.
    Returns the total number of iterations."""
    if tr.mode != "free":
        return solver.solve(tol=tol, max_iter=max_iter, verbose=False, callback=callback)
    import os
    verbose = verbose or os.environ.get("CFD_TRANSITION_VERBOSE") == "1"
    # scale: iteration counts relative to SIMPLEC (the coupled solver
    # converges in ~10x fewer outer iterations)
    chunk, settle = max(int(chunk * scale), 5), max(int(settle * scale), 3)
    n_start1, n_start2 = max(int(150 * scale), 10), max(int(300 * scale), 20)
    total = 0
    prev_F = None
    if tr.auto_start:                     # start: fully turbulent first
        # at least 150 iterations even if the residual is already small: a
        # warm-started field (previous angle, rotated) looks converged at once
        # but its suction peak has not yet adapted to the new angle
        total += solver.solve(tol=0.0, max_iter=min(n_start1, max_iter), verbose=False,
                              callback=callback)
        total += solver.solve(tol=1e-4, max_iter=min(n_start2, max_iter - total), verbose=False,
                              callback=callback)
        tr.initial_guess()
        if verbose:
            h = tr.history[-1]
            print(f"  [transition] it {total:5d}: initial guess from the turbulent flow: upper "
                  f"{h['x_tr']['upper']:.4f} ({h['upper']['how']}), lower {h['x_tr']['lower']:.4f} "
                  f"({h['lower']['how']})", flush=True)
    for k in range(max_updates):
        n = min(settle, max_iter - total)
        total += solver.solve(tol=0.0, max_iter=n, verbose=False, callback=callback)
        if total < max_iter:
            total += solver.solve(tol=1e-5, max_iter=min(chunk, max_iter - total),
                                  verbose=False, callback=callback)
        if total >= max_iter:
            break
        Fp, Fv = solver.force("wall")
        F = Fp + Fv
        # flow not settled: N from it would be noise, only the laminar-
        # separation rule is applied (it is what keeps the flow from settling)
        settled = max(solver.history[-1].values()) <= 1e-4
        move = tr.update(settled=settled)
        # engineering convergence: the transition points still hop by a column
        # or two (e.g. near the trailing edge, where a laminar layer is on the
        # verge of separating) but the forces no longer change
        if settled and not tr.converged and prev_F is not None and move < 0.05 and \
                len(tr.history) >= 15:
            dF = np.abs(F - prev_F) / np.maximum(np.abs(F), 1e-3 * np.linalg.norm(F))
            if np.all(dF < 3e-3):
                tr.converged = True
                tr.converged_by = "forces"
        prev_F = F if settled else None
        # limit cycle (typically a leading-edge laminar bubble: separation <->
        # no separation): freeze the transition points at the mean of the
        # last updates and converge the flow with them
        if not tr.converged and len(tr.history) >= 20:
            last = [h["x_tr"] for h in tr.history[-6:]]
            for sd in ("upper", "lower"):
                tr.x_tr[sd] = float(np.mean([h[sd] for h in last]))
            tr.apply()
            tr.converged = True
            tr.converged_by = "averaged (limit cycle)"
        if verbose:
            h = tr.history[-1]
            print(f"  [transition] it {total:5d}: x_tr upper {h['x_tr']['upper']:.4f} "
                  f"(est {h['upper']['x_est']:.4f}, {h['upper']['how']})  lower "
                  f"{h['x_tr']['lower']:.4f} (est {h['lower']['x_est']:.4f}, {h['lower']['how']})"
                  f"  res {max(solver.history[-1].values()):.1e}", flush=True)
        if tr.converged:
            total += solver.solve(tol=tol, max_iter=max_iter - total, verbose=False,
                                  callback=callback)
            break
    return total
