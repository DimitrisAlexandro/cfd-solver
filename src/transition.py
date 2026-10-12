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
     2.4 ln(Tu)), or at laminar separation + the bubble transition length
     (Mayle 1991: Re_l = 300 Re_theta,s^0.7; Horton's 4e4 nu/U_e optional)
     if the laminar layer separates first
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


def _separated(y, H_sep):
    """laminar separation, or a march that left the physical range: in the
    steep adverse gradient behind a high-alpha suction peak one RK4 sub-step
    can throw H from ~3.5 to large negative values (NACA 2412, Re 2.2e6,
    alpha 10: H 3.55 -> -143); the march then ran on to the trailing edge
    and reported 'laminar to the trailing edge' -> transition jumped aft and
    the leading-edge region stalled"""
    return (not np.all(np.isfinite(y))) or y[1] >= H_sep or y[1] < 1.02 or y[0] <= 0.0


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
            if _separated(y, H_sep):
                break
        if _separated(y, H_sep):
            sep = i
            break
        th[i], H[i] = y
    ret = Ue * th * re
    return th, H, ret, sep


def march_start(Ue, u_min=0.02):
    """index where the laminar march starts: the last of the leading stations
    with U_e ~ 0. Next to the stagnation point the RANS wall pressure can
    slightly exceed the free-stream total pressure (Cp > 1 by a few 1e-3:
    numerical total-pressure error), U_e = sqrt(1 - 2 p) is then clipped to
    0 for one or two columns; marching from there gave theta from U_e ~ 0
    and an immediate spurious 'laminar separation' (NACA 2412, Re 2.2e6,
    alpha 14: transition estimate 0.2 c instead of 0.01 c -> the leading-edge
    laminar region burst into stall and the solution diverged)."""
    Ue = np.asarray(Ue, float)
    i0 = 0
    while i0 + 2 < len(Ue) and Ue[i0 + 1] < u_min:
        i0 += 1
    return i0


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
    pw = p[0]
    if hasattr(solver, "mach"):          # compressible: isentropic relations
        g, P0 = solver.gamma, solver.P0
        Pa = P0 + p
        M2 = (U**2).sum(2) / (g * solver.R * solver.T[C])
        P0inf = P0 * (1 + 0.5 * (g - 1) * solver.mach**2) ** (g / (g - 1))
        p0def = P0inf - Pa * (1 + 0.5 * (g - 1) * M2) ** (g / (g - 1))   # total-pressure deficit
        ratio = np.maximum((P0 + pw) / P0, 1e-6) ** ((g - 1) / g)
        Ue = np.sqrt(np.maximum(1.0 + 2.0 / ((g - 1) * solver.mach**2) * (1.0 - ratio), 1e-12))
    else:
        p0def = 0.5 - (p + 0.5 * (U**2).sum(2))                   # total-pressure deficit
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
# separation-induced transition (laminar separation bubble)
# ----------------------------------------------------------------------
import os
BUBBLE = os.environ.get("CFD_BUBBLE")          # None: "auto" in the low-Re mode, "horton" above
LE_BUBBLE_X = 0.15         # "auto": separation ahead of this x/c is a leading-edge bubble
BUBBLE_K = 1000.0
# low-Re mode (Re below LOWRE_MAX): bubble length "auto" + trip. "auto": Mayle
# (K = 1000) for mid-chord bubbles; for leading-edge bubbles (separation ahead
# of 0.1 c) not shorter than Horton's - Mayle alone made them too short
# (Re_theta,s ~ 40: Eppler 387, Re 2e5, alpha 8 transition at 0.05 c instead
# of ~0.14 c, Cd +60 %), Horton's length everywhere burst the mid-chord
# bubbles at Re 1e5 (alpha 4-10). Above it the
# validated high-Re behaviour is kept exactly (Horton, no trip): a trip of a
# fixed 100 nu is ~10x the eddy viscosity of a thin leading-edge boundary
# layer at Re 2e6 and stalled the NACA 2412 at alpha 14. Near 5e5 Mayle's
# and Horton's lengths coincide (Re_theta,s ~ 200), so the switch is smooth.
LOWRE_MAX = 5e5
# trip behind a separation-induced transition: nuT >= SEED_CHI nu in the
# separated shear layer over SEED_LENGTH (chords) downstream of x_tr, within
# SEED_HEIGHT of the wall. Without it SA grows the eddy viscosity in a free
# shear layer over ~25 shear-layer thicknesses, and a laminar bubble longer
# than ~0.1 c never reattached (Eppler 387, Re 1e5: stall at alpha 4).
SEED = os.environ.get("CFD_SEED", "on") != "off"
SEED_CHI = float(os.environ.get("CFD_SEED_CHI", 100.0))
SEED_LENGTH = 0.02
SEED_HEIGHT = 0.05
# The trip acts on mid-chord bubbles only (separation aft of LE_BUBBLE_X):
# leading-edge bubbles are thin (Re_theta,s ~ 40) and reattach under SA on
# their own; tripping them thickened the layer (Eppler 387, Re 2e5, alpha 8:
# Cd +60 %; the lower-surface bubble at alpha -2: +20 %). Variants tried and
# not kept: a level 0.7 U_e theta_s and a region of 15-30 x 30-60 theta_s
# (mid-chord bubbles at Re 1e5-2e5 burst, alpha 6 Cd +28 ... +190 %).


def bubble_length(re, ue_sep, theta_sep, model="mayle"):
    """distance (chords) from laminar separation to transition in the
    separated shear layer.
      "mayle":  Re_l = K Re_theta,s^0.7 (Mayle 1991; K = 1000, his long-bubble
                constant, K = 300 for short bubbles; "mayle:K" sets it) -
                scales with the boundary layer at separation. With K = 1000
                and the trip below, the Eppler 387 transition points match
                XFOIL's to ~0.02 c at Re 1e5-2e5 (K 300 put them 0.15 c
                early; K 500-800 without the trip stalled);
      "horton": Re_l = 4e4 (Horton 1968), the constant used before: at
                Re 1e5 it gave 0.3 c of laminar free shear layer, which steady
                RANS cannot reattach (Eppler 387 stalled at alpha 4-8)."""
    ue = max(float(ue_sep), 0.1)
    if model == "horton":
        return 4e4 / (re * ue)
    if str(model).startswith("max"):
        k = model.split(":", 1)[1] if ":" in str(model) else str(BUBBLE_K)
        return max(bubble_length(re, ue, theta_sep, f"mayle:{k}"), 4e4 / (re * ue))
    k = float(model.split(":")[1]) if ":" in str(model) else BUBBLE_K
    th = float(theta_sep)
    re_th = max(re * ue * th, 10.0) if math.isfinite(th) else 100.0
    return k * re_th**0.7 / (re * ue)


# ----------------------------------------------------------------------
class Transition:
    """Free (e^N) or fixed transition for an airfoil solved with SA.

    x_tr: chordwise transition positions {"upper": x, "lower": x}; with
    mode "free" they are only the starting guess and are updated by update().
    """

    def __init__(self, solver, mesh, re, mode="free", ncrit=NCRIT, x_tr=None,
                 max_step=0.1, relax=0.8, tol=0.003, bubble=None):
        self.s, self.re, self.mode, self.ncrit = solver, re, mode, ncrit
        self.lowre = re < LOWRE_MAX
        self.bubble = bubble or BUBBLE or ("auto" if self.lowre else "horton")
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
        # intermittency changes are blended in over ramp_n iterations (set by
        # solve_with_transition; 0 = at once). gamma multiplies the eddy
        # viscosity, so switching a region laminar in one iteration removes
        # nu_t ~ 1e4 nu there at once: a cold start deep in stall (NACA 2412,
        # Re 4.4e6, alpha 18; the turbulent start-up flow still unsettled)
        # took the lower surface laminar to the trailing edge, and the
        # trailing-edge cells under the separated upper flow blew up within
        # 10 iterations. The converged solution is the same.
        self.ramp_n = 0
        self._ramp = None
        m = solver.m
        wall = np.where(solver.b_wall)[0]
        # nearest wall face of every cell, and wall face -> column
        from scipy.spatial import cKDTree
        self._wall_faces = wall
        from turbulence import wall_distance
        _, self._nearest = wall_distance(m, wall, return_index=True)
        self._face_col = cKDTree(self.cols["xw"]).query(m.bxf[wall])[1]
        x_te = float(m.bxf[wall][:, 0].max())
        self._before_te = m.xc[:, 0] < x_te - 0.003 * float(np.ptp(m.bxf[wall][:, 0]))
        if self.auto_start:
            solver.turb.gamma = None               # fully turbulent until the first estimate
        else:
            self.apply()

    def initial_guess(self):
        """Transition points estimated directly from the current (fully
        turbulent) flow's edge velocity; then the laminar regions are applied."""
        bl = extract(self.s, self.cols, self.re)
        est = self.estimate(surfaces(bl, self.cols))
        self.est = est
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
            # at high |alpha| the stagnation point lies on the other side of
            # the leading edge: the stretch stagnation point -> leading edge
            # (x decreasing) is laminar whenever transition is downstream of it
            i_le = int(np.argmin(sd["x"]))
            lam = sd["x"] <= self.x_tr[side]
            if self.x_tr[side] > sd["x"][i_le]:
                lam[:i_le + 1] = True
            # laminar only up to the first point past x_tr (monotone in s)
            stop = np.argmax(~lam) if (~lam).any() else len(lam)
            lam_col[sd["i"][:stop]] = True
        lam_col[surf["i_stag"]] = True
        face_lam = lam_col[self._face_col]
        # the wake and the trailing-edge corner cells stay turbulent (as the
        # wake in XFOIL): with a lower surface laminar to the trailing edge
        # these cells (wall distance ~1e-4 c) were laminar too, and under a
        # separated upper-surface flow they blew up (NACA 2412, Re 4.4e6,
        # cold start at alpha 18-22)
        gamma = np.where(face_lam[self._nearest] & self._before_te, 0.0, 1.0)
        self._set_seed(surf, gamma)
        cur = self.s.turb.gamma
        if self.ramp_n > 0:
            g0 = np.ones_like(gamma) if cur is None else np.asarray(cur, float).copy()
            if np.any(g0 != gamma):
                self._ramp = [g0, gamma, 0]
                self.s.turb.gamma = g0
            else:
                self._ramp = None
                self.s.turb.gamma = gamma
        else:
            self.s.turb.gamma = gamma
        self.surf = surf
        return gamma

    def _set_seed(self, surf, gamma):
        """turbulence trip behind a transition caused by laminar separation"""
        turb = self.s.turb
        if not SEED or self.mode != "free" or not self.lowre:
            turb.seed = None
            return
        est = getattr(self, "est", None) or {}
        cell_col = self._face_col[self._nearest]
        x_col = self.cols["xw"][:, 0]
        d = turb.d
        mask = np.zeros(len(gamma), bool)
        val = np.zeros(len(gamma))
        for side in ("upper", "lower"):
            e = est.get(side) or {}
            if e.get("how") != "laminar separation":
                continue
            xs = e.get("x_sep")
            if xs is not None and xs < LE_BUBBLE_X:
                continue                                    # leading-edge bubble: no trip
            v, L_s, H_s = SEED_CHI * turb.nu, SEED_LENGTH, SEED_HEIGHT
            on_side = np.zeros(len(x_col), bool)
            on_side[surf[side]["i"]] = True
            xc = x_col[cell_col]
            x0 = self.x_tr[side]
            m_side = (on_side[cell_col] & (xc >= x0) & (xc <= x0 + L_s) & (d < H_s)
                      & (gamma > 0.5) & self._before_te)
            mask |= m_side
            val[m_side] = v
        idx = np.where(mask)[0]
        turb.seed = (idx, val[idx]) if len(idx) else None
        self.seed_cells = len(idx)

    def step(self):
        """one iteration of an intermittency blend in progress"""
        if self._ramp is None:
            return
        g0, g1, k = self._ramp
        k += 1
        w = min(k / self.ramp_n, 1.0)
        self.s.turb.gamma = g0 + w * (g1 - g0)
        self._ramp = None if w >= 1.0 else [g0, g1, k]

    def estimate(self, surf):
        """Transition estimate per surface from the laminar integral boundary
        layer on the RANS edge velocity (marched over the whole surface, so
        the N_crit point is found directly). Laminar separation before N_crit
        -> transition at separation + Horton's bubble transition length
        (4e4 nu / U_e)."""
        out = {}
        for side in ("upper", "lower"):
            sd = surf[side]
            i0 = march_start(sd["Ue"])
            s, Ue, x = sd["s"][i0:] - sd["s"][i0], sd["Ue"][i0:], sd["x"][i0:]
            i_le = int(np.argmin(x))

            def x_at(sv):
                """chordwise position of the arc length sv; on the stretch
                ahead of the leading edge -> the leading edge"""
                if sv <= s[i_le]:
                    return float(x[i_le])
                return float(min(np.interp(sv, s, x), 1.0)) if sv <= s[-1] else 1.0

            th, H, ret, sep = laminar_bl(s, Ue, self.re)
            end = sep if sep is not None else len(th)
            end = max(end, 2)
            N, rate = integrate_N(s[:end], H[:end], ret[:end], th[:end])
            hit = np.where(N >= self.ncrit)[0]
            if len(hit):
                j = hit[0]
                f = (self.ncrit - N[j - 1]) / max(N[j] - N[j - 1], 1e-12)
                x_est, how = x_at(s[j - 1] + f * (s[j] - s[j - 1])), "N_crit"
            elif sep is not None:
                # separation-induced transition: bubble transition length from
                # separation, measured along the surface
                # (theta at the last attached station: the march stops at separation)
                model = self.bubble
                if model == "auto":
                    # leading-edge bubbles: Mayle's length, but not shorter than Horton's;
                    # mid-chord bubbles: Mayle's (Horton's longer length burst them at Re 1e5)
                    model = "max" if x[sep] < LE_BUBBLE_X else "mayle"
                x_est = x_at(s[sep] + bubble_length(self.re, Ue[sep], th[max(sep - 1, 1)], model))
                how = "laminar separation"
            else:
                x_est, how = 1.0, "laminar to the trailing edge"
            ts = float(th[max(sep - 1, 1)]) if sep is not None else float("nan")
            out[side] = {"x_est": float(x_est), "how": how, "N_end": float(N[end - 1]),
                         "theta_sep": ts, "ue_sep": float(Ue[sep]) if sep is not None else float("nan"),
                         "x_sep": None if sep is None else float(x[sep]),
                         "N": N, "x": x[:end], "H": H[:end], "theta": th[:end]}
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
    tr.ramp_n = settle
    user_cb = callback

    def callback(it, r):
        tr.step()
        return user_cb(it, r) if user_cb is not None else None

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
