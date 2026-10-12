"""
Wing structure, first pass: wing-box sizing from the flight loads, mass,
stiffness, deflection, and the static and dynamic aeroelastic checks
(divergence, aileron reversal, flutter estimate).

    python wing_structure.py ../cases/aircraft/demo_trainer.json ../cases/loads_x/loads.json --out ../cases/struct

Model
  * Wing box between a front and a rear spar (fractions of the chord), the
    depth from the real section at every station (root -> tip airfoil blend).
    Idealised box: spar caps carry the bending, skins and webs the shear flow
    of torsion, the webs the vertical shear (equal shear flow in both webs).
  * Loads: every case of a loads.py run (limit loads; manoeuvre corners,
    gusts, 1 g), times the factor of safety (1.5) -> ultimate. Torsion is
    moved from the loads' structural axis to the box shear centre (mid-box).
  * Sizing per station, the worst case for each member:
        upper cap  A = max(M+ / F_c, |M-| / F_t) / h_eff   (M+ compresses the upper cap)
        lower cap  A = max(M+ / F_t, |M-| / F_c) / h_eff
        webs       t = (|V| / (h_f + h_r) + |T| / 2A_box) / F_s
        skins      t = |T| / (2 A_box F_s)
    with minimum gauges, h_eff = 0.92 x mean box depth. Allowables are design
    ultimate values with knock-downs (compression: crippling / column; shear:
    post-buckled webs; composites: environment and damage) - editable.
  * Mass: caps, webs, skins, ribs (rib pitch, lightened ribs), times a
    non-optimum factor (joints, fasteners, cut-outs), plus secondary
    structure (leading edge, trailing edge, control surfaces) per square
    metre outside the box. Compared with Raymer's Class II wing mass.
  * Stiffness: EI from caps and skins about the box centroid; GJ by Bredt,
    4 A^2 / (sum ds / (G t)). Deflection and twist of the cantilever (root
    clamped at the symmetry plane) under the limit loads.
  * Divergence and aileron reversal: torsion of the elastic axis (finite
    elements, linear) with strip aerodynamics, lift slope from Helmbold's
    formula for the wing's aspect ratio and sweep, aerodynamic centre at
    the quarter chord, aileron lift and moment from thin-airfoil theory.
    Straight-wing model: sweep-back raises the divergence speed (bending-
    torsion coupling), so the estimate is conservative for swept-back wings.
  * Flutter estimate: first bending and first torsion frequencies of the
    wing (beam finite elements on the sized EI, GJ, mass and inertia), then
    the typical section at 75 % of the semi-span with Theodorsen's unsteady
    aerodynamics (V-g method). CS-23.629: flutter-free up to 1.2 V_D. This
    is a screening estimate - not a substitute for a flutter analysis with
    the real modes, control-surface flutter (mass balance) and a ground
    vibration test.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

import aircraft as A

# design ultimate allowables (Pa), moduli (Pa), density (kg/m^3), minimum gauge (m)
# metals: MMPDS A-basis sheet values; F_c includes no crippling knock-down (see k_c)
MATERIALS = {
    "aluminium 2024-T3": {"E_cap": 72.4e9, "E_skin": 72.4e9, "G": 27.6e9, "rho": 2780.0,
                          "F_t": 441e6, "F_c": 269e6, "F_s": 276e6, "min_gauge": 0.5e-3},
    "aluminium 7075-T6": {"E_cap": 71.0e9, "E_skin": 71.0e9, "G": 26.9e9, "rho": 2810.0,
                          "F_t": 538e6, "F_c": 476e6, "F_s": 324e6, "min_gauge": 0.5e-3},
    "CFRP (UD caps, ±45 skins)": {"E_cap": 120e9, "E_skin": 20e9, "G": 30e9, "rho": 1550.0,
                                  "F_t": 600e6, "F_c": 450e6, "F_s": 120e6, "min_gauge": 0.5e-3},
    "GFRP (UD caps, ±45 skins)": {"E_cap": 38e9, "E_skin": 12e9, "G": 10e9, "rho": 1900.0,
                                  "F_t": 450e6, "F_c": 320e6, "F_s": 70e6, "min_gauge": 0.6e-3},
}


def default_settings():
    return {"material": "aluminium 2024-T3", "front_spar": 0.15, "rear_spar": 0.65,
            "safety_factor": 1.5, "k_c": 0.80, "k_s": 0.80, "h_eff": 0.92,
            "rib_pitch": 0.40, "rib_thickness": 1.0e-3, "rib_solidity": 0.5,
            "non_optimum": 1.25, "secondary_kg_m2": 3.0, "cg_chord": 0.42, "gyration_chord": 0.25,
            "fuel_mass": 0.0, "fuel_span": (0.1, 0.5), "flutter_station": 0.75}


# ----------------------------------------------------------------------
# section geometry
# ----------------------------------------------------------------------
_THICK = {}


def thickness_fn(spec):
    """t(x)/c of a section (upper minus lower surface)"""
    key = str(spec)
    if key not in _THICK:
        import airfoil_geom as ag
        n = ag.sharp_outline(spec)
        i = int(np.argmin(n[:, 0]))
        up, lo = n[:i + 1][::-1], n[i:]
        x = np.linspace(0, 1, 401)
        t = np.interp(x, up[:, 0], up[:, 1]) - np.interp(x, lo[:, 0], lo[:, 1])
        _THICK[key] = (x, np.maximum(t, 0.0))
    return _THICK[key]


def box_section(s, eta, xf, xr):
    """chord, front/rear spar depth, mean box depth and width (m) at eta"""
    c = float(s.chord(eta))
    x0, t0 = thickness_fn(s.airfoil)
    if s.tip_airfoil and s.tip_airfoil != s.airfoil:
        x1, t1 = thickness_fn(s.tip_airfoil)
        t = (1 - eta) * t0 + eta * np.interp(x0, x1, t1)
    else:
        t = t0
    hf, hr = np.interp(xf, x0, t) * c, np.interp(xr, x0, t) * c
    m = (x0 >= xf) & (x0 <= xr)
    hm = float(np.trapezoid(t[m], x0[m]) / max(xr - xf, 1e-9)) * c
    return {"c": c, "h_f": float(hf), "h_r": float(hr), "h_mean": hm, "w": (xr - xf) * c,
            "A_box": hm * (xr - xf) * c}


# ----------------------------------------------------------------------
# sizing
# ----------------------------------------------------------------------
def size(ac, loads, settings=None, surface=0):
    """box members, mass and stiffness along the span from a loads.py result"""
    S = default_settings() | dict(settings or {})
    mat = MATERIALS[S["material"]] if isinstance(S["material"], str) else S["material"]
    s = ac.surfaces[surface]
    xf, xr = S["front_spar"], S["rear_spar"]
    xs = 0.5 * (xf + xr)                                        # shear centre of the box
    x_ax = loads.get("x_axis", 0.35)
    FS = S["safety_factor"]
    Fc, Ft, Fs = S["k_c"] * mat["F_c"], mat["F_t"], S["k_s"] * mat["F_s"]
    tmin = mat["min_gauge"]
    cases = loads["cases"]
    eta = np.array(cases[0]["stations"]["eta"])
    order = np.argsort(eta)
    eta = np.clip(eta[order], 0.0, 1.0)
    n = len(eta)
    geo = [box_section(s, e, xf, xr) for e in eta]
    c = np.array([g["c"] for g in geo])
    hf = np.array([g["h_f"] for g in geo])
    hr = np.array([g["h_r"] for g in geo])
    hm = np.array([g["h_mean"] for g in geo])
    w = np.array([g["w"] for g in geo])
    Abox = np.array([g["A_box"] for g in geo])
    he = S["h_eff"] * hm
    A_up = np.zeros(n); A_lo = np.zeros(n); t_w = np.full(n, tmin); t_s = np.full(n, tmin)
    crit = {"upper cap": [""] * n, "lower cap": [""] * n, "webs": [""] * n, "skins": [""] * n}
    for case in cases:
        st = case["stations"]
        V = FS * np.array(st["shear"])[order]
        M = FS * np.array(st["bending"])[order]
        T = FS * (np.array(st["torsion"])[order] + np.array(st["shear"])[order] * (xs - x_ax) * c)
        a_up = np.where(M > 0, M / Fc, -M / Ft) / he
        a_lo = np.where(M > 0, M / Ft, -M / Fc) / he
        tw = (np.abs(V) / (hf + hr) + np.abs(T) / (2 * Abox)) / Fs
        ts = np.abs(T) / (2 * Abox * Fs)
        for arr, new, key in ((A_up, a_up, "upper cap"), (A_lo, a_lo, "lower cap"), (t_w, tw, "webs"),
                              (t_s, ts, "skins")):
            better = new > arr
            arr[better] = new[better]
            for i in np.where(better)[0]:
                crit[key][i] = case["name"]
    for key, arr in (("webs", t_w), ("skins", t_s)):
        for i in np.where(arr <= tmin + 1e-15)[0]:
            crit[key][i] = "minimum gauge"
    # stiffness (about the box centroid; caps at +-h_eff/2, skins at +-h_mean/2)
    zc = (A_up - A_lo) * he / 2 / np.maximum(A_up + A_lo, 1e-12)            # cap centroid offset
    I_caps = A_up * (he / 2 - zc) ** 2 + A_lo * (he / 2 + zc) ** 2
    I_skin = 2 * w * t_s * (hm / 2) ** 2 * mat["E_skin"] / mat["E_cap"]
    EI = mat["E_cap"] * (I_caps + I_skin)
    GJ = 4 * Abox**2 * mat["G"] / (2 * w / t_s + (hf + hr) / t_w)
    # mass per unit span (one side)
    rho = mat["rho"]
    m_caps = rho * (A_up + A_lo)
    m_webs = rho * (hf + hr) * t_w
    m_skins = rho * 2 * w * t_s
    m_ribs = rho * S["rib_thickness"] * w * hm * S["rib_solidity"] / S["rib_pitch"]
    m_box = S["non_optimum"] * (m_caps + m_webs + m_skins + m_ribs)
    m_sec = S["secondary_kg_m2"] * (c - w)
    semi = s.semi
    y = eta * semi
    side = 2.0 if s.symmetric else 1.0
    I = lambda f: float(np.trapezoid(f, y))
    breakdown = {"spar caps": I(m_caps) * S["non_optimum"], "spar webs": I(m_webs) * S["non_optimum"],
                 "skins (box)": I(m_skins) * S["non_optimum"], "ribs": I(m_ribs) * S["non_optimum"],
                 "secondary (LE, TE, controls)": I(m_sec)}
    breakdown = {k: side * v for k, v in breakdown.items()}
    total = sum(breakdown.values())
    return {"settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in S.items()},
            "material": mat, "surface": s.name, "eta": eta, "y": y, "chord": c, "h_f": hf, "h_r": hr,
            "h_mean": hm, "w": w, "A_box": Abox, "A_upper_cap": A_up, "A_lower_cap": A_lo, "t_web": t_w,
            "t_skin": t_s, "EI": EI, "GJ": GJ, "m_span": m_box + m_sec, "m_box": m_box, "critical": crit,
            "mass_breakdown": breakdown, "mass": total, "x_shear_centre": xs, "x_loads_axis": x_ax}


# ----------------------------------------------------------------------
# deflection under the limit loads
# ----------------------------------------------------------------------
def deflection(st, case):
    """bending deflection and elastic twist of one case (limit loads)"""
    order = np.argsort(case["stations"]["eta"])
    M = np.array(case["stations"]["bending"])[order]
    T = np.array(case["stations"]["torsion"])[order] + \
        np.array(case["stations"]["shear"])[order] * (st["x_shear_centre"] - st["x_loads_axis"]) * st["chord"]
    y = st["y"]
    kap = M / st["EI"]
    slope = np.r_[0.0, np.cumsum(0.5 * (kap[1:] + kap[:-1]) * np.diff(y))]
    defl = np.r_[0.0, np.cumsum(0.5 * (slope[1:] + slope[:-1]) * np.diff(y))]
    tw_rate = T / st["GJ"]
    twist = np.r_[0.0, np.cumsum(0.5 * (tw_rate[1:] + tw_rate[:-1]) * np.diff(y))]
    return {"case": case["name"], "n": case["n"], "deflection": defl, "twist_deg": np.degrees(twist),
            "tip_deflection": float(defl[-1]), "tip_twist_deg": float(np.degrees(twist[-1]))}


# ----------------------------------------------------------------------
# static aeroelasticity: torsion FE + strip aerodynamics
# ----------------------------------------------------------------------
def helmbold(ar, sweep_deg=0.0, mach=0.0, kappa=0.95):
    b2 = 1 - mach**2
    t = math.tan(math.radians(sweep_deg))
    return 2 * math.pi * ar / (2 + math.sqrt(ar**2 * b2 / kappa**2 * (1 + t**2 / b2) + 4))


def flap_coeffs(cf):
    """thin-airfoil lift and quarter-chord moment per radian of a plain flap"""
    th = math.acos(2 * cf - 1)
    return 2 * (math.pi - th + math.sin(th)), -0.5 * math.sin(th) * (1 - math.cos(th))


def _grid(st, n=60):
    yn = np.linspace(0, st["y"][-1], n + 1)
    f = lambda k: np.interp(yn, st["y"], st[k])
    return yn, f("GJ"), f("chord"), f("EI"), f("m_span")


def torsion_matrices(y, GJ, c, e, a):
    """K theta = q Q theta for the elastic-axis twist (root clamped);
    Q: strip aerodynamic moment about the elastic axis per unit q and twist"""
    n = len(y) - 1
    K = np.zeros((n + 1, n + 1))
    for i in range(n):
        L = y[i + 1] - y[i]
        k = 0.5 * (GJ[i] + GJ[i + 1]) / L
        K[i:i + 2, i:i + 2] += k * np.array([[1, -1], [-1, 1]])
    dy = np.r_[0.5 * (y[1] - y[0]), 0.5 * (y[2:] - y[:-2]), 0.5 * (y[-1] - y[-2])]
    Q = np.diag(c**2 * e * a * dy)
    return K[1:, 1:], Q[1:, 1:], dy[1:]


def divergence(y, GJ, c, e, a):
    """divergence dynamic pressure (Pa), inf if the elastic axis is ahead of the a.c."""
    if np.all(e <= 0):
        return math.inf
    K, Q, _ = torsion_matrices(y, GJ, c, e, a)
    from scipy.linalg import eigh
    lam = eigh(Q, K, eigvals_only=True)                         # 1/q
    lam = lam[lam > 1e-30]
    return float(1 / lam.max()) if len(lam) else math.inf


def aileron_reversal(y, GJ, c, e, a, eta0, eta1, cf, q_max):
    """reversal dynamic pressure and the roll-effectiveness curve"""
    K, Q, dy = torsion_matrices(y, GJ, c, e, a)
    yy, cc = y[1:], c[1:]
    ee = e[1:] if np.ndim(e) else np.full(len(yy), e)
    aa = a if np.isscalar(a) else a[1:]
    semi = y[-1]
    on = (yy >= eta0 * semi) & (yy <= eta1 * semi)
    cl_d, cm_d = flap_coeffs(cf)
    cl_d = cl_d * aa / (2 * math.pi)
    b = np.where(on, cc**2 * (ee * cl_d + cm_d) * dy, 0.0)          # moment about the e.a. per unit q, delta
    lift_d = np.where(on, cc * cl_d * dy, 0.0)

    def eff(q):
        th = np.linalg.solve(K - q * Q, q * b)
        roll = (yy * (cc * aa * th * dy + lift_d)).sum()
        return roll / (yy * lift_d).sum()
    qs = np.linspace(0, q_max, 200)[1:]
    ef = np.array([eff(q) for q in qs])
    q_r = math.inf
    k = np.where(np.diff(np.sign(ef)) != 0)[0]
    if len(k):
        from scipy.optimize import brentq
        q_r = brentq(eff, qs[k[0]], qs[k[0] + 1])
    return q_r, qs, ef


# ----------------------------------------------------------------------
# flutter
# ----------------------------------------------------------------------
def beam_modes(y, EI, m, GJ, Ip):
    """first bending and first torsion natural frequencies (rad/s) of a
    cantilever: Hermite beam elements (bending), linear elements (torsion),
    consistent mass"""
    from scipy.linalg import eigh
    n = len(y) - 1
    Kb = np.zeros((2 * n + 2, 2 * n + 2)); Mb = np.zeros_like(Kb)
    Kt = np.zeros((n + 1, n + 1)); Mt = np.zeros_like(Kt)
    for i in range(n):
        L = y[i + 1] - y[i]
        ei = 0.5 * (EI[i] + EI[i + 1]); mi = 0.5 * (m[i] + m[i + 1])
        ke = ei / L**3 * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L**2, -6 * L, 2 * L**2],
                                   [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L**2, -6 * L, 4 * L**2]])
        me = mi * L / 420 * np.array([[156, 22 * L, 54, -13 * L], [22 * L, 4 * L**2, 13 * L, -3 * L**2],
                                      [54, 13 * L, 156, -22 * L], [-13 * L, -3 * L**2, -22 * L, 4 * L**2]])
        d = slice(2 * i, 2 * i + 4)
        Kb[d, d] += ke; Mb[d, d] += me
        gj = 0.5 * (GJ[i] + GJ[i + 1]); ip = 0.5 * (Ip[i] + Ip[i + 1])
        Kt[i:i + 2, i:i + 2] += gj / L * np.array([[1, -1], [-1, 1]])
        Mt[i:i + 2, i:i + 2] += ip * L / 6 * np.array([[2, 1], [1, 2]])
    wb = eigh(Kb[2:, 2:], Mb[2:, 2:], eigvals_only=True)
    wt = eigh(Kt[1:, 1:], Mt[1:, 1:], eigvals_only=True)
    return float(math.sqrt(wb[0])), float(math.sqrt(wt[0]))


def theodorsen(k):
    from scipy.special import hankel2
    if k < 1e-8:
        return 1.0 + 0j
    h1, h0 = hankel2(1, k), hankel2(0, k)
    return h1 / (h1 + 1j * h0)


def flutter_typical_section(mu, a, x_a, r2, sigma, ks=None):
    """V-g method with Theodorsen's aerodynamics. Returns the flutter
    reduced velocity U/(b w_alpha) and frequency ratio w/w_alpha (None if no
    crossing in the range), and the V-g branches. a: elastic axis aft of mid-
    chord (semichords); x_a: c.g. aft of the elastic axis (semichords)."""
    ks = np.geomspace(10.0, 0.02, 500) if ks is None else ks
    rows = []
    for k in ks:
        C = theodorsen(k)
        Lh = 1 - 2j * C / k
        La = 0.5 - 1j * (1 + 2 * C) / k - 2 * C / k**2
        Mh = 0.5
        Ma = 3 / 8 - 1j / k
        ap = 0.5 + a
        # det [[mu (1 - s^2 X) + Lh, mu x_a + La - ap Lh],
        #      [mu x_a + Mh - ap Lh, mu r2 (1 - X) + Ma - ap (La + Mh) + ap^2 Lh]] = 0,  X = (w_a/w)^2 (1 + i g)
        a11, b11 = mu + Lh, -mu * sigma**2
        a12 = mu * x_a + La - ap * Lh
        a21 = mu * x_a + Mh - ap * Lh
        a22, b22 = mu * r2 + Ma - ap * (La + Mh) + ap**2 * Lh, -mu * r2
        # (a11 + b11 X)(a22 + b22 X) - a12 a21 = 0
        roots = np.roots([b11 * b22, a11 * b22 + a22 * b11, a11 * a22 - a12 * a21])
        out = []
        for X in roots:
            if X.real <= 0:
                out.append((math.nan, math.nan, math.nan))
                continue
            wr = 1 / math.sqrt(X.real)                       # w / w_alpha
            out.append((wr / k, X.imag / X.real, wr))         # U/(b w_a), g, w/w_a
        rows.append(sorted(out, key=lambda t: (t[2] if t[2] == t[2] else 9)))
    br = np.array(rows)                                       # (k, branch, [V, g, w])
    V_f, w_f = None, None
    for j in range(br.shape[1]):
        V, g, wr = br[:, j, 0], br[:, j, 1], br[:, j, 2]
        ok = np.isfinite(V) & np.isfinite(g)
        Vv, gg, ww = V[ok], g[ok], wr[ok]
        idx = np.where((gg[:-1] < 0) & (gg[1:] >= 0) & (np.diff(Vv) > 0))[0]
        for i in idx:
            f = -gg[i] / (gg[i + 1] - gg[i])
            v = Vv[i] + f * (Vv[i + 1] - Vv[i])
            if V_f is None or v < V_f:
                V_f, w_f = float(v), float(ww[i] + f * (ww[i + 1] - ww[i]))
    return V_f, w_f, br


# ----------------------------------------------------------------------
def aeroelastic(ac, st, V_D=None, rho=1.225, surface=0):
    S = st["settings"]
    s = ac.surfaces[surface]
    y, GJ, c, EI, m = _grid(st)
    ar = s.span**2 / s.area if s.symmetric else 2 * s.span**2 / s.area
    a = helmbold(ar, s.sweep)
    e = np.full_like(y, st["x_shear_centre"] - 0.25)          # e.a. aft of the a.c., chords
    q_d = divergence(y, GJ, c, e, a)
    ail = next((ct for ct in s.controls if ct.name == "aileron"), None)
    rev = None
    if ail is not None:
        q_max = 0.98 * q_d if math.isfinite(q_d) else 50 * 0.5 * rho * (V_D or 100) ** 2
        q_r, qs, ef = aileron_reversal(y, GJ, c, e, a, ail.eta0, ail.eta1, ail.cf, q_max)
        rev = {"q": q_r, "V_eas": math.sqrt(2 * q_r / 1.225) if math.isfinite(q_r) else math.inf,
               "q_curve": qs.tolist(), "effectiveness": ef.tolist()}
        if V_D:
            qD = 0.5 * 1.225 * V_D**2
            rev["effectiveness_at_VD"] = float(np.interp(qD, qs, ef)) if qD <= qs[-1] else None
    # flutter: modes + typical section
    xcg, kg = S["cg_chord"], S["gyration_chord"]
    xs = st["x_shear_centre"]
    mf = np.zeros_like(y)
    if S.get("fuel_mass", 0) > 0:
        f0, f1 = S["fuel_span"]
        on = (y >= f0 * y[-1]) & (y <= f1 * y[-1])
        if on.any():
            mf[on] = 0.5 * S["fuel_mass"] / max(np.trapezoid(on * 1.0, y), 1e-9) * (2 if not s.symmetric else 1)
    mt = m + mf
    Ip = mt * (kg**2 + (xcg - xs) ** 2) * c**2
    w_h, w_a = beam_modes(y, EI, mt, GJ, Ip)
    eta_r = S["flutter_station"]
    cr = float(np.interp(eta_r * y[-1], y, c))
    mr = float(np.interp(eta_r * y[-1], y, mt))
    b = cr / 2
    mu = mr / (math.pi * rho * b**2)
    a_h = 2 * xs - 1
    x_a = 2 * (xcg - xs)
    r2 = (kg**2 + (xcg - xs) ** 2) * 4
    sigma = w_h / w_a
    Vr, wr, br = flutter_typical_section(mu, a_h, x_a, r2, sigma)
    V_f = Vr * b * w_a if Vr else math.inf
    return {"lift_slope": a, "e_chord": float(e[0]), "q_divergence": q_d,
            "V_divergence_eas": math.sqrt(2 * q_d / 1.225) if math.isfinite(q_d) else math.inf,
            "reversal": rev,
            "flutter": {"f_bending_hz": w_h / (2 * math.pi), "f_torsion_hz": w_a / (2 * math.pi),
                        "station": eta_r, "mu": mu, "a": a_h, "x_alpha": x_a, "r2": r2, "sigma": sigma,
                        "V_reduced": Vr, "freq_ratio": wr, "V_flutter": V_f, "rho": rho,
                        "f_flutter_hz": (wr * w_a / (2 * math.pi)) if wr else None,
                        "vg": {"V": (br[:, :, 0] * b * w_a).tolist(), "g": br[:, :, 1].tolist(),
                               "f": (br[:, :, 2] * w_a / (2 * math.pi)).tolist()}},
            "V_D": V_D, "flutter_margin_required": 1.2 * V_D if V_D else None}


def run(ac, loads, settings=None, surface=0, class_ii_mass=None, log=print):
    st = size(ac, loads, settings, surface)
    defl = [deflection(st, c) for c in loads["cases"]]
    V_D = loads.get("vn", {}).get("Vd")
    ae = aeroelastic(ac, st, V_D, surface=surface)
    worst = max(defl, key=lambda d: abs(d["tip_deflection"]))
    one_g = min(defl, key=lambda d: abs(d["n"] - 1.0))
    res = {"aircraft": ac.name, "structure": st, "deflection": defl, "aeroelastic": ae,
           "tip_deflection_1g": one_g["tip_deflection"], "tip_deflection_max": worst["tip_deflection"],
           "tip_deflection_max_case": worst["case"], "tip_twist_max_deg": max(abs(d["tip_twist_deg"]) for d in defl),
           "wing_mass": st["mass"], "wing_mass_assumed_in_loads": loads.get("wing_mass_frac", 0) * ac.mass,
           "class_ii_mass": class_ii_mass}
    log(f"[structure] wing {st['mass']:.1f} kg; tip deflection 1 g {100 * res['tip_deflection_1g']:.1f} cm, "
        f"max {100 * res['tip_deflection_max']:.1f} cm ({worst['case']}); divergence "
        f"{ae['V_divergence_eas']:.0f} m/s EAS; flutter {ae['flutter']['V_flutter']:.0f} m/s; V_D {V_D}")
    return res


def class_ii_wing(ac, n_limit=3.8, cruise_speed=55.0, cruise_alt=1500.0, fuel=0.0):
    """Raymer's GA wing mass for comparison (sizing.class_ii)"""
    import sizing as SZ
    req = SZ.default_requirements() | {"load_factor": n_limit, "cruise_speed": cruise_speed,
                                       "cruise_alt": cruise_alt}
    a2 = copy.deepcopy(ac)
    if len(a2.surfaces) < 3:
        return None
    return SZ.class_ii(req, a2, ac.mass, fuel, 50.0, 12.0)["wing"]


def table(st):
    rows = []
    for i in range(len(st["eta"])):
        rows.append({"eta": st["eta"][i], "y (m)": st["y"][i], "chord (m)": st["chord"][i],
                     "box depth (mm)": 1e3 * st["h_mean"][i], "upper cap (mm²)": 1e6 * st["A_upper_cap"][i],
                     "lower cap (mm²)": 1e6 * st["A_lower_cap"][i], "web t (mm)": 1e3 * st["t_web"][i],
                     "skin t (mm)": 1e3 * st["t_skin"][i], "EI (N m²)": st["EI"][i], "GJ (N m²)": st["GJ"][i],
                     "mass (kg/m)": st["m_span"][i],
                     "upper cap case": st["critical"]["upper cap"][i],
                     "lower cap case": st["critical"]["lower cap"][i],
                     "webs case": st["critical"]["webs"][i]})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aircraft")
    ap.add_argument("loads", help="loads.json of a loads.py run")
    ap.add_argument("--material", default="aluminium 2024-T3", choices=list(MATERIALS))
    ap.add_argument("--spars", default="0.15,0.65")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    ac = A.Aircraft.load(a.aircraft)
    loads = json.loads(Path(a.loads).read_text())
    xf, xr = (float(v) for v in a.spars.split(","))
    res = run(ac, loads, {"material": a.material, "front_spar": xf, "rear_spar": xr})
    if a.out:
        import pandas as pd
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "structure.json").write_text(json.dumps(A._jsonable(res), indent=1))
        pd.DataFrame(table(res["structure"])).to_csv(out / "structure.csv", index=False)


if __name__ == "__main__":
    main()
