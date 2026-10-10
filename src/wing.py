"""
Phase 6: quasi-3D wing analysis - vortex lattice + the CFD section polars.

    python wing.py --airfoil naca2412 --span 10 --root-chord 1.6 --tip-chord 0.8 \\
                   --sweep 5 --dihedral 3 --twist -2 --speed 40 --alpha -4:14:1
    python wing.py --airfoil naca2412 --tip-airfoil naca0012 ... --polar-dir ../cases/polars
    python wing.py --polar root.json --tip-polar tip.json ...   (airfoil.py polar.json files)

Method
  * Weissinger vortex lattice: one horseshoe vortex per spanwise strip, bound
    segment on the quarter-chord line (sweep, dihedral, taper exact), flow
    tangency at the three-quarter chord point, trailing legs to downstream
    infinity; cosine spacing toward the tips. Captures sweep, taper,
    twist, dihedral and low aspect ratio, which classical lifting-line
    theory does not.
  * Nonlinear sections (decambering, Mukherjee & Gopalarathnam 2006): every
    strip gets an extra incidence delta_i such that its VLM lift equals the
    CFD section polar at the strip's effective angle,
        Cl_i(delta) = Cl_2D(alpha_eff,i),  alpha_eff,i = Cl_i/a0 + alpha_L0,i - delta_i
    (a0 = 2 pi / beta, the strip's own lift slope), solved by Newton. Gives the
    nonlinear lift curve, stall onset (where Cl_i reaches Cl_max) and C_L,max.
    Profile drag and section moments from the polar at alpha_eff (strip theory).
  * Induced drag in the Trefftz plane (exact for the lattice's loading).
  * Compressibility: Prandtl-Glauert / Goethert (x stretched by 1/beta), with
    section polars from the compressible solver above M 0.3.
  * Section data: the polars from polar.py (our CFD), at the root and tip
    Reynolds numbers (chord * speed / nu, ISA atmosphere), interpolated in
    log Re and linearly along the span; outside the computed angles the
    Viterna-Corrigan post-stall model. Polars are cached (--polar-dir).
Limits: attached-flow vortex lattice (no vortex lift / leading-edge vortices
on strongly swept or delta wings); post-stall is strip theory (trend only);
no fuselage; steady, symmetric flight (no sideslip).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import airfoil_geom as ag

# ----------------------------------------------------------------------
# atmosphere
# ----------------------------------------------------------------------
def isa(alt_m=0.0):
    """ISA troposphere: (rho, mu, a, T, p)"""
    T = 288.15 - 0.0065 * alt_m
    p = 101325.0 * (T / 288.15) ** 5.25588
    rho = p / (287.053 * T)
    mu = 1.458e-6 * T**1.5 / (T + 110.4)
    return rho, mu, math.sqrt(1.4 * 287.053 * T), T, p


# ----------------------------------------------------------------------
# section polars
# ----------------------------------------------------------------------
class SectionPolar:
    """Cl, Cd, Cm (alpha in deg) from tabulated data + Viterna-Corrigan
    post-stall extension to +-90 deg (aspect ratio ar of the wing)."""

    def __init__(self, alpha, cl, cd, cm, ar=8.0, label=""):
        o = np.argsort(alpha)
        a, cl, cd, cm = (np.asarray(v, float)[o] for v in (alpha, cl, cd, cm))
        self.label = label
        self.a_data = (a[0], a[-1])
        # zero-lift angle and lift slope from the linear range
        lin = (a >= a[0]) & (np.abs(cl) < 0.6 * np.abs(cl).max())
        if lin.sum() >= 2:
            k = np.polyfit(np.radians(a[lin]), cl[lin], 1)
            self.slope, self.alpha0 = k[0], -k[1] / k[0]           # per rad, rad
        else:
            self.slope, self.alpha0 = 2 * math.pi, 0.0
        i = int(np.argmax(cl))
        self.clmax, self.a_clmax = cl[i], a[i]
        # Viterna-Corrigan beyond the data (both ends)
        cdmax = 1.11 + 0.018 * min(ar, 50.0)
        ext_a, ext = [], []
        for end, sgn in ((-1, 1.0), (0, -1.0)):
            a_s = np.radians(a[end]); cl_s, cd_s = cl[end], cd[end]
            B1 = cdmax
            A1 = B1 / 2
            s, c = math.sin(a_s), math.cos(a_s)
            B2 = (cd_s - cdmax * s * s) / max(c, 1e-6)
            A2 = (cl_s - cdmax * s * c) * s / max(c * c, 1e-6)
            rng = np.radians(np.linspace(a[end] + sgn * 2, sgn * 90.0, 30))
            if sgn < 0:
                # negative side: mirror the model about the data end
                s0, c0 = math.sin(-a_s), math.cos(a_s)
                B2 = (cd_s - cdmax * s0 * s0) / max(c0, 1e-6)
                A2 = (-cl_s - cdmax * s0 * c0) * s0 / max(c0 * c0, 1e-6)
                r = -rng
                clv = -(A1 * np.sin(2 * r) + A2 * np.cos(r) ** 2 / np.maximum(np.sin(r), 1e-6))
                cdv = B1 * np.sin(r) ** 2 + B2 * np.cos(r)
            else:
                clv = A1 * np.sin(2 * rng) + A2 * np.cos(rng) ** 2 / np.maximum(np.sin(rng), 1e-6)
                cdv = B1 * np.sin(rng) ** 2 + B2 * np.cos(rng)
            ext_a.append(np.degrees(rng)); ext.append((clv, cdv))
        hi_a, (hi_cl, hi_cd) = ext_a[0], ext[0]
        lo_a, (lo_cl, lo_cd) = ext_a[1][::-1], (ext[1][0][::-1], ext[1][1][::-1])
        self.a = np.r_[lo_a, a, hi_a]
        self.cl = np.r_[lo_cl, cl, hi_cl]
        self.cd = np.r_[np.maximum(lo_cd, cd.min()), cd, np.maximum(hi_cd, cd.min())]
        cm_far = -0.5 * np.sin(np.radians(self.a))          # rough c.p. near mid-chord
        self.cm = np.r_[cm_far[:len(lo_a)] * 0 + cm[0], cm, cm_far[-len(hi_a):] * 0 + cm[-1]]

    def __call__(self, alpha_deg):
        a = np.clip(alpha_deg, -90, 90)
        return (np.interp(a, self.a, self.cl), np.interp(a, self.a, self.cd),
                np.interp(a, self.a, self.cm))

    def dcl(self, alpha_deg, h=0.25):
        """d Cl / d alpha per RAD"""
        return (self(alpha_deg + h)[0] - self(alpha_deg - h)[0]) / math.radians(2 * h)

    @classmethod
    def thin_airfoil(cls, a0=2 * math.pi, alpha0_deg=0.0, cd0=0.0, clmax=None, ar=8.0):
        """linear section (validation): Cl = a0 (alpha - alpha0), optional cap"""
        al = np.linspace(-20, 20, 81)
        cl = a0 * np.radians(al - alpha0_deg)
        if clmax is not None:
            cl = np.clip(cl, -clmax, clmax)
        return cls(al, cl, np.full_like(al, cd0), np.zeros_like(al), ar=ar, label="thin airfoil")

    @classmethod
    def from_results(cls, results, ar=8.0, label=""):
        r = [x for x in results if x.get("Cl") is not None and np.isfinite(x["Cl"])
             and not str(x.get("status", "")).startswith("diverged")]
        return cls([x["alpha"] for x in r], [x["Cl"] for x in r], [x["Cd"] for x in r],
                   [x["Cm"] for x in r], ar=ar, label=label)


class SpanPolars:
    """Section data along the span: polars at several Re (interpolated in
    log Re) for the root and the tip airfoil (interpolated linearly in |y|)."""

    def __init__(self, root, tip=None):
        """root/tip: list of (Re, SectionPolar)"""
        self.root = sorted(root, key=lambda t: t[0])
        self.tip = sorted(tip, key=lambda t: t[0]) if tip else None

    @staticmethod
    def _re_interp(table, re, alpha):
        if len(table) == 1:
            return table[0][1](alpha), table[0][1]
        lr = np.log([t[0] for t in table])
        x = np.clip(math.log(re), lr[0], lr[-1])
        j = int(np.clip(np.searchsorted(lr, x) - 1, 0, len(lr) - 2))
        w = (x - lr[j]) / (lr[j + 1] - lr[j])
        a, b = table[j][1](alpha), table[j + 1][1](alpha)
        return tuple((1 - w) * p + w * q for p, q in zip(a, b)), table[j + int(w > 0.5)][1]

    def __call__(self, alpha, re, eta):
        """alpha (deg), Re, eta = |y|/(b/2) per strip -> Cl, Cd, Cm arrays"""
        out = np.zeros((3, len(alpha)))
        for i, (a, r, e) in enumerate(zip(alpha, re, eta)):
            v, _ = self._re_interp(self.root, r, a)
            if self.tip is not None:
                vt, _ = self._re_interp(self.tip, r, a)
                v = tuple((1 - e) * p + e * q for p, q in zip(v, vt))
            out[:, i] = v
        return out

    @staticmethod
    def _props_re(table, re):
        """(alpha0, Cl_max, alpha at Cl_max), interpolated in log Re"""
        P = [np.array([t[1].alpha0, t[1].clmax, t[1].a_clmax]) for t in table]
        if len(table) == 1:
            return P[0]
        lr = np.log([t[0] for t in table])
        x = np.clip(math.log(re), lr[0], lr[-1])
        j = int(np.clip(np.searchsorted(lr, x) - 1, 0, len(lr) - 2))
        w = (x - lr[j]) / (lr[j + 1] - lr[j])
        return (1 - w) * P[j] + w * P[j + 1]

    def props(self, re, eta):
        """alpha0 (rad), Cl_max, alpha(Cl_max) per strip"""
        out = []
        for r, e in zip(re, eta):
            v = self._props_re(self.root, r)
            if self.tip is not None:
                v = (1 - e) * v + e * self._props_re(self.tip, r)
            out.append(v)
        out = np.array(out)
        return out[:, 0], out[:, 1], out[:, 2]


# ----------------------------------------------------------------------
# geometry + lattice
# ----------------------------------------------------------------------
class Wing:
    """Trapezoidal wing (optionally with a crank: list of sections).
    span b, root/tip chord, quarter-chord sweep, dihedral (deg), linear
    twist root -> tip (deg, negative = washout), root incidence (deg)."""

    def __init__(self, span, root_chord, tip_chord=None, sweep=0.0, dihedral=0.0, twist=0.0,
                 incidence=0.0, n_half=40, chord_fn=None):
        self.b = float(span)
        self.cr = float(root_chord)
        self.ct = float(root_chord if tip_chord is None else tip_chord)
        self.sweep, self.dihedral = sweep, dihedral
        self.twist, self.incidence = twist, incidence
        self.n_half = n_half
        self.chord_fn = chord_fn               # optional c(eta) (e.g. elliptic)
        self._build()

    def chord(self, eta):
        eta = np.abs(eta)
        if self.chord_fn is not None:
            return self.chord_fn(eta)
        return self.cr + (self.ct - self.cr) * eta

    def _build(self):
        n = self.n_half
        th = np.linspace(0.0, math.pi, 2 * n + 1)
        eta = -np.cos(th)                                # -1 .. 1, cosine spacing
        self.eta_nodes = eta
        s = 0.5 * self.b
        y = eta * s
        tl, td = math.tan(math.radians(self.sweep)), math.tan(math.radians(self.dihedral))
        x_qc = np.abs(y) * tl                            # quarter-chord line
        z = np.abs(y) * td
        self.nodes = np.c_[x_qc, y, z]                   # bound-vortex end points
        em = 0.5 * (eta[1:] + eta[:-1])
        self.eta = em                                    # strip mid stations
        self.c = self.chord(em)
        A, B = self.nodes[:-1], self.nodes[1:]
        mid = 0.5 * (A + B)
        self.mid_qc = mid
        self.dl = B - A
        self.dy = self.dl[:, 1]
        # control points: 3/4 chord (half a chord behind the bound vortex)
        self.ctrl = mid + np.c_[0.5 * self.c, 0 * self.c, 0 * self.c]
        # strip normal (dihedral) before incidence
        phi = np.arctan2(self.dl[:, 2], self.dl[:, 1])
        self.phi = phi
        self.twist_strip = self.incidence + self.twist * np.abs(em)
        self.area_strip = self.c * np.abs(self.dy)
        self.S = self.area_strip.sum()
        self.mac = (self.c**2 * np.abs(self.dy)).sum() / self.S
        self.x_mac_qc = (self.mid_qc[:, 0] * self.c * np.abs(self.dy)).sum() / self.S
        self.AR = self.b**2 / self.S

    def describe(self):
        return {"span": self.b, "root_chord": self.cr, "tip_chord": self.ct,
                "taper": self.ct / self.cr if self.chord_fn is None else None,
                "area": self.S, "aspect_ratio": self.AR, "mac": self.mac,
                "sweep_c4_deg": self.sweep, "dihedral_deg": self.dihedral,
                "twist_deg": self.twist, "incidence_deg": self.incidence,
                "strips": 2 * self.n_half}


def _seg(P, A, B, eps=1e-10):
    """velocity at points P (n,3) of unit-strength segments A->B (m,3): (n,m,3)"""
    r1 = P[:, None, :] - A[None]
    r2 = P[:, None, :] - B[None]
    r0 = (B - A)[None]
    cr = np.cross(r1, r2)
    c2 = (cr**2).sum(-1)
    n1, n2 = np.linalg.norm(r1, axis=-1), np.linalg.norm(r2, axis=-1)
    k = np.einsum('nmk,nmk->nm', r0, r1 / np.maximum(n1, eps)[..., None] -
                  r2 / np.maximum(n2, eps)[..., None])
    f = np.where(c2 > eps * (r0**2).sum(-1), k / (4 * math.pi * np.maximum(c2, 1e-300)), 0.0)
    return cr * f[..., None]


def _semi(P, A, u, eps=1e-10):
    """velocity at P of unit-strength semi-infinite filaments from A along u (to infinity)"""
    r = P[:, None, :] - A[None]
    nr = np.linalg.norm(r, axis=-1)
    d = nr - np.einsum('nmk,k->nm', r, u)
    cr = np.cross(u[None, None, :], r)
    ok = (nr * d) > eps
    f = np.where(ok, 1.0 / (4 * math.pi * np.maximum(nr * d, 1e-300)), 0.0)
    return cr * f[..., None]


def horseshoe_matrix(P, A, B, u=np.array([1.0, 0.0, 0.0]), skip_bound=None):
    """(n,m,3) velocity at P per unit circulation of horseshoes inf->A->B->inf"""
    V = _seg(P, A, B) + _semi(P, B, u) - _semi(P, A, u)
    if skip_bound is not None:                   # self-induced bound segment
        V[skip_bound] -= _seg(P, A, B)[skip_bound]
    return V


class VLM:
    """Vortex lattice: nc chordwise x (2 n_half) spanwise flat panels per
    strip, horseshoe vortex on each panel's quarter chord, tangency at its
    three-quarter chord (nc = 1: Weissinger's method). Camber enters through
    the strip incidence (zero-lift angle of the section polar)."""

    def __init__(self, wing, mach=0.0, nc=6):
        self.w = w = wing
        self.mach = mach
        self.nc = nc
        self.beta = math.sqrt(1.0 - mach**2) if mach < 0.98 else 0.2
        st = np.array([1.0 / self.beta, 1.0, 1.0])     # Goethert: x -> x / beta
        ns = len(w.c)
        # strip edge chords and leading edges
        c_node = w.chord(w.eta_nodes)
        le_node = w.nodes - np.c_[0.25 * c_node, 0 * c_node, 0 * c_node]
        f_b = (np.arange(nc) + 0.25) / nc                # bound vortex, chord fraction
        f_c = (np.arange(nc) + 0.75) / nc                # control point
        A = (le_node[:-1, None, :] + f_b[None, :, None] * np.c_[c_node[:-1], 0 * c_node[:-1],
                                                                0 * c_node[:-1]][:, None, :])
        B = (le_node[1:, None, :] + f_b[None, :, None] * np.c_[c_node[1:], 0 * c_node[1:],
                                                               0 * c_node[1:]][:, None, :])
        le_mid = 0.5 * (le_node[:-1] + le_node[1:])
        C = le_mid[:, None, :] + f_c[None, :, None] * np.c_[w.c, 0 * w.c, 0 * w.c][:, None, :]
        self.xb = 0.5 * (A[..., 0] + B[..., 0])          # bound-vortex x per panel (ns, nc)
        A, B, C = (X.reshape(-1, 3) * st for X in (A, B, C))
        V = horseshoe_matrix(C, A, B)
        self.n0 = np.c_[np.zeros_like(w.phi), -np.sin(w.phi), np.cos(w.phi)]
        n0p = np.repeat(self.n0, nc, axis=0)
        self.AIC = np.einsum('nmk,nk->nm', V, n0p)
        self.lu = np.linalg.inv(self.AIC)
        self.strip_of = np.repeat(np.arange(ns), nc)
        self.a0 = 2 * math.pi / self.beta             # strip (2D) lift slope
        self.last_panel = None

    def _rhs(self, alpha_deg, theta):
        a = math.radians(alpha_deg)
        w = self.w
        n = np.c_[np.sin(theta), -np.sin(w.phi) * np.cos(theta), np.cos(w.phi) * np.cos(theta)]
        Vinf = np.array([math.cos(a), 0.0, math.sin(a)])
        return np.repeat(-(n @ Vinf), self.nc)

    def _strip(self, Gp):
        return Gp.reshape(-1, self.nc).sum(1)

    def gamma(self, alpha_deg, theta):
        """strip circulation for free-stream angle alpha and strip incidences
        theta (rad): tangency (V_inf + v).n = 0, n rotated nose-up by theta"""
        Gp = self.lu @ self._rhs(alpha_deg, theta)
        self.last_panel = Gp.reshape(-1, self.nc)
        return self._strip(Gp)

    def dgamma(self, alpha_deg, theta):
        """d(strip circulation)/d(strip incidence), (ns, ns)"""
        a = math.radians(alpha_deg)
        w = self.w
        d = -(math.cos(a) * np.cos(theta) - math.sin(a) * np.cos(w.phi) * np.sin(theta))
        ns = len(theta)
        D = np.zeros((ns * self.nc, ns))
        D[np.arange(ns * self.nc), self.strip_of] = np.repeat(d, self.nc)
        return (self.lu @ D).reshape(ns, self.nc, ns).sum(1)

    def cl_strip(self, G):
        return 2.0 * G / self.w.c                  # V = 1

    def cdi(self, G):
        """induced drag coefficient from a Fourier (sine) series of the span
        loading, Gamma = 2 b sum A_n sin(n theta): CDi = pi AR sum n A_n^2.
        (The point-vortex Trefftz sum with a piecewise-constant Gamma
        overestimates e by ~1.5 % at 40 strips per side; the series is exact
        for the loading it represents.) Planar-wing formula: dihedral is
        neglected in the induced drag."""
        w = self.w
        th = np.arccos(np.clip(-w.eta, -1, 1))
        K = np.arange(1, min(len(G) - 2, 60) + 1)
        A = np.linalg.lstsq(np.sin(np.outer(th, K)), G / (2 * w.b), rcond=None)[0]
        return math.pi * w.AR * (K * A**2).sum()

    def trefftz_cdi(self, G):
        """induced drag coefficient (V = 1, q = 1/2), point vortices in the
        Trefftz plane (kept for reference, see cdi)"""
        w = self.w
        g = np.r_[G, 0.0] - np.r_[0.0, G]                # trailing strengths at nodes
        yz = w.nodes[:, 1:]
        mid = 0.5 * (yz[1:] + yz[:-1])
        r = mid[:, None, :] - yz[None]
        r2 = np.maximum((r**2).sum(-1), 1e-30)
        vy = (-(g[None] * r[..., 1]) / (2 * math.pi * r2)).sum(1)
        vz = ((g[None] * r[..., 0]) / (2 * math.pi * r2)).sum(1)
        wn = vy * self.n0[:, 1] + vz * self.n0[:, 2]
        ds = np.linalg.norm(w.dl[:, 1:], axis=1)
        Di = 0.5 * (G * wn * ds).sum()                   # rho = 1 (wn < 0: downwash)
        return Di / (0.5 * w.S)

    def CL(self, G):
        return (G * self.w.dy).sum() / (0.5 * self.w.S)


# ----------------------------------------------------------------------
# solver: linear or nonlinear (decambering)
# ----------------------------------------------------------------------
def solve(wing, alpha, polars=None, re_strip=None, mach=0.0, vlm=None, delta0=None,
          x_ref=None, tol=1e-6, max_it=60, nc=6):
    """alpha (deg). polars: SpanPolars (None = linear thin-airfoil sections with
    alpha0 = 0). Returns dict of wing coefficients + span distributions."""
    v = vlm or VLM(wing, mach, nc)
    w = wing
    n = len(w.c)
    tw = np.radians(w.twist_strip)
    if polars is None:
        a0s = np.zeros(n)
        clmax = np.full(n, np.inf)
        a_clmax = np.full(n, np.inf)
    else:
        a0s, clmax, a_clmax = polars.props(re_strip, np.abs(w.eta))
    theta_base = tw - a0s                              # camber via the zero-lift angle
    # linear influence of the strip incidences: dG/d(delta)
    Gd = v.dgamma(alpha, theta_base)
    delta = np.zeros(n) if delta0 is None else np.array(delta0, float)
    it, conv = 0, True
    cd = cm = None
    if polars is not None:
        conv = False
        for it in range(1, max_it + 1):
            G = v.gamma(alpha, theta_base + delta)
            cl = v.cl_strip(G)
            a_eff = cl / v.a0 + a0s - delta
            cl2, cd, cm = polars(np.degrees(a_eff), re_strip, np.abs(w.eta))
            R = cl - cl2
            if np.abs(R).max() < tol:
                conv = True
                break
            # Jacobian: dcl/ddelta - dcl2/da_eff * (dcl/ddelta / a0 - I)
            B = 2.0 * Gd / w.c[:, None]
            s = np.array([polars_slope(polars, ae, r, e) for ae, r, e in
                          zip(np.degrees(a_eff), re_strip, np.abs(w.eta))])
            J = B - s[:, None] * (B / v.a0 - np.eye(n))
            try:
                step = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                step = -R / np.diag(J)
            lim = math.radians(4.0)
            step = np.clip(step, -lim, lim)
            delta = delta + (1.0 if it < 30 else 0.5) * step
    G = v.gamma(alpha, theta_base + delta)
    cl = v.cl_strip(G)
    a_eff = cl / v.a0 + a0s - delta
    if polars is not None:
        _, cd, cm = polars(np.degrees(a_eff), re_strip, np.abs(w.eta))
    else:
        cd, cm = np.zeros(n), np.zeros(n)
    CL = v.CL(G)
    CDi = v.cdi(G)
    CDp = (cd * w.area_strip).sum() / w.S
    xr = w.x_mac_qc if x_ref is None else x_ref
    L = G * w.dy                                         # rho = V = 1
    # lift of every panel at its bound vortex (flat-plate moment) + section Cm
    Lp = v.last_panel * w.dy[:, None]
    M = (-(v.xb - xr) * Lp).sum() + (cm * 0.5 * w.c**2 * np.abs(w.dy)).sum()
    Cm = M / (0.5 * w.S * w.mac)
    e = CL**2 / (math.pi * w.AR * CDi) if CDi > 1e-12 else float("nan")
    # stalled: at (98 % of) the section c_l,max or past its angle
    stalled = (cl > 0.98 * clmax) | (np.degrees(a_eff) > a_clmax)
    return {"alpha": alpha, "CL": CL, "CDi": CDi, "CDp": CDp, "CD": CDi + CDp, "Cm": Cm,
            "e": e, "L_D": CL / (CDi + CDp) if CDi + CDp > 0 else float("nan"),
            "converged": conv, "iterations": it, "stall_fraction": float(
                (w.area_strip * stalled).sum() / w.S),
            "span": {"eta": w.eta, "y": w.mid_qc[:, 1], "cl": cl, "cd": cd, "cm": cm,
                     "alpha_eff": np.degrees(a_eff), "gamma": G, "cl_c": cl * w.c,
                     "clmax": clmax, "delta": delta, "stalled": stalled}, "_delta": delta}


def polars_slope(polars, a_deg, re, eta, h=0.25):
    lo = polars(np.array([a_deg - h]), np.array([re]), np.array([eta]))[0][0]
    hi = polars(np.array([a_deg + h]), np.array([re]), np.array([eta]))[0][0]
    return (hi - lo) / math.radians(2 * h)


def sweep_alpha(wing, alphas, polars=None, re_strip=None, mach=0.0, x_ref=None):
    """alpha sweep with continuation of the decambering solution; returns
    list of results (span distributions included)"""
    v = VLM(wing, mach)
    out, d = [], None
    order = np.argsort(np.abs(np.asarray(alphas, float)))
    res = {}
    # sweep upward and downward from the smallest |alpha| (stall continuation)
    al = np.asarray(alphas, float)
    start = al[order[0]]
    for direction in (1, -1):
        d = None
        seq = sorted([a for a in al if (a >= start if direction > 0 else a < start)],
                     reverse=direction < 0)
        for a in seq:
            r = solve(wing, a, polars, re_strip, mach, vlm=v, delta0=d, x_ref=x_ref)
            d = r["_delta"]
            res[a] = r
    for a in al:
        out.append(res[a])
    # CL_max: first maximum of CL(alpha)
    return out


# ----------------------------------------------------------------------
# section polars from the CFD tool (cached)
# ----------------------------------------------------------------------
def cfd_polars(airfoil, re_list, mach, alphas, cache_dir, level=1.0, jobs=2, ar=8.0,
               transition="free", log=print):
    """list of (Re, SectionPolar); runs polar.polar for missing ones"""
    import polar as pl
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for re in re_list:
        slug = f"{ag.section_stem(airfoil)}_Re{re:.3g}_M{mach:.2f}_L{level:g}_{transition}"
        f = cache / (slug.replace("+", "") + ".json")
        if f.exists():
            data = json.loads(f.read_text())
        else:
            log(f"[wing] section polar {airfoil} Re {re:.3g} M {mach:.2f} (CFD, {len(alphas)} angles)")
            t0 = time.time()
            m = mach if mach >= 0.3 else 0.0
            res = pl.polar(airfoil, re, alphas, level=level, jobs=jobs, log=False,
                           transition=transition if m < 0.4 else "off", mach=m)
            data = [{k: r.get(k) for k in ("alpha", "Cl", "Cd", "Cm", "status", "iterations",
                                           "Cd_p", "Cd_v", "x_tr_upper", "x_tr_lower")}
                    for r in res]
            if 0 < mach < 0.3:              # Prandtl-Glauert for the low-speed polars
                b = 1 / math.sqrt(1 - mach**2)
                for r in data:
                    r["Cl"] *= b; r["Cm"] *= b
            f.write_text(json.dumps(data, indent=1))
            log(f"[wing]   done in {time.time() - t0:.0f} s -> {f}")
        # angles that diverged with free transition (near stall the e^N
        # coupling can blow up: NACA 2412, Re 2.2e6, alpha >= 14) -> rerun
        # them fully turbulent (transition is at the leading edge there anyway)
        bad = [r for r in data if not (r.get("Cl") is not None and np.isfinite(r["Cl"]))]
        if bad and transition != "off":
            m = mach if mach >= 0.3 else 0.0
            for r in bad:
                log(f"[wing]   alpha {r['alpha']:g}: diverged with free transition -> "
                    f"fully turbulent")
                try:
                    q, _ = pl.run_point(airfoil, re, r["alpha"], level=level, transition="off",
                                        mach=m)
                    r.update({k: q[k] for k in ("Cl", "Cd", "Cm", "iterations")})
                    r["status"] = q["status"] + " (fully turbulent fallback)"
                    if 0 < mach < 0.3:
                        b = 1 / math.sqrt(1 - mach**2)
                        r["Cl"] *= b; r["Cm"] *= b
                except (ValueError, FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError) as e:
                    r["status"] = f"diverged ({type(e).__name__})"
            f.write_text(json.dumps(data, indent=1))
        out.append((re, SectionPolar.from_results(data, ar=ar, label=f.stem)))
    return out


def load_polar_file(path, ar=8.0):
    d = json.loads(Path(path).read_text())
    res = d["results"] if isinstance(d, dict) else d
    re = d.get("re", 1e6) if isinstance(d, dict) else 1e6
    return [(re, SectionPolar.from_results(res, ar=ar, label=Path(path).stem))]


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------
def report(wing, results, out, title, flight=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    al = np.array([r["alpha"] for r in results])
    CL = np.array([r["CL"] for r in results])
    CD = np.array([r["CD"] for r in results])
    CDi = np.array([r["CDi"] for r in results])
    Cm = np.array([r["Cm"] for r in results])
    i_max = int(np.argmax(CL))
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    ax = axs[0, 0]
    ax.plot(al, CL, "b.-"); ax.set_xlabel("alpha (deg)"); ax.set_ylabel("C_L"); ax.grid(alpha=.3)
    ax.set_title(f"C_L,max {CL[i_max]:.3f} at {al[i_max]:g} deg")
    ax = axs[0, 1]
    ax.plot(CD, CL, "b.-", label="C_D total"); ax.plot(CDi, CL, "g--", label="induced")
    ax.set_xlabel("C_D"); ax.set_ylabel("C_L"); ax.grid(alpha=.3); ax.legend()
    ax = axs[0, 2]
    ax.plot(al, Cm, "b.-"); ax.set_xlabel("alpha (deg)"); ax.set_ylabel("C_m (MAC c/4)")
    ax.grid(alpha=.3)
    # span loading at three angles
    ax = axs[1, 0]
    pick = sorted(set([int(np.argmin(np.abs(al - a))) for a in (2.0, 0.6 * al[i_max], al[i_max])]))
    for k in pick:
        sp = results[k]["span"]
        ax.plot(sp["y"], sp["cl_c"] / wing.mac, label=f"alpha {al[k]:g}")
        CLk = results[k]["CL"]
        ye = np.linspace(-wing.b / 2, wing.b / 2, 200)
        ax.plot(ye, 4 * CLk * wing.S / (math.pi * wing.b) * np.sqrt(1 - (2 * ye / wing.b) ** 2)
                / wing.mac, "k:", lw=0.8)
    ax.set_xlabel("y"); ax.set_ylabel("c_l c / MAC  (dotted: elliptic)"); ax.legend(); ax.grid(alpha=.3)
    ax = axs[1, 1]
    for k in pick:
        sp = results[k]["span"]
        ax.plot(sp["y"], sp["cl"], label=f"alpha {al[k]:g}")
    if np.isfinite(results[0]["span"]["clmax"]).all():
        ax.plot(results[0]["span"]["y"], results[0]["span"]["clmax"], "r--", label="section c_l,max")
    ax.set_xlabel("y"); ax.set_ylabel("local c_l"); ax.legend(); ax.grid(alpha=.3)
    ax = axs[1, 2]                                       # planform
    s = wing.b / 2
    tl = math.tan(math.radians(wing.sweep))
    xs = lambda y, c: abs(y) * tl - 0.25 * c
    ys = np.linspace(-s, s, 101)
    cc = wing.chord(ys / s)
    ax.plot(ys, -np.array([xs(y, c) for y, c in zip(ys, cc)]), "k")
    ax.plot(ys, -np.array([xs(y, c) + c for y, c in zip(ys, cc)]), "k")
    ax.plot([-s, -s], [-xs(-s, cc[0]), -xs(-s, cc[0]) - cc[0]], "k")
    ax.plot([s, s], [-xs(s, cc[-1]), -xs(s, cc[-1]) - cc[-1]], "k")
    sp = results[i_max]["span"]
    st = sp["stalled"]
    if st.any():
        ax.scatter(sp["y"][st], -(wing.mid_qc[st, 0] + 0.25 * wing.c[st]), c="r", s=12,
                   label=f"stalled strips at alpha {al[i_max]:g}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=8)
    ax.set_aspect("equal"); ax.set_title(f"S {wing.S:.3g}, AR {wing.AR:.2f}, MAC {wing.mac:.3g}")
    ax.set_xlabel("y"); ax.set_ylabel("x (flow downward)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / "wing.png", dpi=120)
    plt.close(fig)
    cols = ["alpha", "CL", "CD", "CDi", "CDp", "Cm", "L_D", "e", "stall_fraction", "converged"]
    with open(out / "wing_polar.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    lines = [title, json.dumps(wing.describe()), ""]
    if flight:
        lines += [json.dumps(flight), ""]
    lines.append(f"{'alpha':>6s} {'CL':>7s} {'CD':>8s} {'CDi':>8s} {'CDp':>8s} {'Cm':>7s} "
                 f"{'L/D':>6s} {'e':>6s} {'stall':>6s}")
    for r in results:
        lines.append(f"{r['alpha']:6.2f} {r['CL']:7.4f} {r['CD']:8.5f} {r['CDi']:8.5f} "
                     f"{r['CDp']:8.5f} {r['Cm']:+7.4f} {r['L_D']:6.1f} {r['e']:6.3f} "
                     f"{100 * r['stall_fraction']:5.0f}%" + ("" if r["converged"] else "  (not converged)"))
    lines += ["", f"C_L,max {CL[i_max]:.4f} at alpha {al[i_max]:g} deg",
              "max L/D {:.1f} at alpha {:g} deg".format(*max((r["L_D"], r["alpha"]) for r in results))]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    js = [{k: v for k, v in r.items() if k not in ("span", "_delta")} |
          {"span": {k: np.asarray(v).tolist() for k, v in r["span"].items()}} for r in results]
    (out / "wing.json").write_text(json.dumps({"wing": wing.describe(), "flight": flight,
                                               "results": js}, indent=1, default=float))
    return "\n".join(lines)


# ----------------------------------------------------------------------
def parse_alpha(s):
    if ":" in s:
        a, b, c = (float(x) for x in s.split(":"))
        return list(np.round(np.arange(a, b + 0.5 * c, c), 6))
    return [float(x) for x in s.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--airfoil", help="root airfoil (NACA code, tmr0012 or .dat)")
    ap.add_argument("--tip-airfoil", help="tip airfoil (default: same as root)")
    ap.add_argument("--polar", help="root section polar.json (from airfoil.py) instead of CFD")
    ap.add_argument("--tip-polar")
    ap.add_argument("--span", type=float, required=True)
    ap.add_argument("--root-chord", type=float, required=True)
    ap.add_argument("--tip-chord", type=float)
    ap.add_argument("--sweep", type=float, default=0.0, help="quarter-chord sweep (deg)")
    ap.add_argument("--dihedral", type=float, default=0.0)
    ap.add_argument("--twist", type=float, default=0.0, help="tip twist relative to root (deg)")
    ap.add_argument("--incidence", type=float, default=0.0, help="root incidence (deg)")
    ap.add_argument("--speed", type=float, help="m/s (with --altitude: Re and Mach from ISA)")
    ap.add_argument("--altitude", type=float, default=0.0)
    ap.add_argument("--re", type=float, help="Reynolds number on the MAC (instead of --speed)")
    ap.add_argument("--mach", type=float, help="free-stream Mach (default from --speed)")
    ap.add_argument("--alpha", default="-4:14:1")
    ap.add_argument("--polar-alpha", default="-6:22:2", help="angles of the CFD section polars (past stall)")
    ap.add_argument("--polar-dir", default="../cases/wing_polars")
    ap.add_argument("--refine", type=float, default=1.0, help="CFD mesh level of the section polars")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--transition", choices=["free", "off"], default="free")
    ap.add_argument("--strips", type=int, default=40, help="strips per half span")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    wing = Wing(a.span, a.root_chord, a.tip_chord, a.sweep, a.dihedral, a.twist, a.incidence,
                n_half=a.strips)
    if a.speed:
        rho, mu, snd, T, p = isa(a.altitude)
        mach = a.mach if a.mach is not None else a.speed / snd
        re_per_m = rho * a.speed / mu
        flight = {"speed_m_s": a.speed, "altitude_m": a.altitude, "rho": rho, "mach": mach,
                  "re_mac": re_per_m * wing.mac}
    elif a.re:
        mach = a.mach or 0.0
        re_per_m = a.re / wing.mac
        flight = {"re_mac": a.re, "mach": mach}
    else:
        raise SystemExit("give --speed (m/s) or --re")
    re_strip = re_per_m * wing.c
    re_root, re_tip = re_per_m * wing.cr, re_per_m * wing.ct
    re_list = sorted({float(f"{re_root:.2g}"), float(f"{re_tip:.2g}")})
    al_pol = parse_alpha(a.polar_alpha)
    if a.polar:
        root = load_polar_file(a.polar, wing.AR)
        tip = load_polar_file(a.tip_polar, wing.AR) if a.tip_polar else None
    elif a.airfoil:
        root = cfd_polars(a.airfoil, re_list, mach, al_pol, a.polar_dir, a.refine, a.jobs,
                          wing.AR, a.transition)
        tip = cfd_polars(a.tip_airfoil, re_list, mach, al_pol, a.polar_dir, a.refine, a.jobs,
                         wing.AR, a.transition) if a.tip_airfoil else None
    else:
        raise SystemExit("give --airfoil or --polar")
    polars = SpanPolars(root, tip)
    t0 = time.time()
    res = sweep_alpha(wing, parse_alpha(a.alpha), polars, re_strip, mach)
    name = a.airfoil or Path(a.polar).stem
    out = a.out or f"../cases/wing_{Path(str(name)).stem}_b{a.span:g}"
    title = (f"Wing: {name}" + (f" -> {a.tip_airfoil}" if a.tip_airfoil else "") +
             f", b {a.span:g}, c {wing.cr:g}/{wing.ct:g}, sweep {a.sweep:g}, twist {a.twist:g}, "
             f"M {mach:.3f}, Re_MAC {flight['re_mac']:.3g}")
    txt = report(wing, res, out, title, flight)
    print(txt)
    print(f"[wing] {len(res)} angles in {time.time() - t0:.1f} s -> {out}/ (wing.png, report.txt, "
          f"wing_polar.csv, wing.json)")


if __name__ == "__main__":
    main()
