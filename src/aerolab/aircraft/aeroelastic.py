"""
Aeroelasticity of the wing: structural modes, 3D unsteady aerodynamics,
flutter (p-k), control-surface flutter, and static aeroelasticity (divergence,
aileron reversal, elastic lift and flexible loads) on the vortex lattice.

    python src/run.py aeroelastic ../cases/aircraft/my_aircraft.json --vmax 150 --out ../cases/aeroelastic

Structure (BeamModel)
  * Half wing as a beam along a straight elastic axis (root e.a. -> tip e.a.,
    sweep Lambda_ea), clamped at the symmetry plane. Bending w(s) with
    Hermite cubic elements, torsion theta(s) with linear elements, section
    properties EI, GJ, mass m, static unbalance S = m (x_cg - x_ea) and
    inertia about the e.a. from the wing-structure sizing (or given).
  * Aileron (optional): one rigid rotation beta about the hinge per side,
    hinge spring K_beta (from the uncoupled aileron frequency), aileron mass,
    inertia and static unbalance about the hinge (mass balance < 0 = c.g.
    ahead of the hinge). Kinetic energy of every mass point with
        z = w - d theta - (x - x_h) beta      (d: distance aft of the e.a.)
    so the wing-aileron inertial coupling is exact for a rigid aileron.
  * Modes: generalised eigenproblem K phi = omega^2 M phi.

Aerodynamics (VortexRingLattice, frequency domain)
  * Vortex-ring lattice on the full-span wing (rings at the panel quarter
    chords, collocation at the three-quarter chords, hinge on a panel edge),
    flat wake of rings convected at U; for harmonic motion the wake ring
    shed m steps ago carries Gamma_TE exp(-i omega m dt) (Katz & Plotkin's
    unsteady VLM, linearised). Boundary condition on the mode shape:
        w_n = i omega z + U dz/dx
    loads: steady Kutta-Joukowski on the bound segments + rho i omega Gamma A.
    k -> 0: the steady VLM; high aspect ratio: Theodorsen's 2-D result (tests).
  * Generalised aerodynamic forces Q(k) = Phi^T F(k)/q on a reduced-frequency
    grid; symmetric and antisymmetric families (mirror images of the half-
    wing modes).
  * Strip theory (Theodorsen) as an alternative, for comparison.

Flutter: p-k method (Nastran PK):
    [M p^2 - (rho V b / 2)(Q_I / k) p + (K - q Q_R)] x = 0,  k = Im(p) b / V
matched for every mode at every speed; damping g = 2 Re p / Im p. Flutter
where g of a branch crosses zero. CS-23.629: free of flutter up to 1.2 V_D.

Static aeroelasticity: the same lattice at k = 0 and the same beam:
    (K - q Q0) u = q f       (f: rigid load, Q0: aerodynamic stiffness)
  divergence: det(K - q Q0) = 0 (symmetric family); aileron reversal: rolling
  moment of the elastic wing with the aileron deflected = 0 (antisymmetric);
  elastic CL_alpha and roll effectiveness against speed; and flexible_trim():
  the aircraft model trimmed with the wing incidence changed by the elastic
  twist and bending slope, theta cos(Lambda) - w' sin(Lambda), iterated to
  convergence - used by the flight-loads tool for flexible-wing loads.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eig, eigh

from aerolab.aircraft import aircraft as A


# ----------------------------------------------------------------------
# structure
# ----------------------------------------------------------------------
class BeamModel:
    """half-wing beam. props: dict of arrays along s (elastic-axis length, 0 at
    the root): EI, GJ, m, S (static unbalance per length, + = c.g. aft of the
    e.a.), I (mass moment of inertia per length about the e.a.).
    geometry: function s -> (x_ea, y, z, chord, x_le) at that station (x_ea
    in absolute coordinates). aileron: None or dict(s0, s1, x_h(s) function
    or fraction cf, m, S_h, I_h per length about the hinge, K) - S_h > 0 when
    the aileron c.g. is aft of the hinge."""

    def __init__(self, s, props, sweep_ea_deg=0.0, n_el=24, aileron=None, x_ea_root=0.0, label="wing"):
        self.s_data = np.asarray(s, float)
        self.props = {k: np.asarray(v, float) for k, v in props.items()}
        self.L = float(self.s_data[-1])
        self.sweep = math.radians(sweep_ea_deg)
        self.x_ea_root = x_ea_root
        self.n_el = n_el
        self.s = np.linspace(0, self.L, n_el + 1)
        self.aileron = aileron
        self.label = label
        self._assemble()

    def p(self, key, s):
        return np.interp(s, self.s_data, self.props[key])

    # dof map: node i -> [w, w', theta] at 3i..3i+2; aileron beta last
    @property
    def ndof(self):
        return 3 * (self.n_el + 1) + (1 if self.aileron else 0)

    def _assemble(self):
        n = self.ndof
        K = np.zeros((n, n)); M = np.zeros((n, n))
        g, wg = np.polynomial.legendre.leggauss(4)
        ai = self.aileron
        ib = n - 1 if ai else None
        for e in range(self.n_el):
            s0, s1 = self.s[e], self.s[e + 1]
            L = s1 - s0
            dofs = [3 * e, 3 * e + 1, 3 * e + 3, 3 * e + 4]
            tdofs = [3 * e + 2, 3 * e + 5]
            for xi, wq in zip(g, wg):
                t = 0.5 * (xi + 1)
                sp = s0 + t * L
                jac = 0.5 * L * wq
                Nw = np.array([1 - 3 * t**2 + 2 * t**3, L * (t - 2 * t**2 + t**3), 3 * t**2 - 2 * t**3,
                               L * (-t**2 + t**3)])
                B = np.array([-6 + 12 * t, L * (-4 + 6 * t), 6 - 12 * t, L * (-2 + 6 * t)]) / L**2
                Nt = np.array([1 - t, t])
                dNt = np.array([-1, 1]) / L
                EI, GJ, m, S, I = (self.p(k, sp) for k in ("EI", "GJ", "m", "S", "I"))
                K[np.ix_(dofs, dofs)] += EI * np.outer(B, B) * jac
                K[np.ix_(tdofs, tdofs)] += GJ * np.outer(dNt, dNt) * jac
                M[np.ix_(dofs, dofs)] += m * np.outer(Nw, Nw) * jac
                M[np.ix_(dofs, tdofs)] += -S * np.outer(Nw, Nt) * jac
                M[np.ix_(tdofs, dofs)] += -S * np.outer(Nt, Nw) * jac
                M[np.ix_(tdofs, tdofs)] += I * np.outer(Nt, Nt) * jac
                if ai and ai["s0"] <= sp <= ai["s1"]:
                    ma, Sh, Ih = ai["m"], ai["S_h"], ai["I_h"]
                    d_h = ai["d_h"](sp)                       # hinge distance aft of the e.a.
                    Se = Sh + ma * d_h                        # aileron unbalance about the e.a.
                    Ieh = Ih + d_h * Sh                       # product term (x - x_ea)(x - x_h)
                    # aileron mass is already in m, S, I of the section? no: given separately (added here)
                    M[np.ix_(dofs, dofs)] += ma * np.outer(Nw, Nw) * jac
                    M[np.ix_(dofs, tdofs)] += -Se * np.outer(Nw, Nt) * jac
                    M[np.ix_(tdofs, dofs)] += -Se * np.outer(Nt, Nw) * jac
                    M[np.ix_(tdofs, tdofs)] += (Ih + 2 * d_h * Sh + ma * d_h**2) * np.outer(Nt, Nt) * jac
                    M[dofs, ib] += -Sh * Nw * jac
                    M[ib, dofs] += -Sh * Nw * jac
                    M[tdofs, ib] += Ieh * Nt * jac
                    M[ib, tdofs] += Ieh * Nt * jac
                    M[ib, ib] += Ih * jac
        if ai:
            K[ib, ib] += ai["K"]
        free = np.arange(3, n)                                 # root clamped (w, w', theta = 0)
        self.K_full, self.M_full = K, M
        self.free = free
        self.K, self.M = K[np.ix_(free, free)], M[np.ix_(free, free)]

    def modes(self, n_modes=6):
        w2, V = eigh(self.K, self.M)
        w = np.sqrt(np.maximum(w2[:n_modes], 0.0))
        phi = np.zeros((self.ndof, n_modes))
        phi[self.free] = V[:, :n_modes]
        return w, phi

    def kinematics(self, u, s):
        """w, w', theta at stations s from a full dof vector u (or matrix: columns)"""
        u = np.atleast_2d(np.asarray(u).T).T if np.ndim(u) == 1 else np.asarray(u)
        s = np.clip(np.asarray(s, float), 0, self.L)
        e = np.minimum((s / self.L * self.n_el).astype(int), self.n_el - 1)
        L = self.L / self.n_el
        t = (s - self.s[e]) / L
        Nw = np.stack([1 - 3 * t**2 + 2 * t**3, L * (t - 2 * t**2 + t**3), 3 * t**2 - 2 * t**3, L * (-t**2 + t**3)], 1)
        dNw = np.stack([-6 * t + 6 * t**2, L * (1 - 4 * t + 3 * t**2), 6 * t - 6 * t**2, L * (-2 * t + 3 * t**2)], 1) / L
        idx = np.stack([3 * e, 3 * e + 1, 3 * e + 3, 3 * e + 4], 1)
        w = np.einsum('ij,ijk->ik', Nw, u[idx])
        dw = np.einsum('ij,ijk->ik', dNw, u[idx])
        th = (1 - t)[:, None] * u[3 * e + 2] + t[:, None] * u[3 * e + 5]
        beta = u[-1] if self.aileron else np.zeros(u.shape[1])
        return w, dw, th, beta

    def load_vector(self, s_pts, F_up, M_ea):
        """consistent nodal loads of point forces (up) and moments about the e.a.
        (nose up) at stations s_pts"""
        f = np.zeros(self.ndof)
        s_pts = np.clip(np.asarray(s_pts, float), 0, self.L)
        e = np.minimum((s_pts / self.L * self.n_el).astype(int), self.n_el - 1)
        L = self.L / self.n_el
        t = (s_pts - self.s[e]) / L
        Nw = np.stack([1 - 3 * t**2 + 2 * t**3, L * (t - 2 * t**2 + t**3), 3 * t**2 - 2 * t**3, L * (-t**2 + t**3)], 1)
        for j in range(len(s_pts)):
            for a, d in enumerate([3 * e[j], 3 * e[j] + 1, 3 * e[j] + 3, 3 * e[j] + 4]):
                f[d] += Nw[j, a] * F_up[j]
            f[3 * e[j] + 2] += (1 - t[j]) * M_ea[j]
            f[3 * e[j] + 5] += t[j] * M_ea[j]
        return f

    def static(self, f):
        u = np.zeros(self.ndof)
        u[self.free] = np.linalg.solve(self.K, f[self.free])
        return u


def beam_from_structure(ac, st, aileron=None, n_el=24, surface=0, fuel_mass=0.0, fuel_span=(0.1, 0.5)):
    """BeamModel from a wing_structure.size() result. aileron: dict(freq_hz,
    mass_per_area kg/m2, cg_aft_hinge (fraction of the aileron chord, < 0 =
    mass balanced), gyration (fraction of the aileron chord about its c.g.))"""
    s_ = ac.surfaces[surface]
    S = st["settings"]
    y = np.asarray(st["y"]); c = np.asarray(st["chord"])
    xs = st["x_shear_centre"]
    x_ea = np.array([float(s_.x_le(e)) for e in st["eta"]]) + xs * c
    sweep = math.degrees(math.atan2(x_ea[-1] - x_ea[0], y[-1] - y[0]))
    cl = math.cos(math.radians(sweep))
    s_ax = y / cl
    m = np.asarray(st["m_span"]).copy()
    xcg, kg = S["cg_chord"], S["gyration_chord"]
    if fuel_mass > 0:
        on = (st["eta"] >= fuel_span[0]) & (st["eta"] <= fuel_span[1])
        if on.any():
            add = np.zeros_like(m)
            add[on] = 0.5 * fuel_mass / max(np.trapezoid(on * 1.0, y), 1e-9) * (1 if s_.symmetric else 2)
            m = m + add
    ail = next((ct for ct in s_.controls if ct.name == "aileron"), None) if aileron is not None else None
    m_ai = np.zeros_like(m)
    ai_def = None
    if ail is not None:
        cf = ail.cf
        ca = cf * c
        on = (st["eta"] >= ail.eta0) & (st["eta"] <= ail.eta1)
        m_ai = np.where(on, aileron["mass_per_area"] * ca, 0.0)
        m = np.maximum(m - m_ai, 0.0)                              # aileron mass moved to its own dof
        xa = aileron["cg_aft_hinge"]
        ka = aileron["gyration"]
        s0, s1 = ail.eta0 * y[-1] / cl, ail.eta1 * y[-1] / cl
        sm = 0.5 * (s0 + s1)
        cam = float(np.interp(sm * cl, y, ca))
        ma = float(np.interp(sm * cl, y, m_ai))
        Sh = ma * xa * cam
        Ih = ma * cam**2 * (ka**2 + xa**2)
        span_a = s1 - s0
        K = Ih * span_a * (2 * math.pi * aileron["freq_hz"]) ** 2
        d_h = lambda sp: float(np.interp(sp * cl, y, (1 - cf - xs) * c))
        ai_def = {"s0": s0, "s1": s1, "m": ma, "S_h": Sh, "I_h": Ih, "K": K, "d_h": d_h, "cf": cf,
                  "eta0": ail.eta0, "eta1": ail.eta1, "freq_hz": aileron["freq_hz"], **aileron}
    Sun = m * (xcg - xs) * c
    I = m * (kg**2 + (xcg - xs) ** 2) * c**2
    props = {"EI": np.asarray(st["EI"]), "GJ": np.asarray(st["GJ"]), "m": m, "S": Sun, "I": I}
    b = BeamModel(s_ax, props, sweep, n_el, ai_def, x_ea_root=float(x_ea[0]), label=s_.name)
    b.x_ea_fn = lambda yy: np.interp(yy, y, x_ea)
    b.x_le_fn = lambda yy: np.interp(yy, y, x_ea - xs * c)
    b.chord_fn = lambda yy: np.interp(yy, y, c)
    b.ac, b.surface = ac, surface
    return b


# ----------------------------------------------------------------------
# aerodynamics: vortex-ring lattice, frequency domain
# ----------------------------------------------------------------------
class VortexRingLattice:
    """full-span vortex-ring lattice of one lifting surface with a flat wake"""

    def __init__(self, s, ns=14, nc=8, wake_chords=25.0, dx_wake=None, cf=None):
        self.surf = s
        cfs = [c.cf for c in s.controls] if cf is None else [cf]
        if cfs:
            cf = cfs[0]
            n2 = max(2, int(round(nc * cf)))
            n1 = max(2, nc - n2)
            xe = np.r_[np.linspace(0, 1 - cf, n1 + 1), np.linspace(1 - cf, 1, n2 + 1)[1:]]
        else:
            xe = np.linspace(0, 1, nc + 1)
        e = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, ns + 1))     # root/tip clustered
        for v in [x for c in s.controls for x in (c.eta0, c.eta1)] + s.stations()[0][1:-1].tolist():
            if 1e-6 < v < 1 - 1e-6:
                j = int(np.argmin(np.abs(e - v)))
                if abs(e[j] - v) < 0.4 * np.min(np.diff(e)):
                    e[j] = v
                else:
                    e = np.sort(np.r_[e, v])
        eta = np.r_[-e[::-1], e[1:]] if s.symmetric else e
        self.eta_edges = eta
        nsp, ncp = len(eta) - 1, len(xe) - 1
        self.nsp, self.ncp = nsp, ncp

        def pt(et, f):
            cc = float(s.chord(et))
            x = float(s.x_le(et)) + f * cc
            if s.vertical:
                return np.array([x, s.y, s.z + abs(et) * s.semi])
            return np.array([x, s.y + et * s.semi, s.z + float(s.rise(et))])
        # ring chord fractions: quarter chord of each panel; last ring 1/4 panel past the TE
        fr = xe[:-1] + 0.25 * np.diff(xe)
        fr = np.r_[fr, 1.0 + 0.25 * (xe[-1] - xe[-2])]
        G = np.array([[pt(et, f) for f in fr] for et in eta])           # (nsp+1, ncp+1, 3)
        Pc = np.array([[pt(et, f) for f in xe] for et in eta])          # panel corners
        self.G = G
        n = nsp * ncp
        RA, RB, RC, RD = [], [], [], []
        col, nrm, area, fpt, lseg, strip, row, xmid, ymid, etam, flap = ([] for _ in range(11))
        for j in range(nsp):
            em = 0.5 * (eta[j] + eta[j + 1])
            for i in range(ncp):
                p1, p2 = G[j, i], G[j + 1, i]                # leading segment (y-, y+)
                p3, p4 = G[j + 1, i + 1], G[j, i + 1]
                RA.append(p1); RB.append(p2); RC.append(p3); RD.append(p4)
                c0, c1, c2, c3 = Pc[j, i], Pc[j + 1, i], Pc[j + 1, i + 1], Pc[j, i + 1]
                nv = np.cross(c2 - c0, c1 - c3)
                ar = 0.5 * np.linalg.norm(nv)
                nv = nv / np.linalg.norm(nv)
                if nv[2] < 0 and not s.vertical:
                    nv = -nv
                nrm.append(nv); area.append(ar)
                # collocation at the panel's 3/4 chord, load point at its 1/4 chord (span centre)
                q = lambda f: 0.5 * (pt(eta[j], xe[i] + f * (xe[i + 1] - xe[i])) + pt(eta[j + 1], xe[i] + f * (xe[i + 1] - xe[i])))
                col.append(q(0.75)); fpt.append(q(0.25))
                lseg.append(p2 - p1)
                strip.append(j); row.append(i)
                etam.append(em)
                xmid.append(0.5 * (xe[i] + xe[i + 1]))
                flap.append(0.5 * (xe[i] + xe[i + 1]) > 1 - (cfs[0] if cfs else 0) + 1e-9 if cfs else False)
        self.A, self.B, self.C, self.D = (np.array(v) for v in (RA, RB, RC, RD))
        self.col, self.nrm, self.area, self.fpt, self.lseg = (np.array(v) for v in (col, nrm, area, fpt, lseg))
        self.strip, self.row = np.array(strip), np.array(row)
        self.eta_p, self.xfrac = np.array(etam), np.array(xmid)
        self.on_flap = np.array(flap, bool)
        self.n = n
        # bound AIC
        self.AIC = self._ring_influence(self.col, self.A, self.B, self.C, self.D)
        # wake: rows of rings behind the TE ring of every strip
        te = np.array([j * ncp + ncp - 1 for j in range(nsp)])
        self.te = te
        cref = s.mac
        dx = dx_wake or cref / ncp
        R = int(math.ceil(wake_chords * cref / dx))
        self.dx_w, self.R = dx, R
        ex = np.array([1.0, 0, 0])
        W = np.zeros((R, n, nsp))
        a0 = G[:-1, -1]; b0 = G[1:, -1]                               # TE ring trailing segments
        for m in range(R):
            a1, b1 = a0 + m * dx * ex, b0 + m * dx * ex
            W[m] = self._ring_influence(self.col, a1, b1, b1 + dx * ex, a1 + dx * ex)
        self.W = W
        # closure: semi-infinite legs from the end of the wake (steady)
        aE, bE = a0 + R * dx * ex, b0 + R * dx * ex
        self.W_end = (np.einsum('nmk,nk->nm', A._semi(self.col, bE, ex), self.nrm) -
                      np.einsum('nmk,nk->nm', A._semi(self.col, aE, ex), self.nrm))
        self.x_end = R * dx

    def _ring_influence(self, P, a, b, c, d):
        V = A._seg(P, a, b) + A._seg(P, b, c) + A._seg(P, c, d) + A._seg(P, d, a)
        return np.einsum('nmk,nk->nm', V, self.nrm)

    def system(self, kappa):
        """influence matrix for omega/U = kappa (1/m)"""
        ph = np.exp(-1j * kappa * self.dx_w * np.arange(1, self.R + 1))
        Wk = np.tensordot(ph, self.W, axes=(0, 0))                     # (n, nsp)
        # far wake: the oscillating part averages out; steady closure only at kappa -> 0
        fac = np.exp(-1j * kappa * self.x_end) / (1 + 1j * kappa * self.x_end)
        Wk = Wk + fac * self.W_end
        Amat = self.AIC.astype(complex).copy()
        Amat[:, self.te] += Wk
        return Amat

    def solve(self, kappa, wn):
        """ring strengths (per unit U) for normalwash wn (per unit U) on all panels; columns = cases"""
        Am = self.system(kappa)
        return np.linalg.solve(Am, wn)

    def forces(self, kappa, Gam):
        """normal force per unit q on every panel (positive along the panel normal)"""
        Gam = np.atleast_2d(Gam.T).T if Gam.ndim == 1 else Gam
        prev = np.zeros_like(Gam)
        nz = self.row > 0
        prev[nz] = Gam[np.where(nz)[0] - 1]
        ex = np.array([1.0, 0, 0])
        steady = np.cross(ex[None, :], self.lseg)                       # U x l (U = 1)
        fs = np.einsum('nk,nk->n', steady, self.nrm)
        return 2 * ((Gam - prev) * fs[:, None] + 1j * kappa * Gam * self.area[:, None])


# ----------------------------------------------------------------------
# coupling: beam dofs <-> lattice panels
# ----------------------------------------------------------------------
def mode_at_points(beam, s_, P, phi, family="symmetric", on_flap=None, eta=None):
    """z and dz/dx at points P (n, 3) for mode shapes phi (columns, full beam
    dofs); left half mirrored (antisymmetric: sign change). on_flap: points
    on the aileron (rotate with beta)."""
    cl, sl = math.cos(beam.sweep), math.sin(beam.sweep)
    x, y = P[:, 0], P[:, 1] - s_.y
    side = np.sign(y)
    side[side == 0] = 1
    ya = np.abs(y)
    x_ea = beam.x_ea_fn(ya)
    w, dw, th, beta = beam.kinematics(phi, ya / cl)
    d = (x - x_ea) * cl                                                # distance aft of the e.a., normal to it
    z = w - d[:, None] * th
    dzdx = dw * sl - th * cl
    if beam.aileron is not None and on_flap is not None:
        ai = beam.aileron
        et = np.abs(eta) if eta is not None else ya / s_.semi
        on = on_flap & (et >= ai["eta0"] - 1e-9) & (et <= ai["eta1"] + 1e-9)
        c = s_.chord(et)
        xh = np.array([float(s_.x_le(e)) for e in et]) + (1 - ai["cf"]) * c
        z = z - np.where(on, x - xh, 0.0)[:, None] * beta[None, :]
        dzdx = dzdx - np.where(on, 1.0, 0.0)[:, None] * beta[None, :]
    if family == "antisymmetric":
        z = z * side[:, None]
        dzdx = dzdx * side[:, None]
    return z, dzdx


def panel_modes(beam, lat, phi, family="symmetric"):
    """z and dz/dx at the collocation and load points of the lattice"""
    return {key: mode_at_points(beam, lat.surf, P, phi, family, lat.on_flap, lat.eta_p)
            for key, P in (("col", lat.col), ("fpt", lat.fpt))}


def gaf(beam, lat, phi, ks, b_ref, family="symmetric"):
    """generalised aerodynamic forces per unit dynamic pressure, Q(k) (n x n)
    for reduced frequencies k = omega b_ref / U"""
    pm = panel_modes(beam, lat, phi, family)
    zc, dzc = pm["col"]
    zf, _ = pm["fpt"]
    Qs = []
    for k in ks:
        kap = k / b_ref
        wn = 1j * kap * zc + dzc
        G = lat.solve(kap, wn)
        F = lat.forces(kap, G)
        Qs.append(0.5 * zf.T @ F)                              # half wing (the beam matrices are one half)
    return np.array(Qs)


def gaf_strip(beam, phi, ks, b_ref, n_strip=40, a0=2 * math.pi):
    """strip theory (Theodorsen) generalised forces per unit q for the half
    wing (straight elastic axis, no aileron aerodynamics): the classical
    reference model (Goland). Lift up, moment about the e.a. nose up."""
    from aerolab.aircraft import wing_structure as WS
    s_ = beam.ac.surfaces[beam.surface] if hasattr(beam, "ac") else None
    L = beam.L
    yy = np.linspace(0, L, n_strip + 1)
    ym = 0.5 * (yy[1:] + yy[:-1]); dy = np.diff(yy)
    c = beam.chord_fn(ym)
    b = c / 2
    a = (beam.x_ea_fn(ym) - beam.x_le_fn(ym)) / b - 1
    w, dw, th, _ = beam.kinematics(phi, ym)
    f = a0 / (2 * math.pi)
    out = []
    for k in ks:
        om = k / b_ref                                         # omega / U, U = 1, rho = 2 (per unit q)
        C = np.array([WS.theodorsen(max(om * bb, 1e-9)) for bb in b])
        rho = 2.0
        cols = []
        for j in range(phi.shape[1]):
            h = -w[:, j]
            al = th[:, j]
            hd, hdd = 1j * om * h, -(om**2) * h
            ad, add = 1j * om * al, -(om**2) * al
            wq = al + hd + b * (0.5 - a) * ad                  # three-quarter-chord downwash term
            Lf = math.pi * rho * b**2 * (hdd + ad - b * a * add) + 2 * math.pi * rho * b * C * wq * f
            Mf = math.pi * rho * b**2 * (b * a * hdd - b * (0.5 - a) * ad - b**2 * (1 / 8 + a**2) * add) \
                + 2 * math.pi * rho * b**2 * (a + 0.5) * C * wq * f
            cols.append((Lf, Mf))
        Q = np.zeros((phi.shape[1], phi.shape[1]), complex)
        for i in range(phi.shape[1]):
            for j in range(phi.shape[1]):
                Lf, Mf = cols[j]
                Q[i, j] = np.sum((w[:, i] * Lf + th[:, i] * Mf) * dy)
        out.append(Q)
    return np.array(out)


# ----------------------------------------------------------------------
# flutter: p-k
# ----------------------------------------------------------------------
class GAFTable:
    def __init__(self, ks, Q):
        self.ks = np.asarray(ks, float)
        self.QR = Q.real
        self.QIk = np.array([Q[i].imag / max(k, 1e-6) for i, k in enumerate(ks)])
        if self.ks[0] < 1e-9:                                          # k = 0: Q_I/k from the next point
            self.QIk[0] = self.QIk[1]

    def at(self, k):
        k = float(np.clip(k, self.ks[0], self.ks[-1]))
        i = int(np.clip(np.searchsorted(self.ks, k) - 1, 0, len(self.ks) - 2))
        t = (k - self.ks[i]) / (self.ks[i + 1] - self.ks[i])
        return (1 - t) * self.QR[i] + t * self.QR[i + 1], (1 - t) * self.QIk[i] + t * self.QIk[i + 1]


def pk(Mm, Km, table, Vs, rho, b_ref, damping=0.0, tol=1e-5, it_max=60):
    """p-k flutter solution. Returns per mode: V, g, f (Hz) arrays and the
    flutter points (lowest V where g crosses 0 upward)."""
    n = Mm.shape[0]
    w0 = np.sqrt(np.maximum(np.diag(Km) / np.diag(Mm), 1e-12))
    order = np.argsort(w0)
    res = {"V": np.asarray(Vs, float), "g": np.full((len(Vs), n), np.nan), "f": np.full((len(Vs), n), np.nan),
           "k": np.full((len(Vs), n), np.nan)}
    Minv = np.linalg.inv(Mm)
    prev = {j: complex(-0.0, w0[j]) for j in range(n)}
    for iv, V in enumerate(Vs):
        q = 0.5 * rho * V**2
        for jj, j in enumerate(order):
            p_guess = prev[j]
            for _ in range(it_max):
                k = max(abs(p_guess.imag) * b_ref / max(V, 1e-6), 1e-4)
                QR, QIk = table.at(k)
                Kt = Km * (1 + 1j * damping) - q * QR
                Ct = -0.5 * rho * V * b_ref * QIk
                S = np.block([[np.zeros((n, n)), np.eye(n)], [-Minv @ Kt, -Minv @ Ct]])
                ev = np.linalg.eigvals(S)
                ev = ev[ev.imag > 0] if np.any(ev.imag > 0) else ev
                p_new = ev[np.argmin(np.abs(ev - p_guess))]
                if abs(p_new - p_guess) < tol * max(abs(p_guess), 1.0):
                    p_guess = p_new
                    break
                p_guess = p_new
            prev[j] = p_guess
            res["g"][iv, j] = 2 * p_guess.real / max(abs(p_guess.imag), 1e-9)
            res["f"][iv, j] = abs(p_guess.imag) / (2 * math.pi)
            res["k"][iv, j] = abs(p_guess.imag) * b_ref / max(V, 1e-6)
    pts = []
    for j in range(n):
        g = res["g"][:, j]
        idx = np.where((g[:-1] < 0) & (g[1:] >= 0))[0]
        for i in idx:
            t = -g[i] / (g[i + 1] - g[i])
            pts.append({"mode": int(j), "V": float(Vs[i] + t * (Vs[i + 1] - Vs[i])),
                        "f_hz": float(res["f"][i, j] + t * (res["f"][i + 1, j] - res["f"][i, j]))})
    pts.sort(key=lambda p: p["V"])
    res["flutter"] = pts
    return res


# ----------------------------------------------------------------------
# static aeroelasticity
# ----------------------------------------------------------------------
def static_matrices(beam, lat, family="symmetric"):
    """aerodynamic stiffness Q0 (per unit q, beam dofs) and the rigid load
    vectors per unit q for alpha = 1 rad and aileron = 1 rad, plus the
    lift / rolling-moment rows (per unit q) of the panels"""
    n = beam.ndof
    phi = np.eye(n)
    pm = panel_modes(beam, lat, phi, family)
    zc, dzc = pm["col"]
    zf, _ = pm["fpt"]
    G = lat.solve(0.0, dzc.astype(complex))
    F = lat.forces(0.0, G).real                                        # (panels, dofs)
    Q0 = 0.5 * zf.T @ F                                                # half wing
    # rigid alpha: dz/dx = -1 everywhere (nose up), symmetric
    wa = -np.ones((lat.n, 1)) if family == "symmetric" else -np.sign(lat.col[:, 1] - lat.surf.y)[:, None]
    Fa = lat.forces(0.0, lat.solve(0.0, wa.astype(complex))).real
    fa = 0.5 * zf.T @ Fa
    y = lat.fpt[:, 1] - lat.surf.y
    return {"Q0": Q0, "F": F, "fa": fa[:, 0], "Fa": Fa[:, 0], "y": y, "zf": zf}


def divergence_q(beam, lat):
    Sm = static_matrices(beam, lat, "symmetric")
    K = beam.K_full[np.ix_(beam.free, beam.free)]
    Q = Sm["Q0"][np.ix_(beam.free, beam.free)]
    # K u = q Q u  ->  eigenvalues 1/q of K^-1 Q
    lam = np.linalg.eigvals(np.linalg.solve(K, Q))
    lam = lam[(np.abs(lam.imag) < 1e-6 * np.abs(lam).max()) & (lam.real > 0)].real
    return float(1 / lam.max()) if len(lam) else math.inf


def static_response(beam, lat, qs, family="symmetric"):
    """elastic / rigid ratio of the lift due to alpha (symmetric) or of the
    rolling moment due to the aileron (antisymmetric, needs the aileron dof)
    against dynamic pressure"""
    Sm = static_matrices(beam, lat, family)
    fr = beam.free
    K = beam.K_full[np.ix_(fr, fr)]
    if family == "symmetric":
        Q = Sm["Q0"][np.ix_(fr, fr)]
        lift_row = (Sm["F"] * lat.nrm[:, 2][:, None]).sum(0)[fr]       # lift per unit q per dof
        L0 = float((Sm["Fa"] * lat.nrm[:, 2]).sum())
        out = []
        for q in qs:
            u = np.linalg.solve(K - q * Q, q * Sm["fa"][fr])
            out.append((L0 + lift_row @ u) / L0)
        return np.array(out)
    ib = beam.ndof - 1
    rest = fr[fr != ib]
    Q = Sm["Q0"]
    Kr = beam.K_full[np.ix_(rest, rest)]
    roll_row = (Sm["F"] * (lat.nrm[:, 2] * Sm["y"])[:, None]).sum(0)
    R0 = roll_row[ib]
    out = []
    for q in qs:
        # aileron held at 1 rad (stick fixed): wing dofs respond
        u = np.linalg.solve(Kr - q * Q[np.ix_(rest, rest)], q * Q[rest, ib])
        out.append((R0 + roll_row[rest] @ u) / R0)
    return np.array(out)


def reversal_q(beam, lat, q_max):
    qs = np.linspace(0, q_max, 120)[1:]
    ef = static_response(beam, lat, qs, "antisymmetric")
    k = np.where(np.diff(np.sign(ef)) != 0)[0]
    if not len(k):
        return math.inf, qs, ef
    from scipy.optimize import brentq
    f = lambda q: static_response(beam, lat, [q], "antisymmetric")[0]
    return float(brentq(f, qs[k[0]], qs[k[0] + 1])), qs, ef


# ----------------------------------------------------------------------
# flexible loads on the aircraft model
# ----------------------------------------------------------------------
def strip_incidence(beam, an, u):
    """incidence change (rad) of every lattice strip of the wing from the beam
    displacement u (symmetric: both halves)"""
    L = an.lat
    s_ = an.ac.surfaces[beam.surface]
    m = L.S["surf"] == beam.surface
    d = np.zeros(L.nstrip)
    ya = np.abs(L.S["mid"][m, 1] - s_.y)
    w, dw, th, _ = beam.kinematics(u, ya / math.cos(beam.sweep))
    d[m] = th[:, 0] * math.cos(beam.sweep) - dw[:, 0] * math.sin(beam.sweep)
    return d


def strip_loads_to_beam(beam, an, q2):
    """beam load vector from the last an.solve (right half strips, N and N m)"""
    L = an.lat
    s_ = an.ac.surfaces[beam.surface]
    st = np.where((L.S["surf"] == beam.surface) & (L.S["side"] >= 0))[0]
    Fp = an._last["Fp"] * q2
    cm = an._last["cm"]
    F_up, M_ea, s_pts = [], [], []
    cl = math.cos(beam.sweep)
    for i in st:
        pan = np.where(L.pstrip == i)[0]
        F = Fp[pan].sum(0)
        ya = L.S["mid"][i, 1] - s_.y
        xe = beam.x_ea_fn(ya)
        E = np.array([xe, L.S["mid"][i, 1], L.S["mid"][i, 2]])
        Mv = np.cross(L.P_mid[pan] - E, Fp[pan]).sum(0) + 0.5 * q2 * L.S["c"][i] * L.S["area"][i] * cm[i] * L.S["s"][i]
        F_up.append(F[2])
        M_ea.append(Mv[1] * cl + Mv[0] * math.sin(beam.sweep))          # nose up about the e.a. (pitch: +My)
        s_pts.append(ya / cl)
    return beam.load_vector(np.array(s_pts), np.array(F_up), np.array(M_ea))


def flexible_trim(an, beam, CL_req, control="elevator", n=1.0, it_max=12, tol=1e-5, inertia=None):
    """trim the aircraft with the wing's elastic incidence included (fixed
    point): solve -> wing loads -> beam -> strip incidence -> solve ...
    inertia: beam load vector of the wing's own weight at n g (relief)."""
    base = an.theta_base.copy()
    d = np.zeros_like(base)
    q2 = an.rho * an.V**2
    hist = []
    r = None
    for it in range(it_max):
        an.theta_base = base + d
        r = an.trim(CL_req=CL_req, control=control) if control else an.solve(alpha=math.degrees(CL_req / 5.0))
        r = an.solve(alpha=r.get("alpha_trim", r["alpha"]), controls={control: r["control_trim"]} if control else None)
        f = strip_loads_to_beam(beam, an, q2)
        if inertia is not None:
            f = f + inertia
        u = beam.static(f)
        d_new = strip_incidence(beam, an, u)
        ch = float(np.abs(d_new - d).max())
        hist.append(ch)
        d = 0.5 * d + 0.5 * d_new if it > 4 else d_new
        if ch < tol:
            break
    an.theta_base = base + d
    t = an.trim(CL_req=CL_req, control=control) if control else r
    r = an.solve(alpha=t.get("alpha_trim", t["alpha"]), controls={control: t["control_trim"]} if control else None)
    for k in ("alpha_trim", "control_trim", "trim_control", "CL_req", "L_D"):
        if k in t:
            r[k] = t[k]
    r["elastic_incidence"] = d
    r["beam_u"] = u
    r["elastic_iterations"] = len(hist)
    r["elastic_converged"] = hist[-1] < 10 * tol if hist else True
    an.theta_base_rigid = base
    return r


def inertia_load(beam, n):
    """beam load vector of the wing's own mass at load factor n (down)"""
    s = beam.s
    m = beam.p("m", s)
    S = beam.p("S", s)
    ds = np.gradient(s)
    F = -n * 9.80665 * m * ds
    M = n * 9.80665 * S * ds                                          # c.g. aft of the e.a.: nose-down weight -> -(-S) ...
    return beam.load_vector(s, F, -M)


# ----------------------------------------------------------------------
def analyse(ac, st, aileron=None, Vmax=None, rho=1.225, n_modes=6, ns=12, nc=8, ks=None, log=print,
            fuel_mass=0.0, fuel_span=(0.1, 0.5), V_D=None):
    beam = beam_from_structure(ac, st, aileron, fuel_mass=fuel_mass, fuel_span=fuel_span)
    w, phi = beam.modes(n_modes)
    s_ = ac.surfaces[0]
    lat = VortexRingLattice(s_, ns=ns, nc=nc)
    b_ref = 0.5 * s_.mac
    ks = np.r_[0.0, 0.02, 0.05, 0.08, 0.12, 0.17, 0.25, 0.35, 0.5, 0.7, 1.0, 1.5, 2.2] if ks is None else ks
    Vmax = Vmax or (2.0 * V_D if V_D else 150.0)
    Vs = np.linspace(0.05 * Vmax, Vmax, 80)
    Mm, Km = phi.T @ beam.M_full @ phi, phi.T @ beam.K_full @ phi
    out = {"frequencies_hz": (w / (2 * math.pi)).tolist(), "b_ref": b_ref, "rho": rho, "V_D": V_D}
    shapes = []
    for j in range(n_modes):
        ww, dw, th, be = beam.kinematics(phi[:, j], beam.s)
        sc = max(np.abs(ww).max(), 1e-12)
        shapes.append({"s": beam.s.tolist(), "w": (ww[:, 0] / sc).tolist(), "theta": (th[:, 0] / sc).tolist(),
                       "beta": float(be[0] / sc) if beam.aileron else 0.0, "kind": _mode_kind(ww[:, 0], th[:, 0], be, beam)})
    out["modes"] = shapes
    # mode shapes on the planform grid (panel corners; the aileron's own corners move with beta)
    eta_e = lat.eta_edges
    xe = np.linspace(0, 1, 2 * lat.ncp + 1)
    Pg = np.array([[[float(s_.x_le(e)) + f * float(s_.chord(e)), s_.y + e * s_.semi, s_.z + float(s_.rise(e))]
                    for f in xe] for e in eta_e])
    flat = Pg.reshape(-1, 3)
    cf = beam.aileron["cf"] if beam.aileron else 1.0
    on = np.tile(xe > 1 - cf + 1e-9, len(eta_e))
    et = np.repeat(eta_e, len(xe))
    grid = {"x": Pg[..., 0].tolist(), "y": Pg[..., 1].tolist(), "z": Pg[..., 2].tolist()}
    for fam in ("symmetric", "antisymmetric"):
        zg, _ = mode_at_points(beam, s_, flat, phi, fam, on, et)
        zg = zg / np.maximum(np.abs(zg).max(0), 1e-12)
        grid[fam] = [zg[:, j].reshape(Pg.shape[:2]).tolist() for j in range(phi.shape[1])]
    out["mode_grid"] = grid
    for fam in ("symmetric", "antisymmetric"):
        Q = gaf(beam, lat, phi, ks, b_ref, fam)
        tab = GAFTable(ks, Q)
        r = pk(Mm, Km, tab, Vs, rho, b_ref)
        out[fam] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
        log(f"[aeroelastic] {fam}: flutter " +
            (f"{r['flutter'][0]['V']:.1f} m/s at {r['flutter'][0]['f_hz']:.2f} Hz" if r["flutter"] else "none"))
    q_d = divergence_q(beam, lat)
    out["divergence"] = {"q": q_d, "V_eas": math.sqrt(2 * q_d / 1.225) if math.isfinite(q_d) else math.inf}
    qmax = 0.5 * 1.225 * Vmax**2
    qs = np.linspace(0, min(qmax, 0.95 * q_d) if math.isfinite(q_d) else qmax, 40)[1:]
    out["elastic_lift"] = {"V_eas": np.sqrt(2 * qs / 1.225).tolist(),
                           "ratio": static_response(beam, lat, qs, "symmetric").tolist()}
    if beam.aileron:
        q_r, qq, ef = reversal_q(beam, lat, qs[-1])
        out["reversal"] = {"q": q_r, "V_eas": math.sqrt(2 * q_r / 1.225) if math.isfinite(q_r) else math.inf,
                           "V_curve": np.sqrt(2 * qq / 1.225).tolist(), "effectiveness": ef.tolist()}
    out["aileron"] = {k: v for k, v in (beam.aileron or {}).items() if not callable(v)}
    out["sweep_ea_deg"] = math.degrees(beam.sweep)
    fl = [p | {"family": f} for f in ("symmetric", "antisymmetric") for p in out[f]["flutter"]]
    out["flutter_speed"] = min((p["V"] for p in fl), default=math.inf)
    out["flutter_points"] = sorted(fl, key=lambda p: p["V"])
    return out, beam, lat


def mass_balance_sweep(ac, st, aileron, values, Vmax, rho=1.225, log=None, **kw):
    """flutter speed against the aileron c.g. position (fraction of the aileron
    chord aft of the hinge; negative = overbalanced)"""
    out = []
    for v in values:
        r, _, _ = analyse(ac, st, dict(aileron) | {"cg_aft_hinge": float(v)}, Vmax=Vmax, rho=rho, log=lambda *a: None, **kw)
        out.append({"cg_aft_hinge": float(v), "V_flutter": r["flutter_speed"],
                    "family": r["flutter_points"][0]["family"] if r["flutter_points"] else None})
        if log:
            log(f"[mass balance] c.g. {v:+.2f} c_a: flutter {r['flutter_speed']:.1f} m/s")
    return out


def strip_flutter(ac, st, Vmax, rho=1.225, n_modes=6, fuel_mass=0.0, fuel_span=(0.1, 0.5)):
    """the same structure with strip-theory (Theodorsen) aerodynamics, no aileron dof"""
    beam = beam_from_structure(ac, st, None, fuel_mass=fuel_mass, fuel_span=fuel_span)
    w, phi = beam.modes(n_modes)
    b_ref = 0.5 * ac.surfaces[0].mac
    ks = np.r_[0.0, 0.02, 0.05, 0.08, 0.12, 0.17, 0.25, 0.35, 0.5, 0.7, 1.0, 1.5, 2.2]
    Mm, Km = phi.T @ beam.M_full @ phi, phi.T @ beam.K_full @ phi
    Q = gaf_strip(beam, phi, ks, b_ref)
    r = pk(Mm, Km, GAFTable(ks, Q), np.linspace(0.05 * Vmax, Vmax, 80), rho, b_ref)
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}


def _mode_kind(w, th, be, beam):
    """label from the largest displacement: bending (w), torsion (theta x half
    chord), aileron (beta x quarter chord); number = nodes + 1"""
    c = beam.ac.surfaces[beam.surface].mac
    rw, rt = np.abs(w).max(), np.abs(th).max() * c / 2
    rb = abs(float(be[0])) * c * 0.25 if beam.aileron else 0.0
    if rb > 1.5 * max(rw, rt):
        if rt > 0.2 * rb and rt >= rw:
            return f"aileron + torsion {int(np.sum(np.diff(np.sign(th[1:])) != 0)) + 1}"
        return "aileron"
    if rw >= rt:
        lab = f"bending {int(np.sum(np.diff(np.sign(w[1:])) != 0)) + 1}"
    else:
        lab = f"torsion {int(np.sum(np.diff(np.sign(th[1:])) != 0)) + 1}"
    return lab + (" + aileron" if rb > 0.7 * max(rw, rt) else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aircraft")
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    ac = A.Aircraft.load(a.aircraft)
    if not getattr(ac, "structure", None):
        raise SystemExit("no wing structure saved with this aircraft: run the Wing structure page / wing_structure.py first")
    st = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in ac.structure.items()}
    res, _, _ = analyse(ac, st, ac.structure.get("aileron_dynamics"), a.vmax)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "aeroelastic.json").write_text(json.dumps(A._jsonable(res), indent=1))


if __name__ == "__main__":
    main()
