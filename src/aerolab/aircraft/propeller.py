"""
Propeller analysis and design (blade element momentum theory) and engine
models for the performance tool.

    python src/run.py propeller design --power 75 --rpm 2500 --speed 60 --diameter 1.8 --blades 2 --out ../cases/prop
    python src/run.py propeller analyse ../cases/prop/propeller.json --speed 0:80:5 --rpm 2500

Method
  * Analysis: blade element momentum theory with Prandtl tip and hub
    losses, solved per radial station for the inflow angle phi with Ning's
    single-equation residual (AIAA J. 2014 / Wind Energy 2014) written in a
    form that also holds at zero forward speed:
        R(phi) = sin(phi) (1 - k) - lambda_r cos(phi) (1 + k'),
        k = sigma' c_n / (4 F sin^2 phi),  k' = sigma' c_t / (4 F sin phi cos phi),
    bracketed and solved with Brent's method (always converges where a
    propeller solution exists). Section data: a generic propeller section
    (Clark-Y-like: alpha0 -3.5 deg, lift slope 0.95 x 2 pi, c_l,max 1.25,
    Prandtl-Glauert on the local Mach number, drag rise above M 0.75) or any
    CFD section polar of the polar library. Outputs C_T, C_P, efficiency,
    thrust and power against advance ratio J = V / (n D).
  * Design: minimum induced loss propeller (Adkins & Liebeck, J. Aircraft
    1994): chord and pitch distribution for a given power or thrust, speed,
    rpm, diameter, blade count and design lift coefficient; the design is
    checked by re-analysing it with the BEMT above.
  * Operation: fixed pitch (the rpm settles where the engine power equals
    the absorbed power) or constant speed (pitch adjusted at the governed
    rpm). Part-throttle efficiency at the thrust needed for level flight is
    used for range and endurance.
  * Engines: piston (Gagg-Ferrar altitude lapse, power proportional to rpm
    at full throttle, BSFC), turboprop (power ~ sigma^0.7, PSFC), electric
    (no altitude lapse, motor and controller efficiency, battery energy).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from aerolab.aircraft import wing as W


# ----------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------
class GenericSection:
    """Clark-Y-like propeller section; Reynolds and Mach corrections"""

    def __init__(self, alpha0=-3.5, slope=0.95 * 2 * math.pi, clmax=1.25, clmin=-0.6, cd0=0.0085,
                 k=0.010, cl_cdmin=0.35, t_c=0.10):
        self.alpha0, self.slope, self.clmax, self.clmin = alpha0, slope, clmax, clmin
        self.cd0, self.k, self.cl_cdmin, self.t_c = cd0, k, cl_cdmin, t_c
        al = np.linspace(-12, 16, 113)
        cl = self.slope * np.radians(al - alpha0)
        a_s = alpha0 + math.degrees(clmax / slope)
        a_n = alpha0 + math.degrees(clmin / slope)
        cl = np.where(al > a_s, clmax - 0.06 * (al - a_s), cl)
        cl = np.where(al < a_n, clmin + 0.04 * (al - a_n), cl)
        cd = cd0 + k * (cl - cl_cdmin) ** 2 + np.where(al > a_s, 0.02 * (al - a_s), 0.0) + \
            np.where(al < a_n, 0.02 * (a_n - al), 0.0)
        self.polar = W.SectionPolar(al, cl, cd, np.zeros_like(al), ar=6.0, label="generic propeller section")
        self.a_stall = a_s

    def __call__(self, alpha_deg, re=1e6, mach=0.0):
        cl, cd, _ = self.polar(np.asarray(alpha_deg, float))
        re = np.maximum(np.asarray(re, float), 2e4)
        cd = cd * (np.minimum(re, 3e6) / 1e6) ** -0.2
        m = np.minimum(np.asarray(mach, float), 0.9)
        cl = cl / np.sqrt(1 - np.minimum(m, 0.7) ** 2)
        mcrit = 0.75 - 0.5 * (self.t_c - 0.06)
        cd = cd + 20 * np.maximum(m - mcrit, 0.0) ** 4
        return cl, cd


class PolarSection:
    """a CFD section polar (W.SectionPolar) as a propeller section"""

    def __init__(self, polar, t_c=0.12):
        self.polar, self.t_c = polar, t_c
        self.a_stall = polar.a_clmax
        self.alpha0 = math.degrees(polar.alpha0)

    def __call__(self, alpha_deg, re=1e6, mach=0.0):
        cl, cd, _ = self.polar(np.asarray(alpha_deg, float))
        m = np.minimum(np.asarray(mach, float), 0.9)
        cl = cl / np.sqrt(1 - np.minimum(m, 0.7) ** 2)
        cd = cd + 20 * np.maximum(m - (0.75 - 0.5 * (self.t_c - 0.06)), 0.0) ** 4
        return cl, cd


def section_from_spec(spec=None, polar_dir=None):
    if not spec or spec == "generic":
        return GenericSection()
    if polar_dir:
        from aerolab.design import design_wing_sizing as WS
        got = WS.cached_polars(spec, polar_dir, 0.0)
        if got:
            return PolarSection(got[len(got) // 2][1])
    return GenericSection()


# ----------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------
class Propeller:
    """blade geometry: radii r (m), chord c (m), pitch angle beta (deg, from
    the plane of rotation) at the stations; hub radius; blade count"""

    def __init__(self, diameter, blades, r, chord, beta, hub=None, section=None, name="propeller"):
        self.D, self.R, self.B = float(diameter), 0.5 * float(diameter), int(blades)
        self.r = np.asarray(r, float)
        self.c = np.asarray(chord, float)
        self.beta = np.asarray(beta, float)
        self.hub = float(hub if hub is not None else 0.12 * self.R)
        self.section = section or GenericSection()
        self.name = name

    def to_dict(self):
        return {"name": self.name, "diameter": self.D, "blades": self.B, "hub": self.hub,
                "r": self.r.tolist(), "chord": self.c.tolist(), "beta": self.beta.tolist()}

    @classmethod
    def from_dict(cls, d, section=None):
        return cls(d["diameter"], d["blades"], d["r"], d["chord"], d["beta"], d.get("hub"), section,
                   d.get("name", "propeller"))

    def beta_75(self):
        return float(np.interp(0.75 * self.R, self.r, self.beta))

    def activity_factor(self):
        x = self.r / self.R
        return float(1e5 / 16 * np.trapezoid(self.c / self.D * x**3, x))

    @classmethod
    def simple(cls, diameter, blades=2, pitch_ratio=0.8, chord_ratio=0.08, n=25):
        """generic fixed-pitch blade: constant geometric pitch P/D, elliptic-ish chord"""
        R = 0.5 * diameter
        x = np.linspace(0.15, 1.0, n)
        r = x * R
        beta = np.degrees(np.arctan(pitch_ratio / (math.pi * x)))
        c = chord_ratio * diameter * np.clip(1.2 * np.sqrt(np.maximum(1 - x**2, 0.0)) + 0.35 * (1 - x), 0.25, 1.4)
        return cls(diameter, blades, r, c, beta, hub=0.15 * R)


# ----------------------------------------------------------------------
# BEMT analysis
# ----------------------------------------------------------------------
def analyse(prop, V, rpm, rho=1.225, mu=1.79e-5, a_snd=340.3, dbeta=0.0, n_st=40):
    """thrust, power, torque, efficiency at speed V (m/s) and rpm; dbeta =
    pitch change of the whole blade (constant-speed propellers). All radial
    stations are solved together: bracket on a phi grid, then bisection."""
    n = rpm / 60.0
    Om = 2 * math.pi * n
    R, B = prop.R, prop.B
    x = np.linspace(prop.hub / R, 1.0, n_st + 1)
    r = 0.5 * (x[1:] + x[:-1]) * R
    dr = np.diff(x) * R
    c = np.interp(r, prop.r, prop.c)
    beta = np.radians(np.interp(r, prop.r, prop.beta) + dbeta)
    sig = B * c / (2 * math.pi * r)
    lam = V / max(Om * r.min(), 1e-9) * (r.min() / r)           # V / (Omega r)

    def tiploss(phi):
        s = np.maximum(np.abs(np.sin(phi)), 1e-6)
        ft = B / 2 * (R - r) / (r * s)
        fh = B / 2 * (r - prop.hub) / (prop.hub * s) if prop.hub > 0 else 50.0
        F = (2 / math.pi * np.arccos(np.minimum(np.exp(-ft), 1.0))) * \
            (2 / math.pi * np.arccos(np.minimum(np.exp(-fh), 1.0)))
        return np.maximum(F, 1e-4)

    def coeffs(phi, Wr=None):
        a = np.degrees(beta - phi)
        if Wr is None:
            Wr = Om * r / np.maximum(np.cos(phi), 1e-6)
        cl, cd = prop.section(a, rho * Wr * c / mu, Wr / a_snd)
        sp, cp = np.sin(phi), np.cos(phi)
        return cl * cp - cd * sp, cl * sp + cd * cp

    def resid(phi):
        cn, ct = coeffs(phi)
        F = tiploss(phi)
        sp, cp = np.sin(phi), np.cos(phi)
        k = sig * cn / (4 * F * sp * sp)
        kp = sig * ct / (4 * F * sp * cp)
        return sp * (1 - k) - lam * cp * (1 + kp)

    lo_, hi_ = 1e-4, 0.5 * math.pi - 1e-4
    grid = np.linspace(lo_, hi_, 61)
    Rg = np.array([resid(np.full_like(r, p)) for p in grid])           # (61, n)
    sc = (Rg[:-1] * Rg[1:]) <= 0
    has = sc.any(0)
    j = np.argmax(sc, 0)                                               # first root from small phi
    a_ = grid[j]
    b_ = grid[np.minimum(j + 1, len(grid) - 1)]
    fa = Rg[j, np.arange(len(r))]
    for _ in range(45):
        m = 0.5 * (a_ + b_)
        fm = resid(m)
        left = fa * fm <= 0
        b_ = np.where(left, m, b_)
        a_ = np.where(left, a_, m)
        fa = np.where(left, fa, fm)
    phi = 0.5 * (a_ + b_)
    phi = np.where(has, phi, np.arctan2(V, Om * r) if V > 0 else lo_)  # no propeller solution: unloaded
    F = tiploss(phi)
    sp, cp = np.sin(phi), np.cos(phi)
    cn, ct = coeffs(phi)
    kp = sig * ct / (4 * F * sp * cp)
    Wr = Om * r * (1 - kp / (1 + kp)) / np.maximum(cp, 1e-6)
    cn, ct = coeffs(phi, Wr)                                           # with the converged speed (Re, M)
    q = 0.5 * rho * Wr**2 * c * B
    dT = q * cn * dr
    dQ = q * ct * r * dr
    alpha = np.degrees(beta - phi)
    stalled = alpha > getattr(prop.section, "a_stall", 14.0)
    T = float(dT.sum())
    Q = float(dQ.sum())
    P = Q * Om
    J = V / (n * prop.D) if n > 0 else 0.0
    CT = T / (rho * n**2 * prop.D**4) if n > 0 else 0.0
    CP = P / (rho * n**3 * prop.D**5) if n > 0 else 0.0
    eta = J * CT / CP if CP > 1e-9 and T > 0 else 0.0
    return {"V": V, "rpm": rpm, "J": J, "T": T, "Q": Q, "P": P, "CT": CT, "CP": CP, "eta": eta,
            "dbeta": dbeta, "tip_mach": math.hypot(Om * R, V) / a_snd,
            "stations": {"r": r.tolist(), "dT_dr": (dT / dr).tolist(), "dQ_dr": (dQ / dr).tolist(),
                         "alpha": alpha.tolist(), "phi": np.degrees(phi).tolist(),
                         "stalled": stalled.tolist()},
            "stall_fraction": float(stalled.mean())}


def sweep_J(prop, J, rpm, rho=1.225, mu=1.79e-5, a_snd=340.3, dbeta=0.0):
    n = rpm / 60
    return [analyse(prop, j * n * prop.D, rpm, rho, mu, a_snd, dbeta) for j in J]


# ----------------------------------------------------------------------
# Adkins & Liebeck minimum induced loss design
# ----------------------------------------------------------------------
def design(V, rpm, diameter, blades, power=None, thrust=None, cl_design=0.7, hub_ratio=0.15,
           section=None, rho=1.225, mu=1.79e-5, a_snd=340.3, n=30, tol=1e-7, max_it=100):
    """chord and pitch for minimum induced loss (Adkins & Liebeck 1994).
    Give power (W) or thrust (N)."""
    if (power is None) == (thrust is None):
        raise ValueError("give power or thrust")
    section = section or GenericSection()
    R = 0.5 * diameter
    Om = 2 * math.pi * rpm / 60
    lam = V / (Om * R)
    xi0 = hub_ratio
    xi = xi0 + (1 - xi0) * (0.5 - 0.5 * np.cos(np.linspace(0, math.pi, n)))      # cosine: tip clustered
    xi = np.clip(xi, xi0, 0.9999)
    # design angle of attack and drag/lift ratio at the design cl (iterated with Re)
    al_grid = np.linspace(-6, 14, 401)
    zeta = 0.0
    eps = np.full(n, 0.012)
    a_des = np.zeros(n)
    W_ = np.full(n, V)
    for it in range(max_it):
        phi_t = math.atan(lam * (1 + zeta / 2))
        f = blades / 2 * (1 - xi) / math.sin(phi_t)
        F = 2 / math.pi * np.arccos(np.minimum(np.exp(-f), 1.0))
        phi = np.arctan(math.tan(phi_t) / xi)
        x = Om * xi * R / V
        G = F * x * np.cos(phi) * np.sin(phi)
        Wc = 4 * math.pi * lam * G * V * R * zeta / (cl_design * blades)
        a = zeta / 2 * np.cos(phi) ** 2 * (1 - eps * np.tan(phi))
        ap = zeta / (2 * x) * np.cos(phi) * np.sin(phi) * (1 + eps / np.tan(phi))
        W_ = V * (1 + a) / np.sin(phi)
        c = Wc / np.maximum(W_, 1e-9)
        # section: alpha and eps at cl_design, local Re and Mach
        for i in range(n):
            re = rho * W_[i] * max(c[i], 1e-3) / mu
            cl_g, cd_g = section(al_grid, re, W_[i] / a_snd)
            j = int(np.argmax(cl_g >= cl_design)) if np.any(cl_g >= cl_design) else int(np.argmax(cl_g))
            a_des[i] = al_grid[j]
            eps[i] = float(cd_g[j] / max(cl_g[j], 1e-6))
        I1p = 4 * xi * G * (1 - eps * np.tan(phi))
        I2p = lam * (I1p / (2 * xi)) * (1 + eps / np.tan(phi)) * np.sin(phi) * np.cos(phi)
        J1p = 4 * xi * G * (1 + eps / np.tan(phi))
        J2p = (J1p / 2) * (1 - eps * np.tan(phi)) * np.cos(phi) ** 2
        I1, I2, J1, J2 = (float(np.trapezoid(v, xi)) for v in (I1p, I2p, J1p, J2p))
        if power is not None:
            Pc = 2 * power / (rho * V**3 * math.pi * R**2)
            z_new = -J1 / (2 * J2) + math.sqrt((J1 / (2 * J2)) ** 2 + Pc / J2)
            Tc = I1 * z_new - I2 * z_new**2
        else:
            Tc = 2 * thrust / (rho * V**2 * math.pi * R**2)
            disc = (I1 / (2 * I2)) ** 2 - Tc / I2
            if disc < 0:
                raise ValueError("thrust too high for this diameter and speed")
            z_new = I1 / (2 * I2) - math.sqrt(disc)
            Pc = J1 * z_new + J2 * z_new**2
        if abs(z_new - zeta) < tol:
            zeta = z_new
            break
        zeta = z_new
    beta = np.degrees(phi) + a_des
    c = np.maximum(c, 0.02 * diameter * (1 - xi) + 1e-3)       # finite tip chord for the analysis
    T = Tc * 0.5 * rho * V**2 * math.pi * R**2
    P = Pc * 0.5 * rho * V**3 * math.pi * R**2
    prop = Propeller(diameter, blades, xi * R, c, beta, hub=xi0 * R, section=section,
                     name=f"MIL design {blades} blades, D {diameter:g} m")
    return prop, {"zeta": zeta, "T": T, "P": P, "eta": T * V / P, "iterations": it + 1,
                  "cl_design": cl_design, "J": V / (rpm / 60 * diameter),
                  "eta_ideal": ideal_efficiency(T, V, diameter, rho)}


def ideal_efficiency(T, V, D, rho=1.225):
    """actuator disc (Froude) efficiency: upper bound"""
    Tc = T / (0.5 * rho * V**2 * math.pi * D**2 / 4)
    return 2 / (1 + math.sqrt(1 + Tc))


# ----------------------------------------------------------------------
# engines + propulsion for the performance tool
# ----------------------------------------------------------------------
class Engine:
    """kind: piston | turboprop | electric. power_kw at sea level (rated)."""

    def __init__(self, kind="piston", power_kw=75.0, rpm_max=2700.0, sfc=300.0, eta_motor=0.92,
                 battery_kwh=0.0, usable_fraction=0.8, gear_ratio=1.0):
        self.kind, self.P0, self.rpm_max, self.sfc = kind, power_kw * 1e3, rpm_max, sfc
        self.eta_motor, self.battery_kwh, self.usable = eta_motor, battery_kwh, usable_fraction
        self.gear_ratio = gear_ratio

    def power(self, h, rpm=None):
        """shaft power available (W) at altitude h and propeller rpm"""
        sig = W.isa(h)[0] / 1.225
        if self.kind == "piston":
            lapse = max(sig - (1 - sig) / 7.55, 0.0)
            frac = 1.0 if rpm is None else min(max(rpm / self.rpm_max, 0.0), 1.0)   # torque ~ constant
        elif self.kind == "turboprop":
            lapse, frac = sig**0.7, 1.0
        else:
            lapse, frac = 1.0, 1.0 if rpm is None else min(max(rpm / self.rpm_max, 0.0), 1.0)
        return self.P0 * lapse * frac

    def fuel_flow(self, shaft_power):
        """kg/s (0 for electric)"""
        return 0.0 if self.kind == "electric" else self.sfc / 3.6e9 * shaft_power

    def to_dict(self):
        return {"kind": self.kind, "power_kw": self.P0 / 1e3, "rpm_max": self.rpm_max, "sfc": self.sfc,
                "eta_motor": self.eta_motor, "battery_kwh": self.battery_kwh, "usable_fraction": self.usable}


def _first_crossing(f, lo, hi, n=16, xtol=0.02):
    """smallest x in [lo, hi] with f(x) = 0 (f rising through zero); hi or lo if none"""
    xs = np.linspace(lo, hi, n + 1)
    fv = [f(x) for x in xs]
    for i in range(n):
        if fv[i] <= 0 <= fv[i + 1]:
            return brentq(f, xs[i], xs[i + 1], xtol=xtol) if fv[i] != fv[i + 1] else xs[i]
    return hi if fv[-1] < 0 else lo


class BEMTPropulsion:
    """engine + BEMT propeller with the interface performance.run uses:
    power(h), thrust(V, h), efficiency(V), eta_at_thrust(V, T, h), bsfc,
    fuel_flow(h), kind"""

    def __init__(self, engine, prop, mode="constant_speed", rpm=None, pitch_range=(-15.0, 25.0)):
        self.engine, self.prop, self.mode = engine, prop, mode
        self.rpm = rpm or engine.rpm_max
        self.pitch_range = pitch_range
        self.kind = engine.kind
        self.bsfc = engine.sfc if engine.kind != "electric" else None
        self.D = prop.D
        self.P0 = engine.P0
        self.eta = 0.0
        self._cache = {}

    def _atm(self, h):
        rho, mu, a, _, _ = W.isa(h)
        return rho, mu, a

    def power(self, h):
        return self.engine.power(h, self.rpm)

    def _full_power_point(self, V, h):
        """operating point at full throttle: (thrust, eta, rpm, dbeta)"""
        key = (round(V, 2), round(h, 0))
        if key in self._cache:
            return self._cache[key]
        rho, mu, a = self._atm(h)
        if self.mode == "fixed_pitch":
            def f(rpm):
                return analyse(self.prop, V, rpm, rho, mu, a)["P"] - self.engine.power(h, min(rpm, self.rpm))
            lo, hi = 0.2 * self.rpm, 1.6 * self.rpm
            try:
                rpm = brentq(f, lo, hi, xtol=0.5) if f(lo) * f(hi) < 0 else (hi if f(hi) < 0 else lo)
            except ValueError:
                rpm = self.rpm
            rpm = min(rpm, 1.15 * self.rpm)              # overspeed limit
            r = analyse(self.prop, V, rpm, rho, mu, a)
            out = (r["T"], r["eta"], rpm, 0.0)
        else:
            Pav = self.engine.power(h, self.rpm)

            def f(db):
                return analyse(self.prop, V, self.rpm, rho, mu, a, dbeta=db)["P"] - Pav
            db = _first_crossing(f, *self.pitch_range)       # finest pitch that absorbs the power
            r = analyse(self.prop, V, self.rpm, rho, mu, a, dbeta=db)
            out = (r["T"], r["eta"], self.rpm, db)
        self._cache[key] = out
        return out

    H_STEP = 500.0

    def _table(self, hk):
        """full-throttle thrust, efficiency, rpm and pitch change on a speed
        grid at the table altitude hk"""
        key = ("table", hk)
        if key not in self._cache:
            v_top = 1.5 * self.rpm / 60 * self.prop.D
            Vg = np.r_[0.0, np.linspace(3.0, v_top, 13)]
            pts = np.array([self._full_power_point(v, hk) for v in Vg], float)
            self._cache[key] = (Vg, pts)
        return self._cache[key]

    def _interp(self, V, h, col):
        """table value at speed V, linear between the two table altitudes"""
        h0 = math.floor(max(h, 0.0) / self.H_STEP) * self.H_STEP
        w = (max(h, 0.0) - h0) / self.H_STEP
        out = 0.0
        for hk, wk in ((h0, 1 - w), (h0 + self.H_STEP, w)):
            if wk > 1e-9:
                Vg, pts = self._table(hk)
                out += wk * float(np.interp(V, Vg, pts[:, col]))
        return out

    def thrust(self, V, h):
        return max(self._interp(V, h, 0), 0.0)

    def efficiency(self, V, h=0.0):
        return self._interp(V, h, 1)

    def operating_point(self, V, h=0.0):
        """full throttle: thrust, efficiency, rpm and pitch change at speed V"""
        return {"T": self.thrust(V, h), "eta": self.efficiency(V, h), "rpm": self._interp(V, h, 2),
                "dbeta": self._interp(V, h, 3)}

    def eta_at_thrust(self, V, T, h=0.0):
        """propeller efficiency when it delivers thrust T (part throttle)"""
        rho, mu, a = self._atm(h)
        if T <= 0 or V <= 0:
            return 0.0
        if self.mode == "fixed_pitch":
            def f(rpm):
                return analyse(self.prop, V, rpm, rho, mu, a)["T"] - T
            lo, hi = 0.2 * self.rpm, 1.15 * self.rpm
            if f(lo) * f(hi) > 0:
                return self.efficiency(V, h)
            rpm = brentq(f, lo, hi, xtol=0.5)
            return analyse(self.prop, V, rpm, rho, mu, a)["eta"]
        rpm = 0.85 * self.rpm                              # cruise rpm of a constant-speed propeller

        def g(db):
            return analyse(self.prop, V, rpm, rho, mu, a, dbeta=db)["T"] - T
        if g(self.pitch_range[0]) > 0:
            return self.efficiency(V, h)
        db = _first_crossing(g, *self.pitch_range)
        return analyse(self.prop, V, rpm, rho, mu, a, dbeta=db)["eta"]

    def fuel_flow(self, h):
        return self.engine.fuel_flow(self.power(h))

    def describe(self):
        return {"model": "BEMT", "mode": self.mode, "engine": self.engine.to_dict(),
                "propeller": self.prop.to_dict(), "rpm": self.rpm,
                "activity_factor": self.prop.activity_factor(), "beta_75": self.prop.beta_75()}


def make_propulsion(spec, polar_dir=None):
    """propulsion from a JSON-able dict (app / CLI):
    {"engine": {...Engine kwargs}, "propeller": {Propeller dict} | {"design": {...}},
     "mode": "constant_speed" | "fixed_pitch", "section": "generic" | airfoil}"""
    eng = Engine(**spec.get("engine", {}))
    sec = section_from_spec(spec.get("section"), polar_dir)
    pd_ = spec.get("propeller", {})
    if "design" in pd_:
        d = pd_["design"]
        rho, mu, a, _, _ = W.isa(d.get("altitude", 0.0))
        prop, _ = design(d["speed"], d.get("rpm", eng.rpm_max), d["diameter"], d.get("blades", 2),
                         power=d.get("power_kw", eng.P0 / 1e3 * 0.75) * 1e3, cl_design=d.get("cl", 0.7),
                         section=sec, rho=rho, mu=mu, a_snd=a)
    elif "r" in pd_:
        prop = Propeller.from_dict(pd_, sec)
    else:
        prop = Propeller.simple(pd_.get("diameter", 1.8), pd_.get("blades", 2), pd_.get("pitch_ratio", 0.8))
        prop.section = sec
    return BEMTPropulsion(eng, prop, spec.get("mode", "constant_speed"), spec.get("rpm"))


# ----------------------------------------------------------------------
def plots(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    st = res["design_check"]["stations"] if "design_check" in res else None
    g = res["geometry"]
    r = np.array(g["r"]) / (0.5 * g["diameter"])
    axs[0].plot(r, np.array(g["chord"]) / g["diameter"], label="c / D")
    ax2 = axs[0].twinx()
    ax2.plot(r, g["beta"], color="#C2571A", label="β (deg)")
    axs[0].set_xlabel("r / R"); axs[0].set_ylabel("c / D"); ax2.set_ylabel("β (deg)")
    if "sweep" in res:
        J = [p["J"] for p in res["sweep"]]
        axs[1].plot(J, [p["CT"] for p in res["sweep"]], label="C_T")
        axs[1].plot(J, [p["CP"] for p in res["sweep"]], label="C_P")
        axs[1].set_xlabel("J"); axs[1].legend()
        axs[2].plot(J, [p["eta"] for p in res["sweep"]])
        axs[2].set_xlabel("J"); axs[2].set_ylabel("η")
    if st:
        axs[1].set_title("C_T, C_P")
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out) / "propeller.png", dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design")
    d.add_argument("--power", type=float, required=True, help="shaft power at the design point (kW)")
    d.add_argument("--rpm", type=float, required=True)
    d.add_argument("--speed", type=float, required=True)
    d.add_argument("--diameter", type=float, required=True)
    d.add_argument("--blades", type=int, default=2)
    d.add_argument("--cl", type=float, default=0.7)
    d.add_argument("--alt", type=float, default=0.0)
    d.add_argument("--out", required=True)
    a = sub.add_parser("analyse")
    a.add_argument("propeller")
    a.add_argument("--rpm", type=float, required=True)
    a.add_argument("--speed", default="0:80:5")
    a.add_argument("--alt", type=float, default=0.0)
    args = ap.parse_args(argv)
    if args.cmd == "design":
        rho, mu, snd, _, _ = W.isa(args.alt)
        prop, info = design(args.speed, args.rpm, args.diameter, args.blades, power=args.power * 1e3,
                            cl_design=args.cl, rho=rho, mu=mu, a_snd=snd)
        chk = analyse(prop, args.speed, args.rpm, rho, mu, snd)
        n = args.rpm / 60
        sweep = sweep_J(prop, np.linspace(0.05, 1.3 * info["J"], 25), args.rpm, rho, mu, snd)
        res = {"design": info, "geometry": prop.to_dict(), "design_check": chk, "sweep": sweep}
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "propeller.json").write_text(json.dumps(res, indent=1, default=float))
        plots(res, out)
        print(f"design: T {info['T']:.1f} N, P {info['P'] / 1e3:.2f} kW, eta {info['eta']:.3f} "
              f"(ideal {info['eta_ideal']:.3f}); BEMT check: T {chk['T']:.1f} N, P {chk['P'] / 1e3:.2f} kW, "
              f"eta {chk['eta']:.3f}; beta75 {prop.beta_75():.1f} deg, AF {prop.activity_factor():.0f}, "
              f"n {n:.1f} rev/s")
    else:
        d = json.loads(Path(args.propeller).read_text())
        prop = Propeller.from_dict(d.get("geometry", d))
        rho, mu, snd, _, _ = W.isa(args.alt)
        a0, a1, da = (float(v) for v in args.speed.split(":"))
        for V in np.arange(a0, a1 + 1e-9, da):
            r = analyse(prop, V, args.rpm, rho, mu, snd)
            print(f"V {V:5.1f}  J {r['J']:.3f}  T {r['T']:8.1f} N  P {r['P'] / 1e3:7.2f} kW  eta {r['eta']:.3f}")


if __name__ == "__main__":
    main()
