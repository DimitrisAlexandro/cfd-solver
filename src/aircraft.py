"""
Whole-aircraft aerodynamics, stability and control.

    python aircraft.py demo                       # write the demo aircraft JSON
    python aircraft.py analyse ../cases/aircraft/demo_trainer.json --speed 50 --alt 1000

Model
  * Lifting surfaces (wing, horizontal tail, fin, canard...) as trapezoidal
    panels: span, root/tip chord, quarter-chord sweep, dihedral, twist,
    incidence, root leading-edge position; symmetric (mirrored) or single
    (fin). Control surfaces: span range, chord fraction, symmetric
    (elevator, flap) or antisymmetric (aileron) deflection.
  * One vortex lattice for all surfaces together (horseshoe vortices,
    nc chordwise x ns spanwise panels, hinge lines and control edges on
    panel edges), so wing downwash on the tail, sidewash on the fin and all
    interference are part of the solution. Free stream with angle of attack,
    sideslip and body rotation rates (p, q, r): the dynamic derivatives come
    from the same lattice. Forces by Kutta-Joukowski on every bound segment
    with the full local velocity (induced drag and side force included).
    Compressibility: Prandtl-Glauert (Goethert stretch).
  * Camber through the sections' zero-lift angle; section pitching moment,
    profile drag and (optionally) the nonlinear lift up to and past stall
    from the CFD section polars, as in the wing tool (decambering,
    Mukherjee & Gopalarathnam 2006) - now on every surface.
  * Fuselage: body of revolution; Munk's slender-body moments (pitch and
    yaw, both destabilising, apparent-mass factor k2 - k1 of the equivalent
    prolate spheroid) and turbulent skin-friction drag with Raymer's form
    factor. Wing-body lift carry-over and vertical-wing-position effects on
    C_l_beta are not modelled.

Conventions (as AVL / Etkin): forces CL (up), CD (along the wind), CY
(right); moments about the reference point (default: the centre of
gravity) in STABILITY axes: Cl roll (right wing down +), Cm pitch (nose up
+), Cn yaw (nose right +). Rates p, q, r non-dimensional: p b/2V, q c/2V,
r b/2V. Controls: elevator + = trailing edge down; aileron + = right
aileron trailing edge up (roll right); rudder + = trailing edge left (nose
left, Cn_dr < 0). Geometry axes: x aft, y right, z up, origin at the nose.
"""

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import wing as W

G0 = 9.80665
DEFAULT_GAINS = {"elevator": 1.0, "flap": 1.0, "aileron": -1.0, "rudder": -1.0}


# ----------------------------------------------------------------------
# definition
# ----------------------------------------------------------------------
@dataclass
class Control:
    name: str
    eta0: float = 0.0           # span range, fraction of the semi-span (0 root, 1 tip)
    eta1: float = 1.0
    cf: float = 0.25            # chord fraction aft of the hinge
    mode: str = "symmetric"     # "symmetric" or "antisymmetric" (aileron)
    gain: float | None = None   # incidence change per deflection; default by name


@dataclass
class Surface:
    name: str
    span: float                 # tip to tip (symmetric) or root to tip (single, e.g. fin)
    root_chord: float
    tip_chord: float
    x: float = 0.0              # root leading edge
    y: float = 0.0
    z: float = 0.0
    sweep: float = 0.0          # quarter-chord sweep (deg)
    dihedral: float = 0.0
    twist: float = 0.0          # tip incidence relative to the root (deg, - = washout)
    incidence: float = 0.0
    airfoil: str = "naca0012"
    tip_airfoil: str | None = None
    symmetric: bool = True
    vertical: bool = False
    controls: list = field(default_factory=list)
    ns: int = 12                # spanwise panels per side
    nc: int = 6                 # chordwise panels
    # kink stations between root and tip (cranked / multi-panel surfaces), dicts:
    #   y      distance from the root along the semi-span (m)
    #   chord  local chord (m)
    #   dx     leading-edge x relative to the root leading edge (m)
    #   dz     height relative to the root (m; horizontal surfaces)
    #   twist  incidence relative to the root (deg)
    # The root -> tip definition (tip chord, sweep, dihedral, twist) sets the tip station.
    kinks: list = field(default_factory=list)

    @property
    def semi(self):
        return 0.5 * self.span if self.symmetric else self.span

    def stations(self):
        """(eta, chord, dx_le, dz, twist) at root, kinks and tip; eta = fraction of the semi-span"""
        sm = self.semi
        tl, td = math.tan(math.radians(self.sweep)), math.tan(math.radians(self.dihedral))
        rows = [(0.0, self.root_chord, 0.0, 0.0, 0.0)]
        for k in sorted(self.kinks, key=lambda k: float(k["y"])):
            y = float(k["y"])
            if 1e-9 < y < sm * (1 - 1e-6):
                rows.append((y / sm, float(k["chord"]), float(k.get("dx", 0.0)),
                             0.0 if self.vertical else float(k.get("dz", 0.0)), float(k.get("twist", 0.0))))
        rows.append((1.0, self.tip_chord, 0.25 * self.root_chord + sm * tl - 0.25 * self.tip_chord,
                     0.0 if self.vertical else sm * td, self.twist))
        return np.array(rows, float).T

    def _interp(self, i, eta):
        st = self.stations()
        return np.interp(np.abs(eta), st[0], st[i])

    def chord(self, eta):
        return self._interp(1, eta) if self.kinks else \
            self.root_chord + (self.tip_chord - self.root_chord) * np.abs(eta)

    def x_le(self, eta):
        """leading-edge x at eta"""
        return self.x + self._interp(2, eta)

    def rise(self, eta):
        """height of the leading edge above the root (dihedral), horizontal surfaces"""
        return self._interp(3, eta)

    def twist_at(self, eta):
        """incidence relative to the root (deg)"""
        return self._interp(4, eta)

    def _moments(self):
        """exact integrals over the semi-span (Simpson per linear panel): area,
        int c^2, int c*eta, int c*x_qc (per unit semi-span)"""
        st = self.stations()
        I = np.zeros(4)
        for a, b in zip(range(len(st[0]) - 1), range(1, len(st[0]))):
            e = np.array([st[0, a], 0.5 * (st[0, a] + st[0, b]), st[0, b]])
            c = self.chord(e)
            xq = self.x_le(e) + 0.25 * c
            w = (e[2] - e[0]) / 6 * np.array([1, 4, 1])
            I += [w @ c, w @ c**2, w @ (c * e), w @ (c * xq)]
        return I

    @property
    def area(self):
        return self._moments()[0] * self.semi * (2 if self.symmetric else 1)

    @property
    def mac(self):
        m = self._moments()
        return m[1] / m[0]

    def mac_position(self):
        """leading edge (x) and spanwise station (eta) of the mean aerodynamic chord;
        x from the chord-weighted quarter-chord line"""
        m = self._moments()
        return m[3] / m[0] - 0.25 * m[1] / m[0], m[2] / m[0]


@dataclass
class Fuselage:
    length: float = 0.0
    diameter: float = 0.0
    x: float = 0.0              # nose
    z: float = 0.0              # axis height
    x_max: float = 0.3          # station of the maximum diameter, fraction of the length
    tail_ratio: float = 0.15    # end diameter / maximum diameter

    def radius(self, xs):
        """r(x) of the body of revolution: elliptic nose, tapered tail cone"""
        L, R = self.length, 0.5 * self.diameter
        xi = (np.asarray(xs, float) - self.x) / max(L, 1e-9)
        r = np.zeros_like(xi)
        m = self.x_max
        nose = (xi >= 0) & (xi <= m)
        r[nose] = R * np.sqrt(np.clip(1 - ((m - xi[nose]) / m) ** 2, 0, 1))
        tail = (xi > m) & (xi <= 1)
        t = (xi[tail] - m) / (1 - m)
        r[tail] = R * (1 - (1 - self.tail_ratio) * t**1.6)
        return r

    def volume_and_wetted(self):
        if self.length <= 0 or self.diameter <= 0:
            return 0.0, 0.0
        xs = self.x + np.linspace(0, self.length, 400)
        r = self.radius(xs)
        dr = np.gradient(r, xs)
        return (float(np.trapezoid(math.pi * r**2, xs)),
                float(np.trapezoid(2 * math.pi * r * np.sqrt(1 + dr**2), xs)))

    def k2_minus_k1(self):
        """apparent-mass factor of the equivalent prolate spheroid (Lamb)"""
        f = max(self.length / max(self.diameter, 1e-9), 1.01)
        e = math.sqrt(1 - 1 / f**2)
        lg = math.log((1 + e) / (1 - e))
        a0 = 2 * (1 - e**2) / e**3 * (0.5 * lg - e)
        b0 = 1 / e**2 - (1 - e**2) / (2 * e**3) * lg
        return b0 / (2 - b0) - a0 / (2 - a0)


@dataclass
class Aircraft:
    name: str
    surfaces: list
    fuselage: Fuselage = field(default_factory=Fuselage)
    mass: float = 750.0         # kg
    cg: tuple = (0.0, 0.0, 0.0)
    inertia: tuple | None = None  # Ixx, Iyy, Izz (kg m^2); None = estimated
    S_ref: float | None = None  # default: first surface (the wing)
    b_ref: float | None = None
    c_ref: float | None = None
    cd_extra: float = 0.0       # extra drag area / S_ref (gear, cooling, ...)
    # design data (sizing / weight and balance / propulsion pages); optional
    components: list = field(default_factory=list)   # {name, group, mass, x, z, max}
    gear: dict = field(default_factory=dict)         # x_main, x_nose, track, z_ground
    propulsion: dict = field(default_factory=dict)   # propeller.make_propulsion spec

    # reference values default to the main wing (first surface)
    @property
    def wing(self):
        return self.surfaces[0]

    @property
    def S(self):
        return self.S_ref or self.wing.area

    @property
    def b(self):
        return self.b_ref or self.wing.span

    @property
    def c(self):
        return self.c_ref or self.wing.mac

    def inertias(self):
        """Ixx, Iyy, Izz: given, or Roskam's radii of gyration for single-engine
        propeller aircraft (R_x 0.248, R_y 0.338, R_z 0.393)"""
        if self.inertia:
            return tuple(float(v) for v in self.inertia)
        L = self.fuselage.length if self.fuselage.length > 0 else 0.75 * self.b
        m, b = self.mass, self.b
        return (m * (b * 0.248 / 2) ** 2, m * (L * 0.338 / 2) ** 2,
                m * ((b + L) / 2 * 0.393 / 2) ** 2)

    # serialisation
    def to_dict(self):
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d):
        d = copy.deepcopy(d)
        surfs = []
        for s in d.pop("surfaces"):
            ctrls = [Control(**c) for c in s.pop("controls", [])]
            surfs.append(Surface(**s, controls=ctrls))
        fus = Fuselage(**d.pop("fuselage", {}))
        d["cg"] = tuple(d.get("cg", (0, 0, 0)))
        if d.get("inertia"):
            d["inertia"] = tuple(d["inertia"])
        return cls(surfaces=surfs, fuselage=fus, **d)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def controls(self):
        """{control name: (surface index, Control)}"""
        return {c.name: (i, c) for i, s in enumerate(self.surfaces) for c in s.controls}


def demo_aircraft():
    """generic two-seat trainer, 750 kg (not a real type)"""
    wing = Surface("wing", span=10.4, root_chord=1.45, tip_chord=0.95, x=1.95, z=-0.35,
                   sweep=0.0, dihedral=4.0, twist=-2.0, incidence=1.5, airfoil="naca2412",
                   controls=[Control("aileron", 0.58, 0.95, 0.25, "antisymmetric"),
                             Control("flap", 0.12, 0.56, 0.25, "symmetric")], ns=14, nc=6)
    htail = Surface("horizontal tail", span=3.4, root_chord=0.85, tip_chord=0.6, x=6.25, z=0.15,
                    sweep=5.0, incidence=-1.5, airfoil="naca0012",
                    controls=[Control("elevator", 0.05, 1.0, 0.4, "symmetric")], ns=8, nc=5)
    fin = Surface("fin", span=1.35, root_chord=1.15, tip_chord=0.6, x=6.0, z=0.3, sweep=28.0,
                  airfoil="naca0012", symmetric=False, vertical=True,
                  controls=[Control("rudder", 0.08, 0.95, 0.4, "symmetric")], ns=8, nc=5)
    fus = Fuselage(length=7.3, diameter=1.15, x=0.0, z=0.0, x_max=0.3, tail_ratio=0.15)
    return Aircraft("Demo trainer (generic, 750 kg)", [wing, htail, fin], fus, mass=750.0,
                    cg=(2.30, 0.0, -0.1), cd_extra=0.012)


def surface_quads(s):
    """planform quads for drawing: list of (kind, 4x3 points); kind is
    'surface' or the control name. Mirrored for symmetric surfaces."""
    etas = sorted({0.0, 1.0} | {c.eta0 for c in s.controls} | {c.eta1 for c in s.controls}
                  | set(s.stations()[0].tolist()))

    def pt(eta, f, side):
        c = float(s.chord(eta))
        x = float(s.x_le(eta)) + f * c
        if s.vertical:
            return np.array([x, s.y, s.z + eta * s.semi])
        return np.array([x, s.y + side * eta * s.semi, s.z + float(s.rise(eta))])
    out = []
    for side in ((1, -1) if s.symmetric else (1,)):
        for e0, e1 in zip(etas[:-1], etas[1:]):
            em = 0.5 * (e0 + e1)
            ctl = next((c for c in s.controls if c.eta0 - 1e-9 <= em <= c.eta1 + 1e-9), None)
            fh = 1 - ctl.cf if ctl else 1.0
            out.append(("surface", np.array([pt(e0, 0, side), pt(e1, 0, side), pt(e1, fh, side), pt(e0, fh, side)])))
            if ctl:
                out.append((ctl.name, np.array([pt(e0, fh, side), pt(e1, fh, side), pt(e1, 1, side),
                                                pt(e0, 1, side)])))
    return out


# ----------------------------------------------------------------------
# section data
# ----------------------------------------------------------------------
def camber_properties(spec):
    """thin-airfoil zero-lift angle (rad), Cm_c/4 and thickness of a section"""
    import airfoil_geom as ag
    n = ag.sharp_outline(spec)
    i = int(np.argmin(n[:, 0]))
    up, lo = n[:i + 1][::-1], n[i:]
    th = np.linspace(1e-4, math.pi - 1e-4, 400)
    x = 0.5 * (1 - np.cos(th))
    yu = np.interp(x, up[:, 0], up[:, 1])
    yl = np.interp(x, lo[:, 0], lo[:, 1])
    zc = 0.5 * (yu + yl)
    dz = np.gradient(zc, x)
    a0 = -float(np.trapezoid(dz * (np.cos(th) - 1), th)) / math.pi
    A1 = 2 / math.pi * float(np.trapezoid(dz * np.cos(th), th))
    A2 = 2 / math.pi * float(np.trapezoid(dz * np.cos(2 * th), th))
    return a0, math.pi / 4 * (A2 - A1), float((yu - yl).max())


def estimated_polar(spec, re, ar=8.0):
    """section polar without CFD: thin-airfoil lift and moment of the camber
    line, Cl_max estimate 1.2 + 6 x camber-angle, turbulent flat-plate friction x
    form factor (1 + 2 t + 60 t^4). Marked 'estimated'."""
    a0, cm0, t = camber_properties(spec)
    cf = 0.455 / math.log10(max(re, 1e4)) ** 2.58
    cd0 = 2 * cf * (1 + 2 * t + 60 * t**4)
    clmax = 1.2 + 6.0 * abs(a0)
    al = np.linspace(-16, 18, 69)
    cl = 0.95 * 2 * math.pi * (np.radians(al) - a0)
    cl = np.clip(cl, -clmax + 0.4 * abs(a0) * 6, clmax)
    cd = cd0 + 0.012 * cl**2
    pol = W.SectionPolar(al, cl, cd, np.full_like(al, cm0), ar=ar, label=f"{Path(str(spec)).stem} (estimated)")
    return pol


def surface_polars(s, polar_dir, mach, re_ref, ar=8.0):
    """W.SpanPolars of a surface: cached CFD polars of its airfoil(s) when
    available (any Reynolds number), else the thin-airfoil estimate."""
    import design_wing_sizing as WS

    def one(af):
        got = WS.cached_polars(af, polar_dir, mach) if polar_dir else []
        if got:
            return got, "CFD"
        return [(re_ref, estimated_polar(af, re_ref, ar))], "estimated"
    root, src = one(s.airfoil)
    tip, src_t = (one(s.tip_airfoil) if s.tip_airfoil and s.tip_airfoil != s.airfoil
                  else (None, src))
    return W.SpanPolars(root, tip), (src if src == src_t else f"{src} / {src_t}")


# ----------------------------------------------------------------------
# vortex lattice for the whole aircraft
# ----------------------------------------------------------------------
def _seg(P, A, B, rc2=None):
    """velocity at P (n,3) of unit segments A->B (m,3); optional core radius^2 (n,m)"""
    r1 = P[:, None, :] - A[None]
    r2 = P[:, None, :] - B[None]
    r0 = (B - A)[None]
    cr = np.cross(r1, r2)
    c2 = (cr**2).sum(-1)
    l2 = np.maximum((r0**2).sum(-1), 1e-30)
    n1 = np.maximum(np.linalg.norm(r1, axis=-1), 1e-12)
    n2 = np.maximum(np.linalg.norm(r2, axis=-1), 1e-12)
    k = np.einsum('nmk,nmk->nm', r0, r1 / n1[..., None] - r2 / n2[..., None])
    h2 = c2 / l2                                     # distance^2 from the line
    if rc2 is None:
        f = np.where(h2 > 1e-14 * l2, k / (4 * math.pi * np.maximum(c2, 1e-300)), 0.0)
    else:
        f = k / (4 * math.pi * l2 * np.maximum(h2 + rc2, 1e-300)) * np.where(h2 + rc2 > 1e-14 * l2, 1, 0)
    return cr * f[..., None]


def _semi(P, A, u, rc2=None):
    """velocity at P of unit semi-infinite filaments from A along u"""
    r = P[:, None, :] - A[None]
    nr = np.linalg.norm(r, axis=-1)
    d = nr - np.einsum('nmk,k->nm', r, u)            # nr (1 - cos)
    cr = np.cross(u[None, None, :], r)
    h2 = (cr**2).sum(-1)
    if rc2 is None:
        f = np.where(nr * d > 1e-12, 1.0 / (4 * math.pi * np.maximum(nr * d, 1e-300)), 0.0)
    else:
        # (1 + cos)/(4 pi h) * h^2/(h^2 + rc^2) / h, with 1 + cos = (nr + r.u)/nr
        cosp = 2 - d / np.maximum(nr, 1e-300)
        f = cosp / (4 * math.pi * np.maximum(h2 + rc2, 1e-300))
    return cr * f[..., None]


class Lattice:
    """panels of all surfaces; influence matrices for one Mach number"""

    def __init__(self, ac, mach=0.0):
        self.ac, self.mach = ac, mach
        self.beta = math.sqrt(max(1 - mach**2, 0.04))
        A, B, C, sid, strip, frac = [], [], [], [], [], []
        S = {k: [] for k in ("c", "ds", "area", "eta", "mid", "s", "n0", "t", "surf", "side",
                             "le_mid", "re_c")}
        self.ctrl_panels = {}        # name -> (panel indices, signed gain per panel)
        self.ctrl_hinge = {}         # name -> per-panel hinge x
        np_ = 0
        ns_ = 0
        for k, s in enumerate(ac.surfaces):
            # chordwise division: hinge on a panel edge
            cfs = [c.cf for c in s.controls]
            if cfs:
                cf = cfs[0]
                n2 = max(2, int(round(s.nc * cf)))
                n1 = max(1, s.nc - n2)
                xe = np.r_[np.linspace(0, 1 - cf, n1 + 1), np.linspace(1 - cf, 1, n2 + 1)[1:]]
            else:
                xe = np.linspace(0, 1, s.nc + 1)
            # spanwise stations per side: clustered at the tip, control edges inserted
            e = np.sin(0.5 * math.pi * np.linspace(0, 1, s.ns + 1))
            edges = [v for c in s.controls for v in (c.eta0, c.eta1)] + s.stations()[0][1:-1].tolist()
            for v in edges:                     # control edges and kinks on panel edges
                if 1e-6 < v < 1 - 1e-6:
                    j = int(np.argmin(np.abs(e - v)))
                    if abs(e[j] - v) < 0.35 * np.min(np.diff(e)):
                        e[j] = v
                    else:
                        e = np.sort(np.r_[e, v])
            eta = np.r_[-e[::-1], e[1:]] if s.symmetric else e
            span = s.semi * eta
            cn = s.chord(eta)
            xq = s.x_le(eta) + 0.25 * cn
            if s.vertical:
                qc = np.c_[xq, np.full_like(span, s.y), s.z + span]
            else:
                qc = np.c_[xq, s.y + span, s.z + s.rise(eta)]
            le = qc - np.c_[0.25 * cn, 0 * cn, 0 * cn]
            for i in range(len(eta) - 1):
                em = 0.5 * (eta[i] + eta[i + 1])
                cm_ = s.chord(em)
                sv = qc[i + 1] - qc[i]
                ds = float(np.linalg.norm(sv))
                sh = sv / ds
                ch = np.array([1.0, 0.0, 0.0])
                n0 = np.cross(ch, sh)
                n0 /= np.linalg.norm(n0)
                th = np.cross(sh, n0)
                lem = 0.5 * (le[i] + le[i + 1])
                for j in range(len(xe) - 1):
                    fb = xe[j] + 0.25 * (xe[j + 1] - xe[j])
                    fc = xe[j] + 0.75 * (xe[j + 1] - xe[j])
                    A.append(le[i] + fb * np.array([cn[i], 0, 0]))
                    B.append(le[i + 1] + fb * np.array([cn[i + 1], 0, 0]))
                    C.append(lem + fc * np.array([cm_, 0, 0]))
                    sid.append(k)
                    strip.append(ns_)
                    frac.append(0.5 * (xe[j] + xe[j + 1]))
                side = 0 if not s.symmetric else (1 if em > 0 else -1)
                for c in s.controls:
                    if c.eta0 - 1e-9 <= abs(em) <= c.eta1 + 1e-9:
                        g = c.gain if c.gain is not None else DEFAULT_GAINS.get(c.name, 1.0)
                        sign = (1.0 if side >= 0 else -1.0) if c.mode == "antisymmetric" else 1.0
                        idx, gains = self.ctrl_panels.setdefault(c.name, ([], []))
                        hx = self.ctrl_hinge.setdefault(c.name, [])
                        for j in range(len(xe) - 1):
                            if 0.5 * (xe[j] + xe[j + 1]) > 1 - c.cf - 1e-9:
                                idx.append(np_ + j)
                                gains.append(g * sign)
                                hx.append(lem[0] + (1 - c.cf) * cm_)
                np_ += len(xe) - 1
                S["c"].append(cm_); S["ds"].append(ds); S["area"].append(cm_ * ds)
                S["eta"].append(abs(em)); S["mid"].append(0.5 * (qc[i] + qc[i + 1]))
                S["s"].append(sh); S["n0"].append(n0); S["t"].append(th); S["surf"].append(k)
                S["side"].append(side); S["le_mid"].append(lem)
                ns_ += 1
        self.A, self.B, self.C = (np.array(v, float) for v in (A, B, C))
        self.P_mid = 0.5 * (self.A + self.B)
        self.psurf = np.array(sid)
        self.pstrip = np.array(strip)
        self.pfrac = np.array(frac)
        for k, v in S.items():
            S[k] = np.array(v)
        self.S = S
        self.nstrip = ns_
        self.npan = np_
        self.ctrl_panels = {k: (np.array(i), np.array(g)) for k, (i, g) in self.ctrl_panels.items()}
        self.ctrl_hinge = {k: np.array(v) for k, v in self.ctrl_hinge.items()}
        # influence (Goethert stretch for compressibility)
        st = np.array([1.0 / self.beta, 1.0, 1.0])
        As, Bs, Cs, Ms = self.A * st, self.B * st, self.C * st, self.P_mid * st
        u = np.array([1.0, 0.0, 0.0])
        # vortex core between different surfaces only (a tail can sit in a wing wake leg)
        width = np.linalg.norm(self.B - self.A, axis=1)
        same = self.psurf[:, None] == self.psurf[None, :]
        rc2 = np.where(same, 0.0, (0.05 * width[None, :]) ** 2)

        def infl(P):
            return (_seg(P, As, Bs, rc2) + _semi(P, Bs, u, rc2) - _semi(P, As, u, rc2))
        n0p = self.S["n0"][self.pstrip]
        self.V_ctrl = infl(Cs)
        self.AIC = np.einsum('nmk,nk->nm', self.V_ctrl, n0p)
        self.lu = np.linalg.inv(self.AIC)
        self.V_mid = infl(Ms)                         # at the bound-vortex midpoints (forces)
        self.a0 = 2 * math.pi / self.beta

    # ------------------------------------------------------------------
    def freestream(self, alpha, beta, p=0.0, q=0.0, r=0.0, ref=None, axes="stability"):
        """air velocity (V = 1) at points: function P -> (n,3), geometry axes.
        alpha, beta in deg; p, q, r non-dimensional, stability (or body) axes"""
        ac = self.ac
        a, b = math.radians(alpha), math.radians(beta)
        Vinf = np.array([math.cos(a) * math.cos(b), -math.sin(b), math.sin(a) * math.cos(b)])
        ps, qs, rs = 2 * p / ac.b, 2 * q / ac.c, 2 * r / ac.b      # rad per (m / V)
        if axes == "body":
            pb, rb = ps, rs
        else:
            pb = ps * math.cos(a) - rs * math.sin(a)
            rb = ps * math.sin(a) + rs * math.cos(a)
        om = np.array([-pb, qs, -rb])                               # geometry axes
        ref = np.asarray(ac.cg if ref is None else ref, float)
        return lambda P: Vinf[None, :] - np.cross(om[None, :], P - ref[None, :]), Vinf

    def panel_incidence(self, theta_strip, controls):
        t = theta_strip[self.pstrip].copy()
        for name, d in (controls or {}).items():
            if name in self.ctrl_panels and d:
                idx, g = self.ctrl_panels[name]
                t[idx] += g * math.radians(d)
        return t


class Analysis:
    """aerodynamic model of one aircraft at one flight condition"""

    def __init__(self, ac, speed=50.0, altitude=0.0, polar_dir=None, mach=None):
        self.ac = ac
        rho, mu, a_snd, T, p = W.isa(altitude)
        self.rho, self.mu, self.V, self.alt = rho, mu, speed, altitude
        self.mach = speed / a_snd if mach is None else mach
        self.qbar = 0.5 * rho * speed**2
        self.lat = Lattice(ac, self.mach)
        L = self.lat
        # section data per surface; per-strip Reynolds number
        self.polars, self.polar_source = [], []
        for s in ac.surfaces:
            sp, src = surface_polars(s, polar_dir, self.mach, rho * speed * s.mac / mu,
                                     ar=max(s.span**2 / s.area, 2.0))
            self.polars.append(sp)
            self.polar_source.append(src)
        self.re_strip = rho * speed * L.S["c"] / mu
        a0s = np.zeros(L.nstrip)
        self.clmax = np.zeros(L.nstrip)
        self.a_clmax = np.zeros(L.nstrip)
        for k in range(len(ac.surfaces)):
            m = L.S["surf"] == k
            a0, cm_, am = self.polars[k].props(self.re_strip[m], L.S["eta"][m])
            a0s[m], self.clmax[m], self.a_clmax[m] = a0, cm_, am
        self.alpha0 = a0s
        twist = np.zeros(L.nstrip)
        for k, s in enumerate(ac.surfaces):
            m = L.S["surf"] == k
            twist[m] = np.radians(s.incidence + s.twist_at(L.S["eta"][m]))
        self.theta_base = twist - a0s
        # fuselage
        f = ac.fuselage
        self.fus_vol, self.fus_wet = f.volume_and_wetted()
        if self.fus_vol > 0:
            kk = f.k2_minus_k1()
            self.Cm_a_fus = 2 * kk * self.fus_vol / (ac.S * ac.c)          # per rad, nose-up
            self.Cn_b_fus = -2 * kk * self.fus_vol / (ac.S * ac.b)         # per rad
            re_f = rho * speed * f.length / mu
            cf = 0.455 / math.log10(re_f) ** 2.58 / (1 + 0.144 * self.mach**2) ** 0.65
            fr = f.length / f.diameter
            self.CD_fus = cf * (1 + 60 / fr**3 + fr / 400) * self.fus_wet / ac.S
        else:
            self.Cm_a_fus = self.Cn_b_fus = self.CD_fus = 0.0

    # ------------------------------------------------------------------
    def _section(self, a_eff_deg):
        """cl, cd, cm of every strip at its effective angle"""
        L = self.lat
        cl = np.zeros(L.nstrip); cd = np.zeros(L.nstrip); cm = np.zeros(L.nstrip)
        for k in range(len(self.ac.surfaces)):
            m = L.S["surf"] == k
            v = self.polars[k](a_eff_deg[m], self.re_strip[m], L.S["eta"][m])
            cl[m], cd[m], cm[m] = v
        return cl, cd, cm

    def _slope(self, a_eff_deg, h=0.25):
        lo = self._section(a_eff_deg - h)[0]
        hi = self._section(a_eff_deg + h)[0]
        return (hi - lo) / math.radians(2 * h)

    viscous = True          # profile drag + section moments from the polars

    def solve(self, alpha=0.0, beta=0.0, p=0.0, q=0.0, r=0.0, controls=None, nonlinear=False,
              viscous=None, delta0=None, ref=None, tol=1e-6, max_it=60, axes="stability"):
        """coefficients at one state (see the module docstring for the axes;
        axes="body": rates and Cl, Cn in body axes, as flight simulators use)"""
        L, ac = self.lat, self.ac
        vel, Vinf = L.freestream(alpha, beta, p, q, r, ref, axes)
        Vc = vel(L.C)
        n0p = L.S["n0"][L.pstrip]
        tp = L.S["t"][L.pstrip]
        sumstrip = np.zeros((L.nstrip, L.npan))
        sumstrip[L.pstrip, np.arange(L.npan)] = 1.0

        def gamma(delta):
            t = L.panel_incidence(self.theta_base + delta, controls)
            n = n0p * np.cos(t)[:, None] + tp * np.sin(t)[:, None]
            rhs = -(Vc * n).sum(1)
            return L.lu @ rhs, t

        delta = np.zeros(L.nstrip) if delta0 is None else np.array(delta0, float)
        conv, it = True, 0
        if nonlinear:
            conv = False
            for it in range(1, max_it + 1):
                Gp, t = gamma(delta)
                Gs = sumstrip @ Gp
                cl = 2 * Gs / L.S["c"]
                a_eff = cl / L.a0 + self.alpha0 - delta
                cl2 = self._section(np.degrees(a_eff))[0]
                R = cl - cl2
                if np.abs(R).max() < tol:
                    conv = True
                    break
                # Jacobian d cl / d delta from the lattice
                dn = -n0p * np.sin(t)[:, None] + tp * np.cos(t)[:, None]
                D = np.zeros((L.npan, L.nstrip))
                D[np.arange(L.npan), L.pstrip] = -(Vc * dn).sum(1)
                Bm = 2 * (sumstrip @ (L.lu @ D)) / L.S["c"][:, None]
                s = self._slope(np.degrees(a_eff))
                J = Bm - s[:, None] * (Bm / L.a0 - np.eye(L.nstrip))
                try:
                    step = np.linalg.solve(J, -R)
                except np.linalg.LinAlgError:
                    step = -R / np.diag(J)
                step = np.clip(step, -math.radians(4), math.radians(4))
                delta = delta + (1.0 if it < 30 else 0.5) * step
        Gp, t = gamma(delta)
        Gs = sumstrip @ Gp
        cl = 2 * Gs / L.S["c"]
        a_eff = cl / L.a0 + self.alpha0 - delta
        _, cd, cm = self._section(np.degrees(a_eff))
        if not (self.viscous if viscous is None else viscous):
            cd, cm = 0 * cd, 0 * cm
        # Kutta-Joukowski on every bound segment, full local velocity
        Vm = vel(L.P_mid) + np.einsum('nmk,m->nk', L.V_mid, Gp)
        lvec = L.B - L.A
        Fp = Gp[:, None] * np.cross(Vm, lvec)                      # rho = 1
        q0 = 0.5
        refp = np.asarray(ac.cg if ref is None else ref, float)
        M = np.cross(L.P_mid - refp, Fp).sum(0)
        F = Fp.sum(0)
        # profile drag (along the local wind) and section moments
        Vs = vel(L.S["mid"])
        us = Vs / np.linalg.norm(Vs, axis=1)[:, None]
        Dv = (q0 * L.S["area"] * cd)[:, None] * us
        F = F + Dv.sum(0)
        M = M + np.cross(L.S["mid"] - refp, Dv).sum(0)
        M = M + ((q0 * L.S["c"] * L.S["area"] * cm)[:, None] * L.S["s"]).sum(0)
        # coefficients
        a, b_ = math.radians(alpha), math.radians(beta)
        S, b, c = ac.S, ac.b, ac.c
        CL = F @ np.array([-math.sin(a), 0, math.cos(a)]) / (q0 * S)
        CD = F @ Vinf / (q0 * S)
        CY = F[1] / (q0 * S)
        Clb, Cm, Cnb = -M[0] / (q0 * S * b), M[1] / (q0 * S * c), -M[2] / (q0 * S * b)
        if axes == "body":
            Cl, Cn = Clb, Cnb
        else:
            Cl = Clb * math.cos(a) + Cnb * math.sin(a)
            Cn = -Clb * math.sin(a) + Cnb * math.cos(a)
        # fuselage (Munk moments, friction drag) and extra drag
        if not (self.viscous if viscous is None else viscous):
            pass
        Cm += self.Cm_a_fus * math.sin(a) * math.cos(a)
        Cn += self.Cn_b_fus * math.sin(b_) * math.cos(b_)
        CD += self.CD_fus + ac.cd_extra
        CDi = (Fp.sum(0) @ Vinf) / (q0 * S)
        # per-surface lift and the hinge moments
        surf = {}
        for k, s in enumerate(ac.surfaces):
            m = L.psurf == k
            surf[s.name] = {"CL": float(Fp[m].sum(0) @ np.array([-math.sin(a), 0, math.cos(a)]) / (q0 * S)),
                            "CY": float(Fp[m].sum(0)[1] / (q0 * S))}
        hinge = {}
        n_t = n0p * np.cos(t)[:, None] + tp * np.sin(t)[:, None]
        for name, (idx, g) in L.ctrl_panels.items():
            k = L.psurf[idx[0]]
            right = L.S["side"][L.pstrip[idx]] >= 0
            ii = idx[right]
            Fn = (Fp[ii] * n_t[ii]).sum(1)
            arm = L.P_mid[ii, 0] - L.ctrl_hinge[name][right]
            st = np.unique(L.pstrip[ii])
            cf = ac.controls()[name][1].cf
            Sc = (L.S["area"][st] * cf).sum()
            cc = (L.S["c"][st] ** 2 * L.S["ds"][st]).sum() / max((L.S["c"][st] * L.S["ds"][st]).sum(), 1e-12) * cf
            # in the sense of a positive control deflection: sign of the gain on
            # the panels used (right-hand surface; the list starts on the left wing)
            hinge[name] = float(-(Fn * arm).sum() / (q0 * Sc * cc)) * float(np.sign(g[right][0]) or 1)
        stalled = (cl > 0.98 * self.clmax) | (np.degrees(a_eff) > self.a_clmax)
        # per-strip force (rho V^2 = 1 units) and the panel forces, for structural loads
        Fs = np.zeros((L.nstrip, 3))
        np.add.at(Fs, L.pstrip, Fp)
        Fs += Dv
        self._last = {"Fp": Fp, "Dv": Dv, "cm": cm, "Fs": Fs}
        return {"alpha": alpha, "beta": beta, "p": p, "q": q, "r": r, "controls": dict(controls or {}),
                "CL": float(CL), "CD": float(CD), "CY": float(CY), "Cl": float(Cl), "Cm": float(Cm),
                "Cn": float(Cn), "CDi": float(CDi), "CD_profile": float((q0 * L.S["area"] * cd).sum() / (q0 * S)),
                "CD_fuselage": float(self.CD_fus), "surfaces": surf, "hinge": hinge,
                "converged": bool(conv), "iterations": it,
                "strips": {"cl": cl, "cd": cd, "cm": cm, "gamma": Gs, "a_eff": np.degrees(a_eff),
                           "stalled": stalled}, "_delta": delta}

    # ------------------------------------------------------------------
    def derivatives(self, alpha=0.0, controls=None, ref=None, nonlinear=False, axes="stability",
                    downwash=True):
        """stability and control derivatives (per rad; rates non-dimensional)
        by central differences around (alpha, beta = 0)"""
        ha, hb, hr, hd = 0.5, 0.5, 0.01, 1.0
        keys = ("CL", "CD", "CY", "Cl", "Cm", "Cn")
        base = dict(alpha=alpha, controls=controls, ref=ref, nonlinear=nonlinear, axes=axes)

        def d(var, h, scale):
            kw1, kw2 = dict(base), dict(base)
            kw1[var] = base.get(var, 0.0) + h if var in base else h
            kw2[var] = base.get(var, 0.0) - h if var in base else -h
            r1, r2 = self.solve(**kw1), self.solve(**kw2)
            return {k: (r1[k] - r2[k]) / (2 * h * scale) for k in keys}, r1, r2

        out = {}
        r0 = self.solve(**base)
        da, ra1, ra2 = d("alpha", ha, math.radians(1))
        db, _, _ = d("beta", hb, math.radians(1))
        dp, _, _ = d("p", hr, 1.0)
        dq, _, _ = d("q", hr, 1.0)
        dr, _, _ = d("r", hr, 1.0)
        for k in keys:
            out[f"{k}_a"] = da[k]; out[f"{k}_b"] = db[k]
            out[f"{k}_p"] = dp[k]; out[f"{k}_q"] = dq[k]; out[f"{k}_r"] = dr[k]
        # hinge moment derivatives
        out["hinge_a"] = {n: (ra1["hinge"][n] - ra2["hinge"][n]) / (2 * math.radians(ha))
                          for n in r0["hinge"]}
        out["controls"] = {}
        for name in self.lat.ctrl_panels:
            c0 = dict(controls or {})
            c1, c2 = dict(c0), dict(c0)
            c1[name] = c0.get(name, 0.0) + hd
            c2[name] = c0.get(name, 0.0) - hd
            r1 = self.solve(alpha=alpha, controls=c1, ref=ref, nonlinear=nonlinear, axes=axes)
            r2 = self.solve(alpha=alpha, controls=c2, ref=ref, nonlinear=nonlinear, axes=axes)
            dd = {k: (r1[k] - r2[k]) / (2 * math.radians(hd)) for k in keys}
            dd["Ch_d"] = (r1["hinge"][name] - r2["hinge"][name]) / (2 * math.radians(hd))
            dd["Ch_a"] = out["hinge_a"].get(name)
            dd["Ch_0"] = r0["hinge"].get(name)
            out["controls"][name] = dd
        # downwash at the horizontal tail and alpha-dot derivatives
        if downwash:
            out.update(self._downwash(alpha, controls, ref))
        out["base"] = {k: r0[k] for k in keys + ("CDi",)}
        ac = self.ac
        if abs(out["CL_a"]) > 1e-9:
            xr = (ac.cg if ref is None else ref)[0]
            out["x_np"] = xr - out["Cm_a"] / out["CL_a"] * ac.c
            out["static_margin"] = (out["x_np"] - ac.cg[0]) / ac.c
        return out

    def _tail_index(self):
        for k, s in enumerate(self.ac.surfaces):
            if k > 0 and not s.vertical and s.x > self.ac.wing.x:
                return k
        return None

    def _downwash(self, alpha, controls, ref):
        """d(eps)/d(alpha) at the horizontal tail from the wing's vortices;
        CL_adot, Cm_adot from the tail lag (Etkin / Nelson)"""
        k = self._tail_index()
        if k is None:
            return {"deps_da": 0.0, "CL_adot": 0.0, "Cm_adot": 0.0}
        L = self.lat
        tail_pan = np.where(L.psurf == k)[0]
        wing_pan = np.where(L.psurf == 0)[0]

        def eps(a):
            vel, _ = L.freestream(a, 0.0, ref=ref)
            Vc = vel(L.C)
            n0p = L.S["n0"][L.pstrip]
            tp = L.S["t"][L.pstrip]
            t = L.panel_incidence(self.theta_base, controls)
            n = n0p * np.cos(t)[:, None] + tp * np.sin(t)[:, None]
            G = L.lu @ (-(Vc * n).sum(1))
            w = np.einsum('nmk,m->nk', L.V_ctrl[np.ix_(tail_pan, wing_pan)], G[wing_pan])
            wa = (w * n0p[tail_pan]).sum(1)                           # normal (up) velocity
            area = L.S["area"][L.pstrip[tail_pan]]
            return -float((wa * area).sum() / area.sum())             # downwash angle (rad)
        h = 0.5
        de = (eps(alpha + h) - eps(alpha - h)) / (2 * math.radians(h))
        # tail lift slope (on S_ref) from the full lattice
        r1 = self.solve(alpha=alpha + h, controls=controls, ref=ref)
        r2 = self.solve(alpha=alpha - h, controls=controls, ref=ref)
        name = self.ac.surfaces[k].name
        aT = (r1["surfaces"][name]["CL"] - r2["surfaces"][name]["CL"]) / (2 * math.radians(h))
        aT_isolated = aT / max(1 - de, 0.05)
        ac = self.ac
        s = ac.surfaces[k]
        xr = (ac.cg if ref is None else ref)[0]
        x_t = s.mac_position()[0] + 0.25 * s.mac
        lt = x_t - xr
        CL_adot = 2 * aT_isolated * de * lt / ac.c
        Cm_adot = -2 * aT_isolated * de * (lt / ac.c) ** 2
        return {"deps_da": de, "CL_adot": CL_adot, "Cm_adot": Cm_adot, "tail_lt": lt}

    # ------------------------------------------------------------------
    def trim(self, CL_req=None, n=1.0, control="elevator", ref=None, nonlinear=False,
             alpha0=2.0, tol=1e-6):
        """angle of attack and control deflection for CL = CL_req (default
        n W / q S) and Cm = 0 about the c.g."""
        ac = self.ac
        if CL_req is None:
            CL_req = n * ac.mass * G0 / (self.qbar * ac.S)
        a, d = alpha0, 0.0
        delta = None
        for _ in range(30):
            r = self.solve(alpha=a, controls={control: d}, ref=ref, nonlinear=nonlinear, delta0=delta)
            delta = r["_delta"] if nonlinear else None
            R = np.array([r["CL"] - CL_req, r["Cm"]])
            if np.abs(R).max() < tol:
                break
            ra = self.solve(alpha=a + 0.5, controls={control: d}, ref=ref, nonlinear=nonlinear, delta0=delta)
            rd = self.solve(alpha=a, controls={control: d + 1.0}, ref=ref, nonlinear=nonlinear, delta0=delta)
            J = np.array([[(ra["CL"] - r["CL"]) / 0.5, (rd["CL"] - r["CL"]) / 1.0],
                          [(ra["Cm"] - r["Cm"]) / 0.5, (rd["Cm"] - r["Cm"]) / 1.0]])
            step = np.linalg.solve(J, -R)
            step = np.clip(step, -5, 5)
            a, d = a + step[0], d + step[1]
        r["alpha_trim"], r["control_trim"], r["trim_control"] = a, d, control
        r["CL_req"] = CL_req
        r["L_D"] = r["CL"] / r["CD"] if r["CD"] > 0 else float("nan")
        return r


# ----------------------------------------------------------------------
# flight dynamics: linear modes
# ----------------------------------------------------------------------
def modes(ac, an, der, trim):
    """eigenvalues of the longitudinal and lateral-directional small-
    perturbation equations (stability axes, Nelson ch. 4 / 5), thrust =
    drag and independent of speed, Ixz = 0. Returns the modes with natural
    frequency, damping, period / time constant, and the MIL-F-8785C
    (Class I, Category B) Level 1 limits."""
    V, rho = an.V, an.rho
    Q = 0.5 * rho * V**2
    S, b, c, m = ac.S, ac.b, ac.c, ac.mass
    Ix, Iy, Iz = ac.inertias()
    CL0, CD0 = trim["CL"], trim["CD"]
    CLa, CDa, Cma = der["CL_a"], der["CD_a"], der["Cm_a"]
    CLq, Cmq = der["CL_q"], der["Cm_q"]
    CLad, Cmad = der["CL_adot"], der["Cm_adot"]
    Xu = -2 * CD0 * Q * S / (m * V)
    Xw = -(CDa - CL0) * Q * S / (m * V)
    Zu = -2 * CL0 * Q * S / (m * V)
    Zw = -(CLa + CD0) * Q * S / (m * V)
    Zwd = -CLad * c / (2 * V) * Q * S / (V * m)
    Zq = -CLq * c / (2 * V) * Q * S / m
    Mw = Cma * Q * S * c / (V * Iy)
    Mwd = Cmad * c / (2 * V) * Q * S * c / (V * Iy)
    Mq = Cmq * c / (2 * V) * Q * S * c / Iy
    den = 1 - Zwd
    A_lon = np.array([
        [Xu, Xw, 0.0, -G0],
        [Zu / den, Zw / den, (V + Zq) / den, 0.0],
        [Mwd * Zu / den, Mw + Mwd * Zw / den, Mq + Mwd * (V + Zq) / den, 0.0],
        [0.0, 0.0, 1.0, 0.0]])
    Yb = Q * S * der["CY_b"] / m
    Yp = Q * S * b * der["CY_p"] / (2 * m * V)
    Yr = Q * S * b * der["CY_r"] / (2 * m * V)
    Lb = Q * S * b * der["Cl_b"] / Ix
    Lp = Q * S * b**2 * der["Cl_p"] / (2 * Ix * V)
    Lr = Q * S * b**2 * der["Cl_r"] / (2 * Ix * V)
    Nb = Q * S * b * der["Cn_b"] / Iz
    Np = Q * S * b**2 * der["Cn_p"] / (2 * Iz * V)
    Nr = Q * S * b**2 * der["Cn_r"] / (2 * Iz * V)
    A_lat = np.array([
        [Yb / V, Yp / V, -(1 - Yr / V), G0 / V],
        [Lb, Lp, Lr, 0.0],
        [Nb, Np, Nr, 0.0],
        [0.0, 1.0, 0.0, 0.0]])

    def describe(lam):
        lam = complex(lam)
        if abs(lam.imag) > 1e-9:
            wn = abs(lam)
            z = -lam.real / wn
            return {"real": lam.real, "imag": abs(lam.imag), "omega_n": wn, "zeta": z,
                    "period_s": 2 * math.pi / abs(lam.imag),
                    "t_half_s": math.log(2) / -lam.real if lam.real < 0 else None,
                    "t_double_s": math.log(2) / lam.real if lam.real > 0 else None}
        return {"real": lam.real, "imag": 0.0, "time_constant_s": -1 / lam.real if lam.real < 0 else None,
                "t_half_s": math.log(2) / -lam.real if lam.real < 0 else None,
                "t_double_s": math.log(2) / lam.real if lam.real > 0 else None}

    out = {}
    ev = np.linalg.eigvals(A_lon)
    cpx = sorted([e for e in ev if e.imag > 1e-9], key=lambda e: -abs(e))
    real = sorted([e.real for e in ev if abs(e.imag) <= 1e-9], key=lambda e: -abs(e))
    if len(cpx) == 2:
        out["short period"], out["phugoid"] = describe(cpx[0]), describe(cpx[1])
    else:
        if cpx:
            out["phugoid" if abs(cpx[0]) < 1.0 else "short period"] = describe(cpx[0])
        for i, e in enumerate(real):
            out[f"longitudinal real {i + 1}"] = describe(e)
    ev = np.linalg.eigvals(A_lat)
    cpx = [e for e in ev if e.imag > 1e-9]
    real = sorted([e.real for e in ev if abs(e.imag) <= 1e-9], key=lambda e: -abs(e))
    if cpx:
        out["dutch roll"] = describe(cpx[0])
    if len(real) >= 2:
        out["roll"], out["spiral"] = describe(real[0]), describe(real[-1])
    elif real:
        out["roll"] = describe(real[0])
    # MIL-F-8785C Class I (small light aircraft), Category B, Level 1
    chk = {}
    if "short period" in out:
        z = out["short period"]["zeta"]
        chk["short period"] = (0.30 <= z <= 2.0, "damping ratio 0.30 to 2.0")
    if "phugoid" in out:
        chk["phugoid"] = (out["phugoid"]["zeta"] >= 0.04, "damping ratio at least 0.04")
    if "dutch roll" in out:
        d = out["dutch roll"]
        chk["dutch roll"] = (d["zeta"] >= 0.08 and d["omega_n"] >= 0.4 and d["zeta"] * d["omega_n"] >= 0.15,
                             "ζ ≥ 0.08, ω_n ≥ 0.4 rad/s, ζω_n ≥ 0.15 rad/s")
    if "roll" in out:
        tc = out["roll"].get("time_constant_s")
        chk["roll"] = (tc is not None and tc <= 1.4, "time constant at most 1.4 s")
    if "spiral" in out:
        sp = out["spiral"]
        chk["spiral"] = (sp.get("t_double_s") is None or sp["t_double_s"] >= 20.0,
                         "time to double at least 20 s (or stable)")
    for k, (ok, txt) in chk.items():
        out[k]["level1"] = bool(ok)
        out[k]["criterion"] = txt
    out["_A_lon"], out["_A_lat"] = A_lon.tolist(), A_lat.tolist()
    out["_inertia"] = (Ix, Iy, Iz)
    return out


def responses(ac, an, der, md, t_end=(12.0, 12.0), elevator=-1.0, aileron=5.0, rudder=5.0):
    """linear time responses (stability axes) from the matrices of modes():
    elevator step (deg; - = trailing edge up, nose up), aileron step,
    rudder doublet (1 s each way). Returns time histories in deg, deg/s, m/s."""
    from scipy.linalg import expm
    V, rho = an.V, an.rho
    Q = 0.5 * rho * V**2
    S, b, c, m = ac.S, ac.b, ac.c, ac.mass
    Ix, Iy, Iz = md["_inertia"]
    ctr = der["controls"]
    A_lon, A_lat = np.array(md["_A_lon"]), np.array(md["_A_lat"])
    out = {}

    def sim(Amat, Bcol, u_of_t, T, n=600):
        dt = T / n
        M = np.zeros((5, 5))
        M[:4, :4], M[:4, 4] = Amat * dt, Bcol * dt
        E = expm(M)
        Ad, Bd = E[:4, :4], E[:4, 4]
        x = np.zeros(4)
        hist = [x.copy()]
        for k in range(n):
            x = Ad @ x + Bd * u_of_t(k * dt)
            hist.append(x.copy())
        return np.linspace(0, T, n + 1), np.array(hist)
    if "elevator" in ctr:
        d = ctr["elevator"]
        Zd = -d["CL"] * Q * S / m
        Md = d["Cm"] * Q * S * c / Iy
        Zwd = -der["CL_adot"] * c / (2 * V) * Q * S / (V * m)
        Mwd = der["Cm_adot"] * c / (2 * V) * Q * S * c / (V * Iy)
        Bc = np.array([-d["CD"] * Q * S / m, Zd / (1 - Zwd), Md + Mwd * Zd / (1 - Zwd), 0.0])
        de = math.radians(elevator)
        tt, x = sim(A_lon, Bc, lambda tk: de, t_end[0])
        out["elevator"] = {"t": tt.tolist(), "input_deg": elevator, "u": x[:, 0].tolist(),
                           "alpha": np.degrees(x[:, 1] / V).tolist(), "q": np.degrees(x[:, 2]).tolist(),
                           "theta": np.degrees(x[:, 3]).tolist()}
    lat_in = []
    if "aileron" in ctr:
        lat_in.append(("aileron", aileron, lambda tk: math.radians(aileron)))
    if "rudder" in ctr:
        lat_in.append(("rudder", rudder, lambda tk: math.radians(rudder) * (1 if tk < 1 else (-1 if tk < 2 else 0))))
    for name, amp, u in lat_in:
        d = ctr[name]
        Bc = np.array([Q * S * d["CY"] / (m * V), Q * S * b * d["Cl"] / Ix, Q * S * b * d["Cn"] / Iz, 0.0])
        tt, x = sim(A_lat, Bc, u, t_end[1])
        out[name] = {"t": tt.tolist(), "input_deg": amp, "beta": np.degrees(x[:, 0]).tolist(),
                     "p": np.degrees(x[:, 1]).tolist(), "r": np.degrees(x[:, 2]).tolist(),
                     "phi": np.degrees(x[:, 3]).tolist()}
    return out


# ----------------------------------------------------------------------
def stability_report(ac, speed, altitude, polar_dir=None, nonlinear=False):
    """trim + derivatives + modes at one flight condition (dict, JSON-able)"""
    an = Analysis(ac, speed, altitude, polar_dir)
    ctrl = "elevator" if "elevator" in an.lat.ctrl_panels else None
    if ctrl:
        tr = an.trim(control=ctrl, nonlinear=nonlinear)
        alpha, controls = tr["alpha_trim"], {ctrl: tr["control_trim"]}
    else:
        tr = an.solve(alpha=2.0)
        alpha, controls = 2.0, {}
    der = an.derivatives(alpha, controls)
    md = modes(ac, an, der, tr)
    resp = responses(ac, an, der, md)
    clean = lambda r: {k: v for k, v in r.items() if k not in ("strips", "_delta")}
    return {"aircraft": ac.name, "speed": speed, "altitude": altitude, "mach": an.mach,
            "qbar": an.qbar, "rho": an.rho, "polar_source": dict(zip([s.name for s in ac.surfaces],
                                                                       an.polar_source)),
            "fuselage": {"Cm_a": an.Cm_a_fus, "Cn_b": an.Cn_b_fus, "CD0": an.CD_fus,
                         "volume": an.fus_vol, "wetted": an.fus_wet},
            "trim": clean(tr), "derivatives": der, "modes": md, "responses": resp,
            "reference": {"S": ac.S, "b": ac.b, "c": ac.c, "cg": list(ac.cg), "mass": ac.mass,
                          "inertia": list(ac.inertias())}}


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--out", default="../cases/aircraft/demo_trainer.json")
    a = sub.add_parser("analyse")
    a.add_argument("aircraft")
    a.add_argument("--speed", type=float, default=50.0)
    a.add_argument("--alt", type=float, default=0.0)
    a.add_argument("--polar-dir", default="../cases/wing_polars")
    a.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "demo":
        demo_aircraft().save(args.out)
        print(f"-> {args.out}")
        return
    ac = Aircraft.load(args.aircraft)
    rep = stability_report(ac, args.speed, args.alt, args.polar_dir)
    t, dv = rep["trim"], rep["derivatives"]
    print(f"{ac.name}: V {args.speed} m/s, {args.alt} m, M {rep['mach']:.3f}")
    print(f"  trim: alpha {t['alpha_trim']:.2f} deg, {t['trim_control']} {t['control_trim']:.2f} deg, "
          f"CL {t['CL']:.3f}, CD {t['CD']:.4f}, L/D {t['L_D']:.1f}")
    print(f"  CL_a {dv['CL_a']:.3f}  Cm_a {dv['Cm_a']:.3f}  x_np {dv['x_np']:.3f} m  "
          f"static margin {100 * dv['static_margin']:.1f} % MAC")
    for k, v in rep["modes"].items():
        if not k.startswith("_"):
            print(f"  {k:13s} " + ", ".join(f"{kk} {vv:.3g}" for kk, vv in v.items()
                                           if isinstance(vv, float)))
    if args.out:
        Path(args.out).write_text(json.dumps(_jsonable(rep), indent=1))


if __name__ == "__main__":
    main()
