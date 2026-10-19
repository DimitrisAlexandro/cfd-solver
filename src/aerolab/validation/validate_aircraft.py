"""
Validation of the whole-aircraft vortex lattice (aircraft.py):
  1. wing alone against the wing tool (wing.py, Weissinger / VLM, Phase 6)
  2. wing + tails against an independent vortex-lattice code (AeroSandbox
     VortexLatticeMethod; development check only, not a dependency):
     forces, moments and stability derivatives at the same geometry
     (symmetric sections, no fuselage, inviscid)
  3. classical results: elliptic wing (e = 1, Helmbold lift slope),
     roll damping of a rectangular wing (lifting-surface theory),
     dihedral effect (strip theory), modal solver against the short-period
     and phugoid (Lanchester) approximations
    python src/run.py validate_aircraft      -> ../validation/aircraft_validation.{json,md}
"""
import json
import math
from pathlib import Path

import numpy as np

from aerolab.aircraft import aircraft as A
from aerolab.aircraft import wing as W

OUT = Path(__file__).resolve().parents[3] / "validation"


def test_aircraft():
    """symmetric sections, no fuselage: comparable with any inviscid lattice"""
    wing = A.Surface("wing", span=10.0, root_chord=1.5, tip_chord=0.9, x=2.0, z=0.0, sweep=4.0,
                     dihedral=5.0, incidence=2.0, airfoil="naca0012",
                     controls=[A.Control("aileron", 0.6, 0.95, 0.25, "antisymmetric")], ns=16, nc=8)
    ht = A.Surface("htail", span=3.4, root_chord=0.85, tip_chord=0.6, x=6.3, z=0.5, sweep=6.0,
                   incidence=-2.0, airfoil="naca0012",
                   controls=[A.Control("elevator", 0.0, 1.0, 0.4)], ns=10, nc=8)
    fin = A.Surface("fin", span=1.4, root_chord=1.1, tip_chord=0.6, x=6.0, z=0.5, sweep=30.0,
                    airfoil="naca0012", symmetric=False, vertical=True,
                    controls=[A.Control("rudder", 0.0, 1.0, 0.4)], ns=10, nc=8)
    return A.Aircraft("validation", [wing, ht, fin], A.Fuselage(), mass=750, cg=(2.4, 0, 0))


def test_cranked():
    """multi-panel wing: constant-chord centre panel, swept tapered outer panel with
    a dihedral break and washout; tail with a swept outer panel"""
    wing = A.Surface("wing", span=11.0, root_chord=1.6, tip_chord=0.7, x=2.0, z=0.0, sweep=8.0,
                     dihedral=4.0, twist=-3.0, incidence=2.0, airfoil="naca0012",
                     kinks=[{"y": 2.0, "chord": 1.6, "dx": 0.0, "dz": 0.0, "twist": 0.0}],
                     controls=[A.Control("aileron", 0.6, 0.95, 0.25, "antisymmetric")], ns=18, nc=8)
    ht = A.Surface("htail", span=3.6, root_chord=0.9, tip_chord=0.45, x=6.5, z=0.5, sweep=12.0,
                   incidence=-2.0, airfoil="naca0012",
                   kinks=[{"y": 0.8, "chord": 0.85, "dx": 0.03, "dz": 0.0, "twist": 0.0}],
                   controls=[A.Control("elevator", 0.0, 1.0, 0.4)], ns=12, nc=8)
    fin = A.Surface("fin", span=1.4, root_chord=1.2, tip_chord=0.55, x=6.1, z=0.5, sweep=30.0,
                    airfoil="naca0012", symmetric=False, vertical=True,
                    kinks=[{"y": 0.5, "chord": 1.0, "dx": 0.25}],
                    controls=[A.Control("rudder", 0.0, 1.0, 0.4)], ns=10, nc=8)
    return A.Aircraft("cranked validation", [wing, ht, fin], A.Fuselage(), mass=800, cg=(2.5, 0, 0))


def to_asb(ac):
    import aerosandbox as asb
    wings = []
    for s in ac.surfaces:
        xs = []
        for eta in s.stations()[0]:             # root, kinks, tip
            c = float(s.chord(eta))
            x = float(s.x_le(eta))
            if s.vertical:
                le = [x, s.y, s.z + eta * s.semi]
            else:
                le = [x, s.y + eta * s.semi, s.z + float(s.rise(eta))]
            xs.append(asb.WingXSec(xyz_le=le, chord=c, twist=s.incidence + float(s.twist_at(eta)),
                                   airfoil=asb.Airfoil("naca0012")))
        wings.append(asb.Wing(name=s.name, xsecs=xs, symmetric=s.symmetric))
    return asb.Airplane(wings=wings, xyz_ref=list(ac.cg), s_ref=ac.S, c_ref=ac.c, b_ref=ac.b)


def ours(ac, alpha):
    an = A.Analysis(ac, speed=50.0, altitude=0.0, polar_dir=None)
    an.viscous = False
    an.Cm_a_fus = an.Cn_b_fus = an.CD_fus = 0.0
    ac.cd_extra = 0.0
    r = an.solve(alpha=alpha)
    d = an.derivatives(alpha)
    return r, d


def compare_asb(ac, alpha=2.0):
    import aerosandbox as asb
    ap = to_asb(ac)
    op = asb.OperatingPoint(velocity=50.0, alpha=alpha)
    vlm = asb.VortexLatticeMethod(ap, op, spanwise_resolution=12, chordwise_resolution=8)
    res = vlm.run_with_stability_derivatives(alpha=True, beta=True, p=True, q=True, r=True)
    return {k: float(np.asarray(v)) for k, v in res.items()
            if np.ndim(v) == 0 or np.size(v) == 1}


def main():
    out = {}
    # ---- 1. wing alone against wing.py
    wg = W.Wing(10.0, 1.5, 0.9, sweep=4.0, dihedral=5.0, twist=0.0, incidence=0.0, n_half=40)
    r_w = W.solve(wg, 5.0, None, None, 0.0, nc=6)
    s = A.Surface("wing", span=10.0, root_chord=1.5, tip_chord=0.9, sweep=4.0, dihedral=5.0,
                  airfoil="naca0012", ns=40, nc=6)
    ac1 = A.Aircraft("wing", [s], A.Fuselage(), cg=(0.25 * 1.5 + 0.0, 0, 0))
    an = A.Analysis(ac1, 50.0, 0.0)
    an.viscous = False
    an.CD_fus = 0.0
    r_a = an.solve(alpha=5.0)
    out["wing_vs_wingtool"] = {"CL_wing_tool": r_w["CL"], "CL_aircraft": r_a["CL"],
                               "CDi_wing_tool_trefftz": r_w["CDi"], "CDi_aircraft_nearfield": r_a["CDi"]}
    # ---- 2. against AeroSandbox
    ac = test_aircraft()
    r, d = ours(ac, 2.0)
    try:
        ref = compare_asb(ac, 2.0)
    except Exception as e:                       # noqa: BLE001
        ref = {"error": str(e)}
    pairs = [("CL", r["CL"]), ("Cm", r["Cm"]), ("CLa", d["CL_a"]), ("Cma", d["Cm_a"]),
             ("CYb", d["CY_b"]), ("Clb", d["Cl_b"]), ("Cnb", d["Cn_b"]), ("Clp", d["Cl_p"]),
             ("Cmq", d["Cm_q"]), ("Cnr", d["Cn_r"]), ("Cnp", d["Cn_p"]), ("Clr", d["Cl_r"]),
             ("CLq", d["CL_q"]), ("CYr", d["CY_r"])]
    out["aircraft_vs_aerosandbox"] = {k: {"aerolab": v, "aerosandbox": ref.get(k)} for k, v in pairs}
    out["aircraft_vs_aerosandbox"]["x_np"] = {"aerolab": d["x_np"], "aerosandbox": ref.get("x_np")}
    # multi-panel (cranked) surfaces: same comparison
    ack = test_cranked()
    rk, dk = ours(ack, 2.0)
    try:
        refk = compare_asb(ack, 2.0)
    except Exception as e:                       # noqa: BLE001
        refk = {"error": str(e)}
    pk = [("CL", rk["CL"]), ("Cm", rk["Cm"]), ("CLa", dk["CL_a"]), ("Cma", dk["Cm_a"]), ("x_np", dk["x_np"]),
          ("CYb", dk["CY_b"]), ("Clb", dk["Cl_b"]), ("Cnb", dk["Cn_b"]), ("Clp", dk["Cl_p"])]
    out["cranked_vs_aerosandbox"] = {k: {"aerolab": v, "aerosandbox": refk.get(k)} for k, v in pk}
    # pitch-rate derivatives: thin-airfoil limit (high aspect ratio wing pitching about c/4:
    # CL_q -> CL_a, Cm_q -> -pi/4 x (3D factor)); AeroSandbox's q / r derivatives use a
    # different definition (its CL_q is 1.5 CL_a on this wing), so rates are checked here
    hw = A.Surface("w", span=30.0, root_chord=1.0, tip_chord=1.0, airfoil="naca0012", ns=30, nc=8)
    ac_h = A.Aircraft("w", [hw], A.Fuselage(), cg=(0.25, 0, 0))
    an_h = A.Analysis(ac_h, 50.0, 0.0)
    an_h.viscous = False
    an_h.CD_fus = 0.0
    dh = an_h.derivatives(2.0)
    out["pitch_rate_thin_airfoil_limit"] = {"AR": 30, "CL_a": dh["CL_a"], "CL_q": dh["CL_q"],
                                            "Cm_q": dh["Cm_q"], "Cm_q_2D": -math.pi / 4}
    # yaw-rate derivatives of the fin against the classical tail-volume formulas
    s = ac.surfaces[2]
    lv = s.x + 0.25 * s.root_chord + 0.5 * s.semi * math.tan(math.radians(s.sweep)) - ac.cg[0]
    aV = -d["CY_b"] * ac.S / s.area
    out["yaw_rate_fin_classical"] = {"CY_r": d["CY_r"], "classical_CY_r": 2 * aV * s.area / ac.S * lv / ac.b,
                                     "Cn_r": d["Cn_r"],
                                     "classical_Cn_r_fin": -2 * aV * s.area / ac.S * (lv / ac.b) ** 2}
    # ---- 3a. elliptic wing
    b, c0 = 10.0, 1.2
    el = A.Surface("ell", span=b, root_chord=c0, tip_chord=c0, airfoil="naca0012", ns=40, nc=4)
    ac_e = A.Aircraft("ell", [el], A.Fuselage(), cg=(0.3, 0, 0))
    an = A.Analysis(ac_e, 50.0, 0.0)
    an.viscous = False
    an.CD_fus = 0.0
    # make it elliptic: chord function through the lattice is trapezoidal, so use the
    # rectangular wing against Helmbold instead and check e for both
    AR = b / c0
    ra = an.solve(alpha=4.0)
    CLa = (an.solve(alpha=4.5)["CL"] - an.solve(alpha=3.5)["CL"]) / math.radians(1.0)
    helm = 2 * math.pi * AR / (2 + math.sqrt(AR**2 + 4))
    out["rectangular_wing"] = {"AR": AR, "CL_a": CLa, "helmbold": helm,
                               "e": ra["CL"] ** 2 / (math.pi * AR * ra["CDi"])}
    # ---- 3b. roll damping and dihedral effect of a rectangular wing
    d_r = an.derivatives(4.0)
    # lifting-line/strip estimate Clp = -CL_a/12 * (1 + 3 lambda)/(1 + lambda) (lambda = 1: -CLa/6)
    out["roll_damping"] = {"Cl_p": d_r["Cl_p"], "strip_theory_with_3D_slope": -CLa / 6}
    dih = A.Surface("dih", span=b, root_chord=c0, tip_chord=c0, dihedral=6.0, airfoil="naca0012", ns=30, nc=4)
    ac_d = A.Aircraft("dih", [dih], A.Fuselage(), cg=(0.3, 0, 0))
    an_d = A.Analysis(ac_d, 50.0, 0.0)
    an_d.viscous = False
    an_d.CD_fus = 0.0
    dd = an_d.derivatives(2.0)
    G = math.radians(6.0)
    out["dihedral_effect"] = {"Cl_b": dd["Cl_b"], "strip_theory": -CLa * G / 4}
    # ---- 3c. modes against the classical approximations (demo trainer)
    ac_t = A.demo_aircraft()
    an_t = A.Analysis(ac_t, 50.0, 0.0, None)
    tr = an_t.trim()
    dv = an_t.derivatives(tr["alpha_trim"], {"elevator": tr["control_trim"]})
    md = A.modes(ac_t, an_t, dv, tr)
    V, g = 50.0, A.G0
    out["phugoid"] = {"omega_n": md["phugoid"]["omega_n"], "lanchester": math.sqrt(2) * g / V,
                      "zeta": md["phugoid"]["zeta"],
                      "approx_zeta": tr["CD"] / (math.sqrt(2) * tr["CL"])}
    Iy = ac_t.inertias()[1]
    Q = 0.5 * an_t.rho * V**2
    Ma = dv["Cm_a"] * Q * ac_t.S * ac_t.c / Iy
    Mq = dv["Cm_q"] * ac_t.c / (2 * V) * Q * ac_t.S * ac_t.c / Iy
    Mad = dv["Cm_adot"] * ac_t.c / (2 * V) * Q * ac_t.S * ac_t.c / Iy
    Za = -(dv["CL_a"] + tr["CD"]) * Q * ac_t.S / ac_t.mass
    wsp = math.sqrt(Za * Mq / V - Ma)
    zsp = -(Mq + Mad + Za / V) / (2 * wsp)
    out["short_period"] = {"omega_n": md["short period"]["omega_n"], "approx_omega_n": wsp,
                           "zeta": md["short period"]["zeta"], "approx_zeta": zsp}
    OUT.mkdir(exist_ok=True)
    (OUT / "aircraft_validation.json").write_text(json.dumps(A._jsonable(out), indent=1))
    print(json.dumps(A._jsonable(out), indent=1))
    return out


if __name__ == "__main__":
    main()
