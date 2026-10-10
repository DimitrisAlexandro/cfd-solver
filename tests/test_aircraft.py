"""Whole-aircraft model: lattice, derivatives, trim, modes, database exports, loads."""
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import aircraft as A  # noqa: E402
import wing as W  # noqa: E402


@pytest.fixture(scope="module")
def demo():
    ac = A.demo_aircraft()
    an = A.Analysis(ac, 50.0, 0.0, None)          # estimated sections: no CFD files needed
    return ac, an


def test_symmetry(demo):
    ac, an = demo
    r = an.solve(alpha=4.0)
    assert abs(r["CY"]) < 1e-6 and abs(r["Cl"]) < 1e-6 and abs(r["Cn"]) < 1e-6
    rp, rm = an.solve(alpha=4.0, beta=5.0), an.solve(alpha=4.0, beta=-5.0)
    for k in ("CY", "Cl", "Cn"):
        assert rp[k] == pytest.approx(-rm[k], rel=1e-6, abs=1e-9)
    assert rp["CL"] == pytest.approx(rm["CL"], rel=1e-9)


def test_wing_alone_matches_wing_tool():
    wg = W.Wing(10.0, 1.5, 0.9, sweep=4.0, dihedral=5.0, n_half=40)
    r_w = W.solve(wg, 5.0, None, None, 0.0, nc=6)
    s = A.Surface("wing", 10.0, 1.5, 0.9, sweep=4.0, dihedral=5.0, airfoil="naca0012", ns=40, nc=6)
    an = A.Analysis(A.Aircraft("w", [s], A.Fuselage(), cg=(0.375, 0, 0)), 50.0, 0.0)
    an.viscous = False
    an.CD_fus = 0.0
    r = an.solve(alpha=5.0)
    assert r["CL"] == pytest.approx(r_w["CL"], rel=0.02)
    assert r["CDi"] == pytest.approx(r_w["CDi"], rel=0.03)


def test_pitch_rate_thin_airfoil_limit():
    """high aspect ratio wing pitching about c/4: CL_q -> CL_a"""
    s = A.Surface("w", 30.0, 1.0, 1.0, airfoil="naca0012", ns=30, nc=8)
    an = A.Analysis(A.Aircraft("w", [s], A.Fuselage(), cg=(0.25, 0, 0)), 50.0, 0.0)
    an.viscous = False
    an.CD_fus = 0.0
    d = an.derivatives(2.0, downwash=False)
    assert d["CL_q"] == pytest.approx(d["CL_a"], rel=0.03)
    assert d["Cm_q"] == pytest.approx(-math.pi / 4, rel=0.06)


def test_neutral_point_independent_of_cg(demo):
    ac, _ = demo
    out = []
    for x in (2.2, 2.45):
        a2 = A.Aircraft.from_dict(ac.to_dict())
        a2.cg = (x, 0.0, ac.cg[2])
        an = A.Analysis(a2, 50.0, 0.0, None)
        d = an.derivatives(2.0, downwash=False)
        out.append((d["x_np"], d["static_margin"]))
    assert out[0][0] == pytest.approx(out[1][0], abs=0.01 * ac.c)
    assert out[0][1] - out[1][1] == pytest.approx(0.25 / ac.c, rel=0.05)


def test_signs_and_trim(demo):
    ac, an = demo
    d = an.derivatives(2.0)
    c = d["controls"]
    assert d["Cm_a"] < 0 and d["Cn_b"] > 0 and d["Cl_b"] < 0 and d["Cl_p"] < 0 and d["Cn_r"] < 0
    assert d["Cm_q"] < 0 and 0.1 < d["deps_da"] < 0.7
    assert c["elevator"]["Cm"] < 0 and c["elevator"]["CL"] > 0
    assert c["aileron"]["Cl"] > 0 and c["rudder"]["Cn"] < 0 and c["rudder"]["CY"] > 0
    assert c["flap"]["CL"] > 0
    for n in ("elevator", "aileron", "rudder", "flap"):          # unbalanced controls: restoring
        assert c[n]["Ch_d"] < 0, n
    t = an.trim()
    assert t["CL"] == pytest.approx(ac.mass * A.G0 / (an.qbar * ac.S), rel=1e-4)
    assert abs(t["Cm"]) < 1e-4


def test_modes_against_approximations(demo):
    ac, an = demo
    t = an.trim()
    d = an.derivatives(t["alpha_trim"], {"elevator": t["control_trim"]})
    md = A.modes(ac, an, d, t)
    assert md["phugoid"]["zeta"] == pytest.approx(t["CD"] / (math.sqrt(2) * t["CL"]), rel=0.25)
    assert md["phugoid"]["omega_n"] == pytest.approx(math.sqrt(2) * A.G0 / 50.0, rel=0.25)
    assert md["roll"]["time_constant_s"] > 0 and md["dutch roll"]["zeta"] > 0
    assert md["short period"]["zeta"] > 0.3


def test_fuselage_apparent_mass():
    """Lamb's coefficients, prolate spheroid of fineness 6: k1 0.045, k2 0.918"""
    f = A.Fuselage(length=6.0, diameter=1.0)
    assert f.k2_minus_k1() == pytest.approx(0.918 - 0.045, abs=0.01)


def test_database_exports(tmp_path, demo):
    import aero_database as D
    ac, _ = demo
    db = D.aircraft_database(ac, [-4.0, 0.0, 4.0, 8.0], [-4.0, 0.0, 4.0], [0.15], 0.0,
                             {"elevator": [-10.0, 10.0], "aileron": [10.0], "rudder": [10.0]},
                             None, nonlinear=False, log=lambda *a: None)
    D.write_aircraft_outputs(ac, db, tmp_path)
    root = ET.parse(tmp_path / "jsbsim_aero.xml").getroot()
    axes = {a.get("name") for a in root.findall("axis")}
    assert axes == {"LIFT", "DRAG", "SIDE", "ROLL", "PITCH", "YAW"}
    avl = (tmp_path / "aircraft.avl").read_text()
    assert avl.count("\nSURFACE\n") == 3 and "CONTROL" in avl
    import pandas as pd
    x = pd.ExcelFile(tmp_path / "database.xlsx")
    assert "CL lift M0.15" in x.sheet_names and "Cl roll M0.15" in x.sheet_names


def test_loads_balance_and_vn(demo):
    import loads as Ld
    ac, _ = demo
    vn = Ld.vn_diagram(ac, 1.6, -0.9, 5.0, 60.0, "normal")
    assert vn["n_pos"] == pytest.approx(3.8)                  # 750 kg: formula > 3.8, capped
    assert vn["n_neg"] == pytest.approx(-0.4 * 3.8)
    an = A.Analysis(ac, 50.0, 0.0, None)
    t = an.trim(n=1.0)
    r = an.solve(alpha=t["alpha_trim"], controls={"elevator": t["control_trim"]})
    st, strips = Ld.span_loads(ac, an, r, 1.0, 0, 0.35, 0.0)
    # with no inertia relief the root shear of the half wing = its share of the wing lift
    wing_lift = r["surfaces"]["wing"]["CL"] * an.qbar * ac.S
    assert st["shear"][0] == pytest.approx(0.5 * wing_lift, rel=0.03)
    # shear decreases to zero at the tip
    assert abs(st["shear"][-1]) < 0.1 * abs(st["shear"][0])


def test_propeller_and_flap_models(demo):
    import performance as P
    ac, _ = demo
    pr = P.Propulsion(75.0, 0.8, 1.8, 300.0, v_design=60.0)
    P0 = pr.power(0.0)
    assert pr.thrust(60.0, 0.0) == pytest.approx(0.8 * P0 / 60.0, rel=1e-9)     # design point
    assert pr.thrust(0.0, 0.0) <= 2 * 0.8 * P0 / 60.0 + 1e-9                     # linear decay from rest
    assert pr.power(3000.0) < P0                                                  # altitude lapse
    d10, d30 = P.flap_increment(ac, 10.0), P.flap_increment(ac, 30.0)
    assert 0 < d10 < d30 < 0.6


# ---------------------------------------------------------------------- multi-panel surfaces
def _cranked():
    import validate_aircraft as VA
    return VA.test_cranked()


def test_kink_geometry_exact():
    w = _cranked().wing                    # 2 m constant chord 1.6, then taper to 0.7 at 5.5 m
    assert w.area == pytest.approx(2 * (1.6 * 2.0 + 0.5 * (1.6 + 0.7) * 3.5))
    # MAC of the two panels, area-weighted (analytic)
    s1, m1 = 1.6 * 2.0, 1.6
    lam = 0.7 / 1.6
    s2, m2 = 0.5 * 2.3 * 3.5, 2 / 3 * 1.6 * (1 + lam + lam**2) / (1 + lam)
    assert w.mac == pytest.approx((s1 * m1 + s2 * m2) / (s1 + s2), rel=1e-9)
    assert w.chord(2.0 / 5.5) == pytest.approx(1.6) and w.chord(1.0) == pytest.approx(0.7)
    assert w.rise(2.0 / 5.5) == pytest.approx(0.0) and w.twist_at(2.0 / 5.5) == pytest.approx(0.0)
    assert w.twist_at(1.0) == pytest.approx(-3.0)


def test_collinear_kink_changes_nothing(demo):
    ac, an = demo
    w = ac.wing
    e = 2.0 / w.semi
    b = A.Aircraft.from_dict(ac.to_dict())
    b.wing.kinks = [{"y": 2.0, "chord": float(w.chord(e)), "dx": float(w.x_le(e) - w.x),
                     "dz": float(w.rise(e)), "twist": float(w.twist_at(e))}]
    assert b.S == pytest.approx(ac.S) and b.c == pytest.approx(ac.c)
    d0, d1 = an.derivatives(2.0), A.Analysis(b, 50.0, 0.0, None).derivatives(2.0)
    for k in ("CL_a", "Cm_a", "Cl_b", "Cn_b", "static_margin"):
        assert d1[k] == pytest.approx(d0[k], rel=0.01, abs=1e-3), k


def test_cranked_against_aerosandbox():
    pytest.importorskip("aerosandbox")
    import validate_aircraft as VA
    ac = _cranked()
    r, d = VA.ours(ac, 2.0)
    ref = VA.compare_asb(ac, 2.0)
    assert r["CL"] == pytest.approx(ref["CL"], rel=0.03)
    assert d["CL_a"] == pytest.approx(ref["CLa"], rel=0.02)
    assert d["Cm_a"] == pytest.approx(ref["Cma"], rel=0.02)
    assert d["x_np"] == pytest.approx(ref["x_np"], abs=0.01)
    assert d["Cn_b"] == pytest.approx(ref["Cnb"], rel=0.1)


def test_cranked_loads_balance():
    import loads as Ld
    ac = _cranked()
    an = A.Analysis(ac, 50.0, 0.0, None)
    t = an.trim()
    r = an.solve(alpha=t["alpha_trim"], controls={"elevator": t["control_trim"]})
    st, _ = Ld.span_loads(ac, an, r, 1.0, 0, 0.35, 0.0)
    assert st["shear"][0] == pytest.approx(0.5 * r["surfaces"]["wing"]["CL"] * an.qbar * ac.S, rel=0.03)
