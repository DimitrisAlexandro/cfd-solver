"""Drag buildup and propeller / engine models."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerolab.aircraft import aircraft as A  # noqa: E402
from aerolab.aircraft import drag_buildup as D  # noqa: E402
from aerolab.aircraft import propeller as P  # noqa: E402
from aerolab.aircraft import wing as W  # noqa: E402


# ---------------------------------------------------------------- drag buildup
def test_skin_friction_limits():
    re = 3e6
    assert D.cf_mixed(re, 0.0) == pytest.approx(D.cf_turbulent(re))
    assert D.cf_mixed(re, 1.0) == pytest.approx(1.328 / math.sqrt(re))
    # more laminar flow, less friction; Blasius and Prandtl-Schlichting values
    assert D.cf_mixed(re, 0.3) < D.cf_mixed(re, 0.1) < D.cf_turbulent(re)
    assert D.cf_turbulent(1e7) == pytest.approx(0.455 / 7**2.58)


def test_section_perimeter_matches_raymer():
    per, tc, xm = D.section_shape("naca0012")
    assert tc == pytest.approx(0.12, abs=2e-3)
    assert xm == pytest.approx(0.30, abs=0.02)
    assert per == pytest.approx(1.977 + 0.52 * 0.12, rel=0.003)       # Raymer's wetted-area rule


def test_fuselage_drag_equals_aircraft_model():
    """fully turbulent and smooth: the same formula as aircraft.Analysis"""
    ac = A.demo_aircraft()
    an = A.Analysis(ac, 50.0, 0.0, None)
    rho, mu, a, _, _ = W.isa(0.0)
    fd = D.fuselage_drag(ac.fuselage, rho, mu, 50.0, 50.0 / a, x_tr=0.0, k=0.0, S_ref=ac.S)
    assert fd["CD0"] == pytest.approx(an.CD_fus, rel=1e-9)


def test_wing_wetted_area_exposed():
    ac = A.demo_aircraft()
    rho, mu, a, _, _ = W.isa(0.0)
    s = ac.wing
    r = D.surface_drag(s, A.Fuselage(), rho, mu, 50.0, 0.15, S_ref=ac.S)      # no fuselage: whole wing
    per = D.section_shape(s.airfoil)[0]
    assert r["S_wet"] == pytest.approx(per * s.area, rel=2e-3)
    r2 = D.surface_drag(s, ac.fuselage, rho, mu, 50.0, 0.15, S_ref=ac.S)
    assert r2["hidden_semispan"] > 0.3 and r2["S_wet"] < r["S_wet"]


def test_buildup_totals_and_cfe_check():
    ac = A.demo_aircraft()
    r = D.buildup(ac, 60.0, 1500.0, gear={"type": "fixed_faired"}, power_kw=75.0)
    parts = sum(c["CD0"] for c in r["components"]) + sum(m["CD0"] for m in r["misc"])
    assert r["CD0"] == pytest.approx(parts, rel=1e-12)
    assert 0.018 < r["CD0"] < 0.035
    assert r["CD0"] == pytest.approx(r["cfe_check"]["CD0"], rel=0.25)      # Raymer's equivalent friction
    assert sum(c["share_pct"] for c in r["components"] + r["misc"]) == pytest.approx(100.0)
    # retractable gear and no engine: less drag
    r2 = D.buildup(ac, 60.0, 1500.0, gear={"type": "retractable"}, power_kw=0.0)
    assert r2["CD0"] < r["CD0"]
    assert D.cd_extra_for_model(r) == pytest.approx(r["CD0_misc"])


# ---------------------------------------------------------------- propeller
@pytest.fixture(scope="module")
def designed():
    prop, info = P.design(60.0, 2500.0, 1.8, 2, power=60e3, cl_design=0.7)
    return prop, info


def test_design_reanalysis_consistent(designed):
    """Adkins-Liebeck design re-analysed by BEMT: same thrust, power, efficiency"""
    prop, info = designed
    r = P.analyse(prop, 60.0, 2500.0)
    assert r["P"] == pytest.approx(info["P"], rel=0.01)
    assert r["T"] == pytest.approx(info["T"], rel=0.01)
    assert r["eta"] == pytest.approx(info["eta"], abs=0.005)
    assert info["eta"] < info["eta_ideal"]


def test_inviscid_design_approaches_actuator_disc():
    """no profile drag: the minimum-induced-loss efficiency rises with the
    blade count towards the actuator-disc (Froude) value and stays below it
    (tip loss, swirl, hub)"""
    sec = P.GenericSection(cd0=0.0, k=0.0, t_c=0.0)
    sec.polar = W.SectionPolar.thin_airfoil(a0=2 * math.pi, alpha0_deg=-3.5)
    etas = []
    for B in (2, 4, 8, 20):
        prop, info = P.design(30.0, 2200.0, 2.0, B, power=40e3, cl_design=0.6, section=sec)
        etas.append(info["eta"])
        assert info["eta"] < info["eta_ideal"]
        if B <= 8:
            assert P.analyse(prop, 30.0, 2200.0)["eta"] == pytest.approx(info["eta"], abs=0.005)
    assert np.all(np.diff(etas) > 0)
    assert etas[-1] == pytest.approx(info["eta_ideal"], abs=0.016)


def test_static_thrust_below_momentum_limit(designed):
    prop, _ = designed
    r = P.analyse(prop, 0.0, 2500.0)
    T_ideal = (2 * 1.225 * math.pi * 0.9**2 * r["P"] ** 2) ** (1 / 3)
    assert 0 < r["T"] < T_ideal


def test_constant_speed_absorbs_engine_power(designed):
    prop, _ = designed
    eng = P.Engine("piston", 75.0, 2700.0, 300.0)
    bp = P.BEMTPropulsion(eng, prop, "constant_speed", rpm=2700.0)
    op = bp.operating_point(60.0, 0.0)
    r = P.analyse(prop, 60.0, 2700.0, dbeta=op["dbeta"])
    assert r["P"] == pytest.approx(eng.power(0.0, 2700.0), rel=0.02)
    # thrust falls with altitude, efficiency at part throttle is sensible
    assert bp.thrust(60.0, 3000.0) < bp.thrust(60.0, 0.0)
    assert 0.7 < bp.eta_at_thrust(60.0, 600.0) < 0.95


def test_engines():
    e = P.Engine("electric", 50.0, 2400.0, battery_kwh=20.0)
    assert e.power(3000.0) == pytest.approx(50e3)
    assert e.fuel_flow(50e3) == 0.0
    p = P.Engine("piston", 100.0)
    assert p.power(3000.0) < p.power(0.0)
    t = P.Engine("turboprop", 500.0, sfc=300.0)
    sig = W.isa(5000.0)[0] / 1.225
    assert t.power(5000.0) == pytest.approx(500e3 * sig**0.7)
