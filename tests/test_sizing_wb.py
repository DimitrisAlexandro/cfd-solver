"""Conceptual sizing, weight and balance, scissor plot, landing gear."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerolab.aircraft import aircraft as A  # noqa: E402
from aerolab.aircraft import sizing as SZ  # noqa: E402
from aerolab.aircraft import weight_balance as WB  # noqa: E402

G0 = 9.80665


@pytest.fixture(scope="module")
def sized():
    res, ac = SZ.size({}, log=lambda *a: None)
    return res, ac


# ---------------------------------------------------------------- constraints and weights
def test_constraint_formulas():
    req = SZ.default_requirements()
    con = SZ.constraints(req, 0.025, 0.045)
    assert con["limits"]["stall speed"] == pytest.approx(0.5 * 1.225 * 24.0**2 * 2.0)
    # take-off line back through Roskam's correlation gives the required distance
    ws = np.asarray(con["ws"])
    pw = np.asarray(con["lines"]["take-off distance"])
    i = 50
    wp_lbhp = (SZ.HP / (SZ.LB * G0)) / pw[i]
    top = ws[i] * SZ.PSF * wp_lbhp / req["cl_max_takeoff"]
    s_to = 1.66 * (4.9 * top + 0.009 * top**2) / SZ.FT
    assert s_to == pytest.approx(req["takeoff_distance"], rel=1e-6)
    # cruise power rises with wing loading beyond the minimum-drag loading
    cr = np.asarray(con["lines"]["cruise speed"])
    assert cr[-1] > cr[np.argmin(cr)]


def test_class_i_and_breguet():
    assert SZ.class_i("general aviation, single engine", 1111.0) == pytest.approx(
        2.36 * (1111.0 * SZ.LB) ** -0.18, rel=1e-12)
    req = SZ.default_requirements() | {"range_km": 0.0, "reserve_min": 0.0}
    m0 = SZ.mission(req, 800.0, 11.0, 0.025, 0.045)
    req2 = req | {"range_km": 1000.0}
    m1 = SZ.mission(req2, 800.0, 11.0, 0.025, 0.045)
    cruise = [s for s in m1["segments"] if s["segment"].startswith("cruise")][0]
    c = req["sfc"] / 3.6e9
    assert cruise["fraction"] == pytest.approx(math.exp(-1e6 * c * G0 / (req["eta_prop_cruise"] * cruise["L_D"])))
    assert m1["fuel"] > m0["fuel"]


def test_electric_energy_linear_in_range():
    req = SZ.default_requirements() | {"engine": "electric", "reserve_min": 0.0}
    e1 = SZ.mission(req | {"range_km": 100.0}, 600.0, 9.0, 0.02, 0.035)
    e2 = SZ.mission(req | {"range_km": 200.0}, 600.0, 9.0, 0.02, 0.035)
    e3 = SZ.mission(req | {"range_km": 300.0}, 600.0, 9.0, 0.02, 0.035)
    assert e2["battery"] - e1["battery"] == pytest.approx(e3["battery"] - e2["battery"], rel=1e-9)
    assert e1["fuel"] == 0.0


# ---------------------------------------------------------------- sizing loop
def test_sizing_closes(sized):
    res, ac = sized
    assert res["converged"]
    total = res["empty_mass"] + res["payload"] + res["fuel"] + res["battery"]
    assert total == pytest.approx(res["mtow"], rel=2e-3)
    assert res["mtow"] / res["wing_area"] * G0 == pytest.approx(res["design_point"]["ws"], rel=1e-6)
    # empty fraction: Class II close to the Class I statistics
    assert abs(res["empty_fraction"] - res["class_i_empty_fraction"]) < 0.08
    # the model on disk is complete: components, gear, propulsion
    assert ac.components and ac.gear and ac.propulsion
    d = A.Aircraft.from_dict(json.loads(json.dumps(A._jsonable(ac.to_dict()))))
    assert len(d.components) == len(ac.components)


def test_sized_tail_volumes_and_cg(sized):
    res, ac = sized
    req = res["requirements"]
    vh, vv = SZ.TAIL_VOLUME[req["category"]]
    xw = WB.mac_x(ac.wing, 0.25)
    ht, vt = ac.surfaces[1], ac.surfaces[2]
    assert ht.area * (WB.mac_x(ht, 0.25) - xw) / (ac.S * ac.c) == pytest.approx(vh, rel=0.02)
    assert vt.area * (WB.mac_x(vt, 0.25) - xw) / (ac.S * ac.b) == pytest.approx(vv, rel=0.02)
    ld = WB.loading(ac, ac.components)
    assert ld["cases"]["crew + payload + full fuel"]["pct_mac"] == pytest.approx(req["cg_target"], abs=1.0)
    # the sized aircraft is statically stable at its loaded c.g.
    an = A.Analysis(ac, 50.0, 1000.0, None)
    assert an.derivatives(2.0, downwash=False)["static_margin"] > 0.05


def test_old_aircraft_files_still_load(tmp_path):
    d = A.demo_aircraft().to_dict()
    for k in ("components", "gear", "propulsion"):
        d.pop(k)
    ac = A.Aircraft.from_dict(d)
    assert ac.components == [] and ac.gear == {} and ac.propulsion == {}


# ---------------------------------------------------------------- weight and balance
def test_loading_and_wing_placement():
    ac = A.demo_aircraft()
    comps = WB.default_components(ac)
    m, x, z = WB.cg_of(comps)
    assert m == pytest.approx(sum(c["mass"] for c in comps))
    ld = WB.loading(ac, comps)
    assert ld["forward"]["x"] <= ld["aft"]["x"]
    assert ld["forward"]["pct_mac"] <= ld["cases"]["crew + payload + full fuel"]["pct_mac"] <= ld["aft"]["pct_mac"]
    ac2, comps2 = WB.place_wing(ac, comps, 30.0)
    ld2 = WB.loading(ac2, comps2)
    assert ld2["cases"]["crew + payload + full fuel"]["pct_mac"] == pytest.approx(30.0, abs=0.05)


def test_scissor_plot_trends():
    ac = A.demo_aircraft()
    sc = WB.scissor(ac, sm_min=0.05, cl_land=1.9, factors=[0.6, 1.0, 1.4])
    r = sc["rows"]
    assert r[0]["pct_aft"] < r[1]["pct_aft"] < r[2]["pct_aft"]          # bigger tail: aft limit moves aft
    assert r[0]["pct_fwd"] > r[1]["pct_fwd"] > r[2]["pct_fwd"]          # and the forward limit forward
    an = A.Analysis(ac, sc["speed"], 0.0, None)
    assert r[1]["x_np"] == pytest.approx(an.derivatives(2.0, downwash=False)["x_np"], rel=1e-9)
    # smallest tail that covers a c.g. range inside the window at factor 1
    lo, hi = r[1]["pct_fwd"] + 1.0, r[1]["pct_aft"] - 1.0
    need = WB.min_tail(sc, lo, hi)
    assert need is not None and need < r[1]["Sh_S"]


def test_gear_suggestion_meets_criteria():
    ac = A.demo_aircraft()
    ld = WB.loading(ac)
    g = WB.suggest_gear(ac, ld, prop_diameter=1.8)
    ac.gear = g
    chk = WB.gear_checks(ac, ld, prop_diameter=1.8)
    assert chk["tipback_deg"] >= 15.0
    assert chk["turnover_deg"] <= 63.0
    assert chk["rotation_deg"] >= 12.0
    assert chk["prop_clearance"] >= 0.18
    # nose share: lever rule
    x = ld["aft"]["x"]
    assert chk["nose_load_aft"] == pytest.approx((g["x_main"] - x) / (g["x_main"] - g["x_nose"]))
