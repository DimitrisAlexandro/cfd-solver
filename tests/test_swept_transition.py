"""Falkner-Skan-Cooke solutions and swept-wing transition criteria."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fsc  # noqa: E402
import swept_transition as st  # noqa: E402
import transition as T  # noqa: E402


def test_falkner_skan_reference_values():
    """Blasius and Hiemenz (White, Viscous Fluid Flow, table 4-1)"""
    b = fsc.integrals(0.0)
    assert abs(b["fpp0"] - 0.4696) < 2e-4
    assert abs(b["H"] - 2.591) < 2e-3
    assert abs(b["k"] - 2.193) < 0.01            # Re_V,max / Re_theta (Langtry-Menter's 2.193)
    h = fsc.integrals(1.0)
    assert abs(h["fpp0"] - 1.2326) < 2e-4
    assert abs(h["H"] - 2.216) < 2e-3
    s = fsc.integrals(-0.19)
    assert s["H"] > 3.3 and s["fpp0"] < 0.1       # approaching separation


def test_cooke_spanwise_equals_chordwise_for_zero_pressure_gradient():
    """beta = 0: g = f' exactly, so there is no crossflow"""
    eta, fp, fpp, g = fsc.profile(0.0)
    assert np.abs(g - fp).max() < 1e-6
    assert fsc.crossflow(0.0, 1.0)[0] < 1e-6
    # crossflow grows with the favourable gradient and the sweep
    assert fsc.crossflow(1.0, 1.0)[0] > fsc.crossflow(0.3, 1.0)[0] > 0


def test_attachment_line_reynolds_number_cylinder():
    """circular cylinder of radius R: U_e = 2 V_n sin(s/R) -> dU_e/ds = 2 V_n / R,
    Rbar = V sin(L) / sqrt(nu 2 V cos(L) / R)"""
    R, sweep, re_c = 0.1, 40.0, 1e6
    s = np.linspace(0, 0.05, 30)
    ue = 2 * np.sin(s / R)
    surf = {"upper": (s, s, ue), "lower": (s, s, ue)}
    rbar, due = st.attachment_line(surf, sweep, re_c)
    lam = math.radians(sweep)
    exact = math.sin(lam) / math.sqrt(2 * math.cos(lam) / R / re_c)
    assert abs(due - 2 / R) < 0.01 * 2 / R
    assert abs(rbar - exact) < 0.01 * exact


def test_unswept_equals_two_dimensional_e_n():
    r = st.analyse("naca0012", 0.0, 6e6, 2.0)
    assert r["attachment_line"]["Rbar"] == 0.0
    for side in ("upper", "lower"):
        assert r[side]["chi_max"] == 0.0
        assert "crossflow" not in r[side]["mechanism"]
    assert 0.1 < r["upper"]["x_tr"] < r["lower"]["x_tr"] <= 1.0


def test_sweep_moves_transition_forward_and_crossflow_takes_over():
    xs, mech = [], []
    for sw in (0.0, 35.0, 50.0):
        r = st.analyse("naca0012", sw, 1e7, 2.0)
        xs.append(r["upper"]["x_tr"])
        mech.append(r["upper"]["mechanism"])
    assert xs[0] > xs[1] > xs[2]
    assert "crossflow" in mech[2]


def test_attachment_line_contamination_and_instability():
    r = st.analyse("naca0012", 45.0, 3e7, 0.0, contaminated=True)
    al = r["attachment_line"]
    assert al["Rbar"] > st.RBAR_CONTAMINATION
    assert al["turbulent"]
    assert r["upper"]["x_tr"] < 0.01 and r["lower"]["x_tr"] < 0.01
    r2 = st.analyse("naca0012", 45.0, 3e7, 0.0, contaminated=False)
    assert r2["attachment_line"]["turbulent"] == (r2["attachment_line"]["Rbar"] > st.RBAR_INSTABILITY)


def test_turbulence_level_moves_ts_transition_forward():
    a = st.analyse("naca0012", 0.0, 6e6, 0.0, tu=0.05)
    b = st.analyse("naca0012", 0.0, 6e6, 0.0, tu=0.5)
    assert a["ncrit"] == pytest.approx(T.ncrit_from_turbulence(0.05))
    assert b["ncrit"] < a["ncrit"]
    assert b["upper"]["x_tr"] < a["upper"]["x_tr"]


def test_wing_spanwise_transition():
    """quasi-3D wing: stations along the span, leading-edge sweep from the
    quarter-chord sweep and the taper, crossflow on a 30-degree wing at
    Re 1e7, Tollmien-Schlichting on the same wing unswept"""
    import wing as W
    out = {}
    for sweep in (0.0, 30.0):
        w = W.Wing(30, 6.0, 1.8, sweep, 0, 0, 1)
        sol = W.solve(w, 3.0)
        re = 1e7 / w.mac * w.c
        out[sweep] = st.wing_transition(w, sol, re, "naca0012", n_eval=3)
    t30 = out[30.0]
    assert t30["sweep_le"] > 30.0                       # taper adds LE sweep
    assert all("crossflow" in q["mech_upper"] for q in t30["stations"])
    assert all("crossflow" not in q["mech_upper"] for q in out[0.0]["stations"])
    assert t30["laminar_fraction"] < out[0.0]["laminar_fraction"]
    # Rbar grows with the chord (root > tip)
    r = [q["Rbar"] for q in t30["stations"]]
    assert r[0] > r[-1]
