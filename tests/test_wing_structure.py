"""Wing structure: box sizing, stiffness, beam modes, divergence, reversal, flutter."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerolab.aircraft import aircraft as A  # noqa: E402
from aerolab.aircraft import wing_structure as WS  # noqa: E402


# ---------------------------------------------------------------- closed forms
def test_beam_modes_uniform_cantilever():
    L, EI, m, GJ, Ip = 5.0, 2e5, 6.0, 1.5e5, 0.4
    y = np.linspace(0, L, 41)
    wb, wt = WS.beam_modes(y, np.full_like(y, EI), np.full_like(y, m), np.full_like(y, GJ), np.full_like(y, Ip))
    assert wb == pytest.approx(1.8751**2 * math.sqrt(EI / (m * L**4)), rel=1e-4)
    assert wt == pytest.approx(math.pi / 2 * math.sqrt(GJ / (Ip * L**2)), rel=2e-3)


def test_divergence_uniform_wing():
    L, GJ, c, e, a = 5.0, 1.5e5, 1.2, 0.12, 4.8
    y = np.linspace(0, L, 81)
    q = WS.divergence(y, np.full_like(y, GJ), np.full_like(y, c), np.full_like(y, e), a)
    assert q == pytest.approx(math.pi**2 * GJ / (4 * L**2 * c**2 * e * a), rel=1e-3)
    assert WS.divergence(y, np.full_like(y, GJ), np.full_like(y, c), np.full_like(y, -0.05), a) == math.inf


def test_aileron_reversal_uniform_wing_full_span():
    """closed form (strip theory, uniform clamped wing, full-span aileron):
    a K [(sec(lam l) - 1)/lam^2 - l^2/2] + cl_d l^2/2 = 0,  K = (cl_d + cm_d/e)/a,
    lam^2 = q c^2 e a / GJ"""
    L, GJ, c, e, a, cf = 5.0, 1.5e5, 1.2, 0.12, 4.8, 0.25
    cl_d, cm_d = WS.flap_coeffs(cf)
    cl_d *= a / (2 * math.pi)
    K = (cl_d + cm_d / e) / a

    def f(lam):
        return a * K * ((1 / math.cos(lam * L) - 1) / lam**2 - L**2 / 2) + cl_d * L**2 / 2
    lam = brentq(f, 1e-3, 0.999 * math.pi / (2 * L))
    q_exact = lam**2 * GJ / (c**2 * e * a)
    y = np.linspace(0, L, 161)
    q_d = WS.divergence(y, np.full_like(y, GJ), np.full_like(y, c), np.full_like(y, e), a)
    q_r, qs, ef = WS.aileron_reversal(y, np.full_like(y, GJ), np.full_like(y, c), np.full_like(y, e), a,
                                      0.0, 1.0, cf, 0.98 * q_d)
    assert q_r == pytest.approx(q_exact, rel=5e-3)
    assert ef[0] == pytest.approx(1.0, abs=0.01) and ef[-1] < 0


def test_flap_thin_airfoil():
    cl, cm = WS.flap_coeffs(0.5)                       # half-chord flap: theta_h = pi/2
    assert cl == pytest.approx(2 * (math.pi / 2 + 1))
    assert cm == pytest.approx(-0.5)


def test_flutter_typical_section_textbook():
    """Hodges & Pierce's typical section (a -1/5, x_a 1/10, mu 20, r^2 6/25,
    sigma 2/5): flutter at U/(b w_a) ~ 2.2 with unsteady aerodynamics
    (Peters / Wagner in Aeroelasticity.jl), below the divergence speed
    sqrt(mu r^2 / (1 + 2a)) = 2.83"""
    V, w, _ = WS.flutter_typical_section(20.0, -0.2, 0.1, 0.24, 0.4)
    assert V == pytest.approx(2.2, abs=0.05)
    assert 0.4 < w < 1.0
    assert V < math.sqrt(20 * 0.24 / (1 - 0.4))
    # c.g. moved aft: flutter earlier
    V2, _, _ = WS.flutter_typical_section(20.0, -0.2, 0.2, 0.24, 0.4)
    assert V2 < V


def test_theodorsen_limits():
    assert WS.theodorsen(1e-9) == pytest.approx(1.0)
    C = WS.theodorsen(50.0)
    assert C.real == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------- sizing
def _loads(ac, M, V, T, n_case=1.0, x_axis=0.35):
    eta = np.linspace(0, 1, 21)
    one = lambda v: (v * (1 - eta) ** 2).tolist()
    st = {"eta": eta.tolist(), "y": (eta * ac.wing.semi).tolist(), "shear": one(V), "bending": one(M),
          "torsion": one(T)}
    return {"x_axis": x_axis, "wing_mass_frac": 0.1, "vn": {"Vd": 80.0},
            "cases": [{"name": "case", "n": n_case, "stations": st}]}


def test_box_sizing_formulae():
    ac = A.demo_aircraft()
    M, V, T = 2.0e4, 8e3, 1.5e3
    sets = {"front_spar": 0.2, "rear_spar": 0.6}
    st = WS.size(ac, _loads(ac, M, V, T, x_axis=0.4), sets)       # x_axis = shear centre: no transfer
    mat = WS.MATERIALS[st["settings"]["material"]]
    he = 0.92 * st["h_mean"][0]
    assert st["A_upper_cap"][0] == pytest.approx(1.5 * M / (0.8 * mat["F_c"]) / he, rel=1e-9)
    assert st["A_lower_cap"][0] == pytest.approx(1.5 * M / mat["F_t"] / he, rel=1e-9)
    Ab = st["A_box"][0]
    ts = max(1.5 * T / (2 * Ab * 0.8 * mat["F_s"]), mat["min_gauge"])
    assert st["t_skin"][0] == pytest.approx(ts, rel=1e-9)
    # Bredt
    w, hf, hr, tw = st["w"][0], st["h_f"][0], st["h_r"][0], st["t_web"][0]
    assert st["GJ"][0] == pytest.approx(4 * Ab**2 * mat["G"] / (2 * w / ts + (hf + hr) / tw), rel=1e-9)
    # negative bending swaps the critical allowables
    st2 = WS.size(ac, _loads(ac, -M, V, T, x_axis=0.4), sets)
    assert st2["A_upper_cap"][0] == pytest.approx(st["A_lower_cap"][0], rel=1e-9)
    # stiffer, lighter material in tension: CFRP caps smaller
    st3 = WS.size(ac, _loads(ac, M, V, T, x_axis=0.4), sets | {"material": "CFRP (UD caps, ±45 skins)"})
    assert st3["A_upper_cap"][0] < st["A_upper_cap"][0]


def test_deflection_uniform_moment():
    """constant M and EI: tip deflection M L^2 / (2 EI)"""
    ac = A.demo_aircraft()
    st = WS.size(ac, _loads(ac, 1e4, 0.0, 0.0))
    st = dict(st)
    st["EI"] = np.full_like(st["EI"], 3e5)
    L = st["y"][-1]
    case = {"name": "m", "n": 1.0, "stations": {"eta": st["eta"].tolist(), "bending": [2e3] * len(st["eta"]),
                                                "torsion": [0.0] * len(st["eta"]), "shear": [0.0] * len(st["eta"])}}
    d = WS.deflection(st, case)
    assert d["tip_deflection"] == pytest.approx(2e3 * L**2 / (2 * 3e5), rel=1e-3)


def test_demo_chain_plausible():
    ac = A.demo_aircraft()
    an = A.Analysis(ac, 60.0, 0.0, None)
    from aerolab.aircraft import loads as LD
    res = LD.run(ac, 60.0, "normal", 0.0, None, None, log=lambda *a: None)
    r = WS.run(ac, res, log=lambda *a: None)
    assert 0.06 * ac.mass < r["wing_mass"] < 0.16 * ac.mass                 # light aircraft: ~8-13 % MTOW
    assert 0.0 < r["tip_deflection_1g"] < 0.05 * ac.b
    ae = r["aeroelastic"]
    assert ae["flutter"]["V_flutter"] > 1.2 * ae["V_D"]
    assert ae["V_divergence_eas"] > ae["V_D"]
    assert ae["flutter"]["f_torsion_hz"] > ae["flutter"]["f_bending_hz"]
    _ = an
