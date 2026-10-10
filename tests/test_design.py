"""Design tools: CST, inverse design, wing sizing, drag-rise post-processing, optimizer shape."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import airfoil_geom as ag  # noqa: E402
import cst  # noqa: E402

VAL = Path(__file__).resolve().parents[1] / "validation"


@pytest.mark.parametrize("spec", ["naca0012", "naca2412", str(VAL / "reference" / "rae2822.dat")])
def test_cst_fit(spec):
    au, al, err = cst.fit(ag.sharp_outline(spec), 8)
    assert err < 5e-4
    t, xt, _ = cst.thickness(au, al)
    assert t == pytest.approx(0.12 if "rae" not in spec else 0.121, abs=0.003)


def test_inverse_design_recovers_naca2412():
    """target: NACA 2412 Cp at alpha 3; start: NACA 0012 -> the 2412 shape"""
    import design_inverse as DI
    x, cp, side, shape = DI.target_from_airfoil("naca2412", 3.0)
    s0 = cst.fit(ag.sharp_outline("naca0012"), 8)[:2]
    r = DI.design(x, cp, side, 3.0, s0)
    assert r["rms"] < 1e-3
    assert np.abs(cst.outline(r["au"], r["al"]) - cst.outline(*shape)).max() < 1e-3


def test_inverse_design_follows_a_modified_target():
    """less suction near the leading edge -> the design has it (smooth shape)"""
    import design_inverse as DI
    x, cp, side, shape = DI.target_from_airfoil("naca2412", 4.0)
    up = side == 1
    cp_t = cp.copy()
    cp_t[up] += 0.3 * np.exp(-((x[up] - 0.1) / 0.1) ** 2)
    r = DI.design(x, cp_t, side, 4.0, shape)
    assert r["rms"] < 0.06          # an edited target need not be exactly realisable
    k = up & (np.abs(x - 0.1) < 0.05)
    xs, cps = r["x"][r["side"] == 1], r["cp"][r["side"] == 1]
    o = np.argsort(xs)
    assert np.interp(0.1, xs[o], cps[o]) > np.interp(0.1, np.sort(x[up]), cp[up][np.argsort(x[up])]) + 0.15


def test_wing_sizing_stall_requirement():
    """the sized wing really stalls at the requested speed"""
    import design_wing_sizing as WS
    import wing as W
    pol = json.loads((VAL / "phase3_n2412_polar.json").read_text())
    sp = W.SpanPolars([(pol["re"], W.SectionPolar.from_results(pol["results"]))])
    rho_c, mu_c, a_c, _, _ = W.isa(1500.0)
    m = 750.0 * WS.G
    d = WS.eval_design(m, 27.0, 60.0, rho_c, mu_c, 60.0 / a_c, sp, 10.0, 0.5, -2.0, 0.0, 10.0)
    assert m == pytest.approx(0.5 * 1.225 * 27.0**2 * d["S"] * d["CLmax"], rel=0.01)
    assert d["span"] == pytest.approx(np.sqrt(10.0 * d["S"]), rel=1e-6)
    assert 20 < d["L_D"] < 60


def test_drag_rise_criteria():
    import design_drag_rise as DR
    M = np.arange(0.5, 0.86, 0.04)
    Cd = 0.009 + 0.5 * np.maximum(M - 0.7, 0) ** 2
    res = [{"mach": m, "Cd": c} for m, c in zip(M, Cd)]
    s = DR._summary("naca0012", 6e6, 0.3, 0.12, res)
    # dCd/dM = M - 0.7 = 0.1 at 0.8 (midpoint interpolation); +20 counts: (M-0.7)^2 = 0.004
    assert s["M_dd_douglas"] == pytest.approx(0.8, abs=0.02)
    assert s["M_dd_boeing"] == pytest.approx(0.7 + np.sqrt(0.004), abs=0.01)
    assert s["M_dd_korn_conventional"] == pytest.approx(0.87 - 0.03 - 0.12)
    # a point where the target Cl was not reached (shock stall) is excluded from M_dd
    res2 = [dict(r, Cl=0.3) for r in res] + [{"mach": 0.9, "Cd": 0.002, "Cl": 0.1}]
    s2 = DR._summary("naca0012", 6e6, 0.3, 0.12, res2)
    assert not s2["points"][-1]["cl_reached"]
    assert s2["M_dd_douglas"] == pytest.approx(s["M_dd_douglas"])


def test_optimizer_shape_is_smooth_and_exact_at_zero():
    import design_optimize as DO
    s = DO.Shape("naca2412", n_var=4)
    au, al = s.coeffs(np.zeros(8))
    assert np.allclose(au, s.au0) and np.allclose(al, s.al0)
    au, al = s.coeffs(np.r_[0.02, 0, 0, 0, 0, 0, 0, 0])
    d = au - s.au0
    assert d[0] == pytest.approx(0.02) and np.all(np.diff(d) <= 1e-12)   # smooth hat
