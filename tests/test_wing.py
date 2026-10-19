"""Phase 6: quasi-3D wing tool (vortex lattice + section polars)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerolab.aircraft import wing as W  # noqa: E402
from aerolab.validation import validate_wing as vw  # noqa: E402

VAL = Path(__file__).resolve().parents[1] / "validation"


def test_elliptic_wing_span_efficiency():
    for AR in (4, 10):
        a, r = vw.cla(vw.elliptic(AR))
        assert r["e"] == pytest.approx(1.0, abs=0.003)
    a, _ = vw.cla(vw.elliptic(40))
    assert a == pytest.approx(2 * math.pi / (1 + 2 / 40), rel=0.005)     # lifting-line limit


def test_lift_slope_vs_datcom():
    for AR, taper, sweep in ((2, 1.0, 0), (6, 1.0, 0), (5, 0.5, 30), (5, 0.5, 60)):
        a, _ = vw.cla(vw.trapezoid(AR, taper, sweep))
        assert a == pytest.approx(vw.datcom(AR, sweep, taper), rel=0.07)


def test_compressibility_goethert():
    a0, _ = vw.cla(vw.trapezoid(6, 0.5, 30), mach=0.0)
    a7, _ = vw.cla(vw.trapezoid(6, 0.5, 30), mach=0.7)
    assert a7 / a0 == pytest.approx(vw.datcom(6, 30, 0.5, 0.7) / vw.datcom(6, 30, 0.5), rel=0.02)


def test_symmetry_and_no_lift_at_zero():
    w = W.Wing(8.0, 1.2, 0.6, sweep=20, dihedral=5, n_half=20)
    r = W.solve(w, 0.0)
    assert abs(r["CL"]) < 1e-10
    r = W.solve(w, 5.0)
    g = r["span"]["gamma"]
    assert np.abs(g - g[::-1]).max() < 1e-10 * np.abs(g).max()


def test_nonlinear_matches_section_polar_and_stalls():
    """sections follow the CFD polar; with a capped polar the wing stalls"""
    pol = json.loads((VAL / "phase3_n0012_polar.json").read_text())
    sp = W.SectionPolar.from_results(pol["results"], ar=8)
    w = W.Wing(8.0, 1.0, n_half=30)
    re = np.full(len(w.c), pol["re"])
    r = W.solve(w, 8.0, W.SpanPolars([(pol["re"], sp)]), re)
    s = r["span"]
    cl2 = sp(s["alpha_eff"])[0]
    assert r["converged"] and np.abs(s["cl"] - cl2).max() < 1e-5
    assert 0.6 < r["CL"] < 0.8 and r["CDp"] > 0.008
    capped = W.SectionPolar.thin_airfoil(clmax=1.0, ar=8)
    res = W.sweep_alpha(w, np.arange(0, 21, 2.0), W.SpanPolars([(1e6, capped)]), re)
    CL = np.array([x["CL"] for x in res])
    assert 0.85 < CL.max() < 1.01                          # wing C_L,max below section c_l,max
    assert res[-1]["stall_fraction"] > 0.5
