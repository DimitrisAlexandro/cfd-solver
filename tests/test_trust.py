"""Trust layer: validation envelope, point rules, grid convergence index."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import trust as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_gci_recovers_second_order_exactly():
    h = np.array([1.0, 1.5, 2.25])
    f = 1.0 + 0.3 * h ** 2
    g = T.gci(f, h)
    assert g["p"] == pytest.approx(2.0, abs=1e-9)
    assert g["phi_ext"] == pytest.approx(1.0, abs=1e-12)
    assert not g["oscillatory"]


def test_gci_unequal_ratios():
    h = np.array([1.0, 1.3, 2.0])
    f = 2.0 - 0.1 * h ** 1.6
    g = T.gci(f, h)
    assert g["p"] == pytest.approx(1.6, abs=1e-6)
    assert g["phi_ext"] == pytest.approx(2.0, abs=1e-9)


def test_gci_flat_plate_record():
    """step C flat plate Cf(x=0.97) on 3264/13056/52224 cells -> SA reference 0.002729"""
    g = T.gci_from_cells([0.002757, 0.002735, 0.002730], [3264, 13056, 52224])
    assert g["r21"] == pytest.approx(2.0)
    assert g["phi_ext"] == pytest.approx(0.002729, rel=5e-4)
    assert g["gci_fine"] < 0.001


def test_gci_oscillatory_and_converged():
    g = T.gci([1.0, 1.01, 0.995], [1.0, 1.4, 2.0])
    assert g["oscillatory"]
    g = T.gci([1.0, 1.0, 1.0], [1.0, 1.4, 2.0])
    assert g["converged"] and g["gci_fine"] == 0.0


def test_inside_envelope_is_green():
    a = T.assess_point(3e6, 0.0, 4.0, transition="free")
    assert a.level == "green" and a.distance == 0
    assert a.case == "free_naca0012_3e6"
    assert a.cd_band == pytest.approx(0.03)


def test_distance_grows_band_and_level():
    near = T.assess_point(1.2e7, 0.0, 4.0, transition="free")
    far = T.assess_point(5e7, 0.0, 4.0, transition="free")
    assert 0 < near.distance < far.distance
    assert near.cd_band < far.cd_band
    assert far.level == "red"


def test_low_re_and_unvalidated_re():
    assert T.assess_point(1e5, 0.0, 4.0).level == "red"            # case itself has 47 % Cd error
    assert T.assess_point(3e5, 0.0, 2.0).level == "green"
    r = T.assess_point(5e4, 0.0, 2.0)
    assert r.level == "red" and any("no validation data" in t for _, t in r.reasons)


def test_point_rules():
    base = dict(transition="off")
    assert T.assess_point(6e6, 0.0, 16.0, status="unsteady", **base).level == "red"
    assert T.assess_point(6e6, 0.0, 14.0, alpha_clmax=12.0, **base).level == "amber"
    assert T.assess_point(6e6, 0.0, 8.0, alpha_clmax=12.0, **base).level == "green"
    assert T.assess_point(6e6, 0.0, 4.0, refine=0.7, **base).level == "amber"
    assert T.assess_point(6e6, 0.0, 4.0, yplus_max=3.0, **base).level == "amber"
    assert T.assess_point(6e6, 0.0, 4.0, t_c=0.21, **base).level == "amber"
    assert T.assess_point(6e6, 0.55, 2.0, prandtl_glauert=True, **base).level == "red"
    tr = T.assess_point(6.5e6, 0.734, 2.79, m_max=1.3, **base)
    assert tr.case == "rae2822_case9" and tr.cd_band >= 0.2


def test_grid_study_tightens_band():
    g = {"Cd": T.gci_from_cells([0.0122, 0.01216, 0.01215], [15e3, 29e3, 55e3])}
    a = T.assess_point(6e6, 0.0, 10.0, transition="off", gci=g)
    assert any("Grid study" in t for _, t in a.reasons)


def test_assess_polar_file_and_rerun_args(tmp_path):
    d = {"info": {"name": "NACA 2412", "source": "naca2412", "max_thickness": 0.12}, "re": 3e6,
         "settings": {"mach": 0.0, "transition": "free transition (e^N, N_crit 9)", "refine": 1.0,
                      "airfoil": "naca2412", "re": 3e6, "ncrit": 9.0, "transition_mode": "free",
                      "solver": "auto", "max_iter": 6000},
         "results": [{"alpha": a, "Cl": cl, "Cd": 0.006 + 0.0004 * a, "Cm": -0.05, "status": "converged"}
                     for a, cl in ((0, 0.25), (6, 0.9), (12, 1.5), (16, 1.4))]}
    ass = T.assess_polar(d)
    lv = {p["alpha"]: p["level"] for p in ass["points"]}
    assert lv[0] == "green" and lv[16] in ("amber", "red")
    assert sum(ass["counts"].values()) == 4
    args = T.rerun_args(d["settings"], d["info"], tmp_path / "L0.7", "4,8", 0.7)
    assert args[:3] == ["airfoil.py", "polar", "naca2412"] and "--ncrit" in args and "0.7" in args
    # wing-cache polar: a plain list of rows
    rows = json.loads((ROOT / "data" / "wing_polars" / "naca2412_Re2.2e06_M0.12_L1_free.json").read_text())
    ass2 = T.assess_polar(rows, re=2.2e6, mach=0.12, transition="free")
    assert len(ass2["points"]) == len(rows)


def test_tool_notes():
    n = T.tool_notes("aeroelastic", mach=0.45, antisymmetric=True)
    assert n[0][0] == "amber" and any(l == "red" for l, _ in n)
    assert T.tool_notes("wing", aspect_ratio=2.0)[-1][0] == "amber"
    for c in T.CASES:                                  # every case points to a real report
        assert c.re[0] < c.re[1] and c.alpha[0] <= c.alpha[1]
        assert not c.report or (ROOT / "validation" / c.report).exists(), c.report
