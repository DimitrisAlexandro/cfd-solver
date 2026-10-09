"""Plain trailing-edge flap geometry (airfoil_geom.deflect_flap) and its spec tag."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import airfoil_geom as ag  # noqa: E402
import panel  # noqa: E402


def test_spec_roundtrip_and_stem():
    s = ag.flap_spec("naca2412", 0.25, 20)
    assert s == "naca2412@flap=25c20" and ag.parse_flap(s) == ("naca2412", (0.25, 20.0))
    s = ag.flap_spec("/data/rae.dat", 0.3, 12.5)
    assert ag.parse_flap(s) == ("/data/rae.dat", (0.3, 12.5))
    assert ag.section_stem(s) == "rae@flap=30c12p5"          # no dots: unambiguous file names
    assert ag.flap_spec("naca0012", 0.25, 0) == "naca0012"
    assert ag.section_stem("naca0012") == "naca0012"


@pytest.mark.parametrize("delta", [-10.0, 10.0, 30.0])
def test_geometry_hinge_rotation(delta):
    cf = 0.25
    o = ag.sharp_outline(ag.flap_spec("naca0012", cf, delta))
    te = 0.5 * (o[0] + o[-1])
    d = math.radians(delta)
    assert te[0] == pytest.approx(1 - cf + cf * math.cos(d), abs=0.004)
    assert te[1] == pytest.approx(-cf * math.sin(d), abs=0.004)
    loop, info = ag.load_airfoil(ag.flap_spec("naca0012", cf, delta))   # passes the geometry checks
    assert info["max_thickness"] == pytest.approx(0.12, abs=0.002)
    # the forward part is untouched
    clean = ag.sharp_outline("naca0012")
    i = np.argmin(np.abs(clean[:, 0] - 0.4) + 10 * (clean[:, 1] < 0))
    j = np.argmin(np.abs(o[:, 0] - 0.4) + 10 * (o[:, 1] < 0))
    assert o[j, 1] == pytest.approx(clean[i, 1], abs=2e-3)


def test_flap_lift_against_thin_airfoil_theory():
    """inviscid Delta Cl = 2 pi tau delta (tau = 0.609 at cf 0.25), x ~(1 + 0.77 t/c) for thickness"""
    cf = 0.25
    th = math.acos(2 * cf - 1)
    tau = 1 - (th - math.sin(th)) / math.pi
    base = panel.solve([ag.sharp_outline("naca2412")], 2.0)["Cl"]
    for d in (10.0, 20.0):
        dcl = panel.solve([ag.sharp_outline(ag.flap_spec("naca2412", cf, d))], 2.0)["Cl"] - base
        assert dcl / (2 * math.pi * tau * math.radians(d)) == pytest.approx(1.09, abs=0.05)
