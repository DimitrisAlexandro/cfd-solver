"""Fast regression tests for the FV solver (full validation: src/validate_*.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerolab.core.fvmesh import FVMesh, rect_quad, rect_tri  # noqa: E402
from aerolab.validation import validate_channel as vch                  # noqa: E402
from aerolab.validation import validate_cavity as vca                   # noqa: E402
from aerolab.validation import validate_cylinder as vcy                 # noqa: E402


@pytest.mark.parametrize("mk", [lambda: rect_quad(12, 9), lambda: rect_tri(0.08)])
def test_mesh_geometry(mk):
    m = FVMesh(*mk())
    for k in range(2):     # every cell is closed
        assert np.abs(m.sum_to_cells(m.Sf[:, k], m.bSf[:, k])).max() < 1e-12
    f = lambda x: 2 * x[:, 0] - 3 * x[:, 1] + 1
    assert np.abs(m.grad(f(m.xc), f(m.bxf)) - [2, -3]).max() < 1e-9          # LSQ exact
    g = m.grad_gg(f(m.xc), f(m.bxf), grad_skew=np.tile([2.0, -3.0], (m.ncells, 1)))
    assert np.abs(g - [2, -3]).max() < 1e-9                                    # GG + skew exact


def test_channel_second_order_quads():
    e = [vch.run("quad", n)["err"] for n in (10, 20)]
    assert np.log(e[0] / e[1]) / np.log(2) > 1.8


def test_channel_triangles():
    r10, r20 = vch.run("tri", 10), vch.run("tri", 20)
    assert np.log(r10["err"] / r20["err"]) / np.log(2) > 1.8
    assert abs(r20["dpdx"] / r20["dpdx_exact"] - 1) < 0.005     # pressure gradient < 0.5 %


def test_mass_conservation():
    r = vch.run("tri", 10)
    s = r["solver"]
    assert np.abs(s.m.sum_to_cells(s.F, s.Fb)).max() < 1e-12


def test_cavity_re100_vs_ghia():
    s, it, _ = vca.solve(100, FVMesh(*rect_quad(41, 41)))
    iu = vca.interpolator(s, 0)
    u = iu(np.full_like(vca.GHIA_Y, 0.5), vca.GHIA_Y)
    assert np.abs(u - vca.GHIA_U[100]).max() < 0.02


def test_cylinder_re40():
    r, _, _ = vcy.run(40, 0.7)
    assert r["Cd"] == pytest.approx(1.51, rel=0.03)
    assert 2.1 < r["L/D"] < 2.4
    assert abs(r["theta"] - 53.7) < 1.5
    assert abs(r["Cl"]) < 0.01
