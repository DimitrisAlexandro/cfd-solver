"""Results viewer (postview): readers and figures for 2D cases and 3D wing fields."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import postview as V  # noqa: E402

FOIL = ROOT / "cases" / "foil_a4"


@pytest.mark.skipif(not (FOIL / "results.vtu").exists(), reason="no saved 2D case")
def test_2d_case_reads_and_renders():
    c = V.Case2D(FOIL)
    assert c.has_results and len(c.cells) == len(c.centres)
    assert "p" in c.fields() and len(c.values("U")) == len(c.cells)
    for view in V.VIEWS:
        V.figure_2d(c, "p", view)
    V.figure_2d(c, None, "Leading edge")
    V.figure_2d(c, "U", "Near", streamlines=True)
    assert c.walls() and c.history() is not None


@pytest.fixture(scope="module")
def wing_fields(tmp_path_factory):
    from fv3d.mesh3d import FVMesh3D
    from fv3d.solver3d import Solver3D
    from fv3d.wingmesh import wing_mesh
    from fv3d import wing3d
    pm, info = wing_mesh("naca0012", semi_span=1.0, level=0.25, re=1e6, farfield=6.0)
    m = FVMesh3D(pm)
    a = math.radians(4.0)
    U = (math.cos(a), math.sin(a), 0.0)
    s = Solver3D(m, {"wing": {"type": "wall"}, "symmetry": {"type": "symmetry"},
                     "farfield": {"type": "farfield", "U": U}}, 1e-6, U0=U, turbulence="sa")
    for _ in range(10):
        s.iterate()
    c = wing3d.coefficients(s, info, 4.0)
    f = tmp_path_factory.mktemp("wing") / "fields_a4.npz"
    wing3d.save_fields(f, s, info, pm, 4.0, c)
    return f


def test_3d_fields_round_trip_and_render(wing_fields):
    f = wing_fields
    found = V.find_results(f.parent)
    assert found and found[0]["kind"] == "3d"
    c = V.Case3D(f)
    assert c.info["alpha"] == 4.0
    for fld in c.fields():
        assert np.isfinite(c.values(fld)).all()
    l = c.layer_index(0.5)
    P2, cells, ids = c.span_cut(l)
    assert len(cells) == len(ids) > 0 and (c.layer[ids] == l).all()
    tri, v = c.surface("Cp")
    assert len(tri) == len(v) > 0 and np.isfinite(v).all()
    V.surface_figure(c, "Cp", edges=True, sym_mesh=True)
    V.xcut_figure(c, 1.2, "vorticity_x", zr=(0.5, 1.4), yr=(-0.3, 0.3))
    Z, Y, val, W, Vv, solid = c.x_cut(0.3, "U_mag", zr=(0.0, 0.5), yr=(-0.2, 0.2), nz=40, ny=30)
    assert solid.any() and np.isfinite(val[~solid]).all()
