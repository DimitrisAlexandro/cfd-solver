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

from aerolab.mesh import postview as V  # noqa: E402

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
    from aerolab.fv3d.mesh3d import FVMesh3D
    from aerolab.fv3d.solver3d import Solver3D
    from aerolab.fv3d.wingmesh import wing_mesh
    from aerolab.fv3d import wing3d
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


@pytest.mark.skipif(not (FOIL / "results.vtu").exists(), reason="no saved 2D case")
def test_2d_wake_survey_recovers_the_surface_drag():
    from aerolab.mesh import wakeview as W
    c = V.Case2D(FOIL)
    wk = W.Wake2D(c)
    rows = wk.survey((0.1, 1.0))
    for r in rows:
        assert abs(r["Cd_wake"] / wk.Cd_surface - 1) < 0.08
    r = wk.rake(0.5)
    assert r["u"].min() < 0.9 and abs(r["u"][0] - 1) < 0.1
    assert "u_deficit" in c.fields()


def test_3d_wake_tools(wing_fields):
    from aerolab.mesh import wakeview as W
    c = V.Case3D(wing_fields)
    wk = W.Wake3D(c)
    f = wk.plane(0.5, nt=61, nz=81)
    # the shed circulation equals the circulation around the plane's edge (Stokes)
    t, z = f["t"][:, 0], f["z"][0]
    line = (np.trapezoid(f["w"][-1], z) - np.trapezoid(f["v"][:, -1], t)
            - np.trapezoid(f["w"][0], z) + np.trapezoid(f["v"][:, 0], t))
    assert abs(np.sum(f["xi_p"] * f["dA"]) - line) < 0.05 * abs(line) + 1e-4
    rows = wk.survey((0.5, 1.0))
    for r in rows:
        assert all(np.isfinite(r[k]) for k in ("CD", "CDp", "CDi", "Gamma", "core_z"))
        assert r["CDi"] >= 0
    W.plan_figure(wk, "loss", 3.0)
    W.plan_figure(wk, "xi", 3.0, mode="slice")
    W.planes_figure(wk, (0.5, 1.5), "xi")


def test_maskell_matches_a_vortex_pair():
    """induced drag of a vortex pair with cores: 1/2 int psi xi equals the analytic
    kinetic energy of two Lamb-Oseen vortices mirrored in z = 0"""
    from aerolab.mesh import wakeview as W
    G, b, rc = 1.0, 1.0, 0.1
    t = np.linspace(-1.5, 1.5, 241)
    z = np.linspace(0.0, 3.0, 241)
    T, Z = np.meshgrid(t, z, indexing="ij")
    r2 = T ** 2 + (Z - b) ** 2
    xi = -G / (np.pi * rc ** 2) * np.exp(-r2 / rc ** 2)        # the +z tip vortex of a lifting wing
    dA = np.outer(np.gradient(t), np.gradient(z))
    Di = W.Wake3D._maskell(T, Z, xi, dA)
    # kinetic energy per unit length of a Lamb-Oseen pair (separation 2b) is
    # G^2/(2 pi) (ln(2b/rc) + (gamma - ln 2)/2); the half plane holds half of it
    E_half = G ** 2 / (4 * np.pi) * (np.log(2 * b / rc) + 0.5 * (np.euler_gamma - np.log(2)))
    assert abs(Di / E_half - 1) < 0.02


def test_whole_wake_scene(wing_fields):
    from aerolab.mesh import wakeview as W
    wk = W.Wake3D(V.Case3D(wing_fields))
    vol = W.wake_volume(wk, 3.0, "Coarse")
    for f in ("loss", "nut_ratio", "Q", "Tu"):
        assert vol[f].shape == vol["X"].shape and np.isfinite(vol[f]).all()
    assert vol["nut_ratio"].max() > 1.0
    lines = W.streamlines(wk, 3.0)
    assert lines.shape[2] == 3 and np.isfinite(lines).all() and (lines[:, -1, 0] > 2.5).all()
    xs, ys, zs = W.wake_mesh_lines(wk, 3.0)
    assert len(xs) > 10
    fig = W.wake_scene(wk, vol, W.LAYERS, 3.0, lines=lines)
    assert len(fig.data) >= 4
