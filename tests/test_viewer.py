"""Results files, ParaView compatibility (VTK readers) and the viewer (off-screen)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pv = pytest.importorskip("pyvista")
import case          # noqa: E402
import viewer        # noqa: E402
from io_vtk import write_vtp_lines, write_vtu, write_vtm   # noqa: E402


def test_vtk_roundtrip(tmp_path):
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0.5]], float)
    cells = [[0, 1, 2, 3], [1, 4, 2]]
    cd = {"a": np.array([1.5, -2.0]), "vec": np.array([[1.0, 2.0], [3.0, 4.0]]),
          "k": np.array([7, 8], np.int32)}
    write_vtu(tmp_path / "c.vtu", pts, cells, cell_data=cd)
    write_vtp_lines(tmp_path / "b.vtp", pts, [[0, 1], [1, 4]], {"patch_id": np.array([0, 1])})
    write_vtm(tmp_path / "m.vtm", {"cells": "c.vtu", "boundary": "b.vtp"})
    mb = pv.read(tmp_path / "m.vtm")
    g, b = mb["cells"], mb["boundary"]
    assert np.allclose(g.points[:, :2], pts)
    assert list(g.celltypes) == [pv.CellType.QUAD, pv.CellType.TRIANGLE]
    assert np.array_equal(g.cell_data["a"], cd["a"])
    assert np.array_equal(g.cell_data["vec"][:, :2], cd["vec"])
    assert np.array_equal(g.cell_data["k"], cd["k"])
    assert b.n_cells == 2 and list(b.cell_data["patch_id"]) == [0, 1]


@pytest.fixture(scope="module")
def cyl_case(tmp_path_factory):
    out = tmp_path_factory.mktemp("cyl")
    case.main(["run", "cylinder", "--re", "40", "--refine", "0.5", "--out", str(out)])
    return out


def test_case_files(cyl_case):
    for f in ("mesh.vtm", "results.vtm", "history.csv", "wall_wall.csv", "case.json"):
        assert (cyl_case / f).exists(), f
    meta = json.loads((cyl_case / "case.json").read_text())
    assert meta["converged"] and 1.4 < meta["coefficients"]["Cd"] < 1.65
    g = pv.read(cyl_case / "results.vtu")
    for k in ("U", "Cp", "vorticity", "min_angle", "nonorthogonality"):
        assert k in g.cell_data
    assert np.abs(g.cell_data["continuity_error"]).max() < 1e-10
    h = np.genfromtxt(cyl_case / "history.csv", delimiter=",", names=True)
    assert h["iteration"][-1] == meta["iterations"]


def test_viewer_offscreen(cyl_case, tmp_path):
    v = viewer.Viewer(cyl_case, field="Cp", off_screen=True, show_streamlines=True)
    assert v.fields[0] == "U_mag" and "min_angle" in v.fields
    assert v._stream is not None and v._stream.n_points > 0
    v.i = v.fields.index("min_angle")
    v._draw_field()
    v._worst()
    png = v.save(tmp_path / "shot.png")
    assert Path(png).stat().st_size > 10000
    fig = viewer.plots(cyl_case, show=False, save=tmp_path / "plots.png")
    assert fig is not None and (tmp_path / "plots.png").exists()


def test_viewer_mesh_npz(tmp_path):
    import hybrid_mesher as hm
    m = hm.build_mesh(naca="0012", n_side=60, verbose=False)
    hm.save_npz(m, tmp_path / "m.npz")
    v = viewer.Viewer(tmp_path / "m.npz", off_screen=True)
    assert v.fields[0] == "min_angle" and not v.has_flow
