import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hybrid_mesher as hm   # noqa: E402
import mesh_check            # noqa: E402

CASES = [("2412", 1e6, 0.0, 120), ("0012", 3e6, 8.0, 120), ("4415", 5e5, 4.0, 120),
         ("0012", 1e7, 0.0, 120), ("6409", 2e5, 2.0, 120), ("0006", 1e5, 0.0, 120),
         ("2424", 5e6, 0.0, 120), ("2412", 1e6, 0.0, 200)]


@pytest.mark.parametrize("naca,re,alpha,n_side", CASES)
def test_airfoil_mesh(tmp_path, naca, re, alpha, n_side):
    mesh = hm.build_mesh(naca=naca, reynolds=re, alpha_deg=alpha, n_side=n_side,
                         verbose=False)
    f = tmp_path / "m.npz"
    hm.save_npz(mesh, f)
    errors, s = mesh_check.check(f, verbose=False)
    assert errors == []
    assert s["tri_below_20deg"] == 0
    assert s["nonorth_max"] < 55
    assert s["interface_size_ratio_max"] < 8
    assert s["core_size_ratio_max"] < 10          # no clusters of tiny cells
    back = hm.load_npz(f)
    assert len(back["cells"]) == len(mesh["cells"]) and back["n_bl"] == mesh["n_bl"]


def test_first_cell_height_yplus_scaling():
    from geometry import first_cell_height
    assert first_cell_height(1e6, 2.0) == pytest.approx(2 * first_cell_height(1e6, 1.0))
    assert first_cell_height(1e7) < first_cell_height(1e6)
