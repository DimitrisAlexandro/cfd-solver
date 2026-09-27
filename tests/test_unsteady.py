"""Regression tests for the transient solver (full study: validate_unsteady.py)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import validate_unsteady as vu   # noqa: E402


def test_bdf2_second_order_in_time():
    # - the reference time step must be much smaller than the tested ones
    # - on a very coarse mesh (24^2) a tiny dt-dependent part of the Rhie-Chow
    #   pressure smoothing (~2e-5, < 0.5 % of the spatial error there) does not
    #   scale with dt^2 and masks the order; on 48^2 the order is a clean 2.0
    n = 48
    ref = vu.tgv_run(n=n, dt=0.00625, t_end=1.0, tol=1e-10)
    e = []
    for dt in (0.1, 0.05):
        r = vu.tgv_run(n=n, dt=dt, t_end=1.0, tol=1e-10)
        e.append(np.sqrt((((r["U"] - ref["U"]) ** 2).sum(1) * r["vol"]).sum() / r["vol"].sum()))
    assert math.log(e[0] / e[1]) / math.log(2) > 1.8


def test_transient_converges_to_steady():
    """A long transient run of a steady flow must end on the steady solution."""
    from fvmesh import FVMesh, rect_quad
    from simple_solver import SimpleSolver
    m = FVMesh(*rect_quad(20, 20))
    bcs = {"top": {"type": "wall", "U": (1.0, 0.0)}, "bottom": {"type": "wall"},
           "left": {"type": "wall"}, "right": {"type": "wall"}}
    steady = SimpleSolver(m, bcs, 0.01)
    steady.solve(tol=1e-9, max_iter=5000, verbose=False)
    tr = SimpleSolver(m, bcs, 0.01, alpha_u=0.9)
    for _ in range(150):
        tr.advance(0.5, max_outer=30, tol=1e-9)
    assert np.abs(tr.U - steady.U).max() < 2e-4


def test_shedding_stats_on_synthetic_signal():
    t = np.arange(0, 100, 0.05)
    cl = 0.3 * np.sin(2 * np.pi * 0.165 * t)
    cd = 1.33 + 0.01 * np.sin(4 * np.pi * 0.165 * t)
    s = vu.shedding_stats(t, cd, cl)
    assert s["St"] == pytest.approx(0.165, rel=1e-3)
    assert s["Cl_amp"] == pytest.approx(0.3, rel=1e-2)
    assert s["Cd_mean"] == pytest.approx(1.33, rel=1e-3)


def test_transient_case_files(tmp_path):
    pytest.importorskip("pyvista")
    import case
    import viewer
    out = tmp_path / "tr"
    case.main(["run", "cylinder", "--re", "100", "--refine", "0.5", "--transient", "--dt", "0.2",
               "--t-end", "3", "--write-every", "1", "--no-png", "--out", str(out)])
    meta = json.loads((out / "case.json").read_text())
    assert meta["transient"]["steps"] == 15 and meta["transient"]["snapshots"] == 3
    h = np.genfromtxt(out / "history.csv", delimiter=",", names=True)
    assert len(h) == 15 and h["time"][-1] == pytest.approx(3.0)
    assert (out / "results.pvd").exists() and len(list((out / "snapshots").glob("*.vtu"))) == 3
    gif = viewer.animate(out, field="vorticity", out=tmp_path / "a.gif")
    assert Path(gif).stat().st_size > 1000
    assert viewer.plots(out, show=False, save=tmp_path / "p.png") is not None
