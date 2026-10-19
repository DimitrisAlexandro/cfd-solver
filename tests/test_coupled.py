"""Phase 4b: the coupled (u, v, p) solver reaches the SIMPLEC solution."""
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerolab.core.fvmesh import FVMesh  # noqa: E402
from aerolab.core.simple_solver import SimpleSolver  # noqa: E402

pardiso = pytest.importorskip("pypardiso", reason="pypardiso (MKL PARDISO) not installed")


def test_pardiso_lu_refactor_same_pattern():
    """symbolic analysis once, numeric refactorisation for new values"""
    from aerolab.core.coupled import PardisoLU
    rng = np.random.default_rng(0)
    n = 400
    A = sp.random(n, n, density=0.02, random_state=1, format="csr") + sp.eye(n) * 5
    A = A.tocsr()
    lu = PardisoLU()
    for k in range(3):
        A.data = A.data * (1 + 0.1 * k)
        A.data[A.indptr[:-1]] += 0.0          # same pattern, new values
        lu.factor(A)
        b = rng.random(n)
        x = lu.solve(b)
        assert np.abs(A @ x - b).max() < 1e-10 * np.abs(b).max()


@pytest.mark.parametrize("linear", ["pardiso", "direct"])
def test_coupled_channel_matches_simple(linear):
    """unstructured-triangle channel: the coupled solver converges to the
    SIMPLEC solution (same discretisation), in far fewer iterations"""
    from aerolab.validation import validate_channel as vch
    from aerolab.core.coupled import CoupledSolver
    from aerolab.core.fvmesh import rect_tri
    m = FVMesh(*rect_tri(0.1, 0, 3, -vch.H, vch.H, names=vch.NAMES))
    bcs = {"wallB": {"type": "wall"}, "wallT": {"type": "wall"},
           "inlet": {"type": "inlet", "U": vch.profile}, "outlet": {"type": "outlet"}}
    s0 = SimpleSolver(m, bcs, vch.NU, U0=(0.5, 0))
    s0.solve(tol=1e-10, max_iter=5000, verbose=False)
    s1 = CoupledSolver(m, bcs, vch.NU, U0=(0.5, 0), linear=linear)
    it = s1.solve(tol=1e-10, max_iter=500, verbose=False)
    assert s1.converged and it < 0.5 * len(s0.history)
    assert np.abs(s1.U - s0.U).max() < 1e-6
    assert np.abs(s1.p - s0.p).max() < 1e-6
    assert np.abs(m.sum_to_cells(s1.F, s1.Fb)).max() < 1e-10       # mass conserved


def test_coupled_naca_sa_matches_simple():
    """turbulent NACA 0012 (SA, validated mesh), alpha 4: same lift/drag as
    SIMPLEC within the documented offset (Cl -6e-4 at small angles, see
    claude/phase4b-coupled.md), in far fewer iterations"""
    from aerolab.airfoil import polar as pl
    a, b = (pl.run_point("tmr0012", 6e6, 4.0, transition="off", solver=sv)[0]
            for sv in ("coupled", "simple"))
    assert a["solver"] == "coupled" and a["status"].startswith("converged")
    assert a["iterations"] < 0.5 * b["iterations"]
    assert a["Cl"] == pytest.approx(b["Cl"], abs=1e-3)
    assert a["Cd"] == pytest.approx(b["Cd"], rel=5e-3)


def test_auto_solver_choice(monkeypatch):
    from aerolab.airfoil import polar as pl
    monkeypatch.delenv("CFD_SOLVER", raising=False)
    assert pl.resolve_solver("auto", 6e6) == "simple"
    monkeypatch.setenv("CFD_SOLVER", "coupled")
    assert pl.resolve_solver("auto", 6e6) == "coupled"
    assert pl.resolve_solver("simple", 6e6) == "simple"
    with pytest.raises(ValueError):
        pl.resolve_solver("fast", 6e6)
