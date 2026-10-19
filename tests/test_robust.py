"""Pseudo-time continuation, the corrected Majumdar term and the MUMPS
factorisation of the adjoint matrix."""
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerolab.core.fvmesh import FVMesh, rect_quad  # noqa: E402
from aerolab.core.simple_solver import SimpleSolver  # noqa: E402
from aerolab.validation import validate_cavity as vca  # noqa: E402


def cavity(au=0.9, cfl=None, n=21):
    s = SimpleSolver(FVMesh(*rect_quad(n, n)), vca.BCS, 1.0 / 100, alpha_u=au)
    s.cfl = cfl
    s.solve(tol=1e-10, max_iter=6000, verbose=False)
    return s


@pytest.fixture(scope="module")
def reference():
    return cavity()


@pytest.mark.parametrize("au", [0.3, 0.5])
def test_low_under_relaxation_converges_to_same_solution(reference, au):
    """the Majumdar term on the corrected velocity: SIMPLEC is stable at
    low alpha_u (it diverged for alpha_u <~ 0.8 with the predicted one) and
    the converged solution does not depend on alpha_u"""
    s = cavity(au)
    assert s.converged
    assert np.abs(s.U - reference.U).max() < 1e-7


@pytest.mark.parametrize("cfl", [0.2, 5.0])
def test_pseudo_time_does_not_change_the_solution(reference, cfl):
    s = cavity(cfl=cfl)
    assert s.converged
    assert np.abs(s.U - reference.U).max() < 1e-7
    assert np.abs((s.p - s.p.mean()) - (reference.p - reference.p.mean())).max() < 1e-6


def test_solve_robust_rolls_back_after_a_blow_up(reference):
    s = SimpleSolver(FVMesh(*rect_quad(21, 21)), vca.BCS, 1.0 / 100)
    orig = s.iterate
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 60:                      # one corrupted iteration
            s.U[:] = np.nan
            raise FloatingPointError("injected")
        return orig()
    s.iterate = flaky
    s.solve_robust(tol=1e-10, max_iter=6000)
    assert s.rollbacks == 1 and s.converged
    assert np.abs(s.U - reference.U).max() < 1e-7
    assert s.cfl is None


def test_factorize_backends_agree():
    adjoint = pytest.importorskip("aerolab.design.adjoint")
    rng = np.random.default_rng(1)
    n = 400
    A = sp.random(n, n, 0.02, random_state=2, format="csr") + sp.diags(rng.uniform(1, 3, n))
    b = rng.standard_normal(n)
    x_ref = adjoint.factorize(A, "superlu").solve(b)
    assert np.abs(A @ x_ref - b).max() < 1e-10
    if not adjoint.mumps_available():
        pytest.skip("python-mumps not installed")
    lu = adjoint.factorize(A, "mumps")
    assert type(lu).__name__ == "_MumpsLU"
    assert np.abs(lu.solve(b) - x_ref).max() < 1e-10


def test_solve_primal_robust_same_answer():
    """NACA 0012, Re 3e6, alpha 4: first-order start + pseudo-time gives the
    same converged flow as the plain path"""
    from aerolab.design import adjoint as A
    from aerolab.airfoil import polar as pl
    out = {}
    for rb in (False, True):
        pr = A.solve_primal("naca0012", 3e6, 4.0, 0.5, robust=rb)
        out[rb] = pl.coefficients(pr["solver"], pr["walls"], pr["U"])[0]
        assert pr["residual"] < 1e-9
    assert abs(out[True]["Cl"] - out[False]["Cl"]) < 1e-6
    assert abs(out[True]["Cd"] - out[False]["Cd"]) < 1e-7
