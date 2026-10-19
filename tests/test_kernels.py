"""Phase 4: every compiled kernel against the numpy code it replaces."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerolab.core import kernels  # noqa: E402

pytestmark = pytest.mark.skipif(not kernels.HAVE_NUMBA, reason="numba not installed")


@pytest.fixture(scope="module")
def naca():
    """A coarse NACA 0012 SA case run for 25 iterations (non-trivial fields)."""
    from aerolab.validation import validate_naca0012_sa as v
    from aerolab.core.fvmesh import FVMesh
    from aerolab.core.simple_solver import SimpleSolver
    a = 8.0
    mesh = v.airfoil_mesh(0.6, a)
    U = (math.cos(math.radians(a)), math.sin(math.radians(a)))
    P, ff = mesh["points"], mesh["patches"]["farfield"]
    t = P[ff[:, 1]] - P[ff[:, 0]]
    inflow = t[:, 1] * U[0] - t[:, 0] * U[1] < 0
    fvm = FVMesh(P, mesh["cells"], {"wall": mesh["patches"]["airfoil"], "inlet": ff[inflow],
                                    "outlet": ff[~inflow]})
    s = SimpleSolver(fvm, {"wall": {"type": "wall"}, "inlet": {"type": "inlet", "U": U},
                           "outlet": {"type": "outlet"}}, 1 / 6e6, U0=U, turbulence="sa")
    for _ in range(25):
        s.iterate()
    return s


def test_venkat(naca):
    s, m = naca, naca.m
    Ub, _ = s._boundary_values()
    for k in range(2):
        g = m.grad(s.U[:, k], Ub[:, k])
        kernels.USE_NUMBA = False
        a = s._venkat(s.U[:, k], g, Ub[:, k])
        kernels.USE_NUMBA = True
        b = s._venkat(s.U[:, k], g, Ub[:, k])
        assert np.abs(a - b).max() < 1e-13
        assert a.min() < 0.99                       # the limiter is active somewhere


def test_momentum_coeffs(naca):
    s, m = naca, naca.m
    n, own, nei = m.ncells, m.own, m.nei
    Ub, _ = s._boundary_values()
    gU = [m.grad(s.U[:, k], Ub[:, k]) for k in range(2)]
    nu_f, _ = s.turb.face_viscosity()
    F = s.F
    diag, a_on, a_no, src, tr = kernels.momentum_coeffs(
        n, own, nei, F, np.ascontiguousarray(nu_f), m.coef, m.w, m.Sf, m.Delta, m.xc, m.xf,
        gU[0], gU[1], gU[0], gU[1], True)
    D = nu_f * m.coef
    d0 = np.bincount(own, D + np.maximum(F, 0), n) + np.bincount(nei, D + np.maximum(-F, 0), n)
    assert np.allclose(diag, d0, rtol=1e-13, atol=0)
    assert np.allclose(a_on, -D - np.maximum(-F, 0.0)) and np.allclose(a_no, -D - np.maximum(F, 0))
    Uc = np.where(F >= 0, own, nei)
    for k in range(2):
        gf = m.interp(gU[k])
        s0 = m.sum_to_cells(nu_f * np.einsum('ij,ij->i', gf, m.Sf - m.Delta))
        s0 -= m.sum_to_cells(F * np.einsum('ij,ij->i', gU[k][Uc], m.xf - m.xc[Uc]))
        t0 = m.sum_to_cells(nu_f * np.einsum('ij,ij->i', np.c_[m.interp(gU[0][:, k]),
                                                                 m.interp(gU[1][:, k])], m.Sf))
        assert np.abs(src[:, k] - s0).max() <= 1e-12 * np.abs(s0).max()
        assert np.abs(tr[:, k] - t0).max() <= 1e-12 * np.abs(t0).max()


def test_csr_pattern(naca):
    from aerolab.core.linsolve import csr
    m = naca.m
    n, own, nei = m.ncells, m.own, m.nei
    rng = np.random.default_rng(1)
    d, ao, an = rng.random(n) + 4, -rng.random(len(own)), -rng.random(len(own))
    A = kernels.CsrPattern(n, own, nei).matrix(d, ao, an)
    B = csr(np.r_[d, ao, an], np.r_[np.arange(n), own, nei], np.r_[np.arange(n), nei, own], n)
    assert (A != B).nnz == 0 and A.has_sorted_indices


def test_sgs_matches_pyamg(naca):
    from pyamg.relaxation.relaxation import gauss_seidel
    m = naca.m
    n, own, nei = m.ncells, m.own, m.nei
    rng = np.random.default_rng(2)
    ao, an = -rng.random(len(own)), -rng.random(len(own))
    d = np.bincount(own, -ao, n) + np.bincount(nei, -an, n) + 0.1
    A = kernels.CsrPattern(n, own, nei).matrix(d, ao, an)
    B = rng.random((n, 2))
    X = np.zeros((n, 2))
    kernels.sgs(A.indptr, A.indices, A.data, X, B, 0.0, 3)
    for c in range(2):
        x = np.zeros(n)
        gauss_seidel(A, x, B[:, c].copy(), iterations=3, sweep="symmetric")
        assert np.abs(X[:, c] - x).max() < 1e-12 * np.abs(x).max()


def test_sa_source(naca):
    from aerolab.core import turbulence as tb
    s = naca
    T = s.turb
    gu, gv = s.gU
    Om = np.abs(gv[:, 0] - gu[:, 1])
    S0 = T.source(T.nuT, Om)
    eps = 1e-6 * np.maximum(T.nuT, T.nu)
    J = (T.source(T.nuT + eps, Om) - S0) / eps
    a, b = kernels.sa_source(T.nuT, Om, T.d, float(T.nu), np.ones(1), False, tb.CB1, tb.CW1,
                             tb.CW2, tb.CW3, tb.CV1, tb.CV2, tb.CV3, tb.KAPPA)
    assert np.abs(a - S0).max() <= 1e-13 * np.abs(S0).max()
    assert np.abs(b - J).max() <= 1e-7 * np.abs(J).max()     # one-sided difference


def test_coupled_matvec_matches_assembled(naca):
    """the matrix-free block product of the coupled solver = its assembled matrix"""
    from aerolab.core import coupled
    m = naca.m
    n = m.ncells
    rng = np.random.default_rng(3)
    o = naca.b_outlet
    ops = dict(n=n, diag=rng.random(n) + 1, a_on=-rng.random(len(m.own)),
               a_no=-rng.random(len(m.own)), pb_impl=np.where(o, 0.0, 1.0),
               cf=rng.random(len(m.own)), cb_p=np.where(o, rng.random(len(o)), 0.0))
    K = coupled.CoupledSolver._assemble(naca, ops, m.w, m.Sf[:, 0], m.Sf[:, 1], m.bSf, o)
    x = rng.random(3 * n)
    y = kernels.coupled_matvec(n, m.own, m.nei, m.bcell, ops["diag"], ops["a_on"], ops["a_no"],
                               m.w, m.Sf, m.bSf, ops["pb_impl"], ops["cf"], ops["cb_p"], o, x,
                               True, True)
    assert np.abs(y - K @ x).max() < 1e-12 * np.abs(y).max()
