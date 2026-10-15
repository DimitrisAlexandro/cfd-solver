"""Amplification-factor transport (aft.py), field inversion / learning
(fiml.py), polar surrogate (surrogate.py)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import adjoint as A  # noqa: E402
import aft as AF  # noqa: E402
import fiml  # noqa: E402
import surrogate as SG  # noqa: E402
import transition as T  # noqa: E402


@pytest.fixture(scope="module")
def laminar_plate():
    import validate_flatplate as vf
    from simple_solver import SimpleSolver
    m = vf.plate_mesh(0.5)
    s = SimpleSolver(m, vf.BCS, 1.0 / 5e6, U0=(1.0, 0.0))
    s.solve(tol=1e-8, max_iter=8000, verbose=False)
    a = AF.AFT.__new__(AF.AFT)
    from turbulence import wall_distance
    a.s, a.re, a.ncrit, a.closure = s, 5e6, 9.0, "thwaites"
    wall = np.where(s.b_wall)[0]
    a._wall_faces = wall
    a.d, a._nearest = wall_distance(m, wall, return_index=True)
    a._before_te = m.xc[:, 0] < 1.99
    a.d_max = 0.1
    return s, a


def test_similarity_closure_recovers_blasius_from_single_cells(laminar_plate):
    s, a = laminar_plate
    a.closure = "similarity"
    P, st = a.production()
    m = s.m
    for xq in (0.5, 1.0):
        xcol = m.xc[np.argmin(np.abs(m.xc[:, 0] - xq)), 0]
        col = np.where((np.abs(m.xc[:, 0] - xcol) < 1e-9) & st["inside"])[0]
        k = col[(st["phi"][col] > 0.2) & (st["phi"][col] < 0.9)]
        assert len(k) >= 4
        ret = 0.664 * math.sqrt(5e6 * xcol)
        assert np.all(np.abs(st["Re_theta"][k] / ret - 1) < 0.03)
        assert np.all(np.abs(st["H"][k] - 2.591) < 0.01)
    a.closure = "thwaites"


def test_transported_n_follows_the_envelope(laminar_plate):
    """flat plate: max of n across the layer vs the e^N envelope of the exact
    Blasius layer (H = 2.591, theta = 0.664 sqrt(nu x / U)). (XFOIL's laminar
    closure, transition.laminar_bl, has H = 2.568 here and its envelope runs
    ~0.8 lower at N ~ 7: the criterion is that sensitive to H.)"""
    s, a = laminar_plate
    n = a.solve_n()
    st = a.state
    m = s.m
    xs = np.linspace(1e-6, 2, 40000)
    th = 0.664 * np.sqrt(xs / 5e6)
    N, _ = T.integrate_N(xs, np.full_like(xs, 2.591), 5e6 * th, th)
    thx, Hx, rx, _ = T.laminar_bl(xs, np.ones_like(xs), 5e6)
    Nx, _ = T.integrate_N(xs, Hx, rx, thx)
    for xq in (0.6, 0.8, 1.0):
        xcol = m.xc[np.argmin(np.abs(m.xc[:, 0] - xq)), 0]
        col = np.where((np.abs(m.xc[:, 0] - xcol) < 1e-9) & st["inside"])[0]
        nmax = n[col].max()
        # the exact-Blasius envelope (the classical e^9 result, Re_x ~ 3e6);
        # XFOIL's closure (H = 2.568) runs below it
        assert abs(nmax - np.interp(xcol, xs, N)) < 0.4
        assert nmax > np.interp(xcol, xs, Nx)
    # Thwaites theta from the transported integral vs Blasius
    xcol = m.xc[np.argmin(np.abs(m.xc[:, 0] - 1.0)), 0]
    # the inner layer carries the integral from the leading edge (outer
    # cells entrain fluid that started with I = 0)
    k = np.where(st["inside"] & (np.abs(m.xc[:, 0] - xcol) < 1e-9) & (st["phi"] < 0.6))[0]
    assert len(k) >= 4
    assert np.median(st["Re_theta"][k]) == pytest.approx(0.664 * math.sqrt(5e6 * xcol), rel=0.03)


def test_production_zero_outside_the_layer(laminar_plate):
    s, a = laminar_plate
    P, st = a.production()
    far = s.m.xc[:, 1] > 0.2
    assert np.all(P[far] == 0.0)


@pytest.fixture(scope="module")
def coarse_turbulent():
    pr = A.solve_primal("naca0012", 1e6, 4.0, level=0.3)
    return pr


def test_beta_gradient_vs_finite_difference(coarse_turbulent):
    """sum_i dCl/dbeta_i (adjoint) = dCl/d(eps) for beta = 1 + eps everywhere"""
    pr = coarse_turbulent
    s = pr["solver"]
    adj = A.Adjoint(s, 4.0, pr["vortex"], pr["walls"])
    adj.solve(["Cl"])
    g = adj.gradient_beta("Cl")
    import polar as pl
    st = {"xc": pr["fvm"].xc, "U": s.U, "p": s.p, "nuT": s.turb.nuT}
    eps = 0.02
    cl = []
    for sg in (1, -1):
        p2 = A.solve_primal("naca0012", 1e6, 4.0, level=0.3, init=st)
        s2 = p2["solver"]
        s2.turb.beta = np.full(s2.m.ncells, 1.0 + sg * eps)
        vf = p2["vortex"]
        vf.every = 1
        s2.solve(tol=1e-10, max_iter=4000, verbose=False, callback=vf.update)
        vf.update(0)
        cl.append(pl.coefficients(s2, p2["walls"], p2["U"])[0]["Cl"])
    fd = (cl[0] - cl[1]) / (2 * eps)
    assert abs(g.sum() - fd) < 0.02 * abs(fd), (g.sum(), fd)


def test_mlp_fits_a_smooth_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (2000, 3))
    y = 1.0 + 0.3 * np.tanh(2 * X[:, 0]) * X[:, 1] - 0.2 * X[:, 2] ** 2
    net = fiml.MLP(3, (16, 16)).fit(X, y, epochs=3000)
    assert np.sqrt(np.mean((net.predict(X) - y) ** 2)) < 0.03
    net2 = fiml.MLP.from_json(net.to_json())
    assert np.allclose(net2.predict(X[:10]), net.predict(X[:10]))


def test_gp_interpolates_and_reports_uncertainty():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (40, 2))
    y = np.sin(4 * X[:, 0]) + X[:, 1] ** 2
    g = SG.GP().fit(X, y)
    mu, sd = g.predict(X)
    assert np.abs(mu - y).max() < 0.02
    Xf = np.array([[3.0, 3.0]])                     # far outside the data
    assert g.predict(Xf)[1][0] > 5 * sd.max()
    g2 = SG.GP.from_json(g.to_json())
    assert np.allclose(g2.predict(X[:5])[0], mu[:5])


def test_surrogate_features_and_flap():
    f0 = SG.geometry("naca2412")
    assert f0[0] == pytest.approx(0.12, abs=0.003)      # t/c
    assert f0[2] == pytest.approx(0.02, abs=0.002)      # max camber
    assert f0[3] == pytest.approx(0.4, abs=0.05)        # its position
    f1 = SG.geometry("naca2412@flap=25c20")
    assert f1[:5] == f0[:5] and f1[6] == pytest.approx(2.0)
