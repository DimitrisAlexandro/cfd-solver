"""Amplification-factor transport (aft.py), field inversion / learning
(fiml.py), polar surrogate (surrogate.py)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerolab.design import adjoint as A  # noqa: E402
from aerolab.physics import aft as AF  # noqa: E402
from aerolab.ml import fiml  # noqa: E402
from aerolab.design import surrogate as SG  # noqa: E402
from aerolab.physics import transition as T  # noqa: E402


@pytest.fixture(scope="module")
def laminar_plate():
    from aerolab.validation import validate_flatplate as vf
    from aerolab.core.simple_solver import SimpleSolver
    m = vf.plate_mesh(0.5)
    s = SimpleSolver(m, vf.BCS, 1.0 / 5e6, U0=(1.0, 0.0))
    s.solve(tol=1e-8, max_iter=8000, verbose=False)
    a = AF.AFT.__new__(AF.AFT)
    from aerolab.core.turbulence import wall_distance
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
    from aerolab.airfoil import polar as pl
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


def test_surrogate_same_airfoil_from_a_file(tmp_path):
    """an uploaded NACA 0012 with its own point spacing (features a fraction of a percent
    off) is predicted like the library NACA 0012, not as an unknown airfoil: the shipped
    model was trained on 12 %-thick sections only, so the spread of t and x_t is ~0"""
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 40)))
    y = 0.6 * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    pts = np.vstack([np.c_[x, y][::-1], np.c_[x, -y][1:]])
    f = tmp_path / "my0012.dat"
    f.write_text("my foil\n" + "\n".join(f"{a:.6f} {b:.6f}" for a, b in pts) + "\n")
    sg = SG.Surrogate.load()
    al = np.array([0.0, 4.0, 8.0])
    ref, mine = sg.predict("naca0012", 3e6, al), sg.predict(str(f), 3e6, al)
    assert np.allclose(mine["Cl"], ref["Cl"], atol=0.05)
    assert np.all(mine["Cl_sd"] < 0.5)
    assert np.ptp(mine["Cl"]) > 0.5                      # follows alpha, not a flat line


def test_surrogate_reads_the_results_folder_and_saves_there(tmp_path, monkeypatch):
    """training data from AEROLAB_CASES (the wing tool's cache too, without the shipped
    duplicates); the retrained model goes to the results folder and is then preferred"""
    import shutil
    shipped = sorted((SG.ROOT / "data" / "wing_polars").glob("*.json"))
    monkeypatch.setattr(SG, "CASES", tmp_path)
    monkeypatch.setattr(SG, "USER_MODEL", tmp_path / "_surrogate" / "surrogate_model.json")
    (tmp_path / "wing_polars").mkdir()
    shutil.copy(shipped[0], tmp_path / "wing_polars" / shipped[0].name)          # a duplicate: skipped
    _, base = SG.collect()
    new = shipped[0].name.replace("_Re", "_Re1", 1)                             # a "new" polar
    shutil.copy(shipped[0], tmp_path / "wing_polars" / new)
    rows, sources = SG.collect()
    assert new in sources and len(sources) == len(base) + 1
    assert SG.model_path() == SG.MODEL
    sg = SG.Surrogate.train(rows[:30], sources)
    sg.save()
    assert SG.model_path() == SG.USER_MODEL and SG.USER_MODEL.exists()
    assert SG.Surrogate.load().n == 30
