"""Aeroelasticity: beam, unsteady vortex-ring lattice, p-k flutter, Goland wing,
control-surface flutter, static aeroelasticity and flexible loads."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import aeroelastic as AE  # noqa: E402
import aircraft as A  # noqa: E402
import wing_structure as WS  # noqa: E402


def uniform_beam(L=6.096, c=1.8288, m=35.71, EI=9.77e6, GJ=0.987e6, xea=0.33, xcg=0.43, Iea=8.64, n_el=30,
                 sweep=0.0):
    props = {"EI": [EI] * 2, "GJ": [GJ] * 2, "m": [m] * 2, "S": [m * (xcg - xea) * c] * 2, "I": [Iea] * 2}
    b = AE.BeamModel([0, L / math.cos(math.radians(sweep))], props, sweep, n_el=n_el)
    b.x_ea_fn = lambda y: xea * c + np.asarray(y, float) * math.tan(math.radians(sweep))
    b.x_le_fn = lambda y: np.asarray(y, float) * math.tan(math.radians(sweep))
    b.chord_fn = lambda y: np.full_like(np.asarray(y, float), c)
    surf = A.Surface("w", span=2 * L, root_chord=c, tip_chord=c, airfoil="naca0012",
                     sweep=sweep, controls=[])
    b.ac, b.surface = A.Aircraft("w", [surf], A.Fuselage()), 0
    return b, surf


# ---------------------------------------------------------------- structure
def test_beam_static_and_frequencies():
    b, _ = uniform_beam(xcg=0.33)                        # no unbalance: uncoupled
    w, _ = b.modes(3)
    L, m, EI, GJ, I = 6.096, 35.71, 9.77e6, 0.987e6, 8.64
    assert w[0] == pytest.approx(1.8751**2 * math.sqrt(EI / (m * L**4)), rel=1e-4)
    assert w[1] == pytest.approx(math.pi / 2 * math.sqrt(GJ / (I * L**2)), rel=2e-3)
    f = b.load_vector([L], [1000.0], [500.0])
    u = b.static(f)
    wt, _, tt, _ = b.kinematics(u, [L])
    assert wt[0, 0] == pytest.approx(1000 * L**3 / (3 * EI), rel=1e-6)
    assert tt[0, 0] == pytest.approx(500 * L / GJ, rel=1e-6)


# ---------------------------------------------------------------- aerodynamics
def test_ring_lattice_steady_matches_vlm():
    s = A.Surface("w", span=8.0, root_chord=1.0, tip_chord=1.0, airfoil="naca0012", ns=20, nc=6)
    lat = AE.VortexRingLattice(s, ns=20, nc=6)
    F = lat.forces(0.0, lat.solve(0.0, -np.ones((lat.n, 1), complex))).real
    CLa = (F[:, 0] * lat.nrm[:, 2]).sum() / s.area
    an = A.Analysis(A.Aircraft("w", [s], A.Fuselage(), cg=(0.25, 0, 0)), 50, 0)
    an.viscous = False
    an.CD_fus = 0
    ref = (an.solve(alpha=1.0)["CL"] - an.solve(alpha=-1.0)["CL"]) / math.radians(2)
    assert CLa == pytest.approx(ref, rel=0.015)


@pytest.mark.parametrize("k", [0.2, 0.6])
def test_unsteady_lattice_approaches_theodorsen(k):
    """mid-span section of an aspect-ratio-40 wing, plunge and pitch about mid-chord"""
    b = 0.5
    s = A.Surface("w", span=40.0, root_chord=1.0, tip_chord=1.0, airfoil="naca0012")
    lat = AE.VortexRingLattice(s, ns=30, nc=16, wake_chords=60)
    mid = np.abs(lat.eta_p) < 0.04
    width = sum(abs(lat.B[i, 1] - lat.A[i, 1]) for i in np.where(mid & (lat.row == 0))[0])
    kap = k / b
    C = WS.theodorsen(k)
    for z, dz, hd, al in ((np.ones(lat.n), np.zeros(lat.n), -1.0, 0.0),
                          (-(lat.col[:, 0] - 0.5), -np.ones(lat.n), 0.0, 1.0)):
        F = lat.forces(kap, lat.solve(kap, (1j * kap * z + dz)[:, None]))[:, 0]
        Lsec = (F[mid] * lat.nrm[mid, 2]).sum() / width
        om = kap
        Lt = 2 * (math.pi * b**2 * (om**2 * -hd * -1 * -1 + 1j * om * al) +
                  2 * math.pi * b * C * (al + 1j * om * hd + 0.5 * b * 1j * om * al))
        assert abs(Lsec - Lt) / abs(Lt) < 0.04


# ---------------------------------------------------------------- flutter
def test_pk_strip_reproduces_typical_section():
    b, rho, mu, a, xa, r2, sig = 1.0, 1.0, 20.0, -0.2, 0.1, 0.24, 0.4
    m = mu * math.pi * rho * b**2

    class FB:
        L = 1.0
        chord_fn = staticmethod(lambda y: np.full_like(np.asarray(y, float), 2 * b))
        x_le_fn = staticmethod(lambda y: np.zeros_like(np.asarray(y, float)))
        x_ea_fn = staticmethod(lambda y: np.full_like(np.asarray(y, float), (1 + a) * b))

        def kinematics(self, phi, s):
            n = len(s)
            w = np.zeros((n, 2)); th = np.zeros((n, 2)); w[:, 0] = 1; th[:, 1] = 1
            return w, 0 * w, th, np.zeros(2)
    Mm = np.array([[m, -m * xa * b], [-m * xa * b, m * r2 * b**2]])
    Km = np.diag([m * sig**2, m * r2 * b**2])
    ks = np.r_[0.0, np.geomspace(0.02, 3, 40)]
    r = AE.pk(Mm, Km, AE.GAFTable(ks, AE.gaf_strip(FB(), np.eye(2), ks, b, n_strip=4)),
              np.linspace(0.3, 3.0, 271), rho, b)
    V_vg, _, _ = WS.flutter_typical_section(mu, a, xa, r2, sig)
    assert r["flutter"][0]["V"] == pytest.approx(V_vg, rel=2e-3)


@pytest.fixture(scope="module")
def goland():
    b, surf = uniform_beam()
    w, phi = b.modes(6)
    Mm, Km = phi.T @ b.M_full @ phi, phi.T @ b.K_full @ phi
    ks = np.r_[0.0, 0.02, 0.05, 0.08, 0.12, 0.17, 0.25, 0.35, 0.5, 0.7, 1.0, 1.5]
    return b, surf, phi, Mm, Km, ks


def test_goland_strip_theory_flutter(goland):
    """Goland (1945), strip theory at sea level: 137.2 m/s, 70.7 rad/s"""
    b, surf, phi, Mm, Km, ks = goland
    r = AE.pk(Mm, Km, AE.GAFTable(ks, AE.gaf_strip(b, phi, ks, 1.8288 / 2)), np.linspace(80, 200, 121),
              1.225, 1.8288 / 2)
    p = r["flutter"][0]
    assert p["V"] == pytest.approx(137.2, rel=0.02)
    assert 2 * math.pi * p["f_hz"] == pytest.approx(70.7, rel=0.03)


def test_goland_3d_flutter_above_strip(goland):
    """3D (tip loss) raises the flutter speed over strip theory"""
    b, surf, phi, Mm, Km, ks = goland
    lat = AE.VortexRingLattice(surf, ns=12, nc=8, wake_chords=25)
    r3 = AE.pk(Mm, Km, AE.GAFTable(ks, AE.gaf(b, lat, phi, ks, 1.8288 / 2)), np.linspace(80, 240, 161),
               1.225, 1.8288 / 2)
    assert 137.2 * 1.05 < r3["flutter"][0]["V"] < 137.2 * 1.35
    assert 2 * math.pi * r3["flutter"][0]["f_hz"] == pytest.approx(70.7, rel=0.08)


# ---------------------------------------------------------------- static
def test_divergence_3d_vs_strip_and_sweep():
    L, c, GJ, xea = 5.0, 1.0, 4e4, 0.40
    b0, s0 = uniform_beam(L=L, c=c, GJ=GJ, xea=xea, xcg=xea, m=5.0, Iea=0.5, EI=2e5)
    lat0 = AE.VortexRingLattice(s0, ns=12, nc=6)
    q3 = AE.divergence_q(b0, lat0)
    q_strip = math.pi**2 * GJ / (4 * L**2 * c**2 * (xea - 0.25) * 2 * math.pi)
    assert q_strip < q3 < 2.5 * q_strip                         # 3D lift is lower than 2-D
    b1, s1 = uniform_beam(L=L, c=c, GJ=GJ, xea=xea, xcg=xea, m=5.0, Iea=0.5, EI=2e5, sweep=20.0)
    q3s = AE.divergence_q(b1, AE.VortexRingLattice(s1, ns=12, nc=6))
    assert q3s > 1.3 * q3                                        # sweep-back: bending wash-out
    # elastic lift above rigid for an unswept wing with the e.a. aft of the a.c.
    ratio = AE.static_response(b0, lat0, [0.5 * q3], "symmetric")[0]
    assert ratio > 1.2


def test_aileron_mass_balance_and_flexible_loads():
    ac = A.demo_aircraft()
    import loads as LD
    res = LD.run(ac, 60.0, "normal", 0.0, None, None, log=lambda *a: None)
    st = WS.size(ac, res)
    unbal = {"freq_hz": 15.0, "mass_per_area": 4.0, "cg_aft_hinge": 0.30, "gyration": 0.30}
    bal = unbal | {"cg_aft_hinge": -0.05}
    r1, _, _ = AE.analyse(ac, st, unbal, Vmax=180.0, log=lambda *a: None)
    r2, _, _ = AE.analyse(ac, st, bal, Vmax=180.0, log=lambda *a: None)
    assert r1["flutter_speed"] < 1.2 * res["vn"]["Vd"] * 1.1     # unbalanced aileron flutters early
    assert r2["flutter_speed"] > r1["flutter_speed"] + 30        # mass balance cures it
    # flexible loads: converge, close to rigid for this stiff straight wing; infinitely stiff -> rigid
    ac.structure = WS.to_aircraft(st)
    rf = LD.run(ac, 60.0, "normal", 0.0, None, None, log=lambda *a: None, flexible=True)
    for cr, cf in zip(res["cases"], rf["cases"]):
        assert cf["elastic_converged"]
        assert cf["root_bending"] == pytest.approx(cr["root_bending"], rel=0.15, abs=1000)  # n = 0 case ~ 0
    stiff = dict(ac.structure)
    stiff["EI"] = (np.asarray(stiff["EI"]) * 1e6).tolist()
    stiff["GJ"] = (np.asarray(stiff["GJ"]) * 1e6).tolist()
    ac.structure = stiff
    rs = LD.run(ac, 60.0, "normal", 0.0, None, None, log=lambda *a: None, flexible=True)
    assert rs["cases"][0]["root_bending"] == pytest.approx(res["cases"][0]["root_bending"], rel=1e-3)
