"""3D finite-volume core (fv3d): geometry, C++ kernels, consistency with 2D, wing mesh."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerolab.fv3d import meshgen3d as G  # noqa: E402
from aerolab.fv3d import native as N  # noqa: E402
from aerolab.fv3d.mesh3d import FVMesh3D  # noqa: E402
from aerolab.fv3d.solver3d import Solver3D  # noqa: E402

LIN = np.array([2.0, -3.0, 0.5])


def lin(x):
    return 1.0 + x @ LIN


def tapered_extrusion(levels=4):
    """2D triangles + quads swept through levels that shrink and twist: tapered,
    slightly warped prisms (the situation at a wing tip)"""
    P2 = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0.5, 1.8], [1.5, 1.8]], float)
    cells = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 6], [4, 7, 6], [4, 5, 7]]
    edges = {}
    for c in cells:
        for a, b in zip(c, c[1:] + c[:1]):
            k = (min(a, b), max(a, b))
            edges[k] = edges.get(k, 0) + 1
    ep = {k: "side" for k, v in edges.items() if v == 1}

    def xyz(l):
        s = 1.0 - 0.15 * l
        th = 0.1 * l
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        Q = (P2 - 1.0) @ R.T * s + 1.0
        Q[:, 1] += 0.05 * l * Q[:, 0] ** 2                  # warps the side faces
        return np.c_[Q, np.full(len(P2), 0.3 * l)]
    return G.extrude(P2, cells, ep, levels, xyz=xyz)


def test_extrusion_is_closed_oriented_and_centres_level():
    pm = tapered_extrusion()
    m = FVMesh3D(pm)                                       # closure + positive volumes checked
    assert pm["oriented"]
    assert (m.vol > 0).all()
    # centres of a layer lie at the layer's mid height
    zc = np.round(m.xc[:, 2], 12)
    assert set(zc) <= {0.15, 0.45, 0.75}
    # geometric orientation agrees with the generator's
    pm2 = dict(pm)
    pm2["oriented"] = False
    pm2.pop("centres")
    m2 = FVMesh3D(pm2)
    assert np.allclose(m2.vol, m.vol)


def test_gradients_exact_for_linear_fields_on_warped_cells():
    m = FVMesh3D(tapered_extrusion())
    assert len(m.warp_i) > 0                               # the twist does warp faces
    phi, phib = lin(m.xc), lin(m.bxf)
    g_ls = m.grad(phi, phib)
    g_gg = m.grad_gg(phi, phib, grad_skew=np.tile(LIN, (m.ncells, 1)))
    assert np.abs(g_ls - LIN).max() < 1e-8
    assert np.abs(g_gg - LIN).max() < 1e-8


def test_duplicate_faces_between_agglomerated_cells_are_merged():
    pm = G.box(4, 3, 2)
    n = pm["ncells"]
    group = np.arange(n)
    group[group % 2 == 1] -= 1                             # pairs of cells merged
    m = FVMesh3D(G.agglomerate(pm, group))
    assert m.ncells == n // 2
    assert m.n_merged_faces > 0
    pairs = m.own * m.ncells + m.nei
    assert len(np.unique(pairs)) == len(pairs)
    g = m.grad_gg(lin(m.xc), lin(m.bxf), grad_skew=np.tile(LIN, (m.ncells, 1)))
    assert np.abs(g - LIN).max() < 1e-8


@pytest.mark.skipif(N.lib() is None, reason="C++ kernels not built (no g++)")
def test_native_kernels_match_numpy(monkeypatch):
    pm = G.box(5, 4, 3, grading=(lambda t: t ** 1.3, None, lambda t: np.sin(0.5 * np.pi * t)))
    m = FVMesh3D(pm)
    rng = np.random.default_rng(1)
    n, nf = m.ncells, len(m.own)
    adj = N.Adjacency(m)
    F = rng.normal(size=nf) * 0.1
    nu = 0.01 + 0.001 * rng.random(nf)
    gU = rng.normal(size=(3, n, 3))
    gk = 0.5 * gU
    phi = rng.normal(size=n)
    phib = rng.normal(size=len(m.bcell))
    g = rng.normal(size=(n, 3))
    nat = N.mom_assemble(m, adj, F, nu, gU, gk, True, True)
    psi_n = N.venkat(m, adj, phi, g, phib, 1e-3, None)
    monkeypatch.setattr(N, "lib", lambda: None)
    ref = N.mom_assemble(m, adj, F, nu, gU, gk, True, True)
    psi_r = N.venkat(m, adj, phi, g, phib, 1e-3, None)
    for a, b in zip(nat, ref):
        assert np.allclose(a, b, rtol=1e-12, atol=1e-14)
    assert np.allclose(psi_n, psi_r, atol=1e-12)


def _cavity3d(n, Re):
    pm = G.box(n, n, 1, lz=1.0 / n, names={"xmin": "left", "xmax": "right", "ymin": "bottom",
                                          "ymax": "top", "zmin": "front", "zmax": "back"})
    m = FVMesh3D(pm)
    bcs = {"top": {"type": "wall", "U": (1.0, 0.0, 0.0)}, "bottom": {"type": "wall"},
           "left": {"type": "wall"}, "right": {"type": "wall"},
           "front": {"type": "symmetry"}, "back": {"type": "symmetry"}}
    s = Solver3D(m, bcs, 1.0 / Re)
    s.solve(tol=1e-9, max_iter=4000, verbose=False)
    return m, s


def test_one_layer_extrusion_reproduces_the_2d_cavity():
    from aerolab.core.fvmesh import FVMesh, rect_quad
    from aerolab.core.simple_solver import SimpleSolver
    n, Re = 16, 100.0
    m3, s3 = _cavity3d(n, Re)
    pts, cells, patches = rect_quad(n, n)
    m2 = FVMesh(pts, cells, patches)
    s2 = SimpleSolver(m2, {"top": {"type": "wall", "U": (1.0, 0.0)}, "bottom": {"type": "wall"},
                           "left": {"type": "wall"}, "right": {"type": "wall"}}, 1.0 / Re)
    s2.solve(tol=1e-9, max_iter=4000, verbose=False)
    assert s3.converged
    r2, r3 = np.round(m2.xc, 9), np.round(m3.xc, 9)
    o2 = np.lexsort((r2[:, 1], r2[:, 0]))
    o3 = np.lexsort((r3[:, 1], r3[:, 0]))
    assert np.abs(s3.U[o3, :2] - s2.U[o2]).max() < 1e-5
    assert np.abs(s3.U[:, 2]).max() < 1e-10


def test_wing_mesh_is_valid_and_resolves_the_wall():
    from aerolab.fv3d.wingmesh import wing_mesh
    pm, info = wing_mesh("naca0012", semi_span=1.0, level=0.25, re=1e6, farfield=6.0, taper=0.6,
                         sweep_c4=15.0, twist=-2.0)
    m = FVMesh3D(pm)
    q = m.quality()
    assert q["nonortho_max"] < 80.0 and q["clamped"] == 0
    assert set(m.patch_names) == {"wing", "symmetry", "farfield"}
    assert info["S_half"] == pytest.approx(0.8)
    # the wing surface closes: its area vectors sum to ~0 together with the root section
    Sw = m.bSf[m.patch_faces["wing"]].sum(0)
    assert abs(Sw[0]) < 1e-3 and abs(Sw[1]) < 1e-3


def test_wing_runs_a_few_stable_iterations():
    from aerolab.fv3d.wingmesh import wing_mesh
    pm, info = wing_mesh("naca0012", semi_span=1.0, level=0.25, re=1e6, farfield=6.0)
    m = FVMesh3D(pm)
    a = math.radians(4.0)
    U = (math.cos(a), math.sin(a), 0.0)
    s = Solver3D(m, {"wing": {"type": "wall"}, "symmetry": {"type": "symmetry"},
                     "farfield": {"type": "farfield", "U": U}}, 1e-6, U0=U, turbulence="sa")
    for _ in range(15):
        r = s.iterate()
    assert all(np.isfinite(v) for v in r.values())
    assert np.linalg.norm(s.U, axis=1).max() < 5.0


def test_lateral_face_centres_are_level_with_cell_centres():
    """tapered extrusion of a thin boundary-layer-like strip: the lateral faces are
    trapezoids. Their centres must lie at the layer's mid height (as the cell centres),
    not at the area centroid: otherwise the skewness vector and the wall-face distance
    pick up a spanwise offset of dz dc/(12 c), many first-cell heights in a y+ ~ 1 layer,
    and the wall-normal gradient times that offset drove tapered wings to diverge."""
    h = 1e-5                                               # first-cell height
    P2 = np.array([[0, 0], [1, 0], [1, h], [0, h], [1, 3 * h], [0, 3 * h]], float)
    cells = [[0, 1, 2, 3], [3, 2, 4, 5]]
    ep = {(0, 1): "wall", (1, 2): "side", (2, 4): "side", (4, 5): "top", (3, 5): "side", (0, 3): "side"}
    zl = [0.0, 0.14, 0.3]

    def xyz(l):
        c = 1.0 - 0.3 * zl[l]                              # taper about x = 0.25
        Q = (P2 - [0.25, 0.0]) * c + [0.25, 0.0]
        return np.c_[Q, np.full(len(P2), zl[l])]
    m = FVMesh3D(G.extrude(P2, cells, ep, 3, xyz=xyz))
    lateral = np.abs(m.Sf[:, 2]) < 0.5 * np.linalg.norm(m.Sf, axis=1)
    assert lateral.any()
    assert np.abs(m.rskew[lateral]).max() < 1e-3 * h
    wall = m.patch_faces["wall"]
    nb = m.bSf[wall] / np.linalg.norm(m.bSf[wall], axis=1)[:, None]
    tang = m.db[wall] - np.einsum("ij,ij->i", m.db[wall], nb)[:, None] * nb
    assert np.linalg.norm(tang, axis=1).max() < 1e-3 * h
    phi, phib = lin(m.xc), lin(m.bxf)                       # still exact for linear fields
    g = m.grad_gg(phi, phib, grad_skew=np.tile(LIN, (m.ncells, 1)))
    assert np.abs(g - LIN).max() < 1e-6


def test_tip_section_morphs_from_root_to_tip():
    """tip_airfoil: the section changes linearly from the root's to the tip's; the mesh
    stays valid and the wall-layer columns stay normal to the morphed wall"""
    from aerolab.fv3d.wingmesh import wing_mesh
    pm, info = wing_mesh("naca4415", semi_span=1.0, level=0.25, re=1e6, farfield=6.0, taper=0.6,
                         tip_airfoil="naca0009")
    m = FVMesh3D(pm)
    q = m.quality()
    assert q["nonortho_max"] < 80.0 and q["clamped"] == 0
    assert info["tip_airfoil"] == "naca0009"
    X = m.bxf[m.patch_faces["wing"]]
    zl = np.array(info["z_levels"])
    it = info["itip"]

    def t_over_c(lev):
        s = (X[:, 2] > zl[lev]) & (X[:, 2] < zl[lev + 1])
        x, y = X[s, 0], X[s, 1]
        k = np.digitize(x, np.linspace(x.min(), x.max(), 30))
        return max(np.ptp(y[k == i]) for i in set(k) if (k == i).sum() > 1) / np.ptp(x)
    assert t_over_c(0) == pytest.approx(0.15, abs=0.01)
    assert t_over_c(it - 1) == pytest.approx(0.09, abs=0.01)
    # first-row faces next to the wall: no skewness beyond the plain wing's
    pm0, _ = wing_mesh("naca4415", semi_span=1.0, level=0.25, re=1e6, farfield=6.0, taper=0.6)
    m0 = FVMesh3D(pm0)
    rel = lambda mm: np.percentile(np.linalg.norm(mm.rskew, axis=1) / np.linalg.norm(mm.d, axis=1), 99.9)
    assert rel(m) < 1.5 * rel(m0)


def test_section_walls_resolve_the_leading_edge():
    """coarse levels get more wall points where the nose would turn by more than
    18 deg per panel (NACA 0012 at level 0.3: 21 deg at 48 points, the second-order
    runs diverged at the leading edge); levels that already resolve it are unchanged"""
    from aerolab.fv3d.wingmesh import le_turn, section_walls
    walls, n = section_walls(["naca0012"], 0.3)
    assert n > 48 and le_turn(walls[0]) <= 18.0
    walls, n = section_walls(["naca0012"], 0.4)
    assert n == 64
    # root and tip are repanelled alike: the sharper nose decides for both
    (w0, w1), n = section_walls(["naca0015", "naca0009"], 0.3)
    assert len(w0) == len(w1) and max(le_turn(w0), le_turn(w1)) <= 18.0


def test_tip_section_far_from_the_root_is_refused():
    from aerolab.fv3d.wingmesh import wing_mesh
    with pytest.raises(ValueError):
        wing_mesh("naca0006", semi_span=1.0, level=0.25, re=1e6, farfield=6.0, tip_airfoil="naca9430")


def test_wing_with_tip_section_runs_stable_iterations():
    """a root -> tip morph that turned the wall-layer columns badly diverged in ~10 iterations"""
    from aerolab.fv3d.wingmesh import wing_mesh
    pm, info = wing_mesh("naca2412", semi_span=1.0, level=0.25, re=1e6, farfield=6.0, taper=0.5,
                         tip_airfoil="naca0012")
    m = FVMesh3D(pm)
    a = math.radians(4.0)
    U = (math.cos(a), math.sin(a), 0.0)
    s = Solver3D(m, {"wing": {"type": "wall"}, "symmetry": {"type": "symmetry"},
                     "farfield": {"type": "farfield", "U": U}}, 1e-6, U0=U, turbulence="sa")
    for _ in range(25):
        r = s.iterate()
    assert all(np.isfinite(v) for v in r.values())
    assert np.linalg.norm(s.U, axis=1).max() < 5.0


def test_vortex_lattice_reference_of_a_3d_wing():
    from aerolab.fv3d import wing3d
    d = wing3d.vlm_reference("naca2412", "naca0012", 3.0, 0.5, 5.0, 0.0, -2.0, 1e6, (0.0, 4.0))
    r0, r4 = d["results"]
    assert d["source"] == "estimated"
    assert 0.0 < r0["CL"] < r4["CL"] < 0.5                 # camber: lift at 0 deg
    assert r4["eta"][0] > 0 and len(r4["eta"]) == len(r4["clc"])
