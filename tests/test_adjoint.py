"""Discrete adjoint (adjoint.py): the residual mirrors the solver, the
coloured complex-step Jacobian is exact, and the adjoint gradients agree with
finite differences of complete flow solutions (small meshes)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import adjoint as A  # noqa: E402


@pytest.fixture(scope="module")
def turbulent():
    pr = A.solve_primal("naca0012", 1e6, 3.0, level=0.3, transition="off")
    adj = A.Adjoint(pr["solver"], 3.0, pr["vortex"], pr["walls"])
    adj.solve(["Cl", "Cd", "Cm"])
    return pr, adj


def test_geometry_matches_fvmesh(turbulent):
    pr, adj = turbulent
    m, g = pr["solver"].m, adj.res.geo0
    for k in ("vol", "xc", "Sf", "bSf", "coef", "w", "rskew", "bcoef", "Delta", "db_t"):
        assert np.allclose(getattr(g, k), getattr(m, k), rtol=1e-12, atol=1e-14), k
    assert np.allclose(g.wall_d, pr["solver"].turb.d, rtol=1e-10)
    rng = np.random.default_rng(1)
    phi, pb = rng.random(m.ncells), rng.random(len(m.bcell))
    assert np.allclose(g.grad(phi, pb), m.grad(phi, pb), rtol=1e-9, atol=1e-9)
    assert np.allclose(g.grad_gg(phi, pb, m.grad(phi, pb)), m.grad_gg(phi, pb, m.grad(phi, pb)))


def test_residual_zero_at_converged_solution(turbulent):
    """R(X*) at the level of the solver's convergence in every block"""
    pr, adj = turbulent
    rc = adj.residual_check()
    s = pr["solver"]
    assert max(rc[k] for k in ("u", "v", "p", "F", "Fo")) < 1e-8
    assert rc["nt"] < 1e-12 and rc["G"] < 1e-12
    gp = np.abs(s.gradp).max()
    assert max(rc["gx"], rc["gy"]) < 1e-5 * gp


def test_jacobian_exact(turbulent):
    """coloured Jacobian = directional complex-step derivative"""
    _, adj = turbulent
    for seed in range(2):
        assert adj.jac.check(adj.J, adj.X, seed=seed) < 1e-12


def test_adjoint_equals_tangent(turbulent):
    """duality: psi^T dR/dalpha + dI/dalpha = dI/dX . dX/dalpha (tangent)"""
    from scipy.sparse.linalg import spsolve
    _, adj = turbulent
    dR, dI = adj._partials(alpha=adj.res.alpha + 1j * A.H_CS)
    dX = spsolve(adj.J.tocsc(), -dR)
    ga = adj.gradient_alpha()
    for k in ("Cl", "Cd", "Cm"):
        tangent = dI[k] + adj.res.objective_gradient(adj.X, k) @ dX
        assert abs(tangent - ga[k]) < 1e-6 * max(abs(ga[k]), 1e-6)


def test_dcl_dalpha_vs_finite_difference(turbulent):
    pr, adj = turbulent
    import polar as pl
    st = {"xc": pr["fvm"].xc, "U": pr["solver"].U, "p": pr["solver"].p,
          "nuT": pr["solver"].turb.nuT}
    # small steps: the limiter / upwind switches make larger-step differences
    # noisy at the 1e-3 level (the adjoint is the exact local derivative)
    da = 0.02
    c = [pl.coefficients(q["solver"], q["walls"], q["U"])[0] for q in
         (A.solve_primal("naca0012", 1e6, 3.0 + s * da, level=0.3, init=st) for s in (1, -1))]
    ga = adj.gradient_alpha()
    for k, tol in (("Cl", 1e-4), ("Cm", 2e-2)):
        fd = (c[0][k] - c[1][k]) / (2 * da)
        assert abs(ga[k] - fd) < tol * max(abs(fd), 1e-3), (k, ga[k], fd)


def test_shape_gradient_vs_finite_difference(turbulent):
    """one CST coefficient: mesh deformed by the same IDW map"""
    pr, adj = turbulent
    import polar as pl
    defo = A.IDWDeformation(pr["solver"].m, pr["walls"])
    dP = A.cst_wall_velocities(defo, 8)
    j = 3
    g = adj.gradient_shape([dP[j]])
    st = {"xc": pr["fvm"].xc, "U": pr["solver"].U, "p": pr["solver"].p,
          "nuT": pr["solver"].turb.nuT}
    eps = 2e-4
    P0 = np.asarray(pr["fvm"].points)
    c = [pl.coefficients(q["solver"], q["walls"], q["U"])[0] for q in
         (A.solve_primal("naca0012", 1e6, 3.0, level=0.3, init=st,
                         fvm_override=P0 + s * eps * dP[j]) for s in (1, -1))]
    for k in ("Cl", "Cd"):
        fd = (c[0][k] - c[1][k]) / (2 * eps)
        assert abs(g[k][0] - fd) < 2e-3 * abs(fd), (k, g[k][0], fd)


def test_laminar_adjoint():
    """laminar solver (no SA, no vortex far field): dCd/dalpha vs FD"""
    pr = A.solve_primal("naca0012", 2e3, 2.0, level=0.3)
    adj = A.Adjoint(pr["solver"], 2.0, pr["vortex"], pr["walls"])
    assert adj.res.N == 5 * adj.res.n + adj.res.nf + adj.res.no
    adj.solve(["Cl", "Cd"])
    import polar as pl
    st = {"xc": pr["fvm"].xc, "U": pr["solver"].U, "p": pr["solver"].p}
    da = 0.2            # laminar Re 2e3: small-step differences are noisy
    c = [pl.coefficients(q["solver"], q["walls"], q["U"])[0] for q in
         (A.solve_primal("naca0012", 2e3, 2.0 + s * da, level=0.3, init=st) for s in (1, -1))]
    ga = adj.gradient_alpha()
    fd = (c[0]["Cl"] - c[1]["Cl"]) / (2 * da)
    assert abs(ga["Cl"] - fd) < 2e-3 * abs(fd)
    assert adj.jac.check(adj.J, adj.X) < 1e-12


def test_idw_deformation_is_linear_and_fixes_far_field(turbulent):
    pr, _ = turbulent
    m = pr["solver"].m
    defo = A.IDWDeformation(m, pr["walls"])
    rng = np.random.default_rng(0)
    d1 = rng.standard_normal((defo.nwall, 2))
    d2 = rng.standard_normal((defo.nwall, 2))
    f = defo.field(d1 + 2 * d2)
    assert np.allclose(f, defo.field(d1) + 2 * defo.field(d2))
    ff = m.patch_faces["inlet"]
    assert np.abs(f[m.ba[ff]]).max() == 0.0
    assert np.allclose(f[defo.wall_pts], d1 + 2 * d2)


def test_nested_subdivision_conserves_geometry(turbulent):
    import adjoint_error as E
    from fvmesh import FVMesh
    pr, _ = turbulent
    m = pr["solver"].m
    pats = {n: np.c_[m.ba[m.patch_faces[n]], m.bb[m.patch_faces[n]]] for n in m.patch_names}
    P, cells, pat, parent = E.subdivide(m.points, m.cells, pats)
    mf = FVMesh(P, cells, pat)
    assert mf.ncells == 4 * m.ncells
    assert np.allclose(np.bincount(parent, mf.vol, m.ncells), m.vol, rtol=1e-12)
    for n in m.patch_names:
        assert len(mf.patch_faces[n]) == 2 * len(m.patch_faces[n])
        assert np.allclose(mf.bSf[mf.patch_faces[n]].sum(0), m.bSf[m.patch_faces[n]].sum(0))


def test_deformation_turns_boundary_layer_columns_with_the_wall(turbulent):
    """a nose-radius change: plain IDW leans the boundary-layer columns, the
    normal-following deformation keeps them orthogonal"""
    from fvmesh import FVMesh
    pr, _ = turbulent
    m = pr["solver"].m

    def nonorth(f):
        c = np.abs((f.d * f.Sf).sum(1)) / np.linalg.norm(f.d, axis=1) / np.linalg.norm(f.Sf, axis=1)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))
    n0 = nonorth(m)
    pats = {n: np.c_[m.ba[m.patch_faces[n]], m.bb[m.patch_faces[n]]] for n in m.patch_names}
    # the thin near-wall cells (aspect ratios ~1e3), where a lean makes the
    # explicit non-orthogonal correction diverge
    from turbulence import wall_distance
    d = wall_distance(m, m.patch_faces["wall"])
    near = (m.nverts[m.own] == 4) & (d[m.own] < 0.002)
    inc = {}
    for follow in (None, (0.02, 0.08)):
        defo = A.IDWDeformation(m, pr["walls"], follow=follow)
        dP = A.cst_wall_velocities(defo, 8)
        P = np.asarray(m.points) + 0.02 * dP[9]          # first lower coefficient (nose)
        inc[follow is not None] = (nonorth(FVMesh(P, m.cells, pats)) - n0)[near].max()
    assert inc[True] < 0.25 * inc[False]
