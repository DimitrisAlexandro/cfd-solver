import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cdt2d import CDT                     # noqa: E402
from predicates import incircle, orient2d  # noqa: E402


def build(points, loops, tags=None):
    pts = np.asarray(points, float)
    cdt = CDT(*pts.min(0), *pts.max(0))
    ids = [cdt.insert(x, y) for x, y in pts]
    for k, loop in enumerate(loops):
        tag = tags[k] if tags else 'segment'
        for i in range(len(loop)):
            cdt.insert_segment(ids[loop[i]], ids[loop[(i + 1) % len(loop)]], tag)
    cdt.mark_regions()
    return cdt, ids


def area(cdt, tris):
    x, y = cdt.x, cdt.y
    return sum(0.5 * ((x[b] - x[a]) * (y[c] - y[a]) - (y[b] - y[a]) * (x[c] - x[a]))
               for a, b, c in tris)


def assert_constrained_delaunay(cdt, region):
    """Every non-segment edge inside the region is locally Delaunay."""
    x, y = cdt.x, cdt.y
    for t in range(len(cdt.tv)):
        if not cdt.alive[t] or cdt.region[t] != region:
            continue
        v = cdt.tv[t]
        for i in range(3):
            nt = cdt.tn[t][i]
            u, w = v[(i + 1) % 3], v[(i + 2) % 3]
            if nt < 0 or cdt.is_segment(u, w):
                continue
            s = next(k for k in cdt.tv[nt] if k != u and k != w)
            a, b, c = v
            assert incircle(x[a], y[a], x[b], y[b], x[c], y[c], x[s], y[s]) <= 0


@pytest.mark.parametrize("seed", range(20))
def test_matches_scipy_delaunay(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 1500))
    corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
    pts = np.vstack([corners, rng.uniform(0.001, 0.999, (n, 2))])
    cdt, ids = build(pts, [[0, 1, 2, 3]])
    cdt.check()
    back = {v: i for i, v in enumerate(ids)}
    mine = {tuple(sorted(back[v] for v in t)) for t in cdt.triangles(region=1)}
    ref = {tuple(sorted(s)) for s in Delaunay(pts).simplices.tolist()}
    assert mine == ref


def test_cocircular_grid():
    """Worst case for float predicates: a regular grid (all cells cocircular)."""
    g = np.linspace(0, 1, 21)
    pts = np.array([[a, b] for a in g for b in g])
    cdt = CDT(0, 0, 1, 1)
    for x, y in pts:
        cdt.insert(x, y)
    cdt.check()
    assert len(cdt.triangles()) > 0


@pytest.mark.parametrize("seed", range(10))
def test_constraints_with_hole(seed):
    rng = np.random.default_rng(seed)
    # outer square + star-shaped hole + random interior points
    k = 12
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    rad = 0.2 + 0.1 * (np.arange(k) % 2)
    hole = np.c_[0.5 + rad * np.cos(ang), 0.5 + rad * np.sin(ang)]
    rnd = rng.uniform(0, 1, (400, 2))
    rnd = rnd[np.hypot(rnd[:, 0] - 0.5, rnd[:, 1] - 0.5) > 0.32]
    pts = np.vstack([[[0, 0], [1, 0], [1, 1], [0, 1]], hole, rnd])
    loops = [[0, 1, 2, 3], list(range(4, 4 + k))]
    cdt, ids = build(pts, loops)
    cdt.check()

    # every constraint edge is in the mesh
    for loop in loops:
        for i in range(len(loop)):
            assert cdt.find_edge(ids[loop[i]], ids[loop[(i + 1) % len(loop)]])

    hole_area = 0.5 * abs(np.dot(hole[:, 0], np.roll(hole[:, 1], -1))
                          - np.dot(hole[:, 1], np.roll(hole[:, 0], -1)))
    assert area(cdt, cdt.triangles(region=1)) == pytest.approx(1 - hole_area, rel=1e-12)
    assert area(cdt, cdt.triangles(region=2)) == pytest.approx(hole_area, rel=1e-12)
    assert_constrained_delaunay(cdt, 1)


def test_refinement_quality():
    k = 16
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    hole = np.c_[0.5 + 0.15 * np.cos(ang), 0.5 + 0.15 * np.sin(ang)]
    pts = np.vstack([[[0, 0], [1, 0], [1, 1], [0, 1]], hole])
    cdt, ids = build(pts, [[0, 1, 2, 3], list(range(4, 4 + k))],
                     tags=['farfield', 'farfield'])
    cdt.refine(lambda x, y: 0.05, min_angle_deg=25.0)
    cdt.check()
    tris = cdt.triangles(region=1)
    x, y = cdt.x, cdt.y
    worst = 180.0
    for a, b, c in tris:
        P = [np.array([x[i], y[i]]) for i in (a, b, c)]
        for i in range(3):
            e1, e2 = P[(i + 1) % 3] - P[i], P[(i + 2) % 3] - P[i]
            worst = min(worst, math.degrees(math.acos(
                np.clip(e1 @ e2 / np.linalg.norm(e1) / np.linalg.norm(e2), -1, 1))))
    assert worst >= 24.9
    hole_area = 0.5 * abs(np.dot(hole[:, 0], np.roll(hole[:, 1], -1))
                          - np.dot(hole[:, 1], np.roll(hole[:, 0], -1)))
    assert area(cdt, tris) == pytest.approx(1 - hole_area, rel=1e-9)
    assert_constrained_delaunay(cdt, 1)


def test_predicates_exact_fallback():
    # three points collinear up to the last bit
    assert orient2d(0.0, 0.0, 1.0, 1.0, 0.5, 0.5) == 0.0
    assert orient2d(0.0, 0.0, 1.0, 1.0, 0.5, 0.5 + 2 ** -52) > 0
    # four exactly cocircular points
    assert incircle(0, 0, 1, 0, 1, 1, 0, 1) == 0.0
