"""
2D Delaunay triangulation from scratch — Bowyer-Watson incremental algorithm.

This is Phase 1 of the mesher: the foundational algorithm that everything
else builds on (advancing front, boundary layers, quality metrics all
either build directly on this or on its 3D analogue).

Algorithm (Bowyer-Watson):
  1. Start with a single "super-triangle" that contains all input points.
  2. Insert points one at a time. For each new point:
       a. Find all triangles whose circumcircle contains the point
          ("bad" triangles) -> these violate the Delaunay condition.
       b. Remove them, which leaves a star-shaped polygonal hole.
       c. Re-triangulate the hole by connecting the new point to every
          edge on the hole's boundary.
  3. Remove any triangles that touch the original super-triangle's
     vertices.

Author: Dimitris
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


# --------------------------------------------------------------------------
# Geometry primitives
# --------------------------------------------------------------------------

def circumcircle(p1, p2, p3):
    """
    Return (center, radius) of the circle through 3 points.
    Uses the standard determinant formula. Returns (None, None) if the
    points are collinear (degenerate). Used only for visualization/
    diagnostics now — the actual Delaunay test uses the more numerically
    robust in-circle determinant below.
    """
    ax, ay = p1
    bx, by = p2
    cx, cy = p3

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None, None  # collinear

    ux = ((ax**2 + ay**2) * (by - cy) +
          (bx**2 + by**2) * (cy - ay) +
          (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) +
          (bx**2 + by**2) * (ax - cx) +
          (cx**2 + cy**2) * (bx - ax)) / d

    center = np.array([ux, uy])
    radius = np.linalg.norm(center - np.array(p1))
    return center, radius


def _orient2d(a, b, c):
    """Twice the signed area of triangle abc. >0 if CCW."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def point_in_circumcircle(pt, p1, p2, p3):
    """
    True if pt lies strictly inside the circumcircle of triangle p1,p2,p3.

    Uses the standard in-circle determinant test directly on coordinates
    relative to pt, rather than computing an explicit circumcenter/radius
    and comparing distances. This avoids the severe cancellation error
    that the center/radius approach suffers when the triangle is close
    to degenerate (nearly collinear vertices push the circumcenter far
    away, which was causing missed/incorrect triangles near the point
    set's boundary).
    """
    a, b, c = p1, p2, p3
    # Ensure a,b,c are CCW; the determinant test below assumes that.
    if _orient2d(a, b, c) < 0:
        b, c = c, b

    ax, ay = a[0] - pt[0], a[1] - pt[1]
    bx, by = b[0] - pt[0], b[1] - pt[1]
    cx, cy = c[0] - pt[0], c[1] - pt[1]

    det = ((ax * ax + ay * ay) * (bx * cy - cx * by)
           - (bx * bx + by * by) * (ax * cy - cx * ay)
           + (cx * cx + cy * cy) * (ax * by - bx * ay))

    # Relative tolerance scaled to the magnitude of the terms involved,
    # since an absolute epsilon fails for point sets at very different
    # scales.
    scale = (ax * ax + ay * ay) + (bx * bx + by * by) + (cx * cx + cy * cy)
    tol = 1e-12 * max(scale, 1.0) ** 1.5
    return det > tol


def triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                      (p3[0] - p1[0]) * (p2[1] - p1[1]))


# --------------------------------------------------------------------------
# Bowyer-Watson triangulation
# --------------------------------------------------------------------------

class Triangulation2D:
    """
    Incremental Delaunay triangulation via Bowyer-Watson.

    self.points: (N, 2) array of all vertex coordinates (including the
                 3 super-triangle vertices, which get removed at the end).
    self.triangles: list of (i, j, k) index tuples into self.points.
    """

    def __init__(self, points):
        points = np.asarray(points, dtype=float)
        self.n_real_points = len(points)

        # Build a super-triangle large enough to contain all points.
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        dx, dy = max_x - min_x, max_y - min_y
        # The super-triangle must be MUCH larger than the point set, not
        # just larger. A modest multiplier (e.g. 10x) places super-
        # triangle vertices close enough to real points near the convex
        # hull that the in-circle determinant test for triangles touching
        # a super vertex becomes near-degenerate -- small floating-point
        # errors then flip its sign, misrouting a point's insertion cavity
        # through the super vertex instead of directly between two real
        # points, and silently dropping a legitimate hull-adjacent
        # triangle. A large multiplier keeps those circumcircles far from
        # any real point, avoiding the near-degenerate case entirely.
        delta = max(dx, dy, 1.0) * 1000

        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        st_p1 = (cx - delta, cy - delta)
        st_p2 = (cx + delta, cy - delta)
        st_p3 = (cx, cy + delta)

        # Store super-triangle vertices first, real points after, so we
        # can identify and strip super-triangle indices (0,1,2) at the end.
        self.points = np.vstack([[st_p1, st_p2, st_p3], points])
        self.triangles = [(0, 1, 2)]

        for i in range(self.n_real_points):
            self._insert_point(i + 3)

        self._remove_super_triangle()
        self._legalize_all_edges()
        self._repair_boundary_gaps()

    def _insert_point(self, pt_idx):
        pt = self.points[pt_idx]

        def is_bad(tri):
            return point_in_circumcircle(pt, self.points[tri[0]],
                                          self.points[tri[1]], self.points[tri[2]])

        # Find the "bad" triangles (those whose circumcircle contains pt)
        # via a connected flood-fill, NOT a global scan.
        #
        # Delaunay theory guarantees the set of triangles whose open
        # circumdisk contains pt forms a single connected, star-shaped
        # region around pt. A naive global scan (test every triangle
        # independently) can, under floating-point error, occasionally
        # flag a triangle far from pt as "bad" even though it isn't part
        # of that connected cavity -- corrupting the hole-boundary
        # computation and silently leaving gaps in the final mesh. Flood-
        # filling from a single confirmed-bad seed, only stepping through
        # triangles that share an edge with the growing region, keeps the
        # search restricted to the true cavity and is far more robust.
        edge_to_tris = {}
        for tri in self.triangles:
            for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                edge_to_tris.setdefault(tuple(sorted(e)), []).append(tri)

        seed = next((tri for tri in self.triangles if is_bad(tri)), None)
        bad_triangles = []
        if seed is not None:
            visited = set()
            stack = [seed]
            visited.add(seed)
            while stack:
                tri = stack.pop()
                bad_triangles.append(tri)
                for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                    for neighbor in edge_to_tris.get(tuple(sorted(e)), []):
                        if neighbor != tri and neighbor not in visited and is_bad(neighbor):
                            visited.add(neighbor)
                            stack.append(neighbor)

        # Find the boundary of the polygonal hole: edges that belong to
        # exactly one bad triangle (the "outer" edges of the hole).
        edge_count = {}
        for tri in bad_triangles:
            for edge in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                key = tuple(sorted(edge))
                edge_count[key] = edge_count.get(key, 0) + 1

        boundary_edges = [e for e, count in edge_count.items() if count == 1]

        # Remove bad triangles, then re-triangulate the hole by connecting
        # the new point to every boundary edge.
        for tri in bad_triangles:
            self.triangles.remove(tri)

        for (a, b) in boundary_edges:
            self.triangles.append((a, b, pt_idx))

    def _remove_super_triangle(self):
        """Drop any triangle touching super-triangle vertices (indices 0,1,2),
        then re-index the remaining vertices to drop the 3 super points."""
        self.triangles = [
            tri for tri in self.triangles
            if all(idx >= 3 for idx in tri)
        ]
        # shift indices down by 3, drop the first 3 rows of self.points
        self.points = self.points[3:]
        self.triangles = [(a - 3, b - 3, c - 3) for (a, b, c) in self.triangles]

    def _legalize_all_edges(self, max_passes=10):
        """
        Post-process Lawson edge-flip pass: repairs any residual
        non-Delaunay edges left by the incremental insertion.

        In exact arithmetic, incremental Bowyer-Watson is already fully
        Delaunay after every insertion. In floating point, a rare local
        defect can still slip through (a bad triangle spuriously kept or
        removed near the tolerance boundary of the in-circle test),
        leaving one or two edges that violate the Delaunay condition.
        This pass finds every interior edge shared by two triangles and
        flips it whenever the opposite vertex of the neighboring
        triangle lies inside the first triangle's circumcircle -- the
        standard local repair operation -- and repeats until no edge
        needs flipping (or max_passes is hit, as a safety bound).
        """
        for _ in range(max_passes):
            edge_to_tris = {}
            for tri in self.triangles:
                for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                    edge_to_tris.setdefault(tuple(sorted(e)), []).append(tri)

            flipped_any = False
            for edge, tris in edge_to_tris.items():
                if len(tris) != 2:
                    continue  # boundary edge, nothing to flip against
                t1, t2 = tris
                if t1 not in self.triangles or t2 not in self.triangles:
                    continue  # already replaced by an earlier flip this pass

                a, b = edge
                opp1 = next(v for v in t1 if v not in edge)
                opp2 = next(v for v in t2 if v not in edge)

                if point_in_circumcircle(self.points[opp2], self.points[t1[0]],
                                          self.points[t1[1]], self.points[t1[2]]):
                    self.triangles.remove(t1)
                    self.triangles.remove(t2)
                    self.triangles.append((opp1, opp2, a))
                    self.triangles.append((opp1, opp2, b))
                    flipped_any = True

            if not flipped_any:
                break

    def _repair_boundary_gaps(self, max_passes=10):
        """
        Detect and patch single-triangle "ear" gaps that rare floating-
        point degeneracies in the super-triangle interaction can still
        leave behind after insertion + edge-flip legalization (see the
        comment on the super-triangle size above -- this is a second,
        independent safety net for the same class of defect).

        Signature of the gap: a vertex v with exactly two dangling
        boundary edges (v,a) and (v,b) whose far endpoints a,b are *not*
        already connected by an edge anywhere in the mesh. That is
        precisely a missing triangle (v,a,b) that would close the fan
        around v. Unlike an edge flip, this ADDS a triangle rather than
        swapping a diagonal, since the defect here is a true hole, not a
        bad diagonal.
        """
        for _ in range(max_passes):
            edge_count = {}
            for tri in self.triangles:
                for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                    edge_count[tuple(sorted(e))] = edge_count.get(tuple(sorted(e)), 0) + 1

            boundary_edges = [e for e, c in edge_count.items() if c == 1]

            # group dangling boundary edges by vertex
            by_vertex = {}
            for (a, b) in boundary_edges:
                by_vertex.setdefault(a, []).append(b)
                by_vertex.setdefault(b, []).append(a)

            patched_any = False
            for v, neighbors in by_vertex.items():
                if len(neighbors) != 2:
                    continue  # not the simple single-ear-gap pattern
                a, b = neighbors
                if tuple(sorted((a, b))) in edge_count:
                    continue  # a,b already connected -- not a gap

                # An ordinary convex-hull corner vertex also has exactly
                # two dangling boundary edges whose neighbors aren't
                # directly connected -- that is NOT a gap, it's the hull
                # itself, and must not be patched. The real test is
                # whether (v,a,b) is actually a valid empty-circumcircle
                # Delaunay triangle: only patch when no other point lies
                # inside its circumcircle.
                pv, pa, pb = self.points[v], self.points[a], self.points[b]
                is_valid_triangle = not any(
                    point_in_circumcircle(self.points[k], pv, pa, pb)
                    for k in range(len(self.points))
                    if k not in (v, a, b)
                )
                if not is_valid_triangle:
                    continue

                self.triangles.append((v, a, b))
                patched_any = True

            if not patched_any:
                break

        self._legalize_all_edges()

    # ----------------------------------------------------------------
    # Quality / diagnostics
    # ----------------------------------------------------------------

    def areas(self):
        return np.array([
            triangle_area(self.points[a], self.points[b], self.points[c])
            for (a, b, c) in self.triangles
        ])

    def min_angles_deg(self):
        """Minimum interior angle of each triangle, in degrees (quality metric:
        higher min angle = better shaped triangle; <15 deg is typically 'bad')."""
        out = []
        for (a, b, c) in self.triangles:
            p, q, r = self.points[a], self.points[b], self.points[c]
            angles = []
            for u, v, w in [(p, q, r), (q, r, p), (r, p, q)]:
                e1 = v - u
                e2 = w - u
                cos_a = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-14)
                cos_a = np.clip(cos_a, -1, 1)
                angles.append(np.degrees(np.arccos(cos_a)))
            out.append(min(angles))
        return np.array(out)


# --------------------------------------------------------------------------
# Validation against scipy/matplotlib's Delaunay (Qhull-based)
# --------------------------------------------------------------------------

def validate_against_scipy(points, my_triangles):
    from scipy.spatial import Delaunay as ScipyDelaunay
    ref = ScipyDelaunay(points)

    def normalize(tris):
        return set(tuple(sorted(t)) for t in tris)

    mine = normalize(my_triangles)
    theirs = normalize(ref.simplices.tolist())

    same_count = len(mine) == len(theirs)
    same_set = mine == theirs
    return same_count, same_set, len(mine), len(theirs)


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    points = rng.uniform(0, 1, size=(60, 2))

    tri = Triangulation2D(points)
    print(f"Triangulated {len(points)} points into {len(tri.triangles)} triangles")

    same_count, same_set, n_mine, n_theirs = validate_against_scipy(points, tri.triangles)
    print(f"Validation vs scipy.spatial.Delaunay (Qhull): "
          f"mine={n_mine} triangles, scipy={n_theirs} triangles, "
          f"identical triangle set = {same_set}")
    assert same_set, "Triangulation does not match reference Delaunay!"

    areas = tri.areas()
    min_angles = tri.min_angles_deg()
    print(f"Total triangulated area: {areas.sum():.4f} "
          f"(should equal convex hull area)")
    print(f"Min angle across all triangles: {min_angles.min():.2f} deg "
          f"(worst-quality triangle)")
    print(f"Mean of per-triangle min angles: {min_angles.mean():.2f} deg")

    assert (areas > 0).all(), "Degenerate (zero-area) triangle found!"

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    triang = mtri.Triangulation(points[:, 0], points[:, 1], tri.triangles)
    axes[0].triplot(triang, "b-", lw=0.6)
    axes[0].plot(points[:, 0], points[:, 1], "ko", ms=3)
    axes[0].set_title(f"Bowyer-Watson Delaunay triangulation\n"
                       f"({len(points)} points, {len(tri.triangles)} triangles)")
    axes[0].set_aspect("equal")

    axes[1].hist(min_angles, bins=20, color="steelblue", edgecolor="k")
    axes[1].axvline(30, color="r", linestyle="--", label="good quality (30 deg)")
    axes[1].set_title("Triangle quality: distribution of minimum interior angle")
    axes[1].set_xlabel("min angle (deg)")
    axes[1].set_ylabel("count")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("delaunay2d_demo.png", dpi=150)
    print("Saved plot to delaunay2d_demo.png")
    print("All sanity checks passed.")
