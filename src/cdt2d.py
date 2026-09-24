"""
2D Constrained Delaunay Triangulation (CDT) with Delaunay refinement.

This is the production core of the mesher (delaunay2d.py stays as the simple
teaching version). Differences from delaunay2d.py:

  * Robust predicates (predicates.py) - no epsilon hacks, no repair passes.
  * Real adjacency structure: every triangle knows its 3 neighbours, so
    point location is a walk and insertion is local -> roughly O(N^1.5)
    instead of O(N^2).
  * Constraint edges (segments) are forced into the mesh (Sloan's
    edge-flip recovery). That is how the airfoil boundary layer and the
    farfield enter the triangulation.
  * Regions: triangles are labelled inside/outside by flood-filling
    across constraint edges, so holes (the airfoil) are removed.
  * Ruppert-style refinement: insert circumcentres of triangles that are
    too big (sizing function) or badly shaped (minimum angle), splitting
    boundary segments when needed.

Data layout (triangle t):
    tv[t] = [a, b, c]       vertex ids, counter-clockwise
    tn[t] = [n0, n1, n2]    n_i = neighbour across the edge OPPOSITE vertex i
                            (-1 = none)
"""

import math
import random
from collections import deque

from predicates import orient2d, incircle


def _ekey(u, w):
    return (u, w) if u < w else (w, u)


class CDT:
    def __init__(self, xmin, ymin, xmax, ymax, seed=0):
        self.x, self.y = [], []
        self.tv, self.tn, self.alive, self.region = [], [], [], []
        self.free = []
        self.vtri = []                  # vertex -> one incident triangle
        self.segments = {}              # edge key -> tag (constraint edges)
        self.rng = random.Random(seed)

        # Super-triangle. With exact predicates its size no longer matters
        # for correctness; it only has to contain every point.
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        L = max(xmax - xmin, ymax - ymin, 1e-12) * 50.0
        for px, py in [(cx - 2 * L, cy - L), (cx + 2 * L, cy - L), (cx, cy + 2 * L)]:
            self.x.append(px); self.y.append(py); self.vtri.append(0)
        self._new_tri(0, 1, 2, -1, -1, -1, region=-1)
        self.last = 0
        self.n_super = 3

    # ------------------------------------------------------------------
    # low-level helpers
    # ------------------------------------------------------------------
    def _new_tri(self, a, b, c, n0, n1, n2, region=-1):
        if self.free:
            t = self.free.pop()
            self.tv[t] = [a, b, c]; self.tn[t] = [n0, n1, n2]
            self.alive[t] = True; self.region[t] = region
        else:
            t = len(self.tv)
            self.tv.append([a, b, c]); self.tn.append([n0, n1, n2])
            self.alive.append(True); self.region.append(region)
        self.vtri[a] = t; self.vtri[b] = t; self.vtri[c] = t
        return t

    def _kill(self, t):
        self.alive[t] = False
        self.free.append(t)

    def _orient(self, a, b, c):
        x, y = self.x, self.y
        return orient2d(x[a], y[a], x[b], y[b], x[c], y[c])

    def _orient_p(self, a, b, px, py):
        x, y = self.x, self.y
        return orient2d(x[a], y[a], x[b], y[b], px, py)

    def _incircle_p(self, t, px, py):
        a, b, c = self.tv[t]
        x, y = self.x, self.y
        return incircle(x[a], y[a], x[b], y[b], x[c], y[c], px, py)

    def is_segment(self, u, w):
        return _ekey(u, w) in self.segments

    def _set_neighbor_by_edge(self, t, u, w, new):
        """In triangle t, point the neighbour slot of edge {u,w} to `new`."""
        v = self.tv[t]
        for i in range(3):
            if {v[(i + 1) % 3], v[(i + 2) % 3]} == {u, w}:
                self.tn[t][i] = new
                return
        raise RuntimeError("edge not found in triangle")

    def star(self, v):
        """All triangles incident to vertex v (rotating through neighbours)."""
        t0 = self.vtri[v]
        if not self.alive[t0] or v not in self.tv[t0]:
            # stale pointer (can happen after deletions) -> brute search
            t0 = next(t for t in range(len(self.tv))
                      if self.alive[t] and v in self.tv[t])
            self.vtri[v] = t0
        out, t = [], t0
        while True:
            out.append(t)
            i = self.tv[t].index(v)
            t = self.tn[t][(i + 2) % 3]   # rotate CCW around v
            if t == t0:
                return out
            if t < 0:
                break
        # hit a hull edge: rotate the other way too
        t = t0
        while True:
            i = self.tv[t].index(v)
            t = self.tn[t][(i + 1) % 3]
            if t < 0 or t == t0:
                return out
            out.append(t)

    def find_edge(self, u, w):
        """Return (t, i): triangle containing edge {u,w}, i = local index of
        the vertex opposite that edge. None if the edge does not exist."""
        for t in self.star(u):
            v = self.tv[t]
            if w in v:
                i = next(k for k in range(3) if v[k] != u and v[k] != w)
                return t, i
        return None

    # ------------------------------------------------------------------
    # point location
    # ------------------------------------------------------------------
    def locate(self, px, py, start=None, stop_at_segment=False):
        """Stochastic visibility walk.
        Returns ('in', t) or ('segment', (u, w)) or ('outside', None)."""
        t = start if (start is not None and self.alive[start]) else self.last
        if not self.alive[t]:
            t = next(k for k in range(len(self.tv)) if self.alive[k])
        for _ in range(10 * len(self.tv) + 100):
            v = self.tv[t]
            r = self.rng.randrange(3)
            moved = False
            for k in range(3):
                i = (r + k) % 3
                u, w = v[(i + 1) % 3], v[(i + 2) % 3]
                if self._orient_p(u, w, px, py) < 0:
                    if stop_at_segment and self.is_segment(u, w):
                        return 'segment', (u, w)
                    nt = self.tn[t][i]
                    if nt < 0:
                        return 'outside', None
                    t = nt
                    moved = True
                    break
            if not moved:
                return 'in', t
        raise RuntimeError("point location did not terminate")

    # ------------------------------------------------------------------
    # insertion (Bowyer-Watson restricted by constraint edges)
    # ------------------------------------------------------------------
    def _insert_at(self, px, py, seed, split=None, check_encroach=False):
        """Insert point (px,py) whose cavity starts at triangle `seed`.
        split: edge key of a segment the point lies on (it may be crossed).
        Returns ('ok', vid, new_tris) | ('encroach', (u,w)) | ('invalid',)."""
        tv, tn = self.tv, self.tn
        cavity, in_cav, stack = [seed], {seed}, [seed]
        while stack:
            t = stack.pop()
            for i in range(3):
                nt = tn[t][i]
                if nt < 0 or nt in in_cav:
                    continue
                u, w = tv[t][(i + 1) % 3], tv[t][(i + 2) % 3]
                k = _ekey(u, w)
                if k in self.segments and k != split:
                    continue           # constraints block the cavity
                if self._incircle_p(nt, px, py) > 0:
                    in_cav.add(nt); cavity.append(nt); stack.append(nt)

        bnd = []                       # (u, w, outer_neighbour, region)
        for t in cavity:
            for i in range(3):
                nt = tn[t][i]
                if nt >= 0 and nt in in_cav:
                    continue
                bnd.append((tv[t][(i + 1) % 3], tv[t][(i + 2) % 3], nt, self.region[t]))

        if check_encroach:
            x, y = self.x, self.y
            for u, w, _, _ in bnd:
                if self.is_segment(u, w):
                    # inside diametral circle <=> angle u-p-w > 90 deg
                    if (x[u] - px) * (x[w] - px) + (y[u] - py) * (y[w] - py) < 0:
                        return ('encroach', (u, w))

        # the new fan must be valid (every new triangle CCW, cavity a disk)
        starts = set()
        for u, w, _, _ in bnd:
            if self._orient_p(u, w, px, py) <= 0 or u in starts:
                return ('invalid',)
            starts.add(u)

        # --- commit ---
        p = len(self.x)
        self.x.append(px); self.y.append(py); self.vtri.append(-1)
        for t in cavity:
            self._kill(t)
        start_of, end_of, new = {}, {}, []
        for u, w, nt, reg in bnd:
            t = self._new_tri(p, u, w, nt, -1, -1, region=reg)
            if nt >= 0:
                self._set_neighbor_by_edge(nt, u, w, t)
            start_of[u] = t; end_of[w] = t
            new.append(t)
        for t in new:
            _, u, w = tv[t]
            tn[t][1] = start_of[w]     # edge (w, p): triangle starting at w
            tn[t][2] = end_of[u]       # edge (p, u): triangle ending at u
        if split is not None:
            tag = self.segments.pop(split)
            a, b = split
            self.segments[_ekey(a, p)] = tag
            self.segments[_ekey(p, b)] = tag
        self.last = new[0]
        return ('ok', p, new)

    def insert(self, px, py):
        """Insert a point anywhere. Returns its vertex id (existing id if a
        vertex with identical coordinates is already present)."""
        status, t = self.locate(px, py)
        if status != 'in':
            raise ValueError("point outside super-triangle")
        for v in self.tv[t]:
            if self.x[v] == px and self.y[v] == py:
                return v
        res = self._insert_at(px, py, t)
        if res[0] != 'ok':
            raise RuntimeError(f"insertion failed: {res}")
        return res[1]

    # ------------------------------------------------------------------
    # edge flip
    # ------------------------------------------------------------------
    def _flip(self, t1, i1):
        tv, tn = self.tv, self.tn
        p, u, w = tv[t1][i1], tv[t1][(i1 + 1) % 3], tv[t1][(i1 + 2) % 3]
        A, B, t2 = tn[t1][(i1 + 1) % 3], tn[t1][(i1 + 2) % 3], tn[t1][i1]
        j = next(k for k in range(3) if tv[t2][k] != u and tv[t2][k] != w)
        q = tv[t2][j]
        C, D = tn[t2][(j + 1) % 3], tn[t2][(j + 2) % 3]
        tv[t1] = [p, u, q]; tn[t1] = [C, t2, B]
        tv[t2] = [q, w, p]; tn[t2] = [A, t1, D]
        if A >= 0:
            self._set_neighbor_by_edge(A, w, p, t2)
        if C >= 0:
            self._set_neighbor_by_edge(C, u, q, t1)
        self.vtri[p] = t1; self.vtri[u] = t1; self.vtri[q] = t1; self.vtri[w] = t2
        return p, q

    # ------------------------------------------------------------------
    # constraint recovery (Sloan 1993)
    # ------------------------------------------------------------------
    def _crosses(self, a, b, p, q):
        """Proper crossing of segment ab with segment pq."""
        if p in (a, b) or q in (a, b):
            return False
        o1, o2 = self._orient(a, b, p), self._orient(a, b, q)
        o3, o4 = self._orient(p, q, a), self._orient(p, q, b)
        return o1 * o2 < 0 and o3 * o4 < 0

    def insert_segment(self, a, b, tag='segment'):
        if a == b:
            return
        if self.find_edge(a, b) is not None:
            self.segments[_ekey(a, b)] = tag
            return
        x, y = self.x, self.y

        # 1) find the triangle around a that the ray a->b leaves through
        crossing = None
        for t in self.star(a):
            v = self.tv[t]
            i = v.index(a)
            u, w = v[(i + 1) % 3], v[(i + 2) % 3]
            for c in (u, w):          # a vertex lying ON segment ab?
                if self._orient(a, b, c) == 0 and \
                        (x[c] - x[a]) * (x[b] - x[a]) + (y[c] - y[a]) * (y[b] - y[a]) > 0 and \
                        (x[c] - x[b]) * (x[a] - x[b]) + (y[c] - y[b]) * (y[a] - y[b]) > 0:
                    self.insert_segment(a, c, tag)
                    self.insert_segment(c, b, tag)
                    return
            if self._orient(a, u, b) > 0 and self._orient(a, w, b) < 0:
                crossing = (t, w, u)      # L = w (left of a->b), R = u
                break
        if crossing is None:
            raise RuntimeError("could not start segment recovery")

        # 2) walk from a to b collecting all crossed edges
        t, L, R = crossing
        crossed = [(L, R)]
        while True:
            v = self.tv[t]
            i = next(k for k in range(3) if v[k] != L and v[k] != R)
            t = self.tn[t][i]
            vv = self.tv[t]
            xv = next(k for k in vv if k != L and k != R)
            if xv == b:
                break
            o = self._orient(a, b, xv)
            if o == 0:                    # collinear vertex -> split segment
                self.insert_segment(a, xv, tag)
                self.insert_segment(xv, b, tag)
                return
            if o > 0:
                L = xv
            else:
                R = xv
            crossed.append((L, R))

        # 3) flip crossed edges away
        queue = deque(crossed)
        new_edges = []
        guard = 0
        while queue:
            guard += 1
            if guard > 100000:
                raise RuntimeError("segment recovery did not converge")
            u, w = queue.popleft()
            t1, i1 = self.find_edge(u, w)
            t2 = self.tn[t1][i1]
            p = self.tv[t1][i1]
            q = next(k for k in self.tv[t2] if k != u and k != w)
            if self._orient(p, q, u) * self._orient(p, q, w) < 0:   # convex quad
                self._flip(t1, i1)
                if self._crosses(a, b, p, q):
                    queue.append((p, q))
                else:
                    new_edges.append((p, q))
            else:
                queue.append((u, w))
        self.segments[_ekey(a, b)] = tag

        # 4) restore the Delaunay property among the new edges
        changed = True
        while changed:
            changed = False
            for k, (p, q) in enumerate(new_edges):
                if self.is_segment(p, q):
                    continue
                t1, i1 = self.find_edge(p, q)
                t2 = self.tn[t1][i1]
                s = next(m for m in self.tv[t2] if m != p and m != q)
                if self._incircle_p(t1, self.x[s], self.y[s]) > 0:
                    new_edges[k] = self._flip(t1, i1)
                    changed = True

    # ------------------------------------------------------------------
    # regions / holes
    # ------------------------------------------------------------------
    def mark_regions(self):
        """Depth of each triangle = number of constraint loops crossed from
        the outside. For a farfield loop containing an airfoil loop:
        0 = outside farfield, 1 = fluid domain, 2 = inside the airfoil."""
        for t in range(len(self.tv)):
            if self.alive[t]:
                self.region[t] = -1
        frontier, depth = [self.vtri[0]], 0
        while frontier:
            stack = [t for t in frontier if self.region[t] == -1]
            for t in stack:
                self.region[t] = depth
            nxt = []
            while stack:
                t = stack.pop()
                v = self.tv[t]
                for i in range(3):
                    nt = self.tn[t][i]
                    if nt < 0 or self.region[nt] != -1:
                        continue
                    if self.is_segment(v[(i + 1) % 3], v[(i + 2) % 3]):
                        nxt.append(nt)
                    else:
                        self.region[nt] = depth
                        stack.append(nt)
            frontier, depth = nxt, depth + 1

    # ------------------------------------------------------------------
    # refinement
    # ------------------------------------------------------------------
    def _tri_geometry(self, t):
        a, b, c = self.tv[t]
        x, y = self.x, self.y
        ax, ay, bx, by, cx, cy = x[a], y[a], x[b], y[b], x[c], y[c]
        la = math.hypot(bx - cx, by - cy)
        lb = math.hypot(ax - cx, ay - cy)
        lc = math.hypot(ax - bx, ay - by)
        area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        d = 2.0 * area2
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
              + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
              + (cx * cx + cy * cy) * (bx - ax)) / d
        R = la * lb * lc / (2.0 * abs(area2))
        return R, min(la, lb, lc), (ux, uy), ((ax + bx + cx) / 3, (ay + by + cy) / 3)

    def split_segment(self, u, w):
        found = self.find_edge(u, w)
        if found is None:
            return None
        t, _ = found
        mx, my = 0.5 * (self.x[u] + self.x[w]), 0.5 * (self.y[u] + self.y[w])
        res = self._insert_at(mx, my, t, split=_ekey(u, w))
        return res[2] if res[0] == 'ok' else None

    def refine(self, size_fn, min_angle_deg=25.0, splittable=('farfield',),
               domain_region=1, max_points=500000):
        """Insert circumcentres until every domain triangle satisfies
        circumradius <= h/sqrt(3) and minimum angle >= min_angle_deg.
        Segments whose tag is not in `splittable` are never split (the
        boundary-layer interface must stay conforming with the quads);
        triangles that could only be fixed by splitting them are left."""
        B = 1.0 / (2.0 * math.sin(math.radians(min_angle_deg)))
        sqrt3 = math.sqrt(3.0)
        queue = deque(t for t in range(len(self.tv))
                      if self.alive[t] and self.region[t] == domain_region)
        skip, added = set(), 0
        while queue and added < max_points:
            t = queue.popleft()
            if not self.alive[t] or self.region[t] != domain_region:
                continue
            key = tuple(sorted(self.tv[t]))
            if key in skip:
                continue
            R, lmin, (ux, uy), (gx, gy) = self._tri_geometry(t)
            if not (R > size_fn(gx, gy) / sqrt3 or R / lmin > B):
                continue

            status, info = self.locate(ux, uy, start=t, stop_at_segment=True)
            if status == 'in':
                res = self._insert_at(ux, uy, info, check_encroach=True)
                if res[0] == 'ok':
                    added += 1
                    queue.extend(res[2])
                    continue
                if res[0] == 'invalid':
                    skip.add(key)
                    continue
                seg = res[1]
            elif status == 'segment':
                seg = info
            else:
                skip.add(key)
                continue

            # circumcentre encroaches / lies beyond a segment
            if self.segments.get(_ekey(*seg)) in splittable:
                new = self.split_segment(*seg)
                if new:
                    added += 1
                    queue.extend(new)
                    queue.append(t)
                    continue
            else:
                # Segment must not be split: instead insert the apex of the
                # equilateral triangle standing on it (domain side).
                new = self._insert_apex(*seg, domain_region)
                if new:
                    added += 1
                    queue.extend(new)
                    queue.append(t)
                    continue
            skip.add(key)
        return added

    def _insert_apex(self, u, w, domain_region, height=0.866):
        found = self.find_edge(u, w)
        if found is None:
            return None
        t, i = found
        if self.region[t] != domain_region:
            t = self.tn[t][i]
            if t < 0 or self.region[t] != domain_region:
                return None
        v = self.tv[t]
        k = v.index(u)
        a, b = (u, w) if v[(k + 1) % 3] == w else (w, u)   # CCW order in t
        x, y = self.x, self.y
        ex, ey = x[b] - x[a], y[b] - y[a]
        px = 0.5 * (x[a] + x[b]) - height * ey      # left normal = into t
        py = 0.5 * (y[a] + y[b]) + height * ex
        status, tt = self.locate(px, py, start=t, stop_at_segment=True)
        if status != 'in' or self.region[tt] != domain_region:
            return None
        res = self._insert_at(px, py, tt, check_encroach=True)
        return res[2] if res[0] == 'ok' else None

    # ------------------------------------------------------------------
    # output / checks
    # ------------------------------------------------------------------
    def triangles(self, region=None):
        return [list(self.tv[t]) for t in range(len(self.tv))
                if self.alive[t] and (region is None or self.region[t] == region)]

    def check(self):
        """Structural self-check: symmetric adjacency, CCW triangles."""
        for t in range(len(self.tv)):
            if not self.alive[t]:
                continue
            v = self.tv[t]
            assert self._orient(*v) > 0, f"triangle {t} not CCW"
            for i in range(3):
                nt = self.tn[t][i]
                if nt < 0:
                    continue
                assert self.alive[nt], "neighbour is dead"
                u, w = v[(i + 1) % 3], v[(i + 2) % 3]
                vv = self.tv[nt]
                j = next(k for k in range(3) if vv[k] != u and vv[k] != w)
                assert self.tn[nt][j] == t, "adjacency not symmetric"
        return True
