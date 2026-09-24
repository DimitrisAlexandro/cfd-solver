"""
Independent checker for a saved 2D mesh (.npz from hybrid_mesher.py).
It only reads the file - it shares no code with the mesher - so it checks
what your solver will actually receive.

Validity (hard errors):
  * every cell has positive area (counter-clockwise), quads are convex
  * conformity: every edge is shared by exactly 2 cells, or it is a
    boundary edge listed in exactly one patch (no hanging nodes, no gaps)
  * coverage: sum of cell areas == farfield polygon area - airfoil area
  * boundary faces point outward

Quality (what an FVM solver cares about):
  * triangle minimum angle
  * non-orthogonality: angle between face normal and the line joining
    the two cell centroids (drives the gradient correction error)
  * neighbour size ratio in the triangle region and across the BL edge
    (sudden jumps hurt accuracy)
  * quad aspect ratio (large is fine inside the boundary layer)

Usage:  python mesh_check.py airfoil_mesh.npz
"""

import sys

import numpy as np


def poly_area_centroid(P):
    x, y = P[:, 0], P[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    A = 0.5 * cr.sum()
    cx = ((x + xn) * cr).sum() / (6 * A)
    cy = ((y + yn) * cr).sum() / (6 * A)
    return A, np.array([cx, cy])


def check(path, verbose=True):
    d = np.load(path)
    P = d["points"]
    off, nodes = d["cell_offsets"], d["cell_nodes"]
    cells = [nodes[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    patches = {"airfoil": d["patch_airfoil"], "farfield": d["patch_farfield"]}
    n_bl = int(d["n_bl"]) if "n_bl" in d else 0     # cells [0, n_bl) = boundary layer
    errors = []

    areas, cents = np.empty(len(cells)), np.empty((len(cells), 2))
    for i, c in enumerate(cells):
        areas[i], cents[i] = poly_area_centroid(P[c])
    if (areas <= 0).any():
        errors.append(f"{(areas <= 0).sum()} cells with non-positive area")

    nonconvex = 0
    for c in cells:
        if len(c) == 4:
            Q = P[c]
            e = np.roll(Q, -1, 0) - Q
            cr = e[:, 0] * np.roll(e, -1, 0)[:, 1] - e[:, 1] * np.roll(e, -1, 0)[:, 0]
            nonconvex += (cr <= 0).any()
    if nonconvex:
        errors.append(f"{nonconvex} non-convex quads")

    # edges -> cells
    edges = {}
    for ci, c in enumerate(cells):
        for a, b in zip(c, np.roll(c, -1)):
            edges.setdefault((min(a, b), max(a, b)), []).append((ci, a, b))
    bad = [k for k, v in edges.items() if len(v) > 2]
    if bad:
        errors.append(f"{len(bad)} edges shared by more than 2 cells")

    boundary = {k for k, v in edges.items() if len(v) == 1}
    patch_edges = {}
    for name, E in patches.items():
        for a, b in E:
            k = (min(a, b), max(a, b))
            if k in patch_edges:
                errors.append(f"edge {k} in two patches")
            patch_edges[k] = name
    if boundary != set(patch_edges):
        errors.append(f"boundary edges not matching patches: "
                      f"{len(boundary - set(patch_edges))} unlisted (holes/hanging nodes), "
                      f"{len(set(patch_edges) - boundary)} listed but internal")

    # outward boundary normals + coverage
    def loop_area(E):
        return 0.5 * sum(P[a, 0] * P[b, 1] - P[b, 0] * P[a, 1] for a, b in E)
    for name, E in patches.items():
        for a, b in E:
            ci = edges[(min(a, b), max(a, b))][0][0]
            t = P[b] - P[a]
            n = np.array([t[1], -t[0]])
            if n @ (0.5 * (P[a] + P[b]) - cents[ci]) <= 0:
                errors.append(f"inward-pointing boundary face on {name}")
                break
    expected = loop_area(patches["farfield"]) + loop_area(patches["airfoil"])
    if abs(areas.sum() - expected) > 1e-9 * abs(expected):
        errors.append(f"area mismatch: cells {areas.sum():.12g} vs boundary {expected:.12g}")

    # quality -------------------------------------------------------------
    min_ang = []
    for c in cells[n_bl:]:                          # unstructured triangles only
        if len(c) == 3:
            T = P[c]
            angs = []
            for i in range(3):
                e1, e2 = T[(i + 1) % 3] - T[i], T[(i + 2) % 3] - T[i]
                angs.append(np.degrees(np.arccos(np.clip(
                    e1 @ e2 / np.linalg.norm(e1) / np.linalg.norm(e2), -1, 1))))
            min_ang.append(min(angs))
    min_ang = np.array(min_ang)

    nonorth, ratio, iface_ratio = [], [], []
    for k, v in edges.items():
        if len(v) != 2:
            continue
        (c1, a, b), (c2, _, _) = v
        t = P[b] - P[a]
        n = np.array([t[1], -t[0]]) / np.linalg.norm(t)
        dvec = cents[c2] - cents[c1]
        nonorth.append(np.degrees(np.arccos(min(1.0, abs(n @ dvec) / np.linalg.norm(dvec)))))
        r = max(areas[c1], areas[c2]) / min(areas[c1], areas[c2])
        if c1 >= n_bl and c2 >= n_bl:               # core triangles (BL is
            ratio.append(r)                         # anisotropic by design)
        if (c1 < n_bl) != (c2 < n_bl):              # BL cell next to a core triangle
            iface_ratio.append(r)
    nonorth, ratio = np.array(nonorth), np.array(ratio)

    ar = []
    for c in cells:
        if len(c) == 4:
            Q = P[c]
            L = np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1)
            ar.append(max(L) / min(L))
    ar = np.array(ar)

    stats = {
        "cells": len(cells), "quads": len(ar), "triangles": len(min_ang),
        "tri_min_angle_min": min_ang.min(), "tri_min_angle_p1": np.percentile(min_ang, 1),
        "tri_below_20deg": int((min_ang < 20).sum()),
        "nonorth_max": nonorth.max(), "nonorth_mean": nonorth.mean(),
        "faces_nonorth_gt_45": int((nonorth > 45).sum()),
        "core_size_ratio_max": ratio.max(), "core_size_ratio_p99": np.percentile(ratio, 99),
        "interface_size_ratio_max": max(iface_ratio) if iface_ratio else 0,
        "quad_aspect_max": ar.max(),
    }
    if verbose:
        print("VALIDITY:", "OK" if not errors else "FAILED")
        for e in errors:
            print("  -", e)
        print("QUALITY:")
        for k, v in stats.items():
            print(f"  {k:26s} {v:.4g}" if isinstance(v, float) else f"  {k:26s} {v}")
    return errors, stats


if __name__ == "__main__":
    errs, _ = check(sys.argv[1] if len(sys.argv) > 1 else "airfoil_mesh.npz")
    sys.exit(1 if errs else 0)
