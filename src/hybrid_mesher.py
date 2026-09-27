"""
Hybrid 2D airfoil mesher: structured quad boundary layer + unstructured
triangles, glued conformingly at the boundary-layer edge.

Pipeline
    1. NACA wall loop (geometry.py)
    2. Quad boundary layer, first height from target y+ (boundary_layer.py)
    3. CDT between the BL outer edge (hole, never split -> conforming with
       the quads) and a circular farfield (split freely)       (cdt2d.py)
    4. Delaunay refinement with a sizing field: fine near the airfoil,
       a refined wake, coarse at the farfield
    5. Save to a plain .npz your own solver can read

Usage
    python hybrid_mesher.py --naca 2412 --re 1e6 --yplus 1
"""

import argparse
import math
import time

import numpy as np

from boundary_layer import extrude, cells as bl_cells
from cdt2d import CDT
from geometry import circle, first_cell_height, naca4


def make_size_fn(h_near, grad, h_far, wake_length, wake_h, wake_grad, alpha_deg):
    ca, sa = math.cos(math.radians(alpha_deg)), math.sin(math.radians(alpha_deg))

    def size(x, y):
        # distance to the chord line (0,0)-(1,0)
        cx = min(max(x, 0.0), 1.0)
        d = math.hypot(x - cx, y)
        h = h_near + grad * d
        # wake: a strip leaving the TE along the freestream direction
        xs, ys = x - 1.0, y
        s = xs * ca + ys * sa             # along-wake coordinate
        n = abs(-xs * sa + ys * ca)       # cross-wake distance
        if 0.0 < s < wake_length and n < 0.05 + 0.15 * s:
            h = min(h, wake_h + wake_grad * s)
        return min(h, h_far)
    return size


def build_mesh(naca="2412", n_side=120, reynolds=1e6, yplus=1.0, growth=1.15,
               n_layers=60, farfield_radius=20.0, n_farfield=64,
               h_near=0.02, grad=0.12, wake_length=5.0, wake_h=0.01,
               wake_grad=0.03, alpha_deg=0.0, min_angle=25.0, seed_offset=0.87,
               wall=None, h1=None, label=None, bl_ar_cap=1.0, verbose=True):
    """wall: optional (M,2) closed loop spanning roughly x in [0,1] (any body);
    default is the NACA airfoil. h1: optional first-layer height (default
    from y+ and Re, which assumes a turbulent boundary layer).
    bl_ar_cap: outer boundary-layer cells are capped at this height / spacing
    ratio (1 = square). Use float('inf') for uniform layer heights (a thick
    laminar boundary layer; set its depth with n_layers)."""
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)

    # 1-2) wall + boundary layer ------------------------------------------
    if wall is None:
        wall = naca4(naca, n_side)
        label = label or f"NACA {naca}"
    else:
        wall = np.asarray(wall, float)
        if 0.5 * np.sum(wall[:, 0] * np.roll(wall[:, 1], -1)
                        - np.roll(wall[:, 0], -1) * wall[:, 1]) < 0:
            wall = wall[::-1]                            # make it counter-clockwise
        label = label or "custom wall"
    if h1 is None:
        h1 = first_cell_height(reynolds, yplus)
        h1_note = f"(y+={yplus}, Re={reynolds:.2e})"
    else:
        h1_note = "(user-set)"
    layers, canon = extrude(wall, h1, growth=growth, n_layers=n_layers, ar_cap=bl_ar_cap,
                            verbose=verbose)
    K, M = layers.shape[0] - 1, layers.shape[1]
    points = [p for layer in layers for p in layer]           # index k*M + i
    cells = bl_cells(K, M, canon)
    n_bl = len(cells)
    log(f"[BL]   {label}: {len(wall)} wall points ({M - len(wall)} fan copies), "
        f"first height {h1:.3e} {h1_note}, {K} layers, {n_bl} cells")

    # 3) constrained triangulation ----------------------------------------
    outer = layers[-1]
    ff = circle((0.5, 0.0), farfield_radius, n_farfield)
    R = farfield_radius
    cdt = CDT(0.5 - R, -R, 0.5 + R, R)
    ff_ids = [cdt.insert(x, y) for x, y in ff]
    bl_ids = [cdt.insert(x, y) for x, y in outer]
    for i in range(n_farfield):
        cdt.insert_segment(ff_ids[i], ff_ids[(i + 1) % n_farfield], 'farfield')
    for i in range(M):
        cdt.insert_segment(bl_ids[i], bl_ids[(i + 1) % M], 'interface')
    cdt.mark_regions()

    # Transition row: one point per BL-edge node, offset outward by
    # seed_offset x local spacing, so the first triangles sitting on the
    # interface are near-equilateral and sized like the last quad layer.
    if seed_offset > 0:
        e = np.roll(outer, -1, axis=0) - outer
        le = np.linalg.norm(e, axis=1)
        en = np.c_[e[:, 1], -e[:, 0]] / le[:, None]
        nrm = en + np.roll(en, 1, axis=0)
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        ds = 0.5 * (le + np.roll(le, 1))
        turn = np.einsum('ij,ij->i', en, np.roll(en, 1, axis=0))   # cos of front turn
        for (x, y), n, h, ct in zip(outer, nrm, ds, turn):
            if ct < np.cos(np.radians(25)):
                continue        # sharp front corner (TE fan): let refinement decide
            px, py = x + seed_offset * h * n[0], y + seed_offset * h * n[1]
            status, t = cdt.locate(px, py)
            if status == 'in' and cdt.region[t] == 1:
                # rejected (no-op) if it would encroach the interface
                cdt._insert_at(px, py, t, check_encroach=True)

    # 4) refinement ---------------------------------------------------------
    size = make_size_fn(h_near, grad, 2 * math.pi * R / n_farfield,
                        wake_length, wake_h, wake_grad, alpha_deg)
    added = cdt.refine(size, min_angle_deg=min_angle, splittable=('farfield',))
    cdt.check()
    tris = cdt.triangles(region=1)
    log(f"[CDT]  {added} points inserted by refinement, {len(tris)} triangles")

    # merge: CDT vertex id -> global point id
    to_global = {cid: K * M + i for i, cid in enumerate(bl_ids)}
    for tri in tris:
        for v in tri:
            if v not in to_global:
                to_global[v] = len(points)
                points.append((cdt.x[v], cdt.y[v]))
    cells += [[to_global[v] for v in tri] for tri in tris]

    # drop unused points (fan duplicates on the wall) and renumber
    used = sorted({v for c in cells for v in c})
    renum = {old: new for new, old in enumerate(used)}
    points = np.array(points)[used]
    cells = [[renum[v] for v in c] for c in cells]
    wall_edges = [[renum[canon[i]], renum[canon[(i + 1) % M]]] for i in range(M)
                  if canon[i] != canon[(i + 1) % M]]

    # boundary patches (edge lists, oriented with the owning cell CCW)
    edge_count = {}
    for c in cells:
        for a, b in zip(c, c[1:] + c[:1]):
            edge_count.setdefault((min(a, b), max(a, b)), []).append((a, b))
    wall_set = {(min(a, b), max(a, b)) for a, b in wall_edges}
    farfield_edges = [e[0] for k, e in edge_count.items()
                      if len(e) == 1 and k not in wall_set]
    wall_edges = [edge_count[(min(a, b), max(a, b))][0] for a, b in wall_edges]

    log(f"[MESH] {len(points)} points, {len(cells)} cells "
        f"({n_bl} boundary-layer + {len(tris)} triangles) in {time.time() - t0:.1f} s")
    return {
        "points": points,
        "cells": cells,
        "n_bl": n_bl,
        "patches": {"airfoil": np.array(wall_edges), "farfield": np.array(farfield_edges)},
        "layers": layers,
        "h1": h1,
    }


def save_npz(mesh, path):
    cells = mesh["cells"]
    offsets = np.cumsum([0] + [len(c) for c in cells])
    np.savez_compressed(
        path,
        points=mesh["points"],
        cell_nodes=np.concatenate(cells),
        cell_offsets=offsets,
        patch_airfoil=mesh["patches"]["airfoil"],
        patch_farfield=mesh["patches"]["farfield"],
        n_bl=mesh["n_bl"],
    )


def load_npz(path):
    d = np.load(path)
    off = d["cell_offsets"]
    nodes = d["cell_nodes"]
    cells = [nodes[off[i]:off[i + 1]].tolist() for i in range(len(off) - 1)]
    return {"points": d["points"], "cells": cells, "n_bl": int(d["n_bl"]),
            "patches": {"airfoil": d["patch_airfoil"], "farfield": d["patch_farfield"]}}


def plot_mesh(mesh, path):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    P = mesh["points"]
    nb = mesh["n_bl"]
    q = [P[c] for c in mesh["cells"][:nb]]
    t = [P[c] for c in mesh["cells"][nb:]]
    views = [("Full domain", (-20, 21), (-20.5, 20.5)),
             ("Near field", (-0.6, 2.2), (-0.8, 0.8)),
             ("Leading edge", (-0.12, 0.16), (-0.12, 0.13)),
             ("Trailing edge", (0.88, 1.16), (-0.12, 0.13))]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (title, xl, yl) in zip(axes.flat, views):
        ax.add_collection(PolyCollection(t, facecolor="white", edgecolor="#4a6fa5", lw=0.25))
        ax.add_collection(PolyCollection(q, facecolor="#dce9f5", edgecolor="#c0504d", lw=0.25))
        wall = mesh["layers"][0]
        ax.fill(wall[:, 0], wall[:, 1], color="#333333")
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
        ax.set_title(title)
    fig.suptitle(f"Hybrid mesh: {nb} boundary-layer cells (red) + "
                 f"{len(mesh['cells']) - nb} triangles (blue)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--naca", default="2412")
    ap.add_argument("--n-side", type=int, default=120, help="points per surface side")
    ap.add_argument("--re", type=float, default=1e6)
    ap.add_argument("--yplus", type=float, default=1.0)
    ap.add_argument("--growth", type=float, default=1.15)
    ap.add_argument("--layers", type=int, default=60, help="maximum BL layers")
    ap.add_argument("--farfield", type=float, default=20.0, help="radius in chords")
    ap.add_argument("--alpha", type=float, default=0.0, help="AoA for wake refinement (deg)")
    ap.add_argument("--out", default="airfoil_mesh")
    a = ap.parse_args()

    mesh = build_mesh(naca=a.naca, n_side=a.n_side, reynolds=a.re, yplus=a.yplus,
                      growth=a.growth, n_layers=a.layers, farfield_radius=a.farfield,
                      alpha_deg=a.alpha)
    save_npz(mesh, a.out + ".npz")
    plot_mesh(mesh, a.out + ".png")
    print(f"saved {a.out}.npz and {a.out}.png")
