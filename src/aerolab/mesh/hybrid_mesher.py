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
    python src/run.py hybrid_mesher --naca 2412 --re 1e6 --yplus 1
"""

import argparse
import math
import time

import numpy as np

from aerolab.physics.boundary_layer import extrude, cells as bl_cells
from aerolab.mesh.cdt2d import CDT
from aerolab.mesh.geometry import circle, first_cell_height, naca4


def make_size_fn(h_near, grad, h_far, wake_length, wake_h, wake_grad, alpha_deg,
                 wake=None, te=(1.0, 0.0)):
    """cell size at (x, y): fine near the body, coarse at the far field, and refined
    along the wake.

    wake=None: the original strip (|n| < 0.05 + 0.15 s, s < wake_length, sharp edges).
    wake=dict: a graded wake band that follows the free stream from the trailing edge
        alphas   angles the wake must cover (one mesh for several angles: 3D wings)
        width    (w0, w1): core half-width w0 + w1 s, size h_c(s) inside it
        cross    size growth per unit distance outside the core (smooth, no edge)
        h_c(s) = wake_h + wake_grad min(s, L) + grad max(s - L, 0),  L = wake_length
    """
    ca, sa = math.cos(math.radians(alpha_deg)), math.sin(math.radians(alpha_deg))
    tx, ty = te
    if wake is not None:
        dirs = [(math.cos(math.radians(a)), math.sin(math.radians(a)))
                for a in (wake.get("alphas") or [alpha_deg])]
        w0, w1 = wake.get("width", (0.04, 0.06))
        gn = wake.get("cross", 0.25)
        L = wake_length

    def size(x, y):
        # distance to the chord line (0,0)-(1,0)
        cx = min(max(x, 0.0), 1.0)
        d = math.hypot(x - cx, y)
        h = h_near + grad * d
        xs, ys = x - tx, y - ty
        if wake is None:
            # wake: a strip leaving the TE along the freestream direction
            s = xs * ca + ys * sa             # along-wake coordinate
            n = abs(-xs * sa + ys * ca)       # cross-wake distance
            if 0.0 < s < wake_length and n < 0.05 + 0.15 * s:
                h = min(h, wake_h + wake_grad * s)
            return min(h, h_far)
        for c, sn in dirs:
            s = xs * c + ys * sn
            if s <= 0.0:
                continue
            n = abs(-xs * sn + ys * c)
            hc = wake_h + wake_grad * min(s, L) + grad * max(s - L, 0.0)
            h = min(h, hc + gn * max(0.0, n - (w0 + w1 * s)))
        return min(h, h_far)
    return size


def build_mesh(naca="2412", n_side=120, reynolds=1e6, yplus=1.0, growth=1.15,
               n_layers=60, farfield_radius=20.0, n_farfield=64,
               h_near=0.02, grad=0.12, wake_length=5.0, wake_h=0.01,
               wake_grad=0.03, alpha_deg=0.0, min_angle=25.0, seed_offset=0.87,
               wall=None, h1=None, label=None, bl_ar_cap=1.0, verbose=True, wake=None,
               size_scale=None):
    """wall: optional (M,2) closed loop spanning roughly x in [0,1] (any body);
    default is the NACA airfoil. h1: optional first-layer height (default
    from y+ and Re, which assumes a turbulent boundary layer).
    bl_ar_cap: outer boundary-layer cells are capped at this height / spacing
    ratio (1 = square). Use float('inf') for uniform layer heights (a thick
    laminar boundary layer; set its depth with n_layers).
    size_scale: optional f(x, y) multiplying the triangle size field (goal-
    oriented adaptation, adjoint_error.py)."""
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
    te = tuple(wall[int(np.argmax(wall[:, 0]))]) if wake is not None else (1.0, 0.0)
    size = make_size_fn(h_near, grad, 2 * math.pi * R / n_farfield,
                        wake_length, wake_h, wake_grad, alpha_deg, wake=wake, te=te)
    if size_scale is not None:
        base = size
        size = lambda x, y: base(x, y) * size_scale(x, y)
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


# ----------------------------------------------------------------------
# multi-element airfoils (Phase 3, E5)
# ----------------------------------------------------------------------
def _dense_walls(loops, h=0.002):
    """Points along all wall loops at spacing <= h, with their element index."""
    out, idx = [], []
    for k, L in enumerate(loops):
        a, b = L, np.roll(L, -1, axis=0)
        for p, q in zip(a, b):
            n = max(1, int(math.ceil(np.linalg.norm(q - p) / h)))
            t = np.arange(n)[:, None] / n
            out.append(p + t * (q - p))
            idx.append(np.full(n, k))
    return np.vstack(out), np.concatenate(idx)


def make_multi_size_fn(loops, tes, h_near, grad, h_far, wake_length, wake_h, wake_grad,
                       alpha_deg, fronts=None, front_grad=0.15):
    """Size field: h_near + grad * (distance to the nearest wall), and a
    refined wake strip from every element's trailing edge along the free
    stream (the slat and main-element wakes run over the next element).
    fronts: the boundary-layer edge loops - the triangles then also grow
    smoothly (front_grad) from the local spacing of the layer edge, so the
    fine edge at a cusp, a trailing edge or in a slot is not met by
    triangles 10-50x larger (the slat-cusp shear layer diverged that way on
    the coarse mesh)."""
    from scipy.spatial import cKDTree
    pts, _ = _dense_walls(loops)
    tree = cKDTree(pts)
    if fronts is not None:
        F = np.vstack(fronts)
        ell = np.concatenate([0.5 * (np.linalg.norm(np.roll(f, -1, axis=0) - f, axis=1) +
                                     np.linalg.norm(f - np.roll(f, 1, axis=0), axis=1))
                              for f in fronts])
        ftree = cKDTree(F)
    ca, sa = math.cos(math.radians(alpha_deg)), math.sin(math.radians(alpha_deg))
    tes = [np.asarray(t, float) for t in tes]

    def size(x, y):
        d = tree.query((x, y))[0]
        h = h_near + grad * d
        if fronts is not None:
            df, j = ftree.query((x, y), k=4)
            h = min(h, float(np.min(ell[j] + front_grad * df)))
        for te in tes:
            xs, ys = x - te[0], y - te[1]
            s = xs * ca + ys * sa
            n = abs(-xs * sa + ys * ca)
            if 0.0 < s < wake_length and n < 0.03 + 0.15 * s:
                h = min(h, wake_h + wake_grad * s)
        return min(h, h_far)
    return size


def build_multi_mesh(elements, reynolds=9e6, h1=None, yplus=0.7, growth=1.15, n_layers=60,
                     farfield_radius=100.0, n_farfield=48, h_near=0.03, grad=0.15,
                     wake_length=3.0, wake_h=0.015, wake_grad=0.05, alpha_deg=0.0,
                     min_angle=25.0, seed_offset=0.87, gap_fill=0.4, bl_ar_cap=1.0,
                     bl_max=0.015, label="multi-element airfoil", verbose=True):
    """Hybrid mesh around several bodies (multi_element.load_multi output).

    Each element gets its own quad boundary layer. Where elements are close
    (the slots), an element's layer may fill at most `gap_fill` of the
    distance to the nearest other element (boundary_layer.extrude
    max_thickness: lower local growth, same layer count), so the two layers
    facing each other across a slot leave room for triangles. One CDT fills
    the space between all layer edges and the farfield. The same limit keeps
    the layers out of coves and concave corners (facing_distance), and
    bl_max caps the quad layer at about the physical boundary-layer
    thickness (0.37 Re^-0.2 = 0.015 c at Re 9e6): beyond it, square outer
    quads only follow the wall shape into ragged fronts the triangles
    resolve better.
    Patches: wall_<name> per element, farfield."""
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)
    from scipy.spatial import cKDTree
    from aerolab.airfoil.multi_element import loops_intersect
    walls = [np.asarray(e["loop"], float) for e in elements]
    names = [e["name"] for e in elements]
    if h1 is None:
        h1 = first_cell_height(reynolds, yplus)
    dense, owner = _dense_walls(walls, h=5e-4)
    tree = cKDTree(dense)
    cone = math.cos(math.radians(60.0))

    def facing_distance(P, k, r_max=0.5):
        """Distance from each wall point of element k to the nearest wall
        point (any element, its own included) that lies IN FRONT of it:
        within 60 deg of its outward normal. That is the room its boundary
        layer has before it meets another layer: across a slot, a cove, or
        a concave corner. Points behind (through the body, e.g. the other
        side of a thin trailing edge) or beside it (along the surface) do
        not count."""
        L = walls[k]
        e = np.roll(L, -1, axis=0) - L
        en = np.c_[e[:, 1], -e[:, 0]] / np.linalg.norm(e, axis=1)[:, None]
        vn = en + np.roll(en, 1, axis=0)
        vn /= np.linalg.norm(vn, axis=1)[:, None]
        nrm = vn[cKDTree(L).query(P)[1]]
        d = np.full(len(P), np.inf)
        for i, (p, n) in enumerate(zip(P, nrm)):
            j = np.array(tree.query_ball_point(p, r_max), int)
            if len(j) == 0:
                continue
            v = dense[j] - p
            r = np.linalg.norm(v, axis=1)
            ok = (r > 1e-9) & (v @ n > cone * r)
            if ok.any():
                d[i] = r[ok].min()
        return d

    frac = gap_fill
    for attempt in range(6):
        layers_all, canon_all = [], []
        for k, w in enumerate(walls):
            lim = lambda P, k=k: np.minimum(bl_max, frac * facing_distance(P, k))
            layers, canon = extrude(w, h1, growth=growth, n_layers=n_layers, ar_cap=bl_ar_cap,
                                    verbose=verbose, max_thickness=lim, repair=True)
            layers_all.append(layers)
            canon_all.append(canon)
        # the layer edges must not cross each other or another element
        clash = None
        for a in range(len(walls)):
            for b in range(len(walls)):
                if a != b and (loops_intersect(layers_all[a][-1], layers_all[b][-1]) or
                               loops_intersect(layers_all[a][-1], walls[b])):
                    clash = (names[a], names[b])
        if clash is None:
            break
        frac *= 0.7
        log(f"[BL]   layer edges of '{clash[0]}' and '{clash[1]}' cross - gap fill {frac:.2f}")
    else:
        raise RuntimeError("could not fit the boundary layers between the elements")

    points, cells, offsets, front_ids = [], [], [], []
    for k, (layers, canon) in enumerate(zip(layers_all, canon_all)):
        K, M = layers.shape[0] - 1, layers.shape[1]
        off = len(points)
        points.extend(p for layer in layers for p in layer)
        cells.extend([[v + off for v in c] for c in bl_cells(K, M, canon)])
        offsets.append((off, K, M))
        log(f"[BL]   {names[k]}: {len(walls[k])} wall points, {K} layers, "
            f"first height {h1:.3e}, max thickness "
            f"{np.linalg.norm(layers[-1] - layers[0], axis=1).max():.4f}")
    n_bl = len(cells)

    ff = circle((0.5, 0.0), farfield_radius, n_farfield)
    R = farfield_radius
    cdt = CDT(0.5 - R, -R, 0.5 + R, R)
    ff_ids = [cdt.insert(x, y) for x, y in ff]
    for i in range(n_farfield):
        cdt.insert_segment(ff_ids[i], ff_ids[(i + 1) % n_farfield], 'farfield')
    for layers in layers_all:
        outer = layers[-1]
        ids = [cdt.insert(x, y) for x, y in outer]
        for i in range(len(ids)):
            cdt.insert_segment(ids[i], ids[(i + 1) % len(ids)], 'interface')
        front_ids.append(ids)
    cdt.mark_regions()

    if seed_offset > 0:                       # transition row, as in build_mesh
        for layers in layers_all:
            outer = layers[-1]
            e = np.roll(outer, -1, axis=0) - outer
            le = np.linalg.norm(e, axis=1)
            en = np.c_[e[:, 1], -e[:, 0]] / np.maximum(le, 1e-300)[:, None]
            nrm = en + np.roll(en, 1, axis=0)
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
            ds = 0.5 * (le + np.roll(le, 1))
            turn = np.einsum('ij,ij->i', en, np.roll(en, 1, axis=0))
            for (x, y), n, h, ct in zip(outer, nrm, ds, turn):
                if ct < np.cos(np.radians(25)):
                    continue
                px, py = x + seed_offset * h * n[0], y + seed_offset * h * n[1]
                status, t = cdt.locate(px, py)
                if status == 'in' and cdt.region[t] == 1:
                    cdt._insert_at(px, py, t, check_encroach=True)

    tes = [e["te"] for e in elements]
    size = make_multi_size_fn(walls, tes, h_near, grad, 2 * math.pi * R / n_farfield,
                              wake_length, wake_h, wake_grad, alpha_deg,
                              fronts=[L[-1] for L in layers_all])
    added = cdt.refine(size, min_angle_deg=min_angle, splittable=('farfield',))
    cdt.check()
    tris = cdt.triangles(region=1)
    log(f"[CDT]  {added} points inserted by refinement, {len(tris)} triangles")

    to_global = {}
    for (off, K, M), ids in zip(offsets, front_ids):
        for i, cid in enumerate(ids):
            to_global[cid] = off + K * M + i
    for tri in tris:
        for v in tri:
            if v not in to_global:
                to_global[v] = len(points)
                points.append((cdt.x[v], cdt.y[v]))
    cells += [[to_global[v] for v in tri] for tri in tris]

    used = sorted({v for c in cells for v in c})
    renum = {old: new for new, old in enumerate(used)}
    points = np.array(points)[used]
    cells = [[renum[v] for v in c] for c in cells]
    edge_count = {}
    for c in cells:
        for a, b in zip(c, c[1:] + c[:1]):
            edge_count.setdefault((min(a, b), max(a, b)), []).append((a, b))
    patches, wall_set = {}, set()
    for name, (off, K, M), canon in zip(names, offsets, canon_all):
        we = [[renum[off + canon[i]], renum[off + canon[(i + 1) % M]]] for i in range(M)
              if canon[i] != canon[(i + 1) % M]]
        wall_set |= {(min(a, b), max(a, b)) for a, b in we}
        patches[f"wall_{name}"] = np.array([edge_count[(min(a, b), max(a, b))][0] for a, b in we])
    patches["farfield"] = np.array([e[0] for k, e in edge_count.items()
                                    if len(e) == 1 and k not in wall_set])
    log(f"[MESH] {label}: {len(points)} points, {len(cells)} cells "
        f"({n_bl} boundary-layer + {len(tris)} triangles) in {time.time() - t0:.1f} s")
    return {"points": points, "cells": cells, "n_bl": n_bl, "patches": patches,
            "layers": layers_all, "elements": names, "h1": h1,
            "walls": walls, "multi": True}


def multi_mesh(elements, reynolds, level=1.0, alpha_deg=0.0, farfield=100.0, yplus=0.7,
               verbose=False, label=None):
    """The multi-element mesh settings used by the polar tool (level 1 =
    the 30P30N validation mesh, ~70k cells); level scales all spacings."""
    r = level
    return build_multi_mesh(elements, reynolds=reynolds, h1=first_cell_height(reynolds, yplus) / r,
                            growth=1.15 ** (1 / r), n_layers=int(round(80 * r)),
                            farfield_radius=farfield, n_farfield=int(round(48 * r)),
                            h_near=0.012 / r, grad=0.1 / r, wake_length=3.0,
                            wake_h=0.006 / r, wake_grad=0.04 / r, alpha_deg=alpha_deg,
                            label=label or "multi-element airfoil", verbose=verbose)


def plot_multi_mesh(mesh, path, views=None, title=None):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    P = mesh["points"]
    nb = mesh["n_bl"]
    q = [P[c] for c in mesh["cells"][:nb]]
    t = [P[c] for c in mesh["cells"][nb:]]
    walls = mesh["walls"]
    allw = np.vstack(walls)
    x0, x1 = allw[:, 0].min(), allw[:, 0].max()
    y0, y1 = allw[:, 1].min(), allw[:, 1].max()
    if views is None:
        views = [("Near field", (x0 - 0.3, x1 + 0.6), (y0 - 0.35, y1 + 0.35))]
        for w, name in zip(walls, mesh["elements"]):
            te = w[0]                              # loops start at the trailing edge
            views.append((f"{name} trailing edge", (te[0] - 0.03, te[0] + 0.03),
                          (te[1] - 0.025, te[1] + 0.025)))
    n = len(views)
    cols = 2 if n > 1 else 1
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.5 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (ttl, xl, yl) in zip(axes.flat, views):
        ax.add_collection(PolyCollection(t, facecolor="white", edgecolor="#4a6fa5", lw=0.2))
        ax.add_collection(PolyCollection(q, facecolor="#dce9f5", edgecolor="#c0504d", lw=0.2))
        for w in walls:
            ax.fill(w[:, 0], w[:, 1], color="#333333")
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
        ax.set_title(ttl)
    fig.suptitle(title or f"Hybrid mesh: {nb} boundary-layer cells (red) + "
                 f"{len(mesh['cells']) - nb} triangles (blue)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_npz(mesh, path):
    """Patches are stored as patch_<name> (single airfoil: patch_airfoil,
    patch_farfield; multi-element: patch_wall_<element>, patch_farfield)."""
    cells = mesh["cells"]
    offsets = np.cumsum([0] + [len(c) for c in cells])
    np.savez_compressed(
        path,
        points=mesh["points"],
        cell_nodes=np.concatenate(cells),
        cell_offsets=offsets,
        n_bl=mesh["n_bl"],
        **{f"patch_{k}": np.asarray(v) for k, v in mesh["patches"].items()},
    )


def load_npz(path):
    d = np.load(path)
    off = d["cell_offsets"]
    nodes = d["cell_nodes"]
    cells = [nodes[off[i]:off[i + 1]].tolist() for i in range(len(off) - 1)]
    return {"points": d["points"], "cells": cells, "n_bl": int(d["n_bl"]),
            "patches": {k[6:]: d[k] for k in d.files if k.startswith("patch_")}}


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
