"""Post-processing for the results viewer: every result on disk, 2D and 3D, read
without VTK/PyVista (numpy only), and rendered as matplotlib / plotly figures.

  find_results(root)            every viewable result under cases/
  Case2D(folder)                a 2D case: mesh, cell fields, walls, history
  Case3D(file)                  a 3D wing solution (fields_a*.npz from fv3d.wing3d)

2D cases are the solver's VTU output (results.vtu, else mesh.vtu); 3D cases are the
npz files fv3d.wing3d writes next to wing3d.json.
"""
import base64
import json
import re
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# readers
# ----------------------------------------------------------------------
_DT = {"Float64": np.float64, "Float32": np.float32, "Int64": np.int64, "Int32": np.int32,
       "UInt8": np.uint8, "UInt64": np.uint64}
_ARR = re.compile(r'<DataArray type="(\w+)"(?: Name="([^"]*)")?(?: NumberOfComponents="(\d+)")?'
                  r' format="binary">([^<]*)</DataArray>')


def _arrays(xml):
    out = []
    for typ, name, nc, data in _ARR.findall(xml):
        a = np.frombuffer(base64.b64decode(data.strip())[8:], _DT[typ])
        nc = int(nc or 1)
        out.append((name, a.reshape(-1, nc) if nc > 1 else a))
    return out


def _section(txt, tag):
    i, j = txt.find(f"<{tag}>"), txt.find(f"</{tag}>")
    return txt[i:j] if i >= 0 and j >= 0 else ""


def read_vtu(path):
    """(points (n,3), cells: list of int arrays, cell data {name: array}) of a VTU
    written by io_vtk.write_vtu"""
    txt = Path(path).read_text()
    data = dict(_arrays(_section(txt, "CellData")))
    pts = _arrays(_section(txt, "Points"))[0][1]
    cl = dict(_arrays(_section(txt, "Cells")))
    conn, offs = cl["connectivity"], cl["offsets"]
    start = np.r_[0, offs[:-1]]
    cells = [conn[a:b] for a, b in zip(start, offs)]
    return pts, cells, data


def read_vtp_lines(path):
    """(points (n,3), lines (m,2), cell data) of a VTP written by io_vtk.write_vtp_lines"""
    txt = Path(path).read_text()
    data = dict(_arrays(_section(txt, "CellData")))
    pts = _arrays(_section(txt, "Points"))[0][1]
    ln = dict(_arrays(_section(txt, "Lines")))
    return pts, ln["connectivity"].reshape(-1, 2), data


# ----------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------
def find_results(root):
    """every viewable result under root, newest first: dicts with kind ('2d' / '3d'),
    path, label, group, mtime"""
    root = Path(root)
    out = []
    for f in root.rglob("*.vtu"):
        if f.name not in ("results.vtu", "mesh.vtu"):
            continue
        if f.name == "mesh.vtu" and (f.parent / "results.vtu").exists():
            continue
        rel = f.parent.relative_to(root)
        parts = rel.parts
        label = " / ".join(parts) if parts else f.parent.name
        out.append({"kind": "2d", "path": str(f.parent), "label": label + ("" if f.name == "results.vtu"
                                                                          else "  (mesh only)"),
                    "group": parts[0] if parts else f.parent.name, "mtime": f.stat().st_mtime})
    for f in root.rglob("fields_a*.npz"):
        rel = f.parent.relative_to(root)
        a = f.stem[len("fields_a"):]
        out.append({"kind": "3d", "path": str(f), "label": f"{' / '.join(rel.parts)}  α {a}°  (3D)",
                    "group": rel.parts[0] if rel.parts else f.parent.name, "mtime": f.stat().st_mtime})
    return sorted(out, key=lambda r: -r["mtime"])


# ----------------------------------------------------------------------
# 2D
# ----------------------------------------------------------------------
class Case2D:
    def __init__(self, folder):
        self.folder = Path(folder)
        f = self.folder / "results.vtu"
        self.has_results = f.exists()
        P, self.cells, self.data = read_vtu(f if self.has_results else self.folder / "mesh.vtu")
        self.P = P[:, :2]
        if self.has_results and (self.folder / "mesh.vtu").exists():
            # mesh-quality fields of the mesh file too
            _, _, q = read_vtu(self.folder / "mesh.vtu")
            for k, v in q.items():
                self.data.setdefault(k, v)
        self.centres = np.array([self.P[c].mean(0) for c in self.cells])
        cj = self.folder / "case.json"
        self.meta = json.loads(cj.read_text()) if cj.exists() else {}
        b = self.folder / ("results_boundary.vtp" if self.has_results else "mesh_boundary.vtp")
        self.boundary = read_vtp_lines(b) if b.exists() else None

    def fields(self):
        """scalar fields to colour by (vectors as magnitude), flow fields first"""
        first = ["Cp", "U_mag", "p", "vorticity", "nut_ratio", "Mach", "continuity_error"]
        names = [k for k in self.data if k != "U"]
        return [k for k in first if k in names] + [k for k in names if k not in first]

    def values(self, field):
        v = self.data[field]
        return np.linalg.norm(v[:, :2], axis=1) if v.ndim > 1 else v

    def walls(self):
        """{patch: DataFrame-like dict of arrays} from wall_*.csv"""
        out = {}
        for f in sorted(self.folder.glob("wall_*.csv")):
            rows = np.genfromtxt(f, delimiter=",", names=True)
            out[f.stem[5:]] = {n: rows[n] for n in rows.dtype.names}
        return out

    def history(self):
        f = self.folder / "history.csv"
        if not f.exists():
            return None
        rows = np.genfromtxt(f, delimiter=",", names=True)
        return {n: rows[n] for n in rows.dtype.names}

    def extent(self):
        """chord-scale box of the wall (for view windows)"""
        if self.boundary is None:
            return (self.P.min(0), self.P.max(0))
        pts, lines, d = self.boundary
        names = self.meta.get("patches", [])
        pid = d.get("patch_id")
        walls = [i for i, n in enumerate(names) if n not in ("farfield", "inlet", "outlet")]
        sel = np.isin(pid, walls) if pid is not None and walls else np.ones(len(lines), bool)
        q = pts[np.unique(lines[sel])][:, :2]
        return q.min(0), q.max(0)


VIEWS = {"Full": None, "Near": (-0.4, 1.6, -0.5, 0.5), "Leading edge": (-0.06, 0.14, -0.08, 0.08),
         "Trailing edge": (0.9, 1.1, -0.06, 0.06), "Wake": (0.8, 3.0, -0.6, 0.6)}


def view_box(case, view):
    """(x0, x1, y0, y1) of a named view, scaled to the wall's size"""
    lo, hi = case.extent()
    c = max(hi[0] - lo[0], 1e-9)
    if view == "Full":
        return case.P[:, 0].min(), case.P[:, 0].max(), case.P[:, 1].min(), case.P[:, 1].max()
    x0, x1, y0, y1 = VIEWS[view]
    return lo[0] + c * x0, lo[0] + c * x1, c * y0 + 0.5 * (lo[1] + hi[1]), c * y1 + 0.5 * (lo[1] + hi[1])


def _clim(field, v):
    lo, hi = np.nanpercentile(v, [1, 99]) if len(v) else (0.0, 1.0)
    if field == "Cp":
        lo, hi = max(lo, -4.0), min(hi, 1.0)
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


def field_figure(P, cells, values, box, field="", cmap="turbo", edges=False, mesh_only=False,
                 boundary=None, streamlines=None, clim=None, xlabel="x/c", ylabel="y/c", figsize=(12, 5.2)):
    """matplotlib figure of cell values (or the bare mesh) in the box (x0, x1, y0, y1).
    streamlines: (centres, U2) to draw streamlines from cell velocities"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    x0, x1, y0, y1 = box
    cells = [np.asarray(c)[np.asarray(c) >= 0] for c in cells]
    lo = np.array([P[c].min(0) for c in cells])
    hi = np.array([P[c].max(0) for c in cells])
    sel = np.where((hi[:, 0] >= x0) & (lo[:, 0] <= x1) & (hi[:, 1] >= y0) & (lo[:, 1] <= y1))[0]
    polys = [P[cells[i]] for i in sel]
    fig, ax = plt.subplots(figsize=figsize)
    if mesh_only:
        pc = PolyCollection(polys, facecolors="#f4f5f7", edgecolors="#30343a", linewidths=0.25)
        ax.add_collection(pc)
    else:
        v = values[sel]
        lo, hi = clim or _clim(field, v)
        pc = PolyCollection(polys, array=v, cmap=cmap, edgecolors="#20242a" if edges else "face",
                            linewidths=0.15 if edges else 0.05)
        pc.set_clim(lo, hi)
        ax.add_collection(pc)
        fig.colorbar(pc, ax=ax, label=field, shrink=0.85, pad=0.015)
    if boundary is not None:
        bp, bl = boundary
        ax.add_collection(LineCollection(bp[bl][:, :, :2], colors="k", linewidths=0.9))
    if streamlines is not None and not mesh_only:
        c, U2 = streamlines
        try:
            from scipy.interpolate import griddata
            gx, gy = np.meshgrid(np.linspace(x0, x1, 220), np.linspace(y0, y1, 120))
            near = sel
            gu = griddata(c[near], U2[near, 0], (gx, gy), method="linear")
            gv = griddata(c[near], U2[near, 1], (gx, gy), method="linear")
            ax.streamplot(gx, gy, np.nan_to_num(gu), np.nan_to_num(gv), density=1.6, color="w",
                          linewidth=0.5, arrowsize=0.6)
        except Exception:                                   # noqa: BLE001
            pass
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def figure_2d(case, field, view="Near", box=None, **kw):
    box = box or view_box(case, view)
    st = None
    if kw.pop("streamlines", False) and "U" in case.data:
        st = (case.centres, case.data["U"][:, :2])
    bnd = None
    if case.boundary is not None:
        bp, bl, _ = case.boundary
        bnd = (bp, bl)
    mesh_only = field is None
    return field_figure(case.P, case.cells, None if mesh_only else case.values(field), box, field=field or "",
                        mesh_only=mesh_only, boundary=bnd, streamlines=st, **kw)


# ----------------------------------------------------------------------
# 3D
# ----------------------------------------------------------------------
class Case3D:
    """a 3D wing solution saved by fv3d.wing3d (fields_a<alpha>.npz)"""

    def __init__(self, path):
        self.path = Path(path)
        d = np.load(self.path, allow_pickle=False)
        self.d = {k: d[k] for k in d.files}
        self.info = json.loads(str(self.d["info"]))
        self.P = self.d["points"].astype(float)
        self.n2 = int(self.d["n2"])
        self.cells2 = self.d["cells2"]
        self.layer = self.d["cell_layer"]
        self.c2 = self.d["cell_2d"]
        self.z = self.d["z_levels"].astype(float)
        self.U = self.d["U"].astype(float)
        self.p = self.d["p"].astype(float)
        self.nut = self.d["nut"].astype(float)
        self.nu = float(self.info["nu"])

    def fields(self):
        f = ["Cp", "U_mag", "Ux", "Uy", "Uz", "nut_ratio"]
        return f + (["vorticity_x", "vorticity_mag"] if "vorticity" in self.d else [])

    def values(self, field, idx=None):
        idx = slice(None) if idx is None else idx
        if field == "Cp":
            return 2.0 * self.p[idx]
        if field == "U_mag":
            return np.linalg.norm(self.U[idx], axis=1)
        if field in ("Ux", "Uy", "Uz"):
            return self.U[idx, "xyz".index(field[1])]
        if field == "nut_ratio":
            return self.nut[idx] / self.nu
        if field == "vorticity_x":
            return self.d["vorticity"][idx, 0].astype(float)
        if field == "vorticity_mag":
            return np.linalg.norm(self.d["vorticity"][idx].astype(float), axis=1)
        raise KeyError(field)

    # -- spanwise cut: one layer of the extruded mesh -----------------------------------
    def layer_index(self, zc):
        mid = 0.5 * (self.z[1:] + self.z[:-1])
        return int(np.argmin(np.abs(mid - zc)))

    def span_cut(self, l):
        """(2D points at the layer's mid height, cell polygons, 3D cell ids) of layer l"""
        ids = np.where(self.layer == l)[0]
        lo = self.P[l * self.n2:(l + 1) * self.n2]
        hi = self.P[(l + 1) * self.n2:(l + 2) * self.n2]
        P2 = 0.5 * (lo + hi)[:, :2]
        cells = [r[r >= 0] for r in self.cells2[self.c2[ids]]]
        return P2, cells, ids

    # -- crossflow cut x = const: nearest-cell sampling on a (z, y) grid -------------------
    def x_cut(self, x, field, zr=None, yr=(-0.6, 0.6), nz=320, ny=200):
        from scipy.spatial import cKDTree
        cen = self.d["centres"].astype(float)
        if not hasattr(self, "_tree"):
            self._tree = cKDTree(cen)
        zr = zr or (0.0, float(self.info["semi_span"]) * 1.4)
        Z, Y = np.meshgrid(np.linspace(*zr, nz), np.linspace(*yr, ny))
        q = np.c_[np.full(Z.size, x), Y.ravel(), Z.ravel()]
        dist, j = self._tree.query(q, k=4)
        w = 1.0 / np.maximum(dist, 1e-9)
        w /= w.sum(1, keepdims=True)
        v = (self.values(field, j.ravel()).reshape(j.shape) * w).sum(1).reshape(Z.shape)
        Ui = (self.U[j] * w[..., None]).sum(1)
        inside = self._solid_mask(q).reshape(Z.shape)
        v = np.where(inside, np.nan, v)
        return Z, Y, v, Ui[:, 2].reshape(Z.shape), Ui[:, 1].reshape(Z.shape), inside

    def _solid_mask(self, q):
        """points inside the wing (between the upper and lower wall at that station)"""
        w = self.d["wall_centres"].astype(float)
        s = float(self.info["semi_span"])
        m = np.zeros(len(q), bool)
        inspan = q[:, 2] <= s
        if not inspan.any():
            return m
        from scipy.spatial import cKDTree
        t = cKDTree(w[:, [0, 2]])
        # local section: wall points near (x, z); solid if y between their min and max
        dist, nb = t.query(q[inspan][:, [0, 2]], k=12)
        ys = w[nb, 1]
        xs = w[nb, 0]
        # only over the section itself (not in the wake behind the trailing edge)
        over = (q[inspan, 0] >= xs.min(1)) & (q[inspan, 0] <= xs.max(1)) & (dist[:, 0] < 0.05)
        m[np.where(inspan)[0]] = over & (q[inspan, 1] > ys.min(1)) & (q[inspan, 1] < ys.max(1))
        return m

    # -- surface ------------------------------------------------------------------------
    def surface(self, field="Cp"):
        """triangles of the wing surface and per-triangle values (Cp or Cf)"""
        F = self.d["wall_faces"]
        tris, src = [], []
        for i, f in enumerate(F):
            f = f[f >= 0]
            for k in range(1, len(f) - 1):
                tris.append((f[0], f[k], f[k + 1]))
                src.append(i)
        tris, src = np.array(tris), np.array(src)
        v = self.d["wall_" + field].astype(float)[src]
        return tris, v

    def symmetry_edges(self, max_r=None):
        """edges of the mesh on the symmetry plane (z = 0) for a mesh view"""
        cells = self.cells2
        segs = []
        for c in cells:
            c = c[c >= 0]
            for a, b in zip(c, np.roll(c, -1)):
                if a < b:
                    segs.append((a, b))
        segs = np.unique(np.array(segs), axis=0)
        P = self.P[:self.n2]
        if max_r:
            keep = (np.abs(P[segs[:, 0], 0] - 0.5) < max_r) & (np.abs(P[segs[:, 0], 1]) < max_r)
            segs = segs[keep]
        return P, segs


def surface_figure(case, field="Cp", edges=False, height=620, sym_mesh=False):
    """plotly 3D view of the wing surface coloured by Cp / Cf"""
    import plotly.graph_objects as go
    tris, v = case.surface(field)
    P = case.P
    lo, hi = np.nanpercentile(v, [1, 99])
    if field == "Cp":
        lo, hi = max(lo, -3.0), min(hi, 1.0)
    fig = go.Figure(go.Mesh3d(x=P[:, 0], y=P[:, 2], z=P[:, 1], i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
                              intensity=v, intensitymode="cell", colorscale="Turbo", cmin=lo, cmax=hi,
                              colorbar=dict(title=field), flatshading=True,
                              hovertemplate=f"{field} %{{intensity:.3f}}<extra></extra>"))
    if edges:
        F = case.d["wall_faces"]
        xs, ys, zs = [], [], []
        for f in F:
            f = f[f >= 0]
            q = P[np.r_[f, f[:1]]]
            xs += list(q[:, 0]) + [None]
            ys += list(q[:, 2]) + [None]
            zs += list(q[:, 1]) + [None]
        fig.add_scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="rgba(20,20,25,0.45)", width=1),
                          hoverinfo="skip", showlegend=False)
    if sym_mesh:
        Ps, segs = case.symmetry_edges(max_r=1.5)
        q = Ps[segs]
        xs = np.c_[q[:, 0, 0], q[:, 1, 0], np.full(len(q), np.nan)].ravel()
        zs = np.c_[q[:, 0, 1], q[:, 1, 1], np.full(len(q), np.nan)].ravel()
        fig.add_scatter3d(x=xs, y=np.zeros_like(xs), z=zs, mode="lines",
                          line=dict(color="rgba(60,70,80,0.35)", width=1), hoverinfo="skip", showlegend=False)
    s = float(case.info["semi_span"])
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0),
                      scene=dict(xaxis_title="x/c", yaxis_title="z/c (span)", zaxis_title="y/c",
                                 aspectmode="data", camera=dict(eye=dict(x=-1.2, y=-1.6, z=1.0))),
                      title=f"{field} on the wing, α {case.info['alpha']:g}°, half span {s:g} c")
    return fig


def xcut_figure(case, x, field="vorticity_x", zr=None, yr=(-0.6, 0.6), cmap="RdBu_r", vectors=True):
    """matplotlib crossflow plane x = const (tip vortex), in-plane velocity arrows"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if field == "vorticity_x" and "vorticity" not in case.d:
        field = "U_mag"
    Z, Y, v, W, V, solid = case.x_cut(x, field, zr, yr)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    lo, hi = np.nanpercentile(v, [1, 99])
    if field == "vorticity_x":
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    im = ax.pcolormesh(Z, Y, v, cmap=cmap if field == "vorticity_x" else "turbo", vmin=lo, vmax=hi,
                       shading="auto")
    fig.colorbar(im, ax=ax, label=field, shrink=0.85, pad=0.015)
    if vectors:
        sk = (slice(None, None, 10), slice(None, None, 10))
        ax.quiver(Z[sk], Y[sk], np.where(solid, 0, W)[sk], np.where(solid, 0, V)[sk], color="k",
                  scale=6, width=0.0015)
    ax.axvline(float(case.info["semi_span"]), color="k", lw=0.6, ls=":")
    ax.set_xlabel("z/c (span)")
    ax.set_ylabel("y/c")
    ax.set_aspect("equal")
    ax.set_title(f"Crossflow plane x/c = {x:g}")
    fig.tight_layout()
    return fig
