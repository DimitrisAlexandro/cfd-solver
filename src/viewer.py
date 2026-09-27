"""
Pre- and post-processing viewer (PyVista / VTK).

Open a case folder (results if present, else the mesh), a .vtm/.vtu file,
or a mesher .npz:
    python viewer.py ../cases/cyl_re40
    python viewer.py ../cases/cyl_re40 --field Cp --streamlines
    python viewer.py ../airfoil_mesh.npz                      # mesh only
    python viewer.py ../cases/cyl_re40 --plots                # wall + history plots
    python viewer.py ../cases/cyl_re40 --monitor              # live residuals while running
    python viewer.py CASE --field Cp --view near --screenshot cp.png   # no window
    python viewer.py CASE --plots --screenshot plots.png               # plots to PNG
    python viewer.py CASE --animate vorticity                          # transient: GIF

Keys in the 3D window
    n / b     next / previous field          g   toggle cell edges
    1 2 3 4 5 full / near / leading edge / trailing edge / wake view
    j         jump to the next-worst cell for the current field (quality check)
    m         toggle streamlines              o   open wall + history plots
    x         save screenshot to the case folder
    mouse     left-drag pan, scroll zoom      (q or close window to quit)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv

PATCH_COLORS = {"wall": "white", "inlet": "deepskyblue", "outlet": "tomato",
                "airfoil": "white", "cylinder": "white", "farfield": "silver"}
BACKGROUND = "#2e3238"
QUALITY_FIELDS = {"min_angle", "nonorthogonality", "aspect_ratio", "size_ratio",
                  "cell_area", "cell_type", "boundary_layer", "continuity_error"}

# how to colour known fields: (cmap, clim or None, "low"/"high" = which end is bad)
FIELD_STYLE = {
    "min_angle": ("RdYlGn", (0, 60), "low"),
    "nonorthogonality": ("RdYlGn_r", (0, 70), "high"),
    "aspect_ratio": ("viridis", None, "high"),
    "size_ratio": ("RdYlGn_r", (1, 5), "high"),
    "cell_area": ("viridis", None, None),
    "boundary_layer": ("coolwarm", (0, 1), None),
    "cell_type": ("coolwarm", (3, 4), None),
    "U_mag": ("turbo", None, None),
    "Cp": ("RdBu_r", "sym", None),
    "p": ("RdBu_r", "sym", None),
    "vorticity": ("RdBu_r", "sym", None),
    "continuity_error": ("RdBu_r", "sym", "abs"),
}
MESH_FIELDS_FIRST = ["min_angle", "nonorthogonality", "size_ratio", "aspect_ratio"]
VIEW_KEYS = {"full": "1", "near": "2", "le": "3", "te": "4", "wake": "5"}
FLOW_FIELDS_FIRST = ["U_mag", "Cp", "vorticity", "p"]


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------
def _grid_from_arrays(points, cells):
    P = np.c_[points, np.zeros(len(points))]
    flat, types = [], []
    for c in cells:
        flat += [len(c), *c]
        types.append({3: pv.CellType.TRIANGLE, 4: pv.CellType.QUAD}.get(len(c), pv.CellType.POLYGON))
    return pv.UnstructuredGrid(np.array(flat), np.array(types, np.uint8), P)


def load(path):
    """Returns (grid, boundary PolyData or None, patch names, meta dict, case folder)."""
    path = Path(path)
    folder, meta = None, {}
    if path.is_dir():
        folder = path
        vtm = path / "results.vtm" if (path / "results.vtm").exists() else path / "mesh.vtm"
        if (path / "case.json").exists():
            meta = json.loads((path / "case.json").read_text())
        mb = pv.read(vtm)
        return mb["cells"], mb["boundary"], meta.get("patches", []), meta, folder
    if path.suffix == ".vtm":
        folder = path.parent
        if (folder / "case.json").exists():
            meta = json.loads((folder / "case.json").read_text())
        mb = pv.read(path)
        return mb["cells"], mb["boundary"], meta.get("patches", []), meta, folder
    if path.suffix == ".vtu":
        return pv.read(path), None, [], meta, path.parent
    if path.suffix == ".npz":
        # mesher output: compute quality fields on the fly
        from fvmesh import FVMesh
        from hybrid_mesher import load_npz
        from postproc import mesh_quality
        m = load_npz(path)
        fvm = FVMesh(m["points"], m["cells"], m["patches"])
        grid = _grid_from_arrays(m["points"], m["cells"])
        for k, v in mesh_quality(fvm, m["cells"], m["n_bl"]).items():
            grid.cell_data[k] = v
        names = list(m["patches"])
        lines = np.vstack([m["patches"][n] for n in names])
        pid = np.concatenate([[i] * len(m["patches"][n]) for i, n in enumerate(names)])
        cells_l = np.c_[np.full(len(lines), 2), lines].ravel()
        bnd = pv.PolyData(np.c_[m["points"], np.zeros(len(m["points"]))], lines=cells_l)
        bnd.cell_data["patch_id"] = pid
        return grid, bnd, names, {"case": path.stem}, path.parent
    raise ValueError(f"cannot open {path}")


def field_list(grid):
    names = [k for k in grid.cell_data.keys() if k not in ("U",)]
    first = [f for f in FLOW_FIELDS_FIRST + MESH_FIELDS_FIRST if f in names]
    return first + [n for n in names if n not in first]


# ----------------------------------------------------------------------
# views (chord-relative windows around the wall)
# ----------------------------------------------------------------------
def view_windows(grid, bnd, patches):
    xmin, xmax, ymin, ymax = grid.bounds[:4]
    aspect = 1500 / 950                      # window width / height
    full = ((xmin + xmax) / 2, (ymin + ymax) / 2,
            0.53 * max(ymax - ymin, (xmax - xmin) / aspect))
    wall = None
    if bnd is not None and patches:
        wid = [i for i, n in enumerate(patches) if n in ("wall", "airfoil", "cylinder")]
        if wid:
            wall = bnd.extract_cells(np.where(bnd.cell_data["patch_id"] == wid[0])[0])
    if wall is None or wall.n_points == 0:
        return {"1": full, "2": full, "3": full, "4": full, "5": full}
    wx0, wx1, wy0, wy1 = wall.bounds[:4]
    c = wx1 - wx0
    return {"1": full,
            "2": (wx0 + 0.8 * c, (wy0 + wy1) / 2, 0.8 * c),
            "5": (wx0 + 2.3 * c, (wy0 + wy1) / 2, 1.6 * c),          # body + wake (vortex street)
            "3": (wx0 + 0.05 * c, (wy0 + wy1) / 2, 0.12 * c),
            "4": (wx1 - 0.02 * c, (wy0 + wy1) / 2, 0.12 * c)}


def _boxed(text_actor):
    """Dark semi-transparent box behind overlay text (readable on any colormap)."""
    try:
        text_actor.prop.background_color = BACKGROUND
        text_actor.prop.background_opacity = 0.75
    except Exception:
        pass
    return text_actor


def set_view(pl, win):
    cx, cy, half_h = win
    pl.camera.focal_point = (cx, cy, 0.0)
    pl.camera.position = (cx, cy, 10.0)
    pl.camera.up = (0, 1, 0)
    pl.camera.parallel_scale = half_h


# ----------------------------------------------------------------------
# streamlines
# ----------------------------------------------------------------------
def streamlines(grid, wins):
    if "U" not in grid.cell_data:
        return None
    pg = grid.cell_data_to_point_data()
    cx, cy, h = wins["2"]
    c = h / 0.8                                   # body length (see view_windows)
    x0, x_te = cx - 1.9 * h, cx + 0.2 * c
    seeds = pv.Line((x0, cy - 0.9 * h, 0), (x0, cy + 0.9 * h, 0), resolution=40)
    wake = pv.Line((x_te + 0.4 * c, cy - 0.45 * c, 0), (x_te + 0.4 * c, cy + 0.45 * c, 0),
                   resolution=24)
    src = seeds.merge(wake)
    try:
        return pg.streamlines_from_source(src, vectors="U", surface_streamlines=True,
                                          max_length=8 * h, integration_direction="both")
    except Exception as e:          # never let a visual extra kill the viewer
        print("streamlines failed:", e)
        return None


# ----------------------------------------------------------------------
# the interactive viewer
# ----------------------------------------------------------------------
class Viewer:
    def __init__(self, path, field=None, edges=None, off_screen=False, show_streamlines=False):
        self.grid, self.bnd, self.patches, self.meta, self.folder = load(path)
        self.fields = field_list(self.grid)
        self.has_flow = "U" in self.grid.cell_data
        self.i = self.fields.index(field) if field in self.fields else 0
        self.edges = (not self.has_flow) if edges is None else edges
        self.wins = view_windows(self.grid, self.bnd, self.patches)
        pv.global_theme.font.color = "white"
        self.pl = pv.Plotter(off_screen=off_screen, window_size=(1500, 950))
        self.pl.set_background(BACKGROUND)
        self._points = None                       # point-interpolated copy (smooth plots)
        self.actor = self.stream_actor = self.worst_actor = None
        self.show_stream = show_streamlines
        self.worst_rank = 0
        self._stream = None
        self._draw_boundary()
        self._draw_field()                        # also draws streamlines if requested
        self.pl.enable_parallel_projection()
        set_view(self.pl, self.wins["2"])
        self._keys()

    # -- drawing ---------------------------------------------------------
    def _field_style(self, name, data):
        cmap, clim, _ = FIELD_STYLE.get(name, ("viridis", None, None))
        if clim == "sym":
            a = np.nanpercentile(np.abs(data), 98 if name == "vorticity" else 99.9) or 1.0
            clim = (-a, a)
        elif clim is None:
            lo, hi = np.nanpercentile(data, [0.5, 99.5])
            clim = (lo, hi) if hi > lo else None
        return cmap, clim

    def _draw_field(self):
        name = self.fields[self.i]
        if self.actor is not None:
            self.pl.remove_actor(self.actor)
        data = self.grid.cell_data[name]
        cmap, clim = self._field_style(name, data)
        # quality fields are per cell (flat); flow fields are shown smooth
        if name in QUALITY_FIELDS:
            mesh, pref = self.grid, "cell"
        else:
            if self._points is None:
                self._points = self.grid.cell_data_to_point_data()
            mesh, pref = self._points, "point"
        self.actor = self.pl.add_mesh(mesh, scalars=name, cmap=cmap, clim=clim,
                                      show_edges=self.edges, edge_color="#202020",
                                      line_width=0.3, preference=pref,
                                      scalar_bar_args={"title": name, "vertical": False,
                                                       "position_x": 0.3, "position_y": 0.03,
                                                       "width": 0.4, "height": 0.06,
                                                       "title_font_size": 16,
                                                       "label_font_size": 13, "shadow": True})
        self._text()
        if self.show_stream:                       # streamline colour follows the colormap
            self._toggle_stream(force=True)

    def _draw_boundary(self):
        if self.bnd is None:
            return
        for i, n in enumerate(self.patches):
            sub = self.bnd.extract_cells(np.where(self.bnd.cell_data["patch_id"] == i)[0])
            if sub.n_cells:
                self.pl.add_mesh(sub, color=PATCH_COLORS.get(n, "orange"), line_width=3,
                                 label=n)
        if self.patches:
            self.pl.add_legend(bcolor=BACKGROUND, size=(0.1, 0.1), loc="lower left")

    def _text(self):
        name = self.fields[self.i]
        d = self.grid.cell_data[name]
        lines = [f"{self.meta.get('case', '')}   cells: {self.grid.n_cells}",
                 f"field: {name}   min {np.nanmin(d):.4g}   max {np.nanmax(d):.4g}"]
        c = self.meta.get("coefficients")
        if c:
            lines.append(f"Cd = {c['Cd']:.4f}   Cl = {c['Cl']:.4f}   "
                         f"(Re = {self.meta.get('Re')}, alpha = {self.meta.get('alpha_deg')} deg)")
        lines.append("keys: n/b field  g edges  1-5 views  j worst cell  m streamlines  "
                     "o plots  x screenshot")
        t = self.pl.add_text("\n".join(lines), position="upper_left", font_size=11, name="info")
        _boxed(t)

    def _toggle_stream(self, force=False):
        self.show_stream = True if force else not self.show_stream
        if self.stream_actor is not None:
            self.pl.remove_actor(self.stream_actor)
            self.stream_actor = None
        if self.show_stream:
            if self._stream is None:
                self._stream = streamlines(self.grid, self.wins)
            if self._stream is not None and self._stream.n_points:
                cmap = FIELD_STYLE.get(self.fields[self.i], ("viridis",))[0]
                color = "#111111" if cmap.startswith("RdBu") else "white"
                self.stream_actor = self.pl.add_mesh(self._stream, color=color, line_width=1.0,
                                                     opacity=0.9)

    def _worst(self):
        name = self.fields[self.i]
        sense = FIELD_STYLE.get(name, (None, None, None))[2]
        if sense is None:
            print(f"'{name}' has no 'bad' direction; pick a quality field (n/b)")
            return
        d = np.asarray(self.grid.cell_data[name], float)
        key = np.abs(d) if sense == "abs" else (d if sense == "low" else -d)
        order = np.argsort(key)
        cid = int(order[self.worst_rank % len(order)])
        self.worst_rank += 1
        cell = self.grid.extract_cells([cid])
        if self.worst_actor is not None:
            self.pl.remove_actor(self.worst_actor)
        self.worst_actor = self.pl.add_mesh(cell, style="wireframe", color="magenta", line_width=5)
        b = cell.bounds
        size = max(b[1] - b[0], b[3] - b[2])
        set_view(self.pl, ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, 6 * size))
        msg = (f"worst #{self.worst_rank} by {name}: cell {cid} = {d[cid]:.4g} at "
               f"({(b[0] + b[1]) / 2:.4f}, {(b[2] + b[3]) / 2:.4f})")
        print(msg)
        _boxed(self.pl.add_text(msg, position="lower_right", font_size=10, name="worst"))
        self.pl.render()

    # -- keys --------------------------------------------------------------
    def _keys(self):
        def step(k):
            def f():
                self.i = (self.i + k) % len(self.fields)
                self.worst_rank = 0
                self._draw_field()
                self.pl.render()
            return f

        def edges():
            self.edges = not self.edges
            self._draw_field(); self.pl.render()

        self.pl.add_key_event("n", step(+1))
        self.pl.add_key_event("b", step(-1))
        self.pl.add_key_event("g", edges)
        for k in "12345":
            self.pl.add_key_event(k, lambda k=k: (set_view(self.pl, self.wins[k]), self.pl.render()))
        self.pl.add_key_event("j", self._worst)
        self.pl.add_key_event("m", lambda: (self._toggle_stream(), self.pl.render()))
        self.pl.add_key_event("o", lambda: plots(self.folder, show=True))
        self.pl.add_key_event("x", self.save)

    def save(self, path=None):
        path = path or (self.folder / f"view_{self.fields[self.i]}.png")
        self.pl.screenshot(str(path))
        print("saved", path)
        return path

    def show(self):
        self.pl.enable_2d_style()
        self.pl.show()


# ----------------------------------------------------------------------
# 2D plots: wall distributions + convergence history
# ----------------------------------------------------------------------
def _read_csv(p):
    return np.genfromtxt(p, delimiter=",", names=True)


def _wall_panels(ax_cp, ax_cf, walls):
    w = _read_csv(walls[0])
    for side, mk, lab in ((1, "-", "upper"), (-1, "--", "lower")):
        s = w["side"] == side
        o = np.argsort(w["x"][s])
        ax_cp.plot(w["x"][s][o], w["Cp"][s][o], mk, label=f"Cp {lab}")
        ax_cf.plot(w["x"][s][o], w["Cf"][s][o], mk, label=f"Cf {lab}")
    ax_cp.invert_yaxis()
    ax_cp.set_title("Wall pressure coefficient (Cp axis inverted)")
    ax_cf.axhline(0, color="k", lw=0.8)
    ax_cf.set_title(f"Skin friction (Cf < 0: reversed flow)   y+ max {w['yplus'].max():.2f}")
    for a in (ax_cp, ax_cf):
        a.set_xlabel("x"); a.legend(); a.grid(alpha=.3)


def plots(folder, show=True, save=None):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    hist = folder / "history.csv"
    walls = sorted(folder.glob("wall_*.csv"))
    if not hist.exists() and not walls:
        print("no results in", folder, "(run the case first)")
        return
    h = _read_csv(hist) if hist.exists() else None
    transient = h is not None and "time" in h.dtype.names

    if transient:
        fig, ax = plt.subplots(2, 3, figsize=(18, 8.5))
        t = h["time"]
        ax[0, 0].plot(t, h["Cd"], lw=0.9); ax[0, 0].set_title("Drag coefficient Cd(t)")
        ax[0, 1].plot(t, h["Cl"], lw=0.9, color="C1"); ax[0, 1].set_title("Lift coefficient Cl(t)")
        tail = t > t[0] + 0.5 * (t[-1] - t[0])            # second half: developed flow
        if tail.sum() > 16:
            cd = h["Cd"][tail]
            lo, hi = cd.min(), cd.max()
            pad = max(0.05 * (hi - lo), 1e-3)
            ax[0, 0].set_ylim(lo - 3 * pad - (hi - lo), hi + 3 * pad + (hi - lo))
            y = h["Cl"][tail] - h["Cl"][tail].mean()
            dt = np.median(np.diff(t))
            n = len(y)
            spec = np.abs(np.fft.rfft(y * np.hanning(n))) ** 2
            f = np.fft.rfftfreq(n, dt)
            ax[0, 2].semilogy(f[1:], spec[1:] / spec[1:].max())
            fpk = f[1:][np.argmax(spec[1:])]
            ax[0, 2].axvline(fpk, color="r", ls=":")
            ax[0, 2].set_xlim(0, min(f[-1], 8 * fpk if fpk > 0 else f[-1]))
            ax[0, 2].set_title(f"Lift spectrum (2nd half): peak f = {fpk:.3f} +- {0.5 / (n * dt):.3f}\n"
                               f"(FFT resolution; case.json has the more precise St "
                               f"from zero crossings)", fontsize=10)
            ax[0, 2].set_xlabel("frequency (U / L)")
        for a in ax[0, :2]:
            a.set_xlabel("time (L / U)")
        for k in ("cont", "Ux", "Uy"):
            ax[1, 0].semilogy(t, h[k], lw=0.7, label=k)
        ax[1, 0].set_title("Residual at the end of each time step")
        ax[1, 0].set_xlabel("time (L / U)"); ax[1, 0].legend()
        for a in list(ax[0]) + [ax[1, 0]]:
            a.grid(alpha=.3, which="both")
        if walls:
            _wall_panels(ax[1, 1], ax[1, 2], walls)
            ax[1, 1].set_title("Wall Cp at the final time")
    else:
        fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
        if h is not None:
            for k in ("cont", "Ux", "Uy"):
                ax[0, 0].semilogy(h["iteration"], h[k], label=k)
            ax[0, 0].set_title("Residuals"); ax[0, 0].set_xlabel("iteration"); ax[0, 0].legend()
            ax[0, 0].grid(alpha=.3, which="both")
            ok = np.isfinite(h["Cd"])
            ax[0, 1].plot(h["iteration"][ok], h["Cd"][ok], label="Cd")
            ax[0, 1].plot(h["iteration"][ok], h["Cl"][ok], label="Cl")
            last = h["Cd"][ok][-1] if ok.any() else np.nan
            ax[0, 1].set_ylim(*np.nanpercentile(np.r_[h["Cd"][ok][len(h[ok]) // 5:],
                                                        h["Cl"][ok][len(h[ok]) // 5:]], [0, 100]) +
                              np.array([-0.1, 0.1]) * max(abs(last), 0.1))
            ax[0, 1].set_title("Force coefficients"); ax[0, 1].set_xlabel("iteration")
            ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)
        if walls:
            _wall_panels(ax[1, 0], ax[1, 1], walls)
    fig.suptitle(str(folder))
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
        print("saved", save)
    if show:
        plt.show()
    return fig


def monitor(folder, interval=2.0):
    """Live residual / force plot while a case is running (re-reads history.csv).
    Works for steady (x = iteration) and transient (x = time) runs."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    hist = Path(folder) / "history.csv"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))

    def update(_):
        if not hist.exists():
            return
        try:
            h = _read_csv(hist)
        except Exception:
            return                   # file being written
        if h.size < 2:
            return
        x = h["time"] if "time" in h.dtype.names else h["iteration"]
        xl = "time" if "time" in h.dtype.names else "iteration"
        a1.cla(); a2.cla()
        for k in ("cont", "Ux", "Uy"):
            a1.semilogy(x, h[k], label=k)
        a1.set_title(f"Residuals ({xl} {x[-1]:.6g})"); a1.legend(); a1.set_xlabel(xl)
        a1.grid(alpha=.3, which="both")
        ok = np.isfinite(h["Cd"])
        a2.plot(x[ok], h["Cd"][ok], label="Cd")
        a2.plot(x[ok], h["Cl"][ok], label="Cl")
        if ok.any():
            a2.set_title(f"Cd = {h['Cd'][ok][-1]:.4f}   Cl = {h['Cl'][ok][-1]:.4f}")
        a2.legend(); a2.grid(alpha=.3); a2.set_xlabel(xl)

    anim = FuncAnimation(fig, update, interval=interval * 1000, cache_frame_data=False)
    update(0)
    plt.show()
    return anim


def animate(folder, field="vorticity", out=None, view="5", fps=10, max_frames=80, width=800):
    """GIF of a transient case from its snapshots (results.pvd)."""
    import re
    from PIL import Image
    folder = Path(folder)
    pvd = (folder / "results.pvd").read_text()
    entries = [(float(t), f) for t, f in re.findall(r'timestep="([^"]+)"[^>]*file="([^"]+)"', pvd)]
    if not entries:
        print("no snapshots in", folder)
        return None
    step = max(1, -(-len(entries) // max_frames))          # ceil: at most max_frames
    entries = entries[::step]
    base = Viewer(folder, field=field if field in ("vorticity", "U_mag", "Cp", "p") else None,
                  off_screen=True)
    last = pv.read(folder / entries[-1][1])
    cmap, clim = base._field_style(field, last.cell_data[field])
    if field == "vorticity":
        # scale to the wake, not to the (much stronger) wall vorticity
        a = np.nanpercentile(np.abs(last.cell_data[field]), 90) or 1.0
        clim = (-a, a)
    grid = base.grid
    base.pl.remove_actor(base.actor)            # start clean: no field / colour bar yet
    base.actor = None
    for name in list(base.pl.scalar_bars.keys()):
        base.pl.remove_scalar_bar(name)
    frames = []
    set_view(base.pl, base.wins[view])
    for t, f in entries:
        snap = pv.read(folder / f)
        grid.cell_data[field] = snap.cell_data[field]
        pts = grid.cell_data_to_point_data()
        if base.actor is not None:
            base.pl.remove_actor(base.actor)
        base.actor = base.pl.add_mesh(pts, scalars=field, cmap=cmap, clim=clim,
                                      scalar_bar_args={"title": field, "vertical": False,
                                                       "position_x": 0.3, "position_y": 0.03,
                                                       "width": 0.4, "height": 0.06})
        _boxed(base.pl.add_text(f"{base.meta.get('case', '')}   {field}   t = {t:.2f}",
                                position="upper_left", font_size=12, name="info"))
        im = Image.fromarray(base.pl.screenshot(return_img=True))
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
    out = Path(out or folder / f"{field}.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    base.pl.close()
    print("saved", out, f"({len(frames)} frames)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="case folder, .vtm, .vtu or mesher .npz")
    ap.add_argument("--field", default=None)
    ap.add_argument("--edges", action="store_true", help="start with cell edges on")
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--streamlines", action="store_true")
    ap.add_argument("--view", choices=["full", "near", "le", "te", "wake"], default=None)
    ap.add_argument("--screenshot", default=None, help="render to PNG without a window")
    ap.add_argument("--plots", action="store_true", help="wall + history plots")
    ap.add_argument("--monitor", action="store_true", help="live history while running")
    ap.add_argument("--animate", default=None, metavar="FIELD",
                    help="transient case: make a GIF of FIELD (e.g. vorticity)")
    ap.add_argument("--gif", default=None, help="output file for --animate")
    a = ap.parse_args()

    if a.monitor:
        monitor(a.path); return
    if a.animate:
        animate(a.path, field=a.animate, out=a.gif,
                view=VIEW_KEYS[a.view or "wake"]); return
    if a.plots:
        plots(a.path, show=a.screenshot is None, save=a.screenshot); return
    edges = True if a.edges else (False if a.no_edges else None)
    v = Viewer(a.path, field=a.field, edges=edges, off_screen=a.screenshot is not None,
               show_streamlines=a.streamlines)
    set_view(v.pl, v.wins[VIEW_KEYS[a.view or "near"]])
    if a.screenshot:
        v.save(a.screenshot)
    else:
        v.show()


if __name__ == "__main__":
    main()
