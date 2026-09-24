"""
Interactive mesh viewer (zoom / pan with the mouse).

    python view_mesh.py ../airfoil_mesh.npz
    python view_mesh.py ../airfoil_mesh.npz --color angle     # triangle quality

Mouse:  toolbar magnifier = zoom box, cross arrows = pan, scroll = zoom
Keys:   f = full domain   n = near field   l = leading edge
        t = trailing edge   w = trailing-edge wall close-up
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from hybrid_mesher import load_npz

VIEWS = {
    "f": ((-21, 21), (-21, 21)),
    "n": ((-0.6, 2.2), (-0.8, 0.8)),
    "l": ((-0.12, 0.16), (-0.12, 0.13)),
    "t": ((0.88, 1.16), (-0.12, 0.13)),
    "w": ((0.996, 1.006), (-0.005, 0.004)),
}


def min_angles(P, cells):
    out = np.full(len(cells), np.nan)
    for i, c in enumerate(cells):
        if len(c) != 3:
            continue
        T = P[c]
        a = []
        for k in range(3):
            e1, e2 = T[(k + 1) % 3] - T[k], T[(k + 2) % 3] - T[k]
            a.append(np.degrees(np.arccos(np.clip(
                e1 @ e2 / np.linalg.norm(e1) / np.linalg.norm(e2), -1, 1))))
        out[i] = min(a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", nargs="?", default="../airfoil_mesh.npz")
    ap.add_argument("--color", choices=["type", "angle"], default="type",
                    help="type: BL cells red / triangles blue; angle: triangle min angle")
    a = ap.parse_args()

    m = load_npz(a.mesh)
    P, cells, nb = m["points"], m["cells"], m["n_bl"]
    fig, ax = plt.subplots(figsize=(12, 8))

    bl = PolyCollection([P[c] for c in cells[:nb]], facecolor="#dce9f5",
                        edgecolor="#c0504d", lw=0.3)
    ax.add_collection(bl)
    if a.color == "angle":
        ang = min_angles(P, cells[nb:])
        tri = PolyCollection([P[c] for c in cells[nb:]], array=ang, cmap="RdYlGn",
                             edgecolor="#555555", lw=0.2)
        tri.set_clim(15, 60)
        ax.add_collection(tri)
        fig.colorbar(tri, ax=ax, label="triangle minimum angle (deg)")
    else:
        ax.add_collection(PolyCollection([P[c] for c in cells[nb:]], facecolor="white",
                                         edgecolor="#4a6fa5", lw=0.3))

    wall = m["patches"]["airfoil"]
    ax.add_collection(LineCollection(P[wall], colors="k", lw=1.2))
    ax.set_aspect("equal")
    ax.set_title(f"{a.mesh}: {nb} boundary-layer cells + {len(cells) - nb} triangles   "
                 f"[keys: f n l t w]")

    def set_view(key):
        xl, yl = VIEWS[key]
        ax.set_xlim(*xl); ax.set_ylim(*yl)
        fig.canvas.draw_idle()

    def on_key(ev):
        if ev.key in VIEWS:
            set_view(ev.key)

    def on_scroll(ev):
        if ev.xdata is None:
            return
        s = 0.8 if ev.button == "up" else 1.25
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.set_xlim(ev.xdata + (x0 - ev.xdata) * s, ev.xdata + (x1 - ev.xdata) * s)
        ax.set_ylim(ev.ydata + (y0 - ev.ydata) * s, ev.ydata + (y1 - ev.ydata) * s)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    set_view("n")
    plt.show()


if __name__ == "__main__":
    main()