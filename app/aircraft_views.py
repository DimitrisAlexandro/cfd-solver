"""Aircraft pages: shared picker, 3D view, tables."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
import ui

import aircraft as A

CTRL_COLOR = {"elevator": "#C2571A", "aileron": "#2B7A4B", "rudder": "#7A4FA0", "flap": "#8A7B12"}


def ensure_demo():
    C.AIRCRAFT.mkdir(parents=True, exist_ok=True)
    if not any(C.AIRCRAFT.glob("*.json")):
        A.demo_aircraft().save(C.AIRCRAFT / "demo_trainer.json")


def aircraft_picker(key="acpick", label="Aircraft"):
    """select a saved aircraft (cases/aircraft/*.json); shared across pages"""
    ensure_demo()
    files = sorted(C.AIRCRAFT.glob("*.json"))
    cur = st.session_state.get("aircraft_file")
    idx = next((i for i, f in enumerate(files) if str(f) == cur), 0)

    def name(p):
        try:
            return f"{A.Aircraft.load(p).name}  ({p.name})"
        except Exception:
            return p.name
    f = st.selectbox(label, files, index=idx, format_func=name, key=key)
    st.session_state["aircraft_file"] = str(f)
    return A.Aircraft.load(f), f


def tail_volumes(ac):
    """horizontal and vertical tail volume coefficients"""
    w = ac.wing
    xw = w.mac_position()[0] + 0.25 * w.mac
    vh = vv = 0.0
    for s in ac.surfaces[1:]:
        xs = s.mac_position()[0] + 0.25 * s.mac
        if s.vertical:
            vv += s.area * (xs - xw) / (ac.S * ac.b)
        else:
            vh += s.area * (xs - xw) / (ac.S * ac.c)
    return vh, vv


def view3d(ac, markers=None, height=460):
    """surfaces (controls coloured), fuselage, c.g. and other points"""
    f = go.Figure()
    for s in ac.surfaces:
        for kind, q in A.surface_quads(s):
            col = "#9DB4CF" if kind == "surface" else CTRL_COLOR.get(kind, "#C2571A")
            f.add_trace(go.Mesh3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], i=[0, 0], j=[1, 2], k=[2, 3],
                                  color=col, opacity=0.85 if kind == "surface" else 0.95, flatshading=True,
                                  hoverinfo="text", text=f"{s.name}" + ("" if kind == "surface" else f": {kind}"),
                                  showlegend=False))
            loop = np.r_[q, q[:1]]
            f.add_trace(go.Scatter3d(x=loop[:, 0], y=loop[:, 1], z=loop[:, 2], mode="lines",
                                     line=dict(color=ui.INK, width=2), hoverinfo="skip", showlegend=False))
    fu = ac.fuselage
    if fu.length > 0 and fu.diameter > 0:
        xs = fu.x + np.linspace(0, fu.length, 40)
        r = fu.radius(xs)
        th = np.linspace(0, 2 * np.pi, 28)
        X = np.repeat(xs[:, None], len(th), 1)
        Y = r[:, None] * np.cos(th)[None, :]
        Z = fu.z + r[:, None] * np.sin(th)[None, :]
        f.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0, "#D9DEE4"], [1, "#D9DEE4"]], showscale=False,
                               opacity=0.55, hoverinfo="skip", lighting=dict(ambient=0.8)))
    for lab, p, col, sym in (markers or [("c.g.", ac.cg, ui.INK, "diamond")]):
        f.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text", text=[lab],
                                 textposition="top center", marker=dict(size=5, color=col, symbol=sym),
                                 showlegend=False))
    f.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                    scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                               camera=dict(eye=dict(x=-1.4, y=-1.5, z=0.9), up=dict(x=0, y=0, z=1)),
                               xaxis=dict(backgroundcolor="#FFFFFF", gridcolor=ui.GRID),
                               yaxis=dict(backgroundcolor="#FFFFFF", gridcolor=ui.GRID),
                               zaxis=dict(backgroundcolor="#FFFFFF", gridcolor=ui.GRID)))
    st.plotly_chart(f, width="stretch", theme=None)


def top_view(ac, height=320, markers=None):
    """2D planform, top view (x down the page like a drawing: nose up)"""
    f = go.Figure()
    for s in ac.surfaces:
        for kind, q in A.surface_quads(s):
            if s.vertical:
                q = q.copy()
            loop = np.r_[q, q[:1]]
            col = ui.BLUE if kind == "surface" else CTRL_COLOR.get(kind, "#C2571A")
            f.add_scatter(x=loop[:, 1], y=-loop[:, 0], mode="lines", line=dict(color=col, width=1.3),
                          fill="toself", fillcolor="rgba(29,78,137,0.06)" if kind == "surface" else
                          "rgba(194,87,26,0.15)", hoverinfo="text", text=f"{s.name} {'' if kind == 'surface' else kind}",
                          showlegend=False)
    fu = ac.fuselage
    if fu.length > 0:
        xs = fu.x + np.linspace(0, fu.length, 60)
        r = fu.radius(xs)
        f.add_scatter(x=np.r_[r, -r[::-1]], y=-np.r_[xs, xs[::-1]], mode="lines",
                      line=dict(color=ui.GRAPHITE, width=1), hoverinfo="skip", showlegend=False)
    for lab, p, col, sym in (markers or [("c.g.", ac.cg, ui.INK, "diamond")]):
        f.add_scatter(x=[p[1]], y=[-p[0]], mode="markers+text", text=[lab], textposition="middle right",
                      marker=dict(color=col, size=8, symbol=sym), showlegend=False)
    f.update_layout(height=height, yaxis=dict(scaleanchor="x", title="−x (m)"), xaxis_title="y (m)",
                    title="Top view")
    ui.plot(f)


def surfaces_frame(ac):
    rows = []
    for s in ac.surfaces:
        rows.append({"name": s.name, "span": s.span, "root_chord": s.root_chord, "tip_chord": s.tip_chord,
                     "x": s.x, "y": s.y, "z": s.z, "sweep": s.sweep, "dihedral": s.dihedral,
                     "twist": s.twist, "incidence": s.incidence, "airfoil": s.airfoil,
                     "tip_airfoil": s.tip_airfoil or "", "symmetric": s.symmetric, "vertical": s.vertical,
                     "ns": s.ns, "nc": s.nc})
    return pd.DataFrame(rows)


def controls_frame(ac):
    rows = []
    for s in ac.surfaces:
        for c in s.controls:
            rows.append({"surface": s.name, "name": c.name, "eta0": c.eta0, "eta1": c.eta1, "cf": c.cf,
                         "mode": c.mode})
    return pd.DataFrame(rows, columns=["surface", "name", "eta0", "eta1", "cf", "mode"])


def build_from_frames(base, sdf, cdf, **kw):
    """Aircraft from the edited tables (+ scalar fields in kw)"""
    surfs = []
    for _, r in sdf.iterrows():
        if not str(r["name"]).strip():
            continue
        ctrls = [A.Control(str(c["name"]), float(c["eta0"]), float(c["eta1"]), float(c["cf"]), str(c["mode"]))
                 for _, c in cdf.iterrows() if str(c["surface"]) == str(r["name"]) and str(c["name"]).strip()]
        surfs.append(A.Surface(str(r["name"]), float(r["span"]), float(r["root_chord"]), float(r["tip_chord"]),
                               float(r["x"]), float(r["y"]), float(r["z"]), float(r["sweep"]),
                               float(r["dihedral"]), float(r["twist"]), float(r["incidence"]),
                               str(r["airfoil"]).strip() or "naca0012",
                               (str(r["tip_airfoil"]).strip() or None) if r["tip_airfoil"] is not None else None,
                               bool(r["symmetric"]), bool(r["vertical"]), ctrls, int(r["ns"]), int(r["nc"])))
    d = base.to_dict()
    d.update(kw)
    d["surfaces"] = [A.asdict(s) for s in surfs]
    return A.Aircraft.from_dict(d)


@st.cache_data(show_spinner=False)
def stability(ac_json, speed, altitude, polar_dir):
    ac = A.Aircraft.from_dict(json.loads(ac_json))
    return A._jsonable(A.stability_report(ac, speed, altitude, polar_dir))


def ac_json(ac):
    return json.dumps(ac.to_dict(), sort_keys=True)
