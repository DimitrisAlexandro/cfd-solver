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
    if st.session_state.pop("aircraft_force", False):      # a new file was just saved: select it
        st.session_state[key] = files[idx]

    def name(p):
        try:
            return f"{A.Aircraft.load(p).name}  ({p.name})"
        except Exception:
            return p.name
    f = st.selectbox(label, files, index=idx, format_func=name, key=key)
    st.session_state["aircraft_file"] = str(f)
    return A.Aircraft.load(f), f


def select(path):
    """make a saved aircraft file the selected one on every aircraft page"""
    st.session_state["aircraft_file"] = str(path)
    st.session_state["aircraft_force"] = True


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


def view3d(ac, markers=None, height=460, mesh=None):
    """surfaces (controls coloured), fuselage, c.g. and other points; optionally
    an imported surface mesh (V, T) drawn over the model"""
    f = go.Figure()
    if mesh is not None:
        V, T = mesh                             # wireframe: stays visible on the model's surfaces
        E = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
        if len(E) > 25000:                      # drawing only: a subset of the edges
            E = E[np.random.default_rng(0).choice(len(E), 25000, replace=False)]
        P = np.full((3 * len(E), 3), np.nan)
        P[0::3], P[1::3] = V[E[:, 0]], V[E[:, 1]]
        f.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines", hoverinfo="skip",
                                 line=dict(color="#C2571A", width=1), opacity=0.55, name="STL",
                                 showlegend=False))
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


def _unique_path(slug):
    p = C.AIRCRAFT / f"{slug}.json"
    k = 1
    while p.exists():
        k += 1
        p = C.AIRCRAFT / f"{slug}_{k}.json"
    return p


def upload_panel():
    """import an aircraft from the app's JSON or an AVL geometry (+ mass, airfoil) files"""
    import re
    import tempfile
    import aero_database as D
    import aircraft_import as AI
    left, right = st.columns([5, 6], gap="large")
    with left:
        with st.container(border=True):
            ui.step("Files", first=True)
            ups = st.file_uploader(
                "Aircraft files", type=["json", "avl", "mass", "dat", "txt", "stl"], accept_multiple_files=True,
                key=f"acupload{st.session_state.get('acup_n', 0)}",
                help="An aircraft .json saved from this app; an AVL geometry file (.avl) with, "
                     "optionally, its mass file (.mass) and the airfoil coordinate files its AFILE "
                     "lines name (.dat); or a surface mesh of the aircraft (.stl, binary or ASCII).")
            st.caption(
                "**STL**: wing, tails, fins and fuselage are recognised from cuts through the mesh and "
                "fitted with tapered panels; sections from the mesh. "
                "**AVL**: SURFACE / SECTION geometry with YDUPLICATE, TRANSLATE, SCALE, ANGLE; NACA, "
                "AFILE and inline AIRFOIL sections; CONTROL lines named elevator, aileron, rudder or "
                "flap. Each surface becomes one tapered panel through its root and tip sections. "
                "**Mass file**: total mass, c.g. and inertias about the c.g. from all mass lines.")
            with tempfile.TemporaryDirectory() as td:
                demo = A.demo_aircraft()
                D.write_avl(demo, td)
                avl_t = (Path(td) / "aircraft.avl").read_text()
                mass_t = (Path(td) / "aircraft.mass").read_text()
            import stl_aircraft as S
            c1, c2, c3, c4 = st.columns(4)
            c4.download_button("Example .stl", S.stl_bytes(S.aircraft_mesh(demo), demo.name),
                               file_name="example_aircraft.stl", width="stretch")
            c1.download_button("Example .json", json.dumps(demo.to_dict(), indent=1),
                               file_name="example_aircraft.json", width="stretch")
            c2.download_button("Example .avl", avl_t, file_name="example_aircraft.avl", width="stretch")
            c3.download_button("Example .mass", mass_t, file_name="example_aircraft.mass", width="stretch")
    if st.session_state.get("acup_saved"):
        with left:
            st.success(f"Saved {st.session_state.pop('acup_saved')}. It is now the selected aircraft on "
                       "every aircraft page; check and complete it on the Edit tab.")
    if not ups:
        with right:
            st.info("Upload an aircraft to preview it here before saving.")
        return
    files = {u.name: u.getvalue() for u in ups}
    mesh = None
    stl = [n for n in files if n.lower().endswith(".stl")]
    try:
        if stl:
            with left:
                ac, warns, mesh = _stl_options(stl[0], files[stl[0]])
            if ac is None:
                return
        else:
            ac, warns = AI.read_files(files, C.CASES / "_uploads")
    except Exception as e:                          # noqa: BLE001
        with left:
            st.error(f"These files can't be read as an aircraft: {e}")
        return
    with right:
        view3d(ac, height=360, mesh=mesh)
        if mesh is not None:
            st.caption("Orange lines: the uploaded mesh. Blue surfaces: the fitted model.")
        vh, vv = tail_volumes(ac)
        k = st.columns(4)
        k[0].metric("Wing area", f"{ac.S:.3f} m²")
        k[1].metric("Span", f"{ac.b:.3f} m")
        k[2].metric("Mass", f"{ac.mass:g} kg")
        xm, _ = ac.wing.mac_position()
        k[3].metric("c.g.", f"{100 * (ac.cg[0] - xm) / ac.c:.1f} % MAC")
        sf = surfaces_frame(ac)
        sf["airfoil"] = sf["airfoil"].map(lambda a: C.display_name(Path(str(a)).stem))
        st.dataframe(sf[["name", "span", "root_chord", "tip_chord", "x", "z", "sweep",
                                         "dihedral", "incidence", "airfoil", "vertical"]],
                     hide_index=True, width="stretch")
        cf = controls_frame(ac)
        if len(cf):
            st.dataframe(cf, hide_index=True, width="stretch")
    with left:
        for w in warns:
            st.warning(w)
        with st.container(border=True):
            ui.step("Save")
            c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
            stem = Path(next(iter(n for n in files if n.lower().endswith((".avl", ".json"))), "aircraft")).stem
            fname = c1.text_input("File name", stem, key="acupname")
            if c2.button("Save aircraft", type="primary", width="stretch", key="acupsave"):
                slug = re.sub(r"[^A-Za-z0-9_-]+", "_", fname.strip()) or "aircraft"
                out = _unique_path(slug)
                ac.save(out)
                select(out)
                st.session_state["acup_n"] = st.session_state.get("acup_n", 0) + 1
                st.session_state["acup_saved"] = out.name
                st.rerun()


@st.cache_data(show_spinner=False, max_entries=4)
def _stl_read(data):
    import stl_aircraft as S
    t = S.read_stl(data)
    u = S.guess_units(t)
    return t, u, S.guess_orientation(t, u)


@st.cache_data(show_spinner=False, max_entries=8)
def _stl_import(data, la, nose, up, units, sections, mass, cg, controls, name, af_dir):
    import stl_aircraft as S
    t = S.read_stl(data)
    ac, warns, info = S.to_aircraft(t, la, nose, up, units, sections, af_dir, mass, cg, controls, name)
    return json.dumps(ac.to_dict()), warns, info["V"], info["T"], info["mesh"]


def _stl_options(fname, data):
    """STL import settings (axes, units, sections, mass) -> (aircraft, warnings, (V, T))"""
    import re
    import stl_aircraft as S
    with st.spinner("Reading the mesh"):
        tris, u0, (la0, nose0, up0) = _stl_read(data)
    with st.container(border=True):
        ui.step("Mesh axes and units")
        ext = np.ptp(tris.reshape(-1, 3), axis=0)
        st.caption(f"{len(tris):,} triangles; extent x {ext[0]:.4g}, y {ext[1]:.4g}, z {ext[2]:.4g} "
                   f"(file units). Detected: length along {la0}, nose at its {nose0}imum, up {up0}, {u0}.")
        c1, c2, c3, c4 = st.columns(4)
        units = c1.selectbox("Units", list(S.UNITS), index=list(S.UNITS).index(u0), key="stlu")
        la = c2.selectbox("Length axis", ["x", "y", "z"], index="xyz".index(la0), key="stlla")
        nose = c3.selectbox("Nose at", ["min", "max"], index=["min", "max"].index(nose0),
                            format_func=lambda v: f"{v}. {la}", key="stlnose")
        ups = ["+x", "-x", "+y", "-y", "+z", "-z"]
        up = c4.selectbox("Up", ups, index=ups.index(up0), key="stlup")
        ui.step("Model")
        c1, c2 = st.columns(2)
        sections = c1.radio("Sections", ["naca", "cut"], key="stlsec", horizontal=True,
                            format_func=lambda v: {"naca": "nearest NACA 4-digit", "cut": "cut from the mesh"}[v],
                            help="Cut sections keep the mesh's real shape (a fine mesh is needed); "
                                 "NACA fits are smooth and robust on coarse meshes.")
        controls = c2.checkbox("Add typical control surfaces", True, key="stlctl")
    if la == up[-1]:
        st.error("The length axis and the up axis must differ.")
        return None, [], None
    name = Path(fname).stem
    af_dir = str(C.CASES / "_uploads" / f"stl_{re.sub(r'[^A-Za-z0-9_-]+', '_', name)}")
    with st.spinner("Cutting the mesh and fitting the surfaces"):
        js, warns, V, T, minfo = _stl_import(data, la, nose, up, units, sections, 750.0, 0.25,
                                             bool(controls), name, af_dir)
    ac = A.Aircraft.from_dict(json.loads(js))
    st.caption(f"Model size: length {minfo['length']:.3f} m, span {minfo['span']:.3f} m, "
               f"height {minfo['height']:.3f} m. Check the units if these look wrong.")
    with st.container(border=True):
        ui.step("Mass and balance (not in an STL file)")
        ws = 60.0 if ac.b < 5 else 600.0          # typical wing loading, N/m²: models / light aircraft
        m0 = float(f"{ws * ac.S / A.G0:.2g}")
        c1, c2 = st.columns(2)
        mass = c1.number_input("Mass (kg)", 0.001, 1e6, m0, key=f"stlmass{name}{units}", format="%.4g",
                               help=f"Default from a typical wing loading of {ws:g} N/m²; enter the real mass.")
        cg = c2.number_input("c.g. (% MAC)", -50.0, 100.0, 25.0, 1.0, key="stlcg")
    ac.mass = float(mass)
    xm, _ = ac.wing.mac_position()
    ac.cg = (round(xm + cg / 100 * ac.wing.mac, 4), 0.0, 0.0)
    return ac, warns, (V, T)
