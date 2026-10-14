import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import common as C
import ui

import postview as V

C.page_setup()
ui.header("Results viewer", "Every result on disk, 2D and 3D: meshes, flow fields, surface pressure, "
                            "cuts through the 3D field, wall distributions and convergence.")

CMAPS = ["turbo", "viridis", "plasma", "coolwarm", "RdBu_r", "gray"]


@st.cache_data(ttl=30, show_spinner=False)
def _results():
    return V.find_results(C.CASES)


@st.cache_resource(max_entries=4, show_spinner="Loading the result…")
def _case2d(path, mtime):
    return V.Case2D(path)


@st.cache_resource(max_entries=3, show_spinner="Loading the 3D result…")
def _case3d(path, mtime):
    return V.Case3D(path)


res = _results()
if not res:
    st.info("No results in cases/ yet. Run an airfoil polar, or a 3D wing: "
            "`python -m fv3d.wing3d ... --out ../cases/wing3d`.")
    st.stop()

c1, c2, c3 = st.columns([1, 3, 1], vertical_alignment="bottom")
kinds = c1.segmented_control("Type", ["All", "2D", "3D"], default="All")
pool = [r for r in res if kinds in (None, "All") or r["kind"] == kinds.lower()]
if not pool:
    st.info("No results of that type.")
    st.stop()
sel = c2.selectbox("Result", pool, format_func=lambda r: r["label"])
if c3.button("Refresh list", width="stretch"):
    _results.clear()
    st.rerun()
mt = Path(sel["path"]).stat().st_mtime


def _show(fig):
    st.pyplot(fig, width="stretch")
    import matplotlib.pyplot as plt
    plt.close(fig)


# ======================================================================
# 2D
# ======================================================================
if sel["kind"] == "2d":
    case = _case2d(sel["path"], mt)
    meta = case.meta
    if meta.get("coefficients"):
        k = meta["coefficients"]
        cols = st.columns(5)
        cols[0].metric("α", f"{meta.get('alpha_deg', 0):g}°")
        cols[1].metric("Re", C.fmt_re(meta.get("Re", 0)) if meta.get("Re") else "–")
        for col, name in zip(cols[2:], ("Cl", "Cd", "Cm")):
            v = k.get(name)
            col.metric(name, f"{v:.4f}" if isinstance(v, (int, float)) else "–")
    tabs = st.tabs(["Field", "Mesh", "Wall", "Convergence", "Details"])

    with tabs[0]:
        if not case.has_results:
            st.info("Mesh only (no solution in this folder). The Mesh tab shows it.")
        else:
            a, b, c, d = st.columns([2, 2, 1, 1])
            field = a.selectbox("Field", case.fields(), key="f2")
            view = b.segmented_control("View", list(V.VIEWS), default="Near", key="v2") or "Near"
            cmap = c.selectbox("Colours", CMAPS, key="cm2")
            edges = d.toggle("Cell edges", key="e2")
            stream = d.toggle("Streamlines", key="s2", disabled="U" not in case.data)
            with st.expander("Custom window"):
                x0, x1, y0, y1 = V.view_box(case, view)
                q = st.columns(4)
                box = (q[0].number_input("x min", value=float(x0), format="%.3f"),
                       q[1].number_input("x max", value=float(x1), format="%.3f"),
                       q[2].number_input("y min", value=float(y0), format="%.3f"),
                       q[3].number_input("y max", value=float(y1), format="%.3f"))
            _show(V.figure_2d(case, field, box=box, cmap=cmap, edges=edges, streamlines=stream))

    with tabs[1]:
        a, b = st.columns([3, 2])
        view_m = a.segmented_control("View", list(V.VIEWS), default="Leading edge", key="vm") or "Leading edge"
        qf = [f for f in ("min_angle", "nonorthogonality", "aspect_ratio", "size_ratio", "boundary_layer")
              if f in case.data]
        colour = b.selectbox("Colour by", ["mesh only"] + qf, key="qm")
        if colour == "mesh only":
            _show(V.figure_2d(case, None, view_m))
        else:
            _show(V.figure_2d(case, colour, view_m, edges=True, cmap="viridis"))
        st.caption(f"{len(case.cells):,} cells")
        q = meta.get("quality")
        if q:
            st.json(q, expanded=False)

    with tabs[2]:
        walls = case.walls()
        if not walls:
            st.info("No wall distributions in this folder.")
        for name, w in walls.items():
            fig = go.Figure()
            for side, lab in ((1, "upper"), (-1, "lower")):
                m = w["side"] == side if "side" in w else np.ones(len(w["x"]), bool)
                o = np.argsort(w["x"][m])
                fig.add_scatter(x=w["x"][m][o], y=w["Cp"][m][o], mode="lines", name=f"Cp {lab}")
            fig.update_layout(title=f"Pressure, {name}", xaxis_title="x/c", yaxis=ui.cp_axis([w["Cp"]]))
            ui.plot(fig, key="cp" + name, height=360)
            if "Cf" in w:
                fig = go.Figure()
                for side, lab in ((1, "upper"), (-1, "lower")):
                    m = w["side"] == side if "side" in w else np.ones(len(w["x"]), bool)
                    o = np.argsort(w["x"][m])
                    fig.add_scatter(x=w["x"][m][o], y=w["Cf"][m][o], mode="lines", name=f"Cf {lab}")
                fig.update_layout(title=f"Skin friction, {name}", xaxis_title="x/c", yaxis_title="Cf")
                ui.plot(fig, key="cf" + name, height=320)
            if "yplus" in w:
                st.caption(f"y+ max {np.nanmax(w['yplus']):.2f}, mean {np.nanmean(w['yplus']):.2f}")

    with tabs[3]:
        h = case.history()
        if h is None:
            st.info("No convergence history in this folder.")
        else:
            unsteady = "time" in h and "iteration" not in h               # transient run: per time step
            xk = "time" if unsteady else ("iteration" if "iteration" in h else list(h)[0])
            fig = go.Figure()
            for k in h:
                if k in (xk, "iteration", "step", "time", "outer", "time_s", "Cl", "Cd", "Cm"):
                    continue
                fig.add_scatter(x=h[xk], y=np.abs(h[k]), mode="lines", name=k)
            fig.update_layout(title="Residuals" + (" (end of each time step)" if unsteady else ""),
                              xaxis_title=xk, yaxis_type="log")
            ui.plot(fig, key="res2", height=360)
            if "Cl" in h:
                fig = go.Figure()
                ok = np.isfinite(h["Cl"])
                fig.add_scatter(x=h[xk][ok], y=h["Cl"][ok], mode="lines", name="Cl")
                fig.add_scatter(x=h[xk][ok], y=h["Cd"][ok], mode="lines", name="Cd", yaxis="y2")
                fig.update_layout(title="Forces", xaxis_title=xk, yaxis_title="Cl",
                                  yaxis2=dict(title="Cd", overlaying="y", side="right"))
                ui.plot(fig, key="frc2", height=320)

    with tabs[4]:
        st.caption(sel["path"])
        st.json(meta, expanded=False)

# ======================================================================
# 3D
# ======================================================================
else:
    case = _case3d(sel["path"], mt)
    info = case.info
    cols = st.columns(6)
    cols[0].metric("α", f"{info['alpha']:g}°")
    cols[1].metric("Re (root)", C.fmt_re(info.get("re_root", 0)))
    for col, name in zip(cols[2:], ("CL", "CD", "Cm")):
        col.metric(name, f"{info.get(name, float('nan')):.4f}")
    cols[5].metric("Cells", f"{len(case.layer):,}")
    tabs = st.tabs(["Surface", "Span cut", "Crossflow cut", "Mesh", "Loads", "Details"])

    with tabs[0]:
        a, b, c = st.columns([2, 1, 1])
        sf = a.segmented_control("Colour by", ["Cp", "Cf"], default="Cp", key="sf") or "Cp"
        ed = b.toggle("Surface mesh", key="sed")
        sym = c.toggle("Symmetry-plane mesh", key="ssym")
        ui.plot(V.surface_figure(case, sf, edges=ed, sym_mesh=sym), key="surf", height=640)
        st.caption("Drag to rotate, scroll to zoom. z is the span (root at z = 0), y is up.")

    s_half = float(info["semi_span"])
    with tabs[1]:
        a, b, c, d = st.columns([2, 2, 1, 1])
        zc = a.slider("Span station z/c", 0.0, round(1.5 * s_half, 2), round(0.5 * s_half, 2), 0.01,
                      key="zc")
        field = b.selectbox("Field", case.fields(), key="f3")
        cmap = c.selectbox("Colours", CMAPS, key="cm3")
        edges = d.toggle("Cell edges", key="e3")
        l = case.layer_index(zc)
        P2, cells, ids = case.span_cut(l)
        ch = 1.0 + (float(info.get("taper", 1.0)) - 1.0) * min(zc / s_half, 1.0)
        view = st.segmented_control("View", ["Near", "Leading edge", "Trailing edge", "Wake"],
                                    default="Near", key="v3") or "Near"
        x0, x1, y0, y1 = V.VIEWS[view]
        box = (x0 * ch, x1 * ch, y0 * ch, y1 * ch)
        if zc <= s_half:                                              # sweep moves the section
            xq = 0.25 + zc * np.tan(np.radians(float(info.get("sweep_c4", 0.0))))
            box = (xq + (x0 - 0.25) * ch, xq + (x1 - 0.25) * ch, y0 * ch, y1 * ch)
        mode = st.radio("Show", ["Field", "Mesh"], horizontal=True, key="m3")
        if mode == "Mesh":
            _show(V.field_figure(P2, cells, None, box, mesh_only=True))
        else:
            _show(V.field_figure(P2, cells, case.values(field, ids), box, field=field, cmap=cmap, edges=edges))
        st.caption(f"Layer {l} of {len(case.z) - 1}: z/c {case.z[l]:.4f} to {case.z[l + 1]:.4f}"
                   + ("  (beyond the tip)" if case.z[l] >= s_half - 1e-9 else ""))

    with tabs[2]:
        a, b, c = st.columns([2, 2, 1])
        xc = a.slider("Plane x/c", -0.5, 4.0, 1.2, 0.05, key="xc")
        f = b.selectbox("Field", [x for x in ("vorticity_x", "vorticity_mag", "U_mag", "Uy", "Uz", "Cp",
                                              "nut_ratio") if x in case.fields()], key="fx")
        vec = c.toggle("Velocity arrows", value=True, key="vx")
        zr = st.slider("Span range z/c", 0.0, round(2.0 * s_half, 2), (round(0.6 * s_half, 2),
                                                                      round(1.3 * s_half, 2)), 0.01, key="zr")
        yh = st.slider("Height ± y/c", 0.1, 2.0, 0.4, 0.05, key="yh")
        _show(V.xcut_figure(case, xc, f, zr=zr, yr=(-yh, yh), vectors=vec))
        st.caption("Sampled from the nearest cells (4-point average). Behind the wing this shows the "
                   "tip vortex; streamwise vorticity is negative for a left-turning (port) tip vortex.")

    with tabs[3]:
        st.markdown("**Surface and symmetry-plane mesh**")
        ui.plot(V.surface_figure(case, "Cp", edges=True, sym_mesh=True), key="meshv", height=600)
        st.caption(f"{len(case.layer):,} cells in {len(case.z) - 1} spanwise layers of "
                   f"{len(case.cells2):,} section cells; the Span cut tab shows any layer's mesh.")

    with tabs[4]:
        wj = case.path.parent / "wing3d.json"
        if wj.exists():
            d = json.loads(wj.read_text())
            rows = d.get("results", [])
            st.dataframe([{k: r[k] for k in ("alpha", "CL", "CD", "CDp", "CDv", "Cm", "iterations", "status")
                           if k in r} for r in rows], hide_index=True, width="stretch")
            fig = go.Figure()
            for r in rows:
                sp = r.get("span")
                if sp:
                    fig.add_scatter(x=sp["eta"], y=sp["clc"], mode="lines+markers", name=f"α {r['alpha']:g}°")
            fig.update_layout(title="Span loading", xaxis_title="η = 2y/b", yaxis_title="c_l c / c_root")
            ui.plot(fig, key="span", height=360)
            fig = go.Figure()
            for r in rows:
                h = r.get("history", [])
                if h:
                    fig.add_scatter(x=[x["it"] for x in h], y=[x["CL"] for x in h], mode="lines",
                                    name=f"CL, α {r['alpha']:g}°")
            fig.update_layout(title="Lift during the iterations", xaxis_title="iteration", yaxis_title="CL")
            ui.plot(fig, key="hist3", height=320)
        else:
            st.info("wing3d.json not found next to this result.")

    with tabs[5]:
        st.caption(str(case.path))
        st.json(info, expanded=False)
