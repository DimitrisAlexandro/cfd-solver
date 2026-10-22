import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import common as C
import ui

from aerolab.mesh import postview as V
from aerolab.mesh import wakeview as W

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


@st.cache_resource(max_entries=3, show_spinner="Setting up the wake rake…")
def _wake2d(path, mtime):
    return W.Wake2D(_case2d(path, mtime))


@st.cache_data(max_entries=8, show_spinner="Surveying the wake…")
def _survey2d(path, mtime, ds):
    return _wake2d(path, mtime).survey(ds)


@st.cache_resource(max_entries=2, show_spinner="Preparing the wake…")
def _wake3d(path, mtime):
    return W.Wake3D(_case3d(path, mtime))


@st.cache_data(max_entries=6, show_spinner="Surveying the wake planes (a few seconds per plane)…")
def _survey3d(path, mtime, ds):
    return _wake3d(path, mtime).survey(ds)


@st.cache_resource(max_entries=2, show_spinner="Sampling the wake volume…")
def _volume3d(path, mtime, x_end, res):
    wk = _wake3d(path, mtime)
    return W.wake_volume(wk, x_end, res), W.streamlines(wk, x_end)


res = _results()
if not res:
    st.info("No results in cases/ yet. Run an airfoil polar, or a 3D wing: "
            "`python src/run.py wing3d ... --out ../cases/wing3d`.")
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
    t_field, t_wake, t_mesh, t_wall, t_conv, t_info = st.tabs(["Field", "Wake", "Mesh", "Wall", "Convergence", "Details"])

    with t_field:
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

    with t_wake:
        steady = "U" in case.data and not meta.get("transient")
        if not steady:
            st.info("The wake tools need a steady solution with velocity (this run is unsteady or mesh only); "
                    "the Field tab shows its instantaneous wake.")
        else:
            wk = _wake2d(sel["path"], mt)
            c = wk.chord
            a, b, cc = st.columns([2, 2, 1])
            fw = a.selectbox("Field", ["u_deficit"] + [f for f in ("U_mag", "vorticity", "nut_ratio", "Cp")
                                                       if f in case.data],
                             format_func=lambda k: {"u_deficit": "velocity deficit 1 − u/U∞"}.get(k, k), key="fw2")
            Lw = b.slider("Wake length shown (chords)", 1.0, 10.0, 4.0, 0.5, key="lw2")
            rk = cc.toggle("Rake lines", value=True, key="rk2")
            ed2 = cc.toggle("Cell edges", key="ew2")
            st_d = st.multiselect("Rake stations, chords behind the trailing edge",
                                  [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0],
                                  default=[0.1, 0.5, 1.0, 2.0, 4.0], key="ds2")
            h = c * (0.25 + 0.12 * Lw)
            box = (wk.te[0] - 0.3 * c, wk.te[0] + Lw * c, wk.te[1] - h + 0.5 * Lw * c * wk.es[1],
                   wk.te[1] + h + 0.5 * Lw * c * wk.es[1])
            clim = (0.0, 0.25) if fw == "u_deficit" else None
            fig = V.figure_2d(case, fw, box=box, cmap="magma_r" if fw == "u_deficit" else "turbo",
                              clim=clim, edges=ed2, figsize=(12, 4.2))
            rakes = [wk.rake(d) for d in sorted(st_d)]
            if rk:
                ax = fig.axes[0]
                for r in rakes:
                    col = "#1b6ef3" if fw == "u_deficit" else "w"
                    ax.plot(r["pts"][:, 0], r["pts"][:, 1], "-", color=col, lw=0.8)
                    ax.text(r["pts"][-1, 0], r["pts"][-1, 1], f" {r['d']:g}c", color=col, fontsize=7,
                            va="bottom")
            _show(fig)
            if rakes:
                p1, p2 = st.columns(2)
                fig = go.Figure()
                for r in rakes:
                    fig.add_scatter(x=r["u"], y=r["t"], mode="lines", name=f"{r['d']:g} c")
                fig.update_layout(title="Velocity along the free stream", xaxis_title="u / U∞",
                                  yaxis_title="n / c (normal to the free stream)")
                ui.plot(fig, key="wu2", height=380, container=p1)
                fig = go.Figure()
                for r in rakes:
                    fig.add_scatter(x=r["loss"], y=r["t"], mode="lines", name=f"{r['d']:g} c")
                fig.update_layout(title="Total-pressure loss", xaxis_title="(pt∞ − pt) / q∞",
                                  yaxis_title="n / c")
                ui.plot(fig, key="wl2", height=380, container=p2)
            st.markdown("**Wake survey drag** (Betz wake-rake formula, as in a wind tunnel)")
            rows = _survey2d(sel["path"], mt, (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0))
            if rows:
                fig = go.Figure()
                fig.add_scatter(x=[r["d"] for r in rows], y=[r["Cd_wake"] for r in rows], mode="lines+markers",
                                name="from the wake")
                if wk.Cd_surface:
                    fig.add_hline(y=wk.Cd_surface, line_dash="dash",
                                  annotation_text=f"surface forces {wk.Cd_surface:.5f}")
                fig.update_layout(xaxis_title="rake distance behind the trailing edge (chords)",
                                  yaxis_title="Cd", xaxis_type="log")
                ui.plot(fig, key="wcd2", height=330)
                st.dataframe([{"d / c": r["d"], "Cd (wake)": round(r["Cd_wake"], 5),
                               "Cd (total-pressure loss only)": round(r["Cd_loss"], 5),
                               "min u/U∞": round(r["u_min"], 3), "wake width / c": round(r["width"], 3),
                               "wake centre n/c": round(r["centre"], 3)} for r in rows],
                             hide_index=True, width="stretch")
                st.caption("The wake drag should match the surface-force drag. Close to the trailing edge "
                           "the rake sits in the coarse wake cells; far downstream numerical diffusion "
                           "adds total-pressure loss, so a gap that grows with distance is a mesh "
                           "effect, not physics.")

    with t_mesh:
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

    with t_wall:
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

    with t_conv:
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

    with t_info:
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
    t_surf, t_span, t_xcut, t_wake, t_mesh, t_loads, t_info = st.tabs(
        ["Surface", "Span cut", "Crossflow cut", "Wake", "Mesh", "Loads", "Details"])

    with t_surf:
        a, b, c = st.columns([2, 1, 1])
        sf = a.segmented_control("Colour by", ["Cp", "Cf"], default="Cp", key="sf") or "Cp"
        ed = b.toggle("Surface mesh", key="sed")
        sym = c.toggle("Symmetry-plane mesh", key="ssym")
        ui.plot(V.surface_figure(case, sf, edges=ed, sym_mesh=sym), key="surf", height=640)
        st.caption("Drag to rotate, scroll to zoom. z is the span (root at z = 0), y is up.")

    s_half = float(info["semi_span"])
    with t_span:
        a, b, c, d = st.columns([2, 2, 1, 1])
        zc = a.slider("Span station z/c", 0.0, round(1.5 * s_half, 2), round(0.5 * s_half, 2), 0.01,
                      key="zc")
        field = b.selectbox("Field", case.fields(), key="f3")
        cmap = c.selectbox("Colours", CMAPS, key="cm3")
        edges = d.toggle("Cell edges", key="e3")
        l = case.layer_index(zc)
        P2, cells, ids = case.span_cut(l)
        ch = 1.0 + (float(info.get("taper", 1.0)) - 1.0) * min(zc / s_half, 1.0)
        view = st.segmented_control("View", ["Near", "Leading edge", "Trailing edge", "Wake", "Far wake"],
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

    with t_xcut:
        a, b, c = st.columns([2, 2, 1])
        xc = a.slider("Plane x/c", -0.5, round(1.0 + 0.7 * float(info.get("farfield", 12.0)), 1), 1.2, 0.05,
                      key="xc")
        f = b.selectbox("Field", [x for x in ("vorticity_x", "vorticity_mag", "U_mag", "Uy", "Uz", "Cp",
                                              "nut_ratio") if x in case.fields()], key="fx")
        vec = c.toggle("Velocity arrows", value=True, key="vx")
        zr = st.slider("Span range z/c", 0.0, round(2.0 * s_half, 2), (round(0.6 * s_half, 2),
                                                                      round(1.3 * s_half, 2)), 0.01, key="zr")
        yh = st.slider("Height ± y/c", 0.1, 2.0, 0.4, 0.05, key="yh")
        _show(V.xcut_figure(case, xc, f, zr=zr, yr=(-yh, yh), vectors=vec))
        st.caption("Sampled from the nearest cells (4-point average). Behind the wing this shows the "
                   "tip vortex; on this half wing (tip at +z) it has negative streamwise vorticity at positive lift.")

    with t_wake:
        if "vorticity" not in case.d:
            st.info("This result was saved before the wake data existed; run the wing again to see the wake.")
        else:
            wk = _wake3d(sel["path"], mt)
            xmax = round(float(wk.te[0]) + 0.7 * float(info.get("farfield", 12.0)), 1)
            st.markdown("**The whole wake in 3D**")
            a, b, c = st.columns([5, 2, 1.4])
            lay = a.multiselect("Show", W.LAYERS, default=[W.LAYERS[0], W.LAYERS[2], W.LAYERS[3], W.LAYERS[4]],
                                key="lay3")
            xe = b.slider("Downstream to x/c", 2.0, min(xmax, 12.0), min(6.0, xmax), 0.5, key="xe3")
            res = c.selectbox("Detail", list(W.RES), index=1, key="res3")
            with st.expander("Levels"):
                q = st.columns(2)
                lv = {"loss": q[0].slider("Wake sheet: total-pressure loss", 0.01, 0.3, 0.04, 0.01, key="lvl"),
                      "Q": q[1].slider("Tip vortex: Q as a fraction of its peak", 0.005, 0.3, 0.03, 0.005,
                                       key="lvq")}
            vol, lines = _volume3d(sel["path"], mt, float(xe), res)
            ui.plot(W.wake_scene(wk, vol, lay, float(xe), levels=lv, lines=lines), key="scene3", height=720)
            st.caption(
                "Drag to rotate; click a legend entry to hide it. **Turbulence**: nested shells of the eddy "
                "viscosity ν_t/ν from the Spalart–Allmaras model: the turbulent boundary layers leave the "
                "trailing edge, merge into the wake and spread downstream. A RANS solution is the time-averaged "
                "flow, so the turbulence appears as this modelled eddy viscosity, not as individual eddies "
                "(those need LES/DES). **Turbulence intensity** is estimated from ν_t and the strain rate "
                "(Bradshaw: −u′v′ = 0.3 k), since SA carries no k. **Tip vortex**: Q-criterion (rotation "
                "stronger than strain). **Mesh on the wake**: the cells the wake passes through, on the surface "
                "along the free stream from the trailing edge; where the cells grow, the vortex and the wake "
                "smear out.")

            st.markdown("**The wake seen from above**")
            a, b, c, d = st.columns([2, 2, 2, 2])
            pf = a.selectbox("Field", list(W.LABELS), format_func=lambda k: W.LABELS[k], key="pf3")
            pm = b.radio("Show", ["Through the thickness", "Slice"], horizontal=True, key="pm3")
            x1 = c.slider("Downstream to x/c", 2.0, xmax, min(6.0, xmax), 0.5, key="px3")
            off = d.slider("Slice offset (c)", -0.4, 0.4, 0.0, 0.01, key="po3", disabled=pm != "Slice")
            _show(W.plan_figure(wk, pf, x1, off, "slice" if pm == "Slice" else "max"))
            st.caption("The surface through the trailing edge along the free stream. *Through the thickness* "
                       "shows the largest value within ±0.35 c of it, so the wake sheet stays visible as the "
                       "downwash carries it down; the tip vortex is the streak that rolls in from the tip.")

            st.markdown("**Planes behind the wing**")
            a, b = st.columns([2, 3])
            qf = a.selectbox("Colour by", ["loss", "xi", "deficit", "nut_ratio"],
                             format_func=lambda k: W.LABELS[k], key="qf3")
            qd = b.multiselect("Planes, chords behind the trailing edge", [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
                                                                          6.0, 8.0],
                               default=[0.5, 1.5, 3.0, 5.0], key="qd3")
            if qd:
                ui.plot(W.planes_figure(wk, sorted(qd), qf), key="planes3", height=640)
                st.caption("Only the wake is drawn (values above 6 % of the largest). Rotate to look "
                           "along the stream and watch the sheet roll up into the tip vortex.")

            st.markdown("**Wake survey: drag, circulation and the tip vortex**")
            ds = (0.25, 0.5, 1.0, 2.0, 4.0)
            rows = _survey3d(sel["path"], mt, ds)
            AR = float(info.get("aspect_ratio", 2 * s_half))
            CL = float(info["CL"])
            cdi_ell = CL ** 2 / (np.pi * AR)
            p1, p2 = st.columns(2)
            fig = go.Figure()
            x = [r["d"] for r in rows]
            fig.add_scatter(x=x, y=[r["CD"] for r in rows], mode="lines+markers", name="CD (wake)")
            fig.add_scatter(x=x, y=[r["CDp"] for r in rows], mode="lines+markers", name="profile CDp")
            fig.add_scatter(x=x, y=[r["CDi"] for r in rows], mode="lines+markers", name="induced CDi")
            fig.add_hline(y=float(info["CD"]), line_dash="dash", annotation_text="CD surface")
            fig.add_hline(y=cdi_ell, line_dash="dot", annotation_text="CL²/(π AR)")
            fig.update_layout(title="Drag from the wake", xaxis_title="plane distance behind the trailing edge (c)",
                              yaxis_title="CD")
            ui.plot(fig, key="wcd3", height=380, container=p1)
            G0 = None
            wj = case.path.parent / "wing3d.json"
            if wj.exists():
                for r in json.loads(wj.read_text()).get("results", []):
                    if abs(r["alpha"] - info["alpha"]) < 1e-9 and r.get("span"):
                        G0 = 0.5 * float(r["span"]["clc"][0])
            fig = go.Figure()
            fig.add_scatter(x=x, y=[abs(r["Gamma"]) for r in rows], mode="lines+markers", name="shed (whole plane)")
            fig.add_scatter(x=x, y=[r["core_Gamma"] for r in rows], mode="lines+markers", name="in the tip vortex core")
            if G0:
                fig.add_hline(y=G0, line_dash="dash", annotation_text="root bound circulation")
            fig.update_layout(title="Circulation Γ / (U∞ c)", xaxis_title="plane distance (c)", yaxis_title="Γ")
            ui.plot(fig, key="wg3", height=380, container=p2)
            p1, p2 = st.columns(2)
            fig = go.Figure()
            fig.add_scatter(x=[r["core_x"] for r in rows], y=[r["core_z"] for r in rows], mode="lines+markers",
                            name="z (span)")
            fig.add_scatter(x=[r["core_x"] for r in rows], y=[r["core_y"] for r in rows], mode="lines+markers",
                            name="y (height)", yaxis="y2")
            fig.update_layout(title="Tip vortex path", xaxis_title="x/c", yaxis_title="z/c",
                              yaxis2=dict(title="y/c", overlaying="y", side="right"))
            ui.plot(fig, key="wv3", height=340, container=p1)
            fig = go.Figure()
            fig.add_scatter(x=[r["core_x"] for r in rows], y=[r["peak_xi"] for r in rows], mode="lines+markers",
                            name="peak |ξ|")
            fig.add_scatter(x=[r["core_x"] for r in rows], y=[r["core_r"] for r in rows], mode="lines+markers",
                            name="core radius", yaxis="y2")
            fig.update_layout(title="Tip vortex strength and size", xaxis_title="x/c", yaxis_title="peak |ξ| c/U∞",
                              yaxis2=dict(title="radius / c", overlaying="y", side="right"))
            ui.plot(fig, key="wr3", height=340, container=p2)
            k = st.select_slider("Spanwise profile drag at plane (c behind the TE)", ds, value=1.0, key="sd3")
            r = rows[list(ds).index(k)]
            fig = go.Figure()
            zz = np.array(r["z"])
            fig.add_scatter(x=zz / s_half, y=r["cd_c"], mode="lines", name="c_d c from the wake")
            fig.update_layout(xaxis_title="η = z / (b/2)", yaxis_title="c_d c / c_root (total-pressure loss)",
                              title="Profile drag along the span")
            ui.plot(fig, key="wsd3", height=320)
            st.dataframe([{"d / c": r["d"], "CD wake": round(r["CD"], 5), "CDp": round(r["CDp"], 5),
                           "CDi": round(r["CDi"], 5), "Γ shed": round(abs(r["Gamma"]), 4),
                           "core z/c": round(r["core_z"], 3), "core y/c": round(r["core_y"], 3),
                           "core radius / c": round(r["core_r"], 4)} for r in rows],
                         hide_index=True, width="stretch")
            st.caption(f"Betz / Maskell wake survey on planes normal to the free stream. Profile drag is the "
                       f"total-pressure loss; induced drag is the crossflow kinetic energy, ½∫ψξ dA with the "
                       f"symmetry plane as a mirror. The surface CD is {info['CD']:.5f} and the elliptic-wing "
                       f"induced drag CL²/(πAR) is {cdi_ell:.5f}. Numerical diffusion spreads the tip vortex "
                       f"downstream (its peak falls and its radius grows): on a coarse wake mesh the "
                       f"induced drag read far behind the wing drops for that reason.")

    with t_mesh:
        st.markdown("**Surface and symmetry-plane mesh**")
        ui.plot(V.surface_figure(case, "Cp", edges=True, sym_mesh=True), key="meshv", height=600)
        st.caption(f"{len(case.layer):,} cells in {len(case.z) - 1} spanwise layers of "
                   f"{len(case.cells2):,} section cells; the Span cut tab shows any layer's mesh.")

    with t_loads:
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

    with t_info:
        st.caption(str(case.path))
        st.json(info, expanded=False)
