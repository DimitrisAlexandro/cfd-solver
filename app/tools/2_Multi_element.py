import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
import ui
import views

C.page_setup()
ui.header("Multi-element airfoil", "Slat, main and flap on one hybrid mesh, fully turbulent "
                                   "Spalart–Allmaras. Validated on the JAXA 30P30N.")
from aerolab.airfoil import multi_element as me
from aerolab.airfoil import panel

setup, preview = st.columns([5, 6], gap="large")
with setup:
    with st.container(border=True):
        ui.step("Configuration", first=True)
        mode = st.segmented_control("Configuration", ["Preset", "Build your own"], default="Preset",
                                    label_visibility="collapsed") or "Preset"
        if mode == "Preset":
            label = st.selectbox("Preset", list(C.MULTI_PRESETS))
            spec = C.MULTI_PRESETS[label]
        else:
            st.caption("First row is the main element. Leading edge (x, y) in main-chord units; "
                       "positive deflection is trailing edge down.")
            df = st.data_editor(pd.DataFrame([
                {"airfoil": "naca23012", "chord": 1.0, "x_le": 0.0, "y_le": 0.0, "deflection": 0.0},
                {"airfoil": "naca23012", "chord": 0.3, "x_le": 0.98, "y_le": -0.04, "deflection": 25.0}]),
                num_rows="dynamic", width="stretch")
            parts = []
            for i, r in df.iterrows():
                if i == 0 and r.chord == 1 and r.x_le == 0 and r.y_le == 0 and r.deflection == 0:
                    parts.append(str(r.airfoil))
                else:
                    parts.append(f"{r.airfoil}:{r.chord:g},{r.x_le:g},{r.y_le:g},{r.deflection:g}")
            spec = "+".join(parts)
            st.code(spec, language=None)

        ui.step("Flow and angles")
        c1, c2 = st.columns(2)
        re = c1.number_input("Reynolds number, reference chord", 1e5, 5e8,
                             1.71e6 if "30p30n" in spec else 3e6, format="%.3e")
        mach = c2.number_input("Mach (0 = incompressible)", 0.0, 0.6, 0.0, 0.01)
        alist = st.text_input("Angles of attack (deg)", "4, 8, 12, 16")
        with st.expander("Numerics and output"):
            c1, c2 = st.columns(2)
            level = c1.select_slider("Mesh level", [0.7, 1.0, 1.4], 1.0)
            jobs = c2.number_input("Parallel processes", 1, 32, 2)
            name = st.text_input("Run name", "", placeholder="default: configuration and Re")
        n = len([a for a in alist.split(",") if a.strip()])
        c1, c2 = st.columns(2, vertical_alignment="center")
        go_ = c1.button("Run polar", type="primary", width="stretch")
        c2.caption(f"{n} angles, about {5 * n}–{10 * n} min on two cores")

with preview:
    try:
        elements, info = me.load_multi(spec)
    except Exception as e:
        st.error(f"This configuration can't be built: {e}. Check the element positions and chords.")
        st.stop()
    a = st.slider("Preview α (deg)", -5.0, 25.0, 8.0, 0.5,
                  help="Inviscid panel method, instant")
    pr = panel.solve([e["outline"] for e in elements], a, ref_chord=info["ref_chord"],
                     names=[e["name"] for e in elements])
    f = go.Figure()
    for e in elements:
        f.add_scatter(x=e["loop"][:, 0], y=e["loop"][:, 1], mode="lines", name=e["name"],
                      fill="toself", fillcolor="rgba(29,78,137,0.05)")
    f.update_layout(yaxis=dict(scaleanchor="x", zeroline=False), xaxis=dict(zeroline=False), height=260,
                    title="Smallest gaps: " + ", ".join(f"{k} {v:.4f} c" for k, v in info["min_gaps"].items()))
    ui.plot(f)
    f = go.Figure()
    for e in pr["elements"]:
        f.add_scatter(x=e["x"], y=e["Cp"], mode="lines", name=f"{e['name']}, Cl {e['Cl']:.3f}")
    f.update_layout(title=f"Inviscid pressure at α {a:g}°: Cl {pr['Cl']:.3f}, Cm {pr['Cm']:+.3f}",
                    yaxis=ui.cp_axis([e["Cp"] for e in pr["elements"]]), xaxis_title="x/c", height=300)
    ui.plot(f)

if go_:
    angles = alist.replace(" ", "")
    out = C.CASES / (name or f"multi_{me.parse_spec(spec)['name'][:30]}_Re{re:.2g}").replace(" ", "_") \
        .replace("+", "_").replace(":", "_").replace(",", "_")
    args = ["airfoil.py", "polar", spec, "--re", f"{re:.6g}", f"--alpha={angles}", "--refine", level,
            "--jobs", jobs, "--out", out]
    if mach > 0:
        args += ["--mach", f"{mach:.4g}"]
    st.session_state["last_multi_job"] = C.start_job(
        "polar", f"Multi-element {spec[:40]}, Re {C.fmt_re(re)}", args, out, {"n_angles": n})
jid = st.session_state.get("last_multi_job")
if jid:
    st.markdown("### Run")
    views.live_job(jid, "polar")
    j = C.get_job(jid)
    if j["status"] == "finished":
        views.polar_results(j["out"])
