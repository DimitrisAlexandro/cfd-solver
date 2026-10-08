import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
import views

C.page_setup("Multi-element airfoils", "🪶")
st.title("🪶 Multi-element airfoils")
st.caption("Slat / main / flap systems on one hybrid mesh, fully turbulent SA (validated on "
           "the JAXA 30P30N). Build your own configuration or use a preset.")

mode = st.radio("configuration", ["preset", "build my own"], horizontal=True)
if mode == "preset":
    label = st.selectbox("preset", list(C.MULTI_PRESETS))
    spec = C.MULTI_PRESETS[label]
else:
    st.markdown("Elements in order (first = main). Position: leading edge (x, y) in main-chord "
                "units; deflection positive = trailing edge down.")
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

import multi_element as me
import panel
try:
    elements, info = me.load_multi(spec)
except Exception as e:
    st.error(f"geometry problem: {e}")
    st.stop()
a = st.slider("preview α (deg)", -5.0, 25.0, 8.0, 0.5)
pr = panel.solve([e["outline"] for e in elements], a, ref_chord=info["ref_chord"],
                 names=[e["name"] for e in elements])
c1, c2 = st.columns(2)
f = go.Figure()
for e in elements:
    f.add_scatter(x=e["loop"][:, 0], y=e["loop"][:, 1], mode="lines", name=e["name"], fill="toself")
f.update_layout(yaxis=dict(scaleanchor="x"), height=330, margin=dict(l=10, r=10, t=30, b=10),
                title="geometry; smallest gaps: " + ", ".join(f"{k} {v:.4f}" for k, v in info["min_gaps"].items()))
c1.plotly_chart(f, width="stretch")
f = go.Figure()
for e in pr["elements"]:
    f.add_scatter(x=e["x"], y=e["Cp"], mode="lines", name=f"{e['name']} (Cl {e['Cl']:.3f})")
f.update_yaxes(autorange="reversed", title="Cp"); f.update_xaxes(title="x/c")
f.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10),
                title=f"inviscid panel method: Cl {pr['Cl']:.3f}, Cm {pr['Cm']:+.3f}")
c2.plotly_chart(f, width="stretch")

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    re = c1.number_input("Re (reference chord)", 1e5, 5e8, 1.71e6 if "30p30n" in spec else 3e6, format="%.3e")
    mach = c2.number_input("Mach (0 = incompressible)", 0.0, 0.6, 0.0, 0.01)
    alist = c3.text_input("angles (deg)", "4,8,12,16")
    level = c4.select_slider("mesh level", [0.7, 1.0, 1.4], 1.0)
    jobs = st.number_input("parallel processes", 1, 32, 2)
    name = st.text_input("run name", "")
    go_ = st.button("▶ Run", type="primary")
if go_:
    out = C.CASES / (name or f"multi_{me.parse_spec(spec)['name'][:30]}_Re{re:.2g}").replace(" ", "_").replace("+", "_").replace(":", "_").replace(",", "_")
    args = ["airfoil.py", "polar", spec, "--re", f"{re:.6g}", f"--alpha={alist}", "--refine", level,
            "--jobs", jobs, "--out", out]
    if mach > 0:
        args += ["--mach", f"{mach:.4g}"]
    jid = C.start_job("polar", f"multi-element {spec[:40]} Re {re:.3g}", args, out,
                      {"n_angles": len(alist.split(","))})
    st.session_state["last_multi_job"] = jid
    st.success("started — ~5–10 min per angle on a 2-core machine")
jid = st.session_state.get("last_multi_job")
if jid:
    views.live_job(jid, "polar")
    j = C.get_job(jid)
    if j["status"] == "finished":
        views.polar_results(j["out"])
