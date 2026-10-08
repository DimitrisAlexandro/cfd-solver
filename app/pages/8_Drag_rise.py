import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
import views

C.page_setup("Drag-rise explorer", "🌀")
st.title("🌀 Drag-rise / transonic explorer")
st.caption("Mach sweep at a fixed lift coefficient with the compressible RANS solver: the angle "
           "for the target Cl is found at every Mach, drag rises as shocks form. Drag-divergence "
           "Mach by two criteria + Korn's equation.")
af, aname = views.airfoil_picker("dr")
c1, c2, c3 = st.columns(3)
re = c1.number_input("Re", 1e5, 5e8, 6e6, format="%.3e")
cl = c2.number_input("target Cl", -0.5, 1.5, 0.3, 0.05)
level = c3.select_slider("mesh level", [0.7, 1.0, 1.4], 0.7)
c1, c2, c3 = st.columns(3)
m0 = c1.number_input("Mach from", 0.3, 0.9, 0.5, 0.02)
m1 = c2.number_input("Mach to", 0.4, 0.95, 0.82, 0.02)
dm = c3.number_input("step", 0.01, 0.2, 0.04, 0.01)
tr = st.radio("transition", ["fully turbulent", "fixed trips"], horizontal=True)
xtr = st.text_input("trip x/c upper,lower", "0.05,0.05", disabled=tr != "fixed trips")
nM = int((m1 - m0) / dm + 1e-9) + 1
st.caption(f"{nM} Mach numbers × 2–3 runs ≈ {nM * 2.5 * (0.5 if level == 0.7 else 1.5):.0f}–"
           f"{nM * 2.5 * (1.5 if level == 0.7 else 4):.0f} min")
if st.button("▶ Run Mach sweep", type="primary", disabled=af is None):
    out = C.CASES / f"drag_rise_{aname}_Cl{cl:g}_{int(__import__('time').time())}"
    args = ["design_drag_rise.py", "--airfoil", af, "--re", f"{re:.6g}", "--cl", cl,
            "--mach", f"{m0:g}:{m1:g}:{dm:g}", "--refine", level, "--out", out]
    if tr == "fixed trips":
        args += ["--transition", xtr]
    st.session_state["dr_job"] = C.start_job("drag_rise", f"drag rise {aname} Cl {cl:g}", args, out)


def show(out):
    out = Path(out)
    f = out / "drag_rise.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    P = d["points"]
    M = [p["mach"] for p in P]
    k = st.columns(4)
    for c, (key, lab) in zip(k, (("M_dd_douglas", "M_dd (dCd/dM = 0.1)"),
                                 ("M_dd_boeing", "M_dd (+20 counts)"),
                                 ("M_dd_korn_conventional", "Korn, conventional"),
                                 ("M_dd_korn_supercritical", "Korn, supercritical"))):
        c.metric(lab, f"{d[key]:.3f}" if d.get(key) else "–")
    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=M, y=[p["Cd"] for p in P], mode="lines+markers", name="Cd")
    fig.add_scatter(x=M, y=[p["Cd_p"] for p in P], mode="lines", line=dict(dash="dash"), name="pressure + wave")
    fig.add_scatter(x=M, y=[p["Cd_v"] for p in P], mode="lines", line=dict(dash="dot"), name="friction")
    for key, col in (("M_dd_douglas", "red"), ("M_dd_korn_conventional", "gray")):
        if d.get(key):
            fig.add_vline(x=d[key], line=dict(color=col, dash="dash"))
    fig.update_layout(xaxis_title="Mach", yaxis_title="Cd", height=360, margin=dict(l=10, r=10, t=30, b=10),
                      title=f"Cl = {d['cl']:g}, t/c {d['thickness']:.3f}")
    c1.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    fig.add_scatter(x=M, y=[p["alpha"] for p in P], mode="lines+markers", name="α")
    fig.add_scatter(x=M, y=[p["M_max"] for p in P], mode="lines+markers", name="max local Mach", yaxis="y2")
    fig.update_layout(xaxis_title="Mach", yaxis_title="α (deg)", height=360,
                      yaxis2=dict(overlaying="y", side="right", title="max local Mach"),
                      margin=dict(l=10, r=10, t=30, b=10))
    c2.plotly_chart(fig, width="stretch")
    have = [p for p in P if (Path(p["fields"]) / "results.vtu").exists()]
    if have:
        p = have[st.select_slider("Mach field at", list(range(len(have))), len(have) - 1,
                                  format_func=lambda i: f"M {have[i]['mach']:.3f}", key=f"drm{out}")]
        fig, _ = C.field_image(p["fields"], "Mach", (-0.3, 1.5), (-0.5, 0.6))
        st.pyplot(fig)
    st.dataframe(pd.DataFrame([{k: p[k] for k in ("mach", "alpha", "Cl", "Cd", "Cd_p", "Cd_v", "Cm",
                                                  "M_max", "status", "runs")} for p in P]),
                 hide_index=True, width="stretch")
    if (out / "drag_rise_fields.png").exists():
        st.image(str(out / "drag_rise_fields.png"))


jid = st.session_state.get("dr_job")
if jid:
    views.live_job(jid, "drag_rise")

    @st.fragment(run_every="15s")
    def live():
        show(C.get_job(jid)["out"])
    live()
st.divider()
olds = sorted(C.CASES.glob("drag_rise_*/drag_rise.json"), key=lambda p: -p.stat().st_mtime)
if olds:
    st.subheader("Earlier sweeps")
    pick = st.selectbox("open", olds, format_func=lambda p: p.parent.name)
    if st.button("show sweep"):
        show(pick.parent)
