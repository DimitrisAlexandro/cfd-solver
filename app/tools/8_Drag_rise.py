import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
import ui
import views

C.page_setup()
ui.header("Drag rise", "Mach sweep at fixed lift with the compressible RANS solver. The angle for "
                       "the design Cl is found at every Mach; drag rises as shocks form. "
                       "Drag-divergence Mach by two criteria, compared with Korn's equation.")

setup, info = st.columns([5, 6], gap="large")
with setup:
    with st.container(border=True):
        ui.step("Section", first=True)
        af, aname = views.airfoil_picker("dr")
        ui.step("Design point")
        c1, c2 = st.columns(2)
        re = c1.number_input("Reynolds number", 1e5, 5e8, 6e6, format="%.3e")
        cl = c2.number_input("Design Cl", -0.5, 1.5, 0.3, 0.05)
        ui.step("Mach range")
        c1, c2, c3 = st.columns(3)
        m0 = c1.number_input("From", 0.3, 0.9, 0.5, 0.02)
        m1 = c2.number_input("To", 0.4, 0.95, 0.82, 0.02)
        dm = c3.number_input("Step", 0.01, 0.2, 0.04, 0.01)
        with st.expander("Transition and mesh"):
            tr = st.segmented_control("Transition", ["fully turbulent", "fixed trips"],
                                      default="fully turbulent") or "fully turbulent"
            xtr = st.text_input("Trip x/c, upper and lower", "0.05,0.05", disabled=tr != "fixed trips")
            level = st.select_slider("Mesh level", [0.7, 1.0, 1.4], 0.7)
        nM = int((m1 - m0) / dm + 1e-9) + 1
        c1, c2 = st.columns(2, vertical_alignment="center")
        go_ = c1.button("Run Mach sweep", type="primary", disabled=af is None, width="stretch")
        c2.caption(f"{nM} Mach numbers, about {nM * 2.5 * (0.5 if level == 0.7 else 1.5):.0f}–"
                   f"{nM * 2.5 * (1.5 if level == 0.7 else 4):.0f} min")
with info:
    if af:
        views.section_plot(af, height=200, title=f"Section: {aname}")
    try:
        import airfoil_geom as ag
        tc = ag.load_airfoil(af)[1]["max_thickness"] if af else None
    except Exception:
        tc = None
    if tc:
        st.markdown(f"Korn's equation for this section at Cl {cl:g} (t/c {100 * tc:.1f} %): "
                    f"M_dd {0.87 - cl / 10 - tc:.3f} for a conventional section, "
                    f"{0.95 - cl / 10 - tc:.3f} for a supercritical one.")
    st.caption("Demo: NACA 0012, Re 6e6, Cl 0.3 gave M_dd 0.722 (dCd/dM = 0.1); Korn gives 0.720.")

if go_:
    out = C.CASES / f"drag_rise_{aname}_Cl{cl:g}_{int(time.time())}"
    args = ["design_drag_rise.py", "--airfoil", af, "--re", f"{re:.6g}", "--cl", cl,
            "--mach", f"{m0:g}:{m1:g}:{dm:g}", "--refine", level, "--out", out]
    if tr == "fixed trips":
        args += ["--transition", xtr]
    st.session_state["dr_job"] = C.start_job("drag_rise", f"Drag rise {aname}, Cl {cl:g}", args, out)


def show(out):
    out = Path(out)
    f = out / "drag_rise.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    P = d["points"]
    M = [p["mach"] for p in P]
    k = st.columns(4)
    for c, (key, lab) in zip(k, (("M_dd_douglas", "M_dd, dCd/dM = 0.1"),
                                 ("M_dd_boeing", "M_dd, +20 counts"),
                                 ("M_dd_korn_conventional", "Korn, conventional"),
                                 ("M_dd_korn_supercritical", "Korn, supercritical"))):
        c.metric(lab, f"{d[key]:.3f}" if d.get(key) else "not reached")
    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=M, y=[p["Cd"] for p in P], mode="lines+markers", name="Cd")
    fig.add_scatter(x=M, y=[p["Cd_p"] for p in P], mode="lines", line=dict(dash="dash"),
                    name="pressure and wave")
    fig.add_scatter(x=M, y=[p["Cd_v"] for p in P], mode="lines", line=dict(dash="dot"), name="friction")
    bad = [p for p in P if not p.get("cl_reached", True)]
    if bad:
        fig.add_scatter(x=[p["mach"] for p in bad], y=[p["Cd"] for p in bad], mode="markers",
                        marker=dict(symbol="x", size=9, color="#B3261E"), name="design Cl not reached")
    for key, col, lab in (("M_dd_douglas", ui.SERIES[1], "M_dd"), ("M_dd_korn_conventional", ui.GRAPHITE, "Korn")):
        if d.get(key):
            fig.add_vline(x=d[key], line=dict(color=col, dash="dash", width=1),
                          annotation_text=f"{lab} {d[key]:.3f}", annotation_font_size=10)
    fig.update_layout(title=f"Drag at Cl {d['cl']:g}, t/c {d['thickness']:.3f}", xaxis_title="Mach",
                      yaxis_title="Cd", height=360)
    ui.plot(fig, container=c1, key=f"d1{out}")
    fig = go.Figure()
    fig.add_scatter(x=M, y=[p["alpha"] for p in P], mode="lines+markers", name="α")
    fig.add_scatter(x=M, y=[p["M_max"] for p in P], mode="lines+markers", name="max local Mach",
                    yaxis="y2", line=dict(dash="dash"))
    fig.add_hline(y=1.0, line=dict(color=ui.GRAPHITE, width=1, dash="dot"), yref="y2")
    fig.update_layout(title="Angle for the design Cl and peak Mach", xaxis_title="Mach",
                      yaxis_title="α (deg)", height=360,
                      yaxis2=dict(overlaying="y", side="right", title="max local Mach", showgrid=False,
                                  ticks="inside"))
    ui.plot(fig, container=c2, key=f"d2{out}")
    have = [p for p in P if (Path(p["fields"]) / "results.vtu").exists()]
    if have:
        i = st.select_slider("Mach field at", list(range(len(have))), len(have) - 1,
                             format_func=lambda i: f"M {have[i]['mach']:.3f}", key=f"drm{out}")
        fig, _ = C.field_image(have[i]["fields"], "Mach", (-0.3, 1.5), (-0.5, 0.6))
        st.pyplot(fig)
        st.caption("Black line: sonic line, M = 1.")
    st.dataframe(pd.DataFrame([{k: p[k] for k in ("mach", "alpha", "Cl", "Cd", "Cd_p", "Cd_v", "Cm",
                                                  "M_max", "status", "runs")} for p in P]),
                 hide_index=True, width="stretch")


jid = st.session_state.get("dr_job")
if jid:
    st.markdown("### Run")
    views.live_job(jid, "drag_rise")

    @st.fragment(run_every="15s")
    def live():
        show(C.get_job(jid)["out"])
    live()

olds = sorted(C.CASES.glob("drag_rise_*/drag_rise.json"), key=lambda p: -p.stat().st_mtime)
if olds:
    st.markdown("### Earlier sweeps")
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    pick = c1.selectbox("Result", olds, format_func=lambda p: p.parent.name)
    if c2.button("Open", width="stretch"):
        st.session_state["dr_open"] = str(pick.parent)
    if st.session_state.get("dr_open"):
        show(st.session_state["dr_open"])
