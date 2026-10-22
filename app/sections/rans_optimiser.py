import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
import ui
import views

C.page_setup()
ui.header("Shape optimizer", "Minimum drag at your design lift. Smooth CST changes to the shape, "
                             "every candidate solved with RANS, a surrogate model steering the "
                             "search; the winner is re-checked on the validated mesh.")
from aerolab.airfoil import cst

setup, info = st.columns([5, 6], gap="large")
with setup:
    with st.container(border=True):
        ui.step("Start shape", first=True)
        start, sname = views.airfoil_picker("opt", "Start from")
        ui.step("Design point")
        c1, c2 = st.columns(2)
        re = c1.number_input("Reynolds number", 1e5, 5e8, 3e6, format="%.3e")
        cls = c2.text_input("Design Cl", "0.6", help="Several values, comma separated, minimise "
                                                     "their average drag (a robust design)")
        tr = st.segmented_control("Transition", ["free", "off"], default="free",
                                  format_func=lambda s: {"free": "free (e^N)", "off": "fully turbulent"}[s],
                                  help="Free transition lets the optimizer use laminar flow, where "
                                       "most of the gain is; fully turbulent is more robust") or "free"
        ui.step("Constraints")
        t_start = cst.thickness(*cst.fit(C.raw_outline(start), 8)[:2])[0] if start else 0.12
        c1, c2 = st.columns(2)
        t_min = c1.number_input("Minimum t/c", 0.02, 0.4, round(0.98 * t_start, 4), 0.005)
        use_cm = c2.checkbox("Limit Cm, c/4")
        cm_min = c2.number_input("Cm at least", -0.5, 0.2, -0.08, 0.01, disabled=not use_cm,
                                 label_visibility="collapsed")
        with st.expander("Search settings"):
            c1, c2 = st.columns(2)
            evals = c1.number_input("CFD evaluations", 8, 300, 30)
            nvar = c2.number_input("Design variables per surface", 2, 8, 4)
            bound = c1.number_input("Largest change per variable", 0.005, 0.15, 0.04, 0.005)
            level = c2.select_slider("Mesh level during the search", [0.7, 1.0], 0.7)
        n_cl = len(cls.split(","))
        lo = evals * n_cl * 2.5 * (0.5 if level == 0.7 else 1.2)
        hi = evals * n_cl * 2.5 * (1.0 if level == 0.7 else 2.5)
        c1, c2 = st.columns(2, vertical_alignment="center")
        go_ = c1.button("Start optimization", type="primary", disabled=start is None, width="stretch")
        c2.caption(f"About {lo:.0f}–{hi:.0f} min, plus verification")
with info:
    if start:
        views.section_plot(start, height=220, title=f"Start shape, t/c {100 * t_start:.2f} %")
    st.markdown(
        "Objective: Cd at the design Cl, which is maximum L/D there. Each evaluation finds the angle "
        "for the design Cl with 2–3 CFD runs. Constraint violations are added to the objective as "
        "penalties.\n\n"
        "With free transition on NACA 2412 at Re 3e6, Cl 0.5, the demo found 15 % less drag by "
        "moving transition aft. Fully turbulent, the same search found no real gain.")

if go_:
    out = C.CASES / f"optimize_{sname}_Re{re:.2g}_{int(time.time())}"
    args = ["design_optimize.py", "--start", start, "--re", f"{re:.6g}", "--cl", cls,
            "--t-min", t_min, "--evals", evals, "--n-var", nvar, "--bound", bound,
            "--refine", level, "--transition", tr, "--out", out]
    if use_cm:
        args += ["--cm-min", cm_min]
    st.session_state["opt_job"] = C.start_job("optimize", f"Optimize {sname}, Re {C.fmt_re(re)}, Cl {cls}",
                                              args, out)


def show(out):
    f = Path(out) / "optimize.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    H = d["history"]
    v = d.get("verification_level1")
    if v and v["best"].get("Cd") and v["start"].get("Cd"):
        k = st.columns(4)
        k[0].metric("Cd, start", f"{v['start']['Cd']:.5f}")
        k[1].metric("Cd, optimized", f"{v['best']['Cd']:.5f}",
                    f"{100 * (v['best']['Cd'] / v['start']['Cd'] - 1):+.1f} %", delta_color="inverse")
        k[2].metric("t/c, optimized", f"{100 * v['best']['t']:.2f} %")
        k[3].metric("Evaluations", len(H))
        st.caption("Verified on the validated mesh (level 1).")
    y = [h["obj"] if h["obj"] < 0.9 else None for h in H]
    best = np.minimum.accumulate([q if q is not None else 1 for q in y])
    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=list(range(1, len(y) + 1)), y=y, mode="markers", name="evaluation",
                    marker=dict(color=ui.GRAPHITE))
    fig.add_scatter(x=list(range(1, len(y) + 1)), y=best, mode="lines", name="best so far",
                    line=dict(shape="hv"))
    fig.update_layout(title="Search history", xaxis_title="CFD evaluation",
                      yaxis_title="Cd at design Cl, with penalties", height=340)
    ui.plot(fig, container=c1, key=f"oh{out}")
    B = np.array(d["B"]); n = B.shape[1]
    au0, al0 = np.array(d["au0"]), np.array(d["al0"])
    vb = np.array(d["best"]["v"])
    o0 = cst.outline(au0, al0); ob = cst.outline(au0 + B @ vb[:n], al0 + B @ vb[n:])
    fig = go.Figure()
    fig.add_scatter(x=o0[:, 0], y=o0[:, 1], mode="lines", line=dict(dash="dot", color=ui.GRAPHITE), name="start")
    fig.add_scatter(x=ob[:, 0], y=ob[:, 1], mode="lines", name="best so far")
    fig.update_layout(yaxis=dict(scaleanchor="x", zeroline=False), xaxis=dict(zeroline=False), height=340,
                      title="Shape")
    ui.plot(fig, container=c2, key=f"os{out}")
    if v:
        st.dataframe(pd.DataFrame([{"design": k, **p} for k in ("start", "best")
                                   for p in v[k].get("points", [])]), hide_index=True)
    with st.expander("All evaluations"):
        st.dataframe(pd.DataFrame([{k: h.get(k) for k in ("i", "tag", "obj", "Cd", "t")} for h in H]),
                     hide_index=True, width="stretch")
    views.downloads(out, ("best.dat", "optimize.json"), "o")


jid = st.session_state.get("opt_job")
if jid:
    st.markdown("### Run")
    views.live_job(jid, "optimize")

    @st.fragment(run_every="10s")
    def live():
        show(C.get_job(jid)["out"])
    live()

olds = sorted(C.CASES.glob("optimize_*/optimize.json"), key=lambda p: -p.stat().st_mtime)
if olds:
    st.markdown("### Earlier optimizations")
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    pick = c1.selectbox("Result", olds, format_func=lambda p: p.parent.name)
    if c2.button("Open", width="stretch"):
        st.session_state["opt_open"] = str(pick.parent)
    if st.session_state.get("opt_open"):
        show(st.session_state["opt_open"])
