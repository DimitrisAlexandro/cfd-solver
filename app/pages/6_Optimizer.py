import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
import views

C.page_setup("Airfoil optimizer", "🎯")
st.title("🎯 Airfoil optimizer")
st.caption("The shape is changed (smooth CST perturbations) and every candidate is solved with "
           "the RANS solver at your target lift; a surrogate model steers the search. Slow (each "
           "evaluation = 2–3 CFD runs) but every number is real CFD. The winner is re-checked on "
           "the validated mesh.")

st.subheader("Start shape")
start, sname = views.airfoil_picker("opt")
st.subheader("Goal")
c1, c2, c3 = st.columns(3)
re = c1.number_input("Re", 1e5, 5e8, 3e6, format="%.3e")
cls = c2.text_input("target Cl (several → average drag, robust design)", "0.6")
goal = c3.selectbox("objective", ["minimum drag at target Cl (= max L/D)"])
st.subheader("Constraints")
c1, c2, c3 = st.columns(3)
import cst
t_start = cst.thickness(*cst.fit(C.raw_outline(start), 8)[:2])[0] if start else 0.12
t_min = c1.number_input("min thickness / chord", 0.02, 0.4, round(0.98 * t_start, 4), 0.005)
use_cm = c2.checkbox("limit pitching moment")
cm_min = c3.number_input("Cm (c/4) ≥", -0.5, 0.2, -0.08, 0.01, disabled=not use_cm)
st.subheader("Search settings")
c1, c2, c3, c4 = st.columns(4)
evals = c1.number_input("CFD evaluations", 8, 300, 30)
nvar = c2.number_input("design variables per side", 2, 8, 4)
bound = c3.number_input("max change per variable", 0.005, 0.15, 0.04, 0.005)
level = c4.select_slider("mesh level during search", [0.7, 1.0], 0.7)
tr = st.radio("transition", ["off", "free"], horizontal=True,
              help="fully turbulent is more robust; free transition optimizes laminar flow too")
n_cl = len(cls.split(","))
st.info(f"estimated time: {evals * n_cl * 2.5 * (0.5 if level == 0.7 else 1.2):.0f}–"
        f"{evals * n_cl * 2.5 * (1.0 if level == 0.7 else 2.5):.0f} min (+ verification)")
if st.button("▶ Optimize", type="primary", disabled=start is None):
    out = C.CASES / f"optimize_{sname}_Re{re:.2g}_{int(__import__('time').time())}"
    args = ["design_optimize.py", "--start", start, "--re", f"{re:.6g}", "--cl", cls,
            "--t-min", t_min, "--evals", evals, "--n-var", nvar, "--bound", bound,
            "--refine", level, "--transition", tr, "--out", out]
    if use_cm:
        args += ["--cm-min", cm_min]
    st.session_state["opt_job"] = C.start_job("optimize", f"optimize {sname} Re {re:.3g} Cl {cls}",
                                              args, out)


def show(out):
    f = Path(out) / "optimize.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    H = d["history"]
    y = [h["obj"] if h["obj"] < 0.9 else None for h in H]
    best = np.minimum.accumulate([v if v is not None else 1 for v in y])
    fig = go.Figure()
    fig.add_scatter(x=list(range(1, len(y) + 1)), y=y, mode="markers", name="evaluation")
    fig.add_scatter(x=list(range(1, len(y) + 1)), y=best, mode="lines", name="best so far")
    fig.update_layout(xaxis_title="CFD evaluation", yaxis_title="Cd at target Cl (+ penalties)",
                      height=330, margin=dict(l=10, r=10, t=30, b=10))
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig, width="stretch")
    B = np.array(d["B"]); n = B.shape[1]
    au0, al0 = np.array(d["au0"]), np.array(d["al0"])
    vb = np.array(d["best"]["v"])
    o0 = cst.outline(au0, al0); ob = cst.outline(au0 + B @ vb[:n], al0 + B @ vb[n:])
    fig = go.Figure()
    fig.add_scatter(x=o0[:, 0], y=o0[:, 1], mode="lines", line=dict(dash="dot"), name="start")
    fig.add_scatter(x=ob[:, 0], y=ob[:, 1], mode="lines", name="best so far")
    fig.update_layout(yaxis=dict(scaleanchor="x"), height=330, margin=dict(l=10, r=10, t=30, b=10))
    c2.plotly_chart(fig, width="stretch")
    v = d.get("verification_level1")
    if v and v["best"].get("Cd") and v["start"].get("Cd"):
        k1, k2, k3 = st.columns(3)
        k1.metric("Cd start (validated mesh)", f"{v['start']['Cd']:.5f}")
        k2.metric("Cd optimized", f"{v['best']['Cd']:.5f}",
                  f"{100 * (v['best']['Cd'] / v['start']['Cd'] - 1):+.1f} %", delta_color="inverse")
        k3.metric("thickness", f"{100 * v['best']['t']:.2f} %")
        st.dataframe(pd.DataFrame([{"design": k, **p} for k in ("start", "best")
                                   for p in v[k].get("points", [])]), hide_index=True)
    rows = [{k: h.get(k) for k in ("i", "tag", "obj", "Cd", "t")} for h in H]
    with st.expander("all evaluations"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for fn in ("best.dat", "optimize.json"):
        p = Path(out) / fn
        if p.exists():
            st.download_button(f"⬇ {fn}", p.read_bytes(), file_name=fn, key=f"o{out}{fn}")
    if (Path(out) / "optimize.png").exists():
        st.image(str(Path(out) / "optimize.png"))


jid = st.session_state.get("opt_job")
if jid:
    views.live_job(jid, "optimize")
    j = C.get_job(jid)

    @st.fragment(run_every="10s")
    def live():
        show(C.get_job(jid)["out"])
    live()
st.divider()
st.subheader("Earlier optimizations")
olds = sorted(C.CASES.glob("optimize_*/optimize.json"), key=lambda p: -p.stat().st_mtime)
if olds:
    pick = st.selectbox("open", olds, format_func=lambda p: p.parent.name)
    if st.button("show"):
        show(pick.parent)
