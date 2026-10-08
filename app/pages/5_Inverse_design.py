import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
import views

C.page_setup("Inverse design", "✏️")
st.title("✏️ Inverse design")
st.caption("Say what pressure distribution you want; get the airfoil that produces it. "
           "Shape = CST airfoil, matched with the panel method in seconds (inviscid), then "
           "check it with the viscous CFD analysis.")
import airfoil_geom as ag
import cst
import design_inverse as DI

c1, c2 = st.columns(2)
with c1:
    st.subheader("1 · Base pressure distribution")
    base, bname = views.airfoil_picker("invbase")
    alpha = st.slider("design angle of attack (deg)", -4.0, 10.0, 2.0, 0.25)
with c2:
    st.subheader("2 · Starting shape")
    start, sname = views.airfoil_picker("invstart")
    t_min = st.number_input("minimum thickness / chord (0 = free)", 0.0, 0.3, 0.0, 0.005)
if base is None or start is None:
    st.stop()


@st.cache_data
def base_cp(spec, alpha):
    au, al, _ = cst.fit(C.raw_outline(spec), 8)
    x, cp, side, r = DI.panel_cp(au, al, alpha)
    return x, cp, side, r["Cl"]


x0, cp0, side0, cl0 = base_cp(base, alpha)
st.subheader("3 · Target: edit the base distribution")
mode = st.radio("how", ["smooth changes", "edit the points", "upload CSV (x, Cp, side)"], horizontal=True)
x_t, cp_t, side_t = x0.copy(), cp0.copy(), side0.copy()
if mode == "smooth changes":
    st.markdown("Each row adds a smooth bump ΔCp (negative = more suction) on one surface.")
    bumps = st.data_editor(pd.DataFrame([
        {"surface": "upper", "x_center": 0.15, "width": 0.15, "dCp": 0.0},
        {"surface": "upper", "x_center": 0.6, "width": 0.2, "dCp": 0.0},
        {"surface": "lower", "x_center": 0.8, "width": 0.15, "dCp": 0.0}]),
        num_rows="dynamic", column_config={
            "surface": st.column_config.SelectboxColumn(options=["upper", "lower"]),
            "dCp": st.column_config.NumberColumn(step=0.05)}, width="stretch")
    for _, b in bumps.iterrows():
        s = side_t == (1 if b.surface == "upper" else -1)
        cp_t[s] += b.dCp * np.exp(-((x_t[s] - b.x_center) / max(b.width, 1e-3)) ** 2)
elif mode == "edit the points":
    df = st.data_editor(pd.DataFrame({"x": x0, "Cp": cp0, "side": side0}), height=300,
                        width="stretch", disabled=["x", "side"])
    cp_t = df["Cp"].to_numpy()
else:
    up = st.file_uploader("CSV with header x,Cp,side", type=["csv", "txt"])
    if up is None:
        st.stop()
    d = np.genfromtxt(io.BytesIO(up.getvalue()), delimiter=",", names=True)
    x_t, cp_t, side_t = d["x"], d["Cp"], d["side"].astype(int)

f = go.Figure()
for sd, nm in ((1, "upper"), (-1, "lower")):
    s = side0 == sd
    o = np.argsort(x0[s])
    f.add_scatter(x=x0[s][o], y=cp0[s][o], mode="lines", line=dict(dash="dot", width=1),
                  name=f"base {nm}")
    s = side_t == sd
    o = np.argsort(x_t[s])
    f.add_scatter(x=x_t[s][o], y=cp_t[s][o], mode="lines+markers", marker=dict(size=3),
                  name=f"target {nm}")
f.update_yaxes(autorange="reversed", title="Cp"); f.update_xaxes(title="x/c")
f.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                title=f"base {bname} at α {alpha:g}° (inviscid Cl {cl0:.3f})")
st.plotly_chart(f, width="stretch")

if st.button("▶ Design the airfoil", type="primary"):
    s0 = cst.fit(C.raw_outline(start), 8)[:2]
    with st.spinner("matching the pressure distribution..."):
        r = DI.design(x_t, cp_t, side_t, alpha, s0, t_min=t_min)
    st.session_state["inv_result"] = (r, s0, bname, alpha)

if "inv_result" in st.session_state:
    r, s0, bname_, alpha_ = st.session_state["inv_result"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cp match (rms)", f"{r['rms']:.4f}")
    k2.metric("inviscid Cl", f"{r['Cl']:.3f}")
    k3.metric("thickness", f"{100 * r['thickness']:.2f} % c", f"at x/c {r['x_thickness']:.2f}")
    k4.metric("panel solutions", r["evaluations"])
    c1, c2 = st.columns(2)
    f = go.Figure()
    for sd, nm in ((1, "upper"), (-1, "lower")):
        s = side_t == sd
        o = np.argsort(x_t[s])
        f.add_scatter(x=x_t[s][o], y=cp_t[s][o], mode="markers", marker=dict(size=4), name=f"target {nm}")
        s = r["side"] == sd
        o = np.argsort(r["x"][s])
        f.add_scatter(x=r["x"][s][o], y=r["cp"][s][o], mode="lines", name=f"designed {nm}")
    f.update_yaxes(autorange="reversed", title="Cp"); f.update_xaxes(title="x/c")
    f.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    c1.plotly_chart(f, width="stretch")
    f = go.Figure()
    o = cst.outline(r["au"], r["al"])
    f.add_scatter(x=o[:, 0], y=o[:, 1], mode="lines", name="designed", fill="toself")
    o0 = cst.outline(*s0)
    f.add_scatter(x=o0[:, 0], y=o0[:, 1], mode="lines", line=dict(dash="dot"), name="start")
    f.update_layout(yaxis=dict(scaleanchor="x"), height=360, margin=dict(l=10, r=10, t=30, b=10))
    c2.plotly_chart(f, width="stretch")
    name = st.text_input("name of the new airfoil", "my_inverse_design")
    dat = "\n".join([name] + [f"{x:.7f} {y:.7f}" for x, y in cst.outline(r["au"], r["al"], 160)]) + "\n"
    st.download_button("⬇ airfoil .dat", dat, file_name=f"{name}.dat")
    c1, c2 = st.columns(2)
    re = c1.number_input("Re for the CFD check", 1e4, 5e8, 3e6, format="%.3e")
    angs = c2.text_input("angles", "-2:12:2")
    if st.button("▶ Analyse the designed airfoil with CFD"):
        d = C.CASES / "_designs"
        d.mkdir(exist_ok=True)
        p = d / f"{name}.dat"
        p.write_text(dat)
        out = C.CASES / f"{name}_Re{re:.2g}"
        jid = C.start_job("polar", f"{name} (inverse design) Re {re:.3g}",
                          ["airfoil.py", "polar", p, "--re", f"{re:.6g}", f"--alpha={angs}",
                           "--out", out], out, {"n_angles": C.count_angles(angs)})
        st.session_state["inv_job"] = jid
    if st.session_state.get("inv_job"):
        views.live_job(st.session_state["inv_job"], "polar")
