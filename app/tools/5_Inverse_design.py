import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
import ui
import views

C.page_setup()
ui.header("Inverse design", "Give the pressure distribution you want and get the airfoil that "
                            "produces it. CST shape matched with the panel method in seconds; "
                            "check the result with a viscous polar.")
import cst
import design_inverse as DI


@st.cache_data
def base_cp(spec, alpha):
    au, al, _ = cst.fit(C.raw_outline(spec), 8)
    x, cp, side, r = DI.panel_cp(au, al, alpha)
    return x, cp, side, r["Cl"]


setup, target = st.columns([5, 7], gap="large")
panel = setup.container(border=True)
with panel:
    if True:
        ui.step("Base pressure distribution", first=True)
        base, bname = views.airfoil_picker("invbase", "From section")
        alpha = st.number_input("Design angle of attack (deg)", -4.0, 10.0, 2.0, 0.25)
        ui.step("Starting shape")
        start, sname = views.airfoil_picker("invstart", "Start from")
        t_min = st.number_input("Minimum thickness, t/c (0 = free)", 0.0, 0.3, 0.0, 0.005)
        ui.step("Edit the target")
        mode = st.segmented_control("Edit", ["Smooth changes", "Point by point", "Upload CSV"],
                                    default="Smooth changes", label_visibility="collapsed") or "Smooth changes"
if base is None or start is None:
    st.stop()

x0, cp0, side0, cl0 = base_cp(base, alpha)
x_t, cp_t, side_t = x0.copy(), cp0.copy(), side0.copy()
with panel:
    if mode == "Smooth changes":
        st.caption("Each row adds a smooth bump ΔCp to one surface; negative ΔCp is more suction.")
        bumps = st.data_editor(pd.DataFrame([
            {"surface": "upper", "x_center": 0.15, "width": 0.15, "dCp": 0.0},
            {"surface": "upper", "x_center": 0.6, "width": 0.2, "dCp": 0.0},
            {"surface": "lower", "x_center": 0.8, "width": 0.15, "dCp": 0.0}]),
            num_rows="dynamic", column_config={
                "surface": st.column_config.SelectboxColumn("Surface", options=["upper", "lower"]),
                "x_center": st.column_config.NumberColumn("Centre x/c", step=0.01),
                "width": st.column_config.NumberColumn("Width", step=0.01),
                "dCp": st.column_config.NumberColumn("ΔCp", step=0.05)}, width="stretch")
        for _, b in bumps.iterrows():
            s = side_t == (1 if b.surface == "upper" else -1)
            cp_t[s] += b.dCp * np.exp(-((x_t[s] - b.x_center) / max(b.width, 1e-3)) ** 2)
    elif mode == "Point by point":
        df = st.data_editor(pd.DataFrame({"x": x0, "Cp": cp0, "side": side0}), height=300,
                            width="stretch", disabled=["x", "side"])
        cp_t = df["Cp"].to_numpy()
    else:
        up = st.file_uploader("CSV with columns x, Cp, side (1 upper, −1 lower)", type=["csv", "txt"])
        if up is None:
            st.stop()
        d = np.genfromtxt(io.BytesIO(up.getvalue()), delimiter=",", names=True)
        x_t, cp_t, side_t = d["x"], d["Cp"], d["side"].astype(int)
    go_ = st.button("Design the airfoil", type="primary", width="stretch")

with target:
    f = go.Figure()
    for sd, nm in ((1, "upper"), (-1, "lower")):
        s = side0 == sd
        o = np.argsort(x0[s])
        f.add_scatter(x=x0[s][o], y=cp0[s][o], mode="lines", line=dict(dash="dot", width=1, color=ui.GRAPHITE),
                      name=f"base, {nm}")
        s = side_t == sd
        o = np.argsort(x_t[s])
        f.add_scatter(x=x_t[s][o], y=cp_t[s][o], mode="lines", name=f"target, {nm}",
                      line=dict(color=ui.SERIES[0 if sd == 1 else 1]))
    f.update_layout(yaxis=dict(autorange="reversed", title="Cp"), xaxis_title="x/c", height=420,
                    title=f"Target pressure; base {bname} at α {alpha:g}° has inviscid Cl {cl0:.3f}")
    ui.plot(f)

if go_:
    s0 = cst.fit(C.raw_outline(start), 8)[:2]
    with st.spinner("Matching the pressure distribution..."):
        r = DI.design(x_t, cp_t, side_t, alpha, s0, t_min=t_min)
    st.session_state["inv_result"] = (r, s0, bname, alpha, x_t, cp_t, side_t)

if "inv_result" in st.session_state:
    r, s0, bname_, alpha_, xt_, cpt_, sdt_ = st.session_state["inv_result"]
    st.markdown("### Designed airfoil")
    k = st.columns(4)
    k[0].metric("Cp match, rms", f"{r['rms']:.4f}")
    k[1].metric("Inviscid Cl", f"{r['Cl']:.3f}")
    k[2].metric(f"Thickness, at x/c {r['x_thickness']:.2f}", f"{100 * r['thickness']:.2f} % c")
    k[3].metric("Panel solutions", r["evaluations"])
    c1, c2 = st.columns(2)
    f = go.Figure()
    for sd, nm in ((1, "upper"), (-1, "lower")):
        s = sdt_ == sd
        o = np.argsort(xt_[s])
        f.add_scatter(x=xt_[s][o], y=cpt_[s][o], mode="markers", marker=dict(size=4, color=ui.GRAPHITE),
                      name=f"target, {nm}")
        s = r["side"] == sd
        o = np.argsort(r["x"][s])
        f.add_scatter(x=r["x"][s][o], y=r["cp"][s][o], mode="lines", name=f"designed, {nm}",
                      line=dict(color=ui.SERIES[0 if sd == 1 else 1]))
    f.update_layout(yaxis=dict(autorange="reversed", title="Cp"), xaxis_title="x/c", height=360,
                    title="Pressure: target and designed")
    ui.plot(f, container=c1)
    f = go.Figure()
    o = cst.outline(r["au"], r["al"])
    f.add_scatter(x=o[:, 0], y=o[:, 1], mode="lines", name="designed", fill="toself",
                  fillcolor="rgba(29,78,137,0.06)")
    o0 = cst.outline(*s0)
    f.add_scatter(x=o0[:, 0], y=o0[:, 1], mode="lines", line=dict(dash="dot", color=ui.GRAPHITE), name="start")
    f.update_layout(yaxis=dict(scaleanchor="x", zeroline=False), xaxis=dict(zeroline=False), height=360,
                    title="Shape")
    ui.plot(f, container=c2)
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2], vertical_alignment="bottom")
    name = c1.text_input("Name of the new airfoil", "my_inverse_design")
    dat = "\n".join([name] + [f"{x:.7f} {y:.7f}" for x, y in cst.outline(r["au"], r["al"], 160)]) + "\n"
    c2.download_button("Download .dat", dat, file_name=f"{name}.dat", width="stretch")
    re = c3.number_input("Reynolds number for the CFD check", 1e4, 5e8, 3e6, format="%.3e")
    angs = c4.text_input("Angles (deg)", "-2:12:2")
    if st.button("Run viscous polar of the design"):
        d = C.CASES / "_designs"
        d.mkdir(exist_ok=True)
        p = d / f"{name}.dat"
        p.write_text(dat)
        out = C.CASES / f"{name}_Re{re:.2g}"
        jid = C.start_job("polar", f"{name} (inverse design), Re {C.fmt_re(re)}",
                          ["airfoil.py", "polar", p, "--re", f"{re:.6g}", f"--alpha={angs}",
                           "--out", out], out, {"n_angles": C.count_angles(angs)})
        st.session_state["inv_job"] = jid
    if st.session_state.get("inv_job"):
        views.live_job(st.session_state["inv_job"], "polar")
