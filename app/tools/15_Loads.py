import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import aircraft_views as AV
import common as C
import ui
import views

C.page_setup()
ui.header("Flight loads", "V–n diagram with manoeuvre and gust load factors, trimmed load cases, and "
                          "the spanwise shear, bending and torsion for the structures team, with "
                          "inertia relief and an FEM export.")

setup, info = st.columns([5, 6], gap="large")
with setup:
    with st.container(border=True):
        ui.step("Aircraft", first=True)
        ac, path = AV.aircraft_picker(key="ldac")
        ui.step("Certification basis (simplified CS-23)")
        c1, c2 = st.columns(2)
        cat = c1.selectbox("Category", ["normal", "utility", "aerobatic"])
        alt = c2.number_input("Altitude (m)", 0.0, 15000.0, 0.0, 100.0)
        c1, c2 = st.columns(2)
        vc = c1.number_input("Design cruise speed V_C, EAS (m/s)", 10.0, 300.0, 60.0)
        vd_on = c2.checkbox("Set V_D (otherwise the minimum)")
        vd = c2.number_input("V_D, EAS (m/s)", 10.0, 400.0, 85.0, disabled=not vd_on)
        ui.step("Structure")
        surf_names = [s.name for s in ac.surfaces]
        surf = st.selectbox("Surface", range(len(surf_names)), format_func=lambda i: surf_names[i])
        c1, c2 = st.columns(2)
        axis = c1.number_input("Structural axis (fraction of chord)", 0.1, 0.7, 0.35, 0.01)
        wmf = c2.number_input("Wing structure (fraction of mass)", 0.0, 0.4, 0.10, 0.01)
        c1, c2, c3 = st.columns(3)
        fuel = c1.number_input("Fuel in the wing (kg)", 0.0, 1e5, 0.0)
        f0 = c2.number_input("Fuel from η", 0.0, 1.0, 0.10, 0.05)
        f1 = c3.number_input("Fuel to η", 0.0, 1.0, 0.50, 0.05)
        has_struct = bool(getattr(ac, "structure", None))
        flex = st.checkbox("Flexible wing (elastic twist and bending)", value=False, disabled=not has_struct or surf != 0,
                           help="Uses the wing structure saved with the aircraft on the Wing structure page; every "
                                "case is trimmed again with the deformed wing until it converges"
                                + ("" if has_struct else ". No structure saved with this aircraft yet."))
        c1, c2 = st.columns(2, vertical_alignment="center")
        go_ = c1.button("Compute loads", type="primary", width="stretch")
        c2.caption("About half a minute.")
with info:
    AV.top_view(ac, height=300)
    st.markdown(
        "Load cases: the corners of the manoeuvre envelope (A, C, D, E, F, G), gust load factors "
        "at V_C and V_D, and 1 g cruise. Each case is trimmed with the elevator, so the wing "
        "carries the tail's balancing load. Linear lift is used, which is conservative for the tip "
        "loads near stall.\n\n"
        "Exports: envelope.csv, one CSV per case, strip_loads.csv (forces and moments at the "
        "structural axis) and **loads_nastran.bdf** (GRID, FORCE and MOMENT cards, one load set per "
        "case) for the FEM model.")

if go_:
    out = C.CASES / f"loads_{path.stem}_{int(time.time())}"
    args = ["loads.py", path, "--vc", vc, "--category", cat, "--alt", alt, "--surface", surf,
            "--axis", axis, "--wing-mass", wmf, "--fuel", fuel, "--fuel-span", f"{f0},{f1}",
            "--polar-dir", C.POLAR_CACHE, "--out", out]
    if vd_on:
        args += ["--vd", vd]
    if flex:
        args += ["--flexible"]
    st.session_state["ld_job"] = C.start_job("loads", f"Loads {ac.name}, {cat}", args, out)


def show(out):
    out = Path(out)
    f = out / "loads.json"
    if not f.exists():
        return
    res = json.loads(f.read_text())
    vn = res["vn"]
    k = st.columns(6)
    k[0].metric("n+ / n−", f"{vn['n_pos']:.2f} / {vn['n_neg']:.2f}")
    k[1].metric("V_S1, EAS", f"{vn['Vs1']:.1f} m/s")
    k[2].metric("V_A", f"{vn['Va']:.1f} m/s")
    k[3].metric("V_C", f"{vn['Vc']:.1f} m/s")
    k[4].metric("V_D", f"{vn['Vd']:.1f} m/s")
    k[5].metric("CL max (untrimmed)", f"{vn['CLmax']:.2f}")
    views.trust_notes("loads")
    if res.get("flexible"):
        tw = [(c["name"], c.get("elastic_tip_twist_deg")) for c in res["cases"] if c.get("elastic_tip_twist_deg") is not None]
        st.caption("Flexible wing: every case trimmed with the elastic twist and bending of the saved wing structure. "
                   "Elastic tip twist: " + ", ".join(f"{n} {t:+.2f}°" for n, t in tw[:6]) + (" …" if len(tw) > 6 else ""))
    c1, c2 = st.columns([6, 5])
    cv = vn["curves"]
    V = np.array(cv["V"])
    fig = go.Figure()
    fig.add_scatter(x=V, y=cv["stall_pos"], name="manoeuvre", line=dict(color=ui.BLUE))
    fig.add_scatter(x=V, y=cv["neg_limit"], showlegend=False, line=dict(color=ui.BLUE))
    fig.add_scatter(x=[vn["Va"], vn["Vd"], vn["Vd"]], y=[vn["n_pos"], vn["n_pos"], cv["neg_limit"][-1]],
                    showlegend=False, mode="lines", line=dict(color=ui.BLUE))
    for key, dash, lab in (("gust_vc_pos", "dash", "gust 15.24 m/s"), ("gust_vc_neg", "dash", None),
                           ("gust_vd_pos", "dot", "gust 7.62 m/s"), ("gust_vd_neg", "dot", None)):
        m = V <= (vn["Vc"] if "vc" in key else vn["Vd"])
        fig.add_scatter(x=V[m], y=np.array(cv[key])[m], name=lab, showlegend=lab is not None,
                        line=dict(color=ui.SERIES[1], dash=dash, width=1))
    fig.add_scatter(x=[c["V_eas"] for c in res["cases"]], y=[c["n"] for c in res["cases"]], mode="markers+text",
                    text=[c["name"].split(",")[0] for c in res["cases"]], textposition="top right",
                    textfont=dict(size=10), marker=dict(color=ui.INK, size=7, symbol="square"), name="load cases")
    fig.update_layout(title=f"V–n diagram, {vn['category']} category", xaxis_title="equivalent airspeed (m/s)",
                      yaxis_title="load factor n", height=420)
    ui.plot(fig, container=c1, key=f"vn{out}")
    rows = [{"case": c["name"], "V_EAS": c["V_eas"], "n": c["n"], "α (deg)": c["alpha"],
             "elevator (deg)": c["elevator"], "CL/CLmax": c["CL_over_CLmax"],
             "root shear (kN)": c["root_shear"] / 1e3, "root bending (kN m)": c["root_bending"] / 1e3,
             "root torsion (kN m)": c["root_torsion"] / 1e3} for c in res["cases"]]
    df = pd.DataFrame(rows)
    c2.dataframe(df, hide_index=True, width="stretch", height=420,
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in df.columns if c != "case"})
    names = [c["name"] for c in res["cases"]]
    pick = st.multiselect("Cases on the span plots", names, default=[n for n in names if n[0] in "ACDGF"]
                          or names[:4], key=f"lp{out}")
    cols = st.columns(3)
    env = res["envelope"]
    for col, key, lab in zip(cols, ("shear", "bending", "torsion"),
                             ("Shear (kN)", "Bending, tip up + (kN m)", "Torsion, nose up + (kN m)")):
        fig = go.Figure()
        fig.add_scatter(x=env["eta"], y=np.array(env[f"{key}_max"]) / 1e3, name="envelope",
                        line=dict(color=ui.GRAPHITE, width=1, dash="dot"))
        fig.add_scatter(x=env["eta"], y=np.array(env[f"{key}_min"]) / 1e3, showlegend=False,
                        line=dict(color=ui.GRAPHITE, width=1, dash="dot"))
        for c in res["cases"]:
            if c["name"] in pick:
                fig.add_scatter(x=c["stations"]["eta"], y=np.array(c["stations"][key]) / 1e3,
                                name=c["name"].split(",")[0])
        fig.update_layout(title=lab, xaxis_title="η = 2y/b", height=360)
        ui.plot(fig, container=col, key=f"sp{key}{out}")
    st.caption(f"{res['surface']}, right half; structural axis at {100 * res['x_axis']:.0f} % chord; "
               f"inertia relief with {100 * res['wing_mass_frac']:.0f} % of the mass as wing structure"
               + (f" and {res['fuel_mass']:g} kg fuel" if res["fuel_mass"] else "") + ". Limit loads: "
               "multiply by 1.5 for ultimate.")
    views.downloads(out, ("envelope.csv", "strip_loads.csv", "loads_nastran.bdf", "loads.json"), "ld")


jid = st.session_state.get("ld_job")
if jid:
    st.markdown("### Run")
    views.live_job(jid, "loads")
    if C.get_job(jid)["status"] == "finished":
        show(C.get_job(jid)["out"])
olds = sorted(C.CASES.glob("loads_*/loads.json"), key=lambda p: -p.stat().st_mtime)
if olds:
    st.markdown("### Earlier load sets")
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    pk = c1.selectbox("Result", olds, format_func=lambda p: p.parent.name, key="ldold")
    if c2.button("Open", key="ldopen", width="stretch"):
        st.session_state["ld_open"] = str(pk.parent)
    if st.session_state.get("ld_open"):
        show(st.session_state["ld_open"])

views.workflow_footer("tools/15_Loads.py")
