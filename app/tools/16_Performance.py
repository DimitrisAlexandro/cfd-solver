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

import aircraft as A

C.page_setup()
ui.header("Performance", "Speeds, climb, ceilings, range, endurance, glide, takeoff and landing from "
                         "the trimmed drag polar of the aircraft model and a piston-propeller model.")

setup, info = st.columns([5, 6], gap="large")
with setup:
    with st.container(border=True):
        ui.step("Aircraft", first=True)
        ac, path = AV.aircraft_picker(key="pfac")
        c1, c2 = st.columns(2)
        mass = c1.number_input("Mass (kg)", 0.1, 1e6, float(ac.mass))
        alt = c2.number_input("Altitude (m)", 0.0, 12000.0, 0.0, 100.0)
        ui.step("Engine and propeller")
        has_bemt = bool(ac.propulsion)
        pmodel = st.segmented_control(
            "Propulsion model", ["Saved with the aircraft (BEMT)", "Simple fixed-pitch model"],
            default="Saved with the aircraft (BEMT)" if has_bemt else "Simple fixed-pitch model",
            help="BEMT: the engine and propeller saved on the Propeller page or by Conceptual sizing")
        use_bemt = pmodel == "Saved with the aircraft (BEMT)"
        if use_bemt and not has_bemt:
            st.warning("This aircraft has no saved propulsion: design one on the Propeller page.")
        if use_bemt and has_bemt:
            e_ = ac.propulsion.get("engine", {})
            d_ = ac.propulsion.get("propeller", {}).get("design", {})
            st.caption(f"{e_.get('kind', 'piston')} {e_.get('power_kw', 0):.0f} kW, "
                       f"{ac.propulsion.get('mode', 'fixed_pitch').replace('_', ' ')} propeller "
                       f"⌀ {d_.get('diameter', 0):.2f} m, {d_.get('blades', 2)} blades. The fields below "
                       "except fuel and runway are not used.")
        c1, c2, c3 = st.columns(3)
        power = c1.number_input("Power, sea level (kW)", 1.0, 5000.0, 75.0)
        eta = c2.number_input("Propeller efficiency, max", 0.3, 0.95, 0.80, 0.01)
        vdes = c3.number_input("Propeller design speed (m/s)", 10.0, 200.0, 60.0,
                               help="Fixed pitch: efficiency peaks here; thrust falls linearly from rest to this speed")
        c1, c2 = st.columns(2)
        dprop = c1.number_input("Propeller diameter (m)", 0.2, 6.0, 1.8, 0.05)
        bsfc = c2.number_input("BSFC (g/kWh)", 150.0, 600.0, 300.0, 5.0)
        ui.step("Fuel and runway")
        c1, c2 = st.columns(2)
        fuel = c1.number_input("Usable fuel (kg)", 0.0, 1e5, 80.0)
        reserve = c2.number_input("Reserve (kg)", 0.0, 1e5, 0.0)
        c1, c2 = st.columns(2)
        to_flap = c1.number_input("Takeoff flap (deg)", 0.0, 45.0, 10.0)
        ld_flap = c2.number_input("Landing flap (deg)", 0.0, 45.0, 30.0)
        surface = st.segmented_control("Runway", ["paved", "grass"], default="paved") or "paved"
        c1, c2 = st.columns(2, vertical_alignment="center")
        go_ = c1.button("Compute performance", type="primary", width="stretch")
        c2.caption("About a minute (two with the BEMT propeller).")
with info:
    AV.view3d(ac, height=300)
    st.markdown(
        "The airframe drag is the aircraft model's: induced and profile drag of every surface, fuselage "
        "friction and the extra drag on the Aircraft model page (landing gear, cooling, excrescences). "
        "Range is Breguet's at constant speed and altitude; the flap CL_max increments for takeoff and "
        "landing are Raymer's plain-flap estimates.")

if go_:
    out = C.CASES / f"perf_{path.stem}_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    used = A.Aircraft.from_dict(ac.to_dict())
    used.mass = mass
    used.save(out / "aircraft_used.json")
    mu = (0.04, 0.40) if surface == "paved" else (0.08, 0.20)
    args = ["performance.py", out / "aircraft_used.json", "--power", power, "--eta", eta, "--v-design", vdes,
            "--prop-diameter", dprop, "--bsfc", bsfc, "--fuel", fuel, "--reserve", reserve, "--alt", alt,
            "--to-flap", to_flap, "--ld-flap", ld_flap, "--mu-roll", mu[0], "--mu-brake", mu[1],
            "--polar-dir", C.POLAR_CACHE, "--out", out]
    if use_bemt and ac.propulsion:
        (out / "propulsion.json").write_text(json.dumps(ac.propulsion, indent=1))
        args += ["--propulsion", out / "propulsion.json"]
    st.session_state["pf_job"] = C.start_job("performance", f"Performance {ac.name}, {mass:g} kg", args, out)


def show(out):
    out = Path(out)
    f = out / "performance.json"
    if not f.exists():
        return
    r = json.loads(f.read_text())
    k = st.columns(6)
    k[0].metric("Stall speed, clean", f"{r['V_stall']:.1f} m/s")
    k[1].metric("Max. level speed", f"{r['V_max']:.1f} m/s")
    k[2].metric(f"Best climb, at {r['best_rate']['V']:.0f} m/s", f"{r['best_rate']['roc']:.2f} m/s")
    k[3].metric(f"Best angle, at {r['best_angle']['V']:.0f} m/s", f"{r['best_angle']['gamma_deg']:.1f}°")
    sc = r.get("ceiling_service")
    k[4].metric("Service ceiling", f"{sc:.0f} m" if sc is not None else "above 9000 m")
    k[5].metric("CL max, trimmed", f"{r['CLmax_trimmed']:.2f}")
    k = st.columns(6)
    k[0].metric(f"Range, at {r['best_range']['V']:.0f} m/s", f"{r['best_range']['range_km']:.0f} km")
    k[1].metric(f"Endurance, at {r['best_endurance']['V']:.0f} m/s", f"{r['best_endurance']['endurance_h']:.1f} h")
    k[2].metric("Best glide ratio", f"{r['glide']['best_ratio']:.1f}")
    k[3].metric(f"Min. sink, at {r['glide']['V_min_sink']:.0f} m/s", f"{r['glide']['min_sink']:.2f} m/s")
    TO, LD = r["takeoff"], r["landing"]
    k[4].metric("Takeoff to 15 m", f"{TO['total']:.0f} m" if TO.get("total") else "not possible")
    k[5].metric("Landing from 15 m", f"{LD['total']:.0f} m")
    c = r["curves"]
    V = np.array(c["V"])
    pr = r.get("propulsion", {})
    if pr.get("model") == "BEMT":
        e_ = pr["engine"]
        st.caption(f"Propulsion: {e_['kind']} {e_['power_kw']:.0f} kW, BEMT propeller "
                   f"({pr['mode'].replace('_', ' ')}, ⌀ {pr['propeller']['diameter']:.2f} m, "
                   f"{pr['propeller']['blades']} blades, β 0.75R {pr['beta_75']:.1f}°); range and endurance "
                   "use the propeller efficiency at the thrust needed for level flight"
                   + (f"; battery {e_['battery_kwh']:.1f} kWh." if e_["kind"] == "electric" else "."))
    tabs = st.tabs(["Power and climb", "Range and endurance", "Drag polar", "Takeoff and landing", "Files"])
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        fg = go.Figure()
        fg.add_scatter(x=V, y=np.array(c["P_req"]) / 1e3, name="required, D·V")
        fg.add_scatter(x=V, y=np.array(c["P_av"]) / 1e3, name="available, η·P")
        fg.update_layout(title="Thrust power", xaxis_title="V (m/s)", yaxis_title="kW", height=340)
        ui.plot(fg, container=c1, key=f"p1{out}")
        fg = go.Figure()
        fg.add_scatter(x=V, y=c["roc"], name="rate of climb")
        fg.update_layout(title="Rate of climb", xaxis_title="V (m/s)", yaxis_title="m/s", height=340, showlegend=False)
        ui.plot(fg, container=c2, key=f"p2{out}")
        fg = go.Figure()
        fg.add_scatter(x=c["ceiling_roc"], y=c["ceiling_h"], mode="lines+markers")
        fg.add_vline(x=0.508, line=dict(color=ui.GRAPHITE, dash="dot", width=1),
                     annotation_text="service ceiling", annotation_font_size=10)
        fg.update_layout(title="Best rate of climb against altitude", xaxis_title="m/s", yaxis_title="altitude (m)",
                         height=340, showlegend=False)
        ui.plot(fg, container=c3, key=f"p3{out}")
    with tabs[1]:
        c1, c2, c3 = st.columns(3)
        for col, key, lab in ((c1, "range_km", "Range (km)"), (c2, "endurance_h", "Endurance (h)"),
                              (c3, "sink", "Sink rate, power off (m/s)")):
            fg = go.Figure()
            fg.add_scatter(x=V, y=c[key])
            fg.update_layout(title=lab, xaxis_title="V (m/s)", height=330, showlegend=False)
            ui.plot(fg, container=col, key=f"r{key}{out}")
        st.caption(f"Constant speed and altitude, {r['fuel']:g} kg fuel; propeller efficiency varies with speed.")
    with tabs[2]:
        pol = pd.DataFrame(r["polar"])
        c1, c2 = st.columns(2)
        fg = go.Figure()
        fg.add_scatter(x=pol.CD, y=pol.CL, mode="lines+markers", name="trimmed")
        pf = r["polar_fit"]
        cl = np.linspace(0, pol.CL.max(), 50)
        fg.add_scatter(x=pf["CD0"] + pf["k1"] * cl + pf["k2"] * cl**2, y=cl, name="parabolic fit",
                       line=dict(dash="dot", color=ui.GRAPHITE))
        fg.update_layout(title=f"Trimmed drag polar: CD0 {pf['CD0']:.4f}, e {pf['e_oswald']:.2f}",
                         xaxis_title="CD", yaxis_title="CL", height=360)
        ui.plot(fg, container=c1, key=f"pol{out}")
        c2.dataframe(pol, hide_index=True, width="stretch", height=360)
    with tabs[3]:
        rows = [{"segment": k.replace("_", " "), "takeoff (m)": TO.get(k)} for k in
                ("ground_roll", "rotation", "transition", "climb", "total")]
        rows2 = [{"segment": k.replace("_", " "), "landing (m)": LD.get(k)} for k in
                 ("approach", "flare", "free_roll", "ground_roll", "total")]
        c1, c2 = st.columns(2)
        c1.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        c1.caption(f"Lift-off at {TO['V_lof']:.1f} m/s; CL max with takeoff flap {TO['CLmax']:.2f} "
                   f"(flap increment {TO['flap_dCLmax_estimate']:.2f}, {TO.get('flap_source', 'estimate')}).")
        c2.dataframe(pd.DataFrame(rows2), hide_index=True, width="stretch")
        c2.caption(f"Approach {LD['V_approach']:.1f} m/s, touchdown {LD['V_touchdown']:.1f} m/s; CL max with "
                   f"landing flap {LD['CLmax']:.2f} (increment {LD['flap_dCLmax_estimate']:.2f}, {LD.get('flap_source', 'estimate')}).")
    with tabs[4]:
        views.downloads(out, ("performance.json", "aircraft_used.json", "propulsion.json"), "pf")
        if (out / "performance.png").exists():
            st.image(str(out / "performance.png"))


jid = st.session_state.get("pf_job")
if jid:
    st.markdown("### Run")
    views.live_job(jid, "performance")
    if C.get_job(jid)["status"] == "finished":
        show(C.get_job(jid)["out"])
olds = sorted(C.CASES.glob("perf_*/performance.json"), key=lambda p: -p.stat().st_mtime)
if olds:
    st.markdown("### Earlier results")
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    pk = c1.selectbox("Result", olds, format_func=lambda p: p.parent.name, key="pfold")
    if c2.button("Open", key="pfopen", width="stretch"):
        st.session_state["pf_open"] = str(pk.parent)
    if st.session_state.get("pf_open"):
        show(st.session_state["pf_open"])
