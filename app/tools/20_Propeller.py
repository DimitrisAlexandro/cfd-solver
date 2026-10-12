import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import aircraft_views as AV
import common as C
import ui

import aircraft as A
import propeller as P
import wing as W

C.page_setup()
ui.header("Propeller and engine", "Minimum-induced-loss propeller design (Adkins & Liebeck), blade element "
                                  "momentum analysis with tip losses, fixed-pitch or constant-speed operation "
                                  "with a piston, turboprop or electric engine. Saved with the aircraft, the "
                                  "Performance page uses it.")

top = st.columns([3, 2], vertical_alignment="bottom")
with top[0]:
    ac, path = AV.aircraft_picker(key="prac")
spec0 = dict(ac.propulsion or {})
eng0 = dict(spec0.get("engine") or {})
des0 = dict((spec0.get("propeller") or {}).get("design") or {})
top[1].caption("Starting from the propulsion saved with this aircraft." if spec0 else
               "This aircraft has no propulsion yet; defaults for a light aircraft.")

setup, out = st.columns([5, 7], gap="large")
with setup:
    with st.container(border=True):
        ui.step("Engine", first=True)
        c1, c2, c3 = st.columns(3)
        kinds = ["piston", "turboprop", "electric"]
        kind = c1.selectbox("Type", kinds, index=kinds.index(eng0.get("kind", "piston")))
        pk = c2.number_input("Power, sea level (kW)", 1.0, 5000.0, float(eng0.get("power_kw", 75.0)))
        rpm = c3.number_input("Propeller rpm, max", 500.0, 8000.0, float(eng0.get("rpm_max", 2700.0)), 50.0)
        c1, c2, c3 = st.columns(3)
        if kind == "electric":
            bkwh = c1.number_input("Battery (kWh)", 0.0, 2000.0, float(eng0.get("battery_kwh", 20.0)))
            usable = c2.number_input("Usable fraction", 0.5, 1.0, float(eng0.get("usable_fraction", 0.8)), 0.05)
            etam = c3.number_input("Motor + controller η", 0.7, 0.99, float(eng0.get("eta_motor", 0.92)), 0.01)
            sfc = 0.0
        else:
            sfc = c1.number_input("SFC (g/kWh)", 150.0, 600.0, float(eng0.get("sfc", 300.0)), 5.0)
            bkwh, usable, etam = 0.0, 0.8, 0.92
        ui.step("Propeller design point")
        c1, c2, c3 = st.columns(3)
        vd = c1.number_input("Speed (m/s)", 5.0, 200.0, float(des0.get("speed", 55.0)))
        hd = c2.number_input("Altitude (m)", 0.0, 10000.0, float(des0.get("altitude", 1500.0)), 100.0)
        rpd = c3.number_input("rpm", 500.0, 8000.0, float(des0.get("rpm", 0.92 * rpm)), 50.0)
        c1, c2, c3 = st.columns(3)
        D = c1.number_input("Diameter (m)", 0.2, 6.0, float(des0.get("diameter", 1.8)), 0.02)
        B = c2.number_input("Blades", 2, 8, int(des0.get("blades", 2)))
        pdes = c3.number_input("Shaft power (kW)", 0.5, 5000.0, float(des0.get("power_kw", 0.75 * pk)))
        c1, c2, c3 = st.columns(3)
        cld = c1.number_input("Design section cl", 0.3, 1.1, float(des0.get("cl", 0.7)), 0.05,
                              help="0.6–0.8: best profile L/D of a Clark-Y-like section")
        mode = c2.selectbox("Operation", ["fixed_pitch", "constant_speed"],
                            index=["fixed_pitch", "constant_speed"].index(spec0.get("mode", "fixed_pitch")),
                            format_func=lambda m: m.replace("_", " "))
        secs = ["generic"] + [a for a in ("naca4412", "naca2412", "naca0012") if
                              __import__("design_wing_sizing").cached_polars(a, C.POLAR_CACHE, 0.0)]
        sec = c3.selectbox("Section", secs, help="generic: Clark-Y-like analytic polar; others: CFD polars "
                                                  "from the polar library")

spec = {"engine": {"kind": kind, "power_kw": pk, "rpm_max": rpm, "sfc": sfc or 300.0, "battery_kwh": bkwh,
                   "usable_fraction": usable, "eta_motor": etam},
        "mode": mode, "rpm": rpm, "section": sec,
        "propeller": {"design": {"speed": vd, "rpm": rpd, "diameter": D, "blades": int(B), "power_kw": pdes,
                                 "altitude": hd, "cl": cld}}}


@st.cache_data(show_spinner=False)
def design(spec_json):
    s = json.loads(spec_json)
    d = s["propeller"]["design"]
    rho, mu, a, _, _ = W.isa(d["altitude"])
    section = P.section_from_spec(s.get("section"), C.POLAR_CACHE)
    prop, info = P.design(d["speed"], d["rpm"], d["diameter"], d["blades"], power=d["power_kw"] * 1e3,
                          cl_design=d["cl"], section=section, rho=rho, mu=mu, a_snd=a)
    chk = P.analyse(prop, d["speed"], d["rpm"], rho, mu, a)
    n = d["rpm"] / 60
    J = np.linspace(0.02, 1.35 * info["J"], 28)
    sweep = [P.analyse(prop, j * n * d["diameter"], d["rpm"], rho, mu, a) for j in J]
    maps = {}
    for db in (-6.0, -3.0, 0.0, 3.0, 6.0):
        maps[db] = [P.analyse(prop, j * n * d["diameter"], d["rpm"], rho, mu, a, dbeta=db)
                    for j in np.linspace(0.02, 1.8 * info["J"], 30)]
    return A._jsonable({"prop": prop.to_dict(), "info": info, "check": chk, "sweep": sweep, "maps": maps,
                        "af": prop.activity_factor(), "beta75": prop.beta_75()})


try:
    with st.spinner("Designing the blade"):
        dres = design(json.dumps(spec, sort_keys=True))
    err = None
except Exception as e:                            # noqa: BLE001
    dres, err = None, str(e)

with out:
    if err:
        st.error(f"Design failed: {err}")
    else:
        inf, chk = dres["info"], dres["check"]
        k = st.columns(4)
        k[0].metric("Efficiency at design", f"{inf['eta']:.3f}", f"ideal (disc) {inf['eta_ideal']:.3f}",
                    delta_color="off", delta_arrow="off")
        k[1].metric("Thrust at design", f"{inf['T']:.0f} N", f"BEMT check {chk['T']:.0f} N", delta_color="off", delta_arrow="off")
        k[2].metric("Advance ratio J", f"{inf['J']:.3f}", f"tip Mach {chk['tip_mach']:.2f}", delta_color="off", delta_arrow="off")
        k[3].metric("β at 0.75 R", f"{dres['beta75']:.1f}°", f"activity factor {dres['af']:.0f}", delta_color="off", delta_arrow="off")
        if chk["tip_mach"] > 0.88:
            st.warning("Helical tip Mach number above 0.88: compressibility losses and noise. Reduce rpm or "
                       "diameter.")
        g = dres["prop"]
        x = np.array(g["r"]) / (0.5 * g["diameter"])
        c1, c2 = st.columns(2)
        f = go.Figure()
        f.add_scatter(x=x, y=np.array(g["chord"]) / g["diameter"], name="c / D")
        f.add_scatter(x=x, y=g["beta"], name="β (deg)", yaxis="y2")
        f.update_layout(title="Blade geometry", xaxis_title="r / R", yaxis_title="c / D", height=320,
                        yaxis2=dict(title="β (deg)", overlaying="y", side="right", tickformat=".0f"))
        ui.plot(f, container=c1, key="prgeo")
        sw = dres["sweep"]
        f = go.Figure()
        f.add_scatter(x=[p["J"] for p in sw], y=[p["CT"] for p in sw], name="C_T")
        f.add_scatter(x=[p["J"] for p in sw], y=[p["CP"] for p in sw], name="C_P")
        f.add_scatter(x=[p["J"] for p in sw], y=[p["eta"] for p in sw], name="η", yaxis="y2")
        f.update_layout(title=f"Performance at {rpd:g} rpm", xaxis_title="J = V/(nD)", yaxis_title="C_T, C_P",
                        height=320, yaxis2=dict(title="η", overlaying="y", side="right", range=[0, 1], dtick=0.2, tickformat=".1f"))
        ui.plot(f, container=c2, key="prmap")
        st.caption(f"Design: {B} blades, Adkins & Liebeck minimum induced loss at cl {cld:g}; re-analysed with "
                   f"BEMT (Prandtl tip and hub loss): P {chk['P'] / 1e3:.2f} kW vs {inf['P'] / 1e3:.2f} kW design.")

if dres:
    tabs = st.tabs(["Efficiency map (pitch settings)", "Installed thrust", "Radial loading", "Save"])
    with tabs[0]:
        f = go.Figure()
        for i, (db, pts) in enumerate(dres["maps"].items()):
            pts = [p for p in pts if p["eta"] > 0 and p["T"] > 0]
            f.add_scatter(x=[p["J"] for p in pts], y=[p["eta"] for p in pts], name=f"Δβ {float(db):+g}°",
                          line=dict(color=ui.SERIES[i]))
        f.update_layout(title="Efficiency against advance ratio, pitch change of the whole blade",
                        xaxis_title="J", yaxis_title="η", yaxis_range=[0, 1], height=380)
        ui.plot(f, key="prpitch")
        st.caption("A constant-speed propeller moves along the envelope of these curves; a fixed-pitch one "
                   "stays on the Δβ 0 curve.")
    with tabs[1]:
        c1, c2 = st.columns([1, 3])
        alts = c1.text_input("Altitudes (m)", "0, 1500, 3000")
        vmax = c1.number_input("Up to speed (m/s)", 10.0, 250.0, float(max(1.6 * vd, 60.0)))
        go_t = c1.button("Compute thrust curves", type="primary")
        if go_t:
            bp = P.make_propulsion(spec, C.POLAR_CACHE)
            hs = [float(a) for a in alts.split(",") if a.strip()]
            Vs = np.linspace(0, vmax, 30)
            with st.spinner("Operating points"):
                curves = {h_: [bp.operating_point(v, h_) for v in Vs] for h_ in hs}
            st.session_state["pr_curves"] = (Vs.tolist(), A._jsonable(curves), json.dumps(spec, sort_keys=True))
        got = st.session_state.get("pr_curves")
        if got and got[2] == json.dumps(spec, sort_keys=True):
            Vs, curves, _ = got
            f = go.Figure()
            f2 = go.Figure()
            for i, (h_, pts) in enumerate(curves.items()):
                f.add_scatter(x=Vs, y=[p["T"] for p in pts], name=f"{float(h_):g} m", line=dict(color=ui.SERIES[i]))
                f2.add_scatter(x=Vs, y=[p["eta"] for p in pts], name=f"{float(h_):g} m", line=dict(color=ui.SERIES[i]))
            f.update_layout(title="Full-throttle thrust", xaxis_title="V (m/s)", yaxis_title="N", height=340)
            f2.update_layout(title="Propeller efficiency, full throttle", xaxis_title="V (m/s)", yaxis_title="η",
                             yaxis_range=[0, 1], height=340)
            ui.plot(f, container=c2, key="prT")
            ui.plot(f2, container=c2, key="prE")
            p0 = list(curves.values())[0]
            c1.caption(f"Static thrust {p0[0]['T']:.0f} N ({p0[0]['T'] / pk:.1f} N/kW); "
                       + ("rpm at full throttle where the absorbed power equals the engine's (fixed pitch)."
                          if mode == "fixed_pitch" else "pitch set so the propeller absorbs full power at max rpm."))
    with tabs[2]:
        st_ = dres["check"]["stations"]
        x = np.array(st_["r"]) / (0.5 * D)
        c1, c2 = st.columns(2)
        f = go.Figure()
        f.add_scatter(x=x, y=st_["dT_dr"], name="dT/dr (N/m)")
        f.update_layout(title="Thrust loading at the design point", xaxis_title="r / R", height=300)
        ui.plot(f, container=c1, key="prdT")
        f = go.Figure()
        f.add_scatter(x=x, y=st_["alpha"], name="α")
        f.add_scatter(x=x, y=st_["phi"], name="φ")
        f.update_layout(title="Section angle of attack and inflow angle (deg)", xaxis_title="r / R", height=300)
        ui.plot(f, container=c2, key="pral")
    with tabs[3]:
        st.markdown(f"Saves engine, propeller design and operating mode with **{ac.name}**. The Performance page "
                    "can then use this model instead of its simple propeller.")
        c1, c2 = st.columns(2)
        if c1.button("Save with the aircraft", type="primary", width="stretch"):
            a2 = A.Aircraft.from_dict(ac.to_dict())
            a2.propulsion = spec
            a2.save(path)
            st.success(f"Saved to {path.name}.")
        c2.download_button("Download blade geometry (JSON)", json.dumps(dres["prop"], indent=1),
                           file_name="propeller.json", width="stretch")
        st.dataframe(pd.DataFrame({"r (m)": dres["prop"]["r"], "chord (m)": dres["prop"]["chord"],
                                   "β (deg)": dres["prop"]["beta"]}), hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="%.4f") for k in ("r (m)", "chord (m)", "β (deg)")})
