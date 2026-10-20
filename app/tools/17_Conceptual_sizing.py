import json
import re
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

from aerolab.aircraft import aircraft as A
from aerolab.aircraft import sizing as SZ

G0 = 9.80665
C.page_setup()
ui.header("Conceptual sizing", "Requirements in, aircraft out: constraint diagram, mission fuel or battery, "
                               "Class I and Class II weights, iterated until take-off mass, geometry and drag "
                               "agree. The result is a complete aircraft model for every other page.")

R = SZ.default_requirements()
CATS = list(SZ.CLASS_I)

setup, res_col = st.columns([5, 7], gap="large")
with setup:
    with st.container(border=True):
        ui.step("Mission", first=True)
        name = st.text_input("Name", R["name"])
        c1, c2, c3 = st.columns(3)
        crew = c1.number_input("Crew", 1, 4, R["crew"])
        pax = c2.number_input("Passengers", 0, 8, R["passengers"])
        pm = c3.number_input("Mass per person (kg)", 40.0, 130.0, R["person_mass"])
        c1, c2, c3 = st.columns(3)
        bag = c1.number_input("Baggage (kg)", 0.0, 500.0, R["baggage"])
        rng = c2.number_input("Range (km)", 10.0, 5000.0, R["range_km"], 10.0)
        res_min = c3.number_input("Reserve (min)", 0.0, 120.0, R["reserve_min"], 5.0)
        c1, c2, c3 = st.columns(3)
        vc = c1.number_input("Cruise speed (m/s)", 15.0, 150.0, R["cruise_speed"])
        hc = c2.number_input("Cruise altitude (m)", 0.0, 9000.0, R["cruise_alt"], 100.0)
        pc = c3.number_input("Cruise power (fraction)", 0.3, 1.0, R["cruise_power"], 0.05)

        ui.step("Field, climb and ceiling")
        c1, c2, c3 = st.columns(3)
        vs = c1.number_input("Stall speed, landing flap (m/s)", 10.0, 50.0, R["stall_speed"], 0.5,
                             help="CS-23 single engine: at most 31.4 m/s (61 kt); CS-VLA / LSA: 23.2 m/s (45 kt)")
        s_to = c2.number_input("Take-off over 15 m (m)", 0.0, 3000.0, R["takeoff_distance"], 10.0,
                               help="0 = no requirement")
        s_l = c3.number_input("Landing over 15 m (m)", 0.0, 3000.0, R["landing_distance"], 10.0,
                              help="0 = no requirement")
        c1, c2, c3 = st.columns(3)
        roc = c1.number_input("Rate of climb, sea level (m/s)", 0.5, 20.0, R["climb_rate"], 0.1)
        ceil = c2.number_input("Service ceiling (m)", 0.0, 9000.0, R["ceiling"], 100.0, help="0 = no requirement")
        nz = c3.number_input("Limit load factor", 2.5, 9.0, R["load_factor"], 0.1,
                             help="CS-23 normal 3.8, utility 4.4, aerobatic 6.0")

        ui.step("Configuration")
        cat = st.selectbox("Statistics class", CATS, index=0,
                           help="Class I empty-mass fraction, fuselage length and tail volumes (Raymer)")
        c1, c2, c3 = st.columns(3)
        ar = c1.number_input("Aspect ratio", 4.0, 30.0, R["aspect_ratio"], 0.5)
        tr = c2.number_input("Taper ratio", 0.2, 1.0, R["taper"], 0.05)
        sw = c3.number_input("Sweep, c/4 (deg)", 0.0, 40.0, R["sweep"])
        c1, c2, c3 = st.columns(3)
        airfoil = c1.selectbox("Wing section", C.BUILTIN_AIRFOILS[:6], index=1)
        high = c2.segmented_control("Wing", ["low", "high"], default="low") == "high"
        seats = c3.number_input("Seats abreast", 1, 4, R["seats_abreast"])
        c1, c2, c3 = st.columns(3)
        clm = c1.number_input("CL max, clean", 0.8, 3.0, R["cl_max_clean"], 0.05)
        clto = c2.number_input("CL max, take-off", 0.8, 3.5, R["cl_max_takeoff"], 0.05)
        cll = c3.number_input("CL max, landing", 0.8, 3.5, R["cl_max_landing"], 0.05,
                              help="Plain flap on a NACA 2412: about 1.9–2.0 trimmed; slotted 2.2–2.4")
        c1, c2, c3 = st.columns(3)
        gear = c1.selectbox("Landing gear", list(__import__("aerolab.aircraft.drag_buildup", fromlist=["x"]).GEAR_PRESETS), index=1,
                            format_func=lambda k: k.replace("_", ", ").replace("fixed, ", "fixed: "))
        comp = c2.checkbox("Composite structure", R["composite"],
                           help="0.85 on wing, tail and fuselage mass; smoother surfaces, more laminar flow")
        cgt = c3.number_input("Target c.g. (% MAC)", 10.0, 40.0, R["cg_target"], 1.0)

        ui.step("Propulsion")
        c1, c2, c3 = st.columns(3)
        eng = c1.selectbox("Engine", ["piston", "turboprop", "electric"])
        mode = c2.selectbox("Propeller", ["fixed_pitch", "constant_speed"],
                            format_func=lambda m: m.replace("_", " "))
        rpm = c3.number_input("Propeller rpm (max)", 1000.0, 6000.0, R["rpm"], 50.0)
        c1, c2, c3 = st.columns(3)
        if eng == "electric":
            ebat = c1.number_input("Battery pack (Wh/kg)", 80.0, 500.0, R["battery_wh_kg"], 10.0)
            usable = c2.number_input("Usable fraction", 0.5, 1.0, R["battery_usable"], 0.05)
            etam = c3.number_input("Motor + controller efficiency", 0.7, 0.99, R["eta_motor"], 0.01)
            sfc = R["sfc"]
        else:
            sfc = c1.number_input("SFC (g/kWh)", 150.0, 600.0, 300.0 if eng == "piston" else 330.0, 5.0)
            ebat, usable, etam = R["battery_wh_kg"], R["battery_usable"], R["eta_motor"]
        eta_c = c2.number_input("Propeller η, cruise", 0.5, 0.92, R["eta_prop_cruise"], 0.01)
        eta_cl = c3.number_input("Propeller η, climb", 0.4, 0.9, R["eta_prop_climb"], 0.01)
        with st.expander("Fix the design point (optional)"):
            c1, c2 = st.columns(2)
            ws_fix = c1.number_input("Wing loading (kg/m², 0 = from the constraints)", 0.0, 400.0, 0.0)
            p_fix = c2.number_input("Power (kW, 0 = from the constraints)", 0.0, 5000.0, 0.0)
        go_ = st.button("Size the aircraft", type="primary", width="stretch")

req = {"name": name, "category": cat, "crew": int(crew), "passengers": int(pax), "person_mass": pm, "baggage": bag,
       "range_km": rng, "reserve_min": res_min, "cruise_speed": vc, "cruise_alt": hc, "cruise_power": pc,
       "stall_speed": vs, "takeoff_distance": s_to or None, "landing_distance": s_l or None, "climb_rate": roc,
       "ceiling": ceil or None, "aspect_ratio": ar, "taper": tr, "sweep": sw, "airfoil": airfoil, "high_wing": high,
       "seats_abreast": int(seats), "cl_max_clean": clm, "cl_max_takeoff": clto, "cl_max_landing": cll,
       "gear": gear, "composite": comp, "cg_target": cgt, "engine": eng, "prop_mode": mode, "rpm": rpm, "sfc": sfc,
       "battery_wh_kg": ebat, "battery_usable": usable, "eta_motor": etam, "eta_prop_cruise": eta_c,
       "eta_prop_climb": eta_cl, "load_factor": nz,
       "wing_loading": ws_fix or None, "power_kw": p_fix or None}

if go_:
    try:
        with st.spinner("Sizing"):
            res, ac = SZ.size(req, log=lambda *a: None)
        st.session_state["sz_res"] = A._jsonable(res)
    except Exception as e:                       # noqa: BLE001
        st.session_state.pop("sz_res", None)
        st.error(f"Sizing failed: {e}")

res = st.session_state.get("sz_res")
with res_col:
    if not res:
        st.markdown(
            "Set the requirements on the left and size. The loop takes a few seconds:\n\n"
            "1. **Constraint diagram**: stall and landing fix the largest wing loading; take-off, climb, "
            "cruise speed and ceiling the power needed there.\n"
            "2. **Mission**: fuel by Breguet (or battery energy) for climb, cruise and reserve.\n"
            "3. **Weights**: Raymer's Class II component equations, checked against the Class I statistics.\n"
            "4. **Geometry and drag**: wing from the design point, fuselage and tails from statistics and "
            "volume coefficients, drag polar from the component drag buildup and the vortex lattice.\n\n"
            "Repeated until the take-off mass changes by less than 0.5 kg. The sized aircraft can be "
            "saved and then refined on the other pages.")
    else:
        ac = A.Aircraft.from_dict(res["aircraft"])
        k = st.columns(4)
        k[0].metric("Take-off mass", f"{res['mtow']:.0f} kg")
        k[1].metric("Empty mass", f"{res['empty_mass']:.0f} kg",
                    f"{100 * res['empty_fraction']:.1f} % (Class I {100 * res['class_i_empty_fraction']:.1f} %)",
                    delta_color="off", delta_arrow="off")
        if res["battery"] > 0:
            k[2].metric("Battery", f"{res['battery']:.0f} kg", f"{res['mission']['energy_kWh']:.1f} kWh usable",
                        delta_color="off", delta_arrow="off")
        else:
            k[2].metric("Fuel", f"{res['fuel']:.0f} kg", f"{res['fuel'] / 0.72:.0f} L avgas", delta_color="off", delta_arrow="off")
        k[3].metric("Power", f"{res['power_kw']:.0f} kW", f"{res['power_loading_w_kg']:.0f} W/kg", delta_color="off", delta_arrow="off")
        k = st.columns(4)
        k[0].metric("Wing area", f"{res['wing_area']:.2f} m²", f"{res['wing_loading_kg_m2']:.0f} kg/m²",
                    delta_color="off", delta_arrow="off")
        k[1].metric("Span", f"{res['span']:.2f} m")
        k[2].metric("CD0 / Oswald e", f"{res['CD0']:.4f} / {res['oswald_e']:.2f}")
        k[3].metric("L/D cruise / max", f"{res['L_D_cruise']:.1f} / {res['L_D_max']:.1f}")
        if not res["converged"]:
            st.warning("The loop did not converge in 40 iterations; the last iterate is shown.")
        AV.view3d(ac, height=330)

if res:
    tabs = st.tabs(["Constraint diagram", "Masses", "Mission", "Drag", "Balance and gear",
                    "Iterations", "Save and verify"])
    con, dp = res["constraints"], res["design_point"]
    with tabs[0]:
        ws = np.array(con["ws"]) / G0
        f = go.Figure()
        for i, (kk, v) in enumerate(con["lines"].items()):
            f.add_scatter(x=ws, y=np.array(v) * G0, name=kk, line=dict(color=ui.SERIES[i % len(ui.SERIES)]))
        ymax = 2.6 * dp["pw"] * G0
        for kk, v in con["limits"].items():
            f.add_scatter(x=[v / G0, v / G0], y=[0, ymax], name=kk, mode="lines",
                          line=dict(color=ui.GRAPHITE, dash="dash" if "stall" in kk else "dot", width=1.2))
        # feasible region shading: above every line, left of every limit
        lim = min(con["limits"].values()) / G0
        top = np.max(np.array(list(con["lines"].values())), axis=0) * G0
        m = ws <= lim
        f.add_scatter(x=np.r_[ws[m], ws[m][::-1]], y=np.r_[np.minimum(top[m], ymax), np.full(m.sum(), ymax)],
                      fill="toself", fillcolor="rgba(29,78,137,0.07)", line=dict(width=0), name="feasible",
                      hoverinfo="skip")
        f.add_scatter(x=[dp["ws"] / G0], y=[dp["pw"] * G0], mode="markers", name="design point",
                      marker=dict(symbol="star", size=16, color=ui.INK))
        f.update_layout(title="Power to weight against wing loading", xaxis_title="wing loading W/S (kg/m²)",
                        yaxis_title="power to weight P/W (W/kg)", yaxis_range=[0, ymax], height=460)
        c1, c2 = st.columns([3, 1])
        ui.plot(f, container=c1, key="szcon")
        c2.markdown(f"**Design point**  \n{dp['ws'] / G0:.1f} kg/m², {dp['pw'] * G0:.1f} W/kg")
        c2.markdown(f"Wing loading set by **{dp['ws_limit']}**; power by **{dp['active']}** "
                    f"(+{100 * res['requirements'].get('power_margin', 0.05):.0f} % margin).")
        c2.dataframe(pd.DataFrame([{"constraint": kk, "W/kg": v * G0} for kk, v in dp["pw_lines"].items()]),
                     hide_index=True, width="stretch",
                     column_config={"W/kg": st.column_config.NumberColumn(format="%.1f")})
        st.caption("Take-off and landing lines: Roskam's CS-23 / FAR 23 correlations. Climb, cruise and ceiling: "
                   "power required with the drag polar of the sized geometry, propeller efficiency as entered, "
                   "engine altitude lapse (piston Gagg–Ferrar, turboprop σ^0.7, electric none).")
    with tabs[1]:
        m2 = res["class_ii"]
        rows = [{"group": kk, "mass (kg)": v, "% of MTOW": 100 * v / res["mtow"]} for kk, v in
                sorted(m2.items(), key=lambda t: -t[1])]
        rows += [{"group": "crew and payload", "mass (kg)": res["payload"], "% of MTOW": 100 * res["payload"] / res["mtow"]},
                 {"group": "fuel" if res["fuel"] > 0 else "battery", "mass (kg)": res["fuel"] + res["battery"],
                  "% of MTOW": 100 * (res["fuel"] + res["battery"]) / res["mtow"]}]
        c1, c2 = st.columns([1, 1])
        df = pd.DataFrame(rows)
        f = go.Figure(go.Bar(x=df["mass (kg)"], y=df["group"], orientation="h", marker_color=ui.BLUE))
        f.update_layout(title="Mass breakdown", xaxis_title="kg", height=430, yaxis=dict(autorange="reversed"))
        ui.plot(f, container=c1, key="szmass")
        c2.dataframe(df, hide_index=True, width="stretch", height=430,
                     column_config={"mass (kg)": st.column_config.NumberColumn(format="%.1f"),
                                    "% of MTOW": st.column_config.NumberColumn(format="%.1f")})
        st.caption("Raymer's general-aviation component equations (Aircraft Design, eqs. 15.46–15.58); engine "
                   "dry mass from power; propeller 0.12 kg/kW fixed pitch, 0.18 constant speed. Conceptual "
                   "accuracy: about ±10–15 % on the empty mass.")
    with tabs[2]:
        seg = pd.DataFrame(res["mission"]["segments"])
        st.dataframe(seg, hide_index=True, width="stretch")
        if res["battery"] > 0:
            st.caption(f"Battery {res['battery']:.0f} kg = {res['mission']['energy_kWh']:.1f} kWh needed / "
                       f"({res['requirements']['battery_wh_kg']:.0f} Wh/kg × {res['requirements']['battery_usable']:.2f} usable).")
        else:
            st.caption(f"Fuel {res['fuel']:.1f} kg = 1.06 × (1 − product of the fractions) × MTOW "
                       f"(6 % trapped and unusable). Cruise L/D {res['L_D_cruise']:.1f}.")
    with tabs[3]:
        d = pd.DataFrame(res["drag"])
        c1, c2 = st.columns([1, 1])
        f = go.Figure(go.Bar(x=1e4 * d["CD0"], y=d["component"], orientation="h", marker_color=ui.BLUE))
        f.update_layout(title=f"CD0 = {res['CD0']:.4f}", xaxis_title="drag counts", height=380,
                        yaxis=dict(autorange="reversed"), margin=dict(l=200))
        ui.plot(f, container=c1, key="szdrag")
        c2.dataframe(pd.DataFrame([{k: ("" if v is None else (f"{v:.5f}" if k in ("CD0", "Cf") else
                                         f"{v:.3g}" if k == "Re" else f"{v:.2f}" if isinstance(v, float) else v))
                                     for k, v in r.items()} for r in res["drag"]]),
                     hide_index=True, width="stretch", height=380)
        st.caption("Full options (finish, laminar flow, gear details, struts) on the Drag buildup page.")
    with tabs[4]:
        ld = res["loading"]
        c1, c2 = st.columns([1, 1])
        f = go.Figure()
        for i, (kk, pts) in enumerate(ld["sequences"].items()):
            f.add_scatter(x=[p["pct_mac"] for p in pts], y=[p["mass"] for p in pts], name=kk, mode="lines+markers",
                          line=dict(color=ui.SERIES[i]))
        f.update_layout(title="Loading diagram", xaxis_title="c.g. (% MAC)", yaxis_title="mass (kg)", height=380)
        ui.plot(f, container=c1, key="szld")
        g = res["gear"]
        if g:
            c2.markdown(f"**c.g. range** {ld['forward']['pct_mac']:.1f} – {ld['aft']['pct_mac']:.1f} % MAC  \n"
                        f"**Gear** main at x = {g['x_main']:.2f} m, nose at {g['x_nose_or_tail']:.2f} m, "
                        f"track {g['track']:.2f} m; propeller ⌀ {res['prop_diameter']:.2f} m")
            c2.dataframe(pd.DataFrame([{"check": kk, "ok": "yes" if v else "no"} for kk, v in g["checks"].items()]),
                         hide_index=True, width="stretch")
            c2.caption(f"Tip-back {g['tipback_deg']:.1f}°, turnover {g['turnover_deg']:.1f}°, rotation "
                       f"{g['rotation_deg']:.1f}°, nose wheel {100 * g['nose_load_forward']:.0f} % (fwd c.g.) / "
                       f"{100 * g['nose_load_aft']:.0f} % (aft c.g.). Refine on the Weight and balance page.")
    with tabs[5]:
        h = pd.DataFrame(res["iterations"])
        f = go.Figure()
        f.add_scatter(x=h["iteration"], y=h["W0"], name="take-off mass", mode="lines+markers")
        f.add_scatter(x=h["iteration"], y=h["empty"], name="empty mass", mode="lines+markers")
        f.update_layout(title="Convergence", xaxis_title="iteration", yaxis_title="kg", height=320)
        ui.plot(f, key="szit")
        st.dataframe(h, hide_index=True, width="stretch")
    with tabs[6]:
        c1, c2 = st.columns(2)
        slug = c1.text_input("File name", re.sub(r"[^A-Za-z0-9_-]+", "_", res["requirements"]["name"]).strip("_").lower())
        c1.caption("Saves the aircraft (geometry, component masses and positions, gear, propulsion) to the "
                   "aircraft list. Stability, loads, performance, weight and balance then use it.")
        if c1.button("Save as aircraft model", type="primary"):
            p = C.AIRCRAFT / f"{re.sub(r'[^A-Za-z0-9_-]+', '_', slug) or 'sized_aircraft'}.json"
            A.Aircraft.from_dict(res["aircraft"]).save(p)
            (p.with_suffix(".sizing.json")).write_text(json.dumps(res, indent=1))
            AV.select(p)
            st.success(f"Saved {p.name}. It is now the selected aircraft on every aircraft page.")
        c2.download_button("Download sizing report (JSON)", json.dumps(res, indent=1), file_name="sizing.json")
        c2.download_button("Download aircraft (JSON)", json.dumps(res["aircraft"], indent=1),
                           file_name="aircraft.json")
        st.markdown("##### Verify with the performance tool")
        st.caption("Runs the Performance page's analysis on the sized aircraft: trimmed nonlinear drag polar "
                   "from the vortex lattice and CFD section polars, BEMT propeller designed for the cruise point, "
                   "time-integrated take-off and landing. Two runs: sea level (field, climb, ceiling) and cruise "
                   "altitude (range). About 2 minutes.")
        if st.button("Verify"):
            out = C.CASES / f"verify_{slug}_{int(time.time())}"
            out.mkdir(parents=True, exist_ok=True)
            acv = A.Aircraft.from_dict(res["aircraft"])
            acv.save(out / "aircraft.json")
            spec = dict(acv.propulsion)
            if res["battery"] > 0:                  # only the cruise energy, as for fuel below
                kwh = res["battery"] * res["requirements"]["battery_wh_kg"] / 1e3
                spec["engine"] = dict(spec["engine"]) | {
                    "battery_kwh": kwh * res["mission"].get("cruise_energy_fraction", 1.0)}
            (out / "propulsion.json").write_text(json.dumps(spec, indent=1))
            fuel = res["fuel"]
            # fuel not available for the cruise (start-up, climb, descent, reserve, trapped): kept back, so the
            # verified range at cruise speed compares with the required range like for like
            reserve = max(fuel - res["mission"].get("cruise_fuel", fuel), 0.0)
            jobs = {}
            for tag, alt in (("sl", 0.0), ("cruise", res["requirements"]["cruise_alt"])):
                o = out / tag
                o.mkdir(exist_ok=True)
                args = ["performance.py", out / "aircraft.json", "--propulsion", out / "propulsion.json",
                        "--fuel", fuel, "--reserve", reserve, "--alt", alt, "--to-flap", 10, "--ld-flap", 30,
                        "--polar-dir", C.POLAR_CACHE, "--out", o]
                jobs[tag] = C.start_job("performance", f"Verify {res['requirements']['name']} ({tag})", args, o)
            st.session_state["sz_verify"] = {"jobs": jobs, "out": str(out)}

vj = st.session_state.get("sz_verify")
if res and vj:
    st.markdown("### Verification")
    stat = {t: C.get_job(j)["status"] for t, j in vj["jobs"].items()}
    if any(s in C.ACTIVE for s in stat.values()):
        for t, j in vj["jobs"].items():
            views.live_job(j, "performance")

        @st.fragment(run_every="4s")
        def _watch():
            if not any(C.get_job(j)["status"] in C.ACTIVE for j in vj["jobs"].values()):
                st.rerun(scope="app")
        _watch()
    else:
        out = Path(vj["out"])
        try:
            sl = json.loads((out / "sl" / "performance.json").read_text())
            cr = json.loads((out / "cruise" / "performance.json").read_text())
        except FileNotFoundError:
            st.error("A verification run failed; its log is under Results → Runs.")
        else:
            rq = res["requirements"]
            V = np.array(cr["curves"]["V"])
            rng_at_vc = float(np.interp(rq["cruise_speed"], V, cr["curves"]["range_km"]))
            rows = [
                {"quantity": "stall speed, landing flap (m/s)", "required": rq["stall_speed"],
                 "verified": sl["landing"]["V_stall"], "ok": sl["landing"]["V_stall"] <= rq["stall_speed"] * 1.01},
                {"quantity": "take-off over 15 m (m)", "required": rq.get("takeoff_distance"),
                 "verified": sl["takeoff"].get("total"),
                 "ok": bool(sl["takeoff"].get("total")) and (not rq.get("takeoff_distance") or
                                                            sl["takeoff"]["total"] <= rq["takeoff_distance"] * 1.01)},
                {"quantity": "landing over 15 m (m)", "required": rq.get("landing_distance"),
                 "verified": sl["landing"]["total"],
                 "ok": not rq.get("landing_distance") or sl["landing"]["total"] <= rq["landing_distance"] * 1.01},
                {"quantity": "rate of climb, sea level (m/s)", "required": rq["climb_rate"],
                 "verified": sl["best_rate"]["roc"], "ok": sl["best_rate"]["roc"] >= rq["climb_rate"] * 0.99},
                {"quantity": "service ceiling (m)", "required": rq.get("ceiling"),
                 "verified": sl.get("ceiling_service") or 9000.0,
                 "ok": not rq.get("ceiling") or (sl.get("ceiling_service") or 9e3) >= rq["ceiling"] * 0.99},
                {"quantity": f"maximum level speed at {rq['cruise_alt']:g} m (m/s)", "required": rq["cruise_speed"],
                 "verified": cr["V_max"], "ok": cr["V_max"] >= rq["cruise_speed"]},
                {"quantity": f"range at {rq['cruise_speed']:g} m/s, {rq['cruise_alt']:g} m, cruise fuel (km)",
                 "required": rq["range_km"], "verified": rng_at_vc, "ok": rng_at_vc >= rq["range_km"] * 0.99},
            ]
            df = pd.DataFrame(rows)
            df["ok"] = df["ok"].map({True: "yes", False: "no"})
            c1, c2 = st.columns([3, 2])
            c1.dataframe(df, hide_index=True, width="stretch",
                         column_config={"required": st.column_config.NumberColumn(format="%.1f"),
                                        "verified": st.column_config.NumberColumn(format="%.1f")})
            c2.markdown(f"Trimmed CL max {sl['CLmax_trimmed']:.2f} clean, {sl['landing']['CLmax']:.2f} landing "
                        f"(sizing assumed {rq['cl_max_landing']:.2f}). Polar fit CD0 {sl['polar_fit']['CD0']:.4f}, "
                        f"e {sl['polar_fit']['e_oswald']:.2f} (sizing {res['CD0']:.4f}, {res['oswald_e']:.2f}).")
            c2.caption("Where a line fails, adjust that requirement's assumption (CL max, propeller efficiency) "
                       "or add margin, size again and verify. The performance tool is the higher-fidelity check.")
