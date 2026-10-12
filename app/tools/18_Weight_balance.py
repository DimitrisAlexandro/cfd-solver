import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import aircraft_views as AV
import common as C
import ui

import aircraft as A
import weight_balance as WB

C.page_setup()
ui.header("Weight and balance", "Component masses and positions, loading cases and the c.g. envelope, "
                                "horizontal-tail sizing from the scissor plot, and landing-gear geometry "
                                "checks.")

top = st.columns([3, 2], vertical_alignment="bottom")
with top[0]:
    ac, path = AV.aircraft_picker(key="wbac")
has_list = bool(ac.components)
top[1].caption("Component list saved with the aircraft." if has_list else
               "This aircraft has no component list yet: an estimated one (generic light aircraft, "
               "group fractions of the empty mass) is shown. Edit and save it.")

comps = WB.components_of(ac)
COLS = ["name", "group", "mass", "x", "z", "max"]
df0 = pd.DataFrame([{k: c.get(k) for k in COLS} for c in comps], columns=COLS)

left, right = st.columns([6, 5], gap="large")
with left:
    with st.container(border=True):
        ui.step("Components", first=True)
        st.caption("x from the nose, z up (m). Group: empty (structure, systems, engine), crew, payload, "
                   "fuel, battery. 'max' is the full value of a variable item (fuel, a seat, baggage); "
                   "loading cases fill it from 0 to max.")
        df = st.data_editor(df0, num_rows="dynamic", width="stretch", key=f"wbdf{path}", height=420,
                            column_config={
                                "name": "Component",
                                "group": st.column_config.SelectboxColumn("Group", options=list(WB.GROUPS)),
                                "mass": st.column_config.NumberColumn("Mass (kg)", format="%.1f", min_value=0.0),
                                "x": st.column_config.NumberColumn("x (m)", format="%.3f"),
                                "z": st.column_config.NumberColumn("z (m)", format="%.3f"),
                                "max": st.column_config.NumberColumn("max (kg)", format="%.1f", min_value=0.0)})
        rows = []
        for _, r in df.iterrows():
            if not str(r["name"]).strip() or r["mass"] != r["mass"]:
                continue
            d = {"name": str(r["name"]), "group": str(r["group"] or "empty"), "mass": float(r["mass"]),
                 "x": None if r["x"] != r["x"] else float(r["x"]), "z": None if r["z"] != r["z"] else float(r["z"])}
            if r["max"] == r["max"] and r["max"] is not None:
                d["max"] = float(r["max"])
            rows.append(d)
        comps = WB.place(rows, ac)
        c1, c2 = st.columns(2)
        if c1.button("Estimate the list again", width="stretch",
                     help="Replace the table by the generic estimate for this geometry"):
            a2 = A.Aircraft.from_dict(ac.to_dict())
            a2.components = []
            a2.save(path)
            st.session_state.pop(f"wbdf{path}", None)
            st.rerun()
        target = c2.number_input("Move the wing: loaded c.g. at (% MAC)", 5.0, 45.0, 25.0, 1.0)
        if st.button("Place the wing for this c.g.", width="stretch",
                     help="Moves the wing and the items attached to it (fuel, main gear, flight controls) along x"):
            a2, comps2 = WB.place_wing(ac, comps, target)
            a2.components = comps2
            a2.save(path)
            st.session_state.pop(f"wbdf{path}", None)
            st.rerun()

ld = WB.loading(ac, comps)
with right:
    k = st.columns(3)
    k[0].metric("Empty mass", f"{ld['cases']['empty']['mass']:.0f} kg",
                f"c.g. {ld['cases']['empty']['pct_mac']:.1f} % MAC", delta_color="off", delta_arrow="off")
    k[1].metric("Maximum mass", f"{ld['max_mass']:.0f} kg", f"aircraft file: {ac.mass:.0f} kg", delta_color="off", delta_arrow="off")
    k[2].metric("c.g. range (% MAC)", f"{ld['forward']['pct_mac']:.1f}–{ld['aft']['pct_mac']:.1f}",
                f"{ld['range_pct_mac']:.1f} % wide", delta_color="off", delta_arrow="off")
    f = go.Figure()
    for i, (lab, pts) in enumerate(ld["sequences"].items()):
        f.add_scatter(x=[p["pct_mac"] for p in pts], y=[p["mass"] for p in pts], mode="lines+markers", name=lab,
                      line=dict(color=ui.SERIES[i]))
    for lab, p in ld["cases"].items():
        f.add_scatter(x=[p["pct_mac"]], y=[p["mass"]], mode="markers", marker=dict(symbol="diamond", size=9,
                                                                                   color=ui.INK),
                      name=lab, showlegend=False, hovertext=lab, hoverinfo="text+x+y")
    f.update_layout(title="Loading diagram", xaxis_title="c.g. (% MAC)", yaxis_title="mass (kg)", height=380)
    ui.plot(f, key="wbld")
    st.dataframe(pd.DataFrame([{"case": k_, "mass (kg)": v["mass"], "x (m)": v["x"], "% MAC": v["pct_mac"]}
                               for k_, v in ld["cases"].items()]), hide_index=True, width="stretch",
                 column_config={"mass (kg)": st.column_config.NumberColumn(format="%.1f"),
                                "x (m)": st.column_config.NumberColumn(format="%.3f"),
                                "% MAC": st.column_config.NumberColumn(format="%.1f")})

tabs = st.tabs(["Scissor plot (tail sizing)", "Landing gear", "Save"])
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    sm = c1.number_input("Minimum static margin (% MAC)", 0.0, 30.0, 5.0, 1.0)
    cll = c2.number_input("CL max, landing (trimmed)", 1.0, 3.5, 1.9, 0.05)
    flp = c3.number_input("Landing flap (deg)", 0.0, 45.0, 30.0)
    emax = c4.number_input("Elevator, full nose-up (deg)", -40.0, -5.0, -25.0)
    eff = c5.number_input("Elevator effectiveness at full deflection", 0.3, 1.0, 0.7, 0.05,
                          help="The lattice is linear in the deflection; plain flaps lose effectiveness at large "
                               "angles (DATCOM: about 0.65–0.75 at 25°)")
    if st.button("Compute the scissor plot", type="primary"):
        try:
            with st.spinner("Vortex lattice for seven tail sizes"):
                sc = WB.scissor(ac, sm / 100, cll, flp, emax, polar_dir=C.POLAR_CACHE, elevator_eff=eff)
            st.session_state["wb_sc"] = (str(path), A._jsonable(sc))
        except ValueError as e:
            st.error(str(e))
    got = st.session_state.get("wb_sc")
    if got and got[0] == str(path):
        sc = got[1]
        r = sc["rows"]
        sh = [x["Sh_S"] for x in r]
        need = WB.min_tail(sc, ld["forward"]["pct_mac"], ld["aft"]["pct_mac"])
        f = go.Figure()
        f.add_scatter(x=[x["pct_fwd"] for x in r], y=sh, name="control limit (forward)", mode="lines+markers",
                      line=dict(color=ui.SERIES[1]))
        f.add_scatter(x=[x["pct_aft"] for x in r], y=sh, name=f"stability limit (aft, SM {sm:g} %)",
                      mode="lines+markers", line=dict(color=ui.SERIES[0]))
        f.add_scatter(x=[x["pct_np"] for x in r], y=sh, name="neutral point", mode="lines",
                      line=dict(color=ui.GRAPHITE, dash="dot", width=1))
        cur = sc["Sh_S_current"]
        f.add_scatter(x=[ld["forward"]["pct_mac"], ld["aft"]["pct_mac"]], y=[cur, cur], mode="lines+markers",
                      name="c.g. range, current tail", line=dict(color=ui.INK, width=5))
        if need:
            f.add_hline(y=need, line=dict(color=ui.SERIES[2], dash="dash", width=1),
                        annotation_text=f"smallest tail {need:.3f}", annotation_font_size=10)
        f.update_layout(title="Scissor plot", xaxis_title="c.g. (% MAC)", yaxis_title="S_h / S", height=440)
        c1, c2 = st.columns([3, 2])
        ui.plot(f, container=c1, key="wbsc")
        i = int(np.argmin(np.abs(np.array([x["factor"] for x in r]) - 1.0)))
        c2.markdown(f"**Current tail** S_h/S = {cur:.3f}: c.g. allowed from {r[i]['pct_fwd']:.1f} to "
                    f"{r[i]['pct_aft']:.1f} % MAC (neutral point {r[i]['pct_np']:.1f} %).  \n"
                    f"**Loading** needs {ld['forward']['pct_mac']:.1f} – {ld['aft']['pct_mac']:.1f} % MAC.")
        if need:
            c2.markdown(f"**Smallest tail** covering the loading range with this wing position: "
                        f"S_h/S = {need:.3f} ({need * ac.S:.2f} m², {100 * (need / cur - 1):+.0f} %).")
        else:
            c2.warning("No tail size in the range covers the c.g. range at this wing position. Move the wing "
                       "(left panel) or reduce the c.g. range, then compute again.")
        c2.caption(f"Stability: stick-fixed neutral point from the lattice (downwash, fuselage). Control: "
                   f"landing flap {flp:g}°, CL {cll:g}, elevator {emax:g}° × {eff:g}, at "
                   f"{sc['speed']:.1f} m/s; no ground effect (it moves the forward limit aft - keep margin). "
                   "Tail scaled in area with its aspect ratio and quarter-chord MAC position kept.")
with tabs[1]:
    g0 = dict(ac.gear or {})
    if not g0:
        prop_d = float((ac.propulsion or {}).get("propeller", {}).get("design", {}).get("diameter", 1.8))
        g0 = WB.suggest_gear(ac, ld, prop_diameter=prop_d)
        st.caption("No gear on this aircraft yet: positions suggested for the criteria below.")
    c1, c2, c3, c4, c5 = st.columns(5)
    xm = c1.number_input("Main gear x (m)", -50.0, 200.0, float(g0["x_main"]), 0.01)
    xn = c2.number_input("Nose (or tail) wheel x (m)", -50.0, 200.0, float(g0["x_nose"]), 0.01)
    trk = c3.number_input("Track (m)", 0.3, 30.0, float(g0["track"]), 0.05)
    zg = c4.number_input("Ground z (m)", -20.0, 5.0, float(g0["z_ground"]), 0.01,
                         help="Height of the ground in aircraft axes (wheels on the ground, aircraft level)")
    pdia = c5.number_input("Propeller diameter (m)", 0.0, 6.0,
                           float((ac.propulsion or {}).get("propeller", {}).get("design", {}).get("diameter", 1.8)),
                           0.05)
    if st.button("Suggest positions"):
        g0 = WB.suggest_gear(ac, ld, prop_diameter=pdia)
        a2 = A.Aircraft.from_dict(ac.to_dict())
        a2.gear = g0
        a2.save(path)
        st.rerun()
    gear = {"x_main": xm, "x_nose": xn, "track": trk, "z_ground": zg}
    a2 = A.Aircraft.from_dict(ac.to_dict())
    a2.gear = gear
    gc = WB.gear_checks(a2, ld, prop_diameter=pdia or None)
    c1, c2 = st.columns([2, 3])
    c1.dataframe(pd.DataFrame([{"check": k_, "ok": "yes" if v else "no"} for k_, v in gc["checks"].items()]),
                 hide_index=True, width="stretch")
    vals = {"tip-back angle (°)": gc["tipback_deg"], "turnover angle (°)": gc["turnover_deg"],
            "rotation angle (°)": gc["rotation_deg"],
            "nose wheel load, fwd c.g. (%)": 100 * gc["nose_load_forward"],
            "nose wheel load, aft c.g. (%)": 100 * gc["nose_load_aft"],
            "propeller clearance (m)": gc.get("prop_clearance")}
    c1.dataframe(pd.DataFrame([{"quantity": k_, "value": v} for k_, v in vals.items() if v is not None]),
                 hide_index=True, width="stretch", column_config={"value": st.column_config.NumberColumn(format="%.2f")})
    # side view sketch
    f = go.Figure()
    fu = a2.fuselage
    xs = fu.x + np.linspace(0, fu.length, 60)
    r_ = fu.radius(xs)
    f.add_scatter(x=np.r_[xs, xs[::-1]], y=np.r_[fu.z + r_, (fu.z - r_)[::-1]], fill="toself", mode="lines",
                  line=dict(color=ui.GRAPHITE, width=1), fillcolor="rgba(29,78,137,0.05)", showlegend=False)
    w = a2.wing
    f.add_scatter(x=[w.x, w.x + w.root_chord], y=[w.z, w.z], mode="lines", line=dict(color=ui.BLUE, width=4),
                  name="wing root")
    f.add_hline(y=zg, line=dict(color=ui.INK, width=1))
    for x_, lab in ((xm, "main"), (xn, "nose")):
        f.add_scatter(x=[x_, x_], y=[zg, fu.z - float(fu.radius(np.array([x_]))[0])], mode="lines+markers",
                      line=dict(color=ui.INK, width=2), name=f"{lab} gear", showlegend=False)
    for lab, p, col, tp in (("fwd c.g.", ld["forward"], ui.SERIES[1], "top left"),
                            ("aft c.g.", ld["aft"], ui.SERIES[0], "bottom right")):
        f.add_scatter(x=[p["x"]], y=[p["z"]], mode="markers+text", text=[lab], textposition=tp,
                      marker=dict(symbol="diamond", size=9, color=col), showlegend=False)
    p = ld["aft"]
    f.add_scatter(x=[xm, p["x"]], y=[zg, p["z"]], mode="lines", line=dict(color=ui.SERIES[0], dash="dot", width=1),
                  name="tip-back line")
    if pdia:
        f.add_shape(type="line", x0=fu.x - 0.05, x1=fu.x - 0.05, y0=fu.z - pdia / 2, y1=fu.z + pdia / 2,
                    line=dict(color=ui.SERIES[2], width=3))
    f.update_layout(title="Side view", xaxis_title="x (m)", yaxis=dict(scaleanchor="x", title="z (m)"), height=330)
    ui.plot(f, container=c2, key="wbside")
    st.caption("Criteria (Raymer ch. 11, CS-23.925): nose wheel 8–15 % of the weight ideal (5–20 % acceptable), "
               "tip-back ≥ 15° (main gear behind the aft c.g.), turnover ≤ 63°, rotation (tail strike) ≥ 12°, "
               "propeller clearance ≥ 0.18 m.")
with tabs[2]:
    case = st.selectbox("c.g. stored in the aircraft file (used by stability, loads, performance)",
                        list(ld["cases"]), index=list(ld["cases"]).index("crew + payload + full fuel"))
    st.caption(f"{case}: {ld['cases'][case]['mass']:.1f} kg at x = {ld['cases'][case]['x']:.3f} m "
               f"({ld['cases'][case]['pct_mac']:.1f} % MAC). For the forward and aft limits, save a copy per case.")
    c1, c2 = st.columns(2)
    if c1.button("Save components, gear and this c.g.", type="primary", width="stretch"):
        a2 = A.Aircraft.from_dict(ac.to_dict())
        a2.components = comps
        a2.gear = gear
        a2.mass = float(ld["cases"][case]["mass"])
        a2.cg = (float(ld["cases"][case]["x"]), 0.0, float(ld["cases"][case]["z"]))
        a2.save(path)
        st.success(f"Saved to {path.name}.")
    c2.download_button("Download weight and balance (JSON)",
                       json.dumps(A._jsonable({"loading": {k_: v for k_, v in ld.items()}, "gear": gc}), indent=1),
                       file_name="weight_balance.json", width="stretch")
