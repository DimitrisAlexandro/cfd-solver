import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import aircraft_views as AV
import common as C
import ui

import aircraft as A
import wing_structure as WS

C.page_setup()
ui.header("Wing structure", "First-pass wing box from the flight loads: spar caps, webs, skins and ribs, the "
                            "wing mass, bending and torsion stiffness, deflection, and the aeroelastic checks "
                            "(divergence, aileron reversal, flutter estimate).")

top = st.columns([3, 3], vertical_alignment="bottom")
with top[0]:
    ac, path = AV.aircraft_picker(key="wsac")
runs = []
for f in sorted(C.CASES.glob("loads_*/loads.json"), key=lambda p: -p.stat().st_mtime):
    try:
        d = json.loads(f.read_text())
    except Exception:                                   # noqa: BLE001
        continue
    if d.get("aircraft") == ac.name and d.get("surface") == ac.wing.name:
        runs.append((f, d))
with top[1]:
    if runs:
        k = st.selectbox("Flight loads run", range(len(runs)),
                         format_func=lambda i: f"{runs[i][0].parent.name}  ({runs[i][1]['vn']['category']}, "
                                               f"V_D {runs[i][1]['vn']['Vd']:.0f} m/s, "
                                               f"{len(runs[i][1]['cases'])} cases)")
        lf, loads = runs[k]
    else:
        loads = None
if loads is None:
    st.info(f"No flight loads for {ac.name} yet. Compute them on the Flight loads page (about half a minute), "
            "then come back: this page sizes the wing box for every load case of that run.")
    st.page_link("tools/15_Loads.py", label="Go to Flight loads")
    st.stop()

D = WS.default_settings()
setup, out = st.columns([4, 7], gap="large")
with setup:
    with st.container(border=True):
        ui.step("Material and layout", first=True)
        mat = st.selectbox("Material", list(WS.MATERIALS))
        c1, c2 = st.columns(2)
        xf = c1.number_input("Front spar (x/c)", 0.05, 0.4, D["front_spar"], 0.01)
        xr = c2.number_input("Rear spar (x/c)", 0.4, 0.8, D["rear_spar"], 0.01)
        c1, c2 = st.columns(2)
        fs = c1.number_input("Factor of safety", 1.0, 2.5, D["safety_factor"], 0.05, help="CS-23.303: 1.5")
        nopt = c2.number_input("Non-optimum factor", 1.0, 2.0, D["non_optimum"], 0.05,
                               help="Joints, fasteners, cut-outs, fittings on the primary structure")
        c1, c2 = st.columns(2)
        kc = c1.number_input("Compression knock-down", 0.3, 1.0, D["k_c"], 0.05,
                             help="Cap compression allowable / material F_cy: crippling and column buckling")
        ks = c2.number_input("Shear knock-down", 0.3, 1.0, D["k_s"], 0.05,
                             help="Web and skin shear allowable / F_su: post-buckled (tension-field) webs")
        ui.step("Ribs and secondary structure")
        c1, c2 = st.columns(2)
        pitch = c1.number_input("Rib pitch (m)", 0.1, 2.0, D["rib_pitch"], 0.05)
        trib = c2.number_input("Rib thickness (mm)", 0.3, 5.0, 1e3 * D["rib_thickness"], 0.1)
        sec = st.number_input("Secondary structure (kg/m² outside the box)", 0.0, 15.0, D["secondary_kg_m2"], 0.1,
                              help="Leading edge, trailing edge, ailerons and flaps: about 3 for light metal "
                                   "aircraft, 2–2.5 for composite")
        ui.step("Mass distribution (flutter)")
        c1, c2 = st.columns(2)
        xcg = c1.number_input("Section c.g. (x/c)", 0.25, 0.6, D["cg_chord"], 0.01)
        kg = c2.number_input("Radius of gyration (chords)", 0.1, 0.4, D["gyration_chord"], 0.01)
        c1, c2 = st.columns(2)
        fuel = c1.number_input("Fuel in the wing (kg)", 0.0, 1e5, float(loads.get("fuel_mass", 0.0)))
        fsp = loads.get("fuel_span", [0.1, 0.5])
        c2.caption(f"Between η {fsp[0]:g} and {fsp[1]:g} (from the loads run).")

sets = {"material": mat, "front_spar": xf, "rear_spar": xr, "safety_factor": fs, "non_optimum": nopt, "k_c": kc,
        "k_s": ks, "rib_pitch": pitch, "rib_thickness": trib / 1e3, "secondary_kg_m2": sec, "cg_chord": xcg,
        "gyration_chord": kg, "fuel_mass": fuel, "fuel_span": tuple(fsp)}
try:
    c2m = WS.class_ii_wing(ac, n_limit=loads["vn"]["n_pos"])
except Exception:                                      # noqa: BLE001
    c2m = None
res = WS.run(ac, loads, sets, class_ii_mass=c2m, log=lambda *a: None)
s_ = res["structure"]
ae = res["aeroelastic"]
fl = ae["flutter"]
VD = ae["V_D"]

with out:
    k = st.columns(4)
    k[0].metric("Wing mass", f"{res['wing_mass']:.1f} kg", f"{100 * res['wing_mass'] / ac.mass:.1f} % of {ac.mass:.0f} kg",
                delta_color="off", delta_arrow="off")
    k[1].metric("Class II estimate (Raymer)", f"{c2m:.1f} kg" if c2m else "–",
                f"loads run assumed {res['wing_mass_assumed_in_loads']:.0f} kg", delta_color="off", delta_arrow="off")
    k[2].metric("Tip deflection, 1 g", f"{100 * res['tip_deflection_1g']:.1f} cm",
                f"max {100 * res['tip_deflection_max']:.1f} cm (limit load)", delta_color="off", delta_arrow="off")
    k[3].metric("Tip twist, max", f"{res['tip_twist_max_deg']:.2f}°")
    k = st.columns(4)

    def ok(v, lim):
        return "✓" if v >= lim else "✗"
    k[0].metric("Flutter speed (estimate)", f"{fl['V_flutter']:.0f} m/s" if math.isfinite(fl["V_flutter"]) else "none found",
                f"{ok(fl['V_flutter'], 1.2 * VD)} 1.2 V_D = {1.2 * VD:.0f} m/s", delta_color="off", delta_arrow="off")
    k[1].metric("Divergence speed, EAS", f"{ae['V_divergence_eas']:.0f} m/s" if math.isfinite(ae["V_divergence_eas"]) else "none",
                f"{ok(ae['V_divergence_eas'], 1.2 * VD)} V_D {VD:.0f} m/s", delta_color="off", delta_arrow="off")
    rv = ae["reversal"]
    if rv:
        k[2].metric("Aileron reversal, EAS", f"{rv['V_eas']:.0f} m/s" if math.isfinite(rv["V_eas"]) else "none",
                    f"at V_D: {100 * rv['effectiveness_at_VD']:.0f} % roll power"
                    if rv.get("effectiveness_at_VD") is not None else "", delta_color="off", delta_arrow="off")
    k[3].metric("Bending / torsion frequency", f"{fl['f_bending_hz']:.1f} / {fl['f_torsion_hz']:.1f} Hz")
    if fl["V_flutter"] < 1.2 * VD:
        st.error("The flutter estimate is below 1.2 V_D: stiffen the box in torsion (thicker skins), move mass "
                 "forward (section c.g.), or check with a proper flutter analysis.")
    y = s_["y"]
    c1, c2 = st.columns(2)
    f = go.Figure()
    f.add_scatter(x=y, y=1e6 * s_["A_upper_cap"], name="upper cap")
    f.add_scatter(x=y, y=1e6 * s_["A_lower_cap"], name="lower cap")
    f.update_layout(title="Spar cap area (each)", xaxis_title="y (m)", yaxis_title="mm²", height=300)
    ui.plot(f, container=c1, key="wscap")
    f = go.Figure()
    f.add_scatter(x=y, y=1e3 * s_["t_web"], name="webs")
    f.add_scatter(x=y, y=1e3 * s_["t_skin"], name="skins")
    f.update_layout(title="Web and skin thickness", xaxis_title="y (m)", yaxis_title="mm", height=300)
    ui.plot(f, container=c2, key="wst")

tabs = st.tabs(["Deflection and twist", "Stiffness and mass", "Flutter", "Reversal", "Table and export", "Method"])
with tabs[0]:
    c1, c2 = st.columns(2)
    f, g = go.Figure(), go.Figure()
    for i, d in enumerate(res["deflection"]):
        f.add_scatter(x=y, y=100 * np.asarray(d["deflection"]), name=d["case"], line=dict(color=ui.SERIES[i % 7]))
        g.add_scatter(x=y, y=d["twist_deg"], name=d["case"], line=dict(color=ui.SERIES[i % 7]), showlegend=False)
    f.update_layout(title="Bending deflection, limit loads", xaxis_title="y (m)", yaxis_title="cm", height=380)
    g.update_layout(title="Elastic twist (nose up +)", xaxis_title="y (m)", yaxis_title="deg", height=380)
    ui.plot(f, container=c1, key="wsdf")
    ui.plot(g, container=c2, key="wstw")
with tabs[1]:
    c1, c2 = st.columns(2)
    f = go.Figure()
    f.add_scatter(x=y, y=s_["EI"], name="EI")
    f.add_scatter(x=y, y=s_["GJ"], name="GJ")
    f.update_layout(title="Stiffness", xaxis_title="y (m)", yaxis_title="N m²", yaxis_type="log", height=340)
    ui.plot(f, container=c1, key="wsst")
    mb = res["structure"]["mass_breakdown"]
    f = go.Figure(go.Bar(x=list(mb.values()), y=list(mb.keys()), orientation="h", marker_color=ui.BLUE))
    f.update_layout(title=f"Mass, both sides: {res['wing_mass']:.1f} kg", xaxis_title="kg", height=340,
                    yaxis=dict(autorange="reversed"), margin=dict(l=200))
    ui.plot(f, container=c2, key="wsmb")
    if abs(res["wing_mass"] - res["wing_mass_assumed_in_loads"]) > 0.15 * res["wing_mass"]:
        st.caption(f"The loads run assumed {res['wing_mass_assumed_in_loads']:.0f} kg of wing structure for the "
                   f"inertia relief; this sizing gives {res['wing_mass']:.0f} kg. Set 'Wing structure (fraction of "
                   f"mass)' to {res['wing_mass'] / ac.mass:.3f} on the Flight loads page and run it again.")
with tabs[2]:
    vg = fl["vg"]
    V = np.array(vg["V"]); G = np.array(vg["g"]); F = np.array(vg["f"])
    c1, c2 = st.columns(2)
    f, g = go.Figure(), go.Figure()
    for j in range(V.shape[1]):
        okm = np.isfinite(V[:, j]) & (V[:, j] < 3 * max(VD, 1))
        f.add_scatter(x=V[okm, j], y=G[okm, j], name=f"branch {j + 1}", line=dict(color=ui.SERIES[j]))
        g.add_scatter(x=V[okm, j], y=F[okm, j], name=f"branch {j + 1}", line=dict(color=ui.SERIES[j]), showlegend=False)
    for fig in (f, g):
        fig.add_vline(x=1.2 * VD, line=dict(color=ui.GRAPHITE, dash="dot", width=1), annotation_text="1.2 V_D",
                      annotation_font_size=10)
    f.add_hline(y=0, line=dict(color=ui.INK, width=1))
    f.update_layout(title="V–g diagram (required structural damping)", xaxis_title="V (m/s)", yaxis_title="g",
                    yaxis_range=[-1, 0.5], height=360)
    g.update_layout(title="Frequency", xaxis_title="V (m/s)", yaxis_title="Hz", height=360)
    ui.plot(f, container=c1, key="wsvg")
    ui.plot(g, container=c2, key="wsvf")
    st.caption(f"Typical section at {100 * fl['station']:.0f} % semi-span: mass ratio μ {fl['mu']:.1f}, elastic axis "
               f"a = {fl['a']:+.2f}, c.g. offset x_α = {fl['x_alpha']:+.2f}, r_α² = {fl['r2']:.3f}, ω_h/ω_α = "
               f"{fl['sigma']:.3f}; Theodorsen aerodynamics, sea-level density. Flutter where g crosses zero. "
               "Screening estimate only: no 3D aerodynamics, no aileron or tab flutter (mass-balance the "
               "controls), no engine or fuselage modes.")
with tabs[3]:
    if rv:
        f = go.Figure()
        q = np.array(rv["q_curve"])
        f.add_scatter(x=np.sqrt(2 * q / 1.225), y=100 * np.array(rv["effectiveness"]))
        f.add_hline(y=0, line=dict(color=ui.INK, width=1))
        f.add_vline(x=VD, line=dict(color=ui.GRAPHITE, dash="dot", width=1), annotation_text="V_D",
                    annotation_font_size=10)
        f.update_layout(title="Aileron effectiveness (elastic / rigid rolling moment)", xaxis_title="V_EAS (m/s)",
                        yaxis_title="%", height=360, showlegend=False)
        ui.plot(f, key="wsrev")
    st.caption(f"Strip theory on the elastic-axis torsion: lift slope {ae['lift_slope']:.2f} /rad (Helmbold), "
               f"elastic axis {100 * ae['e_chord']:.0f} % of the chord behind the aerodynamic centre.")
with tabs[4]:
    tab = pd.DataFrame(WS.table(s_))
    st.dataframe(tab, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4g") for c in tab.columns
                                if not c.endswith("case")})
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download table (CSV)", tab.to_csv(index=False), file_name="wing_structure.csv", width="stretch")
    c2.download_button("Download all results (JSON)", json.dumps(A._jsonable(res), indent=1),
                       file_name="wing_structure.json", width="stretch")
    beam = pd.DataFrame({"y_m": y, "EI_Nm2": s_["EI"], "GJ_Nm2": s_["GJ"], "mass_kg_per_m": s_["m_span"],
                         "shear_centre_x_over_c": s_["x_shear_centre"]})
    c3.download_button("Beam properties for FEM (CSV)", beam.to_csv(index=False), file_name="wing_beam.csv",
                       width="stretch")
with tabs[5]:
    st.markdown(
        "- **Loads**: every case of the selected Flight loads run × the factor of safety. Torsion moved to the "
        "box shear centre (mid-box).\n"
        "- **Box**: depth from the real section between the spars. Caps carry bending (cap separation 0.92 × "
        "mean depth); webs carry the vertical shear and the torsion shear flow; skins the torsion (Bredt). "
        "Minimum gauges apply.\n"
        "- **Mass**: members × non-optimum factor, ribs from the pitch, secondary structure per m² outside the box.\n"
        "- **Stiffness**: EI from caps and skins, GJ by Bredt. Deflection and twist under the limit loads, root "
        "clamped at the symmetry plane.\n"
        "- **Divergence and reversal**: torsion finite elements with strip aerodynamics (straight-wing model).\n"
        "- **Flutter**: first bending and torsion frequencies from beam finite elements, then the typical section "
        "with Theodorsen aerodynamics (V–g). CS-23.629 asks for freedom from flutter up to 1.2 V_D.\n\n"
        "Validation (tests): uniform-cantilever frequencies, divergence and full-span reversal against closed "
        "forms, Hodges & Pierce's typical-section flutter (U/bω_α ≈ 2.2).\n\n"
        "This is a conceptual sizing: it does not check skin buckling between stringers, joints, cut-outs, "
        "fatigue, or the fuselage carry-through.")
