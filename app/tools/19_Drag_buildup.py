import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import aircraft_views as AV
import common as C
import ui
import views

from aerolab.aircraft import aircraft as A
from aerolab.aircraft import drag_buildup as DB

C.page_setup()
ui.header("Drag buildup", "Parasite drag component by component: skin friction with laminar runs and "
                          "roughness, form factors, interference, landing gear, cooling, upsweep, leakage. "
                          "The traceable CD0 behind the drag polar.")

top = st.columns([3, 2], vertical_alignment="bottom")
with top[0]:
    ac, path = AV.aircraft_picker(key="dbac")

setup, out = st.columns([5, 7], gap="large")
with setup:
    with st.container(border=True):
        ui.step("Flight condition", first=True)
        c1, c2 = st.columns(2)
        V = c1.number_input("Speed (m/s)", 5.0, 250.0, 55.0)
        h = c2.number_input("Altitude (m)", 0.0, 12000.0, 1500.0, 100.0)
        ui.step("Surfaces")
        c1, c2 = st.columns(2)
        finish = c1.selectbox("Surface finish", list(DB.ROUGHNESS), index=1,
                              help="Equivalent sand roughness (Raymer Table 12.5): caps the Reynolds number")
        leak = c2.number_input("Leakage and protuberances (%)", 0.0, 30.0, 7.0, 0.5,
                               help="Raymer: 5–10 % for propeller aircraft, 2–5 % for jets")
        c1, c2, c3 = st.columns(3)
        lw = c1.number_input("Laminar run, wing", 0.0, 0.7, 0.10, 0.05,
                             help="Fraction of the chord. Riveted metal 0.05–0.10, smooth composite 0.3–0.5. "
                                  "The Airfoil polar page gives x_tr from the e^N model.")
        lt = c2.number_input("Laminar run, tails", 0.0, 0.7, 0.10, 0.05)
        lf = c3.number_input("Laminar run, fuselage", 0.0, 0.5, 0.05, 0.05)
        c1, c2, c3 = st.columns(3)
        qw = c1.number_input("Interference Q, wing", 1.0, 1.5, 1.0, 0.01, help="Filleted 1.0; high wing 1.0–1.1")
        qt = c2.number_input("Interference Q, tails", 1.0, 1.5, 1.04, 0.01, help="Conventional 1.04–1.05, V-tail 1.03")
        qf = c3.number_input("Interference Q, fuselage", 1.0, 1.5, 1.0, 0.01)
        ui.step("Landing gear")
        gp = st.segmented_control("Type", ["retractable", "fixed_faired", "fixed_bare", "fixed_struts", "custom"],
                                  default="fixed_faired",
                                  format_func=lambda k: {"fixed_faired": "fixed, faired", "fixed_bare": "fixed, spring legs",
                                                         "fixed_struts": "fixed, round struts"}.get(k, k))
        gear = {"type": gp or "retractable"}
        if gp and gp != "retractable":
            c1, c2, c3 = st.columns(3)
            nw = c1.number_input("Wheels", 1, 12, 3)
            dt = c2.number_input("Tyre diameter (m)", 0.1, 1.5, 0.38, 0.01)
            wt = c3.number_input("Tyre width (m)", 0.03, 0.6, 0.13, 0.01)
            c1, c2, c3 = st.columns(3)
            wk = c1.selectbox("Wheel", list(DB.GEAR_WHEEL), index=2 if gp == "fixed_faired" else 0)
            lk = c2.selectbox("Legs", list(DB.GEAR_LEG),
                              index=list(DB.GEAR_LEG).index((DB.GEAR_PRESETS.get(gp) or {}).get("leg", "round strut")))
            ll = c3.number_input("Leg length × width (m²)", 0.0, 2.0, 0.02, 0.005, format="%.3f",
                                 help="Frontal area of one leg")
            gear = {"items": [{"item": wk, "kind": wk, "count": nw, "width": wt, "length": dt},
                              {"item": lk, "kind": lk, "count": nw, "width": ll, "length": 1.0}]}
        ui.step("Engine and fuselage")
        prop = ac.propulsion or {}
        p_def = float((prop.get("engine") or {}).get("power_kw", 75.0))
        c1, c2, c3 = st.columns(3)
        ek = c1.selectbox("Engine (cooling drag)", ["piston", "electric", "turboprop", "none"],
                          index=["piston", "electric", "turboprop", "none"].index((prop.get("engine") or {}).get("kind", "piston")))
        pw = c2.number_input("Power in cruise (kW)", 0.0, 5000.0, 0.75 * p_def)
        ups = c3.number_input("Aft fuselage upsweep (°)", 0.0, 25.0, 0.0, 1.0,
                              help="Angle of the aft-fuselage centreline; the body of revolution in the model has none")
        ui.step("Other items (D/q, m²)")
        st.caption("Wing struts, canopy, antennas, steps, external stores. Streamline strut: 0.05 × length × "
                   "thickness; round wire: 0.30 × length × diameter (Raymer Table 12.6).")
        oth = st.data_editor(pd.DataFrame([{"item": "", "D_q": 0.0}]), num_rows="dynamic", width="stretch",
                             key="dbother", column_config={"D_q": st.column_config.NumberColumn("D/q (m²)", format="%.4f")})
        other = [{"item": str(r["item"]), "D_q": float(r["D_q"])} for _, r in oth.iterrows()
                 if str(r["item"]).strip() and r["D_q"] == r["D_q"] and float(r["D_q"]) > 0]

res = DB.buildup(ac, V, h, laminar={"wing": lw, "tail": lt, "fuselage": lf}, roughness=finish,
                 Q={"wing": qw, "tail": qt, "fuselage": qf}, gear=gear, upsweep_deg=ups, power_kw=pw,
                 engine=ek, leakage_pct=leak, other=other)
with out:
    k = st.columns(4)
    k[0].metric("CD0", f"{res['CD0']:.4f}", f"{1e4 * res['CD0']:.0f} counts", delta_color="off", delta_arrow="off")
    k[1].metric("Drag area f", f"{res['f_m2']:.3f} m²", f"{res['f_m2'] * 10.764:.2f} ft²", delta_color="off", delta_arrow="off")
    k[2].metric("Wetted / reference area", f"{res['S_wet_ratio']:.2f}", f"S_wet {res['S_wet_total']:.1f} m²",
                delta_color="off", delta_arrow="off")
    k[3].metric("Cfe check (Raymer)", f"{res['cfe_check']['CD0']:.4f}",
                f"{100 * (res['CD0'] / res['cfe_check']['CD0'] - 1):+.0f} % vs buildup", delta_color="off", delta_arrow="off")
    rows = DB.table(res)
    d = pd.DataFrame(rows)
    f = go.Figure(go.Bar(x=1e4 * d["CD0"], y=d["component"], orientation="h", marker_color=ui.BLUE,
                         text=[f"{v:.1f} %" for v in d["share %"]], textposition="outside"))
    f.update_layout(title="Contributions", xaxis_title="drag counts (1e-4)", height=400,
                    yaxis=dict(autorange="reversed"), margin=dict(l=220),
                    xaxis_range=[0, 1.2e4 * float(d["CD0"].max())])
    ui.plot(f, key="dbbar")
    st.dataframe(pd.DataFrame(DB.table_text(res)), hide_index=True, width="stretch")

tabs = st.tabs(["Drag polar", "Use in the aircraft model", "Method"])
with tabs[0]:
    e = res["oswald_e"]
    c1, c2 = st.columns([3, 2])
    cl = np.linspace(0, 1.6, 60)
    f = go.Figure()
    f.add_scatter(x=DB.polar(res, cl), y=cl, name=f"CD0 + CL²/(π A e), e {e:.2f}")
    LDmax = 0.5 * np.sqrt(np.pi * res["aspect_ratio"] * e / res["CD0"])
    f.update_layout(title=f"Parabolic polar: L/D max {LDmax:.1f} at CL {np.sqrt(res['CD0'] * np.pi * res['aspect_ratio'] * e):.2f}",
                    xaxis_title="CD", yaxis_title="CL", height=360)
    ui.plot(f, container=c1, key="dbpol")
    c2.markdown(f"Oswald factor {e:.2f}: Raymer's estimate for aspect ratio {res['aspect_ratio']:.1f}. "
                "The Stability and Performance pages compute the lift-dependent drag from the vortex lattice "
                "and the CFD section polars instead; this parabola is the quick hand-check.")
with tabs[1]:
    extra = DB.cd_extra_for_model(res)
    st.markdown(f"The aircraft model already computes the lifting surfaces' profile drag (from the section polars) "
                f"and the fuselage friction. What it does not contain is the miscellaneous part of this buildup: "
                f"**ΔCD = {extra:.4f}** (gear, cooling, upsweep, base, other items, leakage). The model's current "
                f"value is {ac.cd_extra:.4f}.")
    if st.button("Set the model's extra drag to this value", type="primary"):
        a2 = A.Aircraft.from_dict(ac.to_dict())
        a2.cd_extra = float(extra)
        a2.save(path)
        st.success(f"Saved ΔCD {extra:.4f} to {path.name}.")
    st.download_button("Download the buildup (JSON)", json.dumps(A._jsonable(res), indent=1),
                       file_name="drag_buildup.json")
with tabs[2]:
    st.markdown(
        "- **Skin friction**: flat plate at the component's Reynolds number; turbulent 0.455/(log Re)^2.58 "
        "with the Mach correction, laminar 1.328/√Re over the laminar run (Prandtl–Schlichting mixed "
        "boundary layer). Roughness caps Re at 38.21 (l/k)^1.053.\n"
        "- **Lifting surfaces**: strip by strip over the exposed span (outside the fuselage), wetted area "
        "from the section's real perimeter, form factor [1 + 0.6/(x/c)_m (t/c) + 100 (t/c)^4] "
        "[1.34 M^0.18 cos^0.28 Λ_m].\n"
        "- **Fuselage**: wetted area of the body of revolution, form factor 1 + 60/f³ + f/400.\n"
        "- **Miscellaneous**: landing gear by Raymer Table 12.6, upsweep 3.83 u^2.5 A_max, base drag, cooling-air "
        "momentum loss (heat 0.7 P for an air-cooled piston, ΔT 50 K, half the momentum lost), leakage as a "
        "percentage.\n"
        "- **Check**: equivalent skin-friction method, CD0 = Cfe S_wet / S with Cfe = 0.0055 (light single).\n\n"
        "Validation: a Cessna 172-like geometry (strut-braced, fixed gear with wheel fairings, riveted metal) "
        "gives CD0 0.027 against 0.027–0.031 quoted for the type.")

views.workflow_footer("tools/19_Drag_buildup.py")
