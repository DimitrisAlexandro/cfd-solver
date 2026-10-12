import json
import math

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
ui.header("Stability and control", "Trim, neutral point, stability and control derivatives, hinge "
                                   "moments, flight modes and responses to the controls, from one "
                                   "vortex lattice of the whole aircraft with the CFD section polars.")

setup, summary = st.columns([4, 7], gap="large")
with setup:
    with st.container(border=True):
        ui.step("Aircraft and flight condition", first=True)
        ac, path = AV.aircraft_picker()
        c1, c2 = st.columns(2)
        V = c1.number_input("True airspeed (m/s)", 5.0, 300.0, 50.0, 1.0)
        alt = c2.number_input("Altitude (m)", 0.0, 15000.0, 1000.0, 100.0)
        c1, c2 = st.columns(2)
        mass = c1.number_input("Mass (kg)", 0.1, 1e6, float(ac.mass))
        cgx = c2.number_input("c.g. x (m)", -100.0, 200.0, float(ac.cg[0]), 0.01)
        st.caption("Changes here are for this analysis only; save the aircraft on the Aircraft model page.")
ac.mass, ac.cg = mass, (cgx, ac.cg[1], ac.cg[2])
with st.spinner("Solving trim, derivatives and modes..."):
    rep = AV.stability(AV.ac_json(ac), V, alt, str(C.POLAR_CACHE))
t, d, md = rep["trim"], rep["derivatives"], rep["modes"]
ctrl = t.get("trim_control")

with summary:
    k = st.columns(4)
    k[0].metric("Trim angle of attack", f"{t.get('alpha_trim', t['alpha']):.2f}°")
    k[1].metric(f"{(ctrl or 'control').capitalize()} to trim", f"{t.get('control_trim', 0):+.2f}°")
    k[2].metric("CL, trimmed", f"{t['CL']:.3f}")
    k[3].metric("L/D, trimmed", f"{t['L_D']:.1f}")
    k = st.columns(4)
    sm = d.get("static_margin")
    k[0].metric("Static margin", f"{100 * sm:.1f} % MAC" if sm is not None else "–")
    k[1].metric("Neutral point x", f"{d.get('x_np', float('nan')):.3f} m")
    xm, _ = ac.wing.mac_position()
    k[2].metric("Neutral point", f"{100 * (d['x_np'] - xm) / ac.c:.1f} % MAC")
    k[3].metric("dε/dα at the tail", f"{d.get('deps_da', 0):.3f}", help="Downwash gradient of the wing at the horizontal tail")
    if sm is not None and sm < 0.05:
        st.warning("Static margin under 5 % MAC: the aircraft is close to neutrally stable or unstable "
                   "in pitch. Move the c.g. forward or enlarge the tail.")
    views.trust_notes("stability")
    src = ", ".join(f"{k_}: {v}" for k_, v in rep["polar_source"].items())
    st.caption(f"M {rep['mach']:.3f}, q {rep['qbar']:.0f} Pa. Section data: {src}. "
               f"Fuselage: Cm_α {rep['fuselage']['Cm_a']:+.3f}, Cn_β {rep['fuselage']['Cn_b']:+.4f}, "
               f"CD0 {rep['fuselage']['CD0']:.4f}.")
    AV.top_view(ac, height=300, markers=[("c.g.", ac.cg, ui.INK, "diamond"),
                                         ("NP", (d["x_np"], 0, 0), ui.SERIES[1], "x")])

tabs = st.tabs(["Derivatives", "Flight modes", "Responses", "Trim and c.g. range", "Files"])
with tabs[0]:
    rows_lon = [("CL_α", d["CL_a"], "per rad"), ("CD_α", d["CD_a"], "per rad"), ("Cm_α", d["Cm_a"], "per rad"),
                ("CL_q", d["CL_q"], "per q̂ = qc/2V"), ("Cm_q", d["Cm_q"], "per q̂"),
                ("CL_α̇", d["CL_adot"], "per α̇c/2V"), ("Cm_α̇", d["Cm_adot"], "per α̇c/2V")]
    rows_lat = [("CY_β", d["CY_b"], "per rad"), ("Cl_β", d["Cl_b"], "per rad"), ("Cn_β", d["Cn_b"], "per rad"),
                ("CY_p", d["CY_p"], "per p̂ = pb/2V"), ("Cl_p", d["Cl_p"], "per p̂"), ("Cn_p", d["Cn_p"], "per p̂"),
                ("CY_r", d["CY_r"], "per r̂ = rb/2V"), ("Cl_r", d["Cl_r"], "per r̂"), ("Cn_r", d["Cn_r"], "per r̂")]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Longitudinal")
        st.dataframe(pd.DataFrame(rows_lon, columns=["derivative", "value", "unit"]), hide_index=True,
                     width="stretch", column_config={"value": st.column_config.NumberColumn(format="%.4f")})
        stab = [("Cm_α < 0 (pitch stiffness)", d["Cm_a"] < 0), ("Cm_q < 0 (pitch damping)", d["Cm_q"] < 0)]
    with c2:
        st.markdown("#### Lateral-directional")
        st.dataframe(pd.DataFrame(rows_lat, columns=["derivative", "value", "unit"]), hide_index=True,
                     width="stretch", column_config={"value": st.column_config.NumberColumn(format="%.4f")})
        stab += [("Cn_β > 0 (weathercock)", d["Cn_b"] > 0), ("Cl_β < 0 (dihedral effect)", d["Cl_b"] < 0),
                 ("Cl_p < 0 (roll damping)", d["Cl_p"] < 0), ("Cn_r < 0 (yaw damping)", d["Cn_r"] < 0)]
    st.markdown("  ".join(f":{'green' if ok else 'red'}-badge[{txt}]" for txt, ok in stab))
    st.markdown("#### Control derivatives and hinge moments (per rad)")
    crow = []
    for name, cd in d["controls"].items():
        crow.append({"control": name, **{f"{k}_δ": cd[k] for k in ("CL", "CD", "CY", "Cl", "Cm", "Cn")},
                     "Ch_δ": cd["Ch_d"], "Ch_α": cd["Ch_a"], "Ch at trim": cd["Ch_0"]})
    st.dataframe(pd.DataFrame(crow), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f") for c in
                                ("CL_δ", "CD_δ", "CY_δ", "Cl_δ", "Cm_δ", "Cn_δ", "Ch_δ", "Ch_α", "Ch at trim")})
    st.caption("Stability axes, moments about the c.g. Hinge moments of the right-hand surface "
               "(or the fin), positive trailing edge down, on control area × control chord.")
with tabs[1]:
    names = [m for m in md if not m.startswith("_")]
    mrows = []
    for m in names:
        v = md[m]
        g = lambda key: v[key] if v.get(key) is not None else float("nan")   # blank cells
        mrows.append({"mode": m, "ω_n (rad/s)": g("omega_n"), "ζ": g("zeta"),
                      "period (s)": g("period_s"), "time constant (s)": g("time_constant_s"),
                      "time to half (s)": g("t_half_s"), "time to double (s)": g("t_double_s"),
                      "Level 1": "yes" if v.get("level1") else ("no" if "level1" in v else ""),
                      "criterion": v.get("criterion", "")})
    st.dataframe(pd.DataFrame(mrows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3g") for c in
                                ("ω_n (rad/s)", "ζ", "period (s)", "time constant (s)", "time to half (s)",
                                 "time to double (s)")})
    st.caption("Small-perturbation equations in stability axes (thrust = drag, independent of speed; "
               "Ixz = 0). Level 1 limits: MIL-F-8785C, Class I (small light aircraft), flight phase "
               f"Category B. Inertias Ixx, Iyy, Izz = {', '.join(f'{v:.0f}' for v in rep['reference']['inertia'])} kg m².")
    ev = []
    for key in ("_A_lon", "_A_lat"):
        for e in np.linalg.eigvals(np.array(md[key])):
            ev.append((e.real, e.imag, "longitudinal" if key == "_A_lon" else "lateral"))
    f = go.Figure()
    for grp, sym in (("longitudinal", "x"), ("lateral", "circle-open")):
        pts = [(a, b) for a, b, g in ev if g == grp]
        f.add_scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", name=grp,
                      marker=dict(symbol=sym, size=10))
    f.update_layout(title="Eigenvalues (complex plane)", xaxis_title="real (1/s)", yaxis_title="imaginary (rad/s)",
                    height=380)
    ui.plot(f)
with tabs[2]:
    resp = rep.get("responses", {})
    c1, c2 = st.columns(2)
    if "elevator" in resp:
        e = resp["elevator"]
        f = go.Figure()
        for k_, lab in (("alpha", "Δα (deg)"), ("theta", "Δθ (deg)"), ("q", "q (deg/s)")):
            f.add_scatter(x=e["t"], y=e[k_], name=lab)
        f.update_layout(title=f"Elevator step {e['input_deg']:+g}° (trailing edge up = nose up)",
                        xaxis_title="time (s)", height=360)
        ui.plot(f, container=c1)
    if "aileron" in resp:
        e = resp["aileron"]
        f = go.Figure()
        for k_, lab in (("p", "p (deg/s)"), ("phi", "φ (deg)"), ("beta", "β (deg)")):
            f.add_scatter(x=e["t"], y=e[k_], name=lab)
        f.update_layout(title=f"Aileron step {e['input_deg']:+g}°", xaxis_title="time (s)", height=360)
        ui.plot(f, container=c2)
    if "rudder" in resp:
        e = resp["rudder"]
        f = go.Figure()
        for k_, lab in (("beta", "β (deg)"), ("r", "r (deg/s)"), ("phi", "φ (deg)")):
            f.add_scatter(x=e["t"], y=e[k_], name=lab)
        f.update_layout(title=f"Rudder doublet ±{e['input_deg']:g}°, 1 s each way", xaxis_title="time (s)",
                        height=360)
        ui.plot(f, container=c1)
    c2.caption("Linear small-perturbation responses about the trimmed state, controls held. They show "
               "the modes' shape and damping; large inputs leave the linear range.")
with tabs[3]:
    st.markdown("Trim over a speed range at this mass and c.g.")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1], vertical_alignment="bottom")
    v0 = c1.number_input("From (m/s)", 5.0, 300.0, float(max(25.0, round(0.6 * V))))
    v1 = c2.number_input("To (m/s)", 5.0, 300.0, float(min(300.0, round(1.6 * V))))
    nv = c3.number_input("Points", 3, 30, 9)
    if c4.button("Compute trim curve", width="stretch"):
        rows = []
        for v in np.linspace(v0, v1, int(nv)):
            an = A.Analysis(ac, float(v), alt, C.POLAR_CACHE)
            try:
                r = an.trim(control=ctrl or "elevator")
                rows.append({"V (m/s)": v, "α (deg)": r["alpha_trim"], "elevator (deg)": r["control_trim"],
                             "CL": r["CL"], "CD": r["CD"], "L/D": r["L_D"],
                             "drag (N)": r["CD"] * an.qbar * ac.S, "power (kW)": r["CD"] * an.qbar * ac.S * v / 1e3})
            except Exception:                    # noqa: BLE001
                pass
        st.session_state["trim_curve"] = rows
    rows = st.session_state.get("trim_curve")
    if rows:
        df = pd.DataFrame(rows)
        c1, c2 = st.columns(2)
        f = go.Figure()
        f.add_scatter(x=df["V (m/s)"], y=df["elevator (deg)"], name="elevator")
        f.add_scatter(x=df["V (m/s)"], y=df["α (deg)"], name="α", line=dict(dash="dash"))
        f.update_layout(title="Elevator and angle of attack to trim", xaxis_title="V (m/s)",
                        yaxis_title="deg", height=340)
        ui.plot(f, container=c1)
        f = go.Figure()
        f.add_scatter(x=df["V (m/s)"], y=df["power (kW)"], name="power required")
        f.update_layout(title="Power required, trimmed (airframe drag)", xaxis_title="V (m/s)",
                        yaxis_title="kW", height=340, showlegend=False)
        ui.plot(f, container=c2)
        st.dataframe(df, hide_index=True, width="stretch")
    st.markdown("#### c.g. range")
    xnp = d["x_np"]
    st.markdown(f"Neutral point x = {xnp:.3f} m. For a 5 % MAC minimum static margin the aft c.g. limit is "
                f"**x = {xnp - 0.05 * ac.c:.3f} m** ({100 * (xnp - 0.05 * ac.c - xm) / ac.c:.1f} % MAC); "
                f"the current c.g. is at {100 * (ac.cg[0] - xm) / ac.c:.1f} % MAC.")
with tabs[4]:
    js = json.dumps(rep, indent=1)
    c1, c2 = st.columns(2)
    c1.download_button("Download stability report (JSON)", js, file_name="stability.json", width="stretch")
    flat = {f"{k}": v for k, v in d.items() if isinstance(v, (int, float))}
    for name, cd in d["controls"].items():
        for k, v in cd.items():
            if isinstance(v, (int, float)):
                flat[f"{k}_{name}"] = v
    c2.download_button("Download derivatives (CSV)", pd.Series(flat).to_csv(header=["value"]),
                       file_name="derivatives.csv", width="stretch")
