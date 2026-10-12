import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import aircraft_views as AV
import common as C
import ui

import aeroelastic as AE
import aircraft as A
import wing as W
import wing_structure as WS

C.page_setup()
ui.header("Aeroelasticity", "Wing modes, 3D unsteady aerodynamics and p-k flutter including aileron flutter; "
                            "divergence, aileron reversal and elastic lift on the vortex lattice.")

ac, path = AV.aircraft_picker(key="aeac")
if not getattr(ac, "structure", None):
    st.info("This aircraft has no wing structure saved yet. On the Wing structure page, size the box from a "
            "Flight loads run and press **Save structure with the aircraft**; then come back here.")
    st.page_link("tools/21_Wing_structure.py", label="Go to Wing structure")
    st.stop()

stc = WS.from_aircraft(ac)
ad0 = dict(stc.get("aileron_dynamics") or {})
has_ail = any(c.name == "aileron" for c in ac.wing.controls)
loads_vd = None
for f in sorted(C.CASES.glob("loads_*/loads.json"), key=lambda p: -p.stat().st_mtime):
    try:
        d = json.loads(f.read_text())
    except Exception:                                   # noqa: BLE001
        continue
    if d.get("aircraft") == ac.name:
        loads_vd = d["vn"]["Vd"]
        break

setup, out = st.columns([4, 7], gap="large")
with setup:
    with st.container(border=True):
        ui.step("Flight condition", first=True)
        c1, c2 = st.columns(2)
        alt = c1.number_input("Altitude (m)", 0.0, 12000.0, 0.0, 250.0,
                              help="Flutter speed is in true airspeed at this density; check the altitudes of the envelope")
        VD = c2.number_input("V_D, EAS (m/s)", 10.0, 400.0, float(loads_vd or 85.0),
                             help="From the latest Flight loads run of this aircraft" if loads_vd else None)
        vmax = st.number_input("Highest speed analysed (m/s)", 20.0, 600.0, float(round(2.0 * (loads_vd or 85.0))), 5.0)
        fuel = st.number_input("Fuel in the wing (kg)", 0.0, 1e5, 0.0,
                               help="Fuel lowers the frequencies and moves the mass: check full and empty tanks")
        ui.step("Aileron (control-surface flutter)")
        use_ail = st.checkbox("Aileron rotation as a degree of freedom", value=has_ail, disabled=not has_ail)
        c1, c2 = st.columns(2)
        f_b = c1.number_input("Aileron frequency, uncoupled (Hz)", 1.0, 100.0, float(ad0.get("freq_hz", 15.0)), 0.5,
                              help="Rotation about the hinge against the control-circuit stiffness, wing held fixed. "
                                   "Light aircraft: 10–25 Hz; a slack circuit is much lower")
        mpa = c2.number_input("Aileron mass (kg/m²)", 0.5, 30.0, float(ad0.get("mass_per_area", 4.0)), 0.5)
        c1, c2 = st.columns(2)
        xcg = c1.number_input("Aileron c.g. aft of the hinge (fraction of aileron chord)", -0.5, 0.6,
                              float(ad0.get("cg_aft_hinge", 0.30)), 0.01,
                              help="0.3–0.4 unbalanced; 0 statically balanced; negative = overbalanced (mass ahead of the hinge)")
        kga = c2.number_input("Aileron radius of gyration (fraction of its chord)", 0.1, 0.6,
                              float(ad0.get("gyration", 0.30)), 0.01)
        ui.step("Model")
        c1, c2, c3 = st.columns(3)
        nm = c1.number_input("Modes", 3, 12, 6)
        ns = c2.number_input("Spanwise panels / side", 6, 30, 12)
        nc = c3.number_input("Chordwise panels", 4, 20, 8)
        go_ = st.button("Analyse", type="primary", width="stretch")

ail = {"freq_hz": f_b, "mass_per_area": mpa, "cg_aft_hinge": xcg, "gyration": kga} if use_ail else None
rho = W.isa(alt)[0]
key = json.dumps({"ac": ac.to_dict(), "ail": ail, "alt": alt, "vmax": vmax, "fuel": fuel, "nm": nm, "ns": ns, "nc": nc},
                 sort_keys=True, default=str)


@st.cache_data(show_spinner=False, max_entries=8)
def compute(key):
    k = json.loads(key)
    a = A.Aircraft.from_dict(k["ac"])
    stc_ = WS.from_aircraft(a)
    res, beam, lat = AE.analyse(a, stc_, k["ail"], Vmax=k["vmax"], rho=W.isa(k["alt"])[0], n_modes=k["nm"],
                                ns=k["ns"], nc=k["nc"], fuel_mass=k["fuel"], log=lambda *x: None)
    strip = AE.strip_flutter(a, stc_, k["vmax"], W.isa(k["alt"])[0], k["nm"], fuel_mass=k["fuel"])
    return A._jsonable(res), A._jsonable(strip)


if go_:
    st.session_state["ae_key"] = key
if st.session_state.get("ae_key") != key:
    with out:
        st.markdown(
            "Press **Analyse** (a few seconds).\n\n"
            "1. **Modes** of the wing from the saved structure (EI, GJ, mass, inertia) with the aileron as a rigid "
            "surface on a hinge spring.\n"
            "2. **3D unsteady aerodynamics**: vortex-ring lattice with an oscillating wake, for every mode and "
            "reduced frequency (symmetric and antisymmetric).\n"
            "3. **p-k flutter**: damping and frequency of every mode against speed; flutter where the damping "
            "crosses zero. CS-23.629: no flutter up to 1.2 V_D.\n"
            "4. **Static aeroelasticity** on the same lattice: divergence, aileron reversal, elastic lift.")
    st.stop()

with st.spinner("Modes, unsteady aerodynamics, p-k"):
    res, strip = compute(key)

VF = res["flutter_speed"]
VDt = VD * math.sqrt(1.225 / rho)                      # V_D as true airspeed at this altitude
req = 1.2 * VDt
with out:
    k = st.columns(4)
    fp = res["flutter_points"][0] if res["flutter_points"] else None
    k[0].metric("Flutter speed (TAS)", f"{VF:.0f} m/s" if math.isfinite(VF) else f"none below {vmax:.0f} m/s",
                f"{'✓' if VF >= req else '✗'} 1.2 V_D = {req:.0f} m/s", delta_color="off", delta_arrow="off")
    k[1].metric("Flutter frequency", f"{fp['f_hz']:.1f} Hz" if fp else "–",
                f"{fp['family']}" if fp else "", delta_color="off", delta_arrow="off")
    dv = res["divergence"]["V_eas"]
    k[2].metric("Divergence, EAS", f"{dv:.0f} m/s" if math.isfinite(dv) else "none",
                f"{'✓' if dv >= 1.2 * VD else '✗'} vs 1.2 V_D", delta_color="off", delta_arrow="off")
    rv = res.get("reversal")
    k[3].metric("Aileron reversal, EAS", (f"{rv['V_eas']:.0f} m/s" if math.isfinite(rv["V_eas"]) else
                                         f"above {max(rv['V_curve']):.0f} m/s") if rv else "–")
    if fp and VF < req:
        st.error(f"Flutter at {VF:.0f} m/s ({fp['family']}, {fp['f_hz']:.1f} Hz) is inside 1.2 V_D. "
                 + ("Mass-balance the aileron (c.g. at or ahead of the hinge) or stiffen the control circuit; "
                    if ail and ail["cg_aft_hinge"] > 0 else "Raise the torsional stiffness or move mass forward; ")
                 + "see the mass-balance tab.")
    fr = pd.DataFrame([{"mode": i + 1, "type": m["kind"], "f (Hz)": f}
                       for i, (m, f) in enumerate(zip(res["modes"], res["frequencies_hz"]))])
    st.dataframe(fr, hide_index=True, width="stretch", column_config={"f (Hz)": st.column_config.NumberColumn(format="%.2f")})


def vg_plot(fam, key_):
    d = res[fam]
    V = np.array(d["V"]); G = np.array(d["g"]); F = np.array(d["f"])
    c1, c2 = st.columns(2)
    f1, f2 = go.Figure(), go.Figure()
    for j in range(G.shape[1]):
        lab = f"{j + 1}: {res['modes'][j]['kind']}"
        f1.add_scatter(x=V, y=G[:, j], name=lab, line=dict(color=ui.SERIES[j % 7]))
        f2.add_scatter(x=V, y=F[:, j], name=lab, line=dict(color=ui.SERIES[j % 7]), showlegend=False)
    for f in (f1, f2):
        f.add_vline(x=req, line=dict(color=ui.GRAPHITE, dash="dot", width=1), annotation_text="1.2 V_D",
                    annotation_font_size=10)
    f1.add_hline(y=0, line=dict(color=ui.INK, width=1))
    for p in d["flutter"]:
        f1.add_scatter(x=[p["V"]], y=[0], mode="markers", marker=dict(size=10, color=ui.SERIES[1], symbol="x"),
                       showlegend=False, hovertext=f"flutter {p['V']:.1f} m/s, {p['f_hz']:.2f} Hz")
    f1.update_layout(title=f"Damping, {fam} modes", xaxis_title="V, TAS (m/s)", yaxis_title="g = 2ζ",
                     yaxis_range=[-0.6, 0.3], height=380)
    f2.update_layout(title="Frequency", xaxis_title="V, TAS (m/s)", yaxis_title="Hz", height=380)
    ui.plot(f1, container=c1, key=f"vg{key_}")
    ui.plot(f2, container=c2, key=f"vf{key_}")


tabs = st.tabs(["Flutter, symmetric", "Flutter, antisymmetric", "Mode shapes", "Static aeroelasticity",
                "Aileron mass balance", "3D vs strip theory", "Method and validation"])
with tabs[0]:
    vg_plot("symmetric", "s")
with tabs[1]:
    vg_plot("antisymmetric", "a")
    st.caption("Antisymmetric: left and right wing move in opposite directions (the ailerons too). The wing root is "
               "clamped in both families; the fuselage roll and pitch freedom is not modelled.")
with tabs[2]:
    j = st.selectbox("Mode", range(len(res["modes"])),
                     format_func=lambda i: f"{i + 1}: {res['modes'][i]['kind']}, {res['frequencies_hz'][i]:.2f} Hz")
    m = res["modes"][j]
    c1, c2 = st.columns([2, 3])
    f = go.Figure()
    f.add_scatter(x=m["s"], y=m["w"], name="bending w (normalised)")
    f.add_scatter(x=m["s"], y=np.array(m["theta"]), name="twist θ (rad per unit w)", yaxis="y2")
    f.update_layout(title="Along the elastic axis", xaxis_title="s (m)", height=360,
                    yaxis2=dict(overlaying="y", side="right", title="θ"))
    ui.plot(f, container=c1, key="msh")
    mg = res["mode_grid"]
    X, Y, Z0 = (np.array(mg[k]) for k in ("x", "y", "z"))
    fam_shape = c2.segmented_control("Family", ["symmetric", "antisymmetric"], default="symmetric", key="mfam") or "symmetric"
    zz = np.array(mg[fam_shape][j])
    span = np.ptp(Y)
    sc = 0.12 * span
    fig = go.Figure()
    fig.add_trace(go.Surface(x=X, y=Y, z=Z0 + sc * zz, surfacecolor=zz, colorscale="RdBu_r", cmin=-1, cmax=1,
                             showscale=False, hoverinfo="skip", lighting=dict(ambient=0.75)))
    fig.add_trace(go.Surface(x=X, y=Y, z=Z0, colorscale=[[0, "#D9DEE4"], [1, "#D9DEE4"]], showscale=False,
                             opacity=0.35, hoverinfo="skip"))
    ext = [np.ptp(X), np.ptp(Y)]
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                      scene=dict(aspectmode="manual", aspectratio=dict(x=ext[0] / ext[1] * 2.5, y=2.5, z=0.6),
                                 camera=dict(eye=dict(x=-1.6, y=-1.3, z=1.0)), xaxis_title="x", yaxis_title="y",
                                 zaxis_title="z"))
    c2.plotly_chart(fig, width="stretch", theme=None, key="m3d")
    c2.caption("Deformed shape, exaggerated; grey = undeformed. Colour: displacement (red up, blue down).")
with tabs[3]:
    c1, c2 = st.columns(2)
    el = res["elastic_lift"]
    f = go.Figure()
    f.add_scatter(x=el["V_eas"], y=100 * np.array(el["ratio"]), name="lift due to α")
    if rv:
        f.add_scatter(x=rv["V_curve"], y=100 * np.array(rv["effectiveness"]), name="roll due to aileron")
    f.add_hline(y=100, line=dict(color=ui.GRAPHITE, width=1, dash="dot"))
    f.add_hline(y=0, line=dict(color=ui.INK, width=1))
    f.add_vline(x=VD, line=dict(color=ui.GRAPHITE, dash="dot", width=1), annotation_text="V_D", annotation_font_size=10)
    f.update_layout(title="Elastic / rigid", xaxis_title="V, EAS (m/s)", yaxis_title="%", height=380)
    ui.plot(f, container=c1, key="stat")
    c2.markdown(
        f"**Divergence** (symmetric): {'%.0f m/s EAS' % dv if math.isfinite(dv) else 'none'} — the elastic axis is "
        f"behind the aerodynamic centre, so lift twists the wing nose up; the elastic lift rises and runs away at "
        f"divergence. Sweep-back (elastic axis sweep {res['sweep_ea_deg']:.1f}°) works against it.\n\n"
        "**Aileron reversal** (antisymmetric): the aileron's lift twists the wing nose down, which cancels its "
        "rolling moment as the speed rises; at reversal the roll power is zero.\n\n"
        "Both on the vortex lattice (3D, sweep included). The Flight loads page can use the same flexible wing for "
        "every load case (option *Flexible wing*).")
with tabs[4]:
    st.caption("Flutter speed as the aileron c.g. moves (fraction of the aileron chord aft of the hinge). About 1 s "
               "per point.")
    c1, c2 = st.columns([1, 3])
    lo = c1.number_input("From", -0.5, 0.5, -0.15, 0.05)
    hi = c1.number_input("To", -0.4, 0.6, 0.40, 0.05)
    npts = c1.number_input("Points", 3, 15, 8)
    if c1.button("Compute sweep", disabled=not ail):
        with st.spinner("Flutter for every c.g. position"):
            sw = AE.mass_balance_sweep(ac, stc, ail, np.linspace(lo, hi, int(npts)), vmax, rho,
                                       n_modes=int(nm), ns=int(ns), nc=int(nc), fuel_mass=fuel)
        st.session_state["ae_sweep"] = (key, sw)
    got = st.session_state.get("ae_sweep")
    if got and got[0] == key:
        sw = sorted(got[1], key=lambda p: p["cg_aft_hinge"])
        x = np.array([p["cg_aft_hinge"] for p in sw])
        y = np.array([p["V_flutter"] for p in sw])
        fin = np.isfinite(y)
        f = go.Figure()
        f.add_scatter(x=x[fin], y=y[fin], mode="lines+markers", name="flutter speed")
        if (~fin).any():
            f.add_scatter(x=x[~fin], y=np.full((~fin).sum(), vmax), mode="markers", name=f"no flutter below {vmax:.0f} m/s",
                          marker=dict(symbol="triangle-up", size=10, color=ui.SERIES[2]))
        f.add_hline(y=req, line=dict(color=ui.SERIES[1], dash="dash", width=1), annotation_text="1.2 V_D",
                    annotation_font_size=10)
        f.update_layout(title="Flutter speed against aileron balance", xaxis_title="aileron c.g. aft of hinge (c_a)",
                        yaxis_title="V flutter, TAS (m/s)", height=380)
        ui.plot(f, container=c2, key="mbal")
        safe = np.where(fin, y, np.inf) >= req
        if safe[0]:
            i = int(np.argmin(safe)) if not safe.all() else len(safe)
            if i == len(safe):
                st.success("Flutter-free to 1.2 V_D over the whole range.")
            else:
                ya, yb = (y[i - 1] if np.isfinite(y[i - 1]) else 2 * req), y[i]
                xl = x[i - 1] + (req - ya) / (yb - ya) * (x[i] - x[i - 1]) if yb != ya else x[i - 1]
                st.success(f"Flutter-free to 1.2 V_D with the aileron c.g. at or ahead of **{xl:+.2f}** of its chord "
                           "(aft of the hinge). Design with margin: aim for static balance (c.g. on the hinge) or "
                           "better.")
                if safe[i:].any():
                    st.caption("Further aft the flutter speed rises again: that region is not a robust design point "
                               "(it depends on the circuit stiffness and the fuel state).")
        else:
            st.warning("Not flutter-free even with the most forward c.g. in the range: stiffen the circuit (higher "
                       "aileron frequency) or the wing in torsion.")

with tabs[5]:
    sp = strip["flutter"]
    st.markdown(f"Same structure, **strip theory** (Theodorsen on every strip, no aileron degree of freedom): flutter "
                f"{'%.0f m/s at %.1f Hz' % (sp[0]['V'], sp[0]['f_hz']) if sp else 'none in the range'}; **3D lattice**: "
                f"{'%.0f m/s' % VF if math.isfinite(VF) else 'none in the range'}. Strip theory ignores the tip loss "
                "and usually predicts flutter earlier (conservative) on a clean wing; it cannot see aileron flutter.")
    V = np.array(strip["V"]); G = np.array(strip["g"])
    f = go.Figure()
    for j in range(G.shape[1]):
        f.add_scatter(x=V, y=G[:, j], name=f"mode {j + 1}", line=dict(color=ui.SERIES[j % 7]))
    f.add_hline(y=0, line=dict(color=ui.INK, width=1))
    f.update_layout(title="Damping, strip theory (symmetric, no aileron)", xaxis_title="V (m/s)", yaxis_title="g",
                    yaxis_range=[-0.6, 0.3], height=360)
    ui.plot(f, key="strip")
with tabs[6]:
    st.markdown(
        "**Structure.** Half wing as a beam on a straight elastic axis (sweep included), clamped at the symmetry "
        "plane: Hermite bending and linear torsion elements with EI, GJ, mass, static unbalance and inertia from the "
        "saved wing structure. The aileron is a rigid surface on a hinge spring; its mass, inertia and unbalance "
        "couple exactly to wing bending and torsion.\n\n"
        "**Aerodynamics.** Vortex-ring lattice on the full-span wing with a wake of rings shed at the trailing edge: "
        "for harmonic motion the ring shed m steps ago carries the trailing-edge circulation with the phase lag "
        "e^(−iωmΔt). Loads from Kutta–Joukowski on the bound segments plus ρ iω Γ A. Generalised forces Q(k) for the "
        "symmetric and antisymmetric families.\n\n"
        "**Flutter.** p-k method (as Nastran's PK): for every mode at every speed, the reduced frequency is matched "
        "to the eigenvalue's frequency. Damping g = 2ζ; flutter where g crosses zero.\n\n"
        "**Static.** Same lattice at k = 0 and same beam: (K − q Q₀) u = q f. Divergence and reversal from it; the "
        "flexible-loads option trims the whole aircraft with the elastic wing incidence θ cos Λ − w′ sin Λ.\n\n"
        "**Validation** (tests): beam frequencies and static deflections exact; the oscillating lattice converges to "
        "Theodorsen's 2-D lift for plunge and pitch (1–3 % at 16 chordwise panels); steady lattice = the aircraft "
        "VLM; p-k with strip theory reproduces the typical-section flutter (2.18) and the **Goland wing**, 137 m/s / "
        "70 rad/s at sea level (Goland 1945: 137.2 m/s, 70.7 rad/s); 3D aerodynamics raises it to ~165 m/s (tip loss); "
        "an unbalanced aileron flutters inside the envelope and mass balance removes it.\n\n"
        "**Limits.** Incompressible (M < 0.3); clamped root (no fuselage freedom); wing only (no tail, engine, "
        "tab flutter); linear structure and aerodynamics; no structural damping (conservative). A screening tool — "
        "certification needs measured modes (ground vibration test) and a flutter analysis on them.")
