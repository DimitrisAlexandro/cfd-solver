import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import common as C
import ui
import views

C.page_setup()
ui.header("Airfoil polar", "Mesh, solve every angle with Spalart–Allmaras RANS, and get the polar, "
                           "surface distributions and flow fields. Defaults are the validated settings.")

setup, preview = st.columns([5, 6], gap="large")
with setup:
    with st.container(border=True):
        ui.step("Section", first=True)
        spec, name = views.airfoil_picker(allow_flap=True)
        derot = st.checkbox("Derotate: put the chord line on the x axis", False,
                            help="For .dat files whose chord isn't horizontal")

        ui.step("Flow")
        how = st.segmented_control("Specify by", ["Re and Mach", "Speed, chord, altitude"],
                                   default="Re and Mach", label_visibility="collapsed")
        c1, c2, c3 = st.columns(3)
        if how == "Speed, chord, altitude":
            V = c1.number_input("Speed (m/s)", 1.0, 400.0, 40.0)
            ch = c2.number_input("Chord (m)", 0.01, 20.0, 1.0)
            alt = c3.number_input("Altitude (m)", 0.0, 15000.0, 0.0, 100.0)
            import wing as W
            rho, mu, a_snd, T, _ = W.isa(alt)
            re, mach = rho * V * ch / mu, V / a_snd
            st.caption(f"Re {C.fmt_re(re)}, M {mach:.3f} (ISA)")
        else:
            re = c1.number_input("Reynolds number", 1e3, 5e8, 3e6, format="%.3e")
            mach = c2.number_input("Mach", 0.0, 0.95, 0.0, 0.01,
                                   help="0 runs incompressible. Above 0: compressible solver or "
                                        "Prandtl–Glauert correction, chosen below.")
        comp_model = "compressible solver"
        if mach > 0:
            comp_model = st.radio("Compressibility", ["compressible solver", "Prandtl–Glauert correction"],
                                  horizontal=True,
                                  help="Compressible: ideal gas and energy equation, captures shocks "
                                       "(validated on RAE 2822). Prandtl–Glauert scales the "
                                       "incompressible result by 1/√(1−M²); use it below M 0.3.")
        regime = "auto"
        if re < 2e4:
            st.caption("Below Re 2e4 the flow is solved laminar, without a turbulence model.")
        elif re < 5e5:
            st.caption("Low-Re mode: laminar separation bubbles with separation-induced transition. "
                       "Eppler 387 against NASA Langley: drag within 6–9 % on average at Re 2e5–3e5, "
                       "transition points within ~0.03 c of XFOIL. Near Re 1e5 a mid-chord bubble can "
                       "burst in steady RANS (flagged 'unsteady', drag too high): check those angles "
                       "against XFOIL or tests.")
            if re < 1e5:
                regime = "laminar" if st.checkbox(
                    "Solve laminar instead (no turbulence model)", False,
                    help="The previous behaviour below Re 1e5. Steady laminar flow cannot reattach a "
                         "separated layer; use it for very clean, low-Re cases only.") else "auto"

        ui.step("Transition")
        tr = st.segmented_control("Transition model", ["free (e^N)", "free (AFT transport)", "fixed trip",
                                                       "fully turbulent"],
                                  default="free (e^N)", label_visibility="collapsed",
                                  help="Free: laminar boundary layer with the e^N envelope method. "
                                       "AFT transport: the same envelope criterion as local transport "
                                       "equations on the CFD mesh (no surface marching; no separation "
                                       "bubbles - use e^N at low Re). Fixed: tripped at the given x/c.")
        tr = tr or "free (e^N)"
        c1, c2, c3 = st.columns(3)
        ncrit = c1.number_input("N_crit", 1.0, 14.0, 9.0, 0.5, disabled=not tr.startswith("free"),
                                help="9 for an average wind tunnel, 11–12 for a quiet tunnel or clean flight")
        tu = c1.number_input("or turbulence level Tu (%)", 0.0, 3.0, 0.0, 0.01,
                             disabled=not tr.startswith("free"),
                             help="> 0: N_crit from Mack's relation N_crit = -8.43 - 2.4 ln(Tu) "
                                  "(overrides N_crit). 0.07 % ~ N_crit 9.")
        xu = c2.number_input("Trip x/c, upper", 0.0, 1.0, 0.05, disabled=not tr.startswith("fixed"))
        xl = c3.number_input("Trip x/c, lower", 0.0, 1.0, 0.05, disabled=not tr.startswith("fixed"))

        ui.step("Angles of attack (deg)")
        c1, c2, c3 = st.columns(3)
        a0 = c1.number_input("From", -20.0, 30.0, -4.0)
        a1 = c2.number_input("To", -20.0, 30.0, 16.0)
        da = c3.number_input("Step", 0.25, 10.0, 2.0)
        alist = st.text_input("Or a list", "", placeholder="0, 2, 5, 10")

        with st.expander("Numerics and output"):
            c1, c2 = st.columns(2)
            level = c1.select_slider("Mesh level", [0.7, 1.0, 1.4], 1.0,
                                     help="1 is the validated mesh; 0.7 is about twice as fast; "
                                          "1.4 for a grid check")
            jobs = c2.number_input("Parallel processes", 1, 32, 2, help="Angles are split into blocks")
            solver = c1.selectbox("Solver", ["auto", "simple", "coupled"],
                                  help="SIMPLEC (default) or coupled u–v–p (needs pypardiso)")
            maxit = c2.number_input("Max iterations per angle", 500, 30000, 6000, 500)
            fields = st.checkbox("Save flow fields", True)
            xfoil = st.checkbox("Overlay XFOIL, if installed", False)
            title = st.text_input("Run name", "", placeholder="default: airfoil and Re")

        if alist.strip():
            angles = alist.replace(" ", "")
            n = len(angles.split(","))
        else:
            angles = f"{a0:g}:{a1:g}:{da:g}"
            n = max(int(math.floor((a1 - a0) / da + 1e-9)) + 1, 0)
        pf = None
        if spec is not None:
            pf = views.prerun_polar(spec, re, angles, mach, "fixed" if tr.startswith("fixed") else
                                    "off" if tr.startswith("fully") else "free",
                                    f"{xu:g},{xl:g}" if tr.startswith("fixed") else None, level, jobs,
                                    mach > 0 and comp_model.startswith("Prandtl"), regime, ncrit)
        blocked = bool(pf) and pf["level"] == "error"
        override = blocked and st.checkbox("Run anyway", False, help="The checks found an error. Run only if "
                                                                      "you know why it doesn't apply.")
        c1, c2 = st.columns([1, 1], vertical_alignment="center")
        go_ = c1.button("Run polar", type="primary", width="stretch",
                        disabled=spec is None or n < 1 or (blocked and not override))
        rt = (pf or {}).get("runtime")
        c2.caption(f"{n} angles, about {C.fmt_duration(rt['expected_s'])} "
                   f"({C.fmt_duration(rt['low_s'])}–{C.fmt_duration(rt['high_s'])})" if rt else f"{n} angles")

with preview:
    if spec is not None:
        views.section_plot(spec, height=210, title=C.display_name(name))
        pa = st.slider("Preview α (deg)", -10.0, 20.0, 4.0, 0.5,
                       help="Inviscid panel method, instant. The CFD run adds viscosity, transition and stall.")
        try:
            (pr,), _ = C.panel_polar(spec, [pa])
            e = pr["elements"][0]
            f = go.Figure()
            f.add_scatter(x=e["x"], y=e["Cp"], mode="lines", name="inviscid Cp",
                          hovertemplate="x/c %{x:.3f}<br>Cp %{y:.3f}<extra></extra>")
            f.update_layout(title=f"Inviscid pressure at α {pa:g}°: Cl {pr['Cl']:.3f}, Cm {pr['Cm']:+.3f}",
                            yaxis=dict(autorange="reversed", title="Cp"), xaxis_title="x/c", height=300,
                            showlegend=False)
            ui.plot(f)
        except Exception as ex:
            st.caption(f"No inviscid preview for this section: {ex}")
    else:
        st.info("Upload a .dat file to see the section and its pressure distribution.")

if go_:
    slug = (title or f"{name}_Re{re:.2g}" + (f"_M{mach:.2f}" if mach else "")).replace(" ", "_")
    out = C.CASES / slug.replace("+", "")
    args = ["airfoil.py", "polar", spec, "--re", f"{re:.6g}", f"--alpha={angles}",
            "--refine", level, "--jobs", jobs, "--max-iter", maxit, "--out", out,
            "--solver", solver]
    if derot:
        args.append("--derotate")
    if tr.startswith("fixed"):
        args += ["--xtr", f"{xu:g},{xl:g}"]
    elif tr.startswith("fully"):
        args += ["--transition", "off"]
    else:
        args += ["--ncrit", ncrit]
        if "AFT" in tr:
            args += ["--transition", "aft"]
        if tu > 0:
            args += ["--tu", f"{tu:g}"]
    if regime != "auto":
        args += ["--regime", regime]
    if mach > 0:
        args += ["--mach", f"{mach:.4g}"]
        if comp_model.startswith("Prandtl"):
            args.append("--prandtl-glauert")
    if not fields:
        args.append("--no-fields")
    if xfoil:
        args.append("--xfoil")
    jid = C.start_job("polar", f"{name} Re {C.fmt_re(re)}" + (f" M {mach:.2f}" if mach else "")
                      + f", {tr}", args, out, {"n_angles": n})
    st.session_state["last_polar_job"] = jid

jid = st.session_state.get("last_polar_job")
if jid:
    st.markdown("### Run")
    views.live_job(jid, "polar")
    j = C.get_job(jid)
    if j["status"] == "finished":
        views.polar_results(j["out"])
