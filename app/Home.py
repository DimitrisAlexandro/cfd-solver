"""Overview page (entry point: AeroLab.py)."""
import time

import streamlit as st

import common as C
import ui

C.page_setup()
ui.header("AeroLab", "Airfoil and wing aerodynamics with your own mesher and RANS solver. "
                     "Each analysis runs in the background; results stay in cases/.")

TOOLS = {
    "Analysis": [
        ("tools/1_Airfoil_analysis.py", "Airfoil polar",
         "Cl, Cd, Cm against α; Cp, Cf and flow fields. Free or fixed transition, compressible "
         "with shocks.", "10–30 min"),
        ("tools/2_Multi_element.py", "Multi-element airfoil",
         "Slat, main and flap on one mesh; forces per element.", "5–10 min per angle"),
        ("tools/3_Wing.py", "Wing",
         "Vortex lattice on CFD section polars: CL–α to stall, drag polar, span loading.",
         "seconds with cached polars"),
    ],
    "Aircraft": [
        ("tools/12_Aircraft_model.py", "Aircraft model",
         "Wing, tails, fin, control surfaces, fuselage, mass and c.g., in 3D.", "instant"),
        ("tools/18_Weight_balance.py", "Weight and balance",
         "Component masses, loading diagram, scissor plot for tail sizing, landing-gear checks.", "seconds"),
        ("tools/19_Drag_buildup.py", "Drag buildup",
         "CD0 component by component: friction, form factors, gear, cooling, leakage.", "instant"),
        ("tools/20_Propeller.py", "Propeller and engine",
         "Minimum-induced-loss blade design, BEMT maps, fixed pitch or constant speed; piston, "
         "turboprop or electric.", "seconds"),
        ("tools/13_Stability.py", "Stability and control",
         "Trim, neutral point, all derivatives, hinge moments, flight modes and responses.", "seconds"),
        ("tools/14_Aero_database.py", "Aero database",
         "Coefficient tables over α, β, Mach and controls; JSBSim, AVL, Excel; section polar matrices.",
         "1–3 min (aircraft)"),
        ("tools/15_Loads.py", "Flight loads",
         "V–n diagram, load cases, shear, bending and torsion along the span, Nastran export.",
         "about 30 s"),
        ("tools/16_Performance.py", "Performance",
         "Speeds, climb, ceilings, range, endurance, glide, takeoff and landing distances.", "about 1 min"),
    ],
    "Design": [
        ("tools/17_Conceptual_sizing.py", "Conceptual sizing",
         "Requirements in, aircraft out: constraint diagram, mission fuel or battery, Class I/II weights, "
         "geometry, gear and propulsion, verified with the performance tool.", "seconds"),
        ("tools/5_Inverse_design.py", "Inverse design",
         "Pressure distribution in, airfoil shape out.", "seconds"),
        ("tools/6_Optimizer.py", "Shape optimizer",
         "Minimum drag at your design Cl, with thickness and Cm limits.", "about 1 h"),
        ("tools/7_Wing_sizing.py", "Wing sizing",
         "Mass, speeds and span limit in; area, aspect ratio, taper and twist out.",
         "about 5 min"),
        ("tools/8_Drag_rise.py", "Drag rise",
         "Mach sweep at fixed lift: drag-divergence Mach, shock growth.", "about 45 min"),
    ],
    "Data": [
        ("tools/4_Geometry_preview.py", "Geometry check",
         "Thickness, trailing edge, inviscid Cp and the CFD mesh before a run.", "instant"),
        ("tools/9_Polar_library.py", "Polar library", "Overlay and download every polar computed.", ""),
        ("tools/10_Jobs_and_results.py", "Runs", "Progress, logs and results of every run.", ""),
        ("tools/11_Validation.py", "Validation",
         "Comparisons with wind-tunnel data and reference codes.", ""),
    ],
}

jobs = C.list_jobs()
left, right = st.columns([7, 4], gap="large")
with left:
    for group, rows in TOOLS.items():
        st.markdown(f"#### {group}")
        for page, title, text, t in rows:
            c1, c2, c3 = st.columns([2.1, 5, 1.6], vertical_alignment="center")
            c1.page_link(page, label=title)
            c2.markdown(text)
            if t:
                c3.caption(t)
with right:
    k = st.columns(3)
    k[0].metric("Running", sum(j["status"] == "running" for j in jobs))
    k[1].metric("Finished", sum(j["status"] == "finished" for j in jobs))
    k[2].metric("Saved polars", len(C.scan_polars()))
    st.markdown("#### Recent runs")
    if not jobs:
        st.caption("No runs yet. Start with an airfoil polar; NACA 2412 at Re 3e6 takes about "
                   "15 minutes on two cores.")
    for j in jobs[:8]:
        when = time.strftime("%d %b %H:%M", time.localtime(j["started"]))
        st.markdown(f"{ui.status_md(j['status'])} {j['title']}  \n"
                    f"<span style='color:{ui.GRAPHITE};font-size:0.8rem'>{when}</span>",
                    unsafe_allow_html=True)
    if jobs:
        st.page_link("tools/10_Jobs_and_results.py", label="All runs")
