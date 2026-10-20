"""Overview page (entry point: AeroLab.py)."""
import time

import streamlit as st

import common as C
import ui

C.page_setup()
ui.header("AeroLab", "Airfoil and wing aerodynamics with your own mesher and RANS solver. "
                     "Each analysis runs in the background; results stay in cases/.")

GUIDE = [
    ("Choose or design the airfoil", "tools/30_Airfoil_polar.py", "Airfoil polar",
     "run the polar of a candidate airfoil at your Reynolds number (first: Polar estimate for a quick look; "
     "Airfoil optimiser to improve one)"),
    ("Size the aircraft", "tools/33_Sizing.py", "Sizing",
     "requirements in, a first aircraft out; or import your own on Aircraft model (STL, AVL)"),
    ("Refine the model", "tools/12_Aircraft_model.py", "Aircraft model",
     "surfaces, controls, fuselage; then Weight and balance, Drag buildup, Propeller and engine"),
    ("Check it flies", "tools/13_Stability.py", "Stability and control",
     "trim, static margin, flight modes; then Performance"),
    ("Check it holds", "tools/15_Loads.py", "Flight loads",
     "V–n diagram and wing loads; then Wing structure and Aeroelasticity"),
    ("Confirm with CFD", "tools/27_Wing_3D.py", "Wing 3D CFD",
     "3D RANS of the wing next to the fast method the aircraft pages use"),
]

TOOLS = {
    "Airfoil": [
        ("tools/30_Airfoil_polar.py", "Airfoil polar",
         "Cl, Cd, Cm against α with RANS; Cp, Cf and flow fields. Free or fixed transition, low-Re bubbles, "
         "compressible. Also: check the geometry and mesh first.", "10–30 min"),
        ("tools/31_Airfoil_optimiser.py", "Airfoil optimiser",
         "Minimum drag at your design lift, with thickness and Cm limits: fast (adjoint) or thorough (RANS "
         "search).", "10 min – 1 h"),
        ("tools/32_Polar_estimate.py", "Polar estimate",
         "Instant Cl and Cd of any airfoil from every polar computed so far, with an uncertainty band.", "instant"),
    ],
    "Wing": [
        ("tools/3_Wing.py", "Wing",
         "Vortex lattice on CFD section polars: CL–α to stall, drag polar, span loading.",
         "seconds with cached polars"),
        ("tools/27_Wing_3D.py", "Wing 3D CFD",
         "3D RANS of a half wing, from a saved aircraft or entered: CL, CD, Cm and span loading.", "1–3 h per angle"),
    ],
    "Aircraft (in this order)": [
        ("tools/33_Sizing.py", "Sizing",
         "Whole aircraft from the requirements (constraint diagram, weights, mission), or the wing only.",
         "seconds – 5 min"),
        ("tools/12_Aircraft_model.py", "Aircraft model",
         "Wing, tails, fin, controls, fuselage, mass and c.g., in 3D; import STL or AVL.", "instant"),
        ("tools/18_Weight_balance.py", "Weight and balance",
         "Component masses, loading diagram, scissor plot for tail sizing, landing gear.", "seconds"),
        ("tools/19_Drag_buildup.py", "Drag buildup",
         "CD0 component by component: friction, form factors, gear, cooling, leakage.", "instant"),
        ("tools/20_Propeller.py", "Propeller and engine",
         "Propeller design and maps; piston, turboprop or electric.", "seconds"),
        ("tools/13_Stability.py", "Stability and control",
         "Trim, neutral point, derivatives, hinge moments, flight modes and responses.", "seconds"),
        ("tools/16_Performance.py", "Performance",
         "Speeds, climb, ceilings, range, endurance, glide, takeoff and landing.", "about 1 min"),
    ],
    "Structure": [
        ("tools/15_Loads.py", "Flight loads", "V–n diagram, load cases, shear, bending and torsion along the span.",
         "about 30 s"),
        ("tools/21_Wing_structure.py", "Wing structure",
         "Wing box from the loads: mass, stiffness, deflection, divergence, flutter estimate.", "instant"),
        ("tools/22_Aeroelasticity.py", "Aeroelasticity", "Wing modes and p-k flutter, divergence, reversal.",
         "seconds"),
    ],
    "Results and trust": [
        ("tools/34_Results.py", "Results", "Every run, every polar and every flow field in one place.", ""),
        ("tools/35_Trust.py", "Validation and confidence",
         "How far a result can be trusted, and the solver against wind-tunnel data.", ""),
        ("tools/23_Projects.py", "Projects",
         "A design as a dependency graph: see what is out of date, recompute only that, compare variants "
         "side by side.", ""),
    ],
    "Advanced": [
        ("tools/25_Research.py", "Research",
         "Shape sensitivity (adjoint), mesh accuracy, swept-wing transition, machine learning.", ""),
        ("tools/8_Drag_rise.py", "Drag rise", "Transonic: drag-divergence Mach at fixed lift.", "about 45 min"),
        ("tools/2_Multi_element.py", "Multi-element airfoil", "Slat, main and flap on one mesh.",
         "5–10 min per angle"),
        ("tools/5_Inverse_design.py", "Inverse design", "Pressure distribution in, airfoil shape out.", "seconds"),
        ("tools/14_Aero_database.py", "Aero database",
         "Coefficient tables for flight simulators (JSBSim, AVL, Excel).", "1–3 min"),
    ],
}

jobs = C.list_jobs()
left, right = st.columns([7, 4], gap="large")
with left:
    st.markdown("#### Design an aircraft, step by step")
    for i, (what, page, title, text) in enumerate(GUIDE, 1):
        c1, c2 = st.columns([2.1, 6.6], vertical_alignment="center")
        c1.markdown(f"**{i}. {what}**")
        c2.page_link(page, label=f"{title}: {text}")
    st.caption("Each aircraft page ends with a link to the next step. Everything runs in the background: "
               "follow it in the sidebar or on Results.")
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
    k[0].metric("Running", sum(j["status"] == "running" for j in jobs),
                delta=f"{sum(j['status'] == 'queued' for j in jobs)} queued", delta_color="off")
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
        st.page_link("tools/34_Results.py", label="All runs", query_params={"view": "Runs"})
