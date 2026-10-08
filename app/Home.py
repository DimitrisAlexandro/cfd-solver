"""AeroLab - interactive front end of the cfd-solver project.

    cd ~/cfd-solver && streamlit run app/Home.py
"""
import streamlit as st

import common as C

C.page_setup("Home")

st.title("✈️ AeroLab")
st.caption("Your own mesher + finite-volume RANS solver, from 2D airfoils to wings — "
           "choose an analysis on the left or below.")

jobs = C.list_jobs()
running = [j for j in jobs if j["status"] == "running"]
c1, c2, c3 = st.columns(3)
c1.metric("Running jobs", len(running))
c2.metric("Saved polars", len(C.scan_polars()))
c3.metric("Finished runs", sum(j["status"] == "finished" for j in jobs))

st.subheader("Analyses")
cards = [
    ("pages/1_Airfoil_analysis.py", "🛩️ Airfoil analysis",
     "Polar of any airfoil (NACA, TMR 0012, RAE 2822, your .dat). Spalart-Allmaras RANS, "
     "free e^N / fixed / no transition, incompressible or **compressible (shocks)**, "
     "laminar below Re 1e5, mesh level, solver choice. Cp, Cf, flow fields."),
    ("pages/2_Multi_element.py", "🪶 Multi-element airfoils",
     "Slat + main + flap (30P30N presets or your own elements and deflections), "
     "per-element forces, inviscid panel-method preview."),
    ("pages/3_Wing.py", "🛫 Wing (quasi-3D)",
     "Vortex lattice + your CFD section polars: taper, sweep, dihedral, twist, root/tip "
     "airfoils. C_L–α with stall, drag polar, C_m, span loading, stalled strips."),
    ("pages/4_Geometry_preview.py", "📐 Geometry & quick look",
     "Load any airfoil, check thickness / trailing edge, see the mesh, inviscid Cp in a "
     "second (panel method)."),
    ("pages/5_Polar_library.py", "📚 Polar library",
     "Every polar you computed — browse, overlay, compare, download."),
    ("pages/6_Jobs_and_results.py", "⏱️ Jobs & results",
     "Live progress of running analyses, logs, cancel; open finished results."),
    ("pages/7_Design_tools.py", "🧪 Design tools",
     "Optimizer, inverse design, wing sizing wizard, drag-rise explorer."),
    ("pages/8_Validation.py", "✅ Validation",
     "How the solver compares with experiments and references (credibility)."),
]
cols = st.columns(2)
for i, (page, title, text) in enumerate(cards):
    with cols[i % 2].container(border=True):
        st.page_link(page, label=title)
        st.markdown(text)

if running:
    st.subheader("Running now")
    for j in running:
        st.write(f"• **{j['title']}** — started {int((__import__('time').time() - j['started']) / 60)} min ago")
