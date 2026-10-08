import streamlit as st

import common as C

C.page_setup("Validation", "✅")
st.title("✅ Validation")
st.caption("How the solver compares with experiments and reference solutions. "
           "Everything here is reproducible with the validate_*.py scripts in src/.")

items = [
    ("Airfoils, low Re (laminar)", ["airfoil_polar.png", "airfoil_distributions.png"], "airfoil_report.txt"),
    ("Flat plate (turbulent BL)", ["flatplate_sa.png"], "flatplate_report.txt"),
    ("NACA 0012, Spalart-Allmaras vs NASA TMR", ["naca0012_sa_polar.png", "naca0012_sa_cp_cf.png"],
     "naca0012_sa_report.txt"),
    ("Transition (e^N) vs experiments", ["phase3_naca0012_polar.png", "phase3_naca2412_polar.png"],
     "transition_report.txt"),
    ("Multi-element 30P30N vs JAXA wind tunnel", ["multi_30p30n_cp.png", "multi_30p30n_grid.png"],
     "multi_30p30n_report.txt"),
    ("Compressible: Ni bump", ["bump_mach.png"], "bump_report.txt"),
    ("Transonic: NACA 0012 Euler, RAE 2822 vs experiment",
     ["transonic_rae2822_cp.png", "transonic_rae2822_mach.png", "transonic_naca0012_euler_cp.png"],
     "transonic_report.txt"),
    ("Wing tool vs theory / DATCOM", ["wing_validation.png", "wing_demo_naca2412.png"], "wing_report.txt"),
    ("Design tools: demos", ["design_drag_rise_naca0012.png", "design_drag_rise_naca0012_fields.png",
                             "design_wing_sizing_demo.png", "design_inverse_rae2822.png",
                             "design_optimize_naca2412.png"], "design_report.md"),
    ("Basic solver: cavity, cylinder, unsteady", ["cavity_profiles.png", "cylinder_flow.png",
                                                  "unsteady_cylinder_forces.png"], "unsteady_report.txt"),
]
for title, imgs, rep in items:
    have = [i for i in imgs if (C.VAL / i).exists()]
    if not have:
        continue
    with st.expander(title):
        for i in have:
            st.image(str(C.VAL / i), caption=i)
        if (C.VAL / rep).exists():
            st.code((C.VAL / rep).read_text()[:4000], language=None)
for md in sorted(C.VAL.glob("phase*_report.md")):
    with st.expander(f"report: {md.name}"):
        st.markdown(md.read_text())
