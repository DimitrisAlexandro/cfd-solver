import streamlit as st

import common as C
import ui

C.page_setup()
ui.header("Validation", "How the solver compares with wind-tunnel data and reference solutions. "
                        "Each case is reproducible with the validate_*.py scripts in src/.")

items = [
    ("Airfoils at low Re, laminar", ["airfoil_polar.png", "airfoil_distributions.png"], "airfoil_report.txt"),
    ("Flat plate, turbulent boundary layer", ["flatplate_sa.png"], "flatplate_report.txt"),
    ("NACA 0012, Spalart–Allmaras against NASA TMR", ["naca0012_sa_polar.png", "naca0012_sa_cp_cf.png"],
     "naca0012_sa_report.txt"),
    ("Transition (e^N) against experiments", ["phase3_naca0012_polar.png", "phase3_naca2412_polar.png"],
     "transition_report.txt"),
    ("Free transition near stall", ["transition_stall_fix.png"], "transition_stall_fix.md"),
    ("Low-Reynolds-number mode: Eppler 387 at Re 1e5–3e5 against NASA Langley LTPT and XFOIL",
     ["lowre_e387.png"], "lowre_report.md"),
    ("Multi-element 30P30N against the JAXA wind tunnel", ["multi_30p30n_cp.png", "multi_30p30n_grid.png"],
     "multi_30p30n_report.txt"),
    ("Compressible: Ni bump", ["bump_mach.png"], "bump_report.txt"),
    ("Transonic: NACA 0012 Euler, RAE 2822 against experiment",
     ["transonic_rae2822_cp.png", "transonic_rae2822_mach.png", "transonic_naca0012_euler_cp.png"],
     "transonic_report.txt"),
    ("Wing tool against theory and DATCOM", ["wing_validation.png", "wing_demo_naca2412.png"], "wing_report.txt"),
    ("Aircraft tools: stability, database, loads, performance (vs AeroSandbox and theory)",
     ["aircraft_stability_demo.png", "aircraft_database_demo.png", "aircraft_loads_vn_demo.png",
      "aircraft_loads_span_demo.png", "aircraft_performance_demo.png"], "aircraft_report.md"),
    ("Plain-flap sections (CFD)", ["flap_sections_demo.png"], "flap_report.md"),
    ("Aeroelasticity: unsteady lattice vs Theodorsen, Goland wing flutter, aileron mass balance, flexible loads",
     ["aeroelastic_validation.png"], "aeroelastic_report.md"),
    ("Conceptual design: drag buildup, propeller, sizing (vs Cessna 172 and Pipistrel Velis Electro), "
     "weight and balance", ["design_tools_validation.png"], "design_tools_report.md"),
    ("Design tools: demonstration cases", ["design_drag_rise_naca0012.png", "design_drag_rise_naca0012_fields.png",
                                           "design_wing_sizing_demo.png", "design_inverse_rae2822.png",
                                           "design_optimize_naca2412.png"], "design_report.md"),
    ("Trust layer and project graph: confidence levels, grid convergence index, variants",
     ["trust_envelope.png", "trust_polar.png", "projects_graph.png", "projects_compare.png"],
     "trust_and_projects.md"),
    ("Basic solver: cavity, cylinder, unsteady", ["cavity_profiles.png", "cylinder_flow.png",
                                                  "unsteady_cylinder_forces.png"], "unsteady_report.txt"),
    ("Discrete adjoint: gradients vs finite differences, adjoint shape design, error estimation",
     ["adjoint_fd_check.png", "adjoint_design_naca0012.png"], "adjoint_report.md"),
    ("Amplification-factor transport (local transition model) vs e^N and XFOIL",
     ["aft_validation.png"], "aft_report.md"),
    ("Swept-wing transition: attachment line, crossflow, Tollmien-Schlichting",
     ["swept_transition.png"], "swept_transition_report.md"),
    ("Field inversion and machine learning (SA near stall, NACA 0012)",
     ["fiml_naca0012.png"], "fiml_report.md"),
    ("Learned laminar-separation-bubble length (Eppler 387, LTPT)", [], "bubble_ml_report.md"),
    ("Gaussian-process polar surrogate with uncertainty", [], "surrogate_report.md"),
]
for title, imgs, rep in items:
    have = [i for i in imgs if (C.VAL / i).exists()]
    if not have and not (C.VAL / rep).exists():
        continue
    with st.expander(title):
        for i in have:
            st.image(str(C.VAL / i), caption=i)
        p = C.VAL / rep
        if p.exists():
            if p.suffix == ".md":
                st.markdown(p.read_text())
            else:
                st.code(p.read_text()[:4000], language=None)
st.markdown("### Phase reports")
for md in sorted(C.VAL.glob("phase*_report.md")):
    with st.expander(md.stem.replace("_", " ").capitalize()):
        st.markdown(md.read_text())
