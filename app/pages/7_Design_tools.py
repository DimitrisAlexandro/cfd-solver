import streamlit as st

import common as C

C.page_setup("Design tools", "🧪")
st.title("🧪 Design tools")
st.caption("Being built next, one by one — each becomes its own page here.")

tools = [
    ("🎯 Airfoil optimizer", "planned",
     "Goal: max L/D at a cruise Cl, min drag, max Cl_max. Constraints: thickness, Cm, "
     "trailing-edge angle. Shape changes (CST parameters) evaluated with the CFD solver on a "
     "coarse mesh, the winner re-checked on the validated mesh."),
    ("✏️ Inverse design", "planned",
     "Draw / edit the Cp distribution you want; the shape that produces it is found "
     "(panel method, seconds), then checked with CFD."),
    ("📏 Wing sizing wizard", "planned",
     "Weight, cruise speed, altitude, stall speed → span, area, taper, twist suggestions with "
     "cruise L/D, stall speed check and where the stall starts."),
    ("🌀 Drag-rise explorer", "planned",
     "Mach sweep at a fixed lift coefficient with the compressible solver: drag-divergence "
     "Mach, Mach field with the shock at every step, Korn-equation comparison."),
]
for name, status, text in tools:
    with st.container(border=True):
        st.markdown(f"### {name}  ·  _{status}_")
        st.markdown(text)
