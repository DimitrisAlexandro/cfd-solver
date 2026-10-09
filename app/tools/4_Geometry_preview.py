import numpy as np
import plotly.graph_objects as go
import streamlit as st

import common as C
import ui
import views

C.page_setup()
ui.header("Geometry check", "Check a section before spending CFD time: shape, thickness, trailing "
                            "edge, inviscid pressure and the CFD mesh.")

c1, c2 = st.columns([5, 6])
with c1:
    spec, name = views.airfoil_picker("geo", allow_flap=True)
if spec is None:
    st.stop()
import airfoil_geom as ag
try:
    loop, info = ag.load_airfoil(spec)
except Exception as e:
    st.error(f"This file can't be read as an airfoil: {e}. Expected two columns x y, Selig or "
             "Lednicer order.")
    st.stop()

c1, c2 = st.columns([3, 1], gap="large")
f = go.Figure()
f.add_scatter(x=loop[:, 0], y=loop[:, 1], mode="lines+markers", marker=dict(size=3),
              name="mesher loop", line=dict(width=1.2))
f.update_layout(yaxis=dict(scaleanchor="x", zeroline=False), xaxis=dict(zeroline=False), height=300,
                title=f"{info['name']}: surface points used by the mesher")
ui.plot(f, container=c1)
with c2:
    st.metric(f"Max thickness, at x/c {info['x_max_thickness']:.3f}", f"{100 * info['max_thickness']:.2f} % c")
    st.metric("Trailing-edge base", f"{100 * info['te_thickness']:.3f} % c")
    st.metric("Surface points", info["points"])

st.markdown("### Inviscid pressure")
a = st.slider("α (deg)", -10.0, 20.0, 4.0, 0.5)
(res,), o = C.panel_polar(spec, [a])
e = res["elements"][0]
c1, c2 = st.columns(2)
f = go.Figure()
f.add_scatter(x=e["x"], y=e["Cp"], mode="lines", name="Cp")
f.update_layout(yaxis=dict(autorange="reversed", title="Cp"), xaxis_title="x/c", height=330,
                showlegend=False, title=f"Panel method at α {a:g}°: Cl {res['Cl']:.4f}, Cm {res['Cm']:+.4f}")
ui.plot(f, container=c1)
al = np.arange(-6, 16.1, 1.0)
rs, _ = C.panel_polar(spec, al)
k = np.polyfit(np.radians(al[(al > -3) & (al < 6)]),
               [r["Cl"] for r, x in zip(rs, al) if -3 < x < 6], 1)
f = go.Figure()
f.add_scatter(x=al, y=[r["Cl"] for r in rs], name="Cl")
f.add_scatter(x=al, y=[r["Cm"] for r in rs], name="Cm, c/4", yaxis="y2", line=dict(dash="dash"))
f.update_layout(height=330, xaxis_title="α (deg)", yaxis_title="Cl",
                yaxis2=dict(overlaying="y", side="right", title="Cm", showgrid=False, ticks="inside"),
                title=f"Lift slope {k[0]:.3f} per rad (2π = 6.283), zero-lift angle "
                      f"{np.degrees(-k[1] / k[0]):.2f}°")
ui.plot(f, container=c2)
st.caption("Inviscid: real lift is lower and there is no stall. Airfoil polar computes both.")

st.markdown("### CFD mesh")
c1, c2, c3, c4 = st.columns([2, 2, 2, 1], vertical_alignment="bottom")
re = c1.number_input("Reynolds number (sets the wall spacing)", 1e3, 5e8, 3e6, format="%.3e")
lev = c2.select_slider("Mesh level", [0.7, 1.0, 1.4], 1.0)
zoom = c3.selectbox("View", ["Whole section", "Leading edge", "Trailing edge", "Far field"])
if c4.button("Build mesh", width="stretch"):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    import polar as pl
    with st.spinner("Meshing..."):
        fvm, mesh, *_ = pl.make_case(spec, re, a, level=lev)
    P = mesh["points"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    lim = {"Whole section": ((-0.2, 1.3), (-0.4, 0.4)), "Leading edge": ((-0.02, 0.06), (-0.03, 0.03)),
           "Trailing edge": ((0.95, 1.05), (-0.025, 0.025)), "Far field": ((-110, 110), (-110, 110))}[zoom]
    ax.add_collection(PolyCollection([P[c] for c in mesh["cells"]], facecolors="none",
                                     edgecolors=ui.INK, linewidths=0.2))
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal")
    ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
    st.pyplot(fig)
    st.caption(f"{fvm.ncells} cells, {mesh.get('n_bl', '?')} in the boundary layer; the wake "
               f"refinement follows α {a:g}°.")
