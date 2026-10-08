import numpy as np
import plotly.graph_objects as go
import streamlit as st

import common as C
import views

C.page_setup("Geometry & quick look", "📐")
st.title("📐 Geometry & quick look")
st.caption("Check an airfoil before spending CFD time: shape, thickness, trailing edge, "
           "inviscid pressure (panel method, instant) and the CFD mesh.")

spec, name = views.airfoil_picker("geo")
if spec is None:
    st.stop()
import airfoil_geom as ag
try:
    loop, info = ag.load_airfoil(spec)
except Exception as e:
    st.error(f"could not load: {e}")
    st.stop()

c1, c2 = st.columns([2, 1])
f = go.Figure()
f.add_scatter(x=loop[:, 0], y=loop[:, 1], mode="lines+markers", marker=dict(size=3), name="mesher loop")
f.update_layout(yaxis=dict(scaleanchor="x"), height=330, margin=dict(l=10, r=10, t=30, b=10),
                title=info["name"])
c1.plotly_chart(f, width="stretch")
c2.metric("max thickness", f"{100 * info['max_thickness']:.2f} % c", f"at x/c {info['x_max_thickness']:.3f}")
c2.metric("trailing-edge base", f"{100 * info['te_thickness']:.3f} % c")
c2.metric("points", info["points"])

st.subheader("Inviscid pressure (panel method)")
a = st.slider("α (deg)", -10.0, 20.0, 4.0, 0.5)
(res,), o = C.panel_polar(spec, [a])
e = res["elements"][0]
f = go.Figure()
f.add_scatter(x=e["x"], y=e["Cp"], mode="lines", name="Cp")
f.update_yaxes(autorange="reversed", title="Cp"); f.update_xaxes(title="x/c")
f.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10),
                title=f"inviscid Cl {res['Cl']:.4f}, Cm {res['Cm']:+.4f}")
c1, c2 = st.columns(2)
c1.plotly_chart(f, width="stretch")
al = np.arange(-6, 16.1, 1.0)
rs, _ = C.panel_polar(spec, al)
f = go.Figure()
f.add_scatter(x=al, y=[r["Cl"] for r in rs], name="Cl inviscid")
f.add_scatter(x=al, y=[r["Cm"] for r in rs], name="Cm inviscid", yaxis="y2")
k = np.polyfit(np.radians(al[(al > -3) & (al < 6)]),
               [r["Cl"] for r, x in zip(rs, al) if -3 < x < 6], 1)
f.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="α (deg)",
                yaxis2=dict(overlaying="y", side="right", title="Cm"), yaxis_title="Cl",
                title=f"lift slope {k[0]:.3f}/rad (2π = 6.283), α_L0 {np.degrees(-k[1] / k[0]):.2f}°")
c2.plotly_chart(f, width="stretch")
st.caption("No viscosity: real lift is lower and there is no stall — run *Airfoil analysis* for that.")

st.subheader("CFD mesh")
c1, c2, c3 = st.columns(3)
re = c1.number_input("Re (sets the wall spacing)", 1e3, 5e8, 3e6, format="%.3e")
lev = c2.select_slider("mesh level", [0.7, 1.0, 1.4], 1.0)
zoom = c3.selectbox("view", ["airfoil", "leading edge", "trailing edge", "far field"])
if st.button("build mesh"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    import polar as pl
    with st.spinner("meshing..."):
        fvm, mesh, *_ = pl.make_case(spec, re, a, level=lev)
    P = mesh["points"]
    fig, ax = plt.subplots(figsize=(10, 5))
    lim = {"airfoil": ((-0.2, 1.3), (-0.4, 0.4)), "leading edge": ((-0.02, 0.06), (-0.03, 0.03)),
           "trailing edge": ((0.95, 1.05), (-0.025, 0.025)), "far field": ((-110, 110), (-110, 110))}[zoom]
    ax.add_collection(PolyCollection([P[c] for c in mesh["cells"]], facecolors="none",
                                     edgecolors="k", linewidths=0.2))
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal")
    st.pyplot(fig)
    st.caption(f"{fvm.ncells} cells, {mesh.get('n_bl', '?')} in the boundary layer; "
               f"wake refinement follows α = {a:g}°")
