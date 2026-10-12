import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
import ui

import trust as T

C.page_setup()
ui.header("Confidence", "How far each result can be trusted: the validated envelope of the solver, the error "
                        "found inside it, and a check of any flow condition before you run it.")

left, right = st.columns([4, 7], gap="large")
with left:
    with st.container(border=True):
        ui.step("Check a flow condition", first=True)
        c1, c2 = st.columns(2)
        re = c1.number_input("Reynolds number", 1e3, 5e8, 3e6, format="%.3e")
        mach = c2.number_input("Mach", 0.0, 0.95, 0.0, 0.01)
        c1, c2 = st.columns(2)
        alpha = c1.number_input("α (deg)", -20.0, 30.0, 4.0, 0.5)
        level = c2.select_slider("Mesh level", [0.7, 1.0, 1.4], 1.0)
        tr = st.segmented_control("Transition", ["free", "fixed", "off", "laminar"], default="free",
                                  format_func={"free": "free (e^N)", "fixed": "fixed trip", "off": "fully turbulent",
                                               "laminar": "laminar solver"}.get) or "free"
        c1, c2 = st.columns(2)
        multi = c1.toggle("Multi-element", False)
        pg = c2.toggle("Prandtl–Glauert", False, disabled=mach == 0,
                       help="Incompressible solver + correction instead of the compressible solver")
        tc = st.slider("Thickness t/c (%)", 4.0, 25.0, 12.0, 0.5) / 100
    a = T.assess_point(re, mach, alpha, transition=tr, elements="multi" if multi else "single",
                       prandtl_glauert=pg, refine=level, t_c=tc)
    st.markdown(f"### {ui.trust_md(a.level, a.label)}")
    k = st.columns(2)
    k[0].metric("Expected Cl uncertainty", f"±{100 * a.cl_band:.1f} %")
    k[1].metric("Expected Cd uncertainty", f"±{100 * a.cd_band:.1f} %")
    st.caption(f"Nearest validated case: {a.case_title} (distance {a.distance:.2f}; 0 = inside its range, "
               "1 ≈ a factor 1.8 in Re, 0.05 in Mach or 2° in α away).")
    for l, t in a.reasons:
        st.markdown(f"{ui.trust_md(l, ui.TRUST[l][1])} {t}")

with right:
    st.markdown("#### Validated envelope")
    el = "multi" if multi else "single"
    cases = [c for c in T.CASES if c.elements == el]
    f = go.Figure()
    col = {"green": "rgba(43,122,75,{})", "amber": "rgba(178,106,0,{})", "red": "rgba(156,61,84,{})"}
    for i, c in enumerate(cases):
        lev = "green" if c.cd_err <= 0.10 else "amber" if c.cd_err <= 0.25 else "red"
        match = tr in c.transition and c.mach[0] - 0.05 <= mach <= c.mach[1] + 0.05
        op = 0.22 if match else 0.05
        x0, x1 = math.log10(c.re[0]), math.log10(c.re[1])
        f.add_shape(type="rect", x0=x0, x1=x1, y0=c.alpha[0], y1=c.alpha[1],
                    fillcolor=col[lev].format(op), line=dict(color=col[lev].format(0.9 if match else 0.3), width=1))
        if c.tested:
            f.add_scatter(x=[math.log10(max(t[0], 1)) for t in c.tested if t[0]],
                          y=[t[2] for t in c.tested if t[0]], mode="markers", name=c.title,
                          marker=dict(size=6, color=col[lev].format(1 if match else 0.35), symbol="circle"),
                          hovertemplate=f"{c.title}<br>Cd error ±{100 * c.cd_err:.0f} %<br>"
                                        "Re 10^%{x:.2f}, α %{y:g}°<extra></extra>", showlegend=False)
    f.add_scatter(x=[math.log10(re)], y=[alpha], mode="markers", name="your condition",
                  marker=dict(size=16, symbol="star", color=ui.INK, line=dict(color="#FFFFFF", width=1)),
                  hovertemplate="your condition<extra></extra>")
    f.update_layout(xaxis=dict(title="log₁₀ Re", range=[2.5, 8]), yaxis=dict(title="α (deg)", range=[-8, 24]),
                    height=430, showlegend=False,
                    title=f"{'Multi' if multi else 'Single'}-element cases; bright = same transition mode and Mach")
    ui.plot(f)
    st.caption("Boxes: the range each validated case covers (green: Cd error ≤ 10 %, amber ≤ 25 %, red above). "
               "Dots: the points actually compared with reference data.")

st.markdown("#### Validated cases")
df = pd.DataFrame(T.envelope_table())
st.dataframe(df.drop(columns=["case"]), hide_index=True, width="stretch",
             column_config={"Cl error": st.column_config.NumberColumn(format="percent"),
                            "Cd error": st.column_config.NumberColumn(format="percent"),
                            "note": st.column_config.TextColumn(width="large"),
                            "title": st.column_config.TextColumn(width="medium")})
st.caption("Reports with the full comparisons are on the Validation page (validation/ in the repository).")

st.markdown("#### Other tools")
st.dataframe(pd.DataFrame([{"tool": k.replace("_", " ").capitalize(), "level": ui.TRUST[v[0]][1],
                            "evidence and limits": v[1], "report": v[2]} for k, v in T.EVIDENCE.items()]),
             hide_index=True, width="stretch",
             column_config={"evidence and limits": st.column_config.TextColumn(width="large")})
st.markdown("#### How the level is set")
st.markdown(
    "- **Validated range** (green): inside a case where this code matched the reference within ~10 % in Cd, "
    "converged, attached flow, validated mesh.\n"
    "- **Use with care** (amber): near a validated case, or a known weak spot: past Cl max, shocks, laminar "
    "bubbles, coarse mesh, y+ above 1, unusual thickness, the transition point cycling.\n"
    "- **Low confidence** (red): far from every validated case, no steady state, Prandtl–Glauert above M 0.5, or "
    "a case where the code itself is known to be off (E387 at Re 1e5).\n\n"
    "The uncertainty bars are the error of the nearest case × (1 + distance), plus terms for the mesh level, "
    "stall and convergence. A grid study (Airfoil polar → *Confidence and grid check*) replaces the mesh term "
    "with the measured grid convergence index.")
