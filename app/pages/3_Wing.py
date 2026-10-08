import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import common as C
import views

C.page_setup("Wing", "🛫")
st.title("🛫 Wing analysis (quasi-3D)")
st.caption("Vortex lattice + section polars from your CFD solver: taper, sweep, dihedral, "
           "twist, root/tip airfoils, stall. Polars are computed once and cached; after that "
           "a wing takes seconds.")
import wing as W

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Planform")
    span = st.number_input("span b (m)", 0.1, 100.0, 10.0)
    cr = st.number_input("root chord (m)", 0.01, 20.0, 1.6)
    ct = st.number_input("tip chord (m)", 0.0, 20.0, 0.8)
    sweep = st.slider("quarter-chord sweep (deg)", -30.0, 60.0, 5.0, 0.5)
    dih = st.slider("dihedral (deg)", -10.0, 15.0, 3.0, 0.5)
    twist = st.slider("tip twist (deg, − = washout)", -8.0, 4.0, -2.0, 0.25)
    inc = st.slider("root incidence (deg)", -5.0, 8.0, 0.0, 0.25)
wg = W.Wing(span, cr, ct, sweep, dih, twist, inc, n_half=30)
with c2:
    s = span / 2
    tl = math.tan(math.radians(sweep))
    ys = np.array([-s, 0, s])
    cs = np.array([ct, cr, ct])
    xle = np.abs(ys) * tl - 0.25 * cs + 0.25 * cr
    f = go.Figure()
    f.add_scatter(x=np.r_[ys, ys[::-1], ys[0]], y=-np.r_[xle, (xle + cs)[::-1], xle[0]],
                  fill="toself", mode="lines", name="planform")
    f.add_scatter(x=ys, y=-(xle + 0.25 * cs), mode="lines", line=dict(dash="dot"), name="c/4")
    f.update_layout(yaxis=dict(scaleanchor="x"), height=330, margin=dict(l=10, r=10, t=30, b=10),
                    title=f"S {wg.S:.3g} m², AR {wg.AR:.2f}, taper {ct / cr:.2f}, MAC {wg.mac:.3g} m")
    st.plotly_chart(f, width="stretch")

st.subheader("Sections")
c1, c2 = st.columns(2)
with c1:
    root, rname = views.airfoil_picker("root")
with c2:
    diff_tip = st.checkbox("different tip airfoil")
    tip, tname = views.airfoil_picker("tip") if diff_tip else (None, None)

st.subheader("Flight condition")
c1, c2, c3, c4 = st.columns(4)
V = c1.number_input("speed (m/s)", 1.0, 300.0, 40.0)
alt = c2.number_input("altitude (m)", 0.0, 15000.0, 0.0, 100.0)
rho, mu, a_snd, T, _ = W.isa(alt)
c3.metric("Re (MAC)", f"{rho * V * wg.mac / mu:.3g}")
c4.metric("Mach", f"{V / a_snd:.3f}")

with st.expander("analysis options", expanded=False):
    c1, c2, c3 = st.columns(3)
    alpha = c1.text_input("wing angles (deg)", "-4:20:1")
    palpha = c2.text_input("section-polar angles (deg)", "-6:22:2",
                           help="must reach past stall for C_L,max")
    tr = c3.selectbox("section transition", ["free", "off"])
    lev = c1.select_slider("CFD mesh level", [0.7, 1.0, 1.4], 1.0)
    jobs = c2.number_input("parallel processes", 1, 32, 2)
    strips = c3.number_input("strips per half span", 10, 100, 40)

# which polars already exist?
re_per_m = rho * V / mu
re_list = sorted({float(f"{re_per_m * cr:.2g}"), float(f"{re_per_m * ct:.2g}")}) if ct > 0 else [float(f"{re_per_m * cr:.2g}")]
mach = V / a_snd
need = []
for af in [root] + ([tip] if tip else []):
    if af is None:
        continue
    from pathlib import Path
    for re in re_list:
        slug = f"{Path(str(af)).stem}_Re{re:.3g}_M{mach:.2f}_L{lev:g}_{tr}".replace("+", "")
        if not (C.POLAR_CACHE / f"{slug}.json").exists():
            need.append(f"{Path(str(af)).stem} @ Re {re:.3g}")
if need:
    st.warning("CFD section polars to compute first (~15–60 min each, once): " + ", ".join(need))
else:
    st.success("all section polars are cached → results in seconds")

if st.button("▶ Analyse wing", type="primary", disabled=root is None):
    out = C.CASES / f"wing_{Path(str(rname)).stem}_b{span:g}_{int(__import__('time').time())}"
    args = ["wing.py", "--airfoil", root, "--span", span, "--root-chord", cr, "--tip-chord", ct,
            "--sweep", sweep, "--dihedral", dih, "--twist", twist, "--incidence", inc,
            "--speed", V, "--altitude", alt, f"--alpha={alpha}", f"--polar-alpha={palpha}",
            "--transition", tr, "--refine", lev, "--jobs", jobs, "--strips", strips,
            "--polar-dir", C.POLAR_CACHE, "--out", out]
    if tip:
        args += ["--tip-airfoil", tip]
    st.session_state["last_wing_job"] = C.start_job("wing", f"wing {rname} b {span:g} m", args, out)
jid = st.session_state.get("last_wing_job")
if jid:
    views.live_job(jid, "wing")
    j = C.get_job(jid)
    if j["status"] == "finished":
        views.wing_results(j["out"])
