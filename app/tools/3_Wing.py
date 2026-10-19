import math
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import common as C
import ui
import views

C.page_setup()
ui.header("Wing", "Vortex lattice with section polars from the CFD solver: taper, sweep, dihedral, "
                  "twist, root and tip sections, stall. Polars are computed once and cached; after "
                  "that a wing takes seconds.")
from aerolab.aircraft import wing as W

setup, preview = st.columns([5, 6], gap="large")
with setup:
    with st.container(border=True):
        ui.step("Planform", first=True)
        c1, c2, c3 = st.columns(3)
        span = c1.number_input("Span (m)", 0.1, 100.0, 10.0)
        cr = c2.number_input("Root chord (m)", 0.01, 20.0, 1.6)
        ct = c3.number_input("Tip chord (m)", 0.0, 20.0, 0.8)
        c1, c2 = st.columns(2)
        sweep = c1.number_input("Quarter-chord sweep (deg)", -30.0, 60.0, 5.0, 0.5)
        dih = c2.number_input("Dihedral (deg)", -10.0, 15.0, 3.0, 0.5)
        twist = c1.number_input("Tip twist (deg, negative = washout)", -8.0, 4.0, -2.0, 0.25)
        inc = c2.number_input("Root incidence (deg)", -5.0, 8.0, 0.0, 0.25)

        ui.step("Sections")
        root, rname = views.airfoil_picker("root", "Root section")
        diff_tip = st.checkbox("Different tip section")
        tip, tname = views.airfoil_picker("tip", "Tip section") if diff_tip else (None, None)

        ui.step("Flight condition")
        c1, c2 = st.columns(2)
        V = c1.number_input("Speed (m/s)", 1.0, 300.0, 40.0)
        alt = c2.number_input("Altitude (m)", 0.0, 15000.0, 0.0, 100.0)

        with st.expander("Analysis options"):
            c1, c2 = st.columns(2)
            alpha = c1.text_input("Wing angles (deg)", "-4:20:1")
            palpha = c2.text_input("Section-polar angles (deg)", "-6:22:2",
                                   help="Must reach past stall to find CL max")
            tr = c1.selectbox("Section transition", ["free", "off"],
                              format_func=lambda s: {"free": "free (e^N)", "off": "fully turbulent"}[s])
            lev = c2.select_slider("CFD mesh level", [0.7, 1.0, 1.4], 1.0)
            jobs = c1.number_input("Parallel processes", 1, 32, 2)
            strips = c2.number_input("Strips per half span", 10, 100, 40)

wg = W.Wing(span, cr, ct, sweep, dih, twist, inc, n_half=30)
rho, mu, a_snd, T, _ = W.isa(alt)
mach = V / a_snd
re_per_m = rho * V / mu
re_list = sorted({float(f"{re_per_m * cr:.2g}"), float(f"{re_per_m * ct:.2g}")}) if ct > 0 \
    else [float(f"{re_per_m * cr:.2g}")]
need = []
for af in [root] + ([tip] if tip else []):
    if af is None:
        continue
    for re in re_list:
        slug = f"{Path(str(af)).stem}_Re{re:.3g}_M{mach:.2f}_L{lev:g}_{tr}".replace("+", "")
        if not (C.POLAR_CACHE / f"{slug}.json").exists():
            need.append(f"{Path(str(af)).stem} at Re {C.fmt_re(re)}")

with preview:
    s = span / 2
    tl = math.tan(math.radians(sweep))
    ys = np.array([-s, 0, s])
    cs = np.array([ct, cr, ct])
    xle = np.abs(ys) * tl - 0.25 * cs + 0.25 * cr
    f = go.Figure()
    f.add_scatter(x=np.r_[ys, ys[::-1], ys[0]], y=-np.r_[xle, (xle + cs)[::-1], xle[0]],
                  fill="toself", fillcolor="rgba(29,78,137,0.06)", mode="lines", name="planform",
                  hoverinfo="skip")
    f.add_scatter(x=ys, y=-(xle + 0.25 * cs), mode="lines", line=dict(dash="dot", color=ui.GRAPHITE),
                  name="quarter-chord line")
    f.update_layout(yaxis=dict(scaleanchor="x", zeroline=False, title="x (m)"),
                    xaxis=dict(title="y (m)", zeroline=False), height=300, title="Planform, top view")
    ui.plot(f)
    k = st.columns(4)
    k[0].metric("Area", f"{wg.S:.3g} m²")
    k[1].metric("Aspect ratio", f"{wg.AR:.2f}")
    k[2].metric("Taper", f"{ct / cr:.2f}")
    k[3].metric("MAC", f"{wg.mac:.3g} m")
    k = st.columns(4)
    k[0].metric("Re, MAC", C.fmt_re(rho * V * wg.mac / mu))
    k[1].metric("Mach", f"{mach:.3f}")
    k[2].metric("Density", f"{rho:.3f} kg/m³")
    k[3].metric("q", f"{0.5 * rho * V ** 2:.0f} Pa")
    if need:
        st.warning("These section polars have to be computed first, about 15–60 min each, once: "
                   + ", ".join(need) + ".")
    else:
        st.caption("All section polars are cached; the wing solves in seconds.")
    go_ = st.button("Run wing analysis", type="primary", disabled=root is None)

if go_:
    out = C.CASES / f"wing_{Path(str(rname)).stem}_b{span:g}_{int(time.time())}"
    args = ["wing.py", "--airfoil", root, "--span", span, "--root-chord", cr, "--tip-chord", ct,
            "--sweep", sweep, "--dihedral", dih, "--twist", twist, "--incidence", inc,
            "--speed", V, "--altitude", alt, f"--alpha={alpha}", f"--polar-alpha={palpha}",
            "--transition", tr, "--refine", lev, "--jobs", jobs, "--strips", strips,
            "--polar-dir", C.POLAR_CACHE, "--out", out]
    if tip:
        args += ["--tip-airfoil", tip]
    st.session_state["last_wing_job"] = C.start_job("wing", f"Wing {rname}, span {span:g} m", args, out)
jid = st.session_state.get("last_wing_job")
if jid:
    st.markdown("### Run")
    views.live_job(jid, "wing")
    j = C.get_job(jid)
    if j["status"] == "finished":
        views.wing_results(j["out"])
