import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import common as C
import ui
import views

C.page_setup()
ui.header("Wing sizing", "Mission in, wing out. For each planform the area follows from the stall "
                         "speed with that planform's own CL max; designs are ranked by cruise L/D "
                         "with a span limit and a root-first stall check. Wing only: no fuselage, "
                         "tail or flaps.")
from aerolab.design import design_wing_sizing as WS

setup, info = st.columns([5, 6], gap="large")
with setup:
    with st.container(border=True):
        ui.step("Mission", first=True)
        c1, c2 = st.columns(2)
        mass = c1.number_input("Mass (kg)", 0.1, 500000.0, 750.0)
        vs = c2.number_input("Stall speed, sea level (m/s)", 3.0, 120.0, 27.0)
        vc = c1.number_input("Cruise speed (m/s)", 5.0, 300.0, 60.0)
        hc = c2.number_input("Cruise altitude (m)", 0.0, 15000.0, 1500.0, 100.0)
        span_max = c1.number_input("Span limit (m, 0 = none)", 0.0, 100.0, 11.0)
        sweep = c2.number_input("Quarter-chord sweep (deg)", 0.0, 45.0, 0.0)
        ui.step("Section")
        af, aname = views.airfoil_picker("size")
        cached = WS.cached_polars(af, C.POLAR_CACHE, 0.0) if af else []
        use_cached = st.checkbox("Use the cached CFD polars of this section"
                                 + (f" (Re {', '.join(f'{r:.2g}' for r, _ in cached)})" if cached else ""),
                                 value=bool(cached), disabled=not cached)
        with st.expander("Design space"):
            ars = st.text_input("Aspect ratios", "6,8,10,12,14")
            tapers = st.text_input("Taper ratios", "0.35,0.5,0.7,1.0")
            twists = st.text_input("Tip twist (deg)", "0,-2,-4")
        n = len(ars.split(",")) * len(tapers.split(",")) * len(twists.split(","))
        c1, c2 = st.columns(2, vertical_alignment="center")
        go_ = c1.button("Size the wing", type="primary", disabled=af is None, width="stretch")
        c2.caption(f"{n} planforms, about {max(n * 4 / 60, 1):.0f}–{n * 8 / 60:.0f} min"
                   + ("" if use_cached else ", plus two section polars (30–90 min, once)"))
with info:
    if af:
        views.section_plot(af, height=200, title=f"Section: {aname}")
    q = 0.5 * 1.225 * vs ** 2
    st.markdown(
        f"First estimate with CL max 1.5: wing loading {1.5 * q:.0f} N/m², area "
        f"{mass * 9.81 / (1.5 * q):.1f} m². The sizing recomputes the area for each planform with "
        "its own CL max.\n\n"
        "Root-first stall check: stall must start inside 60 % of the semi-span, so the ailerons "
        "keep working.")

if go_:
    out = C.CASES / f"sizing_{aname}_{int(mass)}kg_{int(time.time())}"
    args = ["design_wing_sizing.py", "--mass", mass, "--cruise-speed", vc, "--cruise-alt", hc,
            "--stall-speed", vs, "--airfoil", af, "--sweep", sweep, "--ar", ars, "--taper", tapers,
            "--twist", twists, "--polar-dir", C.POLAR_CACHE, "--out", out]
    if span_max > 0:
        args += ["--span-max", span_max]
    if use_cached:
        args.append("--use-cached")
    st.session_state["size_job"] = C.start_job("sizing", f"Wing sizing {aname}, {mass:g} kg", args, out)


def show(out):
    out = Path(out)
    d = json.loads((out / "sizing.json").read_text())
    rk = d["ranked"]
    if not rk:
        st.error("No planform meets the requirements: either the span limit is too small or the "
                 "cruise CL is above CL max. Raise the span limit or the stall speed.")
        return
    b = rk[0]
    st.markdown("### Best planform")
    k = st.columns(6)
    k[0].metric("Area", f"{b['S']:.2f} m²")
    k[1].metric("Span", f"{b['span']:.2f} m")
    k[2].metric("Chords, root / tip", f"{b['root_chord']:.2f} / {b['tip_chord']:.2f} m")
    k[3].metric(f"Cruise L/D, wing (CL {b['CL_cruise']:.3f})", f"{b['L_D']:.1f}")
    k[4].metric("CL max", f"{b['CLmax']:.3f}")
    k[5].metric("Stall onset", f"{100 * (b['stall_onset_eta'] or 1):.0f} % semi-span")
    st.caption(f"Aspect ratio {b['AR']:g}, taper {b['taper']:g}, tip twist {b['twist']:g}°.")
    df = pd.DataFrame([{"AR": x["AR"], "taper": x["taper"], "twist": x["twist"], "S (m²)": x["S"],
                        "span (m)": x["span"], "root c (m)": x["root_chord"], "tip c (m)": x["tip_chord"],
                        "CL max": x["CLmax"], "CL cruise": x["CL_cruise"], "L/D": x["L_D"],
                        "stall onset (%)": 100 * (x["stall_onset_eta"] or 1),
                        "span ok": x["span_ok"], "root-first stall": x["stall_ok"]}
                       for x in d["designs"]]).sort_values("L/D", ascending=False)
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.2f") for k in
                                ("S (m²)", "span (m)", "root c (m)", "tip c (m)")} |
                 {k: st.column_config.NumberColumn(format="%.3f") for k in ("CL max", "CL cruise")} |
                 {"L/D": st.column_config.NumberColumn(format="%.1f"),
                  "stall onset (%)": st.column_config.NumberColumn(format="%.0f")})
    if (out / "sizing.png").exists():
        st.image(str(out / "sizing.png"))
    if (out / "best_wing" / "wing.json").exists():
        st.markdown("### Best wing at cruise")
        views.wing_results(out / "best_wing")


jid = st.session_state.get("size_job")
if jid:
    st.markdown("### Run")
    views.live_job(jid, "sizing")
    if C.get_job(jid)["status"] == "finished":
        show(C.get_job(jid)["out"])
