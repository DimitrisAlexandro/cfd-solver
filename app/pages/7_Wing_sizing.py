import json
from pathlib import Path

import pandas as pd
import streamlit as st

import common as C
import views

C.page_setup("Wing sizing wizard", "📏")
st.title("📏 Wing sizing wizard")
st.caption("Mission in → wing out. For every planform (aspect ratio × taper × washout) the area "
           "comes from the stall speed with that planform's own C_L,max (quasi-3D wing tool + "
           "your CFD polars); designs are ranked by cruise L/D, with the span limit and a "
           "root-first-stall check. Wing only (no fuselage / tail / flaps).")

c1, c2, c3 = st.columns(3)
mass = c1.number_input("mass (kg)", 0.1, 500000.0, 750.0)
vc = c2.number_input("cruise speed (m/s)", 5.0, 300.0, 60.0)
hc = c3.number_input("cruise altitude (m)", 0.0, 15000.0, 1500.0, 100.0)
vs = c1.number_input("stall speed, sea level (m/s)", 3.0, 120.0, 27.0)
span_max = c2.number_input("max span (m, 0 = none)", 0.0, 100.0, 11.0)
sweep = c3.number_input("quarter-chord sweep (deg)", 0.0, 45.0, 0.0)
st.subheader("Airfoil")
af, aname = views.airfoil_picker("size")
import design_wing_sizing as WS
import wing as W
cached = WS.cached_polars(af, C.POLAR_CACHE, 0.0) if af else []
use_cached = st.checkbox(f"use the polars already computed for this airfoil ({len(cached)} cached: "
                         + ", ".join(f"Re {r:.2g}" for r, _ in cached) + ")", value=bool(cached),
                         disabled=not cached)
if not use_cached:
    st.warning("two CFD section polars will be computed first at the mission Reynolds numbers "
               "(~30–90 min, once)")
with st.expander("design space", expanded=False):
    ars = st.text_input("aspect ratios", "6,8,10,12,14")
    tapers = st.text_input("taper ratios", "0.35,0.5,0.7,1.0")
    twists = st.text_input("tip twist (deg)", "0,-2,-4")
n = len(ars.split(",")) * len(tapers.split(",")) * len(twists.split(","))
st.caption(f"{n} planforms, ~{n * 4 / 60:.0f}–{n * 8 / 60:.0f} min once the polars exist")
if st.button("▶ Size the wing", type="primary", disabled=af is None):
    out = C.CASES / f"sizing_{aname}_{int(mass)}kg_{int(__import__('time').time())}"
    args = ["design_wing_sizing.py", "--mass", mass, "--cruise-speed", vc, "--cruise-alt", hc,
            "--stall-speed", vs, "--airfoil", af, "--sweep", sweep, "--ar", ars, "--taper", tapers,
            "--twist", twists, "--polar-dir", C.POLAR_CACHE, "--out", out]
    if span_max > 0:
        args += ["--span-max", span_max]
    if use_cached:
        args.append("--use-cached")
    st.session_state["size_job"] = C.start_job("sizing", f"wing sizing {aname} {mass:g} kg", args, out)


def show(out):
    out = Path(out)
    d = json.loads((out / "sizing.json").read_text())
    rk = d["ranked"]
    if not rk:
        st.error("no planform meets the requirements (span limit / cruise C_L above C_L,max)")
        return
    b = rk[0]
    st.success(f"**Best:** S {b['S']:.2f} m², span {b['span']:.2f} m, chords {b['root_chord']:.2f} / "
               f"{b['tip_chord']:.2f} m (AR {b['AR']:g}, taper {b['taper']:g}, twist {b['twist']:g}°) → "
               f"cruise L/D {b['L_D']:.1f} at C_L {b['CL_cruise']:.3f}, C_L,max {b['CLmax']:.3f}, "
               f"stall starts at {100 * (b['stall_onset_eta'] or 1):.0f} % semi-span")
    df = pd.DataFrame([{"AR": x["AR"], "taper": x["taper"], "twist": x["twist"], "S m²": x["S"],
                        "span m": x["span"], "root c": x["root_chord"], "tip c": x["tip_chord"],
                        "CLmax": x["CLmax"], "CL cruise": x["CL_cruise"], "L/D": x["L_D"],
                        "stall onset %": 100 * (x["stall_onset_eta"] or 1),
                        "span ok": x["span_ok"], "root-first stall": x["stall_ok"]}
                       for x in d["designs"]]).sort_values("L/D", ascending=False)
    st.dataframe(df, hide_index=True, width="stretch")
    if (out / "sizing.png").exists():
        st.image(str(out / "sizing.png"))
    if (out / "best_wing" / "wing.json").exists():
        st.subheader("Best wing, cruise conditions")
        views.wing_results(out / "best_wing")


jid = st.session_state.get("size_job")
if jid:
    views.live_job(jid, "sizing")
    if C.get_job(jid)["status"] == "finished":
        show(C.get_job(jid)["out"])
