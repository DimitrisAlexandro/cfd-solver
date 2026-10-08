import math
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

import common as C
import views

C.page_setup("Airfoil analysis", "🛩️")
st.title("🛩️ Airfoil analysis")
st.caption("Mesh → RANS solve for every angle → polar, Cp/Cf, flow fields. "
           "Every option of the solver is here; defaults are the validated settings.")


with st.container(border=True):
    st.subheader("1 · Geometry")
    spec, name = views.airfoil_picker()
    derot = st.checkbox("derotate (put the chord line on the x axis)", False,
                        help="for .dat files whose chord is not horizontal")

    st.subheader("2 · Flow conditions")
    c1, c2, c3 = st.columns(3)
    how = c1.radio("specify", ["Reynolds number", "speed + chord + altitude"])
    if how == "Reynolds number":
        re = c2.number_input("Re (chord-based)", 1e3, 5e8, 3e6, format="%.3e")
        mach = c3.number_input("Mach", 0.0, 0.95, 0.0, 0.01,
                               help="0 = incompressible. > 0: compressible solver (shocks), "
                                    "or Prandtl-Glauert correction (option below)")
    else:
        V = c2.number_input("speed (m/s)", 1.0, 400.0, 40.0)
        ch = c2.number_input("chord (m)", 0.01, 20.0, 1.0)
        alt = c3.number_input("altitude (m)", 0.0, 15000.0, 0.0, 100.0)
        import wing as W
        rho, mu, a_snd, T, _ = W.isa(alt)
        re, mach = rho * V * ch / mu, V / a_snd
        c3.info(f"Re = {re:.3g}, M = {mach:.3f}")
    comp_model = "compressible solver"
    if mach > 0:
        comp_model = st.radio("compressibility", ["compressible solver", "Prandtl-Glauert correction"],
                              horizontal=True,
                              help="compressible: ideal gas + energy equation, captures shocks "
                                   "(validated on RAE 2822). Prandtl-Glauert: incompressible "
                                   "solution scaled by 1/√(1-M²), only below ~M 0.3")
    if re < 1e5:
        st.info("Re < 1e5 → laminar Navier-Stokes (no turbulence model)")

    st.subheader("3 · Transition (turbulent runs)")
    tr = st.radio("model", ["free (e^N)", "fixed trip", "fully turbulent"], horizontal=True,
                  help="free: laminar boundary layer + e^N envelope method; trips: given x/c")
    c1, c2, c3 = st.columns(3)
    ncrit = c1.number_input("N_crit", 1.0, 14.0, 9.0, 0.5,
                            help="9 = average wind tunnel, 11–12 quiet tunnel / clean flight")
    xu = c2.number_input("trip x/c upper", 0.0, 1.0, 0.05)
    xl = c3.number_input("trip x/c lower", 0.0, 1.0, 0.05)

    st.subheader("4 · Angles of attack")
    c1, c2, c3, c4 = st.columns(4)
    a0 = c1.number_input("from (deg)", -20.0, 30.0, -4.0)
    a1 = c2.number_input("to (deg)", -20.0, 30.0, 16.0)
    da = c3.number_input("step", 0.25, 10.0, 2.0)
    alist = c4.text_input("or list", "", placeholder="0,2,5,10")

    st.subheader("5 · Numerics")
    c1, c2, c3, c4 = st.columns(4)
    level = c1.select_slider("mesh level", [0.7, 1.0, 1.4], 1.0,
                             help="1 = validated mesh; 0.7 coarse/fast (~2× faster); 1.4 fine (grid check)")
    jobs = c2.number_input("parallel processes", 1, 32, 2, help="angles split in blocks")
    solver = c3.selectbox("solver", ["auto", "simple", "coupled"],
                          help="SIMPLEC (default) or coupled u-v-p (needs pypardiso)")
    maxit = c4.number_input("max iterations / angle", 500, 30000, 6000, 500)
    c1, c2 = st.columns(2)
    fields = c1.checkbox("save flow fields (for the field viewer)", True)
    xfoil = c2.checkbox("overlay XFOIL (if installed)", False)
    title = st.text_input("run name", "")
    go = st.button("▶ Run analysis", type="primary")

if go:
    if spec is None:
        st.error("upload an airfoil file first")
        st.stop()
    if alist.strip():
        angles = alist.replace(" ", "")
        n = len(angles.split(","))
    else:
        angles = f"{a0:g}:{a1:g}:{da:g}"
        n = int(math.floor((a1 - a0) / da + 1e-9)) + 1
    slug = (title or f"{name}_Re{re:.2g}" + (f"_M{mach:.2f}" if mach else "")).replace(" ", "_")
    out = C.CASES / slug.replace("+", "")
    args = ["airfoil.py", "polar", spec, "--re", f"{re:.6g}", f"--alpha={angles}",
            "--refine", level, "--jobs", jobs, "--max-iter", maxit, "--out", out,
            "--solver", solver]
    if derot:
        args.append("--derotate")
    if tr.startswith("fixed"):
        args += ["--xtr", f"{xu:g},{xl:g}"]
    elif tr.startswith("fully"):
        args += ["--transition", "off"]
    else:
        args += ["--ncrit", ncrit]
    if mach > 0:
        args += ["--mach", f"{mach:.4g}"]
        if comp_model.startswith("Prandtl"):
            args.append("--prandtl-glauert")
    if not fields:
        args.append("--no-fields")
    if xfoil:
        args.append("--xfoil")
    jid = C.start_job("polar", f"{name} Re {re:.3g}" + (f" M {mach:.2f}" if mach else "")
                      + f" · {tr}", args, out, {"n_angles": n})
    st.session_state["last_polar_job"] = jid
    est = n * (0.6 if level <= 0.7 else 1.2 if level <= 1 else 3.0) * (3 if mach >= 0.5 else 1) / max(jobs, 1)
    st.success(f"started — about {est:.0f}–{3 * est:.0f} min. You can leave this page; "
               "progress is also on *Jobs & results*.")

jid = st.session_state.get("last_polar_job")
if jid:
    st.divider()
    st.subheader("Progress")
    views.live_job(jid, "polar")
    j = C.get_job(jid)
    if j["status"] == "finished":
        st.divider()
        views.polar_results(j["out"])
