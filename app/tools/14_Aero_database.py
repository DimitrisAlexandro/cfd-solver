import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import aircraft_views as AV
import common as C
import ui
import views

C.page_setup()
ui.header("Aero database", "Complete coefficient tables for flight simulation, performance and "
                           "control-law work: the whole aircraft over α, β, Mach and control "
                           "deflections, or section polars from the CFD solver over Reynolds and "
                           "Mach numbers.")

tab_ac, tab_sec = st.tabs(["Aircraft tables", "Section polars (CFD)"])

# ----------------------------------------------------------------------
with tab_ac:
    setup, info = st.columns([5, 6], gap="large")
    with setup:
        with st.container(border=True):
            ui.step("Aircraft", first=True)
            ac, path = AV.aircraft_picker(key="dbac")
            ui.step("Ranges")
            c1, c2 = st.columns(2)
            alphas = c1.text_input("α (deg)", "-6:24:1", help="from:to:step or a list")
            betas = c2.text_input("β (deg)", "-10:10:2")
            c1, c2 = st.columns(2)
            machs = c1.text_input("Mach", "0.15", help="several values give 2-D tables (α, Mach)")
            alt = c2.number_input("Altitude (m, sets the Reynolds number)", 0.0, 15000.0, 1000.0, 100.0)
            ui.step("Control deflections (deg)")
            ctrls = list(ac.controls())
            defaults = {"elevator": "-20,-10,10,20", "aileron": "-15,15", "rudder": "-20,20", "flap": "10,20,30"}
            defl = {}
            cols = st.columns(2)
            for i, n in enumerate(ctrls):
                defl[n] = cols[i % 2].text_input(n.capitalize(), defaults.get(n, "-10,10"), key=f"dbd{n}")
            nonlin = st.checkbox("Nonlinear sections: stall from the CFD polars", True)
            n_a = C.count_angles(alphas)
            n_b = C.count_angles(betas)
            n_m = C.count_angles(machs)
            n_d = sum(C.count_angles(v) for v in defl.values() if v.strip())
            n_states = n_m * n_a * (n_b + n_d)
            c1, c2 = st.columns(2, vertical_alignment="center")
            go_ = c1.button("Build database", type="primary", width="stretch")
            c2.caption(f"{n_states} states, about {max(1, n_states * 0.1 / 60):.0f}–{max(2, n_states * 0.25 / 60):.0f} min")
    with info:
        AV.view3d(ac, height=300)
        st.markdown(
            "Outputs:\n"
            "- **database.xlsx**: one sheet per coefficient (rows α, columns β), control increments, "
            "derivatives against α, the long table\n"
            "- **jsbsim_aero.xml**: a complete `<aerodynamics>` section for JSBSim: CL, CD, Cm tables over "
            "α (and Mach), sideslip, control and damping terms\n"
            "- **aircraft.avl** + **aircraft.mass**: the same geometry for AVL\n"
            "- database.csv / .json, derivatives_vs_alpha.csv")
    if go_:
        out = C.CASES / f"db_{path.stem}_{int(time.time())}"
        args = ["aero_database.py", "aircraft", path, f"--alpha={alphas.replace(' ', '')}",
                f"--beta={betas.replace(' ', '')}", f"--mach={machs.replace(' ', '')}", "--alt", alt,
                "--polar-dir", C.POLAR_CACHE, "--out", out]
        dd = [f"{n}={v.replace(' ', '')}" for n, v in defl.items() if v.strip()]
        if dd:
            args += ["--deflect"] + dd
        if not nonlin:
            args.append("--linear")
        st.session_state["db_job"] = C.start_job("aero_db", f"Aero database {ac.name}", args, out)

    def show_db(out):
        out = Path(out)
        if not (out / "database.csv").exists():
            return
        df = pd.read_csv(out / "database.csv").fillna({"control": ""})
        M = sorted(df.mach.unique())
        m = st.selectbox("Mach", M, key=f"dbm{out}") if len(M) > 1 else M[0]
        d = df[df.mach == m]
        clean = d[d.control == ""]
        c1, c2 = st.columns(2)
        f = go.Figure()
        for b, sub in clean.groupby("beta"):
            if b == 0 or len(clean.beta.unique()) == 1:
                f.add_scatter(x=sub.alpha, y=sub.CL, name=f"β {b:g}°")
        f.update_layout(title="Lift", xaxis_title="α (deg)", yaxis_title="CL", height=330)
        ui.plot(f, container=c1, key=f"dbcl{out}")
        f = go.Figure()
        b0 = 0.0 if 0.0 in clean.beta.values else clean.beta.iloc[0]
        f.add_scatter(x=clean[clean.beta == b0].alpha, y=clean[clean.beta == b0].Cm, name="clean")
        for dv, sub in d[d.control == "elevator"].groupby("deflection"):
            f.add_scatter(x=sub.alpha, y=sub.Cm, name=f"elevator {dv:g}°", line=dict(width=1))
        f.update_layout(title="Pitching moment about the c.g.", xaxis_title="α (deg)", yaxis_title="Cm",
                        height=330)
        ui.plot(f, container=c2, key=f"dbcm{out}")
        a_sel = st.select_slider("α for the sideslip plots (deg)", sorted(clean.alpha.unique()),
                                 value=min(clean.alpha.unique(), key=lambda a: abs(a - 2)), key=f"dba{out}")
        cb = clean[clean.alpha == a_sel]
        c1, c2, c3 = st.columns(3)
        for cc, k, lab in ((c1, "CY", "Side force"), (c2, "Cl", "Rolling moment"), (c3, "Cn", "Yawing moment")):
            f = go.Figure()
            f.add_scatter(x=cb.beta, y=cb[k], name="clean")
            f.update_layout(title=lab, xaxis_title="β (deg)", yaxis_title=f"{k}, stability axes", height=300,
                            showlegend=False)
            ui.plot(f, container=cc, key=f"db{k}{out}")
        views.downloads(out, ("database.xlsx", "jsbsim_aero.xml", "aircraft.avl", "aircraft.mass"), "dba")
        views.downloads(out, ("database.csv", "derivatives_vs_alpha.csv", "database.json"), "dbb")
        if (out / "database.png").exists():
            with st.expander("Overview figure"):
                st.image(str(out / "database.png"))

    jid = st.session_state.get("db_job")
    if jid:
        st.markdown("### Run")
        views.live_job(jid, "aero_db")
        if C.get_job(jid)["status"] == "finished":
            show_db(C.get_job(jid)["out"])
    olds = sorted(C.CASES.glob("db_*/database_done.json"), key=lambda p: -p.stat().st_mtime)
    if olds:
        st.markdown("### Earlier databases")
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        pick = c1.selectbox("Result", olds, format_func=lambda p: p.parent.name, key="dbold")
        if c2.button("Open", key="dbopen", width="stretch"):
            st.session_state["db_open"] = str(pick.parent)
        if st.session_state.get("db_open"):
            show_db(st.session_state["db_open"])

# ----------------------------------------------------------------------
with tab_sec:
    setup, info = st.columns([5, 6], gap="large")
    with setup:
        with st.container(border=True):
            ui.step("Section", first=True)
            af, aname = views.airfoil_picker("sdb")
            c1, c2 = st.columns(2)
            sflaps = c1.text_input("Plain-flap deflections (deg)", "0",
                                   help="0 = clean section only; e.g. 0, 10, 20, 30 adds flapped polars")
            scf = c2.number_input("Flap chord / c", 0.1, 0.5, 0.25, 0.05, key="sdbcf")
            ui.step("Conditions")
            c1, c2 = st.columns(2)
            res_ = c1.text_input("Reynolds numbers", "1e6, 3e6, 6e6")
            ms = c2.text_input("Mach numbers", "0.1",
                               help="Below 0.3 incompressible (one set serves all low Mach numbers); "
                                    "from 0.3 the compressible solver, fully turbulent from 0.4")
            c1, c2 = st.columns(2)
            sal = c1.text_input("α (deg)", "-6:20:2")
            tr = c2.selectbox("Transition", ["free", "off"],
                              format_func=lambda s: {"free": "free (e^N)", "off": "fully turbulent"}[s])
            with st.expander("Numerics"):
                c1, c2 = st.columns(2)
                lev = c1.select_slider("Mesh level", [0.7, 1.0, 1.4], 1.0)
                jobs = c2.number_input("Parallel processes", 1, 32, 2)
            n_re = len([x for x in res_.split(",") if x.strip()])
            n_m = len([x for x in ms.split(",") if x.strip()]) * max(1, len([x for x in sflaps.split(",") if x.strip()]))
            c1, c2 = st.columns(2, vertical_alignment="center")
            go2 = c1.button("Compute polars", type="primary", width="stretch", disabled=af is None)
            c2.caption(f"{n_re * n_m} polars, about {15 * n_re * n_m}–{45 * n_re * n_m} min "
                       "(polars already computed are reused)")
    with info:
        if af:
            views.section_plot(af, height=200, title=f"Section: {aname}")
        st.markdown(
            "Each polar is computed once and kept in the shared section-polar store, so the wing, "
            "aircraft and sizing tools use it too.\n\n"
            "Outputs: section_database.xlsx (Cl, Cd, Cm sheets: rows α, one column per Re and Mach), "
            "CSV / JSON, and one XFOIL-format polar file per condition, which XFLR5 and most "
            "aircraft-design tools read.")
    if go2:
        out = C.CASES / f"sdb_{aname}_{int(time.time())}"
        args = ["aero_database.py", "section", "--airfoil", af, f"--re={res_.replace(' ', '')}",
                f"--mach={ms.replace(' ', '')}", f"--alpha={sal.replace(' ', '')}", "--refine", lev,
                "--jobs", jobs, "--transition", tr, f"--flap={sflaps.replace(' ', '')}", "--flap-chord", scf,
                "--polar-dir", C.POLAR_CACHE, "--out", out]
        st.session_state["sdb_job"] = C.start_job("aero_db", f"Section database {aname}", args, out)

    def show_sdb(out):
        out = Path(out)
        if not (out / "section_database.csv").exists():
            return
        df = pd.read_csv(out / "section_database.csv")
        series = []
        if "flap" not in df.columns:
            df["flap"] = 0.0
        for (re, m, fl), sub in df.groupby(["Re", "mach", "flap"]):
            series.append((f"Re {C.fmt_re(re)}, M {m:g}" + (f", flap {fl:g}°" if fl else ""),
                           sub.sort_values("alpha").to_dict("records")))
        figs = C.polar_figures(series)
        c1, c2 = st.columns(2)
        ui.plot(figs["cl"], container=c1, key=f"s1{out}")
        ui.plot(figs["polar"], container=c2, key=f"s2{out}")
        views.downloads(out, ("section_database.xlsx", "section_database.csv", "section_database.json"), "sdl")
        txt = sorted(out.glob("polar_*.txt"))
        if txt:
            cols = st.columns(min(len(txt), 4))
            for i, p in enumerate(txt):
                cols[i % 4].download_button(f"Download {p.name}", p.read_bytes(), file_name=p.name, key=f"x{p}")

    jid = st.session_state.get("sdb_job")
    if jid:
        st.markdown("### Run")
        views.live_job(jid, "aero_db")
        if C.get_job(jid)["status"] == "finished":
            show_sdb(C.get_job(jid)["out"])
    olds = sorted(C.CASES.glob("sdb_*/database_done.json"), key=lambda p: -p.stat().st_mtime)
    if olds:
        st.markdown("### Earlier section databases")
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        pick = c1.selectbox("Result", olds, format_func=lambda p: p.parent.name, key="sdbold")
        if c2.button("Open", key="sdbopen", width="stretch"):
            st.session_state["sdb_open"] = str(pick.parent)
        if st.session_state.get("sdb_open"):
            show_sdb(st.session_state["sdb_open"])
