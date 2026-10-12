"""Result views shared by several pages."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
import ui

NAMES = {"tmr0012": "NACA 0012, sharp TE (NASA TMR)",
         "../validation/reference/rae2822.dat": "RAE 2822 (transonic)"}


def airfoil_picker(key="af", label="Airfoil", allow_flap=False):
    """library section, any NACA 4/5-digit code, or an uploaded .dat;
    allow_flap: optional plain trailing-edge flap (spec suffix @flap=cf,deg)"""
    spec, name = _airfoil_picker(key, label)
    if allow_flap and spec is not None:
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        on = c1.checkbox("Plain trailing-edge flap", key=key + "flap",
                         help="Sealed plain flap: the camber line aft of the hinge is rotated, "
                              "with a smooth 4 % chord hinge blend; chord reference unchanged")
        cf = c2.number_input("Flap chord / c", 0.1, 0.5, 0.25, 0.05, disabled=not on, key=key + "fcf")
        de = c3.number_input("Deflection (°, + down)", -30.0, 30.0, 20.0, 5.0, disabled=not on, key=key + "fde",
                             help="Validated 0–20°. Beyond about 20° the flow on a plain flap separates and "
                                  "the RANS runs often diverge.")
        if on and abs(de) > 20:
            st.caption("Above 20° the flap flow separates; expect unsteady or diverged points.")
        if on and de != 0:
            import airfoil_geom as ag
            spec = ag.flap_spec(spec, cf, de)
            name = f"{name}_flap{round(100 * cf)}c{de:g}"
    return spec, name


def _airfoil_picker(key="af", label="Airfoil"):
    c1, c2 = st.columns([3, 2])
    pick = c1.selectbox(label, C.BUILTIN_AIRFOILS + ["NACA code", "Upload .dat file"],
                        key=key + "pick", format_func=lambda s: NAMES.get(s, C.display_name(s)))
    if pick == "Upload .dat file":
        up = c2.file_uploader("Selig or Lednicer format", type=["dat", "txt"], key=key + "up")
        if up is None:
            return None, None
        d = C.CASES / "_uploads"
        d.mkdir(exist_ok=True)
        p = d / up.name
        p.write_bytes(up.getvalue())
        return str(p), Path(up.name).stem
    if pick == "NACA code":
        code = c2.text_input("4 or 5 digits", "4415", key=key + "code").strip().lower()
        code = code if code.startswith("naca") else "naca" + code
        return code, code
    return pick, Path(pick).stem


def section_plot(spec, height=200, title=None):
    """airfoil outline, equal axes"""
    title = title and title.replace(str(Path(str(spec)).stem), C.display_name(Path(str(spec)).stem))
    if "@flap=" in str(spec):
        import airfoil_geom as ag
        b, (cf, d) = ag.parse_flap(spec)
        title = f"{C.display_name(Path(str(b)).stem)}, flap {100 * cf:g} % c at {d:g}°"
    try:
        o = C.raw_outline(spec)
    except Exception as e:
        st.caption(f"Can't draw this section: {e}")
        return
    f = go.Figure()
    f.add_scatter(x=o[:, 0], y=o[:, 1], mode="lines", line=dict(color=ui.BLUE, width=1.6),
                  fill="toself", fillcolor="rgba(29,78,137,0.06)", hoverinfo="skip", showlegend=False)
    f.update_layout(height=height, title=title, margin=dict(l=40, r=12, t=30 if title else 8, b=30),
                    yaxis=dict(scaleanchor="x", zeroline=False), xaxis=dict(zeroline=False))
    ui.plot(f)


def live_job(jid, kind="polar"):
    """auto-refreshing progress block of a background run"""
    @st.fragment(run_every="3s")
    def _box():
        try:
            j = C.get_job(jid)
        except FileNotFoundError:
            st.warning("This run no longer exists. Its record was deleted from cases/_jobs.")
            return
        mins = (time.time() - j["started"]) / 60
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{j['title']}**  {ui.status_md(j['status'])}  "
                    f"<span style='color:{ui.GRAPHITE}'>{mins:.1f} min</span>", unsafe_allow_html=True)
        if j["status"] == "running" and c2.button("Cancel run", key=f"cancel{jid}"):
            C.cancel_job(jid)
            st.rerun()
        log = C.job_log(jid)
        pts = C.parse_points(log) if kind == "polar" else []
        n_tot = j["meta"].get("n_angles")
        if n_tot and kind == "polar":
            st.progress(min(len(pts) / n_tot, 1.0), text=f"{len(pts)} of {n_tot} angles solved")
        if pts:
            c1, c2 = st.columns([1, 1])
            figs = C.polar_figures([("live", pts)])
            ui.plot(figs["cl"], key=f"lcl{jid}", container=c1)
            c2.dataframe(pd.DataFrame(pts), hide_index=True, width="stretch",
                         column_config=_num_cols())
        with st.expander("Solver log", expanded=not pts and j["status"] != "finished"):
            st.code(C.job_log(jid, tail=40) or "starting...", language=None)
        if j["status"] == "finished":
            st.caption(f"Results saved in {j['out']}")
            if kind in ("grid_study", "project") and st.button("Show results", key=f"show{jid}"):
                st.rerun(scope="app")
        elif j["status"] == "failed":
            st.error("The run stopped before writing results. The end of the solver log above "
                     "shows why.")
    _box()


def _num_cols():
    f = st.column_config.NumberColumn
    return {"alpha": f("α (deg)", format="%.2f"), "Cl": f("Cl", format="%.4f"),
            "Cd": f("Cd", format="%.5f"), "Cd_p": f("Cd pressure", format="%.5f"),
            "Cd_v": f("Cd friction", format="%.5f"), "Cm": f("Cm", format="%+.4f"),
            "x_tr_upper": f("x_tr upper", format="%.3f"), "x_tr_lower": f("x_tr lower", format="%.3f"),
            "time_s": f("time (s)", format="%.0f"), "M_max": f("M max", format="%.3f")}


def downloads(out, files, key):
    present = [fn for fn in files if (Path(out) / fn).exists()]
    if not present:
        return
    cols = st.columns(len(present) + 2)
    for c, fn in zip(cols, present):
        c.download_button(f"Download {fn}", (Path(out) / fn).read_bytes(), file_name=fn,
                          key=f"{key}{out}{fn}")


def polar_results(out_dir):
    """full results of an airfoil.py polar run"""
    out = Path(out_dir)
    f = out / "polar.json"
    if not f.exists():
        st.info("No results in this folder yet.")
        return
    d = json.loads(f.read_text())
    res = d["results"]
    info = d.get("info", {})
    s = d.get("settings", {})
    ok = [r for r in res if r.get("Cl") is not None and np.isfinite(r["Cl"])]
    st.subheader(f"{C.display_name(info.get('name', out.name))}, Re {C.fmt_re(d.get('re', 0))}"
                 + (f", M {s.get('mach'):g}" if s.get("mach") else ""))
    st.caption(f"Folder: {out}")
    if ok:
        i = int(np.argmax([r["Cl"] for r in ok]))
        ld = [r["Cl"] / r["Cd"] if r["Cd"] else 0 for r in ok]
        j = int(np.argmax(ld))
        z = min(ok, key=lambda r: abs(r["Cl"]))
        k = st.columns(5)
        top = ok[i]["alpha"] == max(r["alpha"] for r in ok)
        k[0].metric(f"Cl at the highest α ({ok[i]['alpha']:g}°)" if top else
                    f"Cl max, at α {ok[i]['alpha']:g}°", f"{ok[i]['Cl']:.3f}",
                    help="The lift curve still rises at the last angle: run higher angles to find "
                         "the stall." if top else None)
        k[1].metric(f"L/D max, at α {ok[j]['alpha']:g}°", f"{ld[j]:.1f}")
        k[2].metric("Cd min", f"{min(r['Cd'] for r in ok):.5f}")
        k[3].metric("Cm near zero lift", f"{z['Cm']:+.4f}")
        bad = sum(not str(r["status"]).startswith("converged") for r in res)
        k[4].metric("points not converged", f"{bad} of {len(res)}")
    ass = polar_trust(out, d)
    bands = {f"{float(p['alpha']):g}": (p["cl_band"], p["cd_band"]) for p in ass["points"]}
    t1, t2, t3, t4, t5 = st.tabs(["Polar", "Pressure and friction", "Flow field", "Confidence and grid check",
                                  "Table and files"])
    with t1:
        series = [("CFD", res)]
        for lab, xd in (d.get("xfoil") or {}).items():
            series.append((lab.split(",")[0], [{"alpha": float(k), "Cl": v[0], "Cd": v[1],
                                                "Cm": v[2]} for k, v in xd.items()]))
        show_bands = st.toggle("Show expected uncertainty", True, key=f"band{out}",
                               help="Bars: the error found on the nearest validated case, widened for the "
                                    "distance from it, the mesh level, stall and convergence (trust layer)")
        figs = C.polar_figures(series, bands if show_bands else None)
        c1, c2 = st.columns(2)
        ui.plot(figs["cl"], container=c1, key=f"pc{out}")
        ui.plot(figs["polar"], container=c2, key=f"pp{out}")
        ui.plot(figs["cm"], container=c1, key=f"pm{out}")
        ui.plot(figs["ld"], container=c2, key=f"pl{out}")
    with t2:
        okw = [r for r in res if r.get("wall") and len(r["wall"].get("x", []))]
        if okw:
            a = st.select_slider("Angle of attack (deg)", [r["alpha"] for r in okw], key=f"cpa{out}")
            r = next(x for x in okw if x["alpha"] == a)
            w = r["wall"]
            x, cp, cf = (np.array(w[k]) for k in ("x", "Cp", "Cf"))
            side, el = np.array(w["side"]), np.array(w.get("element", np.zeros(len(x))))
            f1, f2 = go.Figure(), go.Figure()
            for e in np.unique(el):
                for sd, nm in ((1, "upper"), (-1, "lower")):
                    k = (el == e) & (side == sd)
                    o = np.argsort(x[k])
                    lab = f"{nm}" + (f", element {int(e) + 1}" if len(np.unique(el)) > 1 else "")
                    dash = None if sd == 1 else "dash"
                    f1.add_scatter(x=x[k][o], y=cp[k][o], mode="lines", name=lab, line=dict(dash=dash))
                    f2.add_scatter(x=x[k][o], y=cf[k][o], mode="lines", name=lab, line=dict(dash=dash))
            for xt, nm in ((r.get("x_tr_upper"), "upper"), (r.get("x_tr_lower"), "lower")):
                if xt is not None and xt < 0.999:
                    f2.add_vline(x=xt, line=dict(color=ui.GRAPHITE, width=1, dash="dot"),
                                 annotation_text=f"transition {nm}", annotation_font_size=10)
            f1.update_layout(title="Pressure coefficient", yaxis=dict(autorange="reversed", title="Cp"),
                             xaxis_title="x/c", height=380)
            f2.update_layout(title="Skin friction", yaxis_title="Cf", xaxis_title="x/c", height=380)
            c1, c2 = st.columns(2)
            ui.plot(f1, container=c1, key=f"cp{out}")
            ui.plot(f2, container=c2, key=f"cf{out}")
            xu, xl = r.get("x_tr_upper"), r.get("x_tr_lower")
            trs = "fully turbulent" if xu is None and xl is None else \
                f"transition x/c upper {xu:.3f}, lower {xl:.3f}"
            st.caption(f"Cl {r['Cl']:.4f}, Cd {r['Cd']:.5f}, Cm {r['Cm']:+.4f}; {trs}; {r['status']}")
    with t3:
        folders = sorted(out.glob("alpha_*"))
        if not folders:
            st.info("Flow fields weren't saved for this run. Tick 'Save flow fields' before running.")
        else:
            c1, c2, c3 = st.columns(3)
            fo = c1.selectbox("Angle", folders, format_func=lambda p: p.name.replace("alpha_", "α ") + "°",
                              key=f"ff{out}")
            names = list(C.read_vtu_cell_data(fo / "results.vtu").keys())
            pref = [n for n in ("Mach", "Cp", "U_mag", "vorticity", "nut_ratio", "p_total_loss",
                                "T", "rho") if n in names]
            field = c2.selectbox("Field", pref + [n for n in names if n not in pref], key=f"fd{out}")
            zoom = c3.selectbox("View", ["Whole section", "Leading edge", "Trailing edge", "Far field"],
                                key=f"fz{out}")
            lim = {"Whole section": ((-0.3, 1.5), (-0.45, 0.45)),
                   "Leading edge": ((-0.05, 0.15), (-0.06, 0.06)),
                   "Trailing edge": ((0.85, 1.15), (-0.1, 0.1)), "Far field": ((-2, 4), (-1.5, 1.5))}[zoom]
            fig, _ = C.field_image(fo, field, *lim, cmap="RdBu_r" if field == "vorticity" else "turbo")
            st.pyplot(fig)
    with t4:
        trust_details(out, d, ass)
    with t5:
        cols = ["alpha", "Cl", "Cd", "Cd_p", "Cd_v", "Cm", "x_tr_upper", "x_tr_lower", "status",
                "iterations", "time_s", "M_max"]
        lv = {p["alpha"]: p["level"] for p in ass["points"]}
        df = pd.DataFrame([{**{k: r.get(k) for k in cols}, "confidence": ui.TRUST.get(lv.get(r["alpha"]), ("", ""))[1]}
                           for r in res])
        st.dataframe(df, hide_index=True, width="stretch", column_config=_num_cols())
        downloads(out, ("polar.csv", "report.txt", "polar.json"), "dl")
        for fn in ("polar.png", "distributions.png", "geometry.png"):
            if (out / fn).exists():
                st.image(str(out / fn), caption=fn)


def polar_trust(out, d):
    """confidence line under the polar's headline numbers; returns the assessment"""
    import trust as T
    ass = T.assess_polar(d, T.load_grid_study(out))
    c = ass["counts"]
    parts = " ".join(ui.trust_md(l, f"{c[l]} {ui.TRUST[l][1]}") for l in T.LEVELS if c[l])
    ctx = ass["context"]
    st.markdown(f"**Confidence** {parts}  "
                f"<span style='color:{ui.GRAPHITE};font-size:0.85rem'>against {len(T.CASES)} validated cases; "
                f"see the *Confidence and grid check* tab for the reasons at each angle.</span>",
                unsafe_allow_html=True)
    return ass


def trust_details(out, d, ass):
    """per-angle reasons + grid convergence study (levels 0.7 / 1 / 1.4)"""
    import trust as T
    out = Path(out)
    pts = ass["points"]
    if not pts:
        st.info("This polar doesn't record its Reynolds number, so it can't be assessed.")
        return
    rows = [{"α (deg)": p["alpha"], "confidence": ui.TRUST[p["level"]][1],
             "Cl ±": p["cl_band"], "Cd ±": p["cd_band"], "nearest validated case": p["case_title"],
             "distance": p["distance"],
             "main reason": next((t for l, t in p["reasons"] if l == p["level"]), p["reasons"][0][1])}
            for p in pts]
    pc = st.column_config.NumberColumn
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Cl ±": pc(format="percent"), "Cd ±": pc(format="percent"),
                                "distance": pc(help="0 = inside the case's range of Re, Mach and α; "
                                                    "1 ≈ a factor 1.8 in Re, 0.05 in Mach or 2° away",
                                               format="%.2f"),
                                "main reason": st.column_config.TextColumn(width="large")})
    a = st.selectbox("All reasons at", [p["alpha"] for p in pts], key=f"tra{out}",
                     format_func=lambda x: f"α {x:g}°")
    p = next(p for p in pts if p["alpha"] == a)
    for l, t in p["reasons"]:
        st.markdown(f"{ui.trust_md(l, ui.TRUST[l][1])} {t}")

    st.markdown("#### Grid convergence")
    g = T.load_grid_study(out)
    gfile = out / "_grid" / "grid_study.json"
    jid = st.session_state.get(f"grid{out}")
    if jid:
        live_job(jid, "grid_study")
    if g:
        rr = []
        for ak, e in sorted(g.items(), key=lambda kv: float(kv[0])):
            for q in ("Cl", "Cd", "Cm"):
                if q in e:
                    x = e[q]
                    rr.append({"α (deg)": float(ak), "quantity": q, "coarse": x["phi"][2], "medium": x["phi"][1],
                               "fine": x["phi"][0], "order p": x["p"], "p for GCI": x.get("p_gci", x["p"]),
                               "extrapolated": x["phi_ext"],
                               "GCI fine": x["gci_fine"], "± absolute": x.get("gci_abs"), "convergence": "oscillatory" if x.get("oscillatory")
                               else "converged" if x.get("converged") else "monotonic"})
        st.dataframe(pd.DataFrame(rr), hide_index=True, width="stretch",
                     column_config={"GCI fine": pc(format="percent"), "order p": pc(format="%.2f"),
                                    "p for GCI": pc(format="%.2f", help="min(apparent order, 2): the scheme is "
                                                                         "second order, a higher apparent order "
                                                                         "is not trusted (Roache)"),
                                    "± absolute": pc(format="%.2g", help="GCI as an absolute error: use it for "
                                                                          "coefficients near zero (Cm of a "
                                                                          "symmetric section), where the "
                                                                          "relative value is meaningless"),
                                    **{k: pc(format="%.5g") for k in ("coarse", "medium", "fine", "extrapolated")}})
        st.caption("Celik et al. (2008) procedure: apparent order from three meshes, Richardson extrapolation, "
                   "GCI with safety factor 1.25 and the order capped at 2. Cells from the runs (h = N^-1/2). The GCI also tightens the "
                   "Cd band of these angles.")
    else:
        st.caption("No grid study yet. It reruns the chosen angles on the coarse (0.7) and fine (1.4) meshes "
                   "and gives the discretisation error of every coefficient.")
    s = d.get("settings") or {}
    if not s.get("airfoil"):
        st.caption("This polar was computed before runs recorded their settings: rerun it to enable the grid check.")
        return
    ok = [r["alpha"] for r in d["results"] if r.get("Cl") is not None and np.isfinite(r["Cl"])]
    c1, c2, c3 = st.columns([3, 1, 2], vertical_alignment="bottom")
    pick = c1.multiselect("Angles to check", ok, default=ok[len(ok) // 2: len(ok) // 2 + 1], key=f"gsa{out}",
                          format_func=lambda x: f"{x:g}°")
    jobs = c2.number_input("Processes", 1, 16, 2, key=f"gsj{out}")
    n = len(pick)
    if c3.button(f"Run grid check ({2 * n} runs)", disabled=not n, key=f"gsb{out}", width="stretch"):
        if gfile.exists():
            gfile.unlink()
        jid = C.start_job("grid_study", f"Grid check: {C.display_name(d.get('info', {}).get('name', out.name))} "
                                        f"α {', '.join(f'{x:g}' for x in pick)}",
                          ["trust.py", "grid-study", out, "--alpha", ",".join(f"{x:g}" for x in pick),
                           "--jobs", jobs], out / "_grid")
        st.session_state[f"grid{out}"] = jid
        st.rerun()


def trust_notes(tool, title="Confidence", **ctx):
    import trust as T
    ui.trust_notes(T.tool_notes(tool, **ctx), title)


def wing_results(out_dir):
    out = Path(out_dir)
    f = out / "wing.json"
    if not f.exists():
        st.info("No results in this folder yet.")
        return
    d = json.loads(f.read_text())
    res, w = d["results"], d["wing"]
    al = [r["alpha"] for r in res]
    CL = [r["CL"] for r in res]
    i = int(np.argmax(CL))
    j = int(np.argmax([r["L_D"] for r in res]))
    k = st.columns(5)
    k[0].metric(f"CL at the highest α ({al[i]:g}°)" if i == len(al) - 1 else
                f"CL max, at α {al[i]:g}°", f"{CL[i]:.3f}")
    k[1].metric(f"L/D max, at α {al[j]:g}°", f"{res[j]['L_D']:.1f}")
    k[2].metric("Aspect ratio", f"{w['aspect_ratio']:.2f}")
    k[3].metric("Area", f"{w['area']:.3g} m²")
    k[4].metric("Mean aerodynamic chord", f"{w['mac']:.3g} m")
    fl = d.get("flight") or {}
    trust_notes("wing", aspect_ratio=w["aspect_ratio"], mach=fl.get("mach"))
    if fl.get("speed_m_s") and fl.get("rho"):
        q = 0.5 * fl["rho"] * fl["speed_m_s"] ** 2
        st.caption(f"Flight condition {fl['speed_m_s']:g} m/s at {fl.get('altitude_m', 0):g} m: "
                   f"M {fl['mach']:.3f}, Re (MAC) {C.fmt_re(fl['re_mac'])}. "
                   f"Mass supported at CL max: {q * w['area'] * CL[i] / 9.81:.0f} kg.")
    t1, t2, t3 = st.tabs(["Coefficients", "Span loading", "Table and files"])
    with t1:
        c1, c2, c3 = st.columns(3)
        f = go.Figure(); f.add_scatter(x=al, y=CL, mode="lines+markers", name="CL")
        f.update_layout(title="Lift curve", xaxis_title="α (deg)", yaxis_title="CL", height=330)
        ui.plot(f, container=c1, key=f"w1{out}")
        f = go.Figure()
        f.add_scatter(x=[r["CD"] for r in res], y=CL, mode="lines+markers", name="total")
        f.add_scatter(x=[r["CDi"] for r in res], y=CL, mode="lines", name="induced", line=dict(dash="dash"))
        f.update_layout(title="Drag polar", xaxis_title="CD", yaxis_title="CL", height=330)
        ui.plot(f, container=c2, key=f"w2{out}")
        f = go.Figure(); f.add_scatter(x=al, y=[r["Cm"] for r in res], mode="lines+markers")
        f.update_layout(title="Pitching moment", xaxis_title="α (deg)", yaxis_title="Cm, MAC c/4", height=330)
        ui.plot(f, container=c3, key=f"w3{out}")
    with t2:
        a = st.select_slider("Angle of attack (deg)", al, value=al[i], key=f"wa{out}")
        r = res[al.index(a)]
        sp = r["span"]
        c1, c2 = st.columns(2)
        f = go.Figure()
        f.add_scatter(x=sp["y"], y=sp["cl"], name="local cl")
        f.add_scatter(x=sp["y"], y=sp["clmax"], name="section cl max", line=dict(dash="dash", color=ui.GRAPHITE))
        st_y = [y for y, s in zip(sp["y"], sp.get("stalled", [])) if s]
        st_c = [c for c, s in zip(sp["cl"], sp.get("stalled", [])) if s]
        if st_y:
            f.add_scatter(x=st_y, y=st_c, mode="markers", name="stalled strip",
                          marker=dict(color="#B3261E", size=7, symbol="x"))
        f.update_layout(title="Section lift along the span", xaxis_title="y (m)", yaxis_title="cl", height=360)
        ui.plot(f, container=c1, key=f"w4{out}")
        f = go.Figure()
        f.add_scatter(x=sp["y"], y=sp["cl_c"], name="cl·c")
        b = w["span"]
        ye = np.linspace(-b / 2, b / 2, 200)
        f.add_scatter(x=ye, y=4 * r["CL"] * w["area"] / (np.pi * b) * np.sqrt(1 - (2 * ye / b) ** 2),
                      name="elliptic, same CL", line=dict(dash="dot", color=ui.GRAPHITE))
        f.update_layout(title="Span loading", xaxis_title="y (m)", yaxis_title="cl·c (m)", height=360)
        ui.plot(f, container=c2, key=f"w5{out}")
        st.caption(f"CL {r['CL']:.4f}, CD {r['CD']:.5f} (induced {r['CDi']:.5f}), span efficiency "
                   f"{r['e']:.3f}, stalled area {100 * r['stall_fraction']:.0f} %")
    with t3:
        cols = ["alpha", "CL", "CD", "CDi", "CDp", "Cm", "L_D", "e", "stall_fraction", "converged"]
        st.dataframe(pd.DataFrame([{k: r[k] for k in cols} for r in res]), hide_index=True,
                     width="stretch")
        downloads(out, ("wing_polar.csv", "report.txt", "wing.json"), "wd")
        if (out / "wing.png").exists():
            st.image(str(out / "wing.png"))
