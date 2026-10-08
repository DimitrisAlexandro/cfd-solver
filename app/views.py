"""Result views shared by several pages."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import common as C


def airfoil_picker(key="af"):
    src = st.radio("Airfoil", ["NACA / built-in", "Upload .dat"], horizontal=True, key=key + "src")
    if src == "Upload .dat":
        up = st.file_uploader("Selig or Lednicer .dat file", type=["dat", "txt"], key=key + "up")
        if up is None:
            return None, None
        d = C.CASES / "_uploads"
        d.mkdir(exist_ok=True)
        p = d / up.name
        p.write_bytes(up.getvalue())
        return str(p), Path(up.name).stem
    c1, c2 = st.columns([2, 1])
    pick = c1.selectbox("choose", C.BUILTIN_AIRFOILS + ["other NACA code…"], key=key + "pick",
                        format_func=lambda s: {"tmr0012": "tmr0012 (NASA TMR NACA 0012, sharp TE)",
                                               "../validation/reference/rae2822.dat": "RAE 2822 (transonic)"}
                        .get(s, s))
    if pick == "other NACA code…":
        code = c2.text_input("NACA 4/5-digit", "naca4415", key=key + "code")
        return code, code
    return pick, Path(pick).stem



def live_job(jid, kind="polar"):
    """auto-refreshing progress block of a background job"""
    @st.fragment(run_every="3s")
    def _box():
        try:
            j = C.get_job(jid)
        except FileNotFoundError:
            st.warning("job not found")
            return
        mins = (__import__("time").time() - j["started"]) / 60
        icon = {"running": "🔄", "finished": "✅", "failed": "❌", "cancelled": "⏹️"}[j["status"]]
        st.markdown(f"{icon} **{j['title']}** — {j['status']} ({mins:.1f} min)")
        log = C.job_log(jid)
        pts = C.parse_points(log) if kind == "polar" else []
        n_tot = j["meta"].get("n_angles")
        if n_tot and kind == "polar":
            st.progress(min(len(pts) / n_tot, 1.0), text=f"{len(pts)} / {n_tot} angles")
        if pts:
            c1, c2 = st.columns([1, 1])
            figs = C.polar_figures([("live", pts)])
            c1.plotly_chart(figs["cl"], width="stretch", key=f"lcl{jid}")
            c2.dataframe(pd.DataFrame(pts), hide_index=True, width="stretch")
        with st.expander("log", expanded=not pts):
            st.code(C.job_log(jid, tail=40) or "(starting...)")
        if j["status"] == "running":
            if st.button("⏹ cancel", key=f"cancel{jid}"):
                C.cancel_job(jid)
                st.rerun()
        elif j["status"] == "finished":
            st.success(f"Finished → {j['out']}  (open it below or in *Jobs & results*)")
    _box()


def polar_results(out_dir):
    """full results of an airfoil.py polar run"""
    out = Path(out_dir)
    f = out / "polar.json"
    if not f.exists():
        st.info("no results yet")
        return
    d = json.loads(f.read_text())
    res = d["results"]
    info = d.get("info", {})
    s = d.get("settings", {})
    st.markdown(f"### {info.get('name', out.name)} — Re {d.get('re', 0):.3g}"
                + (f", M {s.get('mach'):g}" if s.get("mach") else ""))
    t1, t2, t3, t4 = st.tabs(["Polar", "Cp / Cf", "Flow field", "Table & files"])
    with t1:
        series = [("CFD", res)]
        for lab, xd in (d.get("xfoil") or {}).items():
            series.append((lab.split(",")[0], [{"alpha": float(k), "Cl": v[0], "Cd": v[1],
                                                "Cm": v[2]} for k, v in xd.items()]))
        figs = C.polar_figures(series)
        c1, c2 = st.columns(2)
        c1.plotly_chart(figs["cl"], width="stretch")
        c2.plotly_chart(figs["polar"], width="stretch")
        c1.plotly_chart(figs["cm"], width="stretch")
        c2.plotly_chart(figs["ld"], width="stretch")
    with t2:
        import plotly.graph_objects as go
        ok = [r for r in res if r.get("wall") and len(r["wall"].get("x", []))]
        if ok:
            a = st.select_slider("α (deg)", [r["alpha"] for r in ok], key=f"cpa{out}")
            r = next(x for x in ok if x["alpha"] == a)
            w = r["wall"]
            x, cp, cf = (np.array(w[k]) for k in ("x", "Cp", "Cf"))
            side, el = np.array(w["side"]), np.array(w.get("element", np.zeros(len(x))))
            f1, f2 = go.Figure(), go.Figure()
            for e in np.unique(el):
                for sd, nm in ((1, "upper"), (-1, "lower")):
                    k = (el == e) & (side == sd)
                    o = np.argsort(x[k])
                    lab = f"{nm}" + (f" el.{int(e)}" if len(np.unique(el)) > 1 else "")
                    f1.add_scatter(x=x[k][o], y=cp[k][o], mode="lines", name=lab)
                    f2.add_scatter(x=x[k][o], y=cf[k][o], mode="lines", name=lab)
            f1.update_yaxes(autorange="reversed", title="Cp"); f1.update_xaxes(title="x/c")
            f2.update_yaxes(title="Cf"); f2.update_xaxes(title="x/c")
            for f_ in (f1, f2):
                f_.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
            c1, c2 = st.columns(2)
            c1.plotly_chart(f1, width="stretch")
            c2.plotly_chart(f2, width="stretch")
            st.caption(f"Cl {r['Cl']:.4f}  Cd {r['Cd']:.5f}  Cm {r['Cm']:+.4f}  "
                       f"transition x/c: upper {r.get('x_tr_upper')}, lower {r.get('x_tr_lower')}"
                       f"  ·  {r['status']}")
    with t3:
        folders = sorted(out.glob("alpha_*"))
        if not folders:
            st.info("flow fields were not saved for this run (option 'save flow fields')")
        else:
            c1, c2, c3 = st.columns(3)
            fo = c1.selectbox("angle", folders, format_func=lambda p: p.name.replace("alpha_", "α "))
            names = list(C.read_vtu_cell_data(fo / "results.vtu").keys())
            pref = [n for n in ("Mach", "Cp", "U_mag", "vorticity", "nut_ratio", "p_total_loss",
                                "T", "rho") if n in names]
            field = c2.selectbox("field", pref + [n for n in names if n not in pref])
            zoom = c3.selectbox("view", ["airfoil", "leading edge", "trailing edge", "wide"])
            lim = {"airfoil": ((-0.3, 1.5), (-0.45, 0.45)), "leading edge": ((-0.05, 0.15), (-0.06, 0.06)),
                   "trailing edge": ((0.85, 1.15), (-0.1, 0.1)), "wide": ((-2, 4), (-1.5, 1.5))}[zoom]
            fig, _ = C.field_image(fo, field, *lim, cmap="RdBu_r" if field == "vorticity" else "jet")
            st.pyplot(fig)
    with t4:
        cols = ["alpha", "Cl", "Cd", "Cd_p", "Cd_v", "Cm", "x_tr_upper", "x_tr_lower", "status",
                "iterations", "time_s", "M_max"]
        df = pd.DataFrame([{k: r.get(k) for k in cols} for r in res])
        st.dataframe(df, hide_index=True, width="stretch")
        for fn in ("polar.csv", "report.txt", "polar.json"):
            p = out / fn
            if p.exists():
                st.download_button(f"⬇ {fn}", p.read_bytes(), file_name=fn, key=f"dl{out}{fn}")
        for fn in ("polar.png", "distributions.png", "geometry.png"):
            if (out / fn).exists():
                st.image(str(out / fn), caption=fn)


def wing_results(out_dir):
    import plotly.graph_objects as go
    out = Path(out_dir)
    f = out / "wing.json"
    if not f.exists():
        st.info("no results yet")
        return
    d = json.loads(f.read_text())
    res, w = d["results"], d["wing"]
    al = [r["alpha"] for r in res]
    CL = [r["CL"] for r in res]
    i = int(np.argmax(CL))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("C_L,max", f"{CL[i]:.3f}", f"at α {al[i]:g}°")
    j = int(np.argmax([r["L_D"] for r in res]))
    k2.metric("max L/D", f"{res[j]['L_D']:.1f}", f"at α {al[j]:g}°")
    k3.metric("aspect ratio", f"{w['aspect_ratio']:.2f}", f"S {w['area']:.3g}")
    k4.metric("MAC", f"{w['mac']:.3g}")
    fl = d.get("flight") or {}
    if fl.get("speed_m_s") and fl.get("rho"):
        q = 0.5 * fl["rho"] * fl["speed_m_s"] ** 2
        st.caption(f"flight: {fl['speed_m_s']:g} m/s, {fl.get('altitude_m', 0):g} m, "
                   f"M {fl['mach']:.3f}, Re_MAC {fl['re_mac']:.3g} — lift at C_L,max: "
                   f"{q * w['area'] * CL[i] / 9.81:.0f} kg")
    t1, t2, t3 = st.tabs(["Coefficients", "Span loading", "Table"])
    with t1:
        c1, c2, c3 = st.columns(3)
        f = go.Figure(); f.add_scatter(x=al, y=CL, mode="lines+markers", name="C_L")
        f.update_layout(xaxis_title="α (deg)", yaxis_title="C_L", height=330)
        c1.plotly_chart(f, width="stretch")
        f = go.Figure()
        f.add_scatter(x=[r["CD"] for r in res], y=CL, mode="lines+markers", name="total")
        f.add_scatter(x=[r["CDi"] for r in res], y=CL, mode="lines", name="induced")
        f.update_layout(xaxis_title="C_D", yaxis_title="C_L", height=330)
        c2.plotly_chart(f, width="stretch")
        f = go.Figure(); f.add_scatter(x=al, y=[r["Cm"] for r in res], mode="lines+markers")
        f.update_layout(xaxis_title="α (deg)", yaxis_title="C_m (MAC c/4)", height=330)
        c3.plotly_chart(f, width="stretch")
    with t2:
        a = st.select_slider("α (deg)", al, value=al[i], key=f"wa{out}")
        r = res[al.index(a)]
        sp = r["span"]
        c1, c2 = st.columns(2)
        f = go.Figure()
        f.add_scatter(x=sp["y"], y=sp["cl"], name="local c_l")
        f.add_scatter(x=sp["y"], y=sp["clmax"], name="section c_l,max", line=dict(dash="dash"))
        st_y = [y for y, s in zip(sp["y"], sp.get("stalled", [])) if s]
        st_c = [c for c, s in zip(sp["cl"], sp.get("stalled", [])) if s]
        if st_y:
            f.add_scatter(x=st_y, y=st_c, mode="markers", name="stalled", marker=dict(color="red"))
        f.update_layout(xaxis_title="y", yaxis_title="c_l", height=360)
        c1.plotly_chart(f, width="stretch")
        f = go.Figure()
        f.add_scatter(x=sp["y"], y=sp["cl_c"], name="c_l·c")
        b = w["span"]
        ye = np.linspace(-b / 2, b / 2, 200)
        f.add_scatter(x=ye, y=4 * r["CL"] * w["area"] / (np.pi * b) * np.sqrt(1 - (2 * ye / b) ** 2),
                      name="elliptic", line=dict(dash="dot"))
        f.update_layout(xaxis_title="y", yaxis_title="c_l·c (span loading)", height=360)
        c2.plotly_chart(f, width="stretch")
        st.caption(f"C_L {r['CL']:.4f}  C_D {r['CD']:.5f} (induced {r['CDi']:.5f})  "
                   f"e {r['e']:.3f}  stalled area {100 * r['stall_fraction']:.0f} %")
    with t3:
        cols = ["alpha", "CL", "CD", "CDi", "CDp", "Cm", "L_D", "e", "stall_fraction", "converged"]
        st.dataframe(pd.DataFrame([{k: r[k] for k in cols} for r in res]), hide_index=True,
                     width="stretch")
        for fn in ("wing_polar.csv", "report.txt", "wing.json"):
            if (out / fn).exists():
                st.download_button(f"⬇ {fn}", (out / fn).read_bytes(), file_name=fn, key=f"wd{out}{fn}")
        if (out / "wing.png").exists():
            st.image(str(out / "wing.png"))
