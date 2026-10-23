import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import aircraft_views as AV
import common as C
import ui
import views

from aerolab.aircraft import wing as W

C.page_setup()
ui.header("Wing 3D CFD", "3D RANS (Spalart–Allmaras) of a half wing: a wing of a saved aircraft or one "
                         "entered here, root and tip sections, taper, sweep, dihedral and twist. CL, CD, Cm "
                         "and span loading, next to the vortex lattice the aircraft pages use.")

LEVELS = {0.3: "coarse (about 180k cells)", 0.4: "medium (about 300k cells)", 0.5: "fine (about 500k cells)"}


def wing3d_results(out):
    """CFD against the vortex lattice: coefficients, span loading, table"""
    out = Path(out)
    f = out / "wing3d.json"
    if not f.exists():
        st.info("No results in this folder yet.")
        return
    d = json.loads(f.read_text())
    su = json.loads((out / "setup.json").read_text()) if (out / "setup.json").exists() else {}
    res = d["results"]
    vl = {r["alpha"]: r for r in (d.get("vlm") or {}).get("results", [])}
    info = d["info"]
    al = [r["alpha"] for r in res]
    ok = [r for r in res if np.isfinite(r["CL"]) and abs(r["CL"]) < 10]
    if len(ok) < len(res):
        st.error(f"{len(res) - len(ok)} angle(s) diverged; their values are left out. A coarser or finer "
                 f"mesh level, or fewer angles, usually gets past it.")
    k = st.columns(4)
    k[0].metric("Aspect ratio", f"{info['aspect_ratio']:.2f}")
    k[1].metric("Taper", f"{info['taper']:.2f}")
    k[2].metric("Re, root chord", C.fmt_re(info["re_root"]))
    k[3].metric("Cells", f"{d['mesh'].get('cells', 0):,}")
    if su.get("speed"):
        st.caption(f"{su.get('source', '')}: span {su['span']:g} m, root chord {su['root_chord']:g} m, tip chord "
                   f"{su['tip_chord']:g} m, {su['speed']:g} m/s at {su['altitude']:g} m. Coefficients on the wing "
                   f"area {su['area']:.4g} m²; Cm about the root quarter chord, on the MAC.")
    nc = [r["alpha"] for r in ok if r["status"] != "converged"]
    if nc:
        st.warning("Not converged at α " + ", ".join(f"{a:g}°" for a in nc) + ": the forces were still moving "
                   "when the iteration limit was reached. Treat those values as rough.")
    rows = []
    for r in res:
        v = vl.get(r["alpha"], {})
        row = {"α (deg)": r["alpha"], "CL CFD": r["CL"], "CL lattice": v.get("CL"), "CD CFD": r["CD"],
               "CD lattice": v.get("CD"), "CD pressure": r["CDp"], "CD friction": r["CDv"], "Cm CFD": r["Cm"],
               "Cm lattice": v.get("Cm"), "iterations": r["iterations"], "status": r["status"]}
        if su.get("speed"):
            row["lift (N)"] = 0.5 * su["rho"] * su["speed"] ** 2 * su["area"] * r["CL"]
        rows.append(row)
    src = (d.get("vlm") or {}).get("source")
    t1, t2, t3 = st.tabs(["Coefficients", "Span loading", "Table and files"])
    with t1:
        c1, c2, c3 = st.columns(3)
        for col, key, title, yl in ((c1, "CL", "Lift curve", "CL"), (c2, "CD", "Drag", "CD"),
                                    (c3, "Cm", "Pitching moment", "Cm, root c/4")):
            fg = go.Figure()
            fg.add_scatter(x=[r["alpha"] for r in ok], y=[r[key] for r in ok], mode="lines+markers", name="3D CFD")
            if vl:
                fg.add_scatter(x=list(vl), y=[v[key] for v in vl.values()], mode="lines+markers",
                               name="vortex lattice", line=dict(dash="dash"))
            fg.update_layout(title=title, xaxis_title="α (deg)", yaxis_title=yl, height=330)
            ui.plot(fg, container=col, key=f"w3{key}{out}")
        if src:
            st.caption(f"Vortex lattice: the method of the Wing and Aircraft pages, with {src} section polars. "
                       "The CFD resolves the tip vortex, the boundary layer and the section shape between root and "
                       "tip directly; a large difference in CL slope points at the section polars, in CD at the "
                       "friction and profile drag the lattice takes from them.")
    with t2:
        if not ok:
            st.info("No converged angle to show.")
        else:
            a = st.select_slider("Angle of attack (deg)", [r["alpha"] for r in ok], value=ok[-1]["alpha"],
                                 key=f"w3a{out}")
            r = next(x for x in ok if x["alpha"] == a)
            v = vl.get(a)
            c1, c2 = st.columns(2)
            fg = go.Figure()
            fg.add_scatter(x=r["span"]["eta"], y=r["span"]["clc"], mode="lines+markers", name="3D CFD")
            if v:
                fg.add_scatter(x=v["eta"], y=v["clc"], mode="lines", name="vortex lattice", line=dict(dash="dash"))
            fg.update_layout(title="Span loading", xaxis_title="η = 2y/b", yaxis_title="c_l c / c_root", height=360)
            ui.plot(fg, container=c1, key=f"w3s{out}")
            fg = go.Figure()
            fg.add_scatter(x=r["span"]["eta"], y=r["span"]["cl"], mode="lines+markers", name="3D CFD")
            if v:
                fg.add_scatter(x=v["eta"], y=v["cl"], mode="lines", name="vortex lattice", line=dict(dash="dash"))
            fg.update_layout(title="Section lift", xaxis_title="η = 2y/b", yaxis_title="c_l", height=360)
            ui.plot(fg, container=c2, key=f"w3c{out}")
            st.caption("Where the local c_l peaks is where the wing stalls first. Results → Flow fields shows the "
                       "surface pressure, flow cuts and the tip vortex of this run.")
    with t3:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={k_: st.column_config.NumberColumn(format="%.5f" if "CD" in k_ else "%.4f")
                                    for k_ in rows[0] if k_.startswith(("CL", "CD", "Cm"))})
        views.downloads(out, ["wing3d.json", "report.txt"], key=f"w3d{out}")
    st.page_link("tools/34_Results.py", label="Flow fields of this run (Results → Flow fields)",
                 query_params={"view": "Flow fields"})


setup, preview = st.columns([5, 6], gap="large")
with setup:
    with st.container(border=True):
        ui.step("Wing", first=True)
        src = st.segmented_control("Geometry from", ["Saved aircraft", "Enter a wing"], default="Saved aircraft",
                                   key="w3src") or "Saved aircraft"
        if src == "Saved aircraft":
            ac, path = AV.aircraft_picker(key="w3ac")
            surfs = [s for s in ac.surfaces if not s.vertical]
            if not surfs:
                st.error("This aircraft has no horizontal lifting surface.")
                st.stop()
            sf = st.selectbox("Surface", surfs, format_func=lambda s: s.name, key="w3surf")
            span = sf.span if sf.symmetric else 2 * sf.span
            cr, ct = sf.root_chord, sf.tip_chord
            sweep, dih, twist = sf.sweep, sf.dihedral, sf.twist
            root, tip = sf.airfoil, (sf.tip_airfoil or None)
            label = f"{ac.name}, {sf.name}"
            if sf.kinks:
                st.warning(f"{sf.name} has {len(sf.kinks)} panel break(s). The 3D mesh is one straight panel "
                           f"from the root to the tip section; the breaks are left out.")
            st.caption(f"Span {span:.4g} m, root chord {cr:.4g} m, tip chord {ct:.4g} m, sweep {sweep:g}°, "
                       f"dihedral {dih:g}°, twist {twist:g}°. Sections: {C.display_name(Path(str(root)).stem)}"
                       + (f" → {C.display_name(Path(str(tip)).stem)}" if tip and tip != root else "") + ".")
        else:
            c1, c2, c3 = st.columns(3)
            span = c1.number_input("Span (m)", 0.1, 100.0, 10.0)
            cr = c2.number_input("Root chord (m)", 0.01, 20.0, 1.6)
            ct = c3.number_input("Tip chord (m)", 0.01, 20.0, 0.8)
            c1, c2, c3 = st.columns(3)
            sweep = c1.number_input("Quarter-chord sweep (deg)", -30.0, 45.0, 0.0, 0.5)
            dih = c2.number_input("Dihedral (deg)", -10.0, 15.0, 0.0, 0.5)
            twist = c3.number_input("Tip twist (deg)", -8.0, 4.0, 0.0, 0.25, help="Negative = washout")
            root, rname = views.airfoil_picker("w3root", "Root section")
            tip = None
            if st.checkbox("Different tip section", key="w3difftip"):
                tip, _ = views.airfoil_picker("w3tip", "Tip section")
            label = f"{C.display_name(rname or '')} wing, span {span:g} m"

        ui.step("Flight condition")
        c1, c2 = st.columns(2)
        V = c1.number_input("Speed (m/s)", 1.0, 120.0, 40.0)
        alt = c2.number_input("Altitude (m)", 0.0, 15000.0, 0.0, 100.0)
        alpha = st.text_input("Angles of attack (deg)", "0,4,8",
                              help="Angle of the root chord to the flow (the wing's incidence on the aircraft is not added). "
                                   "Comma separated; each angle starts from the one before.")
        with st.expander("Mesh and solver"):
            lev = st.select_slider("Mesh", list(LEVELS), 0.3, format_func=lambda x: LEVELS[x])
            c1, c2 = st.columns(2)
            ff = c1.number_input("Far field (root chords)", 6.0, 30.0, 12.0, 1.0)
            iters = c2.number_input("Iterations per angle, at most", 200, 10000, 3000, 100)

try:
    alphas = sorted(float(x) for x in alpha.replace(" ", "").split(",") if x)
except ValueError:
    alphas = []
rho, mu, a_snd, *_ = W.isa(alt)
re_root = rho * V * cr / mu
mach = V / a_snd
taper = ct / cr
area = 0.5 * span * (cr + ct)
problems = []
if root is None:
    problems.append("Choose the root section.")
if not alphas:
    problems.append("Enter the angles as numbers separated by commas, e.g. 0,4,8.")
if not 0.1 <= taper <= 1.5:
    problems.append(f"Taper {taper:.2f} (tip / root chord): the 3D mesher handles 0.1 to 1.5.")
for sec in (root, tip):
    if sec is not None:
        try:
            from aerolab.airfoil import airfoil_geom as ag
            ag.load_airfoil(C.airfoil_file(sec), n_side=48)
        except Exception as e:                       # noqa: BLE001
            problems.append(f"Section {Path(str(sec)).name} cannot be meshed: {e}")
if not 0.5 <= 0.5 * span / cr <= 30:
    problems.append(f"Semi-span {0.5 * span / cr:.2g} root chords: the 3D mesher handles 0.5 to 30.")

with preview:
    s = span / 2
    tl = math.tan(math.radians(sweep))
    ys = np.array([-s, 0, s])
    cs = np.array([ct, cr, ct])
    xle = np.abs(ys) * tl - 0.25 * cs + 0.25 * cr
    fg = go.Figure()
    fg.add_scatter(x=np.r_[ys, ys[::-1], ys[0]], y=-np.r_[xle, (xle + cs)[::-1], xle[0]], fill="toself",
                   fillcolor="rgba(29,78,137,0.06)", mode="lines", name="planform", hoverinfo="skip")
    fg.add_vrect(x0=0, x1=s, fillcolor="rgba(29,78,137,0.07)", line_width=0)
    fg.update_layout(yaxis=dict(scaleanchor="x", zeroline=False, title="x (m)"), xaxis=dict(title="y (m)", zeroline=False),
                     height=280, title="Planform, top view (shaded: the half wing solved)", showlegend=False)
    ui.plot(fg)
    k = st.columns(4)
    k[0].metric("Area", f"{area:.4g} m²")
    k[1].metric("Aspect ratio", f"{span * span / area:.2f}")
    k[2].metric("Re, root chord", C.fmt_re(re_root))
    k[3].metric("Mach", f"{mach:.3f}")
    for p in problems:
        st.error(p)
    if mach > 0.3:
        st.warning(f"Mach {mach:.2f}: the 3D solver is incompressible, so above about Mach 0.3 the lift is "
                   "underestimated (Prandtl–Glauert: about 1/√(1 − M²)).")
    if re_root < 2e5:
        st.warning(f"Re {C.fmt_re(re_root)}: the 3D solver is fully turbulent (no transition model). Below "
                   "about 5e5 laminar flow and separation bubbles matter: expect more drag and less "
                   "lift than the real wing at low angles.")
    st.caption(f"{len(alphas)} angle(s). A run takes roughly 1–3 h per angle on the coarse mesh with 2 cores, "
               "longer on finer meshes. It runs in the background: you can close the page.")
    go_ = st.button("Run 3D CFD", type="primary", disabled=bool(problems))

if go_:
    stem = "".join(ch if ch.isalnum() else "_" for ch in label)[:40].strip("_")
    out = C.CASES / f"wing3d_{stem}_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "setup.json").write_text(json.dumps(
        {"source": label, "span": span, "root_chord": cr, "tip_chord": ct, "sweep": sweep, "dihedral": dih,
         "twist": twist, "airfoil": str(root), "tip_airfoil": str(tip) if tip else None, "speed": V,
         "altitude": alt, "rho": rho, "mach": mach, "area": area, "re_root": re_root}, indent=1))
    args = ["wing3d.py", "--airfoil", root, "--semi-span", f"{0.5 * span / cr:.6g}", "--taper", f"{taper:.6g}",
            "--sweep", sweep, "--dihedral", dih, "--twist", twist, "--re", f"{re_root:.6g}",
            "--alpha", ",".join(f"{a:g}" for a in alphas), "--level", lev, "--farfield", ff,
            "--max-iter", iters, "--polar-dir", C.POLAR_CACHE, "--out", out]
    if tip and tip != root:
        args += ["--tip-airfoil", tip]
    st.session_state["last_wing3d_job"] = C.start_job("wing3d", f"3D CFD: {label}", args, out,
                                                      meta={"n_angles": len(alphas)})

jid = st.session_state.get("last_wing3d_job")
if jid:
    st.markdown("### Run")
    views.live_job(jid, "wing3d")
    j = C.get_job(jid)
    if j["status"] == "finished":
        wing3d_results(j["out"])

past = sorted(C.CASES.glob("wing3d_*/wing3d.json"), key=lambda p: p.stat().st_mtime, reverse=True)
if past:
    st.markdown("### Earlier 3D runs")
    pick = st.selectbox("Result", past, format_func=lambda p: p.parent.name, key="w3past")
    wing3d_results(pick.parent)
