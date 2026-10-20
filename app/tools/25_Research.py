import json
import math
import re as _re
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import aircraft_views as AV
import common as C
import ui
import views

C.page_setup()
ui.header("Research tools", "Advanced tools beyond the everyday airfoil and aircraft pages: which way to "
                            "change an airfoil's shape, an automatic airfoil optimiser, where the flow turns "
                            "turbulent on a swept wing, instant polar estimates without CFD, and a "
                            "machine-learning correction of the turbulence model.")
sys.path.insert(0, str(C.SRC))


def explain(what, when, read):
    """plain-language guide at the top of a tab"""
    with st.expander("What is this?", expanded=False):
        st.markdown(f"**What it does.** {what}\n\n**When to use it.** {when}\n\n**How to read the result.** {read}")


with st.expander("Which tool do I need?", expanded=True):
    st.markdown(
        "| Tab | Question it answers | Time |\n|---|---|---|\n"
        "| **Adjoint gradients** | *How do lift, drag and moment change if I change the shape, the angle or "
        "the Reynolds number a little?* | about 1–5 min |\n"
        "| **Adjoint design** | *Which airfoil shape gives the least drag at the lift I need?* | about 1 min per "
        "iteration, 10–30 min |\n"
        "| **Mesh accuracy** | *How much would this result change on a finer mesh, and where does the "
        "error come from?* | 10–40 min; adaptation study about 1–2 h |\n"
        "| **Swept-wing transition** | *Where does the laminar (low-drag) flow end, on one section or along a "
        "whole swept wing?* | instant (whole wing: seconds) |\n"
        "| **Polar surrogate** | *Roughly what polar will this airfoil have, before I spend time on CFD?* "
        "Also: retrain it on your polars and test how good it is. | instant; retraining a few minutes |\n"
        "| **Learned SA correction** | *How much better does a machine-learned turbulence correction match "
        "wind-tunnel data?* | results only (computed from the command line) |\n"
        "| **Learned bubble model** | *Can machine learning fix the low-Re laminar-bubble model's drag?* | "
        "results only |")
    st.caption("For a normal airfoil polar use Airfoil → Airfoil polar; for wings and aircraft the "
               "Wing and aircraft pages. These tools are for going further.")

tab_g, tab_d, tab_m, tab_s, tab_p, tab_f, tab_b = st.tabs(
    ["Adjoint gradients", "Adjoint design", "Mesh accuracy", "Swept-wing transition", "Polar surrogate",
     "Learned SA correction", "Learned bubble model"])

GOALS = {"Lower drag (Cd)": ("Cd", 1.0), "More lift (Cl)": ("Cl", -1.0),
         "More nose-up moment (Cm)": ("Cm", -1.0)}


def shape_push(spec, grads, goal, order=8, step=0.001):
    """the surface change of steepest improvement within the shape parameters: dA = -s g
    (s = +1 to lower the quantity), so each surface moves up by -s C(x) sum_j g_j B_j(x)
    (the adjoint's shape variables: d y = C(x) B_j(x), C = sqrt(x)(1 - x)). Scaled to move
    the surface at most `step` chords, with the linear change of every coefficient."""
    from aerolab.airfoil import cst
    key, s = GOALS[goal]
    g = np.array(grads[key])
    k = order + 1
    o = C.raw_outline(spec)
    i = int(np.argmin(o[:, 0]))
    sides = {"upper": o[:i + 1][::-1], "lower": o[i:]}          # each from the leading edge back
    x = 0.5 * (1 - np.cos(np.linspace(0.06, 0.97, 22) * np.pi))

    def dy(gs, xx):
        return np.sqrt(xx) * (1 - xx) * (cst.bernstein(xx, order) @ gs)
    xf = np.linspace(0.0, 1.0, 400)
    peak = max(np.abs(dy(g[:k], xf)).max(), np.abs(dy(g[k:], xf)).max(), 1e-300)
    t = step / peak
    dA = -s * t * g
    change = {o_: float(np.dot(grads[o_], dA)) for o_ in grads}
    out = {"change": change, "sides": {}}
    for j, (name, P) in enumerate(sides.items()):
        gs = g[:k] if name == "upper" else g[k:]
        ys = np.interp(x, P[:, 0], P[:, 1])
        out["sides"][name] = {"x": x, "y": ys, "d": -s * dy(gs, x) / peak, "outline": P,
                              "d_full": -s * dy(gs, P[:, 0].clip(0, 1)) / peak}
    return out


def push_figure(spec, res):
    """airfoil with arrows: which way to push each part of the surface"""
    fig = go.Figure()
    for name, sd in res["sides"].items():
        P = sd["outline"]
        fig.add_scatter(x=P[:, 0], y=P[:, 1], mode="lines", line=dict(color=ui.INK, width=2),
                        name="airfoil", showlegend=name == "upper", hoverinfo="skip")
        fig.add_scatter(x=P[:, 0], y=P[:, 1] + 0.025 * sd["d_full"], mode="lines",
                        line=dict(color=ui.GRAPHITE, width=1.5, dash="dot"),
                        name="new shape (exaggerated)", showlegend=name == "upper", hoverinfo="skip")
        outward = 1.0 if name == "upper" else -1.0
        for x, y, d in zip(sd["x"], sd["y"], sd["d"]):
            if abs(d) < 0.08:
                continue
            col = "#C2571A" if d * outward > 0 else "#1D4E89"
            fig.add_annotation(x=x, y=y + 0.07 * d, ax=x, ay=y, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowsize=1.0, arrowwidth=2, arrowcolor=col, text="")
    for lab, col in (("push out (thicker here)", "#C2571A"), ("pull in (thinner here)", "#1D4E89")):
        fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=col, width=3), name=lab)
    fig.update_layout(xaxis=dict(title="x/c", range=[-0.05, 1.05]),
                      yaxis=dict(title="y/c", scaleanchor="x", range=[-0.2, 0.22]), height=430,
                      legend=dict(orientation="h", y=-0.2))
    return fig


# ----------------------------------------------------------------------
with tab_g:
    explain("Runs the CFD once for your airfoil, then one extra *adjoint* solve. Together they give the "
            "gradient (sensitivity) of Cl, Cd and Cm to the angle of attack, the Reynolds number and 18 "
            "shape parameters (CST coefficients, 9 on the upper and 9 on the lower surface, from leading to "
            "trailing edge). Finite differences would need one CFD run per parameter, 20 runs instead of "
            "about 2.",
            "To see which part of the airfoil to change to cut drag or add lift, and how sensitive the airfoil "
            "is to the angle and Re, before redesigning it by hand.",
            "The table gives the change per degree of angle and per unit of Re. The drawing shows the airfoil "
            "with arrows: which way to move each part of the surface to reach your goal (lower drag, more lift "
            "or more nose-up moment), and how strongly; orange pushes the surface out, blue pulls it in, and "
            "the longest arrows mark where the shape matters most. Next to it: what a small move (0.1 % of "
            "the chord) would change, and the side effects on the other coefficients. The raw gradient of the "
            "18 shape coefficients is in the box below the drawing.")
    st.caption("dCl, dCd, dCm with respect to the angle of attack, the Reynolds number and all 18 CST "
               "shape coefficients, for one flow solution + one adjoint solve. Fully turbulent or fixed "
               "transition (a free-transition location is frozen in the adjoint).")
    spec, sname = views.airfoil_picker("g", "Airfoil")
    c2, c3, c4 = st.columns(3)
    re = c2.number_input("Reynolds number", 1e5, 5e7, 3e6, format="%.3g")
    alpha = c3.number_input("Angle of attack (deg)", -10.0, 15.0, 2.0)
    level = c4.selectbox("Mesh level", [0.5, 0.7, 1.0], index=0,
                         help="The adjoint's sparse LU grows faster than linearly: ~1 min at 0.5, "
                              "~2 min at 0.7, several minutes at 1.0")
    go_ = st.button("Compute gradients", type="primary", disabled=spec is None, key="g_go")
    if go_:
        stem = _re.sub(r"[^A-Za-z0-9]+", "_", str(sname))[:30]
        out = C.CASES / f"adjoint_{stem}_Re{re:.0e}_a{alpha:g}".replace("+", "")
        out.mkdir(parents=True, exist_ok=True)
        jid = C.start_job("adjoint", f"adjoint gradients {C.display_name(sname)} alpha {alpha:g}",
                          ["adjoint.py", "gradient", spec, "--re", f"{re:.6g}", "--alpha", f"{alpha:g}",
                           "--level", level, "--out", out / "adjoint.json"], out)
        st.session_state["adj_job"] = jid
    jid = st.session_state.get("adj_job")
    if jid:
        views.live_job(jid, "adjoint")
        j = C.get_job(jid)
        f = Path(j["out"]) / "adjoint.json"
        if j["status"] == "finished" and f.exists():
            g = json.loads(f.read_text())
            st.markdown(f"**Cl {g['Cl']:.4f}, Cd {g['Cd']:.5f}, Cm {g['Cm']:+.4f}** — {g['cells']} cells, "
                        f"{g['unknowns']} adjoint unknowns")
            rows = [{"objective": k, "d/dalpha (per deg)": g["d_dalpha"][k], "d/dRe": g["d_dRe"][k]}
                    for k in ("Cl", "Cd", "Cm")]
            st.dataframe(rows, hide_index=True)
            if "d_dcst" in g:
                goal = st.radio("I want", list(GOALS), horizontal=True, key="adj_goal")
                pr = shape_push(g["airfoil"], g["d_dcst"], goal, g.get("cst_order", 8))
                ch = pr["change"]
                c1, c2 = st.columns([3, 1])
                ui.plot(push_figure(g["airfoil"], pr), key="adj_push", container=c1)
                with c2:
                    st.markdown("**If the surface moves this way, by at most 0.1 % of the chord:**")
                    st.metric("Cd", f"{1e4 * ch['Cd']:+.2f} counts", f"{100 * ch['Cd'] / g['Cd']:+.2f} %",
                              delta_color="inverse")
                    st.metric("Cl", f"{ch['Cl']:+.4f}", f"{100 * ch['Cl'] / g['Cl']:+.2f} %" if g["Cl"] else None)
                    st.metric("Cm", f"{ch['Cm']:+.4f}", delta_color="off")
                    sd = pr["sides"]
                    name, j = max(((n, int(np.argmax(np.abs(v["d"])))) for n, v in sd.items()),
                                  key=lambda t: abs(sd[t[0]]["d"][t[1]]))
                    st.caption(f"Biggest effect: the {name} surface near x/c = {sd[name]['x'][j]:.2f}. "
                               "Linear estimate at the same angle of attack (1 drag count = 0.0001).")
                key_, _ = GOALS[goal]
                others = [o_ for o_ in ("Cd", "Cl") if o_ != key_]
                if any(ch[o_] * (1 if o_ == "Cd" else -1) > 0 for o_ in others):
                    st.caption("Note the side effect: at a fixed angle this change also makes "
                               + " and ".join({"Cd": "the drag higher", "Cl": "the lift lower"}[o_] for o_ in others
                                              if ch[o_] * (1 if o_ == "Cd" else -1) > 0)
                               + "." + (" Adjoint design does the trade-off for you: it lowers the drag while "
                                        "holding the lift you need." if key_ in ("Cd", "Cl") else ""))
                st.caption("Arrows: which way to move each part of the surface, and how strongly (longer = more "
                           "effect). Orange pushes the surface out of the airfoil, blue pulls it in. The dotted line "
                           "is the new shape, exaggerated so it is visible. The movement is the steepest improvement "
                           "the 18 shape parameters can make; it changes the airfoil smoothly, not point by point.")
                with st.expander("Raw shape gradient (the 18 CST coefficients)"):
                    obj = st.radio("Gradient of", ["Cd", "Cl", "Cm"], horizontal=True, key="adj_raw")
                    v = np.array(g["d_dcst"][obj])
                    k = len(v) // 2
                    fig = go.Figure([go.Bar(x=list(range(k)), y=v[:k], name="upper surface"),
                                     go.Bar(x=list(range(k)), y=v[k:], name="lower surface")])
                    fig.update_layout(xaxis_title="CST coefficient (0 = leading edge)",
                                      yaxis_title=f"d{obj}/dA", barmode="group", height=320)
                    ui.plot(fig, key="adj_cst")

# ----------------------------------------------------------------------
with tab_d:
    explain("An optimiser changes the airfoil shape (the 18 CST coefficients) and the angle of attack step by "
            "step to minimise the drag while keeping the lift at your target Cl. The adjoint gradients tell it "
            "which way to move each time, so a design takes minutes instead of days. Optional limits: a "
            "minimum thickness (for the spar and structure) and a minimum Cm (to limit the trim drag of the "
            "tail). The final shape is checked on the validated mesh.",
            "When you know your cruise lift coefficient and want a better airfoil for it than a catalogue one: "
            "start from a similar airfoil (e.g. NACA 2412) and set the Cl you fly at.",
            "Left: the drag (in counts, 1 count = 0.0001) of every design tried, with the lift held at the "
            "target; middle: the start and optimised shapes, with their drag checked on the validated mesh "
            "above; right: the shape gradient of the start airfoil. The result is a design point: run a full polar of the new airfoil on "
            "the Airfoil polar page to check it away from the design Cl.")
    st.caption("Minimum drag at a target lift: SLSQP with adjoint gradients over all 18 CST coefficients "
               "and the angle of attack, every design a deformation of one mesh, then a check on the "
               "validated mesh. ~1 min per design iteration at mesh level 0.5.")
    spec, dname = views.airfoil_picker("d", "Start airfoil")
    c2, c3, c4 = st.columns(3)
    re = c2.number_input("Reynolds number", 1e5, 5e7, 3e6, format="%.3g", key="d_re")
    cl = c3.number_input("Target Cl", 0.0, 1.5, 0.5, key="d_cl")
    tmin = c4.number_input("Minimum t/c (0 = keep the start)", 0.0, 0.3, 0.0, key="d_t")
    c1, c2 = st.columns(2)
    iters = c1.slider("Outer iterations", 2, 30, 12)
    cm = c2.number_input("Minimum Cm (blank = none)", -1.0, 1.0, value=None, key="d_cm")
    go_d = st.button("Optimise", type="primary", disabled=spec is None, key="d_go")
    if go_d:
        out = C.CASES / f"design_adjoint_{_re.sub(r'[^A-Za-z0-9]+', '_', str(dname))[:30]}_cl{cl:g}"
        args = ["design_adjoint.py", "--start", spec, "--re", f"{re:.6g}", "--cl", f"{cl:g}",
                "--iters", iters, "--out", out]
        if tmin > 0:
            args += ["--t-min", f"{tmin:g}"]
        if cm is not None:
            args += ["--cm-min", f"{cm:g}"]
        st.session_state["des_job"] = C.start_job("adjoint_design", f"adjoint design {C.display_name(dname)} Cl {cl:g}", args, out)
    jid = st.session_state.get("des_job")
    if jid:
        views.live_job(jid, "adjoint_design")
        j = C.get_job(jid)
        if (Path(j["out"]) / "design.png").exists():
            st.image(str(Path(j["out"]) / "design.png"))

# ----------------------------------------------------------------------
with tab_m:
    explain("Every CFD result depends a little on the mesh: on a finer mesh the numbers move. This tab "
            "measures how much, for one airfoil and angle. **Error of one result** solves the case on the "
            "normal mesh and on a mesh with every cell split in four, and reports the difference: that is "
            "the real mesh error of the normal result. An adjoint solve also maps **where** the error comes "
            "from (which cells). **Mesh adaptation** uses that map to refine the mesh only where it matters "
            "and compares it with refining everywhere, against a fine reference mesh.",
            "Before trusting small differences between airfoils (a few drag counts), or when a result looks "
            "odd: if the mesh error is bigger than the difference you care about, use a finer mesh level. "
            "For a whole polar the Airfoil polar page has a grid check; this tab looks at one point in depth.",
            "**Normal mesh** is the value a polar gives; **4× finer mesh** the value with every cell split "
            "in four; **mesh error** their difference (1 drag count = 0.0001). Under about 1 % in Cd and "
            "0.005 in Cl is good. The map shows where the error comes from: bright cells matter most. "
            "If most of it is in the boundary-layer cells next to the wall, a finer mesh level helps; if "
            "it is in the wake or around the nose, adaptation helps. **Adjoint estimate** is an "
            "experimental prediction of the fine value without solving it: our validation found it "
            "overshoots the lift and moment error about 2× and does not estimate the drag error, so "
            "trust the solved fine value.")
    mode = st.segmented_control("Study", ["Error of one result", "Mesh adaptation"],
                                default="Error of one result", key="m_mode") or "Error of one result"
    m_spec, m_name = views.airfoil_picker("m", "Airfoil")
    c1, c2, c3, c4 = st.columns(4)
    m_re = c1.number_input("Reynolds number", 2e4, 5e7, 1e6, format="%.3g", key="m_re")
    m_al = c2.number_input("Angle of attack (deg)", -10.0, 15.0, 4.0, key="m_a")
    m_lev = c3.selectbox("Mesh level", [0.5, 0.7, 1.0], key="m_lev",
                         help="The normal mesh. The finer mesh has 4× its cells.")
    m_tr = c4.selectbox("Transition", ["free", "off"], key="m_tr",
                        format_func=lambda s: {"free": "free (e^N)", "off": "fully turbulent"}[s])
    if mode == "Error of one result":
        check = st.checkbox("Measure the real error (solve the 4× finer mesh)", True, key="m_check",
                            help="Off: only the error map and the experimental adjoint estimate, ~5× faster.")
        st.caption("About 10–40 min at level 0.5 with the finer-mesh solve; a few minutes without.")
    else:
        c1, c2 = st.columns(2)
        m_obj = c1.selectbox("Quantity to adapt for", ["Cd", "Cl", "Cm"], key="m_obj")
        m_cyc = c2.slider("Adaptation cycles", 1, 4, 3, key="m_cyc")
        st.caption("About 1–2 h: every cycle is a flow + adjoint solve; then the same number of uniformly "
                   "refined meshes and a fine reference mesh (level 1.4) for comparison.")
    if st.button("Run", type="primary", key="m_go", disabled=m_spec is None):
        stem = _re.sub(r"[^A-Za-z0-9]+", "_", str(m_name))[:30]
        tag = f"{stem}_Re{m_re:.2g}_a{m_al:g}_{int(time.time())}".replace("+", "")
        common = [m_spec, "--re", f"{m_re:.6g}", f"--alpha={m_al:g}", "--level", m_lev, "--transition", m_tr]
        if mode == "Error of one result":
            out = C.CASES / f"mesh_error_{tag}"
            out.mkdir(parents=True, exist_ok=True)
            args = ["adjoint_error.py", "estimate", *common, "--out", out / "estimate.json"] + (["--check"] if check else [])
            st.session_state["m_job"] = C.start_job("mesh_error", f"Mesh error: {m_name} α {m_al:g}", args, out)
        else:
            out = C.CASES / f"mesh_adapt_{tag}"
            args = ["adjoint_error.py", "adapt", *common, "--objective", m_obj, "--cycles", m_cyc, "--out", out]
            st.session_state["m_job"] = C.start_job("mesh_adapt", f"Mesh adaptation: {m_name} α {m_al:g}", args, out)
    jid = st.session_state.get("m_job")
    if jid:
        views.live_job(jid, "mesh_error")

    def mesh_error_view(out):
        e = json.loads((out / "estimate.json").read_text())
        rows = []
        for k, o in e["objectives"].items():
            row = {"": k, "normal mesh": o["coarse"], "4× finer mesh": o.get("fine_solved"),
                   "mesh error": o.get("error_coarse"),
                   "error %": (100 * o["error_coarse"] / o["fine_solved"]
                               if o.get("fine_solved") not in (None, 0) else None),
                   "adjoint estimate (experimental)": o["estimate_fine"],
                   "error from wall cells": f"{100 * o['indicator_share_bl_quads']:.0f} %"}
            rows.append(row)
        st.markdown(f"**{C.display_name(Path(str(e['airfoil'])).stem)}**, Re {C.fmt_re(e['re'])}, α {e['alpha']:g}°, "
                    f"level {e['level']:g}: {e['cells_coarse']:,} cells, finer mesh {e['cells_fine']:,}.")
        cd = e["objectives"].get("Cd", {})
        if cd.get("fine_solved") is not None:
            k = st.columns(3)
            k[0].metric("Mesh error in Cd", f"{1e4 * cd['error_coarse']:+.1f} counts",
                        f"{100 * cd['error_coarse'] / cd['fine_solved']:+.1f} %", delta_color="off")
            cl = e["objectives"]["Cl"]
            k[1].metric("Mesh error in Cl", f"{cl['error_coarse']:+.4f}",
                        f"{100 * cl['error_coarse'] / cl['fine_solved']:+.1f} %" if cl["fine_solved"] else None,
                        delta_color="off")
            ok = abs(cd["error_coarse"]) < 0.01 * abs(cd["fine_solved"]) and abs(cl["error_coarse"]) < 0.005
            k[2].metric("Verdict", "fine as it is" if ok else "use a finer level")
        st.dataframe(rows, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.5f") for c in
                                    ("normal mesh", "4× finer mesh", "mesh error", "adjoint estimate (experimental)")}
                     | {"error %": st.column_config.NumberColumn(format="%+.2f")})
        f = out / "indicator.npz"
        if f.exists():
            d = np.load(f)
            obj = st.radio("Error map of", [k for k in ("Cd", "Cl", "Cm") if f"eta_{k}" in d], horizontal=True,
                           key=f"m_map{out.name}")
            xc, eta = d["xc"], d[f"eta_{obj}"]
            sel = (xc[:, 0] > -0.3) & (xc[:, 0] < 1.6) & (np.abs(xc[:, 1]) < 0.45)
            v = np.log10(np.maximum(eta[sel], eta.max() * 1e-8))
            fig = go.Figure(go.Scattergl(x=xc[sel, 0], y=xc[sel, 1], mode="markers",
                                         marker=dict(size=3, color=v, colorscale="Turbo",
                                                     colorbar=dict(title="log10 share"))))
            fig.update_layout(xaxis_title="x/c", yaxis=dict(title="y/c", scaleanchor="x"), height=420,
                              title=f"Where the mesh error in {obj} comes from (bright = most)")
            ui.plot(fig, key=f"m_fig{out.name}")

    def mesh_adapt_view(out):
        d = json.loads((out / "adapt.json").read_text())
        obj = d["objective"]
        fig = go.Figure()
        for kind, dash in (("adapted", "solid"), ("uniform", "dash")):
            r = [x for x in d["rows"] if x["kind"] == kind]
            fig.add_scatter(x=[x["cells"] for x in r], y=[abs(x["error"]) for x in r], mode="lines+markers",
                            name="adapted mesh" if kind == "adapted" else "refined everywhere", line=dict(dash=dash))
        fig.update_layout(xaxis=dict(type="log", title="cells"), yaxis=dict(type="log", title=f"|error in {obj}|"),
                          height=380, title=f"Error against a fine reference ({d['reference_cells']:,} cells, "
                                            f"{obj} {d['reference']:.5f})")
        ui.plot(fig, key=f"m_ad{out.name}")
        st.dataframe([{"mesh": x["kind"], "cells": x["cells"], obj: x[obj], "error": x["error"]} for x in d["rows"]],
                     hide_index=True, width="stretch")
        st.caption("The lower curve reaches a given error with fewer cells. Only the triangle region is adapted; "
                   "the boundary-layer cells stay as validated.")

    if jid and C.get_job(jid)["status"] == "finished":
        o = Path(C.get_job(jid)["out"])
        (mesh_error_view if (o / "estimate.json").exists() else mesh_adapt_view)(o)
    past = sorted(list(C.CASES.glob("mesh_error_*/estimate.json")) + list(C.CASES.glob("mesh_adapt_*/adapt.json")),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if past:
        with st.expander(f"Earlier results ({len(past)})"):
            pk = st.selectbox("Result", past, format_func=lambda p: p.parent.name, key="m_past")
            (mesh_error_view if pk.name == "estimate.json" else mesh_adapt_view)(pk.parent)

# ----------------------------------------------------------------------
with tab_s:
    explain("A boundary layer starts laminar (low friction) and turns turbulent at some point. On a swept "
            "wing there are three ways that can happen, and this tab checks all three on an infinite swept "
            "wing: (1) the **attachment line**, the line along the leading edge where the flow splits, can "
            "be turbulent from the start; (2) **crossflow**, the sideways flow sweep creates inside the "
            "boundary layer, grows waves that trip it; (3) **Tollmien–Schlichting waves**, the usual "
            "mechanism on straight wings (the e^N method).",
            "For swept wings (more than about 15–20° of sweep) where laminar flow and drag matter: gliders, "
            "business jets, laminar-flow research. On a straight wing (sweep near 0) it reduces to the "
            "ordinary transition prediction of the Airfoil polar page. **One section** checks one airfoil "
            "at one sweep; **Whole wing** takes a saved aircraft's wing (or one you enter), finds each "
            "station's local angle and Reynolds number with the vortex lattice and checks 8 stations "
            "from root to tip.",
            "**Transition, upper / lower** is where the flow turns turbulent, as a fraction of the chord "
            "(0 = leading edge, 1 = trailing edge), and the mechanism that did it. Further aft means more "
            "laminar flow and less drag. **Attachment line Rbar** is the Reynolds number of the leading-edge "
            "flow: below ~245 it stays laminar; above ~245 turbulence from the wing root or fuselage spreads "
            "along it; above ~580 it turns turbulent by itself. In the plot, transition happens where a curve "
            "reaches the dotted line. For a whole wing: the green area of the planform is laminar on the "
            "upper surface, and the laminar fraction is the share of the whole wetted surface.")
    from aerolab.physics import swept_transition as stt
    s_mode = st.segmented_control("Analyse", ["One section", "Whole wing"], default="One section",
                                  key="s_mode") or "One section"
    if s_mode == "One section":
        st.caption("Infinite swept wing: attachment-line contamination / instability (Poll), crossflow "
                   "(Falkner-Skan-Cooke profiles) and Tollmien-Schlichting (e^N).")
        spec, _ = views.airfoil_picker("s", "Airfoil")
        c2, c3, c4 = st.columns(3)
        sweep = c2.slider("Leading-edge sweep (deg)", 0, 60, 30)
        re = c3.number_input("Reynolds number (normal chord)", 1e5, 1e8, 1e7, format="%.3g", key="s_re")
        alpha = c4.number_input("Angle of attack (deg)", -5.0, 10.0, 2.0, key="s_a")
        c1, c2, c3 = st.columns(3)
        tu = c1.number_input("Turbulence Tu (%), 0 = N_crit 9", 0.0, 2.0, 0.0, 0.01)
        chi = c2.number_input("Crossflow transition chi", 50.0, 500.0, 200.0)
        cont = c3.checkbox("Root contaminates the attachment line", False)
        if spec is None:
            st.info("Upload the airfoil's .dat file.")
        else:
            with st.spinner("boundary layers ..."):
                r = stt.analyse(C.airfoil_file(spec), float(sweep), re, alpha, tu or None, None, chi, cont)
            al = r["attachment_line"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Attachment line Rbar", f"{al['Rbar']:.0f}", al["why"],
                      help="Below ~245 laminar; 245–580 turbulent if the root or fuselage contaminates it; "
                           "above ~580 turbulent by itself")
            m2.metric("Transition, upper", f"{r['upper']['x_tr']:.3f} c", r["upper"]["mechanism"])
            m3.metric("Transition, lower", f"{r['lower']['x_tr']:.3f} c", r["lower"]["mechanism"])
            fig = go.Figure()
            for side, dash in (("upper", "solid"), ("lower", "dash")):
                d = r[side]
                fig.add_trace(go.Scatter(x=d["x"], y=d["N"], name=f"N (TS), {side}", line=dict(dash=dash)))
                fig.add_trace(go.Scatter(x=d["x"], y=np.array(d["chi"]) * 9.0 / chi,
                                         name=f"chi x 9/chi_tr, {side}", line=dict(dash=dash)))
            fig.add_hline(y=r["ncrit"], line_dash="dot")
            fig.update_layout(xaxis_title="x/c", yaxis_title="N  (crossflow scaled: transition at the line)",
                              height=380, yaxis_range=[0, 3 * r["ncrit"]])
            ui.plot(fig, key="swept")
    else:
        st.caption("Along the span of a wing: the vortex lattice gives each station's local angle and Reynolds "
                   "number, then the three mechanisms are checked at 8 stations with the local leading-edge sweep.")
        src_w = st.segmented_control("Wing from", ["Saved aircraft", "Enter a wing"], default="Saved aircraft",
                                     key="sw_src") or "Saved aircraft"
        if src_w == "Saved aircraft":
            ac, _ = AV.aircraft_picker(key="sw_ac")
            surfs = [x for x in ac.surfaces if not x.vertical]
            sf = st.selectbox("Surface", surfs, format_func=lambda x: x.name, key="sw_surf")
            w_span = sf.span if sf.symmetric else 2 * sf.span
            w_cr, w_ct, w_sw, w_tw = sf.root_chord, sf.tip_chord, sf.sweep, sf.twist
            w_root, w_tip = sf.airfoil, sf.tip_airfoil or None
            st.caption(f"Span {w_span:.4g} m, chords {w_cr:.4g} / {w_ct:.4g} m, quarter-chord sweep {w_sw:g}°, "
                       f"twist {w_tw:g}°" + (f"; panel breaks left out" if sf.kinks else "") + ".")
        else:
            c1, c2, c3 = st.columns(3)
            w_span = c1.number_input("Span (m)", 0.5, 80.0, 20.0, key="sw_b")
            w_cr = c2.number_input("Root chord (m)", 0.05, 15.0, 3.0, key="sw_cr")
            w_ct = c3.number_input("Tip chord (m)", 0.02, 15.0, 1.2, key="sw_ct")
            c1, c2 = st.columns(2)
            w_sw = c1.number_input("Quarter-chord sweep (deg)", 0.0, 50.0, 25.0, key="sw_sw")
            w_tw = c2.number_input("Tip twist (deg)", -8.0, 4.0, -2.0, key="sw_tw")
            w_root, _ = views.airfoil_picker("swr", "Section")
            w_tip = None
        c1, c2, c3 = st.columns(3)
        w_V = c1.number_input("Speed (m/s)", 5.0, 300.0, 120.0, key="sw_V")
        w_alt = c2.number_input("Altitude (m)", 0.0, 15000.0, 3000.0, 100.0, key="sw_h")
        w_al = c3.number_input("Wing angle of attack (deg)", -4.0, 12.0, 2.0, key="sw_a")
        c1, c2, c3 = st.columns(3)
        w_tu = c1.number_input("Turbulence Tu (%), 0 = N_crit 9", 0.0, 2.0, 0.0, 0.01, key="sw_tu")
        w_chi = c2.number_input("Crossflow transition chi", 50.0, 500.0, 200.0, key="sw_chi")
        w_cont = c3.checkbox("Fuselage contaminates the attachment line", True, key="sw_cont")

        @st.cache_data(show_spinner=False, max_entries=8)
        def wing_tr(span, cr, ct, sweep, twist, root, tip, V, alt, alpha, tu, chi, cont):
            from aerolab.aircraft import aircraft as A
            from aerolab.aircraft import wing as W
            wg = W.Wing(span, cr, ct, sweep, 0.0, twist, 0.0, n_half=30)
            rho, mu, a_snd, *_ = W.isa(alt)
            mach = V / a_snd
            re_strip = rho * V / mu * wg.c
            sfc = A.Surface("wing", span, cr, ct, sweep=sweep, twist=twist, airfoil=root, tip_airfoil=tip)
            pol, src = A.surface_polars(sfc, str(C.POLAR_CACHE), mach, rho * V / mu * wg.mac, wg.AR)
            sol = W.solve(wg, alpha, pol, re_strip, mach)
            t = stt.wing_transition(wg, sol, re_strip, root, tip, tu or None, None, cont, chi)
            t["CL"], t["polars"], t["mach"] = float(sol["CL"]), src, mach
            return t
        if w_root is None:
            st.info("Upload the airfoil's .dat file.")
        else:
            with st.spinner("vortex lattice and boundary layers at 8 stations ..."):
                t = wing_tr(w_span, w_cr, w_ct, w_sw, w_tw, C.airfoil_file(w_root),
                            C.airfoil_file(w_tip) if w_tip else None, w_V, w_alt, w_al, w_tu, w_chi, w_cont)
            st_ = t["stations"]
            k = st.columns(4)
            k[0].metric("Laminar part of the surface", f"{100 * t['laminar_fraction']:.0f} %",
                        help="Area-weighted, both surfaces")
            k[1].metric("Leading-edge sweep", f"{t['sweep_le']:.1f}°")
            k[2].metric("Wing CL", f"{t['CL']:.3f}")
            k[3].metric("Mach", f"{t['mach']:.3f}")
            if t["mach"] > 0.3:
                st.caption("Above Mach 0.3 the boundary layers here are still incompressible: a first estimate.")
            c1, c2 = st.columns(2)
            eta = [q["eta"] for q in st_]
            fig = go.Figure()
            fig.add_scatter(x=eta, y=[q["x_tr_upper"] for q in st_], mode="lines+markers", name="upper",
                            text=[q["mech_upper"] for q in st_], hovertemplate="η %{x:.2f}: x_tr %{y:.3f}<br>%{text}")
            fig.add_scatter(x=eta, y=[q["x_tr_lower"] for q in st_], mode="lines+markers", name="lower",
                            line=dict(dash="dash"), text=[q["mech_lower"] for q in st_],
                            hovertemplate="η %{x:.2f}: x_tr %{y:.3f}<br>%{text}")
            fig.update_layout(xaxis_title="η = 2y/b (root → tip)", yaxis=dict(title="transition x/c", range=[0, 1.05]),
                              height=360, title="Where the flow turns turbulent")
            ui.plot(fig, key="sw_xtr", container=c1)
            # planform: laminar part of the upper surface shaded
            s_ = 0.5 * w_span
            tl = math.tan(math.radians(t["sweep_le"]))
            y = np.array([0.0] + [q["eta"] * s_ for q in st_] + [s_])
            ch = w_cr + (w_ct - w_cr) * y / s_
            xle = y * tl
            xtr = xle + np.interp(y, [q["eta"] * s_ for q in st_], [q["x_tr_upper"] for q in st_]) * ch
            fig = go.Figure()
            fig.add_scatter(x=np.r_[y, y[::-1]], y=-np.r_[xle, (xle + ch)[::-1]], fill="toself", mode="lines",
                            fillcolor="rgba(120,120,120,0.18)", line=dict(color=ui.GRAPHITE), name="turbulent")
            fig.add_scatter(x=np.r_[y, y[::-1]], y=-np.r_[xle, xtr[::-1]], fill="toself", mode="lines",
                            fillcolor="rgba(43,122,75,0.35)", line=dict(width=0), name="laminar (upper surface)")
            fig.update_layout(xaxis_title="y (m)", yaxis=dict(title="x (m)", scaleanchor="x"), height=360,
                              title="Right half wing, top view")
            ui.plot(fig, key="sw_plan", container=c2)
            st.dataframe([{"η": q["eta"], "Re": C.fmt_re(q["re"]), "local α": q["alpha_eff"], "Rbar": q["Rbar"],
                           "attachment line": q["attachment_line"], "x_tr upper": q["x_tr_upper"],
                           "upper: why": q["mech_upper"], "x_tr lower": q["x_tr_lower"], "lower: why": q["mech_lower"]}
                          for q in st_], hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.3f")
                                        for c in ("η", "x_tr upper", "x_tr lower")}
                         | {"local α": st.column_config.NumberColumn(format="%.2f"),
                            "Rbar": st.column_config.NumberColumn(format="%.0f")})
            st.caption(f"Section polars for the lattice: {t['polars']}. The tip section is used outboard of mid-span.")

# ----------------------------------------------------------------------
with tab_p:
    from aerolab.design import surrogate as SG
    explain("A statistical model (a Gaussian process) trained on every CFD polar computed so far. From the "
            "airfoil's shape (thickness, camber, their positions, leading-edge radius, flap) and the Reynolds "
            "number it predicts Cl and Cd at every angle **instantly**, with an uncertainty band. It also "
            "says at which angle a new CFD run would teach it the most.",
            "For a quick first look at an airfoil before spending 15–60 minutes on a CFD polar, and to compare "
            "many airfoils fast. It is an estimate, not a CFD result.",
            "The line is the best estimate and the shaded band the range the CFD result would fall in "
            "(±2 standard deviations, about 95 %). A **narrow band** means the airfoil is similar to polars "
            "already computed: trust it. A **wide band** means it is unlike them: run the real polar. "
            "The suggested angle is the most uncertain point; each polar you run makes it better for "
            "similar airfoils.")
    st.caption("Gaussian process over every polar in the library; mean and ±2 sd (calibrated by "
               "leaving polars out). Retrain it after adding polars: see the box below.")
    spec, _ = views.airfoil_picker("p", "Airfoil", allow_flap=True)
    c2, c3 = st.columns(2)
    re = c2.number_input("Reynolds number", 5e4, 3e7, 3e6, format="%.3g", key="p_re")
    turb = c3.checkbox("Fully turbulent", False)
    if spec is None:
        st.info("Upload the airfoil's .dat file to see its estimate.")
    else:
        spec = C.airfoil_file(spec)
        try:
            sg = SG.Surrogate.load() if SG.model_path().exists() else SG.Surrogate.train()
            al = np.arange(-4, 16.5, 0.5)
            p = sg.predict(spec, re, al, 0.0 if turb else 1.0)
            a_next, _, _ = sg.suggest(spec, re, free=0.0 if turb else 1.0)
            cols = st.columns(2)
            for col, k, lab in ((cols[0], "Cl", "Cl"), (cols[1], "Cd", "Cd")):
                fig = go.Figure([
                    go.Scatter(x=np.r_[al, al[::-1]], y=np.r_[p[k] + 2 * p[k + "_sd"], (p[k] - 2 * p[k + "_sd"])[::-1]],
                               fill="toself", line=dict(width=0), name="+-2 sd", opacity=0.3),
                    go.Scatter(x=al, y=p[k], name="mean")])
                fig.update_layout(xaxis_title="alpha (deg)", yaxis_title=lab, height=340)
                ui.plot(fig, key=f"sg_{k}", container=col)
            st.info(f"Most useful next CFD run for this airfoil and Re: α {a_next:g}° (the most uncertain point; "
                    f"the model is trained on {sg.n} points from {len(sg.sources)} polars).")
        except Exception as e:
            st.warning(f"Surrogate unavailable: {e}")

    with st.expander("Keep it up to date: retrain and test", expanded=False):
        mp = SG.model_path()
        try:
            md = json.loads(mp.read_text())
            n_src, n_pts = len(md["sources"]), md["n"]
        except (OSError, ValueError, KeyError):
            md, n_src, n_pts = {}, 0, 0
        n_files = len({f.name for f in list((SG.ROOT / "data" / "wing_polars").glob("*.json")) +
                       list(C.POLAR_CACHE.glob("*.json"))}) + len(list(C.CASES.rglob("polar.json"))) + \
            len(list((SG.ROOT / "validation").glob("transition_*.json")))
        which = ("retrained on this machine, " + time.strftime("%d %b %Y %H:%M", time.localtime(mp.stat().st_mtime))
                 if mp == SG.USER_MODEL else "the one shipped with AeroLab")
        st.markdown(f"Model in use: **{which}**, {n_pts} points from {n_src} polars. "
                    f"Polars available for training now: about {n_files}"
                    + (" (retrain to use the new ones)." if n_files > n_src else "."))
        st.caption("**Retrain** learns from every converged polar in the library, including the ones you ran "
                   "since (Airfoil polar page and the Wing tool's section polars), and recalibrates the ±2 sd band. "
                   "**Test** leaves each polar out in turn, predicts it from all the others, and compares: "
                   "the honest measure of how good the surrogate is for an airfoil it has not seen. "
                   "Each takes a few minutes and runs in the background.")
        c1, c2 = st.columns(2)
        if c1.button("Retrain now", key="sg_train", width="stretch"):
            out = C.CASES / "_surrogate" / f"train_{int(time.time())}"
            st.session_state["sg_job"] = C.start_job("surrogate_train", "Retrain the polar surrogate",
                                                     ["surrogate.py", "train", "--out", out], out)
        if c2.button("Test it: leave one polar out", key="sg_val", width="stretch"):
            out = C.CASES / "_surrogate" / f"validate_{int(time.time())}"
            st.session_state["sg_job"] = C.start_job("surrogate_validate", "Test the polar surrogate",
                                                     ["surrogate.py", "validate", "--out", out], out)
        if st.session_state.get("sg_job"):
            views.live_job(st.session_state["sg_job"], "surrogate")
        tests = sorted(C.CASES.glob("_surrogate/validate_*/validate.json"))
        if tests:
            v = json.loads(tests[-1].read_text())["polars"]
            st.markdown(f"**Latest test** ({time.strftime('%d %b %Y %H:%M', time.localtime(tests[-1].stat().st_mtime))}): "
                        "each row is a polar predicted by a model that never saw it.")
            st.dataframe([{"polar left out": r["polar"], "points": r["n"], "up to α": r["alpha_max"],
                           "Cl error (rms), α ≤ 12°": r["rms_Cl_prestall"], "Cl error (rms), all": r["rms_Cl"],
                           "Cd error, α ≤ 12°": None if r["rms_Cd_rel_prestall"] is None else 100 * r["rms_Cd_rel_prestall"],
                           "Cd error, all": 100 * r["rms_Cd_rel"],
                           "Cl inside the band": 100 * r["cover_Cl_2sd"], "Cd inside the band": 100 * r["cover_Cd_2sd"]}
                          for r in v], hide_index=True, width="stretch",
                         column_config={"Cl error (rms), α ≤ 12°": st.column_config.NumberColumn(format="%.3f"),
                                        "Cl error (rms), all": st.column_config.NumberColumn(format="%.3f"),
                                        "Cd error, α ≤ 12°": st.column_config.NumberColumn(format="%.1f %%"),
                                        "Cd error, all": st.column_config.NumberColumn(format="%.1f %%"),
                                        "Cl inside the band": st.column_config.NumberColumn(format="%.0f %%"),
                                        "Cd inside the band": st.column_config.NumberColumn(format="%.0f %%")})
            pre = [r["rms_Cl_prestall"] for r in v if r["rms_Cl_prestall"] is not None]
            pcd = [r["rms_Cd_rel_prestall"] for r in v if r["rms_Cd_rel_prestall"] is not None]
            if pre:
                st.caption(f"Below stall (α ≤ 12°) it is typically off by {np.median(pre):.3f} in Cl and "
                           f"{100 * np.median(pcd):.0f} % in Cd for an airfoil it has not seen (median over the "
                           "polars). The band columns use the raw model's ±2 sd: a raw Gaussian process is over-confident, "
                           "so they fall short of 95 %. That is why training widens the band "
                           + (", here by " + ", ".join(f"{k} ×{v:.2f}" for k, v in md.get("calibration", {}).items())
                              if md.get("calibration") else "") +
                           " so that 95 % of left-out points fall inside it. Large errors on one row mean that "
                           "kind of airfoil, Re or flap is under-represented: run a polar of it and retrain.")
        elif (SG.ROOT / "validation" / "surrogate_report.md").exists():
            st.caption("No test run on this machine yet. The test from the AeroLab validation "
                       "(validation/surrogate_report.md) is on the Validation page.")

# ----------------------------------------------------------------------
with tab_f:
    explain("The turbulence model of the CFD (Spalart–Allmaras) is not perfect, especially near stall. "
            "*Field inversion* finds, for airfoils with wind-tunnel data, the correction to the model's "
            "turbulence production at every point of the flow that makes the CFD match the measured lift "
            "and pressures. A neural network then learns that correction from local flow features, so it "
            "can be applied to airfoils it has never seen. The real test: each airfoil family is predicted "
            "by a network that was trained **without** it.",
            "It is a research result to read, not a tool to run here: it shows how far machine learning can "
            "improve the CFD's lift near stall. It does not change the other pages' results.",
            "In **Held-out predictions** compare Cl_SA (plain CFD) and Cl_FIML (CFD with the learned "
            "correction) against cl_exp (wind tunnel): the closer to cl_exp, the better. cp_rms is the "
            "average pressure error over the surface: smaller is better. **Inversions** are the training "
            "cases: how closely the inversion itself matched the data (Cl_inv) and how large the correction "
            "had to be (beta_min, beta_max; 1 = no correction).")
    st.caption("Field inversion of the SA production multiplier against measured lift and pressure "
               "(NACA 0012, NACA 4412, S809, E387), a neural network beta(local flow features), and a "
               "leave-one-airfoil-out test: each family is predicted by a model that never saw it. "
               "Run: `python src/run.py fiml_multi invert`, `loao`, `summary`, `plot`.")
    root = C.SRC.parent
    summ = root / "validation" / "fiml_multi_summary.json"
    png = root / "validation" / "fiml_multi.png"
    if not summ.exists():
        st.info("No results yet: run the commands above (about 3 h on 4 cores at mesh level 0.5).")
    else:
        d = json.loads(summ.read_text())
        import pandas as pd
        st.markdown("**Held-out predictions** (model trained without the airfoil's family)")
        pr = pd.DataFrame(d["predictions"])
        cols = [c for c in ("family", "re", "alpha", "cl_exp", "Cl_SA", "Cl_FIML", "cp_rms_SA",
                            "cp_rms_FIML") if c in pr]
        st.dataframe(pr[cols].sort_values(["family", "re", "alpha"]), hide_index=True,
                     width="stretch")
        st.markdown("**Inversions** (training data)")
        iv = pd.DataFrame(d["inversions"])
        cols = [c for c in ("name", "cl_target", "Cl_SA", "Cl_inv", "cp_rms_SA", "cp_rms_inv",
                            "beta_min", "beta_max", "solutions") if c in iv]
        st.dataframe(iv[cols], hide_index=True, width="stretch")
        if png.exists():
            st.image(str(png))

# ----------------------------------------------------------------------
with tab_b:
    explain("At low Reynolds numbers (model aircraft, UAVs, roughly Re 50 000–500 000) the laminar flow "
            "often separates, turns turbulent in the air and reattaches: a **laminar separation bubble**. "
            "Its length changes the drag a lot. The CFD's low-Re mode sets the bubble length with one "
            "empirical constant K for every case (Mayle's correlation, K = 1000). This study tried to "
            "**learn** K from wind-tunnel data instead: for each Eppler 387 case measured at NASA Langley "
            "(LTPT), it ran the CFD at a ladder of K values, found the K that matches the measured drag, and "
            "trained a statistical model (Gaussian process) to predict that K from the state of the boundary "
            "layer where it separates. Each case was then predicted by a model trained without it.",
            "To see how far to trust the drag of low-Re polars, and why. It is a result to read: the learned "
            "model did **not** beat the constant, so it is not used by the other pages.",
            "**error K = 1000** is the drag error of the normal low-Re model against the wind tunnel; "
            "**error learned** with the learned K. At Re 3e5 the normal model is within 1–2 %; at Re 1e5 "
            "it under-predicts the drag by 13–22 %, and no bubble length fixes that. The lesson: at very "
            "low Re the error is in how the turbulence model treats the reattaching shear layer, not in "
            "where transition starts. For your own low-Re polars expect the drag to be a little low near "
            "Re 1e5 and good from about Re 2–3e5.")
    rep = C.SRC.parent / "validation" / "bubble_ml_report.md"
    if not rep.exists():
        st.info("No results yet: `python src/run.py bubble_ml scan` then `fit` (about 25 min).")
    else:
        txt = rep.read_text()
        rows = []
        for line in txt.splitlines():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) == 9 and _re.match(r"^\d", c[0]):
                pct = lambda x: float(x.replace("%", "").replace("+", "").strip())
                rows.append({"Re": c[0], "α (deg)": float(c[1]), "Cd wind tunnel": float(c[4]),
                             "Cd, K = 1000": float(c[5]), "Cd, learned K": float(c[6]),
                             "error K = 1000 (%)": pct(c[7]), "error learned (%)": pct(c[8]),
                             "K matching the tunnel": c[2], "K predicted": c[3]})
        if rows:
            m = _re.search(r"K = 1000 ([\d.]+) %, learned \(leave-one-out\) ([\d.]+) %", txt)
            if m:
                k = st.columns(2)
                k[0].metric("Mean drag error, normal model (K = 1000)", f"{m.group(1)} %")
                k[1].metric("Mean drag error, learned K", f"{m.group(2)} %",
                            f"{float(m.group(2)) - float(m.group(1)):+.1f} points", delta_color="inverse")
            lab = [f"Re {r['Re']}, α {r['α (deg)']:g}°" for r in rows]
            fig = go.Figure([go.Bar(x=lab, y=[r["error K = 1000 (%)"] for r in rows], name="normal model (K = 1000)"),
                             go.Bar(x=lab, y=[r["error learned (%)"] for r in rows], name="learned K")])
            fig.add_hline(y=0, line_color=ui.GRAPHITE)
            fig.update_layout(barmode="group", yaxis_title="drag error vs wind tunnel (%)", height=360,
                              title="Eppler 387: drag error of the low-Re model (each case held out)")
            ui.plot(fig, key="bub")
            st.dataframe(rows, hide_index=True, width="stretch")
        if "## Findings" in txt:
            with st.expander("Findings (full text of the study)"):
                st.markdown(txt[txt.index("## Findings"):])
