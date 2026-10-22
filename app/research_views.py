"""Shared views of the Research page (tools/25_Research.py): the plain-language guide of a tab,
and the adjoint shape-sensitivity and design views (also used by the Airfoil optimiser)."""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import common as C
import ui


def explain(what, when, read):
    """plain-language guide at the top of a tab"""
    with st.expander("What is this?", expanded=False):
        st.markdown(f"**What it does.** {what}\n\n**When to use it.** {when}\n\n**How to read the result.** {read}")


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


def design_view(out, key=""):
    """results of an adjoint design run (design.json, design.png, best.dat)"""
    out = Path(out)
    f = out / "design.json"
    if f.exists():
        d = json.loads(f.read_text())
        s0, b = d["start_point"], d["best"]
        k = st.columns(4)
        k[0].metric("Drag, start → optimised", f"{b['Cd']:.5f}", f"{100 * (b['Cd'] / s0['Cd'] - 1):+.1f} %",
                    delta_color="inverse")
        k[1].metric("Lift", f"{b['Cl']:.3f}", f"target {d['cl']:g}", delta_color="off")
        v = d.get("verification_level1", {})
        if v.get("start", {}).get("Cd") and v.get("best", {}).get("Cd"):
            k[2].metric("Checked on the fine mesh", f"{v['best']['Cd']:.5f}",
                        f"{100 * (v['best']['Cd'] / v['start']['Cd'] - 1):+.1f} %", delta_color="inverse",
                        help="Start and optimised airfoil re-solved on mesh level 1 at the target Cl")
        k[3].metric("Flow + adjoint solutions", d["evaluations"], f"{d.get('time_min', 0):.0f} min", delta_color="off")
        st.caption(f"Start {C.display_name(Path(str(d['start'])).stem)}, Re {C.fmt_re(d['re'])}"
                   + (f", t/c ≥ {d['t_min']:.3f}" if d.get("t_min") else "")
                   + (f", Cm ≥ {d['cm_min']:g}" if d.get("cm_min") is not None else "") + ".")
    if (out / "design.png").exists():
        st.image(str(out / "design.png"))
    if (out / "best.dat").exists():
        st.download_button("Download the optimised airfoil (.dat)", (out / "best.dat").read_bytes(),
                           file_name=f"{out.name}.dat", key=f"d_dl{key}{out.name}",
                           help="Selig format: upload it on the Airfoil polar page to run its full polar")
def gradient_view(g, key=""):
    """results of an adjoint gradient run (adjoint.json)"""
    st.markdown(f"**{C.display_name(Path(str(g['airfoil'])).stem)}, Re {C.fmt_re(g['re'])}, α {g['alpha']:g}°:** "
                f"**Cl {g['Cl']:.4f}, Cd {g['Cd']:.5f}, Cm {g['Cm']:+.4f}** — {g['cells']} cells, "
                f"{g['unknowns']} adjoint unknowns")
    rows = [{"objective": k, "d/dalpha (per deg)": g["d_dalpha"][k], "d/dRe": g["d_dRe"][k]}
            for k in ("Cl", "Cd", "Cm")]
    st.dataframe(rows, hide_index=True)
    if "d_dcst" in g:
        goal = st.radio("I want", list(GOALS), horizontal=True, key="adj_goal" + key)
        pr = shape_push(g["airfoil"], g["d_dcst"], goal, g.get("cst_order", 8))
        ch = pr["change"]
        c1, c2 = st.columns([3, 1])
        ui.plot(push_figure(g["airfoil"], pr), key="adj_push" + key, container=c1)
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
                       + "." + (" The airfoil optimiser (Airfoil → Airfoil optimiser) does the trade-off for "
                                "you: it lowers the drag while holding the lift you need." if key_ in ("Cd", "Cl") else ""))
        st.caption("Arrows: which way to move each part of the surface, and how strongly (longer = more "
                   "effect). Orange pushes the surface out of the airfoil, blue pulls it in. The dotted line "
                   "is the new shape, exaggerated so it is visible. The movement is the steepest improvement "
                   "the 18 shape parameters can make; it changes the airfoil smoothly, not point by point.")
        with st.expander("Raw shape gradient (the 18 CST coefficients)"):
            obj = st.radio("Gradient of", ["Cd", "Cl", "Cm"], horizontal=True, key="adj_raw" + key)
            v = np.array(g["d_dcst"][obj])
            k = len(v) // 2
            fig = go.Figure([go.Bar(x=list(range(k)), y=v[:k], name="upper surface"),
                             go.Bar(x=list(range(k)), y=v[k:], name="lower surface")])
            fig.update_layout(xaxis_title="CST coefficient (0 = leading edge)",
                              yaxis_title=f"d{obj}/dA", barmode="group", height=320)
            ui.plot(fig, key="adj_cst" + key)
