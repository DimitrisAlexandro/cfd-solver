import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

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
        "| **Swept-wing transition** | *Where does the laminar (low-drag) flow end on a swept wing?* | instant |\n"
        "| **Polar surrogate** | *Roughly what polar will this airfoil have, before I spend time on CFD?* | "
        "instant |\n"
        "| **Learned SA correction** | *How much better does a machine-learned turbulence correction match "
        "wind-tunnel data?* | results only (computed from the command line) |")
    st.caption("For a normal airfoil polar use Airfoil → Airfoil polar; for wings and aircraft the "
               "Wing and aircraft pages. These tools are for going further.")

tab_g, tab_d, tab_s, tab_p, tab_f = st.tabs(["Adjoint gradients", "Adjoint design",
                                             "Swept-wing transition", "Polar surrogate",
                                             "Learned SA correction"])

# ----------------------------------------------------------------------
with tab_g:
    explain("Runs the CFD once for your airfoil, then one extra *adjoint* solve. Together they give the "
            "gradient (sensitivity) of Cl, Cd and Cm to the angle of attack, the Reynolds number and 18 "
            "shape parameters (CST coefficients, 9 on the upper and 9 on the lower surface, from leading to "
            "trailing edge). Finite differences would need one CFD run per parameter, 20 runs instead of "
            "about 2.",
            "To see which part of the airfoil to change to cut drag or add lift, and how sensitive the airfoil "
            "is to the angle and Re, before redesigning it by hand.",
            "The table gives the change per degree of angle and per unit of Re. In the bar chart, a tall bar "
            "is a part of the surface where a small change matters a lot. Making a coefficient bigger moves the "
            "surface **up** there (thicker on the upper surface, thinner on the lower). For Cd, a **negative** "
            "bar means moving the surface up there **lowers** the drag; a positive bar, moving it down. "
            "Coefficient 0 is at the leading edge, 8 at the trailing edge.")
    st.caption("dCl, dCd, dCm with respect to the angle of attack, the Reynolds number and all 18 CST "
               "shape coefficients, for one flow solution + one adjoint solve. Fully turbulent or fixed "
               "transition (a free-transition location is frozen in the adjoint).")
    with st.form("adj_grad"):
        c1, c2, c3, c4 = st.columns(4)
        spec = c1.text_input("Airfoil", "naca0012")
        re = c2.number_input("Reynolds number", 1e5, 5e7, 3e6, format="%.3g")
        alpha = c3.number_input("Angle of attack (deg)", -10.0, 15.0, 2.0)
        level = c4.selectbox("Mesh level", [0.5, 0.7, 1.0], index=0,
                             help="The adjoint's sparse LU grows faster than linearly: ~1 min at 0.5, "
                                  "~2 min at 0.7, several minutes at 1.0")
        go_ = st.form_submit_button("Compute gradients", type="primary")
    if go_:
        out = C.CASES / f"adjoint_{spec}_Re{re:.0e}_a{alpha:g}".replace("+", "")
        out.mkdir(parents=True, exist_ok=True)
        jid = C.start_job("adjoint", f"adjoint gradients {spec} alpha {alpha:g}",
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
                obj = st.radio("Shape gradient of", ["Cd", "Cl", "Cm"], horizontal=True)
                v = np.array(g["d_dcst"][obj])
                k = len(v) // 2
                fig = go.Figure([go.Bar(x=list(range(k)), y=v[:k], name="upper surface"),
                                 go.Bar(x=list(range(k)), y=v[k:], name="lower surface")])
                fig.update_layout(xaxis_title="CST coefficient (0 = leading edge)",
                                  yaxis_title=f"d{obj}/dA", barmode="group", height=360)
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
    with st.form("adj_des"):
        c1, c2, c3, c4 = st.columns(4)
        spec = c1.text_input("Start airfoil", "naca0012", key="d_spec")
        re = c2.number_input("Reynolds number", 1e5, 5e7, 3e6, format="%.3g", key="d_re")
        cl = c3.number_input("Target Cl", 0.0, 1.5, 0.5, key="d_cl")
        tmin = c4.number_input("Minimum t/c (0 = keep the start)", 0.0, 0.3, 0.0, key="d_t")
        c1, c2 = st.columns(2)
        iters = c1.slider("Outer iterations", 2, 30, 12)
        cm = c2.number_input("Minimum Cm (blank = none)", -1.0, 1.0, value=None, key="d_cm")
        go_d = st.form_submit_button("Optimise", type="primary")
    if go_d:
        out = C.CASES / f"design_adjoint_{spec}_cl{cl:g}"
        args = ["design_adjoint.py", "--start", spec, "--re", f"{re:.6g}", "--cl", f"{cl:g}",
                "--iters", iters, "--out", out]
        if tmin > 0:
            args += ["--t-min", f"{tmin:g}"]
        if cm is not None:
            args += ["--cm-min", f"{cm:g}"]
        st.session_state["des_job"] = C.start_job("adjoint_design", f"adjoint design {spec} Cl {cl:g}", args, out)
    jid = st.session_state.get("des_job")
    if jid:
        views.live_job(jid, "adjoint_design")
        j = C.get_job(jid)
        if (Path(j["out"]) / "design.png").exists():
            st.image(str(Path(j["out"]) / "design.png"))

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
            "ordinary transition prediction of the Airfoil polar page.",
            "**Transition, upper / lower** is where the flow turns turbulent, as a fraction of the chord "
            "(0 = leading edge, 1 = trailing edge), and the mechanism that did it. Further aft means more "
            "laminar flow and less drag. **Attachment line Rbar** is the Reynolds number of the leading-edge "
            "flow: below ~245 it stays laminar; above ~245 turbulence from the wing root or fuselage spreads "
            "along it; above ~580 it turns turbulent by itself. In the plot, transition happens where a curve "
            "reaches the dotted line.")
    st.caption("Infinite swept wing: attachment-line contamination / instability (Poll), crossflow "
               "(Falkner-Skan-Cooke profiles) and Tollmien-Schlichting (e^N). For a whole wing: "
               "`wing.py ... --transition-3d ALPHA`.")
    c1, c2, c3, c4 = st.columns(4)
    spec = c1.text_input("Airfoil", "naca0012", key="s_spec")
    sweep = c2.slider("Leading-edge sweep (deg)", 0, 60, 30)
    re = c3.number_input("Reynolds number (normal chord)", 1e5, 1e8, 1e7, format="%.3g", key="s_re")
    alpha = c4.number_input("Angle of attack (deg)", -5.0, 10.0, 2.0, key="s_a")
    c1, c2, c3 = st.columns(3)
    tu = c1.number_input("Turbulence Tu (%), 0 = N_crit 9", 0.0, 2.0, 0.0, 0.01)
    chi = c2.number_input("Crossflow transition chi", 50.0, 500.0, 200.0)
    cont = c3.checkbox("Root contaminates the attachment line", False)
    from aerolab.physics import swept_transition as stt
    with st.spinner("boundary layers ..."):
        r = stt.analyse(spec, float(sweep), re, alpha, tu or None, None, chi, cont)
    al = r["attachment_line"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Attachment line Rbar", f"{al['Rbar']:.0f}", al["why"],
              help="Below ~245 laminar; 245–580 turbulent if the root or fuselage contaminates it; above ~580 "
                   "turbulent by itself")
    m2.metric("Transition, upper", f"{r['upper']['x_tr']:.3f} c", r["upper"]["mechanism"])
    m3.metric("Transition, lower", f"{r['lower']['x_tr']:.3f} c", r["lower"]["mechanism"])
    fig = go.Figure()
    for side, dash in (("upper", "solid"), ("lower", "dash")):
        d = r[side]
        fig.add_trace(go.Scatter(x=d["x"], y=d["N"], name=f"N (TS), {side}", line=dict(dash=dash)))
        fig.add_trace(go.Scatter(x=d["x"], y=np.array(d["chi"]) * 9.0 / chi, name=f"chi x 9/chi_tr, {side}",
                                 line=dict(dash=dash)))
    fig.add_hline(y=r["ncrit"], line_dash="dot")
    fig.update_layout(xaxis_title="x/c", yaxis_title="N  (crossflow scaled: transition at the line)",
                      height=380, yaxis_range=[0, 3 * r["ncrit"]])
    ui.plot(fig, key="swept")

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
    st.caption("Gaussian process over every polar in the library; mean and +-2 sd (calibrated by "
               "leaving polars out). Retrain after adding polars: `python src/run.py surrogate train`.")
    c1, c2, c3 = st.columns(3)
    spec = c1.text_input("Airfoil", "naca2412", key="p_spec")
    re = c2.number_input("Reynolds number", 5e4, 3e7, 3e6, format="%.3g", key="p_re")
    turb = c3.checkbox("Fully turbulent", False)
    try:
        sg = SG.Surrogate.load() if SG.MODEL.exists() else SG.Surrogate.train()
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
