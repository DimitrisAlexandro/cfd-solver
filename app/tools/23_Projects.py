import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
import ui
import views

import project as PR
import trust as T

C.page_setup()
ui.header("Projects", "A design as a dependency graph. Change any input and see what is out of date and why; "
                      "recompute only that; keep variants of the design and compare them side by side. "
                      "Results with identical inputs are shared between variants.")

STATE = {"up to date": "green", "stale": "orange", "not run": "gray", "running": "blue", "failed": "red"}
CHOICES = {"category": ["normal", "utility", "aerobatic"], "transition": ["free", "off", "fixed"]}
try:
    import drag_buildup as _D
    import wing_structure as _W
    CHOICES["gear"] = list(_D.GEAR_PRESETS)
    CHOICES["material"] = list(_W.MATERIALS)
except Exception:                                      # noqa: BLE001
    pass
OVERRIDE_HINTS = ["mass", "cg.0", "surfaces.wing.span", "surfaces.wing.root_chord", "surfaces.wing.tip_chord",
                  "surfaces.wing.sweep", "surfaces.wing.twist", "surfaces.wing.airfoil",
                  "surfaces.horizontal tail.span", "fuselage.length", "cd_extra"]


def state_md(s):
    return f":{STATE.get(s, 'gray')}-badge[{s}]"


def aircraft_files():
    C.AIRCRAFT.mkdir(parents=True, exist_ok=True)
    return ["demo"] + [str(p) for p in sorted(C.AIRCRAFT.glob("*.json"))]


# ----------------------------------------------------------------------
# project picker
# ----------------------------------------------------------------------
projects = PR.Project.list()
top1, top2 = st.columns([4, 1], vertical_alignment="bottom")
with top2.popover("New project", width="stretch"):
    nm = st.text_input("Name", placeholder="Trainer study")
    tpl = st.radio("Start from", ["aircraft", "section", "empty"], horizontal=False,
                   format_func={"aircraft": "Aircraft analysis chain (stability, drag, weight and balance, loads, "
                                            "structure, aeroelasticity, performance)",
                                "section": "Section polar study (CFD)", "empty": "Empty"}.get)
    src = st.selectbox("Aircraft", aircraft_files(), disabled=tpl != "aircraft",
                       format_func=lambda s: "Demo trainer (750 kg)" if s == "demo" else Path(s).stem)
    desc = st.text_input("Description", "")
    if st.button("Create", type="primary", disabled=not nm.strip()):
        p = PR.Project.create(nm.strip(), tpl, src, description=desc)
        st.session_state["proj"] = str(p.dir)
        st.rerun()

if not projects:
    st.info("No projects yet. **New project** creates one: the aircraft chain runs every aircraft tool in "
            "about a minute and keeps track of what each result depends on.")
    st.stop()
paths = [r["path"] for r in projects]
cur = st.session_state.get("proj")
idx = paths.index(cur) if cur in paths else 0
sel = top1.selectbox("Project", paths, index=idx,
                     format_func=lambda s: next(f"{r['name']}  ·  {r['variants']} variant"
                                                f"{'s' if r['variants'] != 1 else ''}" for r in projects
                                                if r["path"] == s))
st.session_state["proj"] = sel
p = PR.Project.load(sel)
if p.data.get("description"):
    st.caption(p.data["description"])

vnames = list(p.variants)
vkey = f"var{sel}"
if st.session_state.get("pending_var") in vnames:         # set by "create" / "delete" on the last run
    st.session_state[vkey] = st.session_state.pop("pending_var")
if st.session_state.get(vkey) not in vnames:
    st.session_state[vkey] = p.baseline
variant = st.segmented_control("Variant", vnames, key=vkey,
                               format_func=lambda v: f"{v} (baseline)" if v == p.baseline and v != "baseline" else v)
variant = variant or p.baseline
status = p.status(variant)
plan = p.plan(variant)
jk = f"pjob{sel}{variant}"

tab_g, tab_e, tab_v, tab_c = st.tabs(["Graph and status", "Edit", "Variants", "Compare variants"])

# ----------------------------------------------------------------------
with tab_g:
    left, right = st.columns([5, 6], gap="large")
    with left:
        if status:
            st.graphviz_chart(p.dot(variant, status), width="stretch")
        else:
            st.info("This variant has no analyses yet: add them on the Edit tab.")
        counts = {s: sum(v["state"] == s for v in status.values()) for s in STATE}
        st.markdown(" ".join(f":{STATE[s]}-badge[{n} {s}]" for s, n in counts.items() if n))
    with right:
        c1, c2 = st.columns(2)
        running = jk in st.session_state and C.get_job(st.session_state[jk])["status"] == "running"
        if c1.button(f"Run what's out of date ({len(plan)})", type="primary", disabled=not plan or running,
                     width="stretch", help="Runs the stale and never-run analyses in dependency order"):
            jdir = p.dir / "_jobs" / time.strftime("%Y%m%d-%H%M%S")
            st.session_state[jk] = C.start_job("project", f"Project {p.name} / {variant}: {len(plan)} analyses",
                                               ["project.py", "run", p.dir, "--variant", variant, "--job-dir", jdir],
                                               jdir)
            st.rerun()
        if c2.button("Recompute everything", disabled=not status or running, width="stretch"):
            jdir = p.dir / "_jobs" / time.strftime("%Y%m%d-%H%M%S")
            st.session_state[jk] = C.start_job("project", f"Project {p.name} / {variant}: all analyses",
                                               ["project.py", "run", p.dir, "--variant", variant, "--force",
                                                "--job-dir", jdir], jdir)
            st.rerun()
        if jk in st.session_state:
            views.live_job(st.session_state[jk], "project")
        mets = p.metrics(variant, status)
        rows = []
        for nid, s in status.items():
            m = mets.get(nid)
            rows.append({"analysis": s["label"], "state": s["state"],
                         "why": "; ".join(s["reasons"]) + (" (solver code updated since this run)"
                                                            if s["code_changed"] else ""),
                         "confidence": ui.TRUST[m["trust"]][1] if m and m["trust"] else "",
                         "time (s)": s.get("elapsed")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"why": st.column_config.TextColumn("why", width="medium"),
                                    "time (s)": st.column_config.NumberColumn(format="%.1f")})
        if any(s["state"] == "stale" for s in status.values()):
            st.caption("Stale results stay visible (marked *) until they are recomputed.")

    st.markdown("#### Key results")
    any_shown = False
    for nid, s in status.items():
        m = mets.get(nid)
        if not m:
            continue
        defs = PR.METRICS.get(s["kind"], [])
        vals = [(lab, unit, m["values"].get(k)) for k, lab, unit, _, _ in defs if m["values"].get(k) is not None]
        if not vals:
            continue
        any_shown = True
        st.markdown(f"**{s['label']}** " + (ui.trust_md(m["trust"]) if m["trust"] else "")
                    + ("" if m["current"] else " :orange-badge[stale *]"))
        cols = st.columns(min(len(vals), 7))
        for c, (lab, unit, v) in zip(cols, vals):
            c.metric(lab + ("" if m["current"] else " *"), f"{v:.4g}" + (f" {unit}" if unit else ""))
        if s["kind"] == "polar" and st.toggle("Show the polar", key=f"showpol{nid}"):
            f, _ = p.output(variant, nid, status=status)
            with st.container(border=True):
                views.polar_results(f.parent)
    if not any_shown:
        st.caption("Results appear here after the first run.")

# ----------------------------------------------------------------------
with tab_e:
    nodes = p.nodes(variant)
    if nodes:
        nid = st.selectbox("Analysis", list(nodes), format_func=lambda k: f"{nodes[k]['label']} ({k})")
        n = nodes[nid]
        kind = PR.KINDS[n["kind"]]
        down = p.downstream(variant, nid)
        st.caption(("Changing it makes these out of date: " + ", ".join(nodes[d]["label"] for d in down) + ".")
                   if down else "Nothing depends on this analysis.")
        if n["inputs"]:
            st.caption("Inputs: " + ", ".join(f"{r} ← {nodes[s]['label']}" for r, s in n["inputs"].items()))
        with st.form(f"edit{sel}{variant}{nid}"):
            new = {}
            for k, v in n["params"].items():
                if k == "overrides":
                    continue
                hlp = kind.get("help", {}).get(k)
                if n["kind"] == "aircraft" and k == "source":
                    opts = aircraft_files()
                    if v not in opts:
                        opts.append(v)
                    new[k] = st.selectbox("Aircraft source", opts, index=opts.index(v), help=hlp,
                                          format_func=lambda s: "Demo trainer (750 kg)" if s == "demo" else Path(s).stem)
                elif k in CHOICES:
                    opts = CHOICES[k] + ([v] if v not in CHOICES[k] else [])
                    new[k] = st.selectbox(k, opts, index=opts.index(v), help=hlp)
                elif isinstance(v, bool):
                    new[k] = st.checkbox(k, v, help=hlp)
                elif isinstance(v, (int, float)):
                    new[k] = st.number_input(k, value=float(v), format="%.6g", help=hlp)
                    if isinstance(v, int):
                        new[k] = int(new[k])
                else:
                    new[k] = st.text_input(k, str(v), help=hlp)
            if n["kind"] == "aircraft":
                st.markdown("**Changes to the aircraft model**")
                ov = n["params"].get("overrides") or {}
                df = pd.DataFrame([{"field": k, "value": json.dumps(v)} for k, v in ov.items()] or
                                  [{"field": "", "value": ""}])
                ed = st.data_editor(df, num_rows="dynamic", hide_index=True, width="stretch",
                                    column_config={"field": st.column_config.TextColumn(
                                        help="A path in the aircraft model, e.g. " + ", ".join(OVERRIDE_HINTS[:4]))})
                st.caption("Fields: " + ", ".join(f"`{h}`" for h in OVERRIDE_HINTS) +
                           ". Surfaces by name or index; values as numbers or JSON.")
            if st.form_submit_button("Save changes", type="primary"):
                if n["kind"] == "aircraft":
                    ovs = {}
                    for _, r in ed.iterrows():
                        f_, v_ = str(r["field"] or "").strip(), str(r["value"] or "").strip()
                        if f_:
                            ovs[f_] = PR._parse_value(v_)
                    new["overrides"] = ovs
                    try:                                   # catch bad paths before they reach a run
                        import aircraft as A
                        base = A.demo_aircraft().to_dict() if new["source"] == "demo" else \
                            json.loads(Path(new["source"]).read_text())
                        base = json.loads(json.dumps(base, default=list))
                        for k_, v_ in ovs.items():
                            PR.set_path(base, k_, v_)
                    except Exception as ex:                # noqa: BLE001
                        st.error(f"Can't apply the changes: {ex}")
                        st.stop()
                p.set_params(variant, nid, **new)
                st.rerun()
        with st.expander("Remove this analysis"):
            if st.button("Remove", key=f"rm{nid}", disabled=bool(down)):
                p.remove_node(variant, nid)
                st.rerun()
            if down:
                st.caption("Remove the analyses that use it first.")
    st.markdown("#### Add an analysis")
    c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    k_new = c1.selectbox("Kind", list(PR.KINDS), format_func=lambda k: PR.KINDS[k]["title"])
    base_id = k_new
    i = 2
    while base_id in p.nodes(variant):
        base_id = f"{k_new}{i}"
        i += 1
    id_new = c2.text_input("Name (id)", base_id)
    ins = {}
    need = PR.KINDS[k_new]["inputs"]
    if need:
        cols = st.columns(len(need))
        for c, role in zip(cols, need):
            cand = [k for k, nn in p.nodes(variant).items() if nn["kind"] == role]
            ins[role] = c.selectbox(f"Input: {role}", cand, key=f"in{role}") if cand else None
    ok_in = all(ins.values())
    if c3.button("Add", disabled=not ok_in or not id_new.strip(), width="stretch"):
        try:
            p.add_node(variant, id_new.strip(), k_new, inputs=ins)
            st.rerun()
        except ValueError as ex:
            st.error(str(ex))
    if need and not ok_in:
        st.caption(f"Needs a {', '.join(r for r in need if not ins.get(r))} analysis in this variant first.")

# ----------------------------------------------------------------------
with tab_v:
    vrows = []
    for v, vd in p.variants.items():
        stv = p.status(v)
        diff = p.diff(p.baseline, v) if v != p.baseline else []
        vrows.append({"variant": v + (" (baseline)" if v == p.baseline else ""), "from": vd.get("parent") or "",
                      "description": vd.get("description", ""),
                      "changes vs baseline": "; ".join(f"{n_}.{k_}: {PR._fmt(a_)} → {PR._fmt(b_)}"
                                                       for n_, k_, a_, b_ in diff) or "—",
                      "up to date": f"{sum(s['state'] == 'up to date' for s in stv.values())} of {len(stv)}"})
    st.dataframe(pd.DataFrame(vrows), hide_index=True, width="stretch",
                 column_config={"changes vs baseline": st.column_config.TextColumn(width="large")})
    st.markdown("#### New variant")
    with st.form(f"newvar{sel}"):
        c1, c2 = st.columns(2)
        vn = c1.text_input("Name", placeholder="heavier, CFRP wing, ...")
        vb = c2.selectbox("Copy of", vnames, index=vnames.index(variant))
        vdsc = st.text_input("Description", "")
        st.markdown("Changes: `analysis.parameter` (e.g. `loads.vc`, `structure.material`) or "
                    "`aircraft.<field>` for the aircraft model (e.g. `aircraft.mass`, `aircraft.surfaces.wing.span`).")
        ch = st.data_editor(pd.DataFrame([{"change": "aircraft.mass", "value": ""}]), num_rows="dynamic",
                            hide_index=True, width="stretch")
        if st.form_submit_button("Create variant", type="primary"):
            changes = {}
            for _, r in ch.iterrows():
                k_, v_ = str(r["change"] or "").strip(), str(r["value"] or "").strip()
                if k_ and v_:
                    changes[k_] = PR._parse_value(v_)
            if not vn.strip() or vn.strip() in p.variants:
                st.error("Give the variant a new name.")
            else:
                bad = [k_ for k_ in changes if k_.split(".")[0] not in p.nodes(vb)]
                if bad:
                    st.error(f"No analysis named {', '.join(b.split('.')[0] for b in bad)} in {vb}.")
                else:
                    p.create_variant(vn.strip(), vb, vdsc, changes)
                    st.session_state["pending_var"] = vn.strip()
                    st.rerun()
    c1, c2 = st.columns(2)
    if c1.button(f"Make '{variant}' the baseline", disabled=variant == p.baseline):
        p.set_baseline(variant)
        st.rerun()
    if c2.button(f"Delete variant '{variant}'", disabled=variant == p.baseline):
        p.delete_variant(variant)
        st.session_state["pending_var"] = p.baseline
        st.rerun()

# ----------------------------------------------------------------------
with tab_c:
    if len(vnames) < 2:
        st.info("Create a second variant (Variants tab) to compare. Each one reuses every result whose inputs "
                "didn't change.")
    else:
        order = [p.baseline] + [v for v in vnames if v != p.baseline]
        vs = st.multiselect("Variants", order, default=order)
        if len(vs) >= 1:
            rows = p.compare(vs)
            if not rows:
                st.caption("No results yet: run the variants first.")
            else:
                tab = []
                for r in rows:
                    rec = {"analysis": r["node"], "result": r["metric"] + (f" ({r['unit']})" if r["unit"] else "")}
                    for v in vs:
                        x = r["values"][v]
                        chg = r["change"].get(v)
                        star = "" if r["current"].get(v) in (True, None) else " *"
                        rec[v] = "—" if x is None else (f"{x:.4g}" + (f"  ({100 * chg:+.1f} %)"
                                                                      if chg and v != vs[0] else "") + star)
                    tab.append(rec)
                df = pd.DataFrame(tab)

                def colour(row_i):
                    r = rows[row_i]
                    out = [""] * 2
                    for v in vs:
                        c = r["change"].get(v)
                        good = None if c is None or abs(c) < 0.005 or not r["better"] or v == vs[0] else \
                            (c > 0) == (r["better"] == "higher")
                        out.append("" if good is None else
                                   ("color: #2B7A4B" if good else "color: #9C3D54"))
                    return out
                sty = df.style.apply(lambda s: colour(s.name), axis=1)
                st.dataframe(sty, hide_index=True, width="stretch", height=min(38 + 35 * len(df), 900))
                st.caption(f"Changes against **{vs[0]}**; green is better, red is worse where there is a better "
                           "direction. * = from a stale run (inputs changed since).")
                labels = [f"{r['node']}: {r['metric']}" for r in rows]
                pick = st.selectbox("Chart", labels)
                r = rows[labels.index(pick)]
                f = go.Figure()
                f.add_bar(x=vs, y=[r["values"][v] for v in vs], marker_color=ui.BLUE,
                          text=[("" if r["values"][v] is None else f"{r['values'][v]:.4g}") for v in vs],
                          textposition="outside")
                f.update_layout(title=pick, yaxis_title=r["unit"], height=320, showlegend=False)
                ui.plot(f)
                if len(vs) >= 2:
                    st.markdown("#### What differs")
                    dd = []
                    for v in vs[1:]:
                        dd += [{"variant": v, "analysis": n_, "parameter": k_, vs[0]: PR._fmt(a_), "this variant": PR._fmt(b_)}
                               for n_, k_, a_, b_ in p.diff(vs[0], v)]
                    st.dataframe(pd.DataFrame(dd) if dd else pd.DataFrame([{"": "no parameter differences"}]),
                                 hide_index=True, width="stretch")
