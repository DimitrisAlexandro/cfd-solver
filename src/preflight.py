"""Pre-run checks: catch a bad setup in seconds instead of after a 40-minute run.

For an airfoil polar: geometry, Reynolds and Mach sanity, angle range, the first-cell
y+ at the suction peak, mesh quality (optional: builds the mesh, ~3 s), the expected
confidence of every angle (trust layer) and the expected runtime (calibrated on the
runs already on this machine).

For a project variant: aircraft source and changes, section data at the flight
Reynolds number of every surface, speeds against the stall speed, propulsion, structure
inputs, and the runtime and expected confidence of every analysis that will run.

    python preflight.py polar naca2412 --re 3e6 --alpha=-4:16:2 [--deep]
    python preflight.py project ../cases/projects/trainer-study [--variant v]

Levels: ok / info / warn / error. An error blocks the run (the app offers "run anyway").
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
CASES = Path(os.environ.get("AEROLAB_CASES", ROOT / "cases"))
POLAR_DIR = CASES / "wing_polars"
RANK = {"ok": 0, "info": 1, "warn": 2, "error": 3}

# seconds per angle on the reference machine (2-core cloud VM, one process), 29k-cell mesh
BASE_S_PER_ANGLE = 110.0
# y+ max ≈ 0.523 (U_e,max/U)^0.81 / level: fitted to 22 converged NACA 0012 / 2412 points (Re 3e6,
# 6e6, α -4…16°; within 5 %), U_e,max from the panel method's minimum Cp
YPLUS_FIT = (0.523, 0.81)
# per-node defaults (s) when this machine has no history yet
NODE_DEFAULT_S = {"aircraft": 2, "stability": 3, "drag": 3, "weight_balance": 3, "loads": 12, "structure": 2,
                  "aeroelastic": 4, "performance": 50}


def item(level, check, detail):
    return {"level": level, "check": check, "detail": detail}


def worst(items):
    return max((i["level"] for i in items), key=lambda l: RANK[l], default="ok")


def parse_alpha(s):
    s = str(s).replace(" ", "")
    if not s:
        return []
    if ":" in s:
        a, b, d = (float(x) for x in s.split(":"))
        if d <= 0:
            return []
        n = int(math.floor((b - a) / d + 1e-9)) + 1
        return [round(a + i * d, 6) for i in range(max(n, 0))]
    return [float(x) for x in s.split(",") if x]


def fmt_time(s):
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


# ----------------------------------------------------------------------
# runtime model
# ----------------------------------------------------------------------
def _cells(level):
    return 29000.0 * level ** 1.9


def angle_cost(alpha, mach=0.0, transition="free", re=3e6, multi=False):
    """relative cost of one angle against a 29k-cell attached incompressible angle"""
    f = 1.0
    if transition == "free":
        f *= 1.35
    if mach >= 0.5:
        f *= 3.0
    elif mach >= 0.3:
        f *= 1.6
    a = abs(alpha)
    if a > 10:
        f *= 1.0 + 0.15 * (a - 10)                    # separated flow: slower convergence
    if re < 5e5 and transition == "free":
        f *= 1.5                                       # low-Re mode: cold starts, bubbles
    if multi:
        f *= 2.6                                       # three elements, ~75k cells at level 1
    return f


def machine_factor():
    """this machine's speed against the reference, from finished polars in cases/
    (median of actual / modelled time per angle); 1.0 without history"""
    r = []
    for f in list(CASES.glob("*/polar.json"))[:60]:
        try:
            d = json.loads(f.read_text())
        except Exception:                               # noqa: BLE001
            continue
        s = d.get("settings") or {}
        mode = s.get("transition_mode") or ("off" if "fully" in str(s.get("transition", "")) else "free")
        for p in d.get("results", []):
            t, n = p.get("time_s"), p.get("cells")
            if not t or not n or not str(p.get("status", "")).startswith("converged"):
                continue
            model = BASE_S_PER_ANGLE * (n / 29000.0) * angle_cost(p["alpha"], s.get("mach") or 0.0, mode,
                                                                  d.get("re") or 3e6, bool(s.get("multi")))
            r.append(t / model)
    return (float(np.median(r)), len(r)) if len(r) >= 3 else (1.0, len(r))


def polar_runtime(alphas, level=1.0, jobs=2, mach=0.0, transition="free", re=3e6, multi=False):
    mf, n_hist = machine_factor()
    cpu = os.cpu_count() or 2
    par = max(1, min(int(jobs), len(alphas) or 1, cpu))
    per = [BASE_S_PER_ANGLE * mf * (_cells(level) / 29000.0) * angle_cost(a, mach, transition, re, multi)
           for a in alphas]
    blocks = np.array_split(np.array(per), par) if per else []
    wall = max((b.sum() for b in blocks), default=0.0) + 10.0     # meshing, start-up
    if par > cpu / 2:                                              # shared cores
        wall *= 1.0 + 0.3 * (par - cpu / 2) / max(cpu / 2, 1)
    return {"expected_s": wall, "low_s": 0.6 * wall, "high_s": 2.0 * wall, "machine_factor": mf,
            "history_points": n_hist, "parallel": par}


# ----------------------------------------------------------------------
# airfoil polar
# ----------------------------------------------------------------------
def check_polar(spec, re, alpha, mach=0.0, transition="free", ncrit=9.0, xtr=None, refine=1.0, jobs=2,
                prandtl_glauert=False, regime="auto", deep=False):
    """returns {"level", "items", "runtime", "confidence", "mesh"}"""
    import airfoil_geom as ag
    import multi_element as me
    import trust as T
    items, mesh = [], None
    alphas = parse_alpha(alpha) if isinstance(alpha, str) else [float(a) for a in alpha]
    multi = me.is_multi(str(spec))
    tmode = "fixed" if (transition or "").startswith("fixed") or xtr else \
        "off" if transition in ("off", "fully turbulent") else "free"

    # geometry
    t_c, outline = None, None
    try:
        if multi:
            elements, info = me.load_multi(spec)
            items.append(item("ok", "Geometry", f"{info['name']}: {len(elements)} elements loaded"))
        else:
            loop, info = ag.load_airfoil(spec)
            t_c = info.get("max_thickness")
            outline = ag.sharp_outline(spec)
            te = info.get("te_thickness", 0.0)
            items.append(item("ok", "Geometry", f"{info['name']}: t/c {100 * t_c:.1f} % at x/c "
                                                f"{info.get('x_max_thickness', 0):.2f}, trailing edge "
                                                f"{100 * te:.2f} % c, {info.get('points', len(loop))} points"))
            if not (T.T_C_RANGE[0] <= t_c <= T.T_C_RANGE[1]):
                items.append(item("warn", "Thickness", f"t/c {100 * t_c:.1f} % is outside the validated sections "
                                                       f"({100 * T.T_C_RANGE[0]:.0f}–{100 * T.T_C_RANGE[1]:.0f} %): "
                                                       "expect larger errors."))
            if te > 0.02:
                items.append(item("warn", "Trailing edge", f"blunt trailing edge {100 * te:.1f} % c: base drag and "
                                                           "unsteady shedding; results may oscillate."))
        if "@flap=" in str(spec):
            b, (cf, d) = ag.parse_flap(spec)
            if abs(d) > 20:
                items.append(item("warn", "Flap", f"{d:g}° plain flap: above 20° the flap flow separates and "
                                                  "runs often diverge (validated 0–20°)."))
    except FileNotFoundError as e:
        items.append(item("error", "Geometry", f"file not found: {e}"))
    except Exception as e:                                         # noqa: BLE001
        items.append(item("error", "Geometry", f"the section can't be loaded: {e}"))

    # Reynolds number
    if re < 1e3:
        items.append(item("error", "Reynolds number", f"Re {re:.3g}: below 1000 is not supported."))
    elif multi and re < 1e5:
        items.append(item("error", "Reynolds number", "multi-element airfoils need Re ≥ 1e5 (turbulent solver)."))
    elif re < 2e4:
        items.append(item("info", "Reynolds number", f"Re {re:.3g}: laminar solver (no turbulence model)."))
    elif re < 8e4:
        items.append(item("warn", "Reynolds number", f"Re {re:.3g}: no validation data between 2e4 and 8e4; "
                                                     "laminar bubbles dominate. Compare with XFOIL."))
    elif re < 1.5e5:
        items.append(item("warn", "Reynolds number", f"Re {re:.3g}: mid-chord bubbles may burst in steady RANS "
                                                     "(E387 at 1e5: drag up to 2.5×)."))
    elif re > 5e7:
        items.append(item("warn", "Reynolds number", f"Re {re:.3g}: beyond the validated range (≤ 1.2e7); the "
                                                     "first cell gets very thin."))
    else:
        items.append(item("ok", "Reynolds number", f"Re {re:.3g}"))

    # Mach number
    if mach >= 0.95:
        items.append(item("error", "Mach number", "M ≥ 0.95: supersonic free streams are not set up."))
    elif mach > 0.8:
        items.append(item("warn", "Mach number", f"M {mach:.2f}: strong shocks and shock-induced separation; "
                                                 "validated up to the RAE 2822 case (M 0.73)."))
    elif mach > 0.3 and prandtl_glauert:
        items.append(item("warn" if mach <= 0.5 else "error", "Compressibility",
                          f"Prandtl–Glauert at M {mach:.2f}: valid to ~0.3; use the compressible solver."))
    elif mach >= 0.4 and tmode == "free":
        items.append(item("info", "Transition", f"M {mach:.2f}: free transition is switched to fully turbulent "
                                                "(e^N stability theory is incompressible)."))
    elif mach > 0:
        items.append(item("ok", "Mach number", f"M {mach:.2f}" + (", Prandtl–Glauert" if prandtl_glauert
                                                                  else ", compressible solver")))
    if multi and tmode == "free":
        items.append(item("info", "Transition", "multi-element: runs fully turbulent (e^N is single-element)."))

    # angles
    if not alphas:
        items.append(item("error", "Angles", "no angles: give a range start:stop:step or a list."))
    else:
        amax = max(abs(a) for a in alphas)
        if amax > 25:
            items.append(item("error", "Angles", f"|α| up to {amax:g}°: far past stall, steady RANS won't converge."))
        elif amax > 18:
            items.append(item("warn", "Angles", f"|α| up to {amax:g}°: past the lift peak the flow is unsteady; "
                                                "those points end as means of oscillations."))
        else:
            items.append(item("ok", "Angles", f"{len(alphas)} angles, {min(alphas):g}° to {max(alphas):g}°"))
        if len(alphas) > 30:
            items.append(item("warn", "Angles", f"{len(alphas)} angles: consider a coarser step first."))

    # y+ at the suction peak (first cell from the mesher's target y+ 0.7 on the turbulent mesh)
    turb = re >= 2e4 and regime != "laminar"
    if turb and outline is not None and alphas:
        try:
            import panel
            ue = []
            for a in (min(alphas), max(alphas)):
                r = panel.solve([outline], a)
                cp = np.asarray(r["elements"][0]["Cp"])
                x = np.asarray(r["elements"][0]["x"])
                cp = cp[(x > 0.002) & (x < 0.98)]
                ue.append(math.sqrt(max(1.0 - cp.min(), 0.0)))
            ue_max = max(ue)
            yp = YPLUS_FIT[0] * ue_max ** YPLUS_FIT[1] / refine
            lev = "ok" if yp <= 1.7 else "warn"
            items.append(item(lev, "Wall resolution (y+)",
                              f"expected y+ max ≈ {yp:.2f} at the suction peak (U_e/U ≈ {ue_max:.2f})"
                              + (": fine (the validated NASA TMR runs reach 1.6 at 16°)" if lev == "ok" else
                                 ": wall cells too thick at the leading edge; mesh level 1.4 lowers it by 1.4×")))
        except Exception as e:                                     # noqa: BLE001
            items.append(item("info", "Wall resolution (y+)", f"not estimated ({e})"))

    # mesh
    if refine < 0.95:
        items.append(item("warn", "Mesh level", f"level {refine:g}: about twice as fast, Cd +2.5 % on NACA 0012; "
                                                "use level 1 for numbers you quote."))
    if deep and not any(i["level"] == "error" for i in items) and alphas:
        try:
            mesh = mesh_quality(spec, re, float(np.median(alphas)), refine, regime)
            lev = "ok"
            msg = (f"{mesh['cells']:,} cells, first cell {mesh['h1']:.2e} c, non-orthogonality max "
                   f"{mesh['nonortho_max']:.0f}° (99 % below {mesh['nonortho_p99']:.0f}°), largest size jump between "
                   f"neighbouring triangles {mesh['size_ratio_max']:.1f}×")
            if mesh["nonortho_max"] > 80 or mesh["clamped"]:
                lev = "error" if mesh["nonortho_max"] > 85 else "warn"
                msg += f"; {mesh['clamped']} faces beyond 80° are clamped (convergence at risk)"
            elif mesh["nonortho_max"] > 70:
                lev = "warn"
                msg += " (validated meshes stay below 66°)"
            items.append(item(lev, "Mesh quality", msg))
        except Exception as e:                                     # noqa: BLE001
            items.append(item("error", "Mesh quality", f"meshing failed: {e}"))

    # expected confidence (trust layer)
    conf = []
    for a in alphas:
        r = T.assess_point(re, mach, a, transition="laminar" if not turb else tmode,
                           elements="multi" if multi else "single", prandtl_glauert=prandtl_glauert,
                           refine=refine, t_c=t_c)
        conf.append({"alpha": a, "level": r.level, "cl_band": r.cl_band, "cd_band": r.cd_band,
                     "reason": next((t for l, t in r.reasons if l == r.level), r.reasons[0][1])})
    counts = {l: sum(c["level"] == l for c in conf) for l in T.LEVELS}
    if conf:
        lv = "ok" if counts["red"] == 0 and counts["amber"] == 0 else "warn" if counts["red"] < len(conf) else "warn"
        txt = ", ".join(f"{counts[l]} {T.LABEL[l]}" for l in T.LEVELS if counts[l])
        items.append(item(lv, "Expected confidence", txt + (
            "" if lv == "ok" else ". Low-confidence angles: " +
            ", ".join(f"{c['alpha']:g}°" for c in conf if c["level"] == "red")[:200])))

    rt = polar_runtime(alphas, refine, jobs, mach, tmode, re, multi) if alphas else None
    if rt:
        items.append(item("info", "Runtime", f"about {fmt_time(rt['expected_s'])} "
                                             f"({fmt_time(rt['low_s'])}–{fmt_time(rt['high_s'])}) with "
                                             f"{rt['parallel']} process{'es' if rt['parallel'] > 1 else ''}"
                                             + (f"; calibrated on {rt['history_points']} earlier angles on this "
                                                "machine" if rt["history_points"] >= 3 else
                                                "; not yet calibrated on this machine")))
    return {"level": worst(items), "items": items, "runtime": rt, "confidence": conf, "mesh": mesh}


def mesh_quality(spec, re, alpha, level=1.0, regime="auto"):
    import polar
    fvm, mesh, *_ = polar.make_case(spec, re, alpha, level, regime=regime)
    d, S = fvm.d, fvm.Sf
    cos = np.abs(np.einsum("ij,ij->i", d, S)) / (np.linalg.norm(d, axis=1) * np.linalg.norm(S, axis=1))
    no = np.degrees(np.arccos(np.clip(cos, 0, 1)))
    a = fvm.vol
    nb = int(mesh.get("n_bl", 0))                       # cells [0, n_bl): anisotropic boundary layer
    tri = (fvm.own >= nb) & (fvm.nei >= nb)
    r = np.maximum(a[fvm.own], a[fvm.nei]) / np.maximum(np.minimum(a[fvm.own], a[fvm.nei]), 1e-300)
    ratio = float(r[tri].max()) if tri.any() else float("nan")
    P = mesh["points"]
    wall = next(v for k, v in mesh["patches"].items() if k in ("airfoil",) or k.startswith("wall"))
    h1 = mesh.get("h1") or float("nan")
    return {"cells": int(fvm.ncells), "nonortho_max": float(no.max()), "nonortho_p99": float(np.percentile(no, 99)),
            "size_ratio_max": ratio, "clamped": int(getattr(fvm, "n_clamped", 0)), "h1": h1,
            "wall_faces": int(len(wall)), "points": int(len(P))}


# ----------------------------------------------------------------------
# project variant
# ----------------------------------------------------------------------
def _section_coverage(ac, speed, alt):
    """per surface: flight Re on its MAC and the Re of the cached CFD polars"""
    import design_wing_sizing as WS
    import wing as W
    rho, mu, a_snd, _, _ = W.isa(alt)
    mach = speed / a_snd
    out = []
    for s in ac.surfaces:
        re = rho * speed * s.mac / mu
        for af in [s.airfoil] + ([s.tip_airfoil] if s.tip_airfoil and s.tip_airfoil != s.airfoil else []):
            got = WS.cached_polars(af, POLAR_DIR, mach) if POLAR_DIR.exists() else []
            res = sorted(r for r, _ in got)
            out.append({"surface": s.name, "airfoil": af, "re": re, "cached_re": res})
    return out


def _stall_speed(ac, alt, clmax=1.4):
    import wing as W
    rho = W.isa(alt)[0]
    return math.sqrt(2 * ac.mass * 9.80665 / (rho * ac.S * clmax))


def check_project(p, variant=None, plan_only=True):
    """returns {"level", "items": [...], "nodes": {nid: {"level", "items", "runtime_s"}}, "runtime_s"}"""
    import aircraft as A
    import project as PR
    import trust as T
    variant = variant or p.baseline
    status = p.status(variant)
    plan = p.plan(variant)
    nodes = p.nodes(variant)
    hist = _node_history()
    res_nodes, total = {}, 0.0
    ac_cache = {}

    def aircraft_of(nid):
        """the aircraft a node will see: built from the aircraft node's params (without running it)"""
        if nid in ac_cache:
            return ac_cache[nid]
        prm = nodes[nid]["params"]
        src = prm.get("source") or "demo"
        if src == "demo":
            d = A.demo_aircraft().to_dict()
        else:
            d = json.loads(Path(src if Path(src).is_absolute() else SRC / src).read_text())
        d = json.loads(json.dumps(d, default=list))
        for k, v in (prm.get("overrides") or {}).items():
            PR.set_path(d, k, v)
        ac_cache[nid] = A.Aircraft.from_dict(d)
        return ac_cache[nid]

    for nid in p.order(variant):
        n = nodes[nid]
        kind, prm = n["kind"], n["params"]
        its = []
        ac = None
        src_nid = n["inputs"].get("aircraft") or (nid if kind == "aircraft" else None)
        if kind == "structure" or kind == "aeroelastic":
            cur = nid
            while src_nid is None and nodes[cur]["inputs"]:
                cur = next(iter(nodes[cur]["inputs"].values()))
                src_nid = cur if nodes[cur]["kind"] == "aircraft" else nodes[cur]["inputs"].get("aircraft")
        if src_nid:
            try:
                ac = aircraft_of(src_nid)
            except FileNotFoundError as e:
                its.append(item("error", "Aircraft source", f"file not found: {e.filename}"))
            except KeyError as e:
                its.append(item("error", "Aircraft changes", f"a change points to a field that doesn't exist: {e}"))
            except Exception as e:                                 # noqa: BLE001
                its.append(item("error", "Aircraft source", f"can't be read: {e}"))
        if kind == "aircraft" and ac is not None:
            from weight_balance import pct_mac
            cg = pct_mac(ac, ac.cg[0])
            its.append(item("ok", "Aircraft", f"{ac.name}: {ac.mass:g} kg, S {ac.S:.2f} m², b {ac.b:.2f} m, "
                                              f"c.g. {cg:.0f} % MAC"))
            if not -10 <= cg <= 50:
                its.append(item("warn", "Centre of gravity", f"c.g. at {cg:.0f} % MAC: outside the usual range; "
                                                             "trim may fail."))
        if kind in ("stability", "loads", "performance", "drag") and ac is not None:
            speed = float(prm.get("speed") or prm.get("vc") or prm.get("v_design") or 50.0)
            alt = float(prm.get("alt") or 0.0)
            vs = _stall_speed(ac, alt)
            if kind in ("stability", "drag") and speed < 1.15 * vs:
                its.append(item("warn", "Speed", f"{speed:g} m/s is within 15 % of the estimated stall speed "
                                                 f"({vs:.0f} m/s at C_L 1.4): trim near C_L,max."))
            if kind == "loads" and speed < 1.3 * vs:
                its.append(item("warn", "V_C", f"V_C {speed:g} m/s is low against the stall speed ≈ {vs:.0f} m/s "
                                               "(CS-23 needs V_C ≥ 2.4 √(W/S) in knots/psf; check the category)."))
            if kind == "stability":
                for c in _section_coverage(ac, speed, alt):
                    if not c["cached_re"]:
                        its.append(item("warn", "Section data", f"{c['surface']} ({c['airfoil']}): no CFD section "
                                                                f"polars; thin-airfoil estimate used (stall and "
                                                                f"profile drag approximate). Compute polars at "
                                                                f"Re {c['re']:.2g}."))
                    elif c["re"] < 0.5 * c["cached_re"][0] or c["re"] > 2 * c["cached_re"][-1]:
                        its.append(item("warn", "Section data",
                                        f"{c['surface']} ({c['airfoil']}): flight Re {c['re']:.2g}, CFD polars at "
                                        f"Re {', '.join(f'{r:.2g}' for r in c['cached_re'])}: extrapolated. "
                                        f"Compute a polar at Re {c['re']:.2g}."))
                    else:
                        its.append(item("ok", "Section data", f"{c['surface']} ({c['airfoil']}): CFD polars cover "
                                                              f"Re {c['re']:.2g}"))
        if kind == "performance":
            if float(prm.get("power", 0)) <= 0:
                its.append(item("info", "Propulsion", "power 0: glider mode (glide polar, stall, sink)."))
            elif prm.get("propulsion") == "aircraft" and ac is not None and not ac.propulsion:
                its.append(item("warn", "Propulsion", "'aircraft' propulsion but the model has none saved: "
                                                      "the simple model will be used."))
            elif ac is not None and (ac.propulsion or {}).get("engine", {}).get("kind") == "electric" and \
                    not (ac.propulsion or {}).get("engine", {}).get("battery_kwh"):
                its.append(item("warn", "Battery", "electric engine without battery energy: range and endurance "
                                                   "will be zero. Set propulsion.engine.battery_kwh."))
        if kind == "structure":
            try:
                f_, r_ = (float(x) for x in str(prm.get("spars", "0.15,0.65")).split(","))
                if not 0 < f_ < r_ < 1:
                    raise ValueError
            except ValueError:
                its.append(item("error", "Spars", f"spars '{prm.get('spars')}' must be 'front,rear' with "
                                                  "0 < front < rear < 1."))
        if kind == "polar":
            pc = check_polar(prm["airfoil"], float(prm["re"]), prm["alpha"], float(prm.get("mach", 0)),
                             prm.get("transition", "free"), float(prm.get("ncrit", 9)), prm.get("xtr") or None,
                             float(prm.get("refine", 1)), int(prm.get("jobs", 2)))
            its += [i for i in pc["items"] if i["check"] != "Runtime"]
            rt = pc["runtime"]["expected_s"] if pc["runtime"] else 0.0
        else:
            rt = hist.get(kind, NODE_DEFAULT_S.get(kind, 5))
        ev = T.EVIDENCE.get(kind)
        if ev and status[nid]["state"] != "up to date":
            its.append(item("info", "Expected confidence", f"{T.LABEL[ev[0]]}: {ev[1]}"))
        will_run = nid in plan
        if will_run:
            total += rt
        res_nodes[nid] = {"level": worst(its), "items": its, "runtime_s": rt if will_run else 0.0,
                          "will_run": will_run, "label": n["label"]}
    top = []
    tpl = p.data.get("template")
    if tpl:
        for note in tpl.get("notes", []):
            top.append(item("info", f"Template: {tpl['title']}", note))
    if plan:
        top.append(item("info", "Runtime", f"{len(plan)} analyses to run, about {fmt_time(total)}"))
    else:
        top.append(item("ok", "Runtime", "everything is up to date"))
    allit = top + [i for v in res_nodes.values() for i in v["items"]]
    return {"level": worst(allit), "items": top, "nodes": res_nodes, "runtime_s": total, "plan": plan}


def _node_history():
    """median elapsed seconds per node kind over every project run on this machine"""
    by = {}
    root = CASES / "projects"
    for m in root.glob("*/runs/*/meta.json") if root.exists() else []:
        try:
            d = json.loads(m.read_text())
        except Exception:                                         # noqa: BLE001
            continue
        if d.get("state") == "ok" and d.get("elapsed") is not None:
            by.setdefault(d["kind"], []).append(d["elapsed"])
    return {k: float(np.median(v)) for k, v in by.items() if v}


# ----------------------------------------------------------------------
def _print(res):
    sym = {"ok": "  ok ", "info": " info", "warn": " WARN", "error": "ERROR"}
    for i in res["items"]:
        print(f"{sym[i['level']]}  {i['check']}: {i['detail']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("polar")
    a.add_argument("airfoil")
    a.add_argument("--re", type=float, required=True)
    a.add_argument("--alpha", default="-4:16:2")
    a.add_argument("--mach", type=float, default=0.0)
    a.add_argument("--transition", default="free", choices=["free", "off", "fixed"])
    a.add_argument("--xtr", default=None)
    a.add_argument("--refine", type=float, default=1.0)
    a.add_argument("--jobs", type=int, default=2)
    a.add_argument("--prandtl-glauert", action="store_true")
    a.add_argument("--deep", action="store_true", help="also build the mesh and check its quality (~3 s)")
    p = sub.add_parser("project")
    p.add_argument("project")
    p.add_argument("--variant", default=None)
    args = ap.parse_args([x for x in (argv if argv is not None else __import__("sys").argv[1:])])
    if args.cmd == "polar":
        r = check_polar(args.airfoil, args.re, args.alpha, args.mach, args.transition, xtr=args.xtr,
                        refine=args.refine, jobs=args.jobs, prandtl_glauert=args.prandtl_glauert, deep=args.deep)
        _print(r)
        print(f"-> {r['level'].upper()}")
    else:
        import project as PR
        pr = PR.Project.load(args.project)
        r = check_project(pr, args.variant)
        _print(r)
        for nid, v in r["nodes"].items():
            flag = "runs" if v["will_run"] else "up to date"
            print(f"  [{v['level']:5s}] {v['label']} ({flag})")
            for i in v["items"]:
                if i["level"] in ("warn", "error"):
                    print(f"        {i['level'].upper()} {i['check']}: {i['detail']}")
        print(f"-> {r['level'].upper()}")


if __name__ == "__main__":
    main()
