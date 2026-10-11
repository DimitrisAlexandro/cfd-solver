"""
Weight and balance, horizontal-tail sizing (scissor plot) and landing-gear
geometry checks.

    python weight_balance.py ../cases/aircraft/my_aircraft.json --out ../cases/wb

Model
  * Component list on the aircraft (`Aircraft.components`): name, group
    ("empty", "crew", "payload", "fuel", "battery"), mass, x, z, and an
    optional maximum (fuel and baggage are variable). Without a list, a
    default one is built from the geometry with Raymer/Roskam-type c.g.
    positions of each group (wing 40 % MAC, tails 40 % of their MAC,
    fuselage 0.42 L, tractor engine 0.10 L, ...).
  * Loading cases and the loading diagram ("potato" plot): mass against
    c.g. in % MAC for the payload-first and fuel-first sequences; forward
    and aft c.g. over all cases; the extreme c.g. positions that the
    stability and control analyses should use.
  * Scissor plot: the vortex lattice of the whole aircraft is solved with the
    horizontal tail scaled in area (aspect ratio and position kept):
      - stability limit: x_cg,aft = x_np - SM_min MAC (stick-fixed neutral
        point from the lattice, downwash and fuselage included);
      - control limit: x_cg,fwd where full nose-up elevator (times a
        large-deflection effectiveness, default 0.7) just trims the landing
        configuration (landing flap, landing CL_max) - moment about
        a reference point, transferred to the c.g.: Cm_cg = Cm_ref + CL (x_cg - x_ref)/c.
    Result: the c.g. range each tail size allows and the smallest tail that
    covers the c.g. range of the loading diagram (wing position fixed).
  * Landing gear (tricycle or taildragger): static nose/tail wheel load at
    forward and aft c.g., tip-back angle, turnover angle, rotation (tail
    strike) angle, propeller ground clearance - against Raymer's and CS-23's
    usual criteria (nose wheel 8-15 % of the weight, tip-back >= 15 deg,
    turnover <= 63 deg, propeller clearance >= 0.18 m).
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

import aircraft as A

GROUPS = ("empty", "crew", "payload", "fuel", "battery")


# ----------------------------------------------------------------------
# component positions
# ----------------------------------------------------------------------
def mac_x(s, frac):
    xm, _ = s.mac_position()
    return xm + frac * s.mac


def surface_z(s, frac_span=0.4):
    if s.vertical:
        return s.z + frac_span * s.semi
    return s.z + float(s.rise(frac_span))


def default_positions(ac, engine="tractor"):
    """x, z (m) for the usual mass groups, from the geometry"""
    f = ac.fuselage
    L = f.length if f.length > 0 else 0.75 * ac.b
    x0, z0 = f.x, f.z
    w = ac.wing
    pos = {"wing": (mac_x(w, 0.40), surface_z(w)),
           "fuselage": (x0 + 0.42 * L, z0),
           "engine": (x0 + 0.10 * L if engine == "tractor" else x0 + 0.92 * L, z0),
           "propeller": (x0 + 0.01 * L if engine == "tractor" else x0 + 1.0 * L, z0),
           "fuel system": (mac_x(w, 0.35), surface_z(w)),
           "fuel": (mac_x(w, 0.35), surface_z(w)),
           "flight controls": (mac_x(w, 0.6), z0),
           "avionics": (x0 + 0.22 * L, z0 + 0.1 * f.diameter),
           "electrical": (x0 + 0.20 * L, z0),
           "furnishings": (x0 + 0.36 * L, z0),
           "hydraulics": (x0 + 0.35 * L, z0),
           "air conditioning and anti-ice": (x0 + 0.25 * L, z0),
           "battery": (x0 + 0.30 * L, z0 - 0.2 * f.diameter),
           "pilot": (x0 + 0.33 * L, z0 + 0.05 * f.diameter),
           "front passenger": (x0 + 0.33 * L, z0 + 0.05 * f.diameter),
           "rear passengers": (x0 + 0.46 * L, z0 + 0.05 * f.diameter),
           "baggage": (x0 + 0.56 * L, z0)}
    for s in ac.surfaces[1:]:
        pos[s.name] = (mac_x(s, 0.40), surface_z(s))
    g = getattr(ac, "gear", None) or {}
    if g.get("x_main") is not None:
        pos["main gear"] = (float(g["x_main"]), float(g.get("z_ground", z0 - 1.0)) + 0.3)
        pos["nose gear"] = (float(g.get("x_nose", x0 + 0.1 * L)), float(g.get("z_ground", z0 - 1.0)) + 0.3)
    else:
        xm = mac_x(w, 0.55)
        pos["main gear"] = (xm, z0 - 0.6 * f.diameter)
        pos["nose gear"] = (x0 + 0.10 * L, z0 - 0.6 * f.diameter)
    return pos


def place(components, ac, engine="tractor"):
    """fill x, z of components that have none (by name, else by group)"""
    pos = default_positions(ac, engine)
    out = []
    for c in components:
        c = dict(c)
        if c.get("x") is None or c.get("z") is None:
            key = c["name"] if c["name"] in pos else c.get("position", c["name"])
            x, z = pos.get(key, (mac_x(ac.wing, 0.3), ac.cg[2]))
            if c.get("x") is None:
                c["x"] = x
            if c.get("z") is None:
                c["z"] = z
        out.append(c)
    return out


def default_components(ac, empty_mass=None, crew=2, crew_mass=86.0, fuel=80.0, baggage=20.0,
                       engine_mass=None):
    """a generic light-aircraft list when the aircraft has none: the empty
    mass split by Raymer-type group fractions, two seats, wing fuel, baggage"""
    m_e = empty_mass if empty_mass is not None else 0.62 * ac.mass
    fr = {"wing": 0.155, "horizontal": 0.025, "vertical": 0.015, "fuselage": 0.17, "engine": 0.24,
          "main gear": 0.05, "nose gear": 0.02, "fuel system": 0.025, "flight controls": 0.03,
          "avionics": 0.04, "electrical": 0.05, "furnishings": 0.09, "propeller": 0.03}
    if engine_mass:
        fr["engine"] = engine_mass / m_e
    s = sum(fr.values())
    comps = [{"name": "wing", "group": "empty", "mass": m_e * fr["wing"] / s}]
    for srf in ac.surfaces[1:]:
        k = "vertical" if srf.vertical else "horizontal"
        comps.append({"name": srf.name, "group": "empty", "mass": m_e * fr[k] / s})
    for k in ("fuselage", "engine", "propeller", "main gear", "nose gear", "fuel system", "flight controls",
              "avionics", "electrical", "furnishings"):
        comps.append({"name": k, "group": "empty", "mass": m_e * fr[k] / s})
    names = ["pilot", "front passenger", "rear passengers"]
    for i in range(crew):
        comps.append({"name": names[min(i, 2)] if i < 2 else "rear passengers", "group": "crew" if i == 0 else "payload",
                      "mass": crew_mass, "max": crew_mass})
    comps.append({"name": "fuel", "group": "fuel", "mass": fuel, "max": fuel})
    comps.append({"name": "baggage", "group": "payload", "mass": baggage, "max": baggage})
    return place(comps, ac)


def components_of(ac):
    if getattr(ac, "components", None):
        return place(ac.components, ac)
    return default_components(ac)


# ----------------------------------------------------------------------
# balance
# ----------------------------------------------------------------------
def cg_of(comps):
    m = sum(c["mass"] for c in comps)
    if m <= 0:
        return 0.0, 0.0, 0.0
    return m, sum(c["mass"] * c["x"] for c in comps) / m, sum(c["mass"] * c["z"] for c in comps) / m


def pct_mac(ac, x):
    xm, _ = ac.wing.mac_position()
    return 100 * (x - xm) / ac.c


def x_of_pct(ac, p):
    xm, _ = ac.wing.mac_position()
    return xm + p / 100 * ac.c


def loading(ac, comps=None, mtow=None):
    """loading cases and the loading diagram"""
    comps = comps or components_of(ac)
    empty = [c for c in comps if c.get("group", "empty") in ("empty", "battery")]
    crew = [c for c in comps if c.get("group") == "crew"]
    pay = [c for c in comps if c.get("group") == "payload"]
    fuel = [c for c in comps if c.get("group") == "fuel"]

    def state(items):
        m, x, z = cg_of(items)
        return {"mass": m, "x": x, "z": z, "pct_mac": pct_mac(ac, x)}

    def scaled(c, f):
        d = dict(c)
        d["mass"] = c.get("max", c["mass"]) * f
        return d
    E = state(empty)
    cases = {"empty": E, "empty + crew": state(empty + crew),
             "crew, full fuel": state(empty + crew + [scaled(c, 1) for c in fuel]),
             "crew + payload, no fuel": state(empty + crew + [scaled(c, 1) for c in pay]),
             "crew + payload + full fuel": state(empty + crew + [scaled(c, 1) for c in pay]
                                                 + [scaled(c, 1) for c in fuel]),
             "as defined": state(comps)}
    # loading diagram: add payload items one by one (front to back and back to front), then fuel
    seqs = {}
    for lab, order in (("payload front to back, then fuel", sorted(pay, key=lambda c: c["x"])),
                       ("payload back to front, then fuel", sorted(pay, key=lambda c: -c["x"])),
                       ("fuel first, then payload front to back", None)):
        pts = []
        cur = list(empty + crew)
        pts.append(state(cur))
        if order is None:
            for f in fuel:
                cur.append(scaled(f, 1))
                pts.append(state(cur))
            for p in sorted(pay, key=lambda c: c["x"]):
                cur.append(scaled(p, 1))
                pts.append(state(cur))
        else:
            for p in order:
                cur.append(scaled(p, 1))
                pts.append(state(cur))
            for f in fuel:
                for k in np.linspace(0.25, 1.0, 4):
                    pts.append(state(cur + [scaled(f, k)]))
                cur.append(scaled(f, 1))
        seqs[lab] = pts
    allp = [p for s in seqs.values() for p in s] + list(cases.values())[1:]
    if mtow:
        allp = [p for p in allp if p["mass"] <= mtow * 1.0001]
    fwd = min(allp, key=lambda p: p["x"])
    aft = max(allp, key=lambda p: p["x"])
    return {"cases": cases, "sequences": seqs, "forward": fwd, "aft": aft,
            "range_pct_mac": aft["pct_mac"] - fwd["pct_mac"],
            "max_mass": max(p["mass"] for p in allp), "components": comps}


def place_wing(ac, comps, target_pct=25.0, case="crew + payload + full fuel", it=8):
    """move the wing (and the components attached to it) along x so that
    the c.g. of the given loading case sits at target % MAC"""
    ac = copy.deepcopy(ac)
    attached = {"wing", "fuel", "fuel system", "flight controls", "main gear"}
    for _ in range(it):
        ld = loading(ac, comps)
        x = ld["cases"][case]["x"]
        dx = x - x_of_pct(ac, target_pct)
        if abs(dx) < 1e-5:
            break
        m_att = sum(c.get("max", c["mass"]) if c["group"] == "fuel" else c["mass"]
                    for c in comps if c["name"] in attached)
        m_tot = ld["cases"][case]["mass"]
        # moving the wing by d moves the c.g. by d m_att/m_tot and the target by d
        d = dx / max(1 - m_att / m_tot, 0.05)
        ac.wing.x += d
        for c in comps:
            if c["name"] in attached:
                c["x"] += d
    return ac, comps


# ----------------------------------------------------------------------
# scissor plot
# ----------------------------------------------------------------------
def tail_index(ac):
    for k, s in enumerate(ac.surfaces):
        if k > 0 and not s.vertical and s.x > ac.wing.x:
            return k
    return None


def scale_tail(ac, k, factor):
    """horizontal tail area x factor, aspect ratio, sweep and quarter-chord
    position of its MAC kept"""
    ac = copy.deepcopy(ac)
    s = ac.surfaces[k]
    xq0 = mac_x(s, 0.25)
    f = math.sqrt(factor)
    s.span *= f
    s.root_chord *= f
    s.tip_chord *= f
    for kk in s.kinks:
        kk["y"] *= f
        kk["chord"] *= f
        kk["dx"] = kk.get("dx", 0.0) * f
        kk["dz"] = kk.get("dz", 0.0) * f
    s.x += xq0 - mac_x(s, 0.25)
    return ac


def scissor(ac, sm_min=0.05, cl_land=None, flap_land=30.0, elevator_max=-25.0, factors=None,
            speed=None, altitude=0.0, polar_dir=None, log=None, elevator_eff=0.7):
    """stability and control limits against the horizontal-tail area.
    elevator_eff: large-deflection effectiveness of the elevator (the
    lattice is linear in the deflection; plain flaps lose effectiveness at
    large angles, DATCOM correlation about 0.65-0.75 at 25 deg)"""
    k = tail_index(ac)
    if k is None:
        raise ValueError("no horizontal tail behind the wing")
    s0 = ac.surfaces[k].area
    factors = factors if factors is not None else np.linspace(0.4, 1.6, 7)
    has_flap = any(c.name == "flap" for c in ac.wing.controls)
    has_elev = any(c.name == "elevator" for c in ac.surfaces[k].controls)
    if not has_elev:
        raise ValueError("the horizontal tail has no elevator")
    if speed is None:
        speed = 1.3 * math.sqrt(2 * ac.mass * A.G0 / (1.225 * ac.S * (cl_land or 1.8)))
    rows = []
    for f in factors:
        a = scale_tail(ac, k, f)
        an = A.Analysis(a, speed, altitude, polar_dir)
        der = an.derivatives(2.0, downwash=False)
        x_np = der["x_np"]
        # control: landing configuration, full nose-up elevator, about the wing MAC leading edge
        xm, _ = a.wing.mac_position()
        ref = (xm, 0.0, a.cg[2])
        ctr = {"elevator": elevator_eff * elevator_max} | ({"flap": flap_land} if has_flap else {})
        r1 = an.solve(alpha=0.0, controls=ctr, ref=ref)
        r2 = an.solve(alpha=8.0, controls=ctr, ref=ref)
        target = cl_land if cl_land is not None else 1.9
        al = 8.0 * (target - r1["CL"]) / (r2["CL"] - r1["CL"])
        r = an.solve(alpha=al, controls=ctr, ref=ref)
        # Cm_cg = Cm_ref + CL (x_cg - x_ref) / c = 0
        x_fwd = xm - r["Cm"] * a.c / r["CL"]
        rows.append({"factor": float(f), "Sh": a.surfaces[k].area, "Sh_S": a.surfaces[k].area / a.S,
                     "x_np": x_np, "x_aft": x_np - sm_min * a.c, "x_fwd": x_fwd,
                     "pct_np": pct_mac(a, x_np), "pct_aft": pct_mac(a, x_np - sm_min * a.c),
                     "pct_fwd": pct_mac(a, x_fwd), "alpha_land": al})
        if log:
            log(f"  tail x{f:.2f}: S_h/S {rows[-1]['Sh_S']:.3f}, fwd {rows[-1]['pct_fwd']:.1f} %, "
                f"aft {rows[-1]['pct_aft']:.1f} % MAC")
    return {"rows": rows, "Sh_S_current": s0 / ac.S, "sm_min": sm_min, "cl_land": cl_land,
            "elevator_max": elevator_max, "elevator_eff": elevator_eff, "flap_land": flap_land, "speed": speed,
            "tail": ac.surfaces[k].name}


def min_tail(sc, fwd_pct, aft_pct):
    """smallest S_h/S whose c.g. window contains [fwd_pct, aft_pct]"""
    r = sc["rows"]
    sh = np.array([x["Sh_S"] for x in r])
    lo = np.array([x["pct_fwd"] for x in r])
    hi = np.array([x["pct_aft"] for x in r])
    ok = (lo <= fwd_pct) & (hi >= aft_pct)
    if not ok.any():
        return None
    # interpolate the crossing of each limit
    need = []
    for lim, val, sign in ((lo, fwd_pct, -1), (hi, aft_pct, 1)):
        g = sign * (lim - val)               # >= 0 where satisfied
        if np.all(g >= 0):
            need.append(sh[0])
            continue
        i = int(np.argmax(g >= 0))
        if i == 0:
            need.append(sh[0])
        else:
            need.append(float(np.interp(0.0, [g[i - 1], g[i]], [sh[i - 1], sh[i]])))
    return max(need)


# ----------------------------------------------------------------------
# landing gear
# ----------------------------------------------------------------------
def gear_checks(ac, ld, gear=None, prop_diameter=None):
    """tricycle (x_nose < x_main) or taildragger (x_tail > x_main) checks"""
    g = dict(getattr(ac, "gear", None) or {}) | dict(gear or {})
    f = ac.fuselage
    L = f.length if f.length > 0 else 0.75 * ac.b
    if g.get("x_main") is None:
        return None
    xm, xn = float(g["x_main"]), float(g.get("x_nose", f.x + 0.1 * L))
    track = float(g.get("track", 2.2))
    zg = float(g.get("z_ground", f.z - 1.0))                    # ground height (geometry axes)
    tri = xn < xm
    out = {"type": "tricycle" if tri else "taildragger", "x_main": xm, "x_nose_or_tail": xn, "track": track,
           "z_ground": zg}
    fwd, aft = ld["forward"], ld["aft"]
    for lab, p in (("forward", fwd), ("aft", aft)):
        h = p["z"] - zg
        share = (xm - p["x"]) / (xm - xn)                       # load on the nose (or tail) wheel
        out[f"nose_load_{lab}"] = share
        out[f"cg_height_{lab}"] = h
    h_aft = aft["z"] - zg
    if tri:
        out["tipback_deg"] = math.degrees(math.atan2(xm - aft["x"], h_aft))
        # turnover: plan-view distance from the c.g. (forward, highest) to the nose-main wheel line
        h_f = fwd["z"] - zg
        dx = xm - xn
        d = abs((fwd["x"] - xn) * (0.5 * track) / math.hypot(dx, 0.5 * track))
        out["turnover_deg"] = math.degrees(math.atan2(h_f, max(d, 1e-6)))
    else:
        out["tipback_deg"] = None
        out["turnover_deg"] = None
    # rotation (tail strike) angle: from the main wheel contact to the lowest aft fuselage point
    xs = f.x + np.linspace(0.6, 1.0, 40) * L
    zb = f.z - f.radius(xs)
    ang = np.degrees(np.arctan2(zb - zg, np.maximum(xs - xm, 1e-6)))
    out["rotation_deg"] = float(ang[xs > xm].min()) if np.any(xs > xm) else None
    # propeller clearance (tractor: propeller at the nose, on the axis)
    if prop_diameter:
        out["prop_clearance"] = (f.z - zg) - 0.5 * prop_diameter
    chk = {}
    if tri:
        chk["nose wheel load 5-20 % (8-15 % ideal)"] = \
            (0.05 <= out["nose_load_aft"] and out["nose_load_forward"] <= 0.20)
        chk["tip-back angle >= 15°"] = out["tipback_deg"] >= 15.0
        chk["turnover angle <= 63°"] = out["turnover_deg"] <= 63.0
    else:
        chk["tail wheel load 5-15 %"] = 0.04 <= out["nose_load_forward"] <= 0.18
    if out["rotation_deg"] is not None:
        chk["rotation angle >= 12°"] = out["rotation_deg"] >= 12.0
    if prop_diameter:
        chk["propeller clearance >= 0.18 m"] = out["prop_clearance"] >= 0.18
    out["checks"] = chk
    return out


def suggest_gear(ac, ld, prop_diameter=1.8, tipback=16.0, clearance=0.23, rotation=13.0, nose_fwd=0.15):
    """tricycle gear placed to meet the criteria: height from the propeller
    clearance (raised until the rotation angle is met), main gear from the
    tip-back angle at the aft c.g., nose gear for at most nose_fwd of the
    weight at the forward c.g. (as far forward as the fuselage allows), track
    for a 55 deg turnover angle"""
    f = ac.fuselage
    L = f.length if f.length > 0 else 0.75 * ac.b
    aft, fwd = ld["aft"], ld["forward"]
    zg = f.z - (0.5 * prop_diameter + clearance)
    for _ in range(40):
        xm = aft["x"] + (aft["z"] - zg) * math.tan(math.radians(tipback))
        trial = dict(getattr(ac, "gear", None) or {}) | {"x_main": xm, "x_nose": f.x + 0.1 * L, "track": 2.0,
                                                          "z_ground": zg}
        g = gear_checks(_with_gear(ac, trial), ld)
        if g["rotation_deg"] is None or g["rotation_deg"] >= rotation:
            break
        zg -= 0.03
    xn = xm - (xm - fwd["x"]) / nose_fwd
    xn = min(max(xn, f.x + 0.06 * L), xm - 0.25 * L)
    h = fwd["z"] - zg
    d_req = h / math.tan(math.radians(55.0))
    dx = xm - xn
    a = max(fwd["x"] - xn, 1e-3)
    ratio = d_req / a
    half = ratio * dx / math.sqrt(max(1 - ratio**2, 1e-6)) if ratio < 1 else 3.0
    return {"x_main": float(xm), "x_nose": float(xn), "track": float(max(2 * half, 1.5)), "z_ground": float(zg)}


def _with_gear(ac, gear):
    a = copy.copy(ac)
    a.gear = gear
    return a


# ----------------------------------------------------------------------
def report(ac, sm_min=0.05, cl_land=None, flap_land=30.0, elevator_max=-25.0, prop_diameter=None,
           polar_dir=None, log=print):
    comps = components_of(ac)
    ld = loading(ac, comps)
    log(f"[w&b] c.g. {ld['forward']['pct_mac']:.1f} … {ld['aft']['pct_mac']:.1f} % MAC")
    sc = None
    try:
        sc = scissor(ac, sm_min, cl_land, flap_land, elevator_max, polar_dir=polar_dir, log=log)
        sc["min_Sh_S"] = min_tail(sc, ld["forward"]["pct_mac"], ld["aft"]["pct_mac"])
    except ValueError as e:
        log(f"[w&b] scissor plot skipped: {e}")
    gc = gear_checks(ac, ld, prop_diameter=prop_diameter)
    return {"aircraft": ac.name, "loading": ld, "scissor": sc, "gear": gc}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aircraft")
    ap.add_argument("--sm-min", type=float, default=0.05)
    ap.add_argument("--cl-land", type=float, default=None)
    ap.add_argument("--flap", type=float, default=30.0)
    ap.add_argument("--elevator-max", type=float, default=-25.0)
    ap.add_argument("--prop-diameter", type=float, default=None)
    ap.add_argument("--polar-dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    ac = A.Aircraft.load(a.aircraft)
    res = report(ac, a.sm_min, a.cl_land, a.flap, a.elevator_max, a.prop_diameter, a.polar_dir)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "weight_balance.json").write_text(json.dumps(A._jsonable(res), indent=1))


if __name__ == "__main__":
    main()
