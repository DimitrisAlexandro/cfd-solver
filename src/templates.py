"""Project templates: a ready-to-run project for each kind of aircraft.

Each template builds an aircraft model with sensible defaults for its class and a
project around it (stability, drag, weight and balance, loads, wing structure,
aeroelasticity, performance) with the flight conditions of that class already set.

- trainer, motor-glider, eVTOL (wing-borne cruise): sized with the conceptual sizing
  tool (constraint diagram, mission, Raymer Class II masses);
- sailplane and small fixed-wing UAV: Raymer's general-aviation statistics don't hold
  there, so the geometry comes from the sizing tool's layout rules and the masses
  from typical fractions of the class (stated in the template).

    python templates.py list
    python templates.py new uav "Survey UAV"
"""
import argparse
import copy
import json
import math

import sizing as SZ
import wing as W

G0 = 9.80665

TEMPLATES = {
    "trainer": {
        "title": "Two-seat trainer",
        "description": "Piston two-seater, 55 m/s cruise at 1500 m, 900 km range, metal, fixed gear. "
                       "Sized from requirements (Raymer Class II).",
        "method": "sized",
        "req": {"name": "Two-seat trainer"},
        "loads_category": "utility",
        "notes": [],
    },
    "motor_glider": {
        "title": "Motor-glider (self-launching, electric)",
        "description": "Two-seat powered sailplane, AR 18, composite; electric self-launch and 60 km sustainer "
                       "range, 30 m/s cruise. Sized from requirements (powered-sailplane statistics).",
        "method": "sized",
        "req": {"name": "Motor-glider", "category": "powered sailplane", "crew": 1, "passengers": 1,
                "baggage": 10.0, "range_km": 60.0, "reserve_min": 10.0, "cruise_speed": 30.0,
                "cruise_alt": 1000.0, "cruise_power": 0.6, "stall_speed": 21.0, "takeoff_distance": 400.0,
                "landing_distance": 400.0, "climb_rate": 2.5, "ceiling": 3000.0, "aspect_ratio": 18.0,
                "taper": 0.45, "dihedral": 3.0, "twist": -2.0, "airfoil": "naca2412", "cl_max_clean": 1.45,
                "cl_max_takeoff": 1.6, "cl_max_landing": 1.75, "flap_takeoff": 5.0, "flap_landing": 15.0,
                "engine": "electric", "battery_wh_kg": 220.0, "composite": True, "seats_abreast": 2,
                "load_factor": 5.3, "avionics_kg": 6.0, "gear": "retractable"},
        "structure": {"material": "CFRP (UD caps, ±45 skins)", "aileron_mass_per_area": 2.5,
                      "aileron_cg_aft_hinge": 0.0, "aileron_freq_hz": 20.0},
        "loads_category": "utility",
        "notes": ["Sized for a self-launch climb and a short sustainer cruise; soaring performance is the "
                  "glide polar on the Performance results."],
    },
    "evtol": {
        "title": "eVTOL, lift + cruise (wing-borne flight)",
        "description": "Two-seat electric lift + cruise configuration, 55 m/s cruise, 100 km range. The wing, "
                       "tail, cruise propulsion and the wing-borne envelope are analysed.",
        "method": "sized",
        "req": {"name": "eVTOL lift + cruise", "crew": 1, "passengers": 1, "baggage": 10.0, "range_km": 100.0,
                "reserve_min": 20.0, "cruise_speed": 55.0, "cruise_alt": 600.0, "stall_speed": 30.0,
                "takeoff_distance": 800.0, "landing_distance": 800.0, "climb_rate": 3.0, "ceiling": 3000.0,
                "aspect_ratio": 9.0, "taper": 0.5, "engine": "electric", "battery_wh_kg": 250.0,
                "composite": True, "load_factor": 3.8, "cg_target": 12.0},
        "structure": {"material": "CFRP (UD caps, ±45 skins)", "aileron_mass_per_area": 3.0,
                      "aileron_cg_aft_hinge": 0.0, "aileron_freq_hz": 20.0},
        "loads_category": "normal",
        "notes": ["c.g. target 12 % MAC: the short wing and long cabin need a forward c.g. for a positive "
                  "static margin.",
                  "Hover and transition (lift rotors, their mass and drag) are NOT modelled: add the "
                  "lift-system mass on the aircraft model and treat take-off/landing distances as "
                  "conventional-aircraft values only."],
    },
    "glider": {
        "title": "Sailplane (15 m class)",
        "description": "Single-seat 15 m sailplane, 400 kg, AR 21, composite, no engine. Glide polar, "
                       "stability, loads (utility n 5.3), structure and flutter.",
        "method": "fixed",
        "req": {"name": "15 m sailplane", "category": "powered sailplane", "crew": 1, "passengers": 0,
                "baggage": 10.0, "cruise_speed": 30.0, "cruise_alt": 1000.0, "aspect_ratio": 21.0,
                "taper": 0.4, "dihedral": 3.0, "twist": -1.5, "airfoil": "naca2412", "engine": "electric",
                "composite": True, "seats_abreast": 1, "load_factor": 5.3, "gear": "retractable",
                "cg_target": 30.0},
        "W0": 400.0, "wing_loading": 38.0, "power_kw": 0.0, "fuselage_diameter": 0.65,
        "masses": {"wing": 115.0, "horizontal tail": 7.0, "fin": 9.0, "fuselage": 70.0, "main gear": 10.0,
                   "nose gear": 3.0, "flight controls": 12.0, "avionics": 8.0, "electrical": 4.0,
                   "furnishings": 8.0},
        "battery": 0.0,
        "structure": {"material": "GFRP (UD caps, ±45 skins)", "aileron_mass_per_area": 2.0,
                      "aileron_cg_aft_hinge": 0.0, "aileron_freq_hz": 20.0},
        "loads_category": "utility",
        "notes": ["Masses from typical 15 m sailplane fractions (empty about 260 kg), not from Raymer's "
                  "statistics. No water ballast. Retractable monowheel; the gear checks assume a tricycle "
                  "layout and are advisory.",
                  "NACA 2412 sections (CFD polars shipped with AeroLab): a laminar sailplane section would "
                  "glide much better (real 15 m class L/D ~40). Change surfaces.wing.airfoil and compute its "
                  "polars to see it."],
    },
    "uav": {
        "title": "Fixed-wing UAV (15 kg, electric)",
        "description": "15 kg electric survey UAV, 3.5 m span, 22 m/s cruise, 4 kg payload. Section data at "
                       "Re ~5e5: compute CFD section polars at that Re for stall and drag you can quote.",
        "method": "fixed",
        "req": {"name": "Fixed-wing UAV", "category": "homebuilt, composite", "crew": 0, "passengers": 0,
                "person_mass": 0.0, "baggage": 4.0, "cruise_speed": 22.0, "cruise_alt": 300.0,
                "aspect_ratio": 10.0, "taper": 0.6, "dihedral": 3.0, "twist": -1.0, "airfoil": "naca2412",
                "engine": "electric", "rpm": 6000.0, "battery_wh_kg": 200.0, "composite": True,
                "seats_abreast": 1, "load_factor": 4.0, "gear": "retractable", "cg_target": 25.0},
        "W0": 15.0, "wing_loading": 12.0, "power_kw": 1.5, "fuselage_diameter": 0.18, "fuselage_length": 1.8,
        "masses": {"wing": 2.0, "horizontal tail": 0.3, "fin": 0.25, "fuselage": 2.3, "engine": 0.8,
                   "propeller": 0.15, "avionics": 1.2, "electrical": 0.5},
        "battery": 3.2,
        "structure": {"material": "CFRP (UD caps, ±45 skins)", "aileron_mass_per_area": 0.6,
                      "aileron_cg_aft_hinge": 0.0, "aileron_freq_hz": 30.0},
        "loads_category": "normal",
        "notes": ["Catapult launch and belly landing: no landing gear (take-off and landing distances are "
                  "not meaningful). The structure sizing uses light-aircraft minimum gauges, so the wing "
                  "box comes out heavier than a real UAV wing.",
                  "Masses from typical small-UAV fractions, not Raymer's statistics. Below Re 1e6 the "
                  "cached NACA polars (Re 2.2e6+) are extrapolated: run section polars at the UAV's Re "
                  "(Airfoil polar, low-Re mode) before quoting stall or drag."],
    },
}

ORDER = ["trainer", "motor_glider", "glider", "uav", "evtol"]


def _eas(v, alt):
    return v * math.sqrt(W.isa(alt)[0] / 1.225)


def build(tid):
    """(Aircraft, info) for a template: info has mass, power, fuel, battery, cruise etc."""
    t = TEMPLATES[tid]
    req = SZ.default_requirements() | copy.deepcopy(t["req"])
    if t["method"] == "sized":
        res, ac = SZ.size(req, log=lambda *a: None)
        info = {"mtow": res["mtow"], "power_kw": res["power_kw"], "fuel": res["fuel"], "battery": res["battery"],
                "wing_area": res["wing_area"], "span": res["span"], "converged": res["converged"]}
    else:
        W0 = t["W0"]
        S = W0 / t["wing_loading"]
        p_kw = t["power_kw"]
        ac = SZ.build_aircraft(req, W0, S, p_kw)
        if t.get("fuselage_length"):
            k = t["fuselage_length"] / ac.fuselage.length
            ac.fuselage.length = t["fuselage_length"]
            for s in ac.surfaces:
                s.x *= k
        if t.get("fuselage_diameter"):
            ac.fuselage.diameter = t["fuselage_diameter"]
        masses = dict(t["masses"])
        m_known = sum(masses.values()) + t.get("battery", 0.0) + req["baggage"] + \
            (req["crew"] + req["passengers"]) * req["person_mass"]
        masses["furnishings"] = masses.get("furnishings", 0.0) + max(W0 - m_known, 0.0)   # balance to W0
        ac, comps, ld, prop_d, gc, db, _ = SZ.finalize(req, ac, W0, masses, 0.0, t.get("battery", 0.0), p_kw,
                                                       class_ii_masses=False)
        info = {"mtow": W0, "power_kw": p_kw, "fuel": 0.0, "battery": t.get("battery", 0.0), "wing_area": ac.S,
                "span": ac.b, "converged": True}
    if req["engine"] == "electric" and ac.propulsion:
        ac.propulsion["engine"]["battery_kwh"] = info["battery"] * req["battery_wh_kg"] / 1000
    ac.name = t["title"]
    info = {k: (bool(v) if k == "converged" else float(v)) for k, v in info.items()}
    info |= {"cruise_speed": req["cruise_speed"], "cruise_alt": req["cruise_alt"], "engine": req["engine"],
             "prop_diameter": (ac.propulsion or {}).get("propeller", {}).get("design", {}).get("diameter", 0.0)}
    return ac, info


def node_params(tid, info):
    """flight conditions and options of every analysis for this class"""
    t = TEMPLATES[tid]
    v, h = info["cruise_speed"], info["cruise_alt"]
    p = info["power_kw"]
    glider = p <= 0
    eng = "none" if glider else ("electric" if info["engine"] == "electric" else "piston")
    return {
        "stability": {"speed": round(v, 1), "alt": h},
        "drag": {"speed": round(v, 1), "alt": h, "gear": TEMPLATES[tid]["req"].get("gear", "fixed_faired"),
                 "power": round(0.75 * p, 2) if eng == "piston" else 0.0, "engine": eng},
        "loads": {"vc": round(_eas(v, h), 1), "category": t["loads_category"], "alt": 0.0,
                  "fuel": round(info["fuel"], 1)},
        "structure": dict(t.get("structure") or {}),
        "performance": {"power": round(p, 2), "fuel": round(info["fuel"], 1), "alt": 0.0,
                        "propulsion": "aircraft" if not glider else "simple",
                        "prop_diameter": round(info.get("prop_diameter") or 1.8, 3), "v_design": round(v, 1)},
    }


def create_project(tid, name=None, root=None, description=None):
    """aircraft json + project with the class's settings (returns the Project)"""
    import project as PR
    ac, info = build(tid)
    t = TEMPLATES[tid]
    p = PR.Project.create(name or t["title"], "aircraft", "demo", root=root,
                          description=description or t["description"])
    src = p.dir / "aircraft_source.json"
    ac.save(src)
    nodes = p.nodes("baseline")
    nodes["aircraft"]["params"]["source"] = str(src)
    for nid, prm in node_params(tid, info).items():
        nodes[nid]["params"].update(prm)
    p.data["template"] = {"id": tid, "title": t["title"], "notes": t["notes"], "sizing": info}
    p.save()
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    n = sub.add_parser("new")
    n.add_argument("template", choices=ORDER)
    n.add_argument("name", nargs="?")
    n.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for k in ORDER:
            print(f"{k:13s} {TEMPLATES[k]['title']}: {TEMPLATES[k]['description']}")
        return
    p = create_project(a.template, a.name, a.root)
    s = p.data["template"]["sizing"]
    print(f"[templates] {p.name}: MTOW {s['mtow']:.1f} kg, S {s['wing_area']:.2f} m², b {s['span']:.2f} m, "
          f"P {s['power_kw']:.1f} kW -> {p.dir}")
    print(json.dumps({k: v["params"] for k, v in p.nodes("baseline").items() if k != "aircraft"}, indent=1))


if __name__ == "__main__":
    main()
