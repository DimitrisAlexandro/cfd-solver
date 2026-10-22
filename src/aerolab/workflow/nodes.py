"""The node kinds of a project (project.py): for each kind its parameters, inputs, files and
the command line that runs it, and the key results shown when variants are compared."""
import json
from pathlib import Path


def _polar_dir():
    """the section polar library: project.POLAR_DIR, read at run time (a test may move it)"""
    from aerolab.workflow import project
    return project.POLAR_DIR


def _ac(inp):
    return str(inp["aircraft"] / "aircraft.json")


def _args_polar(p, inp, out):
    a = ["airfoil.py", "polar", p["airfoil"], "--re", f"{float(p['re']):.6g}", f"--alpha={p['alpha']}",
         "--refine", f"{float(p['refine']):g}", "--jobs", str(p["jobs"]), "--out", str(out), "--no-fields"]
    if p["transition"] == "off":
        a += ["--transition", "off"]
    elif p["transition"].startswith("fixed"):
        a += ["--xtr", p.get("xtr") or "0.05,0.05"]
    else:
        a += ["--ncrit", f"{float(p['ncrit']):g}"]
    if float(p["mach"]) > 0:
        a += ["--mach", f"{float(p['mach']):.4g}"]
    return a


def _args_performance(p, inp, out):
    a = ["performance.py", _ac(inp), "--power", p["power"], "--fuel", p["fuel"], "--alt", p["alt"],
         "--prop-diameter", p.get("prop_diameter", 1.8), "--v-design", p.get("v_design", 60.0),
         "--polar-dir", _polar_dir(), "--out", out]
    if p.get("propulsion") == "aircraft" and float(p["power"]) > 0:
        prop = json.loads(Path(_ac(inp)).read_text()).get("propulsion")
        if prop:
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "propulsion.json").write_text(json.dumps(prop, indent=1))
            a += ["--propulsion", Path(out) / "propulsion.json"]
    return a


KINDS = {
    "aircraft": {
        "title": "Aircraft model", "script": "aircraft.py", "inputs": [], "output": "aircraft.json",
        "params": {"source": "demo", "overrides": {}},
        "help": {"source": "'demo' or the path of an aircraft .json",
                 "overrides": "changes applied to the source: {'mass': 800, 'surfaces.wing.span': 11}"},
        "files": ["source"]},
    "stability": {
        "title": "Stability and control", "script": "aircraft.py", "inputs": ["aircraft"],
        "output": "stability.json", "params": {"speed": 50.0, "alt": 1000.0},
        "args": lambda p, inp, out: ["aircraft.py", "analyse", _ac(inp), "--speed", p["speed"], "--alt", p["alt"],
                                     "--polar-dir", _polar_dir(), "--out", out / "stability.json"]},
    "drag": {
        "title": "Drag buildup", "script": "drag_buildup.py", "inputs": ["aircraft"], "output": "drag_buildup.json",
        "params": {"speed": 60.0, "alt": 0.0, "gear": "fixed_faired", "power": 0.0, "engine": "piston"},
        "args": lambda p, inp, out: ["drag_buildup.py", _ac(inp), "--speed", p["speed"], "--alt", p["alt"],
                                     "--gear", p["gear"], "--power", p["power"],
                                     "--engine", p.get("engine", "piston"), "--out", out]},
    "weight_balance": {
        "title": "Weight and balance", "script": "weight_balance.py", "inputs": ["aircraft"],
        "output": "weight_balance.json", "params": {"sm_min": 0.05},
        "args": lambda p, inp, out: ["weight_balance.py", _ac(inp), "--sm-min", p["sm_min"],
                                     "--polar-dir", _polar_dir(), "--out", out]},
    "loads": {
        "title": "Flight loads", "script": "loads.py", "inputs": ["aircraft"], "output": "loads.json",
        "params": {"vc": 60.0, "category": "utility", "alt": 0.0, "fuel": 0.0},
        "args": lambda p, inp, out: ["loads.py", _ac(inp), "--vc", p["vc"], "--category", p["category"],
                                     "--alt", p["alt"], "--fuel", p["fuel"], "--polar-dir", _polar_dir(),
                                     "--out", out]},
    "structure": {
        "title": "Wing structure", "script": "wing_structure.py", "inputs": ["aircraft", "loads"],
        "output": "structure.json", "params": {"material": "aluminium 2024-T3", "spars": "0.15,0.65",
                                               "aileron_mass_per_area": 4.0, "aileron_cg_aft_hinge": 0.30,
                                               "aileron_freq_hz": 15.0},
        "help": {"aileron_cg_aft_hinge": "aileron c.g. behind its hinge, fraction of the aileron chord "
                                         "(0 = mass balanced, < 0 = overbalanced)",
                 "aileron_mass_per_area": "kg/m² of aileron", "aileron_freq_hz": "aileron rotation frequency "
                                                                                 "(control-circuit stiffness)"},
        "args": lambda p, inp, out: ["wing_structure.py", _ac(inp), str(inp["loads"] / "loads.json"),
                                     "--material", p["material"], "--spars", p["spars"], "--out", out]},
    "aeroelastic": {
        "title": "Aeroelasticity", "script": "aeroelastic.py", "inputs": ["structure"], "output": "aeroelastic.json",
        "params": {},
        "args": lambda p, inp, out: ["aeroelastic.py", str(inp["structure"] / "aircraft_structured.json"),
                                     "--out", out]},
    "performance": {
        "title": "Performance", "script": "performance.py", "inputs": ["aircraft"], "output": "performance.json",
        "params": {"power": 75.0, "fuel": 60.0, "alt": 0.0, "propulsion": "simple", "prop_diameter": 1.8,
                   "v_design": 60.0},
        "help": {"power": "kW at sea level; 0 = glider (glide polar only)",
                 "propulsion": "simple (power × η) or aircraft (the model's saved engine + BEMT propeller)"},
        "args": lambda p, inp, out: _args_performance(p, inp, out)},
    "polar": {
        "title": "Section polar (CFD)", "script": "airfoil.py", "inputs": [], "output": "polar.json",
        "params": {"airfoil": "naca2412", "re": 3e6, "mach": 0.0, "alpha": "-4:16:2", "transition": "free",
                   "ncrit": 9.0, "xtr": "", "refine": 1.0, "jobs": 2},
        "help": {"transition": "free, off (fully turbulent) or fixed (uses xtr)"},
        "files": ["airfoil"], "args": _args_polar},
}

# key results for comparisons: (key, label, unit, better, getter)
METRICS = {
    "aircraft": [("mass", "Mass", "kg", None, lambda d: d["mass"]),
                 ("span", "Wing span", "m", None, lambda d: _wing(d)["span"]),
                 ("area", "Wing area", "m²", None, lambda d: _wing_area(_wing(d))),
                 ("AR", "Aspect ratio", "", None, lambda d: _wing(d)["span"] ** 2 / _wing_area(_wing(d)))],
    "stability": [("static_margin", "Static margin", "% MAC", None, lambda d: 100 * d["derivatives"]["static_margin"]),
                  ("trim_alpha", "Trim α", "deg", None, lambda d: d["trim"]["alpha_trim"]),
                  ("trim_LD", "Trim L/D", "", "higher", lambda d: d["trim"]["L_D"]),
                  ("sp_zeta", "Short-period ζ", "", None, lambda d: d["modes"]["short period"]["zeta"]),
                  ("dr_zeta", "Dutch-roll ζ", "", "higher", lambda d: d["modes"]["dutch roll"]["zeta"])],
    "drag": [("CD0", "CD0", "", "lower", lambda d: d["CD0"]),
             ("oswald", "Oswald e", "", "higher", lambda d: d["oswald_e"])],
    "weight_balance": [("cg_range", "c.g. range", "% MAC", None, lambda d: d["loading"]["range_pct_mac"]),
                       ("min_tail", "Minimum Sh/S", "", "lower", lambda d: d["scissor"]["min_Sh_S"])],
    "loads": [("n_pos", "Limit load factor", "", None, lambda d: d["vn"]["n_pos"]),
              ("VD", "V_D", "m/s", None, lambda d: d["vn"]["Vd"]),
              ("root_M", "Root bending, max", "kN m", "lower", lambda d: d["envelope"]["bending_max"][0] / 1e3),
              ("root_V", "Root shear, max", "kN", "lower", lambda d: d["envelope"]["shear_max"][0] / 1e3)],
    "structure": [("wing_mass", "Wing structure mass", "kg", "lower", lambda d: d["wing_mass"]),
                  ("tip_defl", "Tip deflection, limit", "m", "lower", lambda d: d["tip_deflection_max"]),
                  ("V_flutter_est", "Flutter (typical section)", "m/s", "higher",
                   lambda d: d["aeroelastic"]["flutter"]["V_flutter"]),
                  ("V_div", "Divergence, EAS", "m/s", "higher", lambda d: d["aeroelastic"]["V_divergence_eas"])],
    "aeroelastic": [("V_flutter", "Flutter speed (3D)", "m/s", "higher", lambda d: d["flutter_speed"]),
                    ("V_div3d", "Divergence (3D), EAS", "m/s", "higher", lambda d: d["divergence"]["V_eas"]),
                    ("f1", "First mode", "Hz", "higher", lambda d: d["frequencies_hz"][0])],
    "performance": [("V_stall", "Stall speed", "m/s", "lower", lambda d: d["V_stall"]),
                    ("glide", "Best glide ratio", "", "higher", lambda d: d["glide"]["best_ratio"]),
                    ("min_sink", "Minimum sink", "m/s", "lower", lambda d: d["glide"]["min_sink"]),
                    ("V_max", "Maximum speed", "m/s", "higher", lambda d: d["V_max"]),
                    ("roc", "Best rate of climb", "m/s", "higher", lambda d: d["best_rate"]["roc"]),
                    ("ceiling", "Service ceiling", "m", "higher", lambda d: d["ceiling_service"]),
                    ("range", "Range", "km", "higher", lambda d: d["best_range"]["range_km"]),
                    ("takeoff", "Take-off to 15 m", "m", "lower", lambda d: d["takeoff"]["total"]),
                    ("landing", "Landing from 15 m", "m", "lower", lambda d: d["landing"]["total"])],
    "polar": [("clmax", "Cl max", "", "higher", lambda d: _polar_stats(d)["clmax"]),
              ("ldmax", "L/D max", "", "higher", lambda d: _polar_stats(d)["ldmax"]),
              ("cdmin", "Cd min", "", "lower", lambda d: _polar_stats(d)["cdmin"])],
}


def _wing(d):
    return d["surfaces"][0]


def _wing_area(s):
    c = [s["root_chord"]] + [k["chord"] for k in s.get("kinks") or []] + [s["tip_chord"]]
    y = [0.0] + [k["y"] for k in s.get("kinks") or []] + [s["span"] / 2]
    a = sum((y[i + 1] - y[i]) * (c[i] + c[i + 1]) / 2 for i in range(len(c) - 1))
    return 2 * a if s.get("symmetric", True) else a


def _polar_stats(d):
    import math
    ok = [r for r in d["results"] if r.get("Cl") is not None and math.isfinite(r["Cl"]) and r.get("Cd")]
    return {"clmax": max(r["Cl"] for r in ok), "ldmax": max(r["Cl"] / r["Cd"] for r in ok),
            "cdmin": min(r["Cd"] for r in ok)}
