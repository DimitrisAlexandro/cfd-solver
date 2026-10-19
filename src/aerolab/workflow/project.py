"""Projects: a design as a dependency graph, like a build system.

A project holds one or more *variants* of a design. Each variant is a graph of analysis
nodes (aircraft → stability, drag, weight and balance, loads → structure → aeroelasticity,
performance, section polars, ...). Every node has parameters and inputs (other nodes).

- Fingerprints: a node's fingerprint is a hash of its kind, parameters, external files
  and the fingerprints of its inputs. Change anything upstream and everything downstream
  gets a new fingerprint: it is *stale*, and the reason is shown ("loads changed",
  "mass 750 → 800").
- Recompute only what is needed: `run` executes the stale nodes in dependency order.
- Content-addressed results: outputs live in runs/<kind>-<fingerprint>/. Two variants that
  share a node with the same inputs share its result, so a new variant costs nothing
  until something actually differs.
- Variant comparison: key results of every node side by side, with the change against
  the baseline and the trust level of each result.

    python src/run.py project new "Trainer study" --template aircraft
    python src/run.py project status ../cases/projects/trainer-study
    python src/run.py project variant ../cases/projects/trainer-study heavy --set aircraft.mass=820
    python src/run.py project run ../cases/projects/trainer-study --variant heavy
    python src/run.py project compare ../cases/projects/trainer-study
"""
import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import aerolab

SRC = Path(__file__).resolve().parents[2]
ROOT = SRC.parent
CASES = Path(os.environ.get("AEROLAB_CASES", ROOT / "cases"))
PROJECTS = CASES / "projects"
POLAR_DIR = CASES / "wing_polars"


def ensure_polar_store():
    """starter section polars (as the app does at start-up)"""
    if not POLAR_DIR.exists() or not any(POLAR_DIR.glob("*.json")):
        import shutil
        POLAR_DIR.mkdir(parents=True, exist_ok=True)
        for f in (ROOT / "data" / "wing_polars").glob("*.json"):
            shutil.copy(f, POLAR_DIR / f.name)


def check_environment():
    """fail early, with the fix, when the active Python lacks the solver's packages"""
    import importlib.util
    missing = [m for m in ("numpy", "scipy", "pandas", "matplotlib", "numba", "pyamg")
               if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit(f"[project] this Python ({sys.executable}) is missing {', '.join(missing)}.\n"
                         f"  Activate the environment you run AeroLab with, or install them:\n"
                         f"  pip install -r {ROOT / 'requirements.txt'}")


def polar_cache_signature():
    """the section-polar library the aircraft tools read: adding or recomputing a polar
    changes every aircraft analysis, so it is part of the aircraft node's fingerprint"""
    if not POLAR_DIR.exists():
        return ""
    return _h(sorted((f.name, f.stat().st_size) for f in POLAR_DIR.glob("*.json")))[:16]


def _h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    p = Path(path)
    if not p.is_absolute():
        p = SRC / p
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "project"


# ----------------------------------------------------------------------
# node kinds
# ----------------------------------------------------------------------
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
         "--polar-dir", POLAR_DIR, "--out", out]
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
                                     "--polar-dir", POLAR_DIR, "--out", out / "stability.json"]},
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
                                     "--polar-dir", POLAR_DIR, "--out", out]},
    "loads": {
        "title": "Flight loads", "script": "loads.py", "inputs": ["aircraft"], "output": "loads.json",
        "params": {"vc": 60.0, "category": "utility", "alt": 0.0, "fuel": 0.0},
        "args": lambda p, inp, out: ["loads.py", _ac(inp), "--vc", p["vc"], "--category", p["category"],
                                     "--alt", p["alt"], "--fuel", p["fuel"], "--polar-dir", POLAR_DIR,
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


# ----------------------------------------------------------------------
# aircraft node (in process)
# ----------------------------------------------------------------------
def set_path(d, path, value):
    """'mass', 'cg.0', 'surfaces.wing.span' (surface by name or index), 'fuselage.length'"""
    keys = path.split(".")
    cur = d
    for i, k in enumerate(keys):
        last = i == len(keys) - 1
        if isinstance(cur, list):
            if k.lstrip("-").isdigit():
                k = int(k)
            else:
                idx = next((j for j, e in enumerate(cur) if isinstance(e, dict) and e.get("name") == k), None)
                if idx is None:
                    raise KeyError(f"no element named {k!r} in {path!r}")
                k = idx
        elif k not in cur and not last:
            raise KeyError(f"{k!r} not found in {path!r}")
        if last:
            cur[k] = value
        else:
            cur = cur[k]
    return d


def build_aircraft(params, out):
    from aerolab.aircraft import aircraft as A
    src = params.get("source") or "demo"
    if src == "demo":
        d = A.demo_aircraft().to_dict()
    else:
        p = Path(src) if Path(src).is_absolute() else SRC / src
        d = json.loads(p.read_text())
    d = json.loads(json.dumps(d, default=list))
    for k, v in (params.get("overrides") or {}).items():
        set_path(d, k, v)
    ac = A.Aircraft.from_dict(d)
    ac.save(Path(out) / "aircraft.json")


def structured_aircraft(inp, out, params=None):
    """after wing_structure.py: the aircraft with the sizing attached (input of aeroelasticity)"""
    from aerolab.aircraft import aircraft as A
    from aerolab.aircraft import wing_structure as WS
    ac = A.Aircraft.load(_ac(inp))
    s = json.loads((Path(out) / "structure.json").read_text())["structure"]
    prm = params or {}
    ail = dict((ac.structure or {}).get("aileron_dynamics") or
               {"freq_hz": 15.0, "mass_per_area": 4.0, "cg_aft_hinge": 0.30, "gyration": 0.30})
    for k_p, k_a in (("aileron_mass_per_area", "mass_per_area"), ("aileron_cg_aft_hinge", "cg_aft_hinge"),
                     ("aileron_freq_hz", "freq_hz")):
        if prm.get(k_p) is not None:
            ail[k_a] = float(prm[k_p])
    ac.structure = WS.to_aircraft(s, ail)
    ac.save(Path(out) / "aircraft_structured.json")


# ----------------------------------------------------------------------
TEMPLATES = {
    "aircraft": [("aircraft", "aircraft", {}), ("stability", "stability", {"aircraft": "aircraft"}),
                 ("drag", "drag", {"aircraft": "aircraft"}),
                 ("weight_balance", "weight_balance", {"aircraft": "aircraft"}),
                 ("loads", "loads", {"aircraft": "aircraft"}),
                 ("structure", "structure", {"aircraft": "aircraft", "loads": "loads"}),
                 ("aeroelastic", "aeroelastic", {"structure": "structure"}),
                 ("performance", "performance", {"aircraft": "aircraft"})],
    "section": [("polar", "polar", {})],
    "empty": [],
}


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return True


class Project:
    def __init__(self, path):
        self.dir = Path(path).resolve()        # absolute: runs are executed from src/
        self.data = json.loads((self.dir / "project.json").read_text())

    # -------------------------------------------------------------- create / save
    @classmethod
    def create(cls, name, template="aircraft", aircraft_source="demo", root=None, description=""):
        root = Path(root or PROJECTS)
        d = root / slugify(name)
        i = 2
        while d.exists():
            d = root / f"{slugify(name)}-{i}"
            i += 1
        d.mkdir(parents=True)
        nodes = {}
        for nid, kind, inputs in TEMPLATES[template]:
            params = copy.deepcopy(KINDS[kind]["params"])
            if kind == "aircraft":
                params["source"] = aircraft_source
            nodes[nid] = {"kind": kind, "label": KINDS[kind]["title"], "params": params, "inputs": inputs}
        data = {"name": name, "description": description, "created": time.time(), "baseline": "baseline",
                "variants": {"baseline": {"description": "starting design", "parent": None, "created": time.time(),
                                          "nodes": nodes}}}
        (d / "project.json").write_text(json.dumps(data, indent=1))
        return cls(d)

    @classmethod
    def load(cls, path):
        p = Path(path)
        if not (p / "project.json").exists() and (PROJECTS / str(path) / "project.json").exists():
            p = PROJECTS / str(path)
        return cls(p)

    @staticmethod
    def list(root=None):
        root = Path(root or PROJECTS)
        out = []
        for f in sorted(root.glob("*/project.json")) if root.exists() else []:
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            out.append({"name": d.get("name", f.parent.name), "path": str(f.parent),
                        "variants": len(d.get("variants", {})), "modified": f.stat().st_mtime})
        return sorted(out, key=lambda r: -r["modified"])

    def reload(self):
        self.data = json.loads((self.dir / "project.json").read_text())
        return self

    def save(self):
        tmp = self.dir / "project.json.tmp"
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.dir / "project.json")

    @property
    def name(self):
        return self.data["name"]

    @property
    def variants(self):
        return self.data["variants"]

    @property
    def baseline(self):
        return self.data.get("baseline") or next(iter(self.variants))

    def nodes(self, variant):
        return self.variants[variant]["nodes"]

    # -------------------------------------------------------------- editing
    def add_node(self, variant, nid, kind, params=None, inputs=None, label=None):
        if kind not in KINDS:
            raise ValueError(f"unknown node kind {kind!r}")
        nodes = self.nodes(variant)
        if nid in nodes:
            raise ValueError(f"node {nid!r} exists")
        need = KINDS[kind]["inputs"]
        inputs = dict(inputs or {})
        for role in need:                       # default: the single node of that kind
            if role not in inputs:
                cand = [k for k, n in nodes.items() if n["kind"] == role]
                if len(cand) != 1:
                    raise ValueError(f"{kind} needs input {role!r}")
                inputs[role] = cand[0]
        for role, src in inputs.items():
            if src not in nodes:
                raise ValueError(f"input {src!r} not in variant {variant!r}")
        nodes[nid] = {"kind": kind, "label": label or KINDS[kind]["title"],
                      "params": {**copy.deepcopy(KINDS[kind]["params"]), **(params or {})}, "inputs": inputs}
        self._check_acyclic(variant)
        self.save()

    def remove_node(self, variant, nid):
        nodes = self.nodes(variant)
        users = [k for k, n in nodes.items() if nid in n["inputs"].values()]
        if users:
            raise ValueError(f"{nid!r} is used by {', '.join(users)}")
        nodes.pop(nid)
        self.save()

    def set_params(self, variant, nid, **params):
        self.nodes(variant)[nid]["params"].update(params)
        self.save()

    def set_override(self, variant, nid, path, value):
        """aircraft node: change one field of the model ('mass', 'surfaces.wing.span')"""
        p = self.nodes(variant)[nid]["params"]
        p.setdefault("overrides", {})
        if value is None:
            p["overrides"].pop(path, None)
        else:
            p["overrides"][path] = value
        self.save()

    def create_variant(self, name, base=None, description="", changes=None):
        """copy a variant; changes: {"node.param": value} or {"aircraft.<path>": value}
        for aircraft-model fields (applied as overrides)"""
        base = base or self.baseline
        if name in self.variants:
            raise ValueError(f"variant {name!r} exists")
        self.variants[name] = {"description": description, "parent": base, "created": time.time(),
                               "nodes": copy.deepcopy(self.nodes(base))}
        for key, v in (changes or {}).items():
            self.apply_change(name, key, v, save=False)
        self.save()

    def apply_change(self, variant, key, value, save=True):
        nid, _, rest = key.partition(".")
        node = self.nodes(variant)[nid]
        if node["kind"] == "aircraft" and rest not in ("source", "overrides"):
            node["params"].setdefault("overrides", {})[rest] = value
        else:
            node["params"][rest] = value
        if save:
            self.save()

    def delete_variant(self, name):
        if name == self.baseline:
            raise ValueError("the baseline can't be deleted")
        self.variants.pop(name)
        rf = self._records_file(name)
        if rf.exists():
            rf.unlink()
        self.save()

    def set_baseline(self, name):
        self.data["baseline"] = name
        self.save()

    def _check_acyclic(self, variant):
        self.order(variant)

    # -------------------------------------------------------------- graph
    def order(self, variant, targets=None):
        """topological order (inputs first) of `targets` and their ancestors (all if None)"""
        nodes = self.nodes(variant)
        seen, out, stack = set(), [], set()

        def visit(n):
            if n in seen:
                return
            if n in stack:
                raise ValueError(f"dependency cycle through {n!r}")
            stack.add(n)
            for src in nodes[n]["inputs"].values():
                visit(src)
            stack.discard(n)
            seen.add(n)
            out.append(n)
        for n in (targets or list(nodes)):
            visit(n)
        return out

    def downstream(self, variant, nid):
        nodes = self.nodes(variant)
        out, todo = [], [nid]
        while todo:
            cur = todo.pop()
            for k, n in nodes.items():
                if cur in n["inputs"].values() and k not in out:
                    out.append(k)
                    todo.append(k)
        return out

    def fingerprint(self, variant, nid, _memo=None):
        memo = {} if _memo is None else _memo
        if nid in memo:
            return memo[nid]
        n = self.nodes(variant)[nid]
        kind = KINDS[n["kind"]]
        files = {k: file_hash(n["params"].get(k)) for k in kind.get("files", [])
                 if n["params"].get(k) not in (None, "", "demo") and not str(n["params"].get(k)).lower()
                 .startswith(("naca", "tmr"))}
        if n["kind"] == "aircraft":
            files["section_polars"] = polar_cache_signature()
        fp = _h({"kind": n["kind"], "params": n["params"], "files": files,
                 "inputs": {r: self.fingerprint(variant, s, memo) for r, s in sorted(n["inputs"].items())}})
        memo[nid] = fp
        return fp

    def run_dir(self, kind, fp):
        return self.dir / "runs" / f"{kind}-{fp[:16]}"

    def _records_file(self, variant):
        return self.dir / "variants" / f"{slugify(variant)}.records.json"

    def records(self, variant):
        f = self._records_file(variant)
        return json.loads(f.read_text()) if f.exists() else {}

    def _save_record(self, variant, nid, rec):
        f = self._records_file(variant)
        f.parent.mkdir(parents=True, exist_ok=True)
        r = self.records(variant)
        r[nid] = rec
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, indent=1))
        tmp.replace(f)

    @staticmethod
    def code_hash(kind):
        return file_hash(aerolab.script_path(KINDS[kind]["script"]))

    def _meta(self, kind, fp):
        f = self.run_dir(kind, fp) / "meta.json"
        if not f.exists():
            return None
        m = json.loads(f.read_text())
        if m.get("state") == "running" and not _alive(m.get("pid", -1)):
            m["state"] = "failed"
            m["error"] = "interrupted"
        return m

    def status(self, variant):
        """{node: {state, reasons, fp, dir, code_changed, last_dir}} in graph order.
        states: 'up to date', 'stale', 'not run', 'running', 'failed'"""
        nodes = self.nodes(variant)
        recs = dict(self.records(variant))
        par, seen = self.variants[variant].get("parent"), {variant}
        while par and par in self.variants and par not in seen:   # a new variant: changes vs its parent
            seen.add(par)
            for k, r in self.records(par).items():
                recs.setdefault(k, r)
            par = self.variants[par].get("parent")
        memo, out = {}, {}
        for nid in self.order(variant):
            n = nodes[nid]
            fp = self.fingerprint(variant, nid, memo)
            meta = self._meta(n["kind"], fp)
            rec = recs.get(nid)
            reasons = []
            last_dir = None
            if meta and meta["state"] == "ok":
                state = "up to date"
                if rec is None or rec.get("fp") != fp:
                    other = [v for v in self.variants if v != variant and self.records(v).get(nid, {}).get("fp") == fp]
                    if other:
                        reasons.append(f"result shared with variant {other[0]}")
            elif meta and meta["state"] == "running":
                state = "running"
            elif meta and meta["state"] == "failed":
                state = "failed"
                reasons.append(meta.get("error") or "the run failed (see the log)")
            else:
                if rec and rec.get("fp"):
                    state = "stale"
                    old = rec.get("params") or {}
                    for k in sorted(set(old) | set(n["params"])):
                        if old.get(k) != n["params"].get(k):
                            if k == "overrides":
                                o, c = old.get(k) or {}, n["params"].get(k) or {}
                                for kk in sorted(set(o) | set(c)):
                                    if o.get(kk) != c.get(kk):
                                        reasons.append(f"{kk}: {_fmt(o.get(kk))} → {_fmt(c.get(kk))}")
                            else:
                                reasons.append(f"{k}: {_fmt(old.get(k))} → {_fmt(n['params'].get(k))}")
                    for role, src in n["inputs"].items():
                        if rec.get("inputs", {}).get(role) != memo.get(src):
                            reasons.append(f"input changed: {nodes[src]['label']}")
                    if not reasons and n["kind"] == "aircraft":
                        if "cache" not in rec:
                            reasons.append("the section polar library is now tracked: one-time recompute")
                        elif rec["cache"] != polar_cache_signature():
                            reasons.append("section polar library changed (new or recomputed CFD polars)")
                    if not reasons:
                        reasons.append("an input file changed")
                    od = self.run_dir(n["kind"], rec["fp"])
                    if (od / KINDS[n["kind"]]["output"]).exists():
                        last_dir = od
                else:
                    state = "not run"
            code_changed = bool(meta and meta.get("state") == "ok" and meta.get("code")
                                and meta["code"] != self.code_hash(n["kind"]))
            out[nid] = {"state": state, "reasons": reasons, "fp": fp, "dir": self.run_dir(n["kind"], fp),
                        "kind": n["kind"], "label": n["label"], "code_changed": code_changed,
                        "last_dir": last_dir, "elapsed": (meta or {}).get("elapsed")}
        return out

    def plan(self, variant, targets=None, force=False):
        st = self.status(variant)
        return [n for n in self.order(variant, targets)
                if st[n]["state"] != "running" and (force or st[n]["state"] != "up to date")]

    # -------------------------------------------------------------- execution
    def run(self, variant, targets=None, force=False, log=print):
        """run every node that isn't up to date (and the ancestors of `targets`), in order.
        Returns {node: state}."""
        ensure_polar_store()
        check_environment()
        plan = self.plan(variant, targets, force)
        memo, result = {}, {}
        nodes = self.nodes(variant)
        if not plan:
            log("[project] everything is up to date")
        for nid in plan:
            n = nodes[nid]
            kind = KINDS[n["kind"]]
            st_now = self.status(variant) if any(src not in result for src in n["inputs"].values()) else {}
            if any(result.get(src) in ("failed", "skipped") or
                   (src not in result and st_now.get(src, {}).get("state") != "up to date")
                   for src in n["inputs"].values()):
                result[nid] = "skipped"
                log(f"[project] {nid}: skipped (an input failed or is missing)")
                continue
            fp = self.fingerprint(variant, nid, memo)
            out = self.run_dir(n["kind"], fp)
            out.mkdir(parents=True, exist_ok=True)
            inp = {role: self.run_dir(nodes[src]["kind"], self.fingerprint(variant, src, memo))
                   for role, src in n["inputs"].items()}
            meta = {"state": "running", "fp": fp, "kind": n["kind"], "params": n["params"], "pid": os.getpid(),
                    "started": time.time(), "code": self.code_hash(n["kind"]), "variant": variant, "node": nid}
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
            log(f"[project] {nid}: running {kind['title']}", )
            t0 = time.time()
            err = None
            try:
                if n["kind"] == "aircraft":
                    build_aircraft(n["params"], out)
                else:
                    args = [str(a) for a in kind["args"](n["params"], inp, out)]
                    with open(out / "run.log", "w") as f:
                        rc = subprocess.run([sys.executable, "-u"] + aerolab.script_argv(args), cwd=SRC, stdout=f, stderr=subprocess.STDOUT,
                                            env={**os.environ, "MPLBACKEND": "Agg", "PYTHONWARNINGS": "ignore"}).returncode
                    if rc != 0:
                        err = f"exit code {rc}: " + _tail(out / "run.log")
                    elif n["kind"] == "structure":
                        structured_aircraft(inp, out, n["params"])
                if err is None and not (out / kind["output"]).exists():
                    err = f"no {kind['output']} written: " + _tail(out / "run.log")
            except Exception as ex:                          # noqa: BLE001
                err = f"{type(ex).__name__}: {ex}"
            meta.update(state="failed" if err else "ok", error=err, finished=time.time(),
                        elapsed=round(time.time() - t0, 1))
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
            self._save_record(variant, nid, {"fp": fp, "params": n["params"], "time": time.time(),
                                             "cache": polar_cache_signature() if n["kind"] == "aircraft" else None,
                                             "inputs": {r: self.fingerprint(variant, s, memo)
                                                        for r, s in n["inputs"].items()}})
            result[nid] = "failed" if err else "ok"
            log(f"[project] {nid}: {'FAILED ' + err if err else 'ok'} ({meta['elapsed']} s)")
        return result

    # -------------------------------------------------------------- results
    def output(self, variant, nid, allow_stale=True, status=None):
        """(path of the node's main output, is_current) or (None, False)"""
        st = (status or self.status(variant))[nid]
        f = st["dir"] / KINDS[st["kind"]]["output"]
        if st["state"] == "up to date" and f.exists():
            return f, True
        if allow_stale and st.get("last_dir"):
            return Path(st["last_dir"]) / KINDS[st["kind"]]["output"], False
        return None, False

    def metrics(self, variant, status=None):
        """{node: {"current": bool, "trust": level, "values": {key: value}}}"""
        from aerolab.workflow import trust as T
        status = status or self.status(variant)
        out = {}
        for nid, st in status.items():
            f, cur = self.output(variant, nid, status=status)
            if f is None or not f.exists():
                continue
            d = json.loads(f.read_text())
            vals = {}
            for key, *_rest, get in METRICS.get(st["kind"], []):
                try:
                    vals[key] = float(get(d))
                except Exception:                       # noqa: BLE001 (missing in older outputs)
                    vals[key] = None
            if st["kind"] == "polar":
                lev = T.assess_polar(d)["level"]
            else:                                       # the aircraft model itself has no level
                lev = T.EVIDENCE[st["kind"]][0] if st["kind"] in T.EVIDENCE else None
            out[nid] = {"current": cur, "trust": lev, "values": vals}
        return out

    def compare(self, variants=None):
        """rows for a table: node, metric, unit, better, and one column per variant
        (+ the change against the first variant)"""
        variants = variants or list(self.variants)
        mets = {v: self.metrics(v) for v in variants}
        rows = []
        nodes = self.nodes(variants[0])
        for v in variants[1:]:
            for k, n in self.nodes(v).items():
                nodes.setdefault(k, n)
        for nid, n in nodes.items():
            for key, label, unit, better, _ in METRICS.get(n["kind"], []):
                r = {"node": n["label"], "node_id": nid, "metric": label, "key": key, "unit": unit,
                     "better": better, "values": {}, "current": {}, "trust": {}}
                for v in variants:
                    m = mets[v].get(nid)
                    r["values"][v] = m["values"].get(key) if m else None
                    r["current"][v] = m["current"] if m else None
                    r["trust"][v] = m["trust"] if m else None
                if any(x is not None for x in r["values"].values()):
                    b = r["values"][variants[0]]
                    r["change"] = {v: (None if b in (None, 0) or r["values"][v] is None
                                       else (r["values"][v] - b) / abs(b)) for v in variants}
                    rows.append(r)
        return rows

    def diff(self, a, b):
        """parameter differences between two variants: [(node, param, value_a, value_b)]"""
        out = []
        na, nb = self.nodes(a), self.nodes(b)
        for nid in sorted(set(na) | set(nb)):
            if nid not in na or nid not in nb:
                out.append((nid, "(node)", "present" if nid in na else "—", "present" if nid in nb else "—"))
                continue
            pa, pb = na[nid]["params"], nb[nid]["params"]
            for k in sorted(set(pa) | set(pb)):
                if k == "overrides":
                    oa, ob = pa.get(k) or {}, pb.get(k) or {}
                    out += [(nid, kk, oa.get(kk), ob.get(kk)) for kk in sorted(set(oa) | set(ob))
                            if oa.get(kk) != ob.get(kk)]
                elif pa.get(k) != pb.get(k):
                    out.append((nid, k, pa.get(k), pb.get(k)))
        return out

    def dot(self, variant, status=None):
        """Graphviz source of the graph, coloured by state"""
        st = status or self.status(variant)
        col = {"up to date": ("#E3F1E8", "#2B7A4B"), "stale": ("#FBEFD9", "#B26A00"),
               "not run": ("#EEF1F4", "#56606C"), "running": ("#DDE8F5", "#1D4E89"),
               "failed": ("#F8E1E4", "#9C3D54")}
        lines = ["digraph G {", 'rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.5;',
                 'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13, penwidth=1.4, '
                 'margin="0.15,0.06"];',
                 'edge [color="#9AA4AF", arrowsize=0.7];']
        for nid, s in st.items():
            fill, line = col.get(s["state"], col["not run"])
            lab = f"{s['label']}\\n{s['state']}" + (" · code updated" if s["code_changed"] else "")
            lines.append(f'"{nid}" [label="{lab}", fillcolor="{fill}", color="{line}"];')
        for nid, n in self.nodes(variant).items():
            for src in n["inputs"].values():
                lines.append(f'"{src}" -> "{nid}";')
        lines.append("}")
        return "\n".join(lines)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    return "—" if v is None else str(v)


def _tail(path, n=3):
    try:
        lines = [l for l in Path(path).read_text(errors="replace").splitlines() if l.strip()]
        return " | ".join(lines[-n:])[-400:]
    except OSError:
        return ""


def _parse_value(s):
    try:
        return json.loads(s)
    except Exception:                                    # noqa: BLE001
        return s


# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("name")
    n.add_argument("--template", default="aircraft", choices=list(TEMPLATES))
    n.add_argument("--aircraft", default="demo")
    n.add_argument("--root", default=None)
    s = sub.add_parser("status")
    s.add_argument("project")
    s.add_argument("--variant", default=None)
    v = sub.add_parser("variant")
    v.add_argument("project")
    v.add_argument("name")
    v.add_argument("--base", default=None)
    v.add_argument("--set", action="append", default=[], help="node.param=value, e.g. aircraft.mass=820")
    v.add_argument("--description", default="")
    r = sub.add_parser("run")
    r.add_argument("project")
    r.add_argument("--variant", default=None)
    r.add_argument("--nodes", default=None, help="comma-separated targets (with their inputs)")
    r.add_argument("--force", action="store_true")
    r.add_argument("--job-dir", default=None, help="write run.json here when done (app jobs)")
    c = sub.add_parser("compare")
    c.add_argument("project")
    c.add_argument("--variants", default=None)
    a = ap.parse_args(argv)

    if a.cmd == "new":
        p = Project.create(a.name, a.template, a.aircraft, a.root)
        print(f"[project] -> {p.dir}")
        return
    p = Project.load(a.project)
    if a.cmd == "status":
        for var in ([a.variant] if a.variant else list(p.variants)):
            print(f"{p.name} / {var}")
            for nid, st in p.status(var).items():
                print(f"  {nid:16s} {st['state']:11s} " + "; ".join(st["reasons"]))
    elif a.cmd == "variant":
        p.create_variant(a.name, a.base, a.description,
                         {k: _parse_value(val) for k, _, val in (x.partition("=") for x in a.set)})
        print(f"[project] variant {a.name}: " + ", ".join(f"{nid} {st['state']}" for nid, st in p.status(a.name).items()))
    elif a.cmd == "run":
        var = a.variant or p.baseline
        res = p.run(var, a.nodes.split(",") if a.nodes else None, a.force)
        if a.job_dir:
            Path(a.job_dir).mkdir(parents=True, exist_ok=True)
            (Path(a.job_dir) / "run.json").write_text(json.dumps({"variant": var, "result": res}))
        if any(v == "failed" for v in res.values()):
            sys.exit(1)
    elif a.cmd == "compare":
        vs = a.variants.split(",") if a.variants else None
        rows = p.compare(vs)
        vs = vs or list(p.variants)
        print(f"{'result':34s}" + "".join(f"{v:>16s}" for v in vs))
        for r in rows:
            cells = []
            for v in vs:
                x = r["values"][v]
                ch = r["change"].get(v)
                cells.append("—" if x is None else f"{x:.4g}" + (f" ({100 * ch:+.1f}%)" if ch and v != vs[0] else ""))
            print(f"{(r['node'] + ': ' + r['metric'])[:34]:34s}" + "".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    main()
