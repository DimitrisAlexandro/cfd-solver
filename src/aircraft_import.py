"""Import an aircraft from a file: the app's own JSON, or an AVL geometry file
(.avl) with an optional AVL mass file (.mass) and airfoil coordinate files.

The aircraft model uses one straight-tapered panel per surface. An AVL surface
with more than two sections is fitted with one trapezoid through its root and
tip sections; the fit is reported in the warnings with the area error.

    import aircraft_import as AI
    ac, warnings = AI.read_avl(text, mass_text=None, airfoils={"clarky.dat": "/path/clarky.dat"})
"""
import json
import math
import re
from pathlib import Path

import numpy as np

import aircraft as A

CONTROL_NAMES = (("elev", "elevator"), ("pitch", "elevator"), ("ail", "aileron"), ("roll", "aileron"),
                 ("rud", "rudder"), ("yaw", "rudder"), ("flap", "flap"))


def read_json(text):
    d = json.loads(text)
    ac = A.Aircraft.from_dict(d)
    if not ac.surfaces:
        raise ValueError("no surfaces in the file")
    return ac, []


# ----------------------------------------------------------------------
# AVL
# ----------------------------------------------------------------------
def _clean_lines(text):
    out = []
    for ln in text.splitlines():
        ln = re.split(r"[#!]", ln, maxsplit=1)[0].strip()
        if ln:
            out.append(ln)
    return out


def _nums(line, n=None):
    v = []
    for t in line.replace(",", " ").split():
        try:
            v.append(float(t))
        except ValueError:
            break
    return v if n is None else (v + [0.0] * n)[:n]


def _kw(line):
    """AVL keywords are recognised by their first four characters"""
    return line.split()[0][:4].upper() if line.split() else ""


def _parse_avl(text):
    L = _clean_lines(text)
    if len(L) < 5:
        raise ValueError("too short for an AVL geometry file")
    head = {"name": L[0]}
    i = 1
    head["mach"] = _nums(L[i], 1)[0]; i += 1
    head["sym"] = _nums(L[i], 3); i += 1
    head["ref"] = _nums(L[i], 3); i += 1
    head["xyzref"] = _nums(L[i], 3); i += 1
    head["cdp"] = 0.0
    if i < len(L) and _nums(L[i]) and _kw(L[i]) not in ("SURF", "BODY"):
        head["cdp"] = _nums(L[i], 1)[0]; i += 1
    surfaces, bodies = [], []
    cur = sec = None
    while i < len(L):
        k = _kw(L[i])
        if k == "SURF":
            cur = {"name": L[i + 1], "ydup": None, "scale": [1, 1, 1], "translate": [0, 0, 0],
                   "angle": 0.0, "sections": [], "nchord": None, "nspan": None}
            surfaces.append(cur)
            sec = None
            i += 2
            if i < len(L) and _nums(L[i]):          # Nchord Cspace [Nspan Sspace]
                v = _nums(L[i])
                cur["nchord"] = int(v[0])
                if len(v) >= 3:
                    cur["nspan"] = int(v[2])
                i += 1
            continue
        if k == "BODY":
            bodies.append(L[i + 1])
            cur = sec = None
            i += 2
            continue
        if cur is None:
            i += 1
            continue
        if k == "YDUP":
            cur["ydup"] = _nums(L[i + 1], 1)[0]; i += 2
        elif k == "SCAL":
            cur["scale"] = _nums(L[i + 1], 3); i += 2
        elif k == "TRAN":
            cur["translate"] = _nums(L[i + 1], 3); i += 2
        elif k == "ANGL":
            cur["angle"] = _nums(L[i + 1], 1)[0]; i += 2
        elif k == "SECT":
            v = _nums(L[i + 1], 5)
            sec = {"xle": v[0], "yle": v[1], "zle": v[2], "chord": v[3], "ainc": v[4],
                   "airfoil": "naca0012", "af_source": "default", "controls": []}
            cur["sections"].append(sec)
            i += 2
        elif k == "NACA" and sec is not None:
            code = L[i + 1].split()[0] if i + 1 < len(L) else "0012"
            sec["airfoil"], sec["af_source"] = f"naca{code}", "NACA"
            i += 2
        elif k == "AFIL" and sec is not None:
            parts = L[i].split(maxsplit=1)
            if len(parts) > 1:
                fn = parts[1].strip(); i += 1
            else:
                fn = L[i + 1].strip(); i += 2
            sec["airfoil"], sec["af_source"] = fn.strip('"\''), "AFILE"
        elif k == "AIRF" and sec is not None:      # inline coordinates
            j = i + 1
            pts = []
            while j < len(L) and len(_nums(L[j])) >= 2 and not L[j][0].isalpha():
                pts.append(_nums(L[j], 2)); j += 1
            sec["airfoil"], sec["af_source"], sec["coords"] = f"{cur['name']}_inline", "AIRFOIL", pts
            i = j
        elif k == "CONT" and sec is not None:
            t = L[i + 1].split()
            v = _nums(" ".join(t[1:]), 6)
            sec["controls"].append({"name": t[0], "gain": v[0], "xhinge": v[1], "sgndup": v[5] or 1.0})
            i += 2
        elif k in ("COMP", "INDE", "CLAF", "CDCL", "DESI", "BFIL"):
            i += 2
        else:
            i += 1
    return head, surfaces, bodies


def _canonical_control(name):
    n = name.lower()
    for key, canon in CONTROL_NAMES:
        if key in n:
            return canon
    return None


def _surface_from_avl(s, airfoils, warn, inline_dir):
    sc, tr = np.array(s["scale"], float), np.array(s["translate"], float)
    secs = []
    for q in s["sections"]:
        p = np.array([q["xle"], q["yle"], q["zle"]]) * sc + tr
        secs.append(dict(q, p=p, c=q["chord"] * sc[0]))
    if len(secs) < 2:
        raise ValueError(f"surface '{s['name']}' has fewer than two sections")
    ys = np.array([q["p"][1] for q in secs])
    zs = np.array([q["p"][2] for q in secs])
    vertical = np.ptp(ys) < 0.1 * max(np.ptp(zs), 1e-9)
    if vertical:
        span_c = zs
    else:
        span_c = ys
        # a surface given from tip to tip without YDUPLICATE: keep the +y half
        if s["ydup"] is None and ys.min() < -1e-6 and ys.max() > 1e-6:
            s["ydup"] = 0.0
            secs = [q for q in secs if q["p"][1] >= -1e-9]
            span_c = np.array([q["p"][1] for q in secs])
            warn(f"'{s['name']}' is given from tip to tip; the +y half is used, mirrored about y = 0.")
    order = np.argsort(span_c)
    secs = [secs[k] for k in order]
    span_c = span_c[order]
    r, t = secs[0], secs[-1]
    symmetric = (s["ydup"] is not None) and not vertical
    if vertical:
        semi = t["p"][2] - r["p"][2]
        y0, z0 = r["p"][1], r["p"][2]
        dihedral = 0.0
    else:
        y0 = s["ydup"] if symmetric else r["p"][1]
        semi = t["p"][1] - y0
        z0 = r["p"][2]
        if symmetric and r["p"][1] - y0 > 1e-6:
            warn(f"'{s['name']}': the root section starts at y = {r['p'][1]:.3f} m; the model extends "
                 f"the root chord to the plane of symmetry (y = {y0:g}).")
        dihedral = math.degrees(math.atan2(t["p"][2] - r["p"][2], t["p"][1] - r["p"][1]))
    if semi <= 0:
        raise ValueError(f"surface '{s['name']}' has no span")
    xq_r = r["p"][0] + 0.25 * r["c"]
    xq_t = t["p"][0] + 0.25 * t["c"]
    # root chord on the plane of symmetry / root station: same chord, LE moved along the sweep line
    sweep = math.degrees(math.atan2(xq_t - xq_r, semi if vertical else (t["p"][1] - r["p"][1])))
    x0 = xq_r - (r["p"][1] - y0 if not vertical else 0.0) * math.tan(math.radians(sweep)) - 0.25 * r["c"]
    if not vertical:
        z0 = r["p"][2] - (r["p"][1] - y0) * math.tan(math.radians(dihedral))
    span_len = semi * (2 if symmetric else 1)

    def af_of(q):
        if q["af_source"] == "AFILE":
            fn = Path(q["airfoil"]).name
            if fn in airfoils:
                return str(airfoils[fn])
            warn(f"'{s['name']}': airfoil file {fn} not uploaded; NACA 0012 used instead. Upload it "
                 "with the .avl file.")
            return "naca0012"
        if q["af_source"] == "AIRFOIL":
            p = Path(inline_dir) / f"{re.sub(r'[^A-Za-z0-9_-]+', '_', q['airfoil'])}.dat"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(q["airfoil"] + "\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in q["coords"]) + "\n")
            return str(p)
        if q["af_source"] == "default":
            warn(f"'{s['name']}': a section has no airfoil; NACA 0012 used (AVL's default is a flat plate).")
        return q["airfoil"]
    af_r, af_t = af_of(r), af_of(t)

    # controls: span range from the sections that carry them
    ctrls = []
    names = {c["name"] for q in secs for c in q["controls"]}
    for nm in sorted(names):
        canon = _canonical_control(nm)
        ks = [k for k, q in enumerate(secs) if any(c["name"] == nm for c in q["controls"])]
        if canon is None:
            warn(f"'{s['name']}': control '{nm}' not recognised (elevator, aileron, rudder, flap); skipped.")
            continue
        if len(ks) < 2:
            warn(f"'{s['name']}': control '{nm}' is on one section only; skipped.")
            continue
        cs = [c for k in ks for c in secs[k]["controls"] if c["name"] == nm]
        xh = float(np.mean([abs(c["xhinge"]) for c in cs]))
        eta0 = (span_c[ks[0]] - span_c[0]) / (span_c[-1] - span_c[0])
        eta1 = (span_c[ks[-1]] - span_c[0]) / (span_c[-1] - span_c[0])
        if symmetric and span_c[0] - y0 > 1e-9:          # eta measured from the plane of symmetry
            eta0 = (span_c[ks[0]] - y0) / semi
            eta1 = (span_c[ks[-1]] - y0) / semi
        mode = "antisymmetric" if cs[0]["sgndup"] < 0 else "symmetric"
        if canon == "aileron" and not symmetric:
            mode = "symmetric"
        ctrls.append(A.Control(canon, round(float(eta0), 4), round(float(eta1), 4),
                               round(min(max(1 - xh, 0.05), 0.6), 4), mode))
    nc = int(min(max(s["nchord"] or 6, 2), 12))
    ns = int(min(max(s["nspan"] or 12, 4), 30))
    surf = A.Surface(s["name"], float(span_len), float(r["c"]), float(t["c"]), float(x0),
                     float(y0), float(z0), round(sweep, 6), round(dihedral, 6),
                     round(float(t["ainc"] - r["ainc"]), 4), round(float(r["ainc"] + s["angle"]), 4),
                     af_r, af_t if af_t != af_r else None, bool(symmetric), bool(vertical), ctrls, ns, nc)
    # intermediate sections become kinks, except those on the straight line between their
    # neighbours (e.g. sections AVL needs only at control-surface edges)
    pts = [(0.0, float(r["c"]), 0.0, 0.0, 0.0)]
    for q in secs[1:-1]:
        y = float((q["p"][2] - z0) if vertical else (q["p"][1] - y0))
        if 1e-6 < y < semi * (1 - 1e-6):
            pts.append((y, float(q["c"]), float(q["p"][0] - x0),
                        0.0 if vertical else float(q["p"][2] - z0), float(q["ainc"] - r["ainc"])))
    tip = A.Surface(**{**A.asdict(surf), "controls": []})
    pts.append((semi, float(t["c"]), float(tip.x_le(1.0) - x0), float(tip.rise(1.0)), float(t["ainc"] - r["ainc"])))
    pts = sorted(pts)
    tol = np.array([1.0, 0.002 * r["c"], 0.002 * r["c"], 0.002 * r["c"], 0.02])
    changed = True
    while changed and len(pts) > 2:
        changed = False
        for k in range(1, len(pts) - 1):
            a, m, b = np.array(pts[k - 1]), np.array(pts[k]), np.array(pts[k + 1])
            w = (m[0] - a[0]) / max(b[0] - a[0], 1e-12)
            if np.all(np.abs(m[1:] - (a[1:] + w * (b[1:] - a[1:]))) <= tol[1:]):
                del pts[k]
                changed = True
                break
    kinks = [{"y": round(p[0], 5), "chord": round(p[1], 5), "dx": round(p[2], 5), "dz": round(p[3], 5),
              "twist": round(p[4], 4)} for p in pts[1:-1]]
    surf.kinks = kinks
    if kinks:
        warn(f"'{s['name']}': {len(kinks)} kink station(s) kept (multi-panel surface; see the Panel "
             "breaks table on the Edit tab).")
    return surf


def read_mass(text):
    """AVL mass file -> mass, c.g., (Ixx, Iyy, Izz) about the c.g. (SI units)"""
    lunit = munit = 1.0
    mul, add = np.ones(10), np.zeros(10)
    rows = []
    for raw in text.splitlines():
        ln = re.split(r"[#!]", raw, maxsplit=1)[0].strip()
        if not ln:
            continue
        m = re.match(r"(Lunit|Munit)\s*=\s*([-+0-9.eE]+)\s*(\S*)", ln, re.I)
        if m:
            v = float(m.group(2))
            u = m.group(3).lower()
            conv = {"m": 1, "dm": 0.1, "cm": 0.01, "mm": 1e-3, "in": 0.0254, "ft": 0.3048,
                    "kg": 1, "g": 1e-3, "lb": 0.45359237, "slug": 14.5939}.get(u, 1.0)
            if m.group(1).lower() == "lunit":
                lunit = v * conv
            else:
                munit = v * conv
            continue
        if "=" in ln:
            continue
        if ln.startswith("*"):
            v = _nums(ln[1:]); mul[:len(v)] = v; continue
        if ln.startswith("+"):
            v = _nums(ln[1:]); add[:len(v)] = v; continue
        v = _nums(ln)
        if len(v) >= 4:
            v = np.array((v + [0.0] * 10)[:10]) * mul + add
            rows.append(v)
    if not rows:
        raise ValueError("no mass lines in the mass file")
    R = np.array(rows)
    m = R[:, 0] * munit
    xyz = R[:, 1:4] * lunit
    I = R[:, 4:7] * munit * lunit**2
    M = float(m.sum())
    cg = (m[:, None] * xyz).sum(0) / M
    d = xyz - cg
    Ixx = float((I[:, 0] + m * (d[:, 1]**2 + d[:, 2]**2)).sum())
    Iyy = float((I[:, 1] + m * (d[:, 0]**2 + d[:, 2]**2)).sum())
    Izz = float((I[:, 2] + m * (d[:, 0]**2 + d[:, 1]**2)).sum())
    return M, tuple(float(v) for v in cg), (Ixx, Iyy, Izz)


def read_avl(text, mass_text=None, airfoils=None, inline_dir=None):
    """Aircraft from an AVL geometry file; returns (aircraft, warnings)"""
    warnings = []
    warn = warnings.append
    airfoils = {Path(k).name: v for k, v in (airfoils or {}).items()}
    head, surfs, bodies = _parse_avl(text)
    if not surfs:
        raise ValueError("no SURFACE in the file")
    inline_dir = inline_dir or Path.cwd()
    out = [_surface_from_avl(s, airfoils, warn, inline_dir) for s in surfs]
    # main wing first: the largest horizontal surface
    horiz = [k for k, s in enumerate(out) if not s.vertical]
    if horiz:
        kw = max(horiz, key=lambda k: out[k].area)
        out.insert(0, out.pop(kw))
    # rename duplicated surface names (controls are matched by surface)
    seen = {}
    for s in out:
        if s.name in seen:
            seen[s.name] += 1
            s.name = f"{s.name} {seen[s.name]}"
        else:
            seen[s.name] = 1
    ac = A.Aircraft(head["name"] or "Imported aircraft", out, A.Fuselage(),
                    cg=tuple(float(v) for v in head["xyzref"]), cd_extra=float(head["cdp"]))
    S, c, b = head["ref"]
    w = ac.wing
    for nm, given, own, attr in (("Sref", S, w.area, "S_ref"), ("Cref", c, w.mac, "c_ref"),
                                 ("Bref", b, w.span, "b_ref")):
        if given > 0 and abs(given / own - 1) > 0.02:
            setattr(ac, attr, float(given))
            warn(f"{nm} = {given:g} in the file differs from the wing's own value ({own:.4g}); the "
                 "file's value is kept as the reference.")
    if bodies:
        warn("AVL BODY entries (" + ", ".join(bodies) + ") are not read; enter the fuselage length and "
             "diameter on the Edit tab.")
    else:
        warn("No fuselage in an AVL file: enter its length and diameter on the Edit tab (it adds the "
             "fuselage's pitch and yaw moments).")
    if mass_text:
        M, cg, I = read_mass(mass_text)
        ac.mass, ac.cg, ac.inertia = M, cg, I
    else:
        warn(f"No mass file: mass left at {ac.mass:g} kg and inertias estimated; the c.g. is the file's "
             "moment reference point (Xref, Zref).")
    for s in ac.surfaces:
        for c in s.controls:
            dup = [d for d in s.controls if d.name == c.name]
            if len(dup) > 1:
                warn(f"'{s.name}' has several controls mapped to '{c.name}'; only the first is used.")
                s.controls = [d for d in s.controls if d.name != c.name or d is dup[0]]
                break
    return ac, warnings


def read_files(files, inline_dir=None):
    """files: {file name: bytes}. Returns (aircraft, warnings, airfoil files used)."""
    by_ext = {}
    for n, b in files.items():
        by_ext.setdefault(Path(n).suffix.lower(), []).append((n, b))
    if ".json" in by_ext:
        n, b = by_ext[".json"][0]
        ac, w = read_json(b.decode("utf-8", "replace"))
        return ac, w
    if ".avl" not in by_ext:
        raise ValueError("upload an aircraft .json (from this app) or an AVL .avl geometry file")
    n, b = by_ext[".avl"][0]
    mass = by_ext.get(".mass", [(None, None)])[0][1]
    inline_dir = Path(inline_dir or Path.cwd())
    afs = {}
    for ext in (".dat", ".txt"):
        for fn, fb in by_ext.get(ext, []):
            p = inline_dir / Path(fn).name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(fb)
            afs[Path(fn).name] = str(p)
    return read_avl(b.decode("utf-8", "replace"), mass.decode("utf-8", "replace") if mass else None,
                    afs, inline_dir)
