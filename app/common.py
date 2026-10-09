"""Shared helpers for the Streamlit app: paths, background jobs, readers, plots."""

import base64
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CASES = ROOT / "cases"
VAL = ROOT / "validation"
JOBS = CASES / "_jobs"
POLAR_CACHE = CASES / "wing_polars"
AIRCRAFT = CASES / "aircraft"
for p in (CASES, JOBS):
    p.mkdir(parents=True, exist_ok=True)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BUILTIN_AIRFOILS = ["naca0012", "naca2412", "naca4412", "naca23012", "naca0015", "naca6412",
                    "tmr0012", "../validation/reference/rae2822.dat"]
MULTI_PRESETS = {"30P30N (JAXA wind-tunnel model)": "30p30n_jaxa",
                 "30P30N (original coordinates)": "30p30n"}


def page_setup(title=None, icon=None):
    """style for this page (the entry point AeroLab.py sets the page config)"""
    import ui
    ui.apply()


# ----------------------------------------------------------------------
# background jobs (CFD runs take minutes to hours: separate processes)
# ----------------------------------------------------------------------
def start_job(kind, title, args, out_dir, meta=None):
    """run `python <args>` in src/, stdout+stderr -> log; returns job id"""
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    log = JOBS / f"{jid}.log"
    with open(log, "w") as f:
        p = subprocess.Popen([sys.executable, "-u"] + [str(a) for a in args], cwd=SRC,
                             stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
                             env={**os.environ, "PYTHONWARNINGS": "ignore", "MPLBACKEND": "Agg"})
    job = {"id": jid, "kind": kind, "title": title, "args": [str(a) for a in args],
           "out": str(out_dir), "pid": p.pid, "started": time.time(), "meta": meta or {}}
    (JOBS / f"{jid}.json").write_text(json.dumps(job, indent=1))
    return jid


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:                                              # zombie = finished
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return True


def job_status(job):
    if job.get("cancelled"):
        return "cancelled"
    if _alive(job["pid"]):
        return "running"
    done = {"polar": "polar.json", "wing": "wing.json", "drag_rise": "drag_rise.json",
            "optimize": "optimize.json", "inverse": "inverse.json", "sizing": "sizing.json",
            "aero_db": "database_done.json", "loads": "loads.json", "performance": "performance.json"}
    f = Path(job["out"]) / done.get(job["kind"], "_none_")
    return "finished" if f.exists() else "failed"


def list_jobs():
    jobs = []
    for f in sorted(JOBS.glob("*.json"), reverse=True):
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        j["status"] = job_status(j)
        jobs.append(j)
    return jobs


def get_job(jid):
    j = json.loads((JOBS / f"{jid}.json").read_text())
    j["status"] = job_status(j)
    return j


def cancel_job(jid):
    j = json.loads((JOBS / f"{jid}.json").read_text())
    try:
        os.killpg(j["pid"], signal.SIGTERM)
    except OSError:
        pass
    j["cancelled"] = True
    (JOBS / f"{jid}.json").write_text(json.dumps(j, indent=1))


def job_log(jid, tail=None):
    f = JOBS / f"{jid}.log"
    if not f.exists():
        return ""
    lines = [l for l in f.read_text(errors="replace").splitlines()
             if "Warning" not in l and not l.startswith("  return") and
             not l.strip().startswith(("rho_cur", "alpha = rho", "return np.", "sqnorm", "s = (x"))]
    return "\n".join(lines[-tail:] if tail else lines)


_POINT = re.compile(r"alpha\s+(-?[\d.]+): Cl\s+(\S+)\s+Cd\s+(\S+)\s+Cm\s+(\S+)\s+(.+?)\s+(\d+) it\s+"
                    r"([\d.]+) s")


def parse_points(log_text):
    """per-angle lines printed by polar._chain -> list of dicts (live progress)"""
    out = []
    for m in _POINT.finditer(log_text):
        a, cl, cd, cm, st, it, t = m.groups()
        f = lambda v: float(v) if v not in ("nan",) else float("nan")
        out.append({"alpha": float(a), "Cl": f(cl), "Cd": f(cd), "Cm": f(cm),
                    "status": st.strip(), "iterations": int(it), "time_s": float(t)})
    return sorted(out, key=lambda r: r["alpha"])


def fmt_re(re):
    """3000000 -> '3e6', 2200000 -> '2.2e6' (how engineers write it)"""
    if re is None or not np.isfinite(re) or re <= 0:
        return "?"
    e = int(np.floor(np.log10(re) + 1e-9))
    return f"{re / 10 ** e:.3g}e{e}"


def display_name(name):
    """'naca2412' / 'NACA 2412' -> 'NACA 2412'; anything else unchanged"""
    s = str(name)
    m = re.fullmatch(r"(?i)naca\s*(\d{4,5})", s.strip())
    return f"NACA {m.group(1)}" if m else s


# ----------------------------------------------------------------------
# readers
# ----------------------------------------------------------------------
def read_vtu_cell_data(path):
    import io_vtk
    return io_vtk.read_vtu_cell_data(path)


def read_mesh_npz(path):
    import io_vtk
    return io_vtk.read_case_mesh(Path(path).parent)


def scan_polars():
    """every polar on disk: airfoil.py outputs (cases/*/polar.json) and the
    wing tool's section-polar cache"""
    rows = []
    for f in sorted(CASES.glob("*/polar.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        s = d.get("settings", {})
        res = d.get("results", [])
        rows.append({"name": d.get("info", {}).get("name", f.parent.name), "Re": d.get("re"),
                     "Mach": s.get("mach", res[0].get("mach", 0.0) if res else 0.0),
                     "transition": s.get("transition", res[0].get("transition", "") if res else ""),
                     "angles": len(res), "source": "airfoil analysis", "path": str(f)})
    for f in sorted(POLAR_CACHE.glob("*.json")) if POLAR_CACHE.exists() else []:
        try:
            res = json.loads(f.read_text())
        except Exception:
            continue
        m = re.match(r"(.+)_Re([\d.e+-]+)_M([\d.]+)_L([\d.]+)_(\w+)", f.stem)
        rows.append({"name": m.group(1) if m else f.stem,
                     "Re": float(m.group(2)) if m else None,
                     "Mach": float(m.group(3)) if m else 0.0,
                     "transition": m.group(5) if m else "", "angles": len(res),
                     "source": "wing section cache", "path": str(f)})
    return rows


def load_polar_rows(path):
    d = json.loads(Path(path).read_text())
    return d["results"] if isinstance(d, dict) else d


# ----------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------
def polar_figures(series):
    """series: list of (label, rows) -> four plotly figures"""
    import plotly.graph_objects as go
    figs = {k: go.Figure() for k in ("cl", "polar", "cm", "ld")}
    for lab, rows in series:
        r = [x for x in rows if x.get("Cl") is not None and np.isfinite(x["Cl"])]
        a = [x["alpha"] for x in r]
        cl = [x["Cl"] for x in r]
        cd = [x["Cd"] for x in r]
        cm = [x["Cm"] for x in r]
        txt = [x.get("status", "") for x in r]
        figs["cl"].add_scatter(x=a, y=cl, mode="lines+markers", name=lab, text=txt,
                               hovertemplate="α %{x:g}°<br>Cl %{y:.4f}<br>%{text}<extra>%{fullData.name}</extra>")
        figs["polar"].add_scatter(x=cd, y=cl, mode="lines+markers", name=lab, text=a,
                                  hovertemplate="Cd %{x:.5f}<br>Cl %{y:.4f}<br>α %{text:g}°<extra>%{fullData.name}</extra>")
        figs["cm"].add_scatter(x=a, y=cm, mode="lines+markers", name=lab)
        figs["ld"].add_scatter(x=a, y=[c / d if d else None for c, d in zip(cl, cd)],
                               mode="lines+markers", name=lab)
    lab = {"cl": ("α (deg)", "Cl", "Lift curve"), "polar": ("Cd", "Cl", "Drag polar"),
           "cm": ("α (deg)", "Cm, c/4", "Pitching moment"), "ld": ("α (deg)", "L/D", "Lift-to-drag ratio")}
    for k, f in figs.items():
        f.update_layout(xaxis_title=lab[k][0], yaxis_title=lab[k][1], title=lab[k][2], height=340,
                        showlegend=len(series) > 1)
    return figs


def field_image(case_dir, field, xlim=(-0.4, 1.6), ylim=(-0.5, 0.5), cmap="turbo"):
    """matplotlib figure of a cell field around the airfoil"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    case_dir = Path(case_dir)
    P, cells = read_mesh_npz(case_dir / "mesh.npz")
    data = read_vtu_cell_data(case_dir / "results.vtu")
    v = data[field]
    if v.ndim > 1:
        v = np.linalg.norm(v[:, :2], axis=1)
    cen = np.array([P[c].mean(0) for c in cells])
    sel = np.where((cen[:, 0] > xlim[0] - 0.2) & (cen[:, 0] < xlim[1] + 0.2) &
                   (cen[:, 1] > ylim[0] - 0.2) & (cen[:, 1] < ylim[1] + 0.2))[0]
    fig, ax = plt.subplots(figsize=(13, 5.4))
    vs = v[sel]
    lo, hi = np.nanpercentile(vs, [1, 99])
    if field in ("Cp",):
        lo, hi = max(lo, -3.0), min(hi, 1.0)
    pc = PolyCollection([P[cells[i]] for i in sel], array=vs, cmap=cmap, edgecolors="face",
                        linewidths=0.05)
    pc.set_clim(lo, hi)
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, label=field, shrink=0.85, pad=0.015)
    ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
    if field == "Mach":
        import matplotlib.tri as mtri
        t = mtri.Triangulation(cen[sel, 0], cen[sel, 1])
        try:
            ax.tricontour(t, vs, [1.0], colors="k", linewidths=0.8)
        except Exception:
            pass
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    fig.tight_layout()
    return fig, sorted(data.keys())


# ----------------------------------------------------------------------
# geometry + inviscid preview
# ----------------------------------------------------------------------
def raw_outline(spec):
    import airfoil_geom as ag
    s = str(spec)
    if not s.lower().startswith(("naca", "tmr")) and not Path(s).is_absolute():
        s = str(SRC / s)
    return ag.sharp_outline(s)


def panel_polar(spec, alphas):
    """inviscid panel solutions; the panels at a sharp trailing edge (singular
    Cp, -100 and below) are dropped from the plotted distribution"""
    import panel
    o = raw_outline(spec)
    out = [panel.solve([o], a) for a in alphas]
    for r in out:
        for e in r["elements"]:
            x = np.asarray(e["x"])
            k = x < 0.995
            e["x"], e["Cp"] = list(x[k]), list(np.asarray(e["Cp"])[k])
    return out, o


def count_angles(s):
    s = str(s).replace(" ", "")
    if ":" in s:
        a, b, d = (float(x) for x in s.split(":"))
        return int(np.floor((b - a) / d + 1e-9)) + 1
    return len(s.split(","))
