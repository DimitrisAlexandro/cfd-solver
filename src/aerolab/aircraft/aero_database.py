"""
Aerodynamic database builder.

Aircraft tables (vortex lattice + CFD section polars, aircraft.py):
    python src/run.py aero_database aircraft ../cases/aircraft/demo_trainer.json \\
        --alpha -6:20:1 --beta -10:10:2 --mach 0.15 --alt 1000 \\
        --deflect elevator=-20,-10,10,20 aileron=-15,15 rudder=-20,20 --out ../cases/db_demo
  -> database.csv (long table), database.xlsx (one sheet per table),
     database.json, jsbsim_aero.xml (complete <aerodynamics> section:
     CL/CD/Cm(alpha[, Mach]) tables, control and damping terms),
     aircraft.avl (+ section files) for AVL, database.png

Section polars (CFD, the RANS solver) over Reynolds and Mach numbers:
    python src/run.py aero_database section --airfoil naca2412 --re 1e6,3e6,6e6 --mach 0.1,0.5 \\
        --alpha=-6:20:2 [--flap 0,10,20 --flap-chord 0.25] --out ../cases/sdb_naca2412
  -> section_database.csv / .xlsx / .json and XFOIL-format polar files
     (polar_Re..._M....txt) that XFLR5 and other tools import.

Aircraft tables: the clean aircraft over alpha x beta (nonlinear sections:
stall included), control increments over alpha x deflection (beta = 0),
damping and alpha-dot derivatives against alpha. Moments about the c.g.
Forces CL, CD (wind), CY (body); moments in body axes for flight
simulation (Cl_b, Cm, Cn_b) and stability axes (Cl, Cn) in the long table.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from aerolab.aircraft import aircraft as A
from aerolab.airfoil import airfoil_geom as ag
from aerolab.aircraft import wing as W

KEYS = ("CL", "CD", "CY", "Cl", "Cm", "Cn")
SHEET = {"CL": "CL lift", "CD": "CD drag", "CY": "CY side", "Cl": "Cl roll", "Cm": "Cm pitch",
         "Cn": "Cn yaw"}           # Excel sheet names ignore case: CL / Cl would collide


def parse_range(s):
    s = str(s).replace(" ", "")
    if ":" in s:
        a, b, d = (float(x) for x in s.split(":"))
        return list(np.round(np.arange(a, b + 0.5 * d, d), 6))
    return [float(x) for x in s.split(",") if x]


# ----------------------------------------------------------------------
# aircraft tables
# ----------------------------------------------------------------------
def aircraft_database(ac, alphas, betas=(0.0,), machs=(0.15,), altitude=0.0, deflections=None,
                      polar_dir=None, nonlinear=True, log=print):
    deflections = deflections or {}
    rho, mu, a_snd, *_ = W.isa(altitude)
    rows, damping = [], []
    meta = {"aircraft": ac.name, "altitude": altitude, "alphas": list(alphas), "betas": list(betas),
            "machs": list(machs), "deflections": deflections, "nonlinear": nonlinear,
            "S": ac.S, "b": ac.b, "c": ac.c, "cg": list(ac.cg), "polar_source": {}}
    t0 = time.time()
    al_sorted = sorted(alphas, key=lambda a: abs(a))
    for M in machs:
        V = M * a_snd
        an = A.Analysis(ac, V, altitude, polar_dir, mach=M)
        meta["polar_source"][f"{M:g}"] = dict(zip([s.name for s in ac.surfaces], an.polar_source))
        log(f"[database] Mach {M:g} (V {V:.1f} m/s): clean aircraft, {len(alphas)} x {len(betas)} states")

        def sweep(beta, controls):
            """alpha sweep with continuation from the smallest |alpha| outward"""
            res = {}
            for direction in (1, -1):
                d = None
                seq = sorted([a for a in alphas if (a >= al_sorted[0] if direction > 0 else a < al_sorted[0])],
                             reverse=direction < 0)
                for a in seq:
                    r = an.solve(alpha=a, beta=beta, controls=controls, nonlinear=nonlinear, delta0=d)
                    rb = an.solve(alpha=a, beta=beta, controls=controls, nonlinear=nonlinear,
                                  delta0=r["_delta"], axes="body", max_it=5)
                    d = r["_delta"]
                    res[a] = (r, rb)
            return res

        clean = {}
        for b in betas:
            sw = sweep(b, None)
            clean[b] = sw
            for a in alphas:
                r, rb = sw[a]
                rows.append({"mach": M, "alpha": a, "beta": b, "control": "", "deflection": 0.0,
                             **{k: r[k] for k in KEYS}, "Cl_body": rb["Cl"], "Cn_body": rb["Cn"],
                             "stall_fraction": float(np.mean(r["strips"]["stalled"])),
                             "converged": r["converged"]})
        for name, defl in deflections.items():
            if name not in an.lat.ctrl_panels:
                log(f"[database]   no control named {name!r}: skipped")
                continue
            log(f"[database]   {name}: {len(defl)} deflections")
            b0 = 0.0 if 0.0 in clean else betas[0]
            for dv in defl:
                if dv == 0:
                    continue
                sw = sweep(b0, {name: dv})
                for a in alphas:
                    r, rb = sw[a]
                    r0, rb0 = clean[b0][a]
                    rows.append({"mach": M, "alpha": a, "beta": b0, "control": name, "deflection": dv,
                                 **{k: r[k] for k in KEYS}, "Cl_body": rb["Cl"], "Cn_body": rb["Cn"],
                                 **{f"d{k}": r[k] - r0[k] for k in KEYS},
                                 "dCl_body": rb["Cl"] - rb0["Cl"], "dCn_body": rb["Cn"] - rb0["Cn"],
                                 "stall_fraction": float(np.mean(r["strips"]["stalled"])),
                                 "converged": r["converged"]})
        log(f"[database]   damping derivatives against alpha")
        for a in alphas:
            ds = an.derivatives(a, axes="stability")
            db = an.derivatives(a, axes="body", downwash=False)
            row = {"mach": M, "alpha": a}
            for k in KEYS:
                for v in ("p", "q", "r"):
                    row[f"{k}_{v}"] = ds[f"{k}_{v}"]
                    row[f"{k}_{v}_body"] = db[f"{k}_{v}"]
                row[f"{k}_b"] = ds[f"{k}_b"]
                row[f"{k}_b_body"] = db[f"{k}_b"]
            row["CL_adot"], row["Cm_adot"], row["deps_da"] = ds["CL_adot"], ds["Cm_adot"], ds["deps_da"]
            for name, cd in ds["controls"].items():
                for k in KEYS:
                    row[f"{k}_{name}"] = cd[k]
                for k in ("Cl", "Cn"):
                    row[f"{k}_{name}_body"] = db["controls"][name][k]
                row[f"Ch_{name}_d"] = cd["Ch_d"]
                row[f"Ch_{name}_a"] = cd["Ch_a"]
            damping.append(row)
    meta["time_s"] = round(time.time() - t0, 1)
    return {"meta": meta, "rows": rows, "damping": damping}


# ----------------------------------------------------------------------
def write_aircraft_outputs(ac, db, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(db["rows"])
    dd = pd.DataFrame(db["damping"])
    df.to_csv(out / "database.csv", index=False)
    dd.to_csv(out / "derivatives_vs_alpha.csv", index=False)
    (out / "database.json").write_text(json.dumps(A._jsonable(db), indent=1))
    with pd.ExcelWriter(out / "database.xlsx") as xw:
        pd.DataFrame([{"item": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v}
                      for k, v in db["meta"].items()]).to_excel(xw, sheet_name="info", index=False)
        clean = df[df.control == ""]
        for M in sorted(clean.mach.unique()):
            sub = clean[clean.mach == M]
            for k in KEYS:
                tab = sub.pivot(index="alpha", columns="beta", values=k)
                tab.columns = [f"beta {c:g}" for c in tab.columns]
                tab.to_excel(xw, sheet_name=f"{SHEET[k]} M{M:g}"[:31])
        ctl = df[df.control != ""]
        for (M, name), sub in ctl.groupby(["mach", "control"]):
            for k in ("CL", "Cm", "Cl", "Cn", "CY", "CD"):
                tab = sub.pivot(index="alpha", columns="deflection", values=f"d{k}")
                tab.columns = [f"{name} {c:g} deg" for c in tab.columns]
                tab.to_excel(xw, sheet_name=f"d{SHEET[k]} {name} M{M:g}"[:31])
        dd.to_excel(xw, sheet_name="derivatives vs alpha", index=False)
        df.to_excel(xw, sheet_name="long table", index=False)
    (out / "jsbsim_aero.xml").write_text(jsbsim_xml(ac, db))
    write_avl(ac, out)
    plot_database(db, out)


# ----------------------------------------------------------------------
def _table_xml(rows_alpha, cols, values, indent, row_var="aero/alpha-rad", col_var="velocities/mach"):
    """JSBSim <table>; rows in rad; one column (1D) or several (2D, Mach)"""
    pad = " " * indent
    s = [f"{pad}<table>", f'{pad}  <independentVar lookup="row">{row_var}</independentVar>']
    if len(cols) > 1:
        s.append(f'{pad}  <independentVar lookup="column">{col_var}</independentVar>')
    s.append(f"{pad}  <tableData>")
    if len(cols) > 1:
        s.append(f"{pad}{' ' * 14}" + "".join(f"{c:>12g}" for c in cols))
    for i, a in enumerate(rows_alpha):
        s.append(f"{pad}    {math.radians(a):>10.5f}" + "".join(f"{values[j][i]:>12.5f}" for j in range(len(cols))))
    s.append(f"{pad}  </tableData>")
    s.append(f"{pad}</table>")
    return "\n".join(s)


def jsbsim_xml(ac, db):
    """<aerodynamics> section for a JSBSim aircraft file. Coefficients from
    the database at beta = 0; lateral and damping terms as alpha tables.
    The <metrics> of the aircraft file must hold the same reference area,
    span and chord (given in the header comment)."""
    import pandas as pd
    df = pd.DataFrame(db["rows"])
    dd = pd.DataFrame(db["damping"])
    machs = sorted(df.mach.unique())
    clean = df[(df.control == "") & (df.beta == 0.0)]
    alphas = sorted(clean.alpha.unique())

    def col(frame, key):
        return [[float(frame[(frame.mach == M) & (frame.alpha == a)][key].iloc[0]) for a in alphas]
                for M in machs]

    ft2, ft = 10.7639, 3.28084
    q_S = ["<property>aero/qbar-psf</property>", "<property>metrics/Sw-sqft</property>"]
    out = ['<?xml version="1.0"?>',
           f"<!-- AeroLab aerodynamic database: {ac.name}",
           f"     reference area {ac.S:.4f} m2 ({ac.S * ft2:.3f} ft2), span {ac.b:.4f} m ({ac.b * ft:.3f} ft),",
           f"     chord {ac.c:.4f} m ({ac.c * ft:.3f} ft); moments about the c.g. at x = {ac.cg[0]:.4f} m",
           "     (put the aerodynamic reference point <AERORP> at the c.g.).",
           "     Vortex lattice + CFD section polars; body-axis moments; angles in rad.",
           "     Control conventions: elevator + = trailing edge down, aileron + = roll right,",
           "     rudder + = trailing edge left. -->",
           "<aerodynamics>"]

    def func(name, desc, factors, body, scale=None):
        s = [f'    <function name="aero/coefficient/{name}">',
             f"      <description>{desc}</description>", "      <product>"]
        s += [f"        {f}" for f in factors]
        if scale:
            s += [f"        {f}" for f in scale]
        s.append(body)
        s += ["      </product>", "    </function>"]
        return "\n".join(s)

    def tab(values, ind=8):
        return _table_xml(alphas, machs, values, ind)

    def dtab(key):
        return tab(col(dd, key))

    have = lambda k: k in dd.columns
    ctl = {n for n in df.control.unique() if n}
    var = {"elevator": "fcs/elevator-pos-rad", "aileron": "fcs/left-aileron-pos-rad",
           "rudder": "fcs/rudder-pos-rad", "flap": "fcs/flap-pos-rad"}
    bvar = "<property>aero/bi2vel</property>"
    cvar = "<property>aero/ci2vel</property>"
    # LIFT
    out.append('  <axis name="LIFT">')
    out.append(func("CLalpha", "Lift, clean aircraft (alpha, Mach)", q_S, tab(col(clean, "CL"))))
    for n in ctl & {"elevator", "flap"}:
        if have(f"CL_{n}"):
            out.append(func(f"CL_{n}", f"Lift due to {n}", q_S + [f"<property>{var[n]}</property>"],
                            dtab(f"CL_{n}")))
    out.append(func("CLq", "Lift due to pitch rate", q_S + [cvar, "<property>velocities/q-aero-rad_sec</property>"],
                    dtab("CL_q_body")))
    out.append(func("CLadot", "Lift due to alpha rate", q_S + [cvar, "<property>aero/alphadot-rad_sec</property>"],
                    dtab("CL_adot")))
    out.append("  </axis>")
    # DRAG
    out.append('  <axis name="DRAG">')
    out.append(func("CDalpha", "Drag, clean aircraft (alpha, Mach): induced + profile + fuselage", q_S,
                    tab(col(clean, "CD"))))
    cb = df[(df.control == "") & (df.alpha == alphas[np.argmin(np.abs(alphas))])]
    if len(cb.beta.unique()) > 2:
        k = np.polyfit(np.radians(cb.beta), cb.CD, 2)[0]
        out.append(func("CDbeta", "Drag due to sideslip (quadratic fit)", q_S,
                        f"        <property>aero/beta-rad</property>\n        <property>aero/beta-rad</property>\n"
                        f"        <value>{k:.5f}</value>"))
    out.append("  </axis>")
    # SIDE
    out.append('  <axis name="SIDE">')
    out.append(func("CYb", "Side force due to sideslip", q_S + ["<property>aero/beta-rad</property>"],
                    dtab("CY_b_body")))
    out.append(func("CYp", "Side force due to roll rate", q_S + [bvar, "<property>velocities/p-aero-rad_sec</property>"],
                    dtab("CY_p_body")))
    out.append(func("CYr", "Side force due to yaw rate", q_S + [bvar, "<property>velocities/r-aero-rad_sec</property>"],
                    dtab("CY_r_body")))
    if "rudder" in ctl and have("CY_rudder"):
        out.append(func("CYdr", "Side force due to rudder", q_S + [f"<property>{var['rudder']}</property>"],
                        dtab("CY_rudder")))
    out.append("  </axis>")
    # ROLL / PITCH / YAW
    qsb = q_S + ["<property>metrics/bw-ft</property>"]
    qsc = q_S + ["<property>metrics/cbarw-ft</property>"]
    out.append('  <axis name="ROLL">')
    out.append(func("Clb", "Roll due to sideslip", qsb + ["<property>aero/beta-rad</property>"], dtab("Cl_b_body")))
    out.append(func("Clp", "Roll damping", qsb + [bvar, "<property>velocities/p-aero-rad_sec</property>"],
                    dtab("Cl_p_body")))
    out.append(func("Clr", "Roll due to yaw rate", qsb + [bvar, "<property>velocities/r-aero-rad_sec</property>"],
                    dtab("Cl_r_body")))
    for n in ("aileron", "rudder"):
        if n in ctl and have(f"Cl_{n}_body"):
            out.append(func(f"Cl_{n}", f"Roll due to {n}", qsb + [f"<property>{var[n]}</property>"],
                            dtab(f"Cl_{n}_body")))
    out.append("  </axis>")
    out.append('  <axis name="PITCH">')
    out.append(func("Cmalpha", "Pitching moment, clean aircraft (alpha, Mach)", qsc, tab(col(clean, "Cm"))))
    for n in ("elevator", "flap"):
        if n in ctl and have(f"Cm_{n}"):
            out.append(func(f"Cm_{n}", f"Pitch due to {n}", qsc + [f"<property>{var[n]}</property>"],
                            dtab(f"Cm_{n}")))
    out.append(func("Cmq", "Pitch damping", qsc + [cvar, "<property>velocities/q-aero-rad_sec</property>"],
                    dtab("Cm_q_body")))
    out.append(func("Cmadot", "Pitch due to alpha rate", qsc + [cvar, "<property>aero/alphadot-rad_sec</property>"],
                    dtab("Cm_adot")))
    out.append("  </axis>")
    out.append('  <axis name="YAW">')
    out.append(func("Cnb", "Yaw due to sideslip", qsb + ["<property>aero/beta-rad</property>"], dtab("Cn_b_body")))
    out.append(func("Cnp", "Yaw due to roll rate", qsb + [bvar, "<property>velocities/p-aero-rad_sec</property>"],
                    dtab("Cn_p_body")))
    out.append(func("Cnr", "Yaw damping", qsb + [bvar, "<property>velocities/r-aero-rad_sec</property>"],
                    dtab("Cn_r_body")))
    for n in ("aileron", "rudder"):
        if n in ctl and have(f"Cn_{n}_body"):
            out.append(func(f"Cn_{n}", f"Yaw due to {n}", qsb + [f"<property>{var[n]}</property>"],
                            dtab(f"Cn_{n}_body")))
    out.append("  </axis>")
    out.append("</aerodynamics>")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------
def write_avl(ac, out):
    """AVL geometry file (+ coordinate files for non-NACA sections)"""
    out = Path(out)
    L = [ac.name, "0.0                       | Mach", "0  0  0.0                 | iYsym  iZsym  Zsym",
         f"{ac.S:.4f} {ac.c:.4f} {ac.b:.4f}   | Sref Cref Bref",
         f"{ac.cg[0]:.4f} {ac.cg[1]:.4f} {ac.cg[2]:.4f}   | Xref Yref Zref",
         f"{ac.cd_extra:.4f}                    | CDp (extra parasite drag; fuselage not modelled in AVL)", ""]

    def af_lines(spec):
        s = str(spec).lower().replace(" ", "")
        if s.startswith("naca") and len(s) == 8:
            return ["NACA", s[4:]]
        from aerolab.airfoil import airfoil_geom as ag
        o = ag.sharp_outline(spec)
        fn = f"{Path(str(spec)).stem}.dat"
        (out / fn).write_text(Path(str(spec)).stem + "\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in o) + "\n")
        return ["AFILE", fn]

    for s in ac.surfaces:
        L += ["#" + "=" * 60, "SURFACE", s.name,
              f"{s.nc}  1.0  {s.ns}  -2.0      | Nchord Cspace Nspan Sspace"]
        if s.symmetric:
            L += ["YDUPLICATE", f"{s.y:.4f}"]
        L += ["ANGLE", "0.0", ""]
        etas = sorted({0.0, 1.0} | {c.eta0 for c in s.controls} | {c.eta1 for c in s.controls}
                      | set(s.stations()[0].tolist()))
        for e in etas:
            c = float(s.chord(e))
            x = float(s.x_le(e))
            if s.vertical:
                y, z = s.y, s.z + e * s.semi
            else:
                y, z = s.y + e * s.semi, s.z + float(s.rise(e))
            inc = s.incidence + float(s.twist_at(e))
            L += ["SECTION", f"{x:.4f} {y:.4f} {z:.4f} {c:.4f} {inc:.3f}   | Xle Yle Zle Chord Ainc"]
            af = s.airfoil if (s.tip_airfoil is None or e < 0.5) else s.tip_airfoil
            L += af_lines(af)
            for ctl in s.controls:
                if ctl.eta0 - 1e-9 <= e <= ctl.eta1 + 1e-9:
                    g = ctl.gain if ctl.gain is not None else A.DEFAULT_GAINS.get(ctl.name, 1.0)
                    if s.vertical:
                        g = -g            # AVL: + rotation about +z hinge = trailing edge right
                    dup = -1.0 if ctl.mode == "antisymmetric" else 1.0
                    L += ["CONTROL", f"{ctl.name} {g:.1f} {1 - ctl.cf:.3f} 0. 0. 0. {dup:.0f}"
                          "   | name gain Xhinge XYZhvec SgnDup"]
            L.append("")
    (out / "aircraft.avl").write_text("\n".join(L) + "\n")
    # mass file for AVL eigenmode runs
    Ix, Iy, Iz = ac.inertias()
    (out / "aircraft.mass").write_text(
        "Lunit = 1.0 m\nMunit = 1.0 kg\nTunit = 1.0 s\ng = 9.81\nrho = 1.225\n"
        "# mass x y z Ixx Iyy Izz\n"
        f"{ac.mass:.3f} {ac.cg[0]:.4f} {ac.cg[1]:.4f} {ac.cg[2]:.4f} {Ix:.2f} {Iy:.2f} {Iz:.2f}\n")


# ----------------------------------------------------------------------
def plot_database(db, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(db["rows"])
    M = sorted(df.mach.unique())[0]
    d = df[df.mach == M]
    clean = d[d.control == ""]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    b0 = 0.0 if 0.0 in clean.beta.values else clean.beta.iloc[0]
    c0 = clean[clean.beta == b0]
    axs[0, 0].plot(c0.alpha, c0.CL, "o-", ms=3); axs[0, 0].set_ylabel("CL")
    axs[0, 1].plot(c0.CD, c0.CL, "o-", ms=3); axs[0, 1].set_xlabel("CD"); axs[0, 1].set_ylabel("CL")
    axs[0, 2].plot(c0.alpha, c0.Cm, "o-", ms=3, label="clean")
    for dv, sub in d[d.control == "elevator"].groupby("deflection"):
        axs[0, 2].plot(sub.alpha, sub.Cm, "-", lw=1, label=f"elevator {dv:g}°")
    axs[0, 2].axhline(0, color="k", lw=0.5); axs[0, 2].set_ylabel("Cm (c.g.)"); axs[0, 2].legend(fontsize=7)
    a0 = clean.alpha.iloc[np.argmin(np.abs(clean.alpha.values - 2.0))]
    cb = clean[clean.alpha == a0]
    for k, ax in zip(("CY", "Cl", "Cn"), axs[1]):
        ax.plot(cb.beta, cb[k], "o-", ms=3, label=f"clean, α {a0:g}°")
        ax.set_xlabel("β (deg)"); ax.set_ylabel(k + " (stability axes)")
    for name, k, ax in (("aileron", "Cl", axs[1, 1]), ("rudder", "Cn", axs[1, 2])):
        sub = d[(d.control == name) & (d.alpha == a0)]
        if len(sub):
            ax.plot(sub.deflection * 0 + b0, sub[k], "s", label=f"{name} " + ", ".join(f"{v:g}°" for v in sub.deflection))
        ax.legend(fontsize=7)
    for ax in axs[0, ::2]:
        ax.set_xlabel("α (deg)")
    for ax in axs.ravel():
        ax.grid(alpha=0.3)
    fig.suptitle(f"{db['meta']['aircraft']}: aerodynamic database, Mach {M:g}")
    fig.tight_layout()
    fig.savefig(out / "database.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------
# section polars (CFD) over Re x Mach
# ----------------------------------------------------------------------
def section_database(airfoil, re_list, machs, alphas, polar_dir, level=1.0, jobs=2,
                     transition="free", out=None, log=print, flaps=(0.0,), flap_chord=0.25):
    """runs (or reuses) the CFD section polars over Re x Mach x flap
    deflection (plain flap, airfoil_geom.deflect_flap); returns long-table rows"""
    rows = []
    for fd in flaps:
        spec = ag.flap_spec(airfoil, flap_chord, fd)
        for M in machs:
            log(f"[section database] flap {fd:g} deg, Mach {M:g}: Re " + ", ".join(f"{r:.3g}" for r in re_list))
            W.cfd_polars(spec, re_list, M, alphas, polar_dir, level=level, jobs=jobs,
                         transition=transition, log=log)
            for re in re_list:
                slug = f"{ag.section_stem(spec)}_Re{re:.3g}_M{M:.2f}_L{level:g}_{transition}".replace("+", "")
                f = Path(polar_dir) / f"{slug}.json"
                if not f.exists():
                    continue
                for r in json.loads(f.read_text()):
                    rows.append({"airfoil": ag.section_stem(airfoil), "Re": re, "mach": M, "flap": fd,
                                 "flap_chord": flap_chord if fd else 0.0, **r})
    return rows


def write_section_outputs(rows, out, airfoil, transition="free"):
    import pandas as pd
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "section_database.csv", index=False)
    (out / "section_database.json").write_text(json.dumps(A._jsonable(rows), indent=1))
    with pd.ExcelWriter(out / "section_database.xlsx") as xw:
        if "flap" not in df.columns:
            df["flap"] = 0.0
        for k, nm in (("Cl", "Cl lift"), ("Cd", "Cd drag"), ("Cm", "Cm pitch")):
            t = df.pivot_table(index="alpha", columns=["Re", "mach", "flap"], values=k)
            t.columns = [f"Re {r:.3g} M {m:g}" + (f" flap {f:g}" if f else "") for r, m, f in t.columns]
            t.to_excel(xw, sheet_name=nm)
        df.to_excel(xw, sheet_name="long table", index=False)
    name = ag.section_stem(airfoil)
    for (re, M, fl), sub in df.groupby(["Re", "mach", "flap"]):
        sub = sub.sort_values("alpha")
        ftxt = f", plain flap {100 * sub.flap_chord.iloc[0]:g}% c at {fl:g} deg" if fl else ""
        L = ["", "       XFOIL         Version 6.99", "",
             f" Calculated polar for: {name.upper()}{ftxt}  (AeroLab RANS, Spalart-Allmaras"
             f"{', e^N transition' if transition == 'free' else ', fully turbulent'})", "",
             " 1 1 Reynolds number fixed          Mach number fixed", "",
             " xtrf =   1.000 (top)        1.000 (bottom)",
             f" Mach = {M:7.3f}     Re = {re / 1e6:9.3f} e 6     Ncrit =   9.000", "",
             "   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr",
             "  ------ -------- --------- --------- -------- -------- --------"]
        for _, r in sub.iterrows():
            if not (r.get("Cl") is not None and np.isfinite(r["Cl"])):
                continue
            cdp = r.get("Cd_p")
            cdp = cdp if cdp is not None and np.isfinite(cdp) else r["Cd"]
            xu = r.get("x_tr_upper") if r.get("x_tr_upper") is not None and np.isfinite(r.get("x_tr_upper") or np.nan) else 0.0
            xl = r.get("x_tr_lower") if r.get("x_tr_lower") is not None and np.isfinite(r.get("x_tr_lower") or np.nan) else 0.0
            L.append(f"  {r['alpha']:6.3f} {r['Cl']:8.4f} {r['Cd']:9.5f} {cdp:9.5f} {r['Cm']:8.4f} "
                     f"{xu:8.4f} {xl:8.4f}")
        fn = f"polar_{name}_Re{re:.3g}_M{M:.2f}" + (f"_flap{fl:g}" if fl else "")
        (out / (fn.replace("+", "").replace(".", "p") + ".txt")).write_text("\n".join(L) + "\n")
    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for (re, M, fl), sub in df.groupby(["Re", "mach", "flap"]):
        sub = sub.sort_values("alpha")
        lab = f"Re {re:.3g}, M {M:g}" + (f", flap {fl:g}°" if fl else "")
        axs[0].plot(sub.alpha, sub.Cl, "o-", ms=3, label=lab)
        axs[1].plot(sub.Cd, sub.Cl, "o-", ms=3)
        axs[2].plot(sub.alpha, sub.Cm, "o-", ms=3)
    axs[0].set_xlabel("α (deg)"); axs[0].set_ylabel("Cl"); axs[0].legend(fontsize=7)
    axs[1].set_xlabel("Cd"); axs[1].set_ylabel("Cl"); axs[2].set_xlabel("α (deg)"); axs[2].set_ylabel("Cm")
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.suptitle(f"{name}: section database")
    fig.tight_layout(); fig.savefig(out / "section_database.png", dpi=110); plt.close(fig)


# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("aircraft")
    a.add_argument("aircraft")
    a.add_argument("--alpha", default="-6:20:1")
    a.add_argument("--beta", default="-10:10:2")
    a.add_argument("--mach", default="0.15")
    a.add_argument("--alt", type=float, default=0.0)
    a.add_argument("--deflect", nargs="*", default=[], help="name=d1,d2,... (deg)")
    a.add_argument("--linear", action="store_true", help="linear sections (no stall)")
    a.add_argument("--polar-dir", default="../cases/wing_polars")
    a.add_argument("--out", required=True)
    s = sub.add_parser("section")
    s.add_argument("--airfoil", required=True)
    s.add_argument("--re", required=True)
    s.add_argument("--mach", default="0.0")
    s.add_argument("--alpha", default="-6:20:2")
    s.add_argument("--refine", type=float, default=1.0)
    s.add_argument("--jobs", type=int, default=2)
    s.add_argument("--transition", default="free")
    s.add_argument("--flap", default="0", help="plain-flap deflections (deg), e.g. 0,10,20")
    s.add_argument("--flap-chord", type=float, default=0.25)
    s.add_argument("--polar-dir", default="../cases/wing_polars")
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "aircraft":
        ac = A.Aircraft.load(args.aircraft)
        defl = {}
        for item in args.deflect:
            k, v = item.split("=")
            defl[k] = [float(x) for x in v.split(",")]
        db = aircraft_database(ac, parse_range(args.alpha), parse_range(args.beta), parse_range(args.mach),
                               args.alt, defl, args.polar_dir, nonlinear=not args.linear)
        write_aircraft_outputs(ac, db, args.out)
        print(f"[database] {len(db['rows'])} states in {db['meta']['time_s']:.0f} s -> {args.out}/")
        (Path(args.out) / "database_done.json").write_text(json.dumps({"rows": len(db["rows"])}))
    else:
        rows = section_database(args.airfoil, parse_range(args.re), parse_range(args.mach),
                                parse_range(args.alpha), args.polar_dir, args.refine, args.jobs,
                                args.transition, flaps=parse_range(args.flap), flap_chord=args.flap_chord)
        write_section_outputs(rows, args.out, args.airfoil, args.transition)
        print(f"[section database] {len(rows)} points -> {args.out}/")
        (Path(args.out) / "database_done.json").write_text(json.dumps({"rows": len(rows)}))


if __name__ == "__main__":
    main()
