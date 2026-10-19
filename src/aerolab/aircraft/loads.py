"""
Flight loads: V-n diagram (manoeuvre + gust, CS-23 style) and the spanwise
shear, bending and torsion of the wing (or any lifting surface) for every
load case, with inertia relief, envelopes and an FEM export.

    python src/run.py loads ../cases/aircraft/demo_trainer.json --vc 60 --category normal --out ../cases/loads_demo

Method
  * V-n diagram, equivalent airspeed, sea-level reference density:
    manoeuvre limit load factors n+ = 2.1 + 24000/(W[lb] + 10000) (normal,
    at most 3.8), 4.4 (utility), 6.0 (aerobatic); n- = -0.4 n+ (-0.5
    aerobatic); V_A = V_S1 sqrt(n+); V_D = max(1.25 V_C, k V_C,min) with
    V_C,min = 33 sqrt(W/S [lb/ft2]) kt and k = 1.40 / 1.50 / 1.55. Gust load
    factors n = 1 +- K_g rho0 U_de V a / (2 W/S), K_g = 0.88 mu/(5.3 + mu),
    mu = 2 (W/S)/(rho c a g), U_de = 15.24 m/s at V_C, 7.62 m/s at V_D.
    Negative manoeuvre line: n- up to V_C, then linear to 0 at V_D
    (simplified; check against the regulation that applies).
  * Every case is trimmed (angle of attack + elevator, Cm = 0 about the
    c.g.) with the aircraft model (vortex lattice on all surfaces, linear
    lift: conservative for the tip loads near stall), so the wing load
    includes the tail's balancing load and the real span loading.
  * Strip forces integrated from the tip inward about a structural axis at
    a given fraction of the chord: shear (along the wing normal), bending
    (tip up +), torsion (nose up +), in-plane shear and bending. Inertia
    relief: wing structural mass (fraction of the take-off mass) spread
    with the strip area, optional fuel between two span stations, both
    loaded with n g at a chordwise centroid.
Outputs: loads.json, loads_<case>.csv (stations), envelope.csv,
strip_loads.csv (forces / moments at the structural-axis points, all
cases), loads_nastran.bdf (GRID, FORCE, MOMENT cards, one load set per
case), vn_diagram.png, span_loads.png.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from aerolab.aircraft import aircraft as A
from aerolab.aircraft import wing as W

LB, FT, KT = 0.45359237, 0.3048, 0.514444


def vn_diagram(ac, CLmax, CLmin, CLa, Vc, category="normal", Vd=None, altitude=0.0):
    """limit load factors, design speeds (EAS, m/s) and the envelope curves"""
    rho0 = 1.225
    rho = W.isa(altitude)[0]
    Wn = ac.mass * A.G0
    WS = Wn / ac.S
    W_lb = ac.mass / LB
    WS_psf = WS / (LB * A.G0 / FT**2)
    if category == "normal":
        n_pos = min(2.1 + 24000 / (W_lb + 10000), 3.8)
    else:
        n_pos = {"utility": 4.4, "aerobatic": 6.0}[category]
    n_neg = -(0.5 if category == "aerobatic" else 0.4) * n_pos
    Vs1 = math.sqrt(2 * WS / (rho0 * CLmax))
    Vs_neg = math.sqrt(2 * WS / (rho0 * abs(CLmin)))
    Va = Vs1 * math.sqrt(n_pos)
    Vg = Vs_neg * math.sqrt(abs(n_neg))
    k = {"normal": 1.40, "utility": 1.50, "aerobatic": 1.55}[category]
    Vc_min = (33 if category != "aerobatic" else 36) * math.sqrt(WS_psf) * KT
    if Vd is None:
        Vd = max(1.25 * Vc, k * Vc_min)
    mu = 2 * WS / (rho * ac.c * CLa * A.G0)
    Kg = 0.88 * mu / (5.3 + mu)

    def gust(V, U):
        return Kg * rho0 * U * V * CLa / (2 * WS)
    V = np.linspace(0, Vd, 300)
    curves = {
        "V": V.tolist(),
        "stall_pos": np.minimum(rho0 * V**2 * CLmax / (2 * WS), n_pos).tolist(),
        "stall_neg": np.maximum(-rho0 * V**2 * abs(CLmin) / (2 * WS), n_neg).tolist(),
        "gust_vc_pos": (1 + gust(V, 15.24)).tolist(), "gust_vc_neg": (1 - gust(V, 15.24)).tolist(),
        "gust_vd_pos": (1 + gust(V, 7.62)).tolist(), "gust_vd_neg": (1 - gust(V, 7.62)).tolist(),
    }
    neg_line = np.where(V <= Vc, n_neg, n_neg * (Vd - V) / max(Vd - Vc, 1e-9))
    curves["neg_limit"] = np.maximum(curves["stall_neg"], neg_line).tolist()
    cases = [
        ("A  manoeuvre, V_A", Va, n_pos), ("C  manoeuvre, V_C", Vc, n_pos), ("D  manoeuvre, V_D", Vd, n_pos),
        ("G  negative, V_G", Vg, n_neg), ("F  negative, V_C", Vc, n_neg), ("E  V_D, n = 0", Vd, 0.0),
        ("gust + V_C", Vc, 1 + gust(Vc, 15.24)), ("gust - V_C", Vc, 1 - gust(Vc, 15.24)),
        ("gust + V_D", Vd, 1 + gust(Vd, 7.62)), ("gust - V_D", Vd, 1 - gust(Vd, 7.62)),
        ("cruise 1 g, V_C", Vc, 1.0)]
    return {"category": category, "n_pos": n_pos, "n_neg": n_neg, "Vs1": Vs1, "Vs_neg": Vs_neg,
            "Va": Va, "Vg": Vg, "Vc": Vc, "Vd": Vd, "Vc_min": Vc_min, "mu_g": mu, "Kg": Kg,
            "W_S": WS, "CLmax": CLmax, "CLmin": CLmin, "CLa": CLa, "curves": curves,
            "cases": [{"name": n, "V_eas": v, "n": nn} for n, v, nn in cases]}


def clmax_estimate(ac, altitude, polar_dir, Vref):
    """trimmed-free CL_max / CL_min of the aircraft (nonlinear sections) and CL_alpha"""
    an = A.Analysis(ac, Vref, altitude, polar_dir)
    res, d = [], None
    for a in np.arange(0, 30.1, 1.0):
        r = an.solve(alpha=a, nonlinear=True, delta0=d)
        d = r["_delta"]
        res.append(r["CL"])
    cl = np.array(res)
    i = int(np.argmax(cl))
    CLmax = float(cl[i]) if i < len(cl) - 1 else float(cl[-1])
    neg, d = [], None
    for a in np.arange(0, -25.1, -1.0):
        r = an.solve(alpha=a, nonlinear=True, delta0=d)
        d = r["_delta"]
        neg.append(r["CL"])
    CLmin = float(np.min(neg))
    CLa = (an.solve(alpha=3.0)["CL"] - an.solve(alpha=1.0)["CL"]) / math.radians(2.0)
    return CLmax, CLmin, CLa


# ----------------------------------------------------------------------
def span_loads(ac, an, res, n, surface=0, x_axis=0.35, wing_mass_frac=0.10, fuel_mass=0.0,
               fuel_span=(0.1, 0.5), x_mass=0.40, scale=None):
    """station loads of one surface (right half or the single fin) for a
    solved state `res` (an.solve called last for this state). Units: N, N m."""
    L = an.lat
    s = ac.surfaces[surface]
    st = np.where((L.S["surf"] == surface) & (L.S["side"] >= 0))[0]
    st = st[np.argsort(L.S["eta"][st])]                       # root -> tip
    q2 = an.rho * an.V**2 if scale is None else scale         # rho V^2 (code units: rho = V = 1)
    Fp = an._last["Fp"] * q2
    Dv = an._last["Dv"] * q2
    cm = an._last["cm"]
    # strip forces and moments about the strip's structural-axis point
    E = L.S["le_mid"][st] + np.c_[x_axis * L.S["c"][st], 0 * st, 0 * st]
    F = np.zeros((len(st), 3))
    Mpt = np.zeros((len(st), 3))
    for j, i in enumerate(st):
        pan = np.where(L.pstrip == i)[0]
        F[j] = Fp[pan].sum(0) + Dv[i]
        Mpt[j] = np.cross(L.P_mid[pan] - E[j], Fp[pan]).sum(0) + np.cross(L.S["mid"][i] - E[j], Dv[i])
        Mpt[j] += 0.5 * q2 * L.S["c"][i] * L.S["area"][i] * cm[i] * L.S["s"][i]
    # inertia: structure ~ strip area, fuel between two stations; load factor n along body z
    area = L.S["area"][st]
    half = 0.5 if s.symmetric else 1.0
    m_struct = wing_mass_frac * ac.mass * half * area / area.sum() if surface == 0 else 0 * area
    eta = L.S["eta"][st]
    infuel = (eta >= fuel_span[0]) & (eta <= fuel_span[1])
    m_fuel = np.zeros_like(area)
    if surface == 0 and fuel_mass > 0 and infuel.any():
        m_fuel[infuel] = fuel_mass * half * area[infuel] / area[infuel].sum()
    m = m_struct + m_fuel
    g_dir = np.array([0.0, 0.0, -1.0])                         # weight direction (geometry axes, level)
    Fi = (n * A.G0 * m)[:, None] * g_dir[None, :]
    Pm = L.S["le_mid"][st] + np.c_[x_mass * L.S["c"][st], 0 * st, 0 * st]
    Mi = np.cross(Pm - E, Fi)
    F_net, M_net = F + Fi, Mpt + Mi
    # integrate from the tip: loads at the inboard edge of every strip
    e_s, e_n = L.S["s"][st], L.S["n0"][st]
    out = {k: [] for k in ("eta", "y", "shear", "bending", "torsion", "inplane_shear", "inplane_bending")}
    for j in range(len(st)):
        o = slice(j, None)
        Ej = E[j] - 0.5 * L.S["ds"][st[j]] * e_s[j]            # inboard edge of strip j
        Ft = F_net[o].sum(0)
        Mt = (np.cross(E[o] - Ej, F_net[o]) + M_net[o]).sum(0)
        ec = np.cross(e_s[j], e_n[j])
        out["eta"].append(float(eta[j] - 0.5 * L.S["ds"][st[j]] / s.semi))
        out["y"].append(float(Ej[2] if s.vertical else Ej[1]))
        out["shear"].append(float(Ft @ e_n[j]))
        out["bending"].append(float(Mt @ ec))
        out["torsion"].append(float(Mt @ e_s[j]))
        out["inplane_shear"].append(float(Ft @ ec))
        out["inplane_bending"].append(float(-Mt @ e_n[j]))
    strips = {"eta": eta.tolist(), "E": E.tolist(), "F_aero": F.tolist(), "F_inertia": Fi.tolist(),
              "M_net": M_net.tolist(), "F_net": F_net.tolist(), "mass": m.tolist()}
    return out, strips


# ----------------------------------------------------------------------
def run(ac, Vc, category="normal", altitude=0.0, Vd=None, polar_dir=None, surface=0, x_axis=0.35,
        wing_mass_frac=0.10, fuel_mass=0.0, fuel_span=(0.1, 0.5), x_mass=0.40, log=print, flexible=False):
    """flexible=True: the wing structure saved with the aircraft (Aircraft.structure)
    deforms under every case (aeroelastic.flexible_trim): elastic twist and
    bending slope change the wing incidence, the case is trimmed again until
    the deformation converges"""
    rho0 = 1.225
    rho = W.isa(altitude)[0]
    log("[loads] CL_max, CL_min and lift slope of the aircraft")
    CLmax, CLmin, CLa = clmax_estimate(ac, altitude, polar_dir, Vc * math.sqrt(rho0 / rho))
    vn = vn_diagram(ac, CLmax, CLmin, CLa, Vc, category, Vd, altitude)
    beam = None
    if flexible:
        from aerolab.aircraft import aeroelastic as AE
        from aerolab.aircraft import wing_structure as WS
        if not getattr(ac, "structure", None) or surface != 0:
            raise ValueError("flexible loads need the wing structure saved with the aircraft (Wing structure page)")
        beam = AE.beam_from_structure(ac, WS.from_aircraft(ac), None, fuel_mass=fuel_mass, fuel_span=fuel_span)
    cases = []
    for c in vn["cases"]:
        Vt = c["V_eas"] * math.sqrt(rho0 / rho)
        an = A.Analysis(ac, Vt, altitude, polar_dir)
        ctrl = "elevator" if "elevator" in an.lat.ctrl_panels else None
        if abs(c["n"]) < 1e-9:
            CLr = 0.0
        else:
            CLr = c["n"] * ac.mass * A.G0 / (an.qbar * ac.S)
        flex = None
        if beam is not None:
            r = AE.flexible_trim(an, beam, CLr, ctrl, inertia=AE.inertia_load(beam, c["n"]))
            flex = r
        elif ctrl:
            r = an.trim(CL_req=CLr, control=ctrl, alpha0=CLr / max(CLa, 1e-3) * 57.3)
        else:
            a = math.degrees(CLr / CLa)
            r = an.solve(alpha=a)
        # last solve must be the trimmed state for the strip forces
        r = an.solve(alpha=r.get("alpha_trim", r["alpha"]),
                     controls={ctrl: r["control_trim"]} if ctrl else None)
        st_loads, strips = span_loads(ac, an, r, c["n"], surface, x_axis, wing_mass_frac, fuel_mass,
                                      fuel_span, x_mass)
        tail = {s.name: v["CL"] * an.qbar * ac.S for s, v in zip(ac.surfaces, r["surfaces"].values())}
        case = {**c, "V_true": Vt, "CL": r["CL"], "alpha": r["alpha"],
                "elevator": r["controls"].get(ctrl) if ctrl else None, "lift_by_surface_N": tail,
                "root_shear": st_loads["shear"][0], "root_bending": st_loads["bending"][0],
                "root_torsion": st_loads["torsion"][0], "stations": st_loads, "strips": strips,
                "CL_over_CLmax": r["CL"] / CLmax}
        if flex is not None:
            case["elastic_tip_twist_deg"] = float(np.degrees(flex["elastic_incidence"][
                np.argmax(np.abs(an.lat.S["eta"]) * (an.lat.S["surf"] == 0))]))
            case["elastic_iterations"] = flex["elastic_iterations"]
            case["elastic_converged"] = flex["elastic_converged"]
        cases.append(case)
        log(f"[loads] {c['name']:20s} V_EAS {c['V_eas']:5.1f} m/s  n {c['n']:+5.2f}  α {r['alpha']:5.2f}°  "
            f"root: shear {st_loads['shear'][0] / 1e3:7.2f} kN  bending {st_loads['bending'][0] / 1e3:7.2f} kN m  "
            f"torsion {st_loads['torsion'][0] / 1e3:6.2f} kN m")
    # envelope over the cases
    keys = ("shear", "bending", "torsion")
    env = {"eta": cases[0]["stations"]["eta"], "y": cases[0]["stations"]["y"]}
    for k in keys:
        arr = np.array([c["stations"][k] for c in cases])
        env[f"{k}_max"] = arr.max(0).tolist()
        env[f"{k}_min"] = arr.min(0).tolist()
        env[f"{k}_max_case"] = [cases[i]["name"] for i in arr.argmax(0)]
        env[f"{k}_min_case"] = [cases[i]["name"] for i in arr.argmin(0)]
    return {"aircraft": ac.name, "surface": ac.surfaces[surface].name, "altitude": altitude,
            "x_axis": x_axis, "wing_mass_frac": wing_mass_frac, "fuel_mass": fuel_mass,
            "fuel_span": list(fuel_span), "vn": vn, "cases": cases, "envelope": env, "flexible": bool(flexible)}


def write_outputs(res, out):
    import pandas as pd
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "loads.json").write_text(json.dumps(A._jsonable(res), indent=1))
    for c in res["cases"]:
        slug = c["name"].split()[0] if c["name"][0].isalpha() and c["name"][1] == " " else \
            c["name"].replace(" ", "_").replace("+", "pos").replace("-", "neg").replace(",", "")
        pd.DataFrame(c["stations"]).to_csv(out / f"loads_{slug}.csv", index=False)
    pd.DataFrame(res["envelope"]).to_csv(out / "envelope.csv", index=False)
    rows = []
    for c in res["cases"]:
        s = c["strips"]
        for j, eta in enumerate(s["eta"]):
            rows.append({"case": c["name"], "eta": eta, "x": s["E"][j][0], "y": s["E"][j][1], "z": s["E"][j][2],
                         "Fx": s["F_net"][j][0], "Fy": s["F_net"][j][1], "Fz": s["F_net"][j][2],
                         "Mx": s["M_net"][j][0], "My": s["M_net"][j][1], "Mz": s["M_net"][j][2],
                         "mass_kg": s["mass"][j]})
    pd.DataFrame(rows).to_csv(out / "strip_loads.csv", index=False)
    # Nastran bulk data (free field): structural-axis grid points + one load set per case
    c0 = res["cases"][0]["strips"]
    L = ["$ AeroLab flight loads: " + res["aircraft"] + ", " + res["surface"] + " (right half)",
         "$ units N, m; geometry axes x aft, y right, z up (AeroLab); one FORCE/MOMENT set per case",
         "$ net loads = aerodynamic + inertia relief (n g on the wing structure and fuel)"]
    for j, E in enumerate(c0["E"]):
        L.append(f"GRID,{1001 + j},,{E[0]:.5f},{E[1]:.5f},{E[2]:.5f}")
    for k, c in enumerate(res["cases"]):
        sid = 101 + k
        L.append(f"$ load set {sid}: {c['name']}, V_EAS {c['V_eas']:.2f} m/s, n {c['n']:.3f}")
        s = c["strips"]
        for j in range(len(s["eta"])):
            F, M = s["F_net"][j], s["M_net"][j]
            fm = math.sqrt(sum(v * v for v in F)) or 1.0
            mm = math.sqrt(sum(v * v for v in M)) or 1.0
            L.append(f"FORCE,{sid},{1001 + j},0,{fm:.6g},{F[0] / fm:.6f},{F[1] / fm:.6f},{F[2] / fm:.6f}")
            L.append(f"MOMENT,{sid},{1001 + j},0,{mm:.6g},{M[0] / mm:.6f},{M[1] / mm:.6f},{M[2] / mm:.6f}")
    (out / "loads_nastran.bdf").write_text("\n".join(L) + "\n")
    plots(res, out)


def plots(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vn = res["vn"]
    cv = vn["curves"]
    V = np.array(cv["V"])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(V, cv["stall_pos"], "C0", label="manoeuvre (stall / n+)")
    ax.plot(V, cv["neg_limit"], "C0")
    ax.plot([vn["Va"], vn["Vd"]], [vn["n_pos"]] * 2, "C0")
    ax.plot([vn["Vd"], vn["Vd"]], [cv["neg_limit"][-1], vn["n_pos"]], "C0")
    for k, ls in (("gust_vc_pos", "--"), ("gust_vc_neg", "--"), ("gust_vd_pos", ":"), ("gust_vd_neg", ":")):
        m = V <= (vn["Vc"] if "vc" in k else vn["Vd"])
        ax.plot(V[m], np.array(cv[k])[m], "C1" + ls, lw=1,
                label={"gust_vc_pos": "gust, 15.24 m/s (V_C)", "gust_vd_pos": "gust, 7.62 m/s (V_D)"}.get(k))
    for c in res["cases"]:
        ax.plot(c["V_eas"], c["n"], "ks", ms=4)
        ax.annotate(c["name"].split(",")[0], (c["V_eas"], c["n"]), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    for k in ("Vs1", "Va", "Vc", "Vd"):
        ax.axvline(vn[k], color="0.7", lw=0.6)
        ax.text(vn[k], vn["n_neg"] - 0.6, {"Vs1": "$V_{S1}$", "Va": "$V_A$", "Vc": "$V_C$", "Vd": "$V_D$"}[k],
                fontsize=9, ha="center")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("equivalent airspeed (m/s)"); ax.set_ylabel("load factor n")
    ax.set_title(f"{res['aircraft']}: V-n diagram ({vn['category']} category, simplified CS-23)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "vn_diagram.png", dpi=110); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for c in res["cases"]:
        st = c["stations"]
        for ax, k, sc in zip(axs, ("shear", "bending", "torsion"), (1e-3, 1e-3, 1e-3)):
            ax.plot(st["eta"], np.array(st[k]) * sc, lw=1, label=c["name"])
    for ax, lab in zip(axs, ("shear (kN)", "bending, tip up + (kN m)", "torsion, nose up + (kN m)")):
        ax.set_xlabel("η = 2y/b"); ax.set_ylabel(lab); ax.grid(alpha=0.3); ax.axhline(0, color="k", lw=0.5)
    axs[0].legend(fontsize=6)
    fig.suptitle(f"{res['surface']}: limit loads at the structural axis ({100 * res['x_axis']:.0f} % chord), "
                 "aerodynamic + inertia")
    fig.tight_layout(); fig.savefig(out / "span_loads.png", dpi=110); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aircraft")
    ap.add_argument("--vc", type=float, required=True, help="design cruise speed V_C, EAS (m/s)")
    ap.add_argument("--vd", type=float, default=None)
    ap.add_argument("--category", default="normal", choices=["normal", "utility", "aerobatic"])
    ap.add_argument("--alt", type=float, default=0.0)
    ap.add_argument("--surface", type=int, default=0)
    ap.add_argument("--axis", type=float, default=0.35, help="structural axis, fraction of the chord")
    ap.add_argument("--wing-mass", type=float, default=0.10, help="wing structure, fraction of the mass")
    ap.add_argument("--fuel", type=float, default=0.0, help="fuel in the wing (kg, both sides)")
    ap.add_argument("--fuel-span", default="0.1,0.5")
    ap.add_argument("--polar-dir", default="../cases/wing_polars")
    ap.add_argument("--flexible", action="store_true",
                    help="flexible wing: uses the wing structure saved with the aircraft")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ac = A.Aircraft.load(a.aircraft)
    fs = tuple(float(x) for x in a.fuel_span.split(","))
    res = run(ac, a.vc, a.category, a.alt, a.vd, a.polar_dir, a.surface, a.axis, a.wing_mass, a.fuel, fs, flexible=a.flexible)
    write_outputs(res, a.out)
    print(f"-> {a.out}/ (loads.json, envelope.csv, strip_loads.csv, loads_nastran.bdf, vn_diagram.png, span_loads.png)")


if __name__ == "__main__":
    main()
