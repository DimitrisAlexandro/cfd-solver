"""
Validation of the conceptual-design tools: drag buildup, propeller, sizing,
weight and balance.

    python src/run.py validate_design_tools                 # ~10 s
    python src/run.py validate_design_tools --verify        # + sizing -> performance-tool verification (~3 min)

Writes validation/design_tools_validation.png and design_tools_report.md.

References
  * Cessna 172 (UIUC / FlightGear aerodynamic model, from Roskam): CD0 0.031;
    commonly quoted 0.027-0.031. 172S: MTOW 1111 kg, empty about 757 kg, wing
    16.2 m^2, 119 kW, 53 US gal usable fuel (144 kg).
  * Pipistrel Velis Electro (type-certified electric trainer): MTOW 600 kg,
    empty 428 kg including 2 x 70 kg batteries, 24.8 kWh, span 10.71 m, wing
    9.51 m^2, 57.6 kW take-off / 49.2 kW continuous, cruise 170 km/h, stall
    83 km/h flaps down, endurance 50 min plus VFR reserve.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from aerolab.aircraft import aircraft as A
from aerolab.aircraft import drag_buildup as DB
from aerolab.aircraft import propeller as P
from aerolab.aircraft import sizing as SZ
from aerolab.aircraft import weight_balance as WB
from aerolab.aircraft import wing as W

VAL = Path(__file__).resolve().parents[3] / "validation"


def c172_like():
    """strut-braced high wing, constant-chord inboard panel, tapered outboard"""
    wing = A.Surface("wing", span=11.0, root_chord=1.63, tip_chord=1.12, x=2.0, z=0.75, dihedral=1.73, twist=-3,
                     incidence=1.5, airfoil="naca2412", kinks=[{"y": 2.6, "chord": 1.63, "dx": 0.0,
                                                                "dz": 2.6 * 0.0302, "twist": -0.5}])
    ht = A.Surface("horizontal tail", span=3.45, root_chord=1.15, tip_chord=0.8, x=6.95, z=0.0, airfoil="naca0012")
    fin = A.Surface("fin", span=1.5, root_chord=1.55, tip_chord=0.75, x=6.6, z=0.35, sweep=35, airfoil="naca0012",
                    symmetric=False, vertical=True)
    fus = A.Fuselage(length=8.0, diameter=1.16, x=0, z=0.1, x_max=0.3, tail_ratio=0.1)
    return A.Aircraft("C172-like", [wing, ht, fin], fus, mass=1111)


C172_DRAG = dict(laminar={"wing": 0.05, "tail": 0.05, "fuselage": 0.0}, roughness="production sheet metal",
                 gear={"items": [{"item": "wheel with fairing", "kind": "wheel with fairing", "count": 3,
                                  "width": 0.13, "length": 0.38},
                                 {"item": "main spring leg", "kind": "flat spring leg", "count": 2, "width": 0.04,
                                  "length": 0.5},
                                 {"item": "nose strut", "kind": "round strut", "count": 1, "width": 0.07,
                                  "length": 0.4}]},
                 upsweep_deg=8, power_kw=119, other=[{"item": "wing struts (streamline)", "D_q": 0.05 * 2 * 3.1 * 0.09}],
                 leakage_pct=10)

C172_REQ = {"name": "C172-class sizing check", "category": "general aviation, single engine",
            "crew": 1, "passengers": 1, "person_mass": 86, "baggage": 25,
            "range_km": 1100, "reserve_min": 45, "cruise_speed": 57, "cruise_alt": 2400, "cruise_power": 0.65,
            "stall_speed": 24.7, "takeoff_distance": 497, "landing_distance": 407, "climb_rate": 3.7, "ceiling": 4270,
            "aspect_ratio": 7.45, "taper": 0.67, "high_wing": True, "airfoil": "naca2412",
            "cl_max_takeoff": 1.6, "cl_max_landing": 1.8, "engine": "piston", "sfc": 290, "prop_mode": "fixed_pitch",
            "rpm": 2700, "gear": "fixed_faired", "seats_abreast": 2, "load_factor": 3.8,
            "drag": {"roughness": "production sheet metal", "laminar": {"wing": 0.05, "tail": 0.05, "fuselage": 0.0},
                     "upsweep_deg": 8, "leakage_pct": 10, "other": [{"item": "wing struts", "D_q": 0.028}]}}
C172_REF = {"MTOW (kg)": 1111, "empty mass (kg)": 757, "fuel (kg)": 144, "wing area (m²)": 16.2, "power (kW)": 119}

VELIS_REQ = {"name": "Velis-Electro-class sizing check", "engine": "electric", "category": "homebuilt, composite",
             "composite": True, "crew": 1, "passengers": 1, "person_mass": 86, "baggage": 0,
             "range_km": 141, "reserve_min": 20, "cruise_speed": 47, "cruise_alt": 600, "cruise_power": 0.85,
             "stall_speed": 23.1, "takeoff_distance": 450, "landing_distance": 450, "climb_rate": 3.5, "ceiling": 3000,
             "aspect_ratio": 12.04, "taper": 0.6, "high_wing": True, "battery_wh_kg": 177, "battery_usable": 0.85,
             "prop_mode": "fixed_pitch", "rpm": 2400, "gear": "fixed_faired", "load_factor": 4.0}
VELIS_REF = {"MTOW (kg)": 600, "empty mass without battery (kg)": 288, "battery (kg)": 140, "wing area (m²)": 9.51,
             "span (m)": 10.71, "power (kW)": 49.2}


def run(verify=False, log=print):
    out = {}
    # 1. drag buildup
    ac = c172_like()
    db = DB.buildup(ac, 55.0, 1500.0, **C172_DRAG)
    out["drag_c172"] = {"CD0": db["CD0"], "reference": "0.027-0.031 (UIUC/Roskam model 0.031)",
                        "cfe_check": db["cfe_check"]["CD0"], "f_m2": db["f_m2"], "table": DB.table(db)}
    per, tc, _ = DB.section_shape("naca0012")
    out["perimeter_naca0012"] = {"exact": per, "raymer": 1.977 + 0.52 * tc}
    log(f"[drag] C172-like CD0 {db['CD0']:.4f} (ref 0.027-0.031), Cfe {db['cfe_check']['CD0']:.4f}")
    # 2. propeller
    prop, info = P.design(60.0, 2500.0, 1.8, 2, power=60e3, cl_design=0.7)
    chk = P.analyse(prop, 60.0, 2500.0)
    out["prop_consistency"] = {"design_T": info["T"], "bemt_T": chk["T"], "design_P": info["P"], "bemt_P": chk["P"],
                               "design_eta": info["eta"], "bemt_eta": chk["eta"], "eta_ideal": info["eta_ideal"]}
    sec = P.GenericSection(cd0=0.0, k=0.0, t_c=0.0)
    sec.polar = W.SectionPolar.thin_airfoil(a0=2 * math.pi, alpha0_deg=-3.5)
    blades = [2, 3, 4, 6, 8, 12, 20]
    etas, ideal = [], None
    for B in blades:
        _, inf = P.design(30.0, 2200.0, 2.0, B, power=40e3, cl_design=0.6, section=sec)
        etas.append(inf["eta"])
        ideal = inf["eta_ideal"]
    out["prop_inviscid"] = {"blades": blades, "eta": etas, "eta_ideal": ideal}
    sweep = P.sweep_J(prop, np.linspace(0.05, 1.05, 21), 2500.0)
    log(f"[prop] design T {info['T']:.1f} / BEMT {chk['T']:.1f} N, P {info['P'] / 1e3:.2f} / {chk['P'] / 1e3:.2f} kW")
    # 3. sizing
    rows = []
    for req, ref, lab in ((C172_REQ, C172_REF, "C172"), (VELIS_REQ, VELIS_REF, "Velis")):
        res, _ = SZ.size(req, log=lambda *a: None)
        got = {"MTOW (kg)": res["mtow"], "fuel (kg)": res["fuel"], "wing area (m²)": res["wing_area"],
               "power (kW)": res["power_kw"], "span (m)": res["span"], "battery (kg)": res["battery"],
               "empty mass (kg)": res["empty_mass"], "empty mass without battery (kg)": res["empty_mass"]}
        for k, v in ref.items():
            rows.append({"case": lab, "quantity": k, "reference": v, "sized": got[k], "diff_pct": 100 * (got[k] / v - 1)})
        out[f"sizing_{lab}"] = {"mtow": res["mtow"], "CD0": res["CD0"], "e": res["oswald_e"],
                                "empty_fraction": res["empty_fraction"], "class_i": res["class_i_empty_fraction"]}
        log(f"[sizing] {lab}: MTOW {res['mtow']:.0f} kg (ref {ref['MTOW (kg)']})")
    out["sizing_rows"] = rows
    # 4. weight and balance: scissor plot of the demo trainer
    demo = A.demo_aircraft()
    sc = WB.scissor(demo, 0.05, 1.9, factors=np.linspace(0.4, 1.6, 7))
    ld = WB.loading(demo)
    out["scissor_demo"] = {"rows": sc["rows"], "current": sc["Sh_S_current"],
                           "min_tail": WB.min_tail(sc, ld["forward"]["pct_mac"], ld["aft"]["pct_mac"]),
                           "cg_range": [ld["forward"]["pct_mac"], ld["aft"]["pct_mac"]]}
    # 5. optional: sizing -> performance-tool verification
    if verify:
        from aerolab.aircraft import performance as PF
        res, acs = SZ.size({}, log=lambda *a: None)
        spec = acs.propulsion
        pr = P.make_propulsion(spec)
        reserve = res["fuel"] - res["mission"]["cruise_fuel"]
        sl = PF.run(acs, pr, 0.0, str(VAL.parent / "data" / "wing_polars"), res["fuel"], reserve, log=lambda *a: None)
        cr = PF.run(acs, P.make_propulsion(spec), res["requirements"]["cruise_alt"],
                    str(VAL.parent / "data" / "wing_polars"), res["fuel"], reserve, log=lambda *a: None)
        rq = res["requirements"]
        out["verify"] = {
            "stall speed, landing flap (m/s)": (rq["stall_speed"], sl["landing"]["V_stall"]),
            "take-off over 15 m (m)": (rq["takeoff_distance"], sl["takeoff"]["total"]),
            "landing over 15 m (m)": (rq["landing_distance"], sl["landing"]["total"]),
            "rate of climb (m/s)": (rq["climb_rate"], sl["best_rate"]["roc"]),
            "service ceiling (m)": (rq["ceiling"], sl["ceiling_service"]),
            "range at cruise speed, cruise fuel (km)": (rq["range_km"], float(np.interp(
                rq["cruise_speed"], cr["curves"]["V"], cr["curves"]["range_km"]))),
            "landing CL max": (rq["cl_max_landing"], sl["landing"]["CLmax"]),
            "Oswald e": (res["oswald_e"], sl["polar_fit"]["e_oswald"])}
        log("[verify] " + ", ".join(f"{k}: {a:.3g} -> {b:.3g}" for k, (a, b) in out["verify"].items()))
    return out, sweep


def figure(out, sweep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    t = out["drag_c172"]["table"]
    ax[0, 0].barh([r["component"] for r in t][::-1], [1e4 * r["CD0"] for r in t][::-1], color="#1D4E89")
    ax[0, 0].set_xlabel("drag counts")
    ax[0, 0].set_title(f"C172-like drag buildup: CD0 {out['drag_c172']['CD0']:.4f} (quoted 0.027–0.031)")
    J = [p["J"] for p in sweep]
    ax[0, 1].plot(J, [p["CT"] for p in sweep], label="C_T")
    ax[0, 1].plot(J, [p["CP"] for p in sweep], label="C_P")
    a2 = ax[0, 1].twinx()
    a2.plot(J, [p["eta"] for p in sweep], color="#2B7A4B", label="η")
    a2.set_ylim(0, 1)
    a2.set_ylabel("η")
    pi = out["prop_inviscid"]
    ax[0, 1].set_title(f"MIL propeller (2 blades): design η {out['prop_consistency']['design_eta']:.3f}, "
                       f"BEMT {out['prop_consistency']['bemt_eta']:.3f}")
    ax[0, 1].set_xlabel("J"); ax[0, 1].legend(loc="upper left")
    ins = ax[0, 1].inset_axes([0.12, 0.08, 0.35, 0.35])
    ins.plot(pi["blades"], pi["eta"], "o-", ms=3)
    ins.axhline(pi["eta_ideal"], color="0.4", ls="--", lw=0.8)
    ins.set_title("no profile drag: η vs blades → disc", fontsize=7); ins.tick_params(labelsize=6)
    ins.set_xticks(pi["blades"][::2])
    rows = out["sizing_rows"]
    lab = [f"{r['case']}: {r['quantity']}" for r in rows]
    ax[1, 0].barh(lab[::-1], [r["diff_pct"] for r in rows][::-1], color="#C2571A")
    ax[1, 0].axvline(0, color="k", lw=0.8)
    for x in (-15, 15):
        ax[1, 0].axvline(x, color="0.6", ls=":", lw=0.8)
    ax[1, 0].set_xlabel("sized − reference (%)")
    ax[1, 0].set_title("Conceptual sizing against two real aircraft")
    sc = out["scissor_demo"]
    r = sc["rows"]
    ax[1, 1].plot([x["pct_fwd"] for x in r], [x["Sh_S"] for x in r], "o-", label="control (fwd)")
    ax[1, 1].plot([x["pct_aft"] for x in r], [x["Sh_S"] for x in r], "o-", label="stability (aft, SM 5 %)")
    ax[1, 1].plot(sc["cg_range"], [sc["current"]] * 2, "k-", lw=4, label="c.g. range, current tail")
    if sc["min_tail"]:
        ax[1, 1].axhline(sc["min_tail"], color="#2B7A4B", ls="--", lw=1, label=f"smallest tail {sc['min_tail']:.3f}")
    ax[1, 1].set_xlabel("c.g. (% MAC)"); ax[1, 1].set_ylabel("S_h / S"); ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("Scissor plot, demo trainer (vortex lattice)")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def report(out):
    pc = out["prop_consistency"]
    L = ["# Design tools: validation", "",
         "Reproduce with `python src/run.py validate_design_tools [--verify]`.", "",
         "## Drag buildup", "",
         f"- Cessna 172-like geometry (strut-braced high wing, fixed gear with wheel fairings, riveted metal, "
         f"8° upsweep): **CD0 {out['drag_c172']['CD0']:.4f}** against 0.027–0.031 quoted for the type "
         f"(UIUC/FlightGear model from Roskam: 0.031). Equivalent-skin-friction check (Cfe 0.0055): "
         f"{out['drag_c172']['cfe_check']:.4f}. Drag area f = {out['drag_c172']['f_m2']:.3f} m².",
         f"- NACA 0012 perimeter (wetted area per chord): {out['perimeter_naca0012']['exact']:.4f} from the "
         f"section, Raymer's rule {out['perimeter_naca0012']['raymer']:.4f}.",
         "- Fully turbulent, smooth: the fuselage term equals the aircraft model's fuselage drag to machine "
         "precision (test).", "",
         "## Propeller", "",
         f"- Adkins–Liebeck design re-analysed with BEMT (2 blades, 60 m/s, 2500 rpm, 60 kW): thrust "
         f"{pc['design_T']:.1f} / {pc['bemt_T']:.1f} N, power {pc['design_P'] / 1e3:.2f} / {pc['bemt_P'] / 1e3:.2f} kW, "
         f"efficiency {pc['design_eta']:.3f} / {pc['bemt_eta']:.3f} (actuator disc {pc['eta_ideal']:.3f}).",
         "- Without profile drag the design efficiency rises with the blade count towards the actuator-disc "
         f"value: " + ", ".join(f"{b}: {e:.3f}" for b, e in zip(out["prop_inviscid"]["blades"], out["prop_inviscid"]["eta"]))
         + f" (disc {out['prop_inviscid']['eta_ideal']:.3f}; the rest is swirl and hub).", "",
         "## Conceptual sizing against real aircraft", "",
         "| case | quantity | reference | sized | difference |", "|---|---|---|---|---|"]
    for r in out["sizing_rows"]:
        L.append(f"| {r['case']} | {r['quantity']} | {r['reference']:g} | {r['sized']:.1f} | {r['diff_pct']:+.1f} % |")
    L += ["", "C172: requirements from the 172S handbook class (2 occupants + 25 kg, 1100 km at 65 % power, 8000 ft, "
          "45 min reserve; stall 48 kt with flaps, take-off 497 m, landing 407 m over 50 ft), drag details as above. "
          "The largest difference is the empty mass: Raymer's general-aviation Class II equations put the C172 "
          "structure and systems 16 % lighter, and the lighter aircraft then needs less wing (the stall line fixes "
          "the wing loading) and less fuel.",
          "Velis Electro: 2 × 86 kg, 50 min at 170 km/h + 20 min reserve, 177 Wh/kg pack (24.8 kWh / 140 kg), 85 % "
          "usable, composite.", "",
          "Conceptual methods are usually quoted at ±10–15 %. Outside ±15 % here: "
          + (", ".join(f"{r['case']} {r['quantity'].split(' (')[0]} ({r['diff_pct']:+.0f} %)" for r in out["sizing_rows"]
                       if abs(r["diff_pct"]) > 15) or "none") + "."]
    if "verify" in out:
        L += ["", "## Sizing checked by the performance tool (default two-seat trainer)", "",
              "| quantity | sizing / required | performance tool |", "|---|---|---|"]
        for k, (a, b) in out["verify"].items():
            L.append(f"| {k} | {a:.3g} | {b:.3g} |")
        L += ["", "The performance tool uses the trimmed nonlinear polar (vortex lattice + CFD sections), the BEMT "
              "propeller designed for cruise and a time-integrated take-off. The take-off line is the one to watch: "
              "a fixed-pitch propeller pitched for cruise gives less static thrust than Roskam's statistics assume."]
    sc = out["scissor_demo"]
    L += ["", "## Weight and balance", "",
          f"- Demo trainer: c.g. range {sc['cg_range'][0]:.1f}–{sc['cg_range'][1]:.1f} % MAC (estimated component "
          f"list), tail S_h/S {sc['current']:.3f}; smallest tail for this range at the current wing position "
          f"{sc['min_tail']:.3f}.",
          "- Tests: neutral point of the scissor plot = the stability page's at the same tail; wing placement puts "
          "the loaded c.g. at the target to 0.05 % MAC; suggested gear meets tip-back, turnover, rotation and "
          "propeller clearance."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    out, sweep = run(a.verify)
    VAL.mkdir(exist_ok=True)
    figure(out, sweep, VAL / "design_tools_validation.png")
    (VAL / "design_tools_report.md").write_text(report(out))
    (VAL / "design_tools_validation.json").write_text(json.dumps(A._jsonable(out), indent=1))
    print(f"-> {VAL}/design_tools_validation.png, design_tools_report.md")


if __name__ == "__main__":
    main()
