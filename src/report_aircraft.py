"""
Demonstration + validation report of the aircraft tools (stability and
control, aero database, flight loads, performance) on the demo trainer.

    python report_aircraft.py        -> ../validation/aircraft_report.md + figures
"""
import json
import math
import shutil
from pathlib import Path

import numpy as np

import aero_database as D
import aircraft as A
import loads as Ld
import performance as P
import validate_aircraft as V

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "validation"
CASES = ROOT / "cases"
POL = CASES / "wing_polars"


def fig_stability(ac, rep, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    md, resp = rep["modes"], rep["responses"]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2))
    for key, mk, lab in (("_A_lon", "x", "longitudinal"), ("_A_lat", "o", "lateral")):
        ev = np.linalg.eigvals(np.array(md[key]))
        axs[0].plot(ev.real, ev.imag, mk, mfc="none", ms=8, label=lab)
    axs[0].axvline(0, color="k", lw=0.5); axs[0].set_xlabel("real (1/s)"); axs[0].set_ylabel("imag (rad/s)")
    axs[0].set_title("eigenvalues"); axs[0].legend(fontsize=8)
    e = resp["elevator"]
    for k, lab in (("alpha", "Δα (deg)"), ("theta", "Δθ (deg)"), ("q", "q (deg/s)")):
        axs[1].plot(e["t"], e[k], label=lab)
    axs[1].set_title(f"elevator step {e['input_deg']:+g}°")
    a = resp["aileron"]
    for k, lab in (("p", "p (deg/s)"), ("phi", "φ (deg)"), ("beta", "β (deg)")):
        axs[2].plot(a["t"], a[k], label=lab)
    axs[2].set_title(f"aileron step {a['input_deg']:+g}°")
    r = resp["rudder"]
    for k, lab in (("beta", "β (deg)"), ("r", "r (deg/s)"), ("phi", "φ (deg)")):
        axs[3].plot(r["t"], r[k], label=lab)
    axs[3].set_title(f"rudder doublet ±{r['input_deg']:g}°")
    for ax in axs[1:]:
        ax.set_xlabel("time (s)"); ax.legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.suptitle(f"{ac.name}: flight modes and control responses, {rep['speed']:g} m/s, {rep['altitude']:g} m")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def flap_report():
    """CFD polars of the plain-flapped NACA 2412 (25 % chord) vs clean; increments
    against thin-airfoil theory (lift) and Raymer's table (max lift)"""
    import airfoil_geom as ag
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wing as W
    cf, re = 0.25, 2.2e6
    th = math.acos(2 * cf - 1)
    tau = 1 - (th - math.sin(th)) / math.pi
    data = {}
    for d in (0, 10, 20):
        f = POL / f"{ag.section_stem(ag.flap_spec('naca2412', cf, d))}_Re{re:.3g}_M0.12_L1_free.json".replace("+", "")
        if f.exists():
            data[d] = [r for r in json.loads(f.read_text()) if r.get("Cl") is not None and np.isfinite(r["Cl"])]
    if len(data) < 2:
        return None
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    rows = []
    base = data[0]
    sp0 = W.SectionPolar.from_results(base)
    for d, res in data.items():
        a = [r["alpha"] for r in res]
        axs[0].plot(a, [r["Cl"] for r in res], "o-", ms=3, label="clean" if d == 0 else f"flap {d}°")
        axs[1].plot([r["Cd"] for r in res], [r["Cl"] for r in res], "o-", ms=3)
        axs[2].plot(a, [r["Cm"] for r in res], "o-", ms=3)
        if d:
            sp = W.SectionPolar.from_results(res)
            cl0 = float(sp(0.0)[0]) - float(sp0(0.0)[0])
            theory = 2 * math.pi * tau * math.radians(d)
            rows.append({"deflection": d, "dCl_alpha0": cl0, "thin_airfoil": theory,
                         "ratio": cl0 / theory, "clmax": sp.clmax, "dclmax": sp.clmax - sp0.clmax,
                         "raymer_dclmax": 0.9 * min(d / 40, 1), "a_clmax": sp.a_clmax})
    axs[0].set_xlabel("α (deg)"); axs[0].set_ylabel("Cl"); axs[0].legend(fontsize=8)
    axs[1].set_xlabel("Cd"); axs[1].set_ylabel("Cl"); axs[1].set_xlim(0, 0.08)
    axs[2].set_xlabel("α (deg)"); axs[2].set_ylabel("Cm, c/4")
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.suptitle(f"NACA 2412 with a {100 * cf:g} % plain flap, Re {re:.2g}, free transition (RANS)")
    fig.tight_layout(); fig.savefig(VAL / "flap_sections_demo.png", dpi=110); plt.close(fig)
    L = ["# Plain-flap sections (CFD)", "",
         "Geometry: `airfoil_geom.deflect_flap` (spec `naca2412@flap=25c20`): camber line rotated aft of the "
         "hinge at 75 % chord, smooth 4 %-chord sealed-hinge blend, chord reference unchanged. Inviscid check "
         "(panel method): ΔCl = 1.06–1.09 × thin-airfoil flap theory for 5–30° (`tests/test_flap.py`).", "",
         f"RANS (Spalart–Allmaras, e^N transition), Re {re:.2g}:", "",
         "| flap | ΔCl at α 0 | thin airfoil 2πτδ | ratio (viscous effectiveness) | Cl_max | ΔCl_max | Raymer ΔCl_max | α at Cl_max |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['deflection']}° | {r['dCl_alpha0']:.3f} | {r['thin_airfoil']:.3f} | {r['ratio']:.2f} | "
                 f"{r['clmax']:.3f} | {r['dclmax']:+.3f} | {r['raymer_dclmax']:.3f} | {r['a_clmax']:g}° |")
    L += ["", f"Clean section: Cl_max {sp0.clmax:.3f} at {sp0.a_clmax:g}°.", "",
          "The viscous lift increment falls below thin-airfoil theory as the deflection grows (boundary "
          "layer on the flap's upper surface), the usual plain-flap behaviour; the performance tool uses "
          "these CFD Cl_max increments for takeoff and landing when the polars exist.", "",
          "Validated range: 0–20°. At 25° the lift at α 0 is already lower than at 20° (flow separated on "
          "the flap) and cold starts at higher angles diverge; at 30° no angle converged, also after "
          "widening the hinge blend (mesh non-orthogonality 58° → 44°) and with coarse-mesh sequencing. "
          "Larger deflections need a slotted-flap model (multi-element tool) or unsteady RANS.", "",
          "![flap sections](flap_sections_demo.png)"]
    (VAL / "flap_report.md").write_text("\n".join(L) + "\n")
    return rows


def main():
    VAL.mkdir(exist_ok=True)
    ac = A.demo_aircraft()
    ac.save(CASES / "aircraft" / "demo_trainer.json")
    # ---- validation
    val = V.main()
    # ---- stability
    rep = A._jsonable(A.stability_report(ac, 50.0, 1000.0, POL))
    fig_stability(ac, rep, VAL / "aircraft_stability_demo.png")
    # ---- database
    db = D.aircraft_database(ac, D.parse_range("-6:24:1"), D.parse_range("-10:10:2"), [0.15], 1000.0,
                             {"elevator": [-20, -10, 10, 20], "aileron": [-15, 15], "rudder": [-20, 20],
                              "flap": [10, 20, 30]}, POL, nonlinear=True, log=lambda *a: None)
    dbo = CASES / "db_demo_trainer_report"
    D.write_aircraft_outputs(ac, db, dbo)
    shutil.copy(dbo / "database.png", VAL / "aircraft_database_demo.png")
    # ---- loads
    ld = Ld.run(ac, 60.0, "utility", 0.0, None, POL, fuel_mass=80.0, log=lambda *a: None)
    lo = CASES / "loads_demo_trainer_report"
    Ld.write_outputs(ld, lo)
    shutil.copy(lo / "vn_diagram.png", VAL / "aircraft_loads_vn_demo.png")
    shutil.copy(lo / "span_loads.png", VAL / "aircraft_loads_span_demo.png")
    # ---- performance
    pr = P.Propulsion(75.0, 0.80, 1.8, 300.0, 60.0)
    pf = A._jsonable(P.run(ac, pr, 0.0, POL, 80.0, 0.0, 10.0, 20.0, log=lambda *a: None))
    po = CASES / "perf_demo_trainer_report"
    po.mkdir(parents=True, exist_ok=True)
    (po / "performance.json").write_text(json.dumps(pf, indent=1))
    P.plots(pf, po)
    shutil.copy(po / "performance.png", VAL / "aircraft_performance_demo.png")
    write_md(ac, val, rep, db, ld, pf)
    print(flap_report())


def _f(v, fmt=".3f"):
    return "–" if v is None else format(v, fmt)


def write_md(ac, val, rep, db, ld, pf):
    d, t, md = rep["derivatives"], rep["trim"], rep["modes"]
    vs = val["aircraft_vs_aerosandbox"]
    L = [f"# Aircraft tools: stability and control, aero database, flight loads, performance",
         "", "Generated by `src/report_aircraft.py` (demo trainer, 750 kg, generic). App pages: Aircraft model, "
         "Stability and control, Aero database, Flight loads, Performance.", "",
         "## Method in one paragraph",
         "One vortex lattice for all lifting surfaces (horseshoe vortices, control-surface hinges and edges on "
         "panel edges, sideslip and body rates in the free stream, Kutta–Joukowski forces with the full local "
         "velocity), camber, profile drag, section moments and stall from the CFD section polars "
         "(decambering on every strip), Munk slender-body fuselage moments and friction drag. Derivatives by "
         "central differences; flight modes from the small-perturbation equations; loads by trimming every "
         "V–n case and integrating the strip forces with inertia relief; performance from the trimmed "
         "nonlinear drag polar and a piston/fixed-pitch propeller model.", "",
         "## Validation (`validate_aircraft.py`, `aircraft_validation.json`)", "",
         "Same geometry in AeroSandbox's vortex-lattice method (independent code; symmetric sections, no "
         "fuselage, inviscid):", "",
         "| quantity | AeroLab | AeroSandbox | difference |", "|---|---|---|---|"]
    for k in ("CL", "Cm", "CLa", "Cma", "x_np", "CYb", "Clb", "Cnb", "Clp"):
        a, b = vs[k]["aerolab"], vs[k]["aerosandbox"]
        diff = f"{100 * (a - b) / abs(b):+.1f} %" if b else "–"
        L.append(f"| {k} | {a:.4f} | {b:.4f} | {diff} |")
    if "cranked_vs_aerosandbox" in val:
        ck = val["cranked_vs_aerosandbox"]
        L += ["", "Multi-panel surfaces (kinks): wing with a 2 m constant-chord centre panel and a swept, "
              "tapered outer panel with a dihedral break and 3° washout; tail and fin with a kink:", "",
              "| quantity | AeroLab | AeroSandbox | difference |", "|---|---|---|---|"]
        for k in ("CL", "Cm", "CLa", "Cma", "x_np", "CYb", "Clb", "Cnb", "Clp"):
            a, b = ck[k]["aerolab"], ck[k]["aerosandbox"]
            diff = f"{100 * (a - b) / abs(b):+.1f} %" if b else "–"
            L.append(f"| {k} | {a:.4f} | {b:.4f} | {diff} |")
    pr, yr = val["pitch_rate_thin_airfoil_limit"], val["yaw_rate_fin_classical"]
    L += ["", "Rate derivatives (AeroSandbox's q and r derivatives use another definition: its CL_q is "
          "1.5 CL_a on a plain high-aspect-ratio wing, where theory gives CL_q = CL_a):", "",
          f"- AR 30 wing pitching about c/4: CL_q {pr['CL_q']:.3f} vs CL_a {pr['CL_a']:.3f} (theory: equal); "
          f"Cm_q {pr['Cm_q']:.3f} vs 2-D −π/4 = {pr['Cm_q_2D']:.3f}",
          f"- fin yaw-rate terms vs tail-volume formulas: CY_r {yr['CY_r']:.3f} vs {yr['classical_CY_r']:.3f}, "
          f"Cn_r {yr['Cn_r']:.3f} vs {yr['classical_Cn_r_fin']:.3f} (fin only)",
          f"- wing alone vs the wing tool (Phase 6): CL {val['wing_vs_wingtool']['CL_aircraft']:.4f} vs "
          f"{val['wing_vs_wingtool']['CL_wing_tool']:.4f}, CDi {val['wing_vs_wingtool']['CDi_aircraft_nearfield']:.5f} "
          f"(near field) vs {val['wing_vs_wingtool']['CDi_wing_tool_trefftz']:.5f} (Trefftz)",
          f"- modes vs approximations: phugoid ζ {val['phugoid']['zeta']:.4f} vs CD/(√2 CL) "
          f"{val['phugoid']['approx_zeta']:.4f}; short period ω_n {val['short_period']['omega_n']:.2f} vs "
          f"{val['short_period']['approx_omega_n']:.2f} rad/s, ζ {val['short_period']['zeta']:.3f} vs "
          f"{val['short_period']['approx_zeta']:.3f}",
          "- Lamb's apparent-mass factor (fineness 6): test in `tests/test_aircraft.py`",
          "", "## Stability and control (demo, 50 m/s, 1000 m)", "",
          f"Trim α {t['alpha_trim']:.2f}°, elevator {t['control_trim']:+.2f}°, CL {t['CL']:.3f}, L/D {t['L_D']:.1f}. "
          f"Neutral point x {d['x_np']:.3f} m, static margin {100 * d['static_margin']:.1f} % MAC, "
          f"dε/dα {d['deps_da']:.3f}.", "",
          "| derivative | value | | derivative | value |", "|---|---|---|---|---|"]
    pairs = [("CL_a", "CY_b"), ("Cm_a", "Cl_b"), ("CL_q", "Cn_b"), ("Cm_q", "Cl_p"), ("CL_adot", "Cn_p"),
             ("Cm_adot", "Cl_r"), ("CD_a", "Cn_r")]
    for a, b in pairs:
        L.append(f"| {a} | {d[a]:.4f} | | {b} | {d[b]:.4f} |")
    L += ["", "| control | CL_δ | Cl_δ | Cm_δ | Cn_δ | CY_δ | Ch_δ | Ch_α |", "|---|---|---|---|---|---|---|---|"]
    for n, c in d["controls"].items():
        L.append(f"| {n} | {c['CL']:.4f} | {c['Cl']:.4f} | {c['Cm']:.4f} | {c['Cn']:.4f} | {c['CY']:.4f} | "
                 f"{c['Ch_d']:.3f} | {_f(c['Ch_a'])} |")
    L += ["", "| mode | ω_n (rad/s) | ζ | period / time constant (s) | Level 1 (MIL-F-8785C, Class I, Cat. B) |",
          "|---|---|---|---|---|"]
    for n, m in md.items():
        if n.startswith("_"):
            continue
        tc = m.get("period_s") or m.get("time_constant_s") or m.get("t_double_s")
        L.append(f"| {n} | {_f(m.get('omega_n'))} | {_f(m.get('zeta'))} | {_f(tc, '.2f')} | "
                 f"{'yes' if m.get('level1') else 'no'} ({m.get('criterion', '')}) |")
    L += ["", "![stability](aircraft_stability_demo.png)", "",
          "## Aero database", "",
          f"{len(db['rows'])} states (α −6…24° × β −10…10°, elevator, aileron, rudder, flap deflections), "
          f"nonlinear sections, in {db['meta']['time_s']:.0f} s. Files: database.xlsx, jsbsim_aero.xml "
          "(complete `<aerodynamics>` section), aircraft.avl + aircraft.mass, CSV/JSON.", "",
          "![database](aircraft_database_demo.png)", "",
          "## Flight loads (utility category, V_C 60 m/s EAS, 80 kg fuel)", "",
          f"n+ {ld['vn']['n_pos']:.2f}, n− {ld['vn']['n_neg']:.2f}; V_S1 {ld['vn']['Vs1']:.1f}, V_A {ld['vn']['Va']:.1f}, "
          f"V_D {ld['vn']['Vd']:.1f} m/s; gust factor K_g {ld['vn']['Kg']:.3f}.", "",
          "| case | V_EAS (m/s) | n | root shear (kN) | root bending (kN m) | root torsion (kN m) |",
          "|---|---|---|---|---|---|"]
    for c in ld["cases"]:
        L.append(f"| {c['name']} | {c['V_eas']:.1f} | {c['n']:+.2f} | {c['root_shear'] / 1e3:.2f} | "
                 f"{c['root_bending'] / 1e3:.2f} | {c['root_torsion'] / 1e3:.2f} |")
    L += ["", "Check: with inertia relief off, the root shear of the half wing equals half the wing's lift "
          "(test). 1 g cruise: bending / shear puts the load centroid at ≈ 42 % semi-span (elliptic: 42.4 %).", "",
          "![vn](aircraft_loads_vn_demo.png)", "", "![span loads](aircraft_loads_span_demo.png)", "",
          "## Performance (75 kW, fixed-pitch propeller, 80 kg fuel, sea level)", "",
          f"Trimmed CL_max {pf['CLmax_trimmed']:.2f}; polar fit CD0 {pf['polar_fit']['CD0']:.4f}, Oswald e "
          f"{pf['polar_fit']['e_oswald']:.2f}. Stall {pf['V_stall']:.1f} m/s, V_max {pf['V_max']:.1f} m/s, best "
          f"climb {pf['best_rate']['roc']:.2f} m/s at {pf['best_rate']['V']:.0f} m/s, service ceiling "
          f"{_f(pf['ceiling_service'], '.0f')} m, best glide {pf['glide']['best_ratio']:.1f}, range "
          f"{pf['best_range']['range_km']:.0f} km at {pf['best_range']['V']:.0f} m/s. Takeoff to 15 m "
          f"{_f(pf['takeoff']['total'], '.0f')} m (ground roll {_f(pf['takeoff']['ground_roll'], '.0f')} m), landing "
          f"from 15 m {pf['landing']['total']:.0f} m. Flap CL_max increments: takeoff "
          f"{pf['takeoff']['flap_dCLmax_estimate']:.2f} ({pf['takeoff']['flap_source']}), landing "
          f"{pf['landing']['flap_dCLmax_estimate']:.2f} ({pf['landing']['flap_source']}).", "",
          "![performance](aircraft_performance_demo.png)", "",
          "## Limits",
          "- Vortex lattice: attached flow, no vortex lift; post-stall is strip theory (trend). Wake straight along x.",
          "- Fuselage: Munk slender-body moments; no wing-body lift carry-over, no effect of the wing's vertical "
          "position on C_l_β, no propeller slipstream or thrust-line effects.",
          "- Hinge moments from the inviscid lattice (no boundary-layer effect on the control; real values are "
          "usually smaller in magnitude).",
          "- Modes: thrust independent of speed, Ixz = 0, inertias estimated unless entered.",
          "- Loads: simplified CS-23 envelope (check the negative-gust and V_D lines against the applicable "
          "regulation), symmetric manoeuvres only (no rolling or fin load cases yet).",
          "- Performance: Breguet at constant speed and altitude; flap increments from Raymer unless CFD polars "
          "of the flapped section exist."]
    (VAL / "aircraft_report.md").write_text("\n".join(L) + "\n")
    print(f"-> {VAL / 'aircraft_report.md'}")


if __name__ == "__main__":
    main()
