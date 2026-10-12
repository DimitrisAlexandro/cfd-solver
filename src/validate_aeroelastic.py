"""
Validation of the aeroelastic tools.

    python validate_aeroelastic.py        # ~2 min

Writes validation/aeroelastic_validation.png and aeroelastic_report.md.

  1. Unsteady vortex-ring lattice -> Theodorsen (2-D) on an aspect-ratio-40
     wing, plunge and pitch, refinement 8 / 16 / 32 chordwise panels.
  2. Goland wing (Goland 1945; the standard flutter benchmark): half span
     6.096 m, chord 1.8288 m, e.a. 33 %, c.g. 43 % chord, 35.71 kg/m,
     I_ea 8.64 kg m, EI 9.77e6, GJ 0.987e6 N m^2. Strip theory p-k against
     Goland's 137.2 m/s / 70.7 rad/s (sea level); the 3D lattice.
  3. Demo trainer: aileron flutter against the aileron balance; divergence,
     elastic lift and roll effectiveness; flexible against rigid loads.
"""

import json
import math
from pathlib import Path

import numpy as np

import aeroelastic as AE
import aircraft as A
import wing_structure as WS

VAL = Path(__file__).resolve().parents[1] / "validation"


def theodorsen_check():
    rows = []
    b = 0.5
    s = A.Surface("w", span=40.0, root_chord=1.0, tip_chord=1.0, airfoil="naca0012")
    for nc in (8, 16, 32):
        lat = AE.VortexRingLattice(s, ns=30, nc=nc, wake_chords=60)
        mid = np.abs(lat.eta_p) < 0.04
        width = sum(abs(lat.B[i, 1] - lat.A[i, 1]) for i in np.where(mid & (lat.row == 0))[0])
        for k in (0.05, 0.1, 0.2, 0.4, 0.6, 1.0):
            kap = k / b
            C = WS.theodorsen(k)
            for mode, z, dz, hd, al in (("plunge", np.ones(lat.n), np.zeros(lat.n), -1.0, 0.0),
                                        ("pitch", -(lat.col[:, 0] - 0.5), -np.ones(lat.n), 0.0, 1.0)):
                F = lat.forces(kap, lat.solve(kap, (1j * kap * z + dz)[:, None]))[:, 0]
                L = (F[mid] * lat.nrm[mid, 2]).sum() / width
                Lt = 2 * (math.pi * b**2 * (-(kap**2) * hd + 1j * kap * al) +
                          2 * math.pi * b * C * (al + 1j * kap * hd + 0.5 * b * 1j * kap * al))
                rows.append({"nc": nc, "k": k, "mode": mode, "L": L, "L_th": Lt, "err": abs(L - Lt) / abs(Lt)})
    return rows


def goland(rho=1.225):
    L, c, m = 6.096, 1.8288, 35.71
    props = {"EI": [9.77e6] * 2, "GJ": [0.987e6] * 2, "m": [m] * 2, "S": [m * 0.1 * c] * 2, "I": [8.64] * 2}
    b = AE.BeamModel([0, L], props, 0.0, n_el=30)
    b.x_ea_fn = lambda y: np.full_like(np.asarray(y, float), 0.33 * c)
    b.x_le_fn = lambda y: np.zeros_like(np.asarray(y, float))
    b.chord_fn = lambda y: np.full_like(np.asarray(y, float), c)
    surf = A.Surface("goland", span=2 * L, root_chord=c, tip_chord=c, airfoil="naca0012")
    b.ac, b.surface = A.Aircraft("g", [surf], A.Fuselage()), 0
    w, phi = b.modes(6)
    Mm, Km = phi.T @ b.M_full @ phi, phi.T @ b.K_full @ phi
    ks = np.r_[0.0, 0.02, 0.05, 0.08, 0.12, 0.17, 0.25, 0.35, 0.5, 0.7, 1.0, 1.5]
    Vs = np.linspace(60, 230, 171)
    rs = AE.pk(Mm, Km, AE.GAFTable(ks, AE.gaf_strip(b, phi, ks, c / 2)), Vs, rho, c / 2)
    out = {"frequencies_rad_s": w.tolist(), "strip": rs}
    lat = AE.VortexRingLattice(surf, ns=16, nc=12, wake_chords=25)
    out["uvlm"] = AE.pk(Mm, Km, AE.GAFTable(ks, AE.gaf(b, lat, phi, ks, c / 2)), Vs, rho, c / 2)
    return out


def demo():
    import loads as LD
    ac = A.demo_aircraft()
    res = LD.run(ac, 60.0, "normal", 0.0, None, None, log=lambda *a: None)
    st = WS.size(ac, res)
    ail = {"freq_hz": 15.0, "mass_per_area": 4.0, "cg_aft_hinge": 0.30, "gyration": 0.30}
    Vmax = 2 * res["vn"]["Vd"]
    r, beam, lat = AE.analyse(ac, st, ail, Vmax=Vmax, log=lambda *a: None)
    sweep = AE.mass_balance_sweep(ac, st, ail, np.linspace(-0.15, 0.40, 12), Vmax)
    ac.structure = WS.to_aircraft(st)
    rf = LD.run(ac, 60.0, "normal", 0.0, None, None, log=lambda *a: None, flexible=True)
    flex = [{"case": a["name"], "rigid": a["root_bending"], "flexible": b["root_bending"],
             "tip_twist_deg": b["elastic_tip_twist_deg"]} for a, b in zip(res["cases"], rf["cases"])]
    return {"analysis": r, "mass_balance": sweep, "V_D": res["vn"]["Vd"], "flex": flex}


def figure(th, gl, dm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    for nc, ls in ((8, ":"), (16, "--"), (32, "-")):
        for mode, col in (("plunge", "#1D4E89"), ("pitch", "#C2571A")):
            r = [x for x in th if x["nc"] == nc and x["mode"] == mode]
            ax[0, 0].plot([x["k"] for x in r], [100 * x["err"] for x in r], ls, color=col, label=f"{mode}, {nc} panels")
    ax[0, 0].set_xlabel("reduced frequency k"); ax[0, 0].set_ylabel("|L − L_Theodorsen| / |L_T| (%)")
    ax[0, 0].set_title("Unsteady lattice vs Theodorsen (AR 40, mid-span)"); ax[0, 0].legend(fontsize=7)
    for key, lab, col in (("strip", "strip theory (Theodorsen)", "#1D4E89"), ("uvlm", "3D vortex-ring lattice", "#C2571A")):
        r = gl[key]
        G = np.array(r["g"])
        for j in range(min(G.shape[1], 3)):
            ax[0, 1].plot(r["V"], G[:, j], color=col, lw=1.2, label=lab if j == 0 else None)
            ax[0, 2].plot(r["V"], np.array(r["f"])[:, j] * 2 * math.pi, color=col, lw=1.2, label=lab if j == 0 else None)
    ax[0, 1].axhline(0, color="k", lw=0.8); ax[0, 1].axvline(137.2, color="0.5", ls="--", lw=0.8)
    ax[0, 1].text(138, 0.05, "Goland 137.2 m/s", fontsize=8)
    ax[0, 1].set_ylim(-0.4, 0.2); ax[0, 1].set_xlabel("V (m/s)"); ax[0, 1].set_ylabel("g")
    ax[0, 1].set_title("Goland wing, sea level: damping"); ax[0, 1].legend(fontsize=8)
    ax[0, 2].axhline(70.7, color="0.5", ls="--", lw=0.8)
    ax[0, 2].set_xlabel("V (m/s)"); ax[0, 2].set_ylabel("ω (rad/s)"); ax[0, 2].set_title("Goland wing: frequency")
    mb = dm["mass_balance"]
    x = [p["cg_aft_hinge"] for p in mb]
    y = [p["V_flutter"] if math.isfinite(p["V_flutter"]) else np.nan for p in mb]
    ax[1, 0].plot(x, y, "o-")
    for xi, yi in zip(x, y):
        if not np.isfinite(yi):
            ax[1, 0].plot(xi, 2 * dm["V_D"], "^", color="#2B7A4B")
    ax[1, 0].axhline(1.2 * dm["V_D"], color="#C2571A", ls="--", lw=1, label="1.2 V_D")
    ax[1, 0].set_xlabel("aileron c.g. aft of hinge (aileron chords)"); ax[1, 0].set_ylabel("flutter speed (m/s)")
    ax[1, 0].set_title("Demo trainer: aileron mass balance (▲ = none below 2 V_D)"); ax[1, 0].legend(fontsize=8)
    a = dm["analysis"]
    ax[1, 1].plot(a["elastic_lift"]["V_eas"], 100 * np.array(a["elastic_lift"]["ratio"]), label="lift due to α")
    if a.get("reversal"):
        ax[1, 1].plot(a["reversal"]["V_curve"], 100 * np.array(a["reversal"]["effectiveness"]), label="roll due to aileron")
    ax[1, 1].axhline(100, color="0.6", lw=0.8, ls=":"); ax[1, 1].axvline(dm["V_D"], color="0.5", lw=0.8, ls="--")
    ax[1, 1].set_xlabel("V EAS (m/s)"); ax[1, 1].set_ylabel("elastic / rigid (%)")
    ax[1, 1].set_title(f"Static aeroelasticity (divergence {a['divergence']['V_eas']:.0f} m/s)"); ax[1, 1].legend(fontsize=8)
    fl = dm["flex"]
    xs = np.arange(len(fl))
    ax[1, 2].bar(xs - 0.2, [f["rigid"] / 1e3 for f in fl], 0.4, label="rigid")
    ax[1, 2].bar(xs + 0.2, [f["flexible"] / 1e3 for f in fl], 0.4, label="flexible")
    ax[1, 2].set_xticks(xs, [f["case"].split(",")[0][:10] for f in fl], rotation=60, fontsize=7)
    ax[1, 2].set_ylabel("root bending (kN m)"); ax[1, 2].set_title("Flight loads, rigid vs flexible wing")
    ax[1, 2].legend(fontsize=8)
    for a_ in ax.ravel():
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def report(th, gl, dm):
    gs, gu = gl["strip"]["flutter"][0], gl["uvlm"]["flutter"][0]
    e16 = max(x["err"] for x in th if x["nc"] == 16 and x["k"] <= 0.6)
    e32 = max(x["err"] for x in th if x["nc"] == 32 and x["k"] <= 0.6)
    a = dm["analysis"]
    fp = a["flutter_points"][0] if a["flutter_points"] else None
    L = ["# Aeroelasticity: validation", "", "Reproduce with `python src/validate_aeroelastic.py`.", "",
         "## Unsteady aerodynamics", "",
         f"Vortex-ring lattice with an oscillating wake against Theodorsen's 2-D lift (aspect ratio 40, mid-span), "
         f"plunge and pitch, k ≤ 0.6: largest error {100 * e16:.1f} % with 16 chordwise panels, {100 * e32:.1f} % with 32 "
         "(first-order convergence). The steady limit equals the aircraft vortex lattice (test).", "",
         "## Goland wing (sea level)", "",
         "| | flutter speed | flutter frequency |", "|---|---|---|",
         "| Goland (1945), strip theory | 137.2 m/s | 70.7 rad/s |",
         f"| this tool, strip theory (Theodorsen) + p-k | {gs['V']:.1f} m/s | {2 * math.pi * gs['f_hz']:.1f} rad/s |",
         f"| this tool, 3D vortex-ring lattice + p-k | {gu['V']:.1f} m/s | {2 * math.pi * gu['f_hz']:.1f} rad/s |", "",
         f"Modes (coupled bending-torsion beam FE): {', '.join(f'{v:.1f}' for v in gl['frequencies_rad_s'][:4])} rad/s "
         "(uncoupled: 49.5 and 87.1 rad/s exactly, test). The 3D lattice includes the tip loss that strip theory "
         "ignores: the flutter speed rises; the frequency stays close.", "",
         "## Demo trainer", "",
         f"- Aileron 15 Hz, c.g. 0.30 aileron chord aft of the hinge: flutter "
         + (f"**{fp['V']:.0f} m/s** ({fp['family']}, {fp['f_hz']:.1f} Hz) < 1.2 V_D = {1.2 * dm['V_D']:.0f} m/s"
            if fp else "none") + " — the classic unbalanced-aileron flutter.",
         "- Mass-balance sweep (flutter speed against the aileron c.g.):",
         "", "| aileron c.g. aft of hinge | flutter speed |", "|---|---|"]
    for p in dm["mass_balance"]:
        L.append(f"| {p['cg_aft_hinge']:+.2f} | {'%.1f m/s' % p['V_flutter'] if math.isfinite(p['V_flutter']) else 'none below 2 V_D'} |")
    L += ["", f"- Divergence {a['divergence']['V_eas']:.0f} m/s EAS; aileron reversal "
          + (f"{a['reversal']['V_eas']:.0f} m/s EAS" if a.get("reversal") and math.isfinite(a["reversal"]["V_eas"])
             else f"above {2 * dm['V_D']:.0f} m/s EAS (2 V_D)") + ".",
          "- Flexible against rigid wing loads (root bending, kN m): "
          + ", ".join(f"{f['case'].split(',')[0]} {f['rigid'] / 1e3:.2f} → {f['flexible'] / 1e3:.2f}" for f in dm["flex"]) + ".",
          "", "Limits: incompressible, clamped root, wing only, linear, no structural damping."]
    return "\n".join(L) + "\n"


def main():
    th = theodorsen_check()
    gl = goland()
    dm = demo()
    VAL.mkdir(exist_ok=True)
    figure(th, gl, dm, VAL / "aeroelastic_validation.png")
    (VAL / "aeroelastic_report.md").write_text(report(th, gl, dm))
    (VAL / "aeroelastic_validation.json").write_text(json.dumps(A._jsonable(
        {"theodorsen": [{**x, "L": [x["L"].real, x["L"].imag], "L_th": [x["L_th"].real, x["L_th"].imag]} for x in th],
         "goland": {"frequencies_rad_s": gl["frequencies_rad_s"], "strip_flutter": gl["strip"]["flutter"],
                    "uvlm_flutter": gl["uvlm"]["flutter"]},
         "demo": {"mass_balance": dm["mass_balance"], "flex": dm["flex"], "V_D": dm["V_D"]}}), indent=1))
    print(f"-> {VAL}/aeroelastic_validation.png, aeroelastic_report.md")


if __name__ == "__main__":
    main()
