"""
Component drag buildup: parasite drag CD0 of a whole aircraft, component by
component, in the form engineers check by hand (Raymer ch. 12, Hoerner).

    python drag_buildup.py ../cases/aircraft/demo_trainer.json --speed 60 --alt 1500 --gear fixed_faired

Method
  * Every component:  CD0_i = Cf_i FF_i Q_i S_wet,i / S_ref
    - Cf: flat-plate skin friction at the component's Reynolds number, mixed
      laminar-turbulent (Prandtl-Schlichting: turbulent from the leading edge
      minus the laminar-run correction), turbulent part
      0.455 / (log10 Re)^2.58 / (1 + 0.144 M^2)^0.65, laminar 1.328 / sqrt(Re).
      Surface roughness caps the Reynolds number at the cutoff value
      38.21 (l / k)^1.053 (Raymer eq. 12.28).
    - Lifting surfaces strip by strip over the EXPOSED span (outside the
      fuselage): local chord Reynolds number, wetted area from the section's
      real perimeter, form factor
      [1 + 0.6 / (x/c)_m (t/c) + 100 (t/c)^4] [1.34 M^0.18 cos(L_m)^0.28]
      with t/c and its chordwise position (x/c)_m from the section shape and
      L_m the sweep of the maximum-thickness line.
    - Fuselage: wetted area of the body of revolution, FF = 1 + 60/f^3 + f/400.
    - Interference factors Q (Raymer): wing 1.0, conventional tail 1.04,
      fuselage 1.0 (editable).
  * Miscellaneous drag (D/q, m^2, divided by S_ref):
    - fixed landing gear per wheel and leg from Raymer's Table 12.6 (wheel and
      tyre 0.25, with fairing 0.13, round strut 0.30, streamline strut 0.05,
      flat spring leg 1.40 on the frontal area);
    - fuselage aft upsweep 3.83 u^2.5 A_max and base drag
      (0.139 + 0.419 (M - 0.161)^2) A_base (Raymer eqs. 12.36, 12.37);
    - engine cooling: momentum loss of the cooling air
      D = m_dot V k_loss, m_dot = q_heat P / (c_p dT) (air-cooled piston
      default: heat 0.7 P, dT 50 K, k_loss 0.5; electric: heat (1 - eta) P);
    - leakage and protuberances: a percentage of the component total
      (Raymer: 5-10 % for propeller aircraft; default 7 %).
  * Cross-check: Raymer's equivalent skin-friction method CD0 = Cfe S_wet / S_ref
    (light single-engine aircraft Cfe = 0.0055).
  * Oswald factor estimate for the drag polar: e = 1.78 (1 - 0.045 A^0.68) - 0.64
    (straight wings; swept: 4.61 (1 - 0.045 A^0.68) cos(L_le)^0.15 - 3.1).

The aircraft model (aircraft.py) already contains the lifting surfaces'
profile drag (CFD section polars) and the fuselage friction; what it does not
contain is the miscellaneous part, so `cd_extra_for_model` returns the value
for the model's "extra drag" field: gear + cooling + upsweep + base + other +
leakage.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import wing as W

# surface roughness k (m), Raymer Table 12.5
ROUGHNESS = {"camouflage paint on aluminium": 1.015e-5, "smooth paint": 0.634e-5,
             "production sheet metal": 0.405e-5, "polished sheet metal": 0.152e-5,
             "smooth moulded composite": 0.052e-5}
# landing gear drag D/q on the frontal area (Raymer Table 12.6)
GEAR_WHEEL = {"wheel and tyre": 0.25, "streamline wheel and tyre": 0.18, "wheel with fairing": 0.13}
GEAR_LEG = {"round strut": 0.30, "streamline strut": 0.05, "flat spring leg": 1.40, "none": 0.0}
GEAR_PRESETS = {
    "retractable": None,
    "fixed_faired": {"wheel": "wheel with fairing", "leg": "streamline strut"},
    "fixed_bare": {"wheel": "wheel and tyre", "leg": "flat spring leg"},
    "fixed_struts": {"wheel": "wheel and tyre", "leg": "round strut"},
}
CFE = {"light aircraft, single engine": 0.0055, "light aircraft, twin engine": 0.0045,
       "transport / bomber": 0.0030, "propeller seaplane": 0.0065, "sailplane": 0.0030}


# ----------------------------------------------------------------------
# skin friction
# ----------------------------------------------------------------------
def cf_turbulent(re, mach=0.0):
    re = max(float(re), 1e3)
    return 0.455 / math.log10(re) ** 2.58 / (1 + 0.144 * mach**2) ** 0.65


def cf_laminar(re):
    return 1.328 / math.sqrt(max(float(re), 1.0))


def cf_mixed(re, x_tr=0.0, mach=0.0):
    """average flat-plate friction with laminar flow over the first x_tr
    of the length (Prandtl-Schlichting): turbulent over the whole length
    minus the turbulent friction over the laminar run, plus its laminar
    friction"""
    re = max(float(re), 1e3)
    x_tr = min(max(float(x_tr), 0.0), 1.0)
    cf = cf_turbulent(re, mach)
    if x_tr > 0:
        rt = x_tr * re
        cf += x_tr * (cf_laminar(rt) - cf_turbulent(rt, mach))
        if x_tr >= 1.0:
            cf = cf_laminar(re)
    return max(cf, cf_laminar(re))


def re_cutoff(length, k):
    """roughness cutoff Reynolds number (subsonic), Raymer eq. 12.28"""
    if not k or k <= 0:
        return math.inf
    return 38.21 * (length / k) ** 1.053


# ----------------------------------------------------------------------
# section geometry
# ----------------------------------------------------------------------
_SEC_CACHE = {}


def section_shape(spec):
    """(perimeter / chord, t/c, x/c of max thickness) of a section"""
    key = str(spec)
    if key in _SEC_CACHE:
        return _SEC_CACHE[key]
    try:
        import airfoil_geom as ag
        n = ag.sharp_outline(spec)
        per = float(np.linalg.norm(np.diff(n, axis=0), axis=1).sum())
        i = int(np.argmin(n[:, 0]))
        up, lo = n[:i + 1][::-1], n[i:]
        x = np.linspace(0.0, 1.0, 801)
        t = np.interp(x, up[:, 0], up[:, 1]) - np.interp(x, lo[:, 0], lo[:, 1])
        j = int(np.argmax(t))
        out = (per, float(t[j]), float(x[j]))
    except Exception:                                   # noqa: BLE001  unknown section: 12 %, Raymer area
        out = (1.977 + 0.52 * 0.12, 0.12, 0.30)
    _SEC_CACHE[key] = out
    return out


def form_factor_wing(tc, xm, mach, sweep_m_deg):
    return (1 + 0.6 / max(xm, 0.1) * tc + 100 * tc**4) * \
        (1.34 * max(mach, 1e-3) ** 0.18 * math.cos(math.radians(sweep_m_deg)) ** 0.28)


def form_factor_body(fineness):
    f = max(fineness, 1.0)
    return 1 + 60 / f**3 + f / 400


def oswald(ar, sweep_le_deg=0.0):
    """Raymer eqs. 12.48 / 12.49"""
    if abs(sweep_le_deg) > 30:
        e = 4.61 * (1 - 0.045 * ar**0.68) * math.cos(math.radians(sweep_le_deg)) ** 0.15 - 3.1
    else:
        e = 1.78 * (1 - 0.045 * ar**0.68) - 0.64
    return float(min(max(e, 0.5), 0.95))


# ----------------------------------------------------------------------
# components
# ----------------------------------------------------------------------
def hidden_semispan(s, fus):
    """part of a surface's (semi)span buried inside the fuselage (m)"""
    if fus.length <= 0 or fus.diameter <= 0:
        return 0.0
    xs = s.x + 0.5 * s.root_chord
    r = float(fus.radius(np.array([xs]))[0])
    if r <= 0:
        return 0.0
    if s.vertical:
        dz = s.z - fus.z                              # fin root above the axis
        return float(min(max(r - dz, 0.0) if dz >= 0 else r, 0.95 * s.semi))
    dz = abs(s.z - fus.z)
    if dz >= r or abs(s.y) > r:
        return 0.0
    return float(min(math.sqrt(r * r - dz * dz) - abs(s.y), 0.95 * s.semi))


def surface_drag(s, fus, rho, mu, V, mach, x_tr=0.1, k=0.634e-5, Q=1.0, S_ref=1.0, n=40):
    """CD0 contribution of one lifting surface (exposed part, both sides)"""
    h = hidden_semispan(s, fus)
    eta0 = h / s.semi
    # strips over the exposed (semi)span
    e = eta0 + (1 - eta0) * (0.5 - 0.5 * np.cos(np.linspace(0, math.pi, n + 1)))
    em = 0.5 * (e[1:] + e[:-1])
    dy = np.diff(e) * s.semi
    c = s.chord(em)
    st = s.stations()
    secs = []
    for eta in em:                                     # root -> tip section blend (shape factors)
        per_r, tc_r, xm_r = section_shape(s.airfoil)
        if s.tip_airfoil and s.tip_airfoil != s.airfoil:
            per_t, tc_t, xm_t = section_shape(s.tip_airfoil)
            secs.append(tuple(a + (b - a) * eta for a, b in zip((per_r, tc_r, xm_r), (per_t, tc_t, xm_t))))
        else:
            secs.append((per_r, tc_r, xm_r))
    secs = np.array(secs)
    # sweep of the max-thickness line from the leading-edge line, per segment
    xle = s.x_le(em)
    xm_line = xle + secs[:, 2] * c
    span_pos = em * s.semi
    sw = np.degrees(np.arctan(np.gradient(xm_line, span_pos))) if len(em) > 2 else np.zeros_like(em)
    sw = np.where(np.isfinite(sw), sw, s.sweep)
    re = rho * V * c / mu
    re_eff = np.minimum(re, [re_cutoff(ci, k) for ci in c])
    cf = np.array([cf_mixed(r, x_tr, mach) for r in re_eff])
    ff = np.array([form_factor_wing(t, xm, mach, a) for t, xm, a in zip(secs[:, 1], secs[:, 2], sw)])
    sides = 2 if s.symmetric else 1
    swet_strip = sides * secs[:, 0] * c * dy
    swet = float(swet_strip.sum())
    D_q = float((cf * ff * Q * swet_strip).sum())
    w = swet_strip / max(swet, 1e-12)
    return {"name": s.name, "kind": "fin" if s.vertical else ("wing" if Q == 1.0 else "surface"),
            "S_wet": swet, "S_exposed": float(sides * (c * dy).sum()), "hidden_semispan": h,
            "Re": float((re * w).sum()), "Re_cutoff_active": bool(np.any(re_eff < re)),
            "Cf": float((cf * w).sum()), "FF": float((ff * w).sum()), "Q": Q,
            "t_c": float((secs[:, 1] * w).sum()), "laminar_fraction": x_tr,
            "D_q": D_q, "CD0": D_q / S_ref}


def fuselage_drag(fus, rho, mu, V, mach, x_tr=0.05, k=0.634e-5, Q=1.0, S_ref=1.0):
    vol, swet = fus.volume_and_wetted()
    if swet <= 0:
        return None
    re = rho * V * fus.length / mu
    re_eff = min(re, re_cutoff(fus.length, k))
    cf = cf_mixed(re_eff, x_tr, mach)
    fr = fus.length / fus.diameter
    ff = form_factor_body(fr)
    D_q = cf * ff * Q * swet
    return {"name": "fuselage", "kind": "body", "S_wet": swet, "S_exposed": None, "Re": re,
            "Re_cutoff_active": re_eff < re, "Cf": cf, "FF": ff, "Q": Q, "fineness": fr,
            "laminar_fraction": x_tr, "D_q": D_q, "CD0": D_q / S_ref}


def gear_drag(gear):
    """landing gear D/q (m^2). gear: dict with preset or explicit wheels"""
    if gear and gear.get("items"):                    # explicit list: kind, count, width, length
        rows = []
        for it in gear["items"]:
            kind = it["kind"]
            coef = GEAR_WHEEL.get(kind, GEAR_LEG.get(kind, 0.0))
            rows.append({"item": f"{int(it.get('count', 1))} × {it.get('item', kind)}",
                         "D_q": int(it.get("count", 1)) * coef * float(it["width"]) * float(it["length"])})
        return float(sum(r["D_q"] for r in rows)), rows
    if not gear or gear.get("type", "retractable") == "retractable":
        return 0.0, []
    pre = GEAR_PRESETS.get(gear.get("type"), {}) or {}
    wheel = gear.get("wheel", pre.get("wheel", "wheel and tyre"))
    leg = gear.get("leg", pre.get("leg", "round strut"))
    n_w = int(gear.get("wheels", 3))
    d_t, w_t = float(gear.get("tyre_diameter", 0.38)), float(gear.get("tyre_width", 0.13))
    leg_len, leg_w = float(gear.get("leg_length", 0.5)), float(gear.get("leg_width", 0.04))
    rows = [{"item": f"{n_w} × {wheel}", "D_q": n_w * GEAR_WHEEL[wheel] * d_t * w_t}]
    if GEAR_LEG.get(leg, 0) > 0:
        rows.append({"item": f"{n_w} × {leg}", "D_q": n_w * GEAR_LEG[leg] * leg_len * leg_w})
    return float(sum(r["D_q"] for r in rows)), rows


def cooling_drag(V, power_kw=0.0, kind="piston", heat_ratio=None, dT=50.0, k_loss=0.5, eta_motor=0.93, rho=1.225):
    """cooling-air momentum loss as D/q (m^2) at speed V and density rho"""
    if power_kw <= 0 or kind == "none":
        return 0.0
    if heat_ratio is None:
        heat_ratio = {"piston": 0.7, "electric": 1 - eta_motor, "turboprop": 0.05}.get(kind, 0.7)
    mdot = heat_ratio * power_kw * 1e3 / (1005.0 * dT)
    D = mdot * V * k_loss
    return D / (0.5 * rho * V**2)


# ----------------------------------------------------------------------
def buildup(ac, speed=60.0, altitude=0.0, laminar=None, roughness="smooth paint", Q=None,
            gear=None, upsweep_deg=0.0, power_kw=0.0, engine="piston", cooling=None,
            leakage_pct=7.0, other=None, cfe_class="light aircraft, single engine"):
    """parasite drag of the aircraft. laminar: {"wing": 0.1, "tail": 0.1,
    "fuselage": 0.05} (fraction of chord / length); Q: interference factors by
    the same keys; other: list of {"item", "D_q"} (canopy, antennas, ...)"""
    rho, mu, a, _, _ = W.isa(altitude)
    mach = speed / a
    S = ac.S
    lam = {"wing": 0.10, "tail": 0.10, "fuselage": 0.05} | dict(laminar or {})
    QQ = {"wing": 1.0, "tail": 1.04, "fuselage": 1.0} | dict(Q or {})
    k = ROUGHNESS.get(roughness, roughness if isinstance(roughness, (int, float)) else 0.634e-5)
    comps = []
    for i, s in enumerate(ac.surfaces):
        key = "wing" if i == 0 else "tail"
        comps.append(surface_drag(s, ac.fuselage, rho, mu, speed, mach, lam[key], k, QQ[key], S))
    fd = fuselage_drag(ac.fuselage, rho, mu, speed, mach, lam["fuselage"], k, QQ["fuselage"], S)
    if fd:
        comps.append(fd)
    comp_total = sum(c["CD0"] for c in comps)
    misc = []
    g_dq, g_rows = gear_drag(gear)
    if g_dq > 0:
        misc.append({"item": "landing gear", "D_q": g_dq, "detail": g_rows})
    fu = ac.fuselage
    if fu.length > 0 and fu.diameter > 0:
        A_max = math.pi * fu.diameter**2 / 4
        if upsweep_deg > 0:
            misc.append({"item": f"fuselage upsweep {upsweep_deg:g}°",
                         "D_q": 3.83 * math.radians(upsweep_deg) ** 2.5 * A_max})
        A_base = A_max * fu.tail_ratio**2
        if A_base > 1e-6:
            misc.append({"item": "fuselage base", "D_q": (0.139 + 0.419 * (mach - 0.161) ** 2) * A_base})
    cool = cooling_drag(speed, power_kw, engine, rho=rho, **(cooling or {})) if power_kw > 0 else 0.0
    if cool > 0:
        misc.append({"item": f"engine cooling ({engine})", "D_q": cool})
    for o in other or []:
        if o.get("D_q", 0):
            misc.append({"item": str(o.get("item", "other")), "D_q": float(o["D_q"])})
    for m in misc:
        m["CD0"] = m["D_q"] / S
    misc_total = sum(m["CD0"] for m in misc)
    leak = leakage_pct / 100.0 * (comp_total + misc_total)
    misc.append({"item": f"leakage and protuberances ({leakage_pct:g} %)", "D_q": leak * S, "CD0": leak})
    total = comp_total + misc_total + leak
    swet = sum(c["S_wet"] for c in comps)
    cfe = CFE.get(cfe_class, 0.0055)
    w = ac.wing
    ar = w.span**2 / w.area
    sweep_le = math.degrees(math.atan((w.x_le(1.0) - w.x_le(0.0)) / max(w.semi, 1e-9)))
    e = oswald(ar, sweep_le)
    for c in comps + misc:
        c["share_pct"] = 100 * c["CD0"] / total if total > 0 else 0.0
    return {"aircraft": ac.name, "speed": speed, "altitude": altitude, "mach": mach,
            "S_ref": S, "q": 0.5 * rho * speed**2, "components": comps, "misc": misc,
            "CD0_components": comp_total, "CD0_misc": misc_total + leak, "CD0": total,
            "f_m2": total * S, "S_wet_total": swet, "S_wet_ratio": swet / S,
            "cfe_check": {"class": cfe_class, "Cfe": cfe, "CD0": cfe * swet / S},
            "oswald_e": e, "aspect_ratio": ar, "K": 1 / (math.pi * ar * e),
            "settings": {"laminar": lam, "Q": QQ, "roughness": roughness, "k": k, "gear": gear,
                         "upsweep_deg": upsweep_deg, "power_kw": power_kw, "engine": engine,
                         "leakage_pct": leakage_pct}}


def cd_extra_for_model(res):
    """the part of CD0 the vortex-lattice aircraft model does not contain
    (it has the surfaces' profile drag and the fuselage friction): misc items
    + leakage"""
    return float(res["CD0_misc"])


def polar(res, CL):
    CL = np.asarray(CL, float)
    return res["CD0"] + res["K"] * CL**2


def table(res):
    rows = []
    for c in res["components"]:
        rows.append({"component": c["name"], "S_wet (m²)": c["S_wet"], "Re": c["Re"], "Cf": c["Cf"],
                     "FF": c["FF"], "Q": c["Q"], "CD0": c["CD0"], "share %": c["share_pct"]})
    for m in res["misc"]:
        rows.append({"component": m["item"], "S_wet (m²)": None, "Re": None, "Cf": None, "FF": None, "Q": None,
                     "CD0": m["CD0"], "share %": m["share_pct"]})
    return rows


def table_text(res):
    """table() formatted as text for display (blank where a column does not apply)"""
    fmt = {"S_wet (m²)": "{:.2f}", "Re": "{:.3g}", "Cf": "{:.5f}", "FF": "{:.3f}", "Q": "{:.2f}",
           "CD0": "{:.5f}", "share %": "{:.1f}"}
    return [{k: (fmt[k].format(v) if k in fmt and v is not None else ("" if v is None else v))
             for k, v in r.items()} for r in table(res)]


def plots(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = table(res)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    names = [r["component"] for r in rows]
    vals = [1e4 * r["CD0"] for r in rows]
    a1.barh(names[::-1], vals[::-1], color="#1D4E89")
    a1.set_xlabel("drag counts (1e-4)")
    a1.set_title(f"CD0 = {res['CD0']:.4f}  (f = {res['f_m2']:.3f} m²)")
    cl = np.linspace(0, 1.5, 50)
    a2.plot(polar(res, cl), cl, label=f"CD0 + CL²/(π A e), e {res['oswald_e']:.2f}")
    a2.set_xlabel("CD"); a2.set_ylabel("CL"); a2.legend(); a2.grid(alpha=0.3)
    for a in (a1,):
        a.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(Path(out) / "drag_buildup.png", dpi=110)
    plt.close(fig)


def main(argv=None):
    import aircraft as A
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aircraft")
    ap.add_argument("--speed", type=float, default=60.0)
    ap.add_argument("--alt", type=float, default=0.0)
    ap.add_argument("--gear", default="fixed_faired", choices=list(GEAR_PRESETS))
    ap.add_argument("--power", type=float, default=0.0, help="engine power (kW) for cooling drag")
    ap.add_argument("--engine", default="piston", choices=["piston", "electric", "turboprop", "none"])
    ap.add_argument("--roughness", default="smooth paint", choices=list(ROUGHNESS))
    ap.add_argument("--laminar-wing", type=float, default=0.10)
    ap.add_argument("--leakage", type=float, default=7.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    ac = A.Aircraft.load(a.aircraft)
    res = buildup(ac, a.speed, a.alt, laminar={"wing": a.laminar_wing, "tail": a.laminar_wing},
                  roughness=a.roughness, gear={"type": a.gear}, power_kw=a.power, engine=a.engine,
                  leakage_pct=a.leakage)
    for r in table(res):
        print(f"  {r['component']:38s} CD0 {r['CD0']:.5f}  {r['share %']:5.1f} %")
    print(f"CD0 {res['CD0']:.4f}, f {res['f_m2']:.3f} m², S_wet/S {res['S_wet_ratio']:.2f}, "
          f"Cfe check {res['cfe_check']['CD0']:.4f}, Oswald e {res['oswald_e']:.2f}")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "drag_buildup.json").write_text(json.dumps(A._jsonable(res), indent=1))
        plots(res, out)


if __name__ == "__main__":
    main()
