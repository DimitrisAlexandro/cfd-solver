"""
Aircraft performance from the trimmed aerodynamic model (aircraft.py) and a
piston-propeller propulsion model.

    python performance.py ../cases/aircraft/demo_trainer.json --power 75 --fuel 80 --out ../cases/perf_demo

Method
  * Trimmed drag polar CD(CL) of the aircraft (elevator trim, nonlinear
    CFD sections up to the trimmed CL_max) at the given altitude.
  * Propulsion (simple model; --propulsion uses propeller.py instead: BEMT
    propeller, fixed pitch or constant speed, piston / turboprop / electric):
    shaft power P(h) = P0 (sigma - (1 - sigma)/7.55) (Gagg-
    Ferrar); fixed-pitch propeller with eta(V) = eta_max (2 r - r^2),
    r = V / V_design, i.e. thrust falling linearly from 2 eta_max P/V_design
    at rest to eta_max P/V_design at the design speed, capped by momentum-
    theory static thrust (figure of merit 0.7). Fuel flow = BSFC x P.
  * Level flight: power required D V vs available eta_p P; maximum speed;
    rate of climb (P_av - D V)/W and climb angle (T - D)/W, best-rate and
    best-angle speeds; service (0.5 m/s) and absolute ceilings.
  * Range and endurance at constant speed and altitude (Breguet, propeller):
    R = eta_p / (BSFC g) (L/D) ln(W0/W1), E = R / V, against speed.
  * Glide: best ratio and minimum sink.
  * Takeoff (to 15 m) and landing (from 15 m), Raymer's segments: ground
    roll integrated with rolling / braking friction, rotation, transition
    arc and climb; flap CL_max increment for takeoff / landing from
    Raymer's plain-flap estimate (Delta cl_max 0.9 at 40 deg, linear,
    x 0.9 S_flapped/S cos(sweep)) - an estimate, shown as such.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import aircraft as A
import wing as W


class Propulsion:
    """piston engine + fixed-pitch propeller.
    eta(V) = eta_max (2 r - r^2), r = V / V_design (r <= 1): thrust
    eta P / V falls linearly from 2 eta_max P / V_design at rest to
    eta_max P / V_design at the design speed (the usual fixed-pitch
    behaviour); above the design speed eta drops (overspeed). Static thrust
    capped by momentum theory with a figure of merit."""

    def __init__(self, power_kw=75.0, eta=0.80, prop_diameter=1.8, bsfc=300.0, v_design=60.0, fom=0.7):
        self.P0 = power_kw * 1e3
        self.eta, self.D, self.bsfc, self.Vd, self.fom = eta, prop_diameter, bsfc, v_design, fom
        self.kind = "piston"

    def power(self, h):
        sig = W.isa(h)[0] / 1.225
        return self.P0 * max(sig - (1 - sig) / 7.55, 0.0)

    def efficiency(self, V):
        r = V / self.Vd
        return self.eta * (2 * r - r * r) if r <= 1 else self.eta * max(1 - 0.8 * (r - 1) ** 2, 0.2)

    def thrust(self, V, h):
        P = self.power(h)
        rho = W.isa(h)[0]
        A_disc = math.pi * self.D**2 / 4
        T_static = self.fom * (2 * rho * A_disc * P**2) ** (1 / 3)
        r = V / self.Vd
        T_prop = self.eta * P / self.Vd * (2 - r) if r <= 1 else self.efficiency(V) * P / V
        return min(T_prop, T_static)

    def fuel_flow(self, h):
        """kg/s at full power"""
        return self.bsfc / 3.6e9 * self.power(h)


def trimmed_polar(ac, altitude, polar_dir, n_pts=14, control="elevator"):
    """CD(CL) trimmed, nonlinear sections; CL_max trimmed from where trim fails
    or the lift stops rising"""
    rows = []
    CLs = np.r_[np.linspace(0.15, 1.2, 8), np.linspace(1.3, 2.4, 12)]
    a0 = 2.0
    for CL in CLs:
        V = math.sqrt(2 * ac.mass * A.G0 / (W.isa(altitude)[0] * ac.S * CL))
        an = A.Analysis(ac, V, altitude, polar_dir)
        try:
            t = an.trim(CL_req=CL, control=control, nonlinear=True, alpha0=a0)
        except Exception:                                   # noqa: BLE001
            break
        ok = abs(t["CL"] - CL) < 1e-3 and abs(t["Cm"]) < 1e-3 and t["converged"]
        if not ok or (rows and t["alpha_trim"] > 25):
            break
        if rows and t["alpha_trim"] < rows[-1]["alpha"]:     # past the maximum
            break
        rows.append({"CL": float(t["CL"]), "CD": float(t["CD"]), "alpha": float(t["alpha_trim"]),
                     "elevator": float(t["control_trim"]), "V": V, "L_D": float(t["CL"] / t["CD"]),
                     "stall_fraction": float(np.mean(t["strips"]["stalled"]))})
        a0 = t["alpha_trim"]
    # refine the trimmed CL_max between the last trimmed and the first failed CL
    if rows and len(rows) < len(CLs):
        lo = rows[-1]["CL"]
        hi = CLs[len(rows)]
        for _ in range(4):
            CL = 0.5 * (lo + hi)
            V = math.sqrt(2 * ac.mass * A.G0 / (W.isa(altitude)[0] * ac.S * CL))
            an = A.Analysis(ac, V, altitude, polar_dir)
            try:
                t = an.trim(CL_req=CL, control=control, nonlinear=True, alpha0=rows[-1]["alpha"])
                ok = abs(t["CL"] - CL) < 1e-3 and abs(t["Cm"]) < 1e-3 and t["converged"] and \
                    t["alpha_trim"] > rows[-1]["alpha"] and t["alpha_trim"] < 25
            except Exception:                               # noqa: BLE001
                ok = False
            if ok:
                rows.append({"CL": float(t["CL"]), "CD": float(t["CD"]), "alpha": float(t["alpha_trim"]),
                             "elevator": float(t["control_trim"]), "V": V, "L_D": float(t["CL"] / t["CD"]),
                             "stall_fraction": float(np.mean(t["strips"]["stalled"]))})
                lo = CL
            else:
                hi = CL
    return rows


def flap_increment(ac, deflection):
    """Raymer's plain-flap CL_max increment (estimate)"""
    if deflection <= 0:
        return 0.0
    w = ac.wing
    fl = [c for c in w.controls if c.name == "flap"]
    if not fl:
        return 0.0
    c = fl[0]
    # flapped wing area (both sides)
    e = np.linspace(c.eta0, c.eta1, 50)
    Sf = 2 * np.trapezoid(w.chord(e), e * w.semi)
    dcl = 0.9 * min(deflection / 40.0, 1.0)
    return 0.9 * dcl * Sf / ac.S * math.cos(math.radians(w.sweep))


def flap_increment_best(ac, deflection, polar_dir=None, mach=0.0):
    """(Delta CL_max, source): section increment from the CFD polars of the
    flapped wing section when they are cached (airfoil@flap=..., same flap
    chord), else Raymer's table; 3D conversion 0.9 S_flapped/S cos(sweep)
    in both cases"""
    w = ac.wing
    fl = [c for c in w.controls if c.name == "flap"]
    if deflection <= 0 or not fl:
        return 0.0, "none"
    est = flap_increment(ac, deflection)
    if not polar_dir:
        return est, "Raymer estimate"
    import airfoil_geom as ag
    import design_wing_sizing as WS
    cf = fl[0].cf
    clean = WS.cached_polars(w.airfoil, polar_dir, mach)
    flapped = WS.cached_polars(ag.flap_spec(w.airfoil, cf, deflection), polar_dir, mach)
    if not (clean and flapped):
        return est, "Raymer estimate"
    re_f, pf = flapped[0]
    re_c, pc = min(clean, key=lambda t: abs(math.log(t[0] / re_f)))
    e = np.linspace(fl[0].eta0, fl[0].eta1, 50)
    Sf = 2 * np.trapezoid(w.chord(e), e * w.semi)
    dcl = pf.clmax - pc.clmax
    return 0.9 * dcl * Sf / ac.S * math.cos(math.radians(w.sweep)), f"CFD section polars (Re {re_f:.3g})"


def run(ac, prop, altitude=0.0, polar_dir=None, fuel=60.0, reserve=0.0, to_flap=10.0, ld_flap=30.0,
        mu_roll=0.04, mu_brake=0.40, h_obs=15.0, log=print):
    g = A.G0
    Wt = ac.mass * g
    log("[performance] trimmed drag polar")
    pol = trimmed_polar(ac, altitude, polar_dir)
    if len(pol) < 4:
        raise RuntimeError("trim failed for most lift coefficients: check the elevator and the c.g.")
    CL = np.array([p["CL"] for p in pol])
    CD = np.array([p["CD"] for p in pol])
    CLmax = float(CL.max())
    k2, k1, k0 = np.polyfit(CL[CL < 0.8 * CLmax], CD[CL < 0.8 * CLmax], 2)

    def cd_of(cl):
        cl = np.asarray(cl, float)
        v = np.interp(cl, CL, CD, right=np.nan)
        lo = cl < CL[0]
        return np.where(lo, k0 + k1 * cl + k2 * cl**2, v)

    def level(h):
        rho = W.isa(h)[0]
        Vs = math.sqrt(2 * Wt / (rho * ac.S * CLmax))
        V = np.linspace(1.02 * Vs, 3.6 * Vs, 300)
        cl = 2 * Wt / (rho * V**2 * ac.S)
        cd = cd_of(cl)
        D = 0.5 * rho * V**2 * ac.S * cd
        T = np.array([prop.thrust(v, h) for v in V])
        Pav = T * V
        return {"V": V, "cl": cl, "cd": cd, "D": D, "T": T, "P_req": D * V, "P_av": Pav,
                "roc": (Pav - D * V) / Wt, "gamma": np.degrees(np.arcsin(np.clip((T - D) / Wt, -1, 1))),
                "Vs": Vs, "rho": rho}
    L0 = level(altitude)
    i = int(np.argmax(L0["roc"]))
    j = int(np.argmax(L0["gamma"]))
    exc = L0["P_av"] - L0["P_req"]
    k = np.where(exc >= 0)[0]
    Vmax = float(L0["V"][k[-1]]) if len(k) else float("nan")
    # ceilings
    hs, rocs = [], []
    for h in np.arange(0, 9001, 250.0):
        L = level(h)
        hs.append(h)
        rocs.append(float(L["roc"].max()))
        if rocs[-1] < 0:
            break
    hs, rocs = np.array(hs), np.array(rocs)

    def ceiling(lim):
        m = np.where(rocs < lim)[0]
        if not len(m):
            return None
        a = m[0]
        return float(np.interp(lim, [rocs[a], rocs[a - 1]], [hs[a], hs[a - 1]])) if a > 0 else 0.0
    # range / endurance (Breguet, propeller), mean conditions
    LD = L0["cl"] / L0["cd"]
    if hasattr(prop, "eta_at_thrust"):
        # BEMT propeller: efficiency at the thrust needed for level flight (part throttle)
        ok = np.isfinite(L0["D"])
        Vg = np.linspace(L0["V"][ok].min(), L0["V"][ok].max(), 12)
        Dg = np.interp(Vg, L0["V"][ok], L0["D"][ok])
        eg = [prop.eta_at_thrust(v, d, altitude) for v, d in zip(Vg, Dg)]
        eta_V = np.interp(L0["V"], Vg, eg)
    else:
        eta_V = np.array([prop.efficiency(v) for v in L0["V"]])
    if getattr(prop, "kind", "piston") == "electric":
        # electric: constant mass; R = E_shaft eta_p (L/D) / W
        e = prop.engine
        E_shaft = e.battery_kwh * 3.6e6 * e.usable * e.eta_motor
        R = E_shaft * eta_V * LD / Wt
    else:
        c_kg = prop.bsfc / 3.6e9                                   # kg per J (shaft)
        W1 = (ac.mass - (fuel - reserve)) * g
        lnr = math.log(Wt / W1) if fuel > reserve else 0.0
        R = eta_V / (c_kg * g) * LD * lnr
    E = R / L0["V"]
    ir, ie = int(np.nanargmax(R)), int(np.nanargmax(E))
    # glide (power off)
    sink = L0["V"] * L0["cd"] / L0["cl"]
    ig, iS = int(np.nanargmax(LD)), int(np.nanargmin(sink))
    # takeoff / landing
    rho = W.isa(altitude)[0]
    an = A.Analysis(ac, 30.0, altitude, polar_dir)
    has_flap = "flap" in an.lat.ctrl_panels

    def ground_CL_CD(flap):
        r = an.solve(alpha=0.0, controls={"flap": flap} if has_flap else None)
        return r["CL"], r["CD"]

    def takeoff():
        dCL, src = flap_increment_best(ac, to_flap, polar_dir, an.mach) if has_flap else (0.0, "none")
        CLmax_to = CLmax + dCL
        Vs = math.sqrt(2 * Wt / (rho * ac.S * CLmax_to))
        V_lof = 1.1 * Vs
        CLg, CDg = ground_CL_CD(to_flap)
        V, s, t, dt = 0.0, 0.0, 0.0, 0.02
        while V < V_lof and t < 300:
            q = 0.5 * rho * V**2
            Ff = prop.thrust(V, altitude) - q * ac.S * CDg - mu_roll * max(Wt - q * ac.S * CLg, 0.0)
            a = Ff / ac.mass
            if a <= 0:
                return {"ground_roll": None, "total": None, "V_lof": V_lof, "note": "thrust too low"}
            V += a * dt
            s += V * dt
            t += dt
        s_rot = 1.0 * V_lof                                       # 1 s rotation (light aircraft, Raymer)
        V_tr = 1.15 * Vs
        cl_tr = 2 * Wt / (rho * V_tr**2 * ac.S)
        D_tr = 0.5 * rho * V_tr**2 * ac.S * float(cd_of(min(cl_tr, 0.95 * CLmax)))
        gam = math.asin(max(min((prop.thrust(V_tr, altitude) - D_tr) / Wt, 0.5), 1e-4))
        Rtr = V_tr**2 / (g * 0.2)                                 # n = 1.2 in the transition
        h_tr = Rtr * (1 - math.cos(gam))
        if h_tr > h_obs:
            s_tr = math.sqrt(Rtr**2 - (Rtr - h_obs) ** 2)
            s_cl = 0.0
        else:
            s_tr = Rtr * math.sin(gam)
            s_cl = (h_obs - h_tr) / math.tan(gam)
        return {"V_stall": Vs, "V_lof": V_lof, "CLmax": CLmax_to, "flap_dCLmax_estimate": dCL, "flap_source": src,
                "ground_roll": s, "rotation": s_rot, "transition": s_tr, "climb": s_cl,
                "total": s + s_rot + s_tr + s_cl, "time_ground_s": t}

    def landing():
        dCL, src = flap_increment_best(ac, ld_flap, polar_dir, an.mach) if has_flap else (0.0, "none")
        CLmax_l = CLmax + dCL
        Vs = math.sqrt(2 * Wt / (rho * ac.S * CLmax_l))
        Va, Vf, Vtd = 1.3 * Vs, 1.23 * Vs, 1.15 * Vs
        th = math.radians(3.0)
        Rf = Vf**2 / (g * 0.2)
        hf = Rf * (1 - math.cos(th))
        s_app = (h_obs - hf) / math.tan(th)
        s_fl = Rf * math.sin(th)
        s_free = 1.0 * Vtd                                        # 1 s free roll
        CLg, CDg = ground_CL_CD(ld_flap)
        V, s, dt = Vtd, 0.0, 0.02
        while V > 0.5:
            q = 0.5 * rho * V**2
            Ff = -q * ac.S * CDg - mu_brake * max(Wt - q * ac.S * CLg, 0.0)
            V += Ff / ac.mass * dt
            s += max(V, 0) * dt
        return {"V_stall": Vs, "V_approach": Va, "V_touchdown": Vtd, "CLmax": CLmax_l,
                "flap_dCLmax_estimate": dCL, "flap_source": src, "approach": s_app, "flare": s_fl, "free_roll": s_free,
                "ground_roll": s, "total": s_app + s_fl + s_free + s}
    TO, LDG = takeoff(), landing()
    res = {
        "aircraft": ac.name, "altitude": altitude, "mass": ac.mass, "fuel": fuel,
        "propulsion": (prop.describe() if hasattr(prop, "describe") else
                       {"model": "simple", "power_kw": prop.P0 / 1e3, "eta": prop.eta, "prop_diameter": prop.D,
                        "bsfc": prop.bsfc, "v_design": prop.Vd}),
        "eta_prop_cruise": eta_V.tolist(),
        "polar": pol, "polar_fit": {"CD0": k0, "k1": k1, "k2": k2, "e_oswald": 1 / (math.pi * ac.b**2 / ac.S * k2)},
        "CLmax_trimmed": CLmax, "V_stall": L0["Vs"], "V_max": Vmax,
        "best_rate": {"V": float(L0["V"][i]), "roc": float(L0["roc"][i])},
        "best_angle": {"V": float(L0["V"][j]), "gamma_deg": float(L0["gamma"][j])},
        "ceiling_service": ceiling(0.508), "ceiling_absolute": ceiling(1e-6),
        "best_range": {"V": float(L0["V"][ir]), "range_km": float(R[ir] / 1e3), "L_D": float(LD[ir])},
        "best_endurance": {"V": float(L0["V"][ie]), "endurance_h": float(E[ie] / 3600)},
        "glide": {"best_ratio": float(LD[ig]), "V_best_glide": float(L0["V"][ig]),
                  "min_sink": float(sink[iS]), "V_min_sink": float(L0["V"][iS])},
        "takeoff": TO, "landing": LDG,
        "curves": {k_: L0[k_].tolist() for k_ in ("V", "P_req", "P_av", "roc", "gamma", "D", "T")}
        | {"range_km": (R / 1e3).tolist(), "endurance_h": (E / 3600).tolist(), "sink": sink.tolist(),
           "ceiling_h": hs.tolist(), "ceiling_roc": rocs.tolist()},
    }
    return res


def plots(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = res["curves"]
    V = np.array(c["V"])
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    axs[0, 0].plot(V, np.array(c["P_req"]) / 1e3, label="required (D V)")
    axs[0, 0].plot(V, np.array(c["P_av"]) / 1e3, label="available (η P)")
    axs[0, 0].set_ylabel("power (kW)"); axs[0, 0].legend(fontsize=8)
    axs[0, 1].plot(V, c["roc"]); axs[0, 1].set_ylabel("rate of climb (m/s)")
    axs[0, 2].plot(c["ceiling_roc"], c["ceiling_h"], "o-", ms=3); axs[0, 2].axvline(0.508, color="0.6", lw=0.8)
    axs[0, 2].set_xlabel("max rate of climb (m/s)"); axs[0, 2].set_ylabel("altitude (m)")
    axs[1, 0].plot(V, c["range_km"]); axs[1, 0].set_ylabel("range (km)")
    axs[1, 1].plot(V, c["endurance_h"]); axs[1, 1].set_ylabel("endurance (h)")
    axs[1, 2].plot(V, c["sink"]); axs[1, 2].set_ylabel("sink rate, power off (m/s)")
    for ax in axs.ravel():
        if ax is not axs[0, 2]:
            ax.set_xlabel("true airspeed (m/s)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{res['aircraft']}: performance at {res['altitude']:g} m, {res['mass']:g} kg")
    fig.tight_layout(); fig.savefig(Path(out) / "performance.png", dpi=110); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aircraft")
    ap.add_argument("--power", type=float, default=75.0, help="engine power at sea level (kW)")
    ap.add_argument("--eta", type=float, default=0.80)
    ap.add_argument("--prop-diameter", type=float, default=1.8)
    ap.add_argument("--bsfc", type=float, default=300.0, help="g/kWh")
    ap.add_argument("--v-design", type=float, default=60.0, help="propeller design (cruise) speed, m/s")
    ap.add_argument("--fuel", type=float, default=60.0, help="usable fuel (kg)")
    ap.add_argument("--reserve", type=float, default=0.0)
    ap.add_argument("--alt", type=float, default=0.0)
    ap.add_argument("--to-flap", type=float, default=10.0)
    ap.add_argument("--ld-flap", type=float, default=30.0)
    ap.add_argument("--mu-roll", type=float, default=0.04, help="rolling friction (0.04 paved, 0.08 grass)")
    ap.add_argument("--mu-brake", type=float, default=0.40, help="braking friction (0.4 dry paved, 0.2 grass)")
    ap.add_argument("--polar-dir", default="../cases/wing_polars")
    ap.add_argument("--propulsion", default=None,
                    help="JSON file: engine + BEMT propeller (propeller.make_propulsion); replaces the simple model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ac = A.Aircraft.load(a.aircraft)
    if a.propulsion:
        import propeller as PR
        prop = PR.make_propulsion(json.loads(Path(a.propulsion).read_text()), a.polar_dir)
    else:
        prop = Propulsion(a.power, a.eta, a.prop_diameter, a.bsfc, a.v_design)
    res = run(ac, prop, a.alt, a.polar_dir, a.fuel, a.reserve, a.to_flap, a.ld_flap, a.mu_roll, a.mu_brake)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "performance.json").write_text(json.dumps(A._jsonable(res), indent=1))
    plots(res, out)
    print(f"V_stall {res['V_stall']:.1f} m/s, V_max {res['V_max']:.1f} m/s, best ROC {res['best_rate']['roc']:.2f} m/s "
          f"at {res['best_rate']['V']:.1f}, ceiling {res['ceiling_service']}, range {res['best_range']['range_km']:.0f} km, "
          f"takeoff {res['takeoff']['total']:.0f} m, landing {res['landing']['total']:.0f} m")
    print(f"-> {out}/performance.json, performance.png")


if __name__ == "__main__":
    main()
