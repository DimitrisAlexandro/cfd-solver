"""
Conceptual aircraft sizing: requirements -> constraint diagram -> mission
fuel (or battery) -> Class I and Class II weights -> converged take-off mass
-> a complete aircraft model (geometry, masses and c.g. positions, landing
gear, propulsion) that the analysis pages use directly.

    python sizing.py requirements.json --out ../cases/sizing_demo
    python sizing.py --demo --out ../cases/sizing_demo

Method (single-engine or twin propeller aircraft, piston / turboprop / electric)
  * Constraint diagram, power-to-weight P/W against wing loading W/S (SI),
    thrust from T = eta P / V:
      - stall speed (landing flap)        W/S <= 1/2 rho0 V_s^2 CL_max,L
      - landing distance over 15 m        Roskam (CS-23): s_L [ft] = 0.5136 V_s,L^2 [kt]
      - take-off distance over 15 m       Roskam: s_TOG = 4.9 TOP + 0.009 TOP^2, s_TO = 1.66 s_TOG,
                                          TOP = (W/S)(W/P) / (sigma CL_max,TO)  [lb/ft^2, lb/hp]
      - rate of climb at sea level        P/W = [ROC + V (q CD0 / (W/S) + K (W/S) / q)] / eta at V_y
      - cruise speed at cruise altitude   at the cruise power setting, altitude lapse of the engine
      - service ceiling                   0.5 m/s rate of climb at the ceiling
      - sustained turn (optional)         load factor n at a given speed
    Design point: the largest wing loading the stall and landing lines allow
    (or a chosen one), the smallest power that meets every line there plus a
    margin.
  * Drag polar from the component drag buildup of the current geometry
    (drag_buildup.py) and the span efficiency of the vortex lattice
    (aircraft.py) with Kroo's viscous lift-dependent term,
    1/e = 1/e_inv + 0.38 CD0 pi A - updated every iteration, so geometry,
    drag and weights stay consistent.
  * Mission: warm-up and take-off, climb, cruise (Breguet, propeller),
    reserve (loiter at minimum-power speed), landing; fuel fractions of
    single-engine propeller aircraft (Roskam, Airplane Design I, Table 2.1:
    0.995 x 0.997 x 0.998, 0.992, Breguet, Breguet, 0.993 x 0.993) and 6 %
    trapped-fuel allowance (Raymer). Electric: battery energy for climb, cruise and reserve
    at constant mass, usable fraction and motor efficiency.
  * Class I empty-mass fraction (Raymer Table 3.1, W0 in lb):
    W_e / W_0 = A W_0^C K_vs (general aviation single engine A 2.36, C -0.18;
    twin 1.51, -0.10; homebuilt metal/wood 1.19, -0.09; composite 1.15, -0.09).
  * Class II component masses (Raymer eqs. 15.46-15.58, general aviation):
    wing, horizontal and vertical tail, fuselage, main and nose gear,
    installed engine, fuel system, flight controls, hydraulics, avionics,
    electrical, furnishings; engine dry mass from power (certified flat
    piston engines m = 0.68 P[kW] + 35; turboprops 0.25 P + 40; electric
    motor + controller P/5 + P/15), propeller 0.12 kg/kW (fixed pitch) or
    0.18 (constant speed); a technology factor for composite structure.
    Take-off mass iterated to convergence: W0 = W_e (Class II) + W_crew +
    W_payload + W_fuel (or battery).
  * Geometry: wing from W/S, aspect ratio, taper, sweep; fuselage length by
    Raymer's statistics (L = a W0^C), diameter from the seating; tails from
    volume coefficients (Raymer Table 6.4); wing placed so the loaded c.g.
    sits at the target % MAC (weight_balance.place_wing); tricycle gear
    placed for tip-back, nose-wheel load, turnover and propeller clearance.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

import aircraft as A
import drag_buildup as DB
import weight_balance as WB
import wing as W

G0 = 9.80665
LB, FT, FT2, HP, KT, GAL = 2.20462, 3.28084, 10.7639, 745.7, 1.94384, 0.264172
PSF = 0.0208854                                    # Pa -> lb/ft^2

CLASS_I = {"general aviation, single engine": (2.36, -0.18),
           "general aviation, twin engine": (1.51, -0.10),
           "homebuilt, metal or wood": (1.19, -0.09),
           "homebuilt, composite": (1.15, -0.09),
           "powered sailplane": (0.91, -0.05)}
FUS_LENGTH = {"general aviation, single engine": (4.37, 0.23), "general aviation, twin engine": (0.86, 0.42),
              "homebuilt, metal or wood": (3.68, 0.23), "homebuilt, composite": (3.50, 0.23),
              "powered sailplane": (0.71, 0.48)}
TAIL_VOLUME = {"general aviation, single engine": (0.70, 0.04), "general aviation, twin engine": (0.80, 0.07),
               "homebuilt, metal or wood": (0.50, 0.04), "homebuilt, composite": (0.50, 0.04),
               "powered sailplane": (0.50, 0.02)}


def default_requirements():
    """a two-seat touring trainer (generic)"""
    return {
        "name": "Sized two-seat trainer",
        "category": "general aviation, single engine",
        "crew": 1, "passengers": 1, "person_mass": 86.0, "baggage": 20.0,
        "range_km": 900.0, "reserve_min": 45.0,
        "cruise_speed": 55.0, "cruise_alt": 1500.0, "cruise_power": 0.75,
        "stall_speed": 24.0, "takeoff_distance": 450.0, "landing_distance": 450.0,
        "climb_rate": 3.5, "ceiling": 4000.0, "turn_n": None, "turn_speed": None,
        "aspect_ratio": 8.0, "taper": 0.6, "sweep": 0.0, "dihedral": 4.0, "twist": -2.0,
        "airfoil": "naca2412", "tail_airfoil": "naca0012", "high_wing": False,
        "cl_max_clean": 1.5, "cl_max_takeoff": 1.7, "cl_max_landing": 2.0,
        "flap_takeoff": 10.0, "flap_landing": 30.0,
        "engine": "piston", "engines": 1, "sfc": 300.0, "eta_prop_cruise": 0.80, "eta_prop_climb": 0.70,
        "prop_mode": "fixed_pitch", "rpm": 2600.0,
        "battery_wh_kg": 220.0, "battery_usable": 0.8, "eta_motor": 0.92,
        "gear": "fixed_faired", "seats_abreast": 2, "load_factor": 3.8,
        "composite": False, "power_margin": 0.05, "cg_target": 25.0,
        "wing_loading": None, "power_kw": None,
        "drag": {},          # drag-buildup options: roughness, laminar, upsweep_deg, leakage_pct, other
    }


# ----------------------------------------------------------------------
# constraints
# ----------------------------------------------------------------------
def lapse(kind, h):
    sig = W.isa(h)[0] / 1.225
    if kind == "piston":
        return max(sig - (1 - sig) / 7.55, 1e-3)
    if kind == "turboprop":
        return sig**0.7
    return 1.0


def constraints(req, CD0, K, ws=None):
    """P/W (W/N) of each constraint on a W/S grid (N/m^2); vertical lines as W/S limits"""
    rho0 = 1.225
    kind = req["engine"]
    ws = np.linspace(150, 1800, 140) if ws is None else np.asarray(ws, float)
    out = {"ws": ws.tolist(), "lines": {}, "limits": {}}
    # stall (landing configuration)
    out["limits"]["stall speed"] = 0.5 * rho0 * req["stall_speed"] ** 2 * req["cl_max_landing"]
    # landing distance (Roskam, CS-23 / FAR 23): s [ft] = 0.5136 V_s,L^2 [kt]
    if req.get("landing_distance"):
        v_sl = math.sqrt(req["landing_distance"] * FT / 0.5136) / KT
        out["limits"]["landing distance"] = 0.5 * rho0 * v_sl**2 * req["cl_max_landing"]
    # take-off (Roskam): s_TO = 1.66 (4.9 TOP + 0.009 TOP^2) ft
    if req.get("takeoff_distance"):
        s_tog = req["takeoff_distance"] * FT / 1.66
        top = (-4.9 + math.sqrt(4.9**2 + 4 * 0.009 * s_tog)) / (2 * 0.009)
        wp = top * req["cl_max_takeoff"] / (ws * PSF)                    # lb/hp
        out["lines"]["take-off distance"] = (HP / (LB * G0)) / wp            # W/N
    # climb at sea level, best-rate speed
    eta_c = req["eta_prop_climb"]
    vy = np.sqrt(2 / rho0 * ws * math.sqrt(K / (3 * CD0)))
    q = 0.5 * rho0 * vy**2
    out["lines"]["rate of climb"] = (req["climb_rate"] + vy * (q * CD0 / ws + K * ws / q)) / eta_c
    # cruise speed at altitude, at the cruise power setting
    rho_c = W.isa(req["cruise_alt"])[0]
    V = req["cruise_speed"]
    qc = 0.5 * rho_c * V**2
    out["lines"]["cruise speed"] = V * (qc * CD0 / ws + K * ws / qc) / \
        (req["eta_prop_cruise"] * req["cruise_power"] * lapse(kind, req["cruise_alt"]))
    # service ceiling: 0.5 m/s
    if req.get("ceiling"):
        rh = W.isa(req["ceiling"])[0]
        vyh = np.sqrt(2 / rh * ws * math.sqrt(K / (3 * CD0)))
        qh = 0.5 * rh * vyh**2
        out["lines"]["service ceiling"] = (0.508 + vyh * (qh * CD0 / ws + K * ws / qh)) / \
            (eta_c * lapse(kind, req["ceiling"]))
    # sustained turn
    if req.get("turn_n") and req.get("turn_speed"):
        n, Vt = req["turn_n"], req["turn_speed"]
        qt = 0.5 * rho0 * Vt**2
        out["lines"][f"turn n = {n:g}"] = Vt * (qt * CD0 / ws + K * n**2 * ws / qt) / req["eta_prop_cruise"]
    out["lines"] = {k: np.asarray(v).tolist() for k, v in out["lines"].items()}
    return out


def design_point(req, con):
    ws_max = min(con["limits"].values())
    ws = req.get("wing_loading") * G0 if req.get("wing_loading") else ws_max
    grid = np.asarray(con["ws"])
    pw = {k: float(np.interp(ws, grid, v)) for k, v in con["lines"].items()}
    active = max(pw, key=pw.get)
    p_w = pw[active] * (1 + req.get("power_margin", 0.05))
    return {"ws": ws, "pw": p_w, "ws_max": ws_max, "active": active, "pw_lines": pw,
            "ws_limit": min(con["limits"], key=con["limits"].get)}


# ----------------------------------------------------------------------
# mission
# ----------------------------------------------------------------------
def mission(req, W0, S, CD0, K):
    """fuel mass (or battery mass) and the segment table"""
    rho_c = W.isa(req["cruise_alt"])[0]
    V = req["cruise_speed"]
    eta = req["eta_prop_cruise"]
    Wn = W0 * G0
    segs = []
    if req["engine"] == "electric":
        e_spec = req["battery_wh_kg"] * 3600.0 * req["battery_usable"]       # J/kg usable
        eta_m = req["eta_motor"]
        h = req["cruise_alt"]
        # climb: potential energy + drag at V_y over the climb time
        roc = max(req["climb_rate"], 1.0)
        vy = math.sqrt(2 / 1.225 * Wn / S * math.sqrt(K / (3 * CD0)))
        D_y = 0.5 * 1.225 * vy**2 * S * (CD0 + K * (Wn / (0.5 * 1.225 * vy**2 * S)) ** 2)
        t_cl = h / roc
        E_climb = (Wn * h + D_y * vy * t_cl) / (req["eta_prop_climb"] * eta_m)
        dist_cl = vy * t_cl
        CL = Wn / (0.5 * rho_c * V**2 * S)
        D = 0.5 * rho_c * V**2 * S * (CD0 + K * CL**2)
        E_cruise = D * max(req["range_km"] * 1e3 - dist_cl, 0.0) / (eta * eta_m)
        vmp = math.sqrt(2 / rho_c * Wn / S * math.sqrt(K / (3 * CD0)))
        CLm = Wn / (0.5 * rho_c * vmp**2 * S)
        Pmp = 0.5 * rho_c * vmp**3 * S * (CD0 + K * CLm**2)
        E_res = Pmp * req["reserve_min"] * 60 / (eta * eta_m)
        E_to = 0.02 * (E_climb + E_cruise)
        E = E_to + E_climb + E_cruise + E_res
        for lab, e in (("take-off", E_to), ("climb", E_climb), ("cruise", E_cruise), ("reserve", E_res)):
            segs.append({"segment": lab, "energy_kWh": e / 3.6e6})
        mb = E / e_spec
        return {"fuel": 0.0, "battery": mb, "energy_kWh": E / 3.6e6, "segments": segs,
                "L_D_cruise": CL / (CD0 + K * CL**2), "reserve_energy_fraction": E_res / E,
                "cruise_energy_fraction": E_cruise / E}
    c = req["sfc"] / 3.6e9                                                 # kg/J (shaft)
    f = 1.0
    # Roskam (Airplane Design I, Table 2.1), single-engine propeller aircraft
    for lab, fr in (("engine start, warm-up, taxi, take-off", 0.995 * 0.997 * 0.998), ("climb", 0.992)):
        f *= fr
        segs.append({"segment": lab, "fraction": fr})
    # cruise at the mean mass of the cruise
    Wc = Wn * f * 0.96
    CL = Wc / (0.5 * rho_c * V**2 * S)
    LD = CL / (CD0 + K * CL**2)
    fr = math.exp(-req["range_km"] * 1e3 * c * G0 / (eta * LD))
    cruise_fuel = W0 * f * (1 - fr)
    f *= fr
    segs.append({"segment": f"cruise {req['range_km']:g} km at {V:g} m/s", "fraction": fr, "L_D": LD})
    # reserve: loiter at minimum-power speed (L/D = 0.866 L/D_max)
    LDmax = 1 / (2 * math.sqrt(CD0 * K))
    E = req["reserve_min"] * 60
    vmp = math.sqrt(2 / rho_c * Wc / S * math.sqrt(K / (3 * CD0)))
    fr = math.exp(-E * vmp * c * G0 / (eta * 0.866 * LDmax))
    reserve_fuel = W0 * f * (1 - fr)
    f *= fr
    segs.append({"segment": f"reserve {req['reserve_min']:g} min at {vmp:.0f} m/s", "fraction": fr})
    f *= 0.993 * 0.993
    segs.append({"segment": "descent, landing, taxi", "fraction": 0.993 * 0.993})
    wf = 1.06 * (1 - f)
    return {"fuel": wf * W0, "battery": 0.0, "fuel_fraction": wf, "segments": segs, "L_D_cruise": LD,
            "reserve_fuel": reserve_fuel, "cruise_fuel": cruise_fuel}


# ----------------------------------------------------------------------
# weights
# ----------------------------------------------------------------------
def class_i(category, W0):
    a, c = CLASS_I.get(category, CLASS_I["general aviation, single engine"])
    return a * (W0 * LB) ** c


def engine_mass(kind, p_kw):
    if kind == "piston":
        return 0.68 * p_kw + 35.0
    if kind == "turboprop":
        return 0.25 * p_kw + 40.0
    return p_kw / 5.0 + p_kw / 15.0


def class_ii(req, ac, W0, fuel, p_kw, L_D, battery=0.0):
    """Raymer's general-aviation component weights (eqs. 15.46-15.58), kg"""
    w, ht = ac.wing, ac.surfaces[1]
    vt = ac.surfaces[2]
    Nz = 1.5 * req["load_factor"]
    Wdg = W0 * LB
    q = 0.5 * W.isa(req["cruise_alt"])[0] * req["cruise_speed"] ** 2 * PSF
    tc = DB.section_shape(w.airfoil)[1]
    tct = DB.section_shape(ht.airfoil)[1]
    ar = w.span**2 / w.area
    lam_w = w.tip_chord / w.root_chord
    cosL = math.cos(math.radians(w.sweep))
    Wfw = fuel * LB
    Sw = w.area * FT2
    m = {}
    m["wing"] = 0.036 * Sw**0.758 * max(Wfw, 1.0) ** 0.0035 * (ar / cosL**2) ** 0.6 * q**0.006 * \
        lam_w**0.04 * (100 * tc / cosL) ** -0.3 * (Nz * Wdg) ** 0.49
    ar_h = ht.span**2 / ht.area
    cos_h = math.cos(math.radians(ht.sweep))
    m[ht.name] = 0.016 * (Nz * Wdg) ** 0.414 * q**0.168 * (ht.area * FT2) ** 0.896 * \
        (100 * tct / cos_h) ** -0.12 * (ar_h / cos_h**2) ** 0.043 * (ht.tip_chord / ht.root_chord) ** -0.02
    ar_v = vt.span**2 / vt.area
    cos_v = math.cos(math.radians(vt.sweep))
    m[vt.name] = 0.073 * (1 + 0.2 * 0.0) * (Nz * Wdg) ** 0.376 * q**0.122 * (vt.area * FT2) ** 0.873 * \
        (100 * DB.section_shape(vt.airfoil)[1] / cos_v) ** -0.49 * (ar_v / cos_v**2) ** 0.357 * \
        (vt.tip_chord / vt.root_chord) ** 0.039
    f = ac.fuselage
    _, swet_f = f.volume_and_wetted()
    lt = (WB.mac_x(ht, 0.25) - WB.mac_x(w, 0.25)) * FT
    m["fuselage"] = 0.052 * (swet_f * FT2) ** 1.086 * (Nz * Wdg) ** 0.177 * max(lt, 1.0) ** -0.051 * \
        max(L_D, 1.0) ** -0.072 * q**0.241
    Nl = 1.5 * 3.0                                                     # ultimate landing load factor
    Wl = Wdg
    Lm, Ln = 0.6 * 39.37, 0.55 * 39.37                                  # gear lengths (in)
    m["main gear"] = 0.095 * (Nl * Wl) ** 0.768 * (Lm / 12) ** 0.409
    m["nose gear"] = 0.125 * (Nl * Wl) ** 0.566 * (Ln / 12) ** 0.845
    n_en = req.get("engines", 1)
    m_en = engine_mass(req["engine"], p_kw / n_en)
    m["engine"] = 2.575 * (m_en * LB) ** 0.922 * n_en                   # installed (mounts, cowl, ...)
    if req["engine"] != "electric":
        vt_gal = fuel / 0.72 * GAL
        m["fuel system"] = 2.49 * max(vt_gal, 1.0) ** 0.726 * (1 / (1 + 0.0)) ** 0.363 * 2**0.242 * n_en**0.157
    m["flight controls"] = 0.053 * (f.length * FT) ** 1.536 * (w.span * FT) ** 0.371 * \
        (Nz * Wdg * 1e-4) ** 0.80
    m["hydraulics"] = 0.001 * Wdg if req.get("hydraulics", False) else 0.0
    w_uav = req.get("avionics_kg", 15.0) * LB
    m["avionics"] = 2.117 * w_uav**0.933
    m["electrical"] = 12.57 * (m.get("fuel system", 0.0) + m["avionics"]) ** 0.51
    np_ = req["crew"] + req["passengers"]
    m["furnishings"] = max(0.0582 * Wdg - 65, 0.0)
    out = {k: v / LB for k, v in m.items() if v > 0}
    # composite structure: technology factor on wing, tails, fuselage
    if req.get("composite"):
        for k in ("wing", ht.name, vt.name, "fuselage"):
            out[k] *= 0.85
    prop_kg = (0.18 if req.get("prop_mode") == "constant_speed" else 0.12) * p_kw if req["engine"] != "jet" else 0
    out["propeller"] = prop_kg
    if req["engine"] == "electric":
        out["electrical"] = out.get("electrical", 0.0) + 0.05 * battery          # wiring, BMS
    _ = np_
    return out


# ----------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------
def build_aircraft(req, W0, S, p_kw, lt_frac=0.6):
    ar, tr = req["aspect_ratio"], req["taper"]
    b = math.sqrt(ar * S)
    cr = 2 * S / (b * (1 + tr))
    ct = tr * cr
    cat = req["category"]
    a, c = FUS_LENGTH.get(cat, FUS_LENGTH["general aviation, single engine"])
    L = a * (W0 * LB) ** c / FT
    d = {1: 0.95, 2: 1.20}.get(int(req.get("seats_abreast", 2)), 1.2 + 0.5 * (req.get("seats_abreast", 2) - 2))
    fus = A.Fuselage(length=L, diameter=d, x=0.0, z=0.0, x_max=0.3, tail_ratio=0.12)
    zw = 0.35 * d if req.get("high_wing") else -0.3 * d
    wing = A.Surface("wing", span=b, root_chord=cr, tip_chord=ct, x=0.27 * L, z=zw,
                     sweep=req["sweep"], dihedral=1.5 if req.get("high_wing") else req["dihedral"],
                     twist=req["twist"], incidence=1.5, airfoil=req["airfoil"],
                     controls=[A.Control("aileron", 0.58, 0.95, 0.25, "antisymmetric"),
                               A.Control("flap", 0.12, 0.56, 0.25, "symmetric")], ns=14, nc=6)
    vh, vv = TAIL_VOLUME.get(cat, TAIL_VOLUME["general aviation, single engine"])
    mac = wing.mac
    lt = lt_frac * L
    Sh = vh * mac * S / lt
    Sv = vv * b * S / lt
    bh = math.sqrt(4.5 * Sh)
    crh = 2 * Sh / (bh * 1.6)
    hv = math.sqrt(1.5 * Sv)
    crv = 2 * Sv / (hv * 1.5)
    xw_q = wing.mac_position()[0] + 0.25 * mac
    ht = A.Surface("horizontal tail", span=bh, root_chord=crh, tip_chord=0.6 * crh, x=0.0, z=0.1 * d,
                   sweep=5.0, incidence=-1.5, airfoil=req["tail_airfoil"],
                   controls=[A.Control("elevator", 0.05, 1.0, 0.4, "symmetric")], ns=8, nc=5)
    ht.x = xw_q + lt - (ht.mac_position()[0] - ht.x + 0.25 * ht.mac)
    vt = A.Surface("fin", span=hv, root_chord=crv, tip_chord=0.5 * crv, x=0.0, z=0.25 * d, sweep=30.0,
                   airfoil=req["tail_airfoil"], symmetric=False, vertical=True,
                   controls=[A.Control("rudder", 0.08, 0.95, 0.4, "symmetric")], ns=8, nc=5)
    vt.x = xw_q + lt - (vt.mac_position()[0] - vt.x + 0.25 * vt.mac)
    # tails must stay on the fuselage: shorten the arm if they would overhang
    over = max(ht.x + ht.root_chord, vt.x + vt.root_chord) - (L + 0.05)
    if over > 0:
        ht.x -= over
        vt.x -= over
    return A.Aircraft(req.get("name", "Sized aircraft"), [wing, ht, vt], fus, mass=W0,
                      cg=(wing.mac_position()[0] + 0.25 * mac, 0.0, -0.1 * d))


def resize_tails(ac, req):
    """tail areas from the volume coefficients for the current tail arm"""
    vh, vv = TAIL_VOLUME.get(req["category"], TAIL_VOLUME["general aviation, single engine"])
    w = ac.wing
    xw = WB.mac_x(w, 0.25)
    for s in ac.surfaces[1:]:
        xq = WB.mac_x(s, 0.25)
        lt = max(xq - xw, 0.2 * ac.fuselage.length)
        need = (vv * ac.b * ac.S if s.vertical else vh * ac.c * ac.S) / lt
        f = math.sqrt(need / s.area)
        s.span *= f
        s.root_chord *= f
        s.tip_chord *= f
        s.x += xq - WB.mac_x(s, 0.25)
    return ac


def oswald_lattice(ac):
    """span efficiency of the whole configuration from the lattice (inviscid)"""
    an = A.Analysis(ac, 50.0, 0.0, None)
    r1 = an.solve(alpha=2.0, viscous=False)
    r2 = an.solve(alpha=6.0, viscous=False)
    ar = ac.b**2 / ac.S
    dcl2 = r2["CL"] ** 2 - r1["CL"] ** 2
    dcdi = r2["CDi"] - r1["CDi"]
    return float(dcl2 / (math.pi * ar * dcdi)) if dcdi > 0 else 0.8


def drag(req, ac, p_kw):
    """component drag buildup of the current geometry; req["drag"] may set
    roughness, laminar fractions, upsweep, leakage and other D/q items
    (struts, antennas, ...) - see drag_buildup.buildup"""
    opt = {"laminar": {"wing": 0.30, "tail": 0.30, "fuselage": 0.10} if req.get("composite") else None,
           "roughness": "smooth moulded composite" if req.get("composite") else "smooth paint"}
    opt |= dict(req.get("drag") or {})
    return DB.buildup(ac, req["cruise_speed"], req["cruise_alt"], gear={"type": req["gear"]},
                      power_kw=p_kw * req.get("cruise_power", 0.75), engine=req["engine"], **opt)


def oswald_total(e_inv, CD0, ar):
    """Oswald factor: inviscid span efficiency plus the viscous lift-dependent
    drag (Kroo: 1/e = 1/e_inv + 0.38 CD0 pi A)"""
    return 1.0 / (1.0 / e_inv + 0.38 * CD0 * math.pi * ar)


def components_from(req, ac, m2, fuel, battery):
    comps = [{"name": k, "group": "empty", "mass": v} for k, v in m2.items()]
    names = ["pilot", "front passenger", "rear passengers", "rear passengers"]
    persons = req["crew"] + req["passengers"]
    for i in range(persons):
        nm = names[min(i, 3)]
        grp = "crew" if i < req["crew"] else "payload"
        comps.append({"name": nm, "group": grp, "mass": req["person_mass"], "max": req["person_mass"]})
    if req.get("baggage", 0) > 0:
        comps.append({"name": "baggage", "group": "payload", "mass": req["baggage"], "max": req["baggage"]})
    if fuel > 0:
        comps.append({"name": "fuel", "group": "fuel", "mass": fuel, "max": fuel})
    if battery > 0:
        comps.append({"name": "battery", "group": "battery", "mass": battery})
    return WB.place(comps, ac)


# ----------------------------------------------------------------------
def size(req=None, tol=0.5, max_it=40, log=print):
    """the full sizing loop; returns (result dict, Aircraft)"""
    req = default_requirements() | dict(req or {})
    m_people = (req["crew"] + req["passengers"]) * req["person_mass"]
    m_pay = m_people + req.get("baggage", 0.0)
    W0 = 3.0 * m_pay + 200.0                                           # first guess
    CD0, e = 0.028, 0.78
    hist = []
    ac = None
    for it in range(1, max_it + 1):
        ar = req["aspect_ratio"]
        K = 1 / (math.pi * ar * e)
        con = constraints(req, CD0, K)
        dp = design_point(req, con)
        S = W0 * G0 / dp["ws"]
        p_kw = req["power_kw"] if req.get("power_kw") else dp["pw"] * W0 * G0 / 1e3
        if ac is None:
            ac = build_aircraft(req, W0, S, p_kw)
        else:                                                          # rescale the wing, keep its position
            ac = rescale(ac, req, W0, S)
        # drag polar of the current geometry
        db = drag(req, ac, p_kw)
        CD0 = db["CD0"]
        if it <= 3 or it % 4 == 0:                                     # span efficiency from the lattice
            e_inv = min(oswald_lattice(ac), 1.0)
        e = oswald_total(e_inv, CD0, ar)
        ms = mission(req, W0, S, CD0, 1 / (math.pi * ar * e))
        m2 = class_ii(req, ac, W0, ms["fuel"], p_kw, ms["L_D_cruise"], ms["battery"])
        m_e = sum(m2.values())
        W0_new = m_e + m_pay + ms["fuel"] + ms["battery"]
        hist.append({"iteration": it, "W0": W0, "W0_new": W0_new, "S": S, "P_kW": p_kw, "CD0": CD0, "e": e,
                     "empty": m_e, "fuel": ms["fuel"], "battery": ms["battery"]})
        log(f"[sizing] it {it}: W0 {W0:.1f} -> {W0_new:.1f} kg, S {S:.2f} m², P {p_kw:.1f} kW, "
            f"CD0 {CD0:.4f}, e {e:.3f}, empty {m_e:.0f} kg, fuel {ms['fuel']:.1f} kg, battery {ms['battery']:.1f} kg")
        if abs(W0_new - W0) < tol and it > 3:
            W0 = W0_new
            break
        W0 = W0 + 0.8 * (W0_new - W0)
        if not np.isfinite(W0) or W0 > 1e5:
            raise RuntimeError("sizing diverged: the mission cannot be met with these assumptions "
                               "(range or battery energy too high?)")
    converged = abs(hist[-1]["W0_new"] - hist[-1]["W0"]) < tol
    # final consistent model: masses, positions, wing placement, tail volumes, gear
    S = W0 * G0 / dp["ws"]
    ac = rescale(ac, req, W0, S)
    ac, comps, ld, prop_d, gc, db, m2 = finalize(req, ac, W0, m2, ms["fuel"], ms["battery"], p_kw,
                                                 ms["L_D_cruise"])
    m_e = sum(c["mass"] for c in comps if c["group"] == "empty")
    con = constraints(req, db["CD0"], 1 / (math.pi * req["aspect_ratio"] * e))
    res = {"requirements": req, "converged": converged, "iterations": hist,
           "mtow": W0, "empty_mass": m_e, "empty_fraction": m_e / W0,
           "class_i_empty_fraction": class_i(req["category"], W0),
           "payload": m_pay, "fuel": ms["fuel"], "battery": ms["battery"], "mission": ms,
           "wing_area": S, "span": ac.b, "mac": ac.c, "power_kw": p_kw,
           "wing_loading_kg_m2": W0 / S, "power_loading_w_kg": p_kw * 1e3 / W0,
           "design_point": dp, "constraints": con, "CD0": db["CD0"], "oswald_e": e,
           "L_D_cruise": ms["L_D_cruise"], "L_D_max": 0.5 * math.sqrt(math.pi * req["aspect_ratio"] * e / db["CD0"]),
           "class_ii": {c["name"]: c["mass"] for c in comps if c["group"] == "empty"},
           "drag": DB.table(db), "loading": {k: v for k, v in ld.items() if k != "components"},
           "gear": gc, "prop_diameter": prop_d, "aircraft": ac.to_dict()}
    return res, ac


def finalize(req, ac, W0, m2, fuel, battery, p_kw, L_D=12.0, class_ii_masses=True):
    """masses and positions, wing placed for the c.g. target, tails from the volume
    coefficients, landing gear, extra drag and propulsion of a sized geometry.
    class_ii_masses=False keeps the given empty masses (templates outside Raymer's
    statistics: sailplanes, small UAVs). p_kw = 0: no propulsion (glider)."""
    comps = components_from(req, ac, m2, fuel, battery)
    for _ in range(3):
        ac, comps = WB.place_wing(ac, comps, req["cg_target"])
        ac = resize_tails(ac, req)
        if class_ii_masses:
            m2 = class_ii(req, ac, W0, fuel, p_kw, L_D, battery)
            for c in comps:
                if c["group"] == "empty" and c["name"] in m2:
                    c["mass"] = m2[c["name"]]
        comps = WB.place([{k: v for k, v in c.items() if k not in ("x", "z") or c["name"] not in
                           (ac.surfaces[1].name, ac.surfaces[2].name)} for c in comps], ac)
    ld = WB.loading(ac, comps)
    prop_d = prop_diameter(p_kw / req.get("engines", 1), req) if p_kw > 0 else 0.0
    ac.gear = WB.suggest_gear(ac, ld, prop_diameter=prop_d)
    comps = WB.place([{k: v for k, v in c.items() if k not in ("x", "z") or c["name"] not in
                       ("main gear", "nose gear")} for c in comps], ac)
    ld = WB.loading(ac, comps)
    ac.components = comps
    ac.mass = W0
    ac.cg = (ld["cases"]["crew + payload + full fuel"]["x"], 0.0, ld["cases"]["crew + payload + full fuel"]["z"])
    db = drag(req, ac, p_kw)
    ac.cd_extra = DB.cd_extra_for_model(db)
    ac.propulsion = propulsion_spec(req, p_kw, prop_d) if p_kw > 0 else {}
    gc = WB.gear_checks(ac, ld, prop_diameter=prop_d)
    return ac, comps, ld, prop_d, gc, db, m2


def rescale(ac, req, W0, S):
    """new wing area and fuselage length, positions scaled"""
    ac = copy.deepcopy(ac)
    w = ac.wing
    b = math.sqrt(req["aspect_ratio"] * S)
    cr = 2 * S / (b * (1 + req["taper"]))
    xq_old = WB.mac_x(w, 0.25)
    w.span, w.root_chord, w.tip_chord = b, cr, req["taper"] * cr
    w.x += xq_old - WB.mac_x(w, 0.25)
    a, c = FUS_LENGTH.get(req["category"], FUS_LENGTH["general aviation, single engine"])
    L = a * (W0 * LB) ** c / FT
    k = L / ac.fuselage.length
    ac.fuselage.length = L
    for s in ac.surfaces:
        s.x *= k
    ac.mass = W0
    return resize_tails(ac, req)


def prop_diameter(p_kw, req):
    """propeller diameter: Raymer's statistical D = K P^0.25 (2 blades: 0.56 m/kW^0.25
    for piston light aircraft), capped by a 0.85 helical tip Mach number"""
    d = 0.56 * p_kw**0.25 * (1.0 if req.get("prop_mode") == "fixed_pitch" else 0.95)
    n = req.get("rpm", 2600) / 60
    vt_max = 0.85 * 340.3
    d_mach = 2 * math.sqrt(max(vt_max**2 - req["cruise_speed"] ** 2, 1.0)) / (2 * math.pi * n)
    return float(min(d, d_mach))


def propulsion_spec(req, p_kw, d):
    kind = req["engine"]
    eng = {"kind": kind, "power_kw": p_kw / req.get("engines", 1), "rpm_max": req.get("rpm", 2600),
           "sfc": req.get("sfc", 300.0)}
    if kind == "electric":
        eng |= {"eta_motor": req["eta_motor"], "usable_fraction": req["battery_usable"]}
    return {"engine": eng, "mode": req.get("prop_mode", "fixed_pitch"), "rpm": req.get("rpm", 2600),
            "propeller": {"design": {"speed": req["cruise_speed"], "rpm": 0.92 * req.get("rpm", 2600),
                                     "diameter": d, "blades": 2,
                                     "power_kw": req.get("cruise_power", 0.75) * p_kw / req.get("engines", 1),
                                     "altitude": req["cruise_alt"]}}}


# ----------------------------------------------------------------------
def plots(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    con, dp = res["constraints"], res["design_point"]
    ws = np.array(con["ws"]) / G0
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    for k, v in con["lines"].items():
        a1.plot(ws, np.array(v) * G0, label=k)
    for k, v in con["limits"].items():
        a1.axvline(v / G0, ls="--", lw=1, label=k, color="0.4" if "stall" in k else "0.65")
    a1.plot(dp["ws"] / G0, dp["pw"] * G0, "k*", ms=14, label="design point")
    a1.set_ylim(0, 3 * dp["pw"] * G0)
    a1.set_xlabel("wing loading (kg/m²)"); a1.set_ylabel("power to weight (W/kg)")
    a1.legend(fontsize=8); a1.grid(alpha=0.3)
    h = res["iterations"]
    a2.plot([x["iteration"] for x in h], [x["W0"] for x in h], "o-", label="take-off mass")
    a2.plot([x["iteration"] for x in h], [x["empty"] for x in h], "s-", label="empty mass")
    a2.set_xlabel("iteration"); a2.set_ylabel("kg"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out) / "sizing.png", dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("requirements", nargs="?", default=None, help="JSON file (keys of default_requirements)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    req = json.loads(Path(a.requirements).read_text()) if a.requirements else {}
    res, ac = size(req)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sizing.json").write_text(json.dumps(A._jsonable(res), indent=1))
    ac.save(out / "aircraft.json")
    plots(res, out)
    print(f"MTOW {res['mtow']:.0f} kg, empty {res['empty_mass']:.0f} kg ({100 * res['empty_fraction']:.1f} %, "
          f"Class I {100 * res['class_i_empty_fraction']:.1f} %), fuel {res['fuel']:.0f} kg, battery "
          f"{res['battery']:.0f} kg; S {res['wing_area']:.2f} m², b {res['span']:.2f} m, P {res['power_kw']:.0f} kW; "
          f"CD0 {res['CD0']:.4f}, e {res['oswald_e']:.2f}, L/D cruise {res['L_D_cruise']:.1f}")
    print(f"-> {out}/sizing.json, aircraft.json, sizing.png")


if __name__ == "__main__":
    main()
