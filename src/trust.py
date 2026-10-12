"""Trust layer: how far can a result be trusted?

Every number the tools produce gets
  - a confidence level (green / amber / red),
  - an expected uncertainty band (Cl, Cd) from the closest validated case,
  - the reasons, written as an engineer would say them, and the reference behind them.

Three sources of evidence:
  1. the validation envelope: the cases this code was checked against (NASA TMR, XFOIL,
     NASA LTPT, JAXA wind tunnel, RAE 2822 experiment, ...), each with the range of
     Re, Mach, angle and transition mode it covers and the error found there;
  2. point rules: convergence, stall, shocks, y+, mesh level, laminar bubbles, geometry;
  3. grid studies: Richardson extrapolation and the grid convergence index (GCI,
     Celik et al. 2008, ASME J. Fluids Eng. 130 078001) from three mesh levels.

    python trust.py check --re 3e6 --alpha 4 --transition free
    python trust.py polar ../cases/naca2412_Re3e06
    python trust.py grid-study ../cases/naca2412_Re3e06 --alpha 4 --jobs 2
    python trust.py envelope
"""
import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

LEVELS = ("green", "amber", "red")
RANK = {k: i for i, k in enumerate(LEVELS)}
LABEL = {"green": "validated range", "amber": "use with care", "red": "low confidence", "info": "note"}


def worst(*levels):
    """highest of green < amber < red; 'info' notes don't count"""
    return max((l for l in levels if l in RANK), key=lambda l: RANK[l], default="green")


# ----------------------------------------------------------------------
# 1. validation envelope
# ----------------------------------------------------------------------
@dataclass
class Case:
    id: str
    title: str
    reference: str
    elements: str                 # "single" | "multi"
    transition: tuple             # modes covered: "free", "fixed", "off", "laminar"
    re: tuple                     # (lo, hi) covered
    mach: tuple                   # (lo, hi) covered
    alpha: tuple                  # (lo, hi) deg covered
    cl_err: float                 # typical relative error found (fraction)
    cd_err: float
    report: str = ""              # file in validation/
    note: str = ""
    compressible: bool = False    # needs the compressible solver (shocks)
    tested: list = field(default_factory=list)   # (Re, M, alpha) points actually compared


CASES = [
    Case("tmr_naca0012", "NACA 0012, fully turbulent SA against NASA TMR (CFL3D, grid-converged)",
         "NASA Turbulence Modeling Resource; Ladson (lift slope)", "single", ("off",),
         (2e6, 1.2e7), (0.0, 0.3), (-4.0, 16.0), 0.015, 0.023, "naca0012_sa_report.txt",
         "Cd within 0.7–2.5 % (finest to coarsest mesh), Cl within 0.1 % once the M 0.15 of the "
         "reference is accounted for.", tested=[(6e6, 0.15, a) for a in (0, 10, 15)]),
    Case("tripped_naca2412", "NACA 2412, trip at x/c 0.01, against XFOIL (tripped)", "XFOIL 6.99",
         "single", ("fixed",), (1e6, 6e6), (0.0, 0.3), (-2.0, 6.0), 0.03, 0.019,
         "phase3_report.txt", "Cd within 0.2–1.9 %; XFOIL's lift is typically 1–3 % higher.",
         tested=[(3e6, 0, a) for a in (-2, 0, 2, 4, 6)]),
    Case("free_naca0012_3e6", "NACA 0012, free transition (e^N, N_crit 9), attached range",
         "XFOIL (free transition)", "single", ("free",), (7e5, 6e6), (0.0, 0.3), (-2.0, 8.0),
         0.02, 0.03, "transition_report.txt",
         "Transition points within ~0.01 c of XFOIL; Cd within ±3 %.",
         tested=[(3e6, 0, a) for a in (-2, 0, 2, 4, 6, 8)] + [(1e6, 0, a) for a in (0, 2, 4, 6, 8)]),
    Case("free_naca0012_high_alpha", "NACA 0012, free transition at 10–12°", "XFOIL (free transition)",
         "single", ("free",), (7e5, 6e6), (0.0, 0.3), (9.0, 12.0), 0.03, 0.07,
         "transition_report.txt", "Cd 5–7 % above XFOIL as the transition point nears the leading edge.",
         tested=[(3e6, 0, 10), (3e6, 0, 12)]),
    Case("free_naca2412_3e6", "NACA 2412, free transition (e^N, N_crit 9)", "XFOIL (free transition)",
         "single", ("free",), (7e5, 6e6), (0.0, 0.3), (-2.0, 10.0), 0.03, 0.025,
         "transition_report.txt", "Cd within −2.5…+1.7 %, transition within ~0.01 c.",
         tested=[(3e6, 0, a) for a in range(-2, 11, 2)]),
    Case("lowre_e387_3e5", "Eppler 387, low-Re mode, Re 2e5–3e5", "NASA Langley LTPT (TM-4062), XFOIL",
         "single", ("free",), (1.5e5, 5e5), (0.0, 0.2), (-2.0, 8.0), 0.06, 0.08, "lowre_report.md",
         "Mean |ΔCd| 6 % (3e5) and 9 % (2e5); α 6° is +29 % where the bubble is unsteady.",
         tested=[(r, 0, a) for r in (2e5, 3e5) for a in range(-2, 9, 2)]),
    Case("lowre_e387_1e5", "Eppler 387, low-Re mode, Re 1e5", "NASA Langley LTPT (TM-4062)",
         "single", ("free",), (8e4, 1.5e5), (0.0, 0.2), (-2.0, 8.0), 0.12, 0.47, "lowre_report.md",
         "Mean |ΔCd| 47 %: at α 4–6° the mid-chord bubble bursts in steady RANS (Cd ~2.5×).",
         tested=[(1e5, 0, a) for a in range(-2, 9, 2)]),
    Case("laminar_naca0012", "NACA 0012, laminar, Re 1000–5000", "XFOIL; Di Ilio et al. 2020; Kurtulus 2015",
         "single", ("laminar",), (5e2, 1e4), (0.0, 0.1), (0.0, 7.0), 0.03, 0.03, "airfoil_report.txt",
         "Cd within ~3 % of XFOIL and the LBM reference; leading-edge suction peak 12–15 % low.",
         tested=[(1e3, 0, a) for a in range(0, 8)]),
    Case("multi_30p30n", "30P30N three-element high-lift airfoil, fully turbulent",
         "JAXA LWT2 wind tunnel (Cp); SU2 V&V (forces)", "multi", ("off", "fixed", "free"),
         (1e6, 1e7), (0.0, 0.25), (4.0, 14.0), 0.02, 0.05, "multi_30p30n_report.txt",
         "Cl +0.04 %, Cd −3.5 % vs SU2 grid-converged; slat suction peak and cove are the weak spot.",
         tested=[(1.71e6, 0.17, a) for a in (5.5, 9.5, 14)]),
    Case("rae2822_case9", "RAE 2822 case 9, transonic, SA", "Cook, McDonald & Firmin (AGARD AR-138)",
         "single", ("off", "fixed"), (3e6, 1.2e7), (0.65, 0.76), (1.5, 3.5), 0.014, 0.20,
         "transonic_report.txt",
         "Cl +1.4 %, Cm within 1 %; Cd +20 % against the experiment (shock 2–3 % c aft) — typical of "
         "SA, published SA results are Cd 0.018–0.020.", compressible=True,
         tested=[(6.5e6, 0.734, 2.79)]),
    Case("subsonic_compressible", "Compressible solver at subsonic Mach (Euler NACA 0012 M 0.5, Ni bump)",
         "AGARD AR-211; Vassberg & Jameson 2010", "single", ("off", "fixed", "free"), (1e6, 1.2e7),
         (0.3, 0.65), (-2.0, 6.0), 0.02, 0.05, "transonic_report.txt",
         "Euler NACA 0012 M 0.5: Cl 0.178 vs ≈0.18; viscous subsonic runs inherit the SA validation.",
         compressible=True, tested=[(0, 0.5, 1.25)]),
]

# what each scaled unit of extrapolation means: a factor ~1.8 in Re, 0.05 in Mach, 2° in alpha
SCALE = {"re": 0.25, "mach": 0.05, "alpha": 2.0}
T_C_RANGE = (0.07, 0.16)         # thickness ratios of the validated sections (E387 9 %, NACA/RAE 12 %)


def _excess(v, lo, hi):
    return lo - v if v < lo else v - hi if v > hi else 0.0


def envelope_distance(case, re, mach, alpha):
    """0 inside the case's range; otherwise the root-sum-square of the scaled distances"""
    dr = _excess(math.log10(max(re, 1.0)), math.log10(case.re[0]), math.log10(case.re[1])) / SCALE["re"]
    dm = _excess(mach, *case.mach) / SCALE["mach"]
    da = _excess(alpha, *case.alpha) / SCALE["alpha"]
    return math.sqrt(dr * dr + dm * dm + da * da)


def nearest_case(re, mach, alpha, transition="free", elements="single", compressible=None):
    """closest validated case with the same transition mode and element count"""
    pool = [c for c in CASES if c.elements == elements and transition in c.transition]
    if compressible is not None and mach >= 0.3:
        pool = [c for c in pool if c.compressible == compressible] or pool
    if not pool:
        pool = [c for c in CASES if c.elements == elements] or CASES
        penalty = 1.0                    # another transition mode: weaker evidence
    else:
        penalty = 0.0
    best = min(pool, key=lambda c: envelope_distance(c, re, mach, alpha))
    return best, envelope_distance(best, re, mach, alpha) + penalty


# ----------------------------------------------------------------------
# 2. assessment of a result
# ----------------------------------------------------------------------
@dataclass
class Assessment:
    level: str
    cl_band: float                # expected relative uncertainty (fraction)
    cd_band: float
    case: str                     # nearest validated case id
    case_title: str
    distance: float               # 0 = inside its range
    reasons: list                 # [(level, text)]

    @property
    def label(self):
        return LABEL[self.level]

    def to_dict(self):
        d = asdict(self)
        d["label"] = self.label
        return d


def transition_mode(settings):
    """'free' / 'fixed' / 'off' / 'laminar' from polar settings (new or old polar.json)"""
    s = settings or {}
    if s.get("turbulent") is False:
        return "laminar"
    if s.get("transition_mode"):
        return s["transition_mode"]
    t = str(s.get("transition", "")).lower()
    if "fixed" in t:
        return "fixed"
    if "fully" in t or t in ("off", "turbulent"):
        return "off"
    if "laminar" == t:
        return "laminar"
    return "free"


def assess_point(re, mach, alpha, *, transition="free", elements="single", compressible=None,
                 prandtl_glauert=False, refine=1.0, status="converged", alpha_clmax=None,
                 clmax_is_last=False, yplus_max=None, m_max=None, t_c=None, low_re_mode=None,
                 transition_note=None, gci=None):
    """confidence of one polar point. Everything except re/mach/alpha is optional."""
    if compressible is None:
        compressible = mach >= 0.3 and not prandtl_glauert
    case, d = nearest_case(re, mach, alpha, transition, elements, compressible)
    reasons = []
    if d < 1e-9:
        lev = "green" if case.cd_err <= 0.10 else "amber" if case.cd_err <= 0.25 else "red"
        reasons.append((lev, f"Inside the validated range of: {case.title}. {case.note}"))
    elif d <= 1.5:
        lev = "amber" if case.cd_err <= 0.25 else "red"
        reasons.append((lev, f"Close to, but outside, the validated range of: {case.title} "
                             f"(Re {case.re[0]:.2g}–{case.re[1]:.2g}, M {case.mach[0]:g}–{case.mach[1]:g}, "
                             f"α {case.alpha[0]:g}–{case.alpha[1]:g}°). Error grows with the distance."))
    else:
        lev = "red"
        reasons.append(("red", f"Outside every validated case (nearest: {case.title}). "
                               "No reference data supports this point: check it against an experiment, "
                               "XFOIL or a finer model before using it."))
    cl_b = case.cl_err * (1 + d)
    cd_b = case.cd_err * (1 + d)

    st = str(status or "")
    if st.startswith("diverged") or st in ("failed", "nan"):
        reasons.append(("red", "The solution diverged: no result."))
    elif "unsteady" in st or st.startswith("not converged"):
        reasons.append(("red", "No steady state: the values are means over oscillating iterations. "
                               "Steady RANS is not reliable here (separated or unsteady flow)."))
        cl_b += 0.05
        cd_b += 0.25
    elif "transition not converged" in st:
        reasons.append(("amber", "The transition point was still moving at the iteration limit."))
        cd_b += 0.05
    elif "small oscillation" in st:
        reasons.append(("green", "Converged to a small oscillation: the mean is reported."))

    if alpha_clmax is not None and not clmax_is_last:
        if alpha > alpha_clmax + 1e-9:
            reasons.append(("amber", "Past the lift peak. Spalart–Allmaras tends to over-predict Cl_max "
                                     "and post-stall lift for trailing-edge stall: compare with wind-tunnel "
                                     "data before designing to it."))
            cl_b += 0.10
            cd_b += 0.20
        elif abs(alpha - alpha_clmax) < 1e-9:
            reasons.append(("amber", "At Cl_max: SA usually over-predicts the maximum lift by a few percent."))
            cl_b += 0.05
    if prandtl_glauert and mach > 0.3:
        reasons.append(("red" if mach > 0.5 else "amber",
                        f"Prandtl–Glauert correction at M {mach:.2f}: valid up to ~0.3; use the "
                        "compressible solver."))
        cl_b += 0.02 if mach <= 0.5 else 0.08
    if m_max is not None and m_max > 1.0:
        reasons.append(("amber", f"Supersonic pocket (M_max {m_max:.2f}) with a shock. On RAE 2822 the "
                                 "shock sits 2–3 % c aft of the experiment and Cd is ~20 % high (typical of SA)."))
        cd_b = max(cd_b, 0.20)
    if transition == "off" and elements == "single" and re < 1e7 and transition != "laminar":
        reasons.append(("info", "Fully turbulent: Cd is higher than for a real section with laminar "
                                 "flow (use free transition to predict that)."))
    if transition == "laminar" and re >= 2e4:
        reasons.append(("amber", "Laminar solver at Re ≥ 2e4: a separated laminar layer cannot "
                                 "reattach, so bubbles become open separation."))
    if (low_re_mode or (transition == "free" and re < 5e5)) and re < 1.5e5:
        reasons.append(("amber", "Re below 1.5e5: mid-chord laminar bubbles may burst in steady RANS "
                                 "(E387 at Re 1e5: Cd up to 2.5×). Check against XFOIL or tests."))
    if re < 8e4 and transition != "laminar":
        reasons.append(("red", "Re 2e4–8e4 has no validation data yet (SD7003 at 6e4 is the planned case)."))
    if transition_note:
        reasons.append(("amber", transition_note))
    if refine is not None and refine < 0.95:
        reasons.append(("amber", f"Coarse mesh (level {refine:g}): on NACA 0012 the coarse level is "
                                 "+2.5 % in Cd. Use level 1 for numbers you quote."))
        cd_b += 0.02
        cl_b += 0.005
    if yplus_max is not None and yplus_max > 1.5:
        reasons.append(("amber", f"y+ max {yplus_max:.1f} (> 1): wall cells too thick for SA without "
                                 "wall functions; skin friction is under-resolved."))
        cd_b += 0.03
    if t_c is not None and not (T_C_RANGE[0] <= t_c <= T_C_RANGE[1]):
        reasons.append(("amber", f"Thickness {100 * t_c:.1f} % c is outside the validated sections "
                                 f"({100 * T_C_RANGE[0]:.0f}–{100 * T_C_RANGE[1]:.0f} %)."))
        cl_b *= 1.3
        cd_b *= 1.3
    if elements == "multi" and alpha > 12:
        reasons.append(("amber", "High-lift configuration near Cl_max: slat-cove recirculation and "
                                 "confluent boundary layers are where steady RANS is weakest."))
    if gci:
        g = gci.get("Cd")
        if g and math.isfinite(g.get("gci_fine", math.nan)):
            cd_b = max(cd_b, g["gci_fine"])
            pu = g.get("p_gci", g["p"])
            msg = f"Grid study: GCI {100 * g['gci_fine']:.2f} % on Cd (apparent order {g['p']:.2f}" + \
                (f", {pu:g} used)" if abs(pu - g["p"]) > 1e-9 else ")")
            lev = "green" if g["gci_fine"] < 0.02 else "amber" if g["gci_fine"] < 0.06 else "red"
            if g.get("oscillatory"):
                msg += ", oscillatory convergence"
            reasons.append((lev, msg + "."))
    level = worst(*(r[0] for r in reasons))
    return Assessment(level, cl_b, cd_b, case.id, case.title, round(d, 3), reasons)


def assess_polar(d, gci_by_alpha=None, re=None, mach=None, transition=None):
    """assess every point of a polar.json dict (airfoil.py output) or a list of rows (wing
    cache: pass re/mach/transition). Returns {"level", "counts", "points", "context"}."""
    if isinstance(d, list):
        d = {"results": d, "settings": {}, "info": {}}
    s = d.get("settings") or {}
    info = d.get("info") or {}
    res = d.get("results") or []
    re = float(re or s.get("re") or d.get("re") or 0) or None
    mach = float(mach if mach is not None else (s.get("mach") or (res[0].get("mach") if res else 0) or 0))
    if transition:
        tr = transition_mode({"transition": transition})
    elif s:
        tr = transition_mode(s)
    else:
        tr = transition_mode({"transition": (res[0].get("transition") if res else None) or "free"})
    elements = "multi" if s.get("multi") or "+" in str(s.get("airfoil", "")) or \
        "30p30n" in str(s.get("airfoil", info.get("source", ""))).lower() else "single"
    ok = [r for r in res if r.get("Cl") is not None and math.isfinite(r["Cl"])]
    a_max = max(ok, key=lambda r: r["Cl"])["alpha"] if ok else None
    is_last = bool(ok) and a_max == max(r["alpha"] for r in ok)
    pts = []
    for r in res:
        if re is None:
            break
        note = None
        if str(r.get("transition_converged_by") or "").startswith("averaged"):
            note = "Transition point cycled around a leading-edge bubble and was fixed at its mean."
        a = assess_point(re, mach, r["alpha"], transition=tr, elements=elements,
                         prandtl_glauert=bool(s.get("prandtl_glauert")), refine=s.get("refine"),
                         status=r.get("status", "converged"),
                         alpha_clmax=a_max if r.get("Cl") is not None else None, clmax_is_last=is_last,
                         yplus_max=r.get("yplus_max"), m_max=r.get("M_max"),
                         t_c=info.get("max_thickness"), low_re_mode=s.get("low_re_mode"),
                         transition_note=note,
                         gci=(gci_by_alpha or {}).get(_akey(r["alpha"])))
        pts.append({"alpha": r["alpha"], **a.to_dict()})
    counts = {l: sum(p["level"] == l for p in pts) for l in LEVELS}
    return {"level": worst(*(p["level"] for p in pts)) if pts else "amber", "counts": counts,
            "points": pts, "context": {"re": re, "mach": mach, "transition": tr, "elements": elements}}


def _akey(a):
    return f"{float(a):g}"


def summary_line(ass):
    c = ass["counts"]
    n = sum(c.values())
    if not n:
        return "No points to assess."
    parts = [f"{c[l]} {LABEL[l]}" for l in LEVELS if c[l]]
    return f"{n} points: " + ", ".join(parts)


# ----------------------------------------------------------------------
# tool-level evidence and context notes (non-polar tools)
# ----------------------------------------------------------------------
EVIDENCE = {
    "wing": ("green", "Vortex lattice + CFD sections. Lift slope within 0.4–6 % of DATCOM (AR 1–20, sweep "
                      "0–60°), elliptic wing e = 1.000, compressibility ratio within 1.5 %.", "wing_report.txt"),
    "stability": ("green", "Against AeroSandbox VLM: CL, CLα, x_np within 0.5 %; Cmα 1.6 %, Cnβ 0.5 %, "
                           "Clβ 7 %, Clp 3 %. Modes within 3–7 % of the classical approximations.",
                  "aircraft_report.md"),
    "loads": ("amber", "Simplified CS-23 envelope, symmetric manoeuvres and gusts only (no rolling, fin or "
                       "asymmetric gust cases yet). Root shear check against half the wing lift exact.",
              "aircraft_report.md"),
    "structure": ("amber", "First-pass box sizing: no skin/stringer buckling, joints, cut-outs or fatigue. "
                           "Beam frequencies and divergence within 0.2 % of closed forms.", "aircraft_report.md"),
    "aeroelastic": ("amber", "Goland wing flutter 137.1 m/s vs 137.2 (strip p-k); lattice vs Theodorsen within "
                             "1.5–6 %. Incompressible, clamped root, no structural damping (conservative).",
                    "aeroelastic_report.md"),
    "performance": ("green", "Trimmed nonlinear polar; Breguet at constant speed and altitude. Sizing → "
                             "performance loop agrees within 1–5 % on stall, landing, Oswald e.",
                    "design_tools_report.md"),
    "drag": ("green", "Cessna 172-like geometry: CD0 0.0273 vs 0.027–0.031 quoted.", "design_tools_report.md"),
    "sizing": ("amber", "Statistical (Raymer Class II): Cessna 172 MTOW −14 %, Velis Electro +10 %.",
               "design_tools_report.md"),
    "weight_balance": ("green", "Component build-up; scissor plot from the vortex lattice (linear, no ground "
                                "effect, no tail stall: keep margin).", "design_tools_report.md"),
    "drag_rise": ("amber", "NACA 0012 Cl 0.3: M_dd 0.722 vs Korn's equation 0.720; SA drag is high once a shock "
                           "forms (RAE 2822 +20 %).", "design_report.md"),
}


def tool_notes(tool, **ctx):
    """[(level, text)] for a tool and the context of the current run"""
    notes = []
    ev = EVIDENCE.get(tool)
    if ev:
        notes.append((ev[0], ev[1]))
    mach = ctx.get("mach")
    if tool == "wing":
        if ctx.get("alpha") is not None and ctx.get("alpha_clmax") is not None and \
                ctx["alpha"] > ctx["alpha_clmax"]:
            notes.append(("amber", "Past C_L,max the strip-theory solution is not unique: trend only."))
        if ctx.get("aspect_ratio") is not None and ctx["aspect_ratio"] < 3:
            notes.append(("amber", f"Aspect ratio {ctx['aspect_ratio']:.1f}: leading-edge vortex lift is "
                                   "not modelled (attached-flow lattice), C_L is low at high α."))
        if mach is not None and mach > 0.6:
            notes.append(("red", f"M {mach:.2f}: Prandtl–Glauert/Göthert on the lattice is not valid with "
                                 "shocks; use the compressible section polars and expect errors."))
    if tool == "stability":
        notes.append(("info", "Hinge moments come from the inviscid lattice: real values are usually smaller "
                               "in magnitude. No propeller slipstream or wing-body carry-over."))
    if tool == "aeroelastic":
        if mach is not None and mach > 0.3:
            notes.append(("red", f"M {mach:.2f}: the unsteady lattice is incompressible; flutter speeds "
                                 "above M 0.3 are not conservative."))
        if ctx.get("antisymmetric"):
            notes.append(("amber", "Antisymmetric/aileron modes with a clamped root: the fuselage roll "
                                   "freedom is missing (it usually raises the flutter speed)."))
        notes.append(("amber", "Screening level: confirm with a ground vibration test and a certified "
                               "flutter analysis before flight."))
    if tool == "structure":
        if ctx.get("flutter_margin") is not None and ctx["flutter_margin"] < 1.0:
            notes.append(("red", "Flutter estimate below 1.2 V_D."))
    if tool == "performance" and ctx.get("flap_source", "").lower().startswith("raymer"):
        notes.append(("amber", "Flap C_L,max increment from Raymer's statistics, not from CFD sections."))
    if tool == "loads" and ctx.get("flexible") is False:
        notes.append(("info", "Rigid wing; the flexible option changes root bending by 1–3 % on the demo."))
    return notes


# ----------------------------------------------------------------------
# 3. grid convergence index (Celik et al. 2008)
# ----------------------------------------------------------------------
def gci(phi, h, fs=1.25, max_iter=200, p_formal=2.0):
    """phi, h: values and representative cell sizes on three grids, finest first
    (h1 < h2 < h3). Returns the apparent order p, the extrapolated value, the relative
    errors and the fine-grid GCI (fractions). The GCI uses min(p, p_formal): an apparent
    order above the scheme's formal order (2) is usually luck, not accuracy (Roache)."""
    f1, f2, f3 = (float(x) for x in phi)
    h1, h2, h3 = (float(x) for x in h)
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    out = {"phi": [f1, f2, f3], "r21": r21, "r32": r32}
    if abs(e21) < 1e-14 * max(abs(f1), 1e-30) or abs(e32) < 1e-30:
        out.update(p=float("nan"), phi_ext=f1, e_a21=abs(e21 / f1) if f1 else 0.0, gci_fine=0.0, gci_abs=0.0,
                   oscillatory=False, converged=True)
        return out
    s = 1.0 if e32 / e21 > 0 else -1.0
    p = abs(math.log(abs(e32 / e21))) / math.log(r21)
    for _ in range(max_iter):
        q = math.log((r21 ** p - s) / (r32 ** p - s)) if (r32 ** p - s) * (r21 ** p - s) > 0 else 0.0
        pn = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(pn - p) < 1e-10:
            p = pn
            break
        p = pn
    p = max(p, 0.5)                                    # guard a near-zero apparent order
    phi_ext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    e_a21 = abs((f1 - f2) / f1) if f1 else abs(f1 - f2)
    pg = min(p, p_formal) if p_formal else p
    out.update(p=p, p_gci=pg, phi_ext=phi_ext, e_a21=e_a21,
               e_ext21=abs((phi_ext - f1) / phi_ext) if phi_ext else float("nan"),
               gci_fine=fs * e_a21 / (r21 ** pg - 1), gci_abs=fs * abs(f1 - f2) / (r21 ** pg - 1),
               oscillatory=s < 0, converged=False)
    return out


def gci_from_cells(values, cells, dim=2, fs=1.25):
    """values and cell counts on three meshes in any order -> gci() with h = N^(-1/dim)"""
    z = sorted(zip(cells, values), key=lambda t: -t[0])          # finest first
    h = [n ** (-1.0 / dim) for n, _ in z]
    return gci([v for _, v in z], h, fs)


# ----------------------------------------------------------------------
# grid study of a polar run (levels 0.7 / 1 / 1.4)
# ----------------------------------------------------------------------
LEVELS_GRID = (0.7, 1.0, 1.4)


def rerun_args(settings, info, out, alpha, level, jobs=1, re=None):
    """airfoil.py arguments that repeat a polar at another mesh level"""
    s = settings or {}
    spec = s.get("airfoil") or info.get("source")
    re = s.get("re") or re
    if not spec or not re:
        raise ValueError("this polar does not record its airfoil and Re (rerun it with the current code)")
    a = ["airfoil.py", "polar", spec, "--re", f"{float(re):.6g}", f"--alpha={alpha}",
         "--refine", f"{level:g}", "--jobs", str(jobs), "--out", str(out), "--no-fields"]
    mode = transition_mode(s)
    if mode == "fixed" and s.get("xtr"):
        a += ["--xtr", s["xtr"]]
    elif mode == "off":
        a += ["--transition", "off"]
    elif s.get("ncrit"):
        a += ["--ncrit", f"{s['ncrit']:g}"]
    if s.get("regime") and s["regime"] != "auto":
        a += ["--regime", s["regime"]]
    if s.get("derotate"):
        a.append("--derotate")
    if s.get("mach"):
        a += ["--mach", f"{float(s['mach']):.4g}"]
        if s.get("prandtl_glauert"):
            a.append("--prandtl-glauert")
    if s.get("solver"):
        a += ["--solver", s["solver"]]
    if s.get("max_iter"):
        a += ["--max-iter", str(s["max_iter"])]
    return a


def grid_study(polar_dir, alphas, jobs=1, quantities=("Cl", "Cd", "Cm"), run=True):
    """rerun chosen angles of a polar at the two other mesh levels and compute the GCI.
    Writes <polar_dir>/_grid/grid_study.json; returns it."""
    polar_dir = Path(polar_dir)
    d = json.loads((polar_dir / "polar.json").read_text())
    s, info = d.get("settings") or {}, d.get("info") or {}
    base_level = float(s.get("refine") or 1.0)
    gdir = polar_dir / "_grid"
    gdir.mkdir(exist_ok=True)
    alist = ",".join(f"{float(a):g}" for a in alphas)
    rows = {base_level: {_akey(r["alpha"]): r for r in d["results"]}}
    for lev in LEVELS_GRID:
        if abs(lev - base_level) < 1e-9:
            continue
        out = gdir / f"L{lev:g}"
        if run and not (out / "polar.json").exists():
            print(f"[trust] mesh level {lev:g}: alpha {alist}", flush=True)
            subprocess.run([sys.executable, "-u"] + rerun_args(s, info, out, alist, lev, jobs, d.get("re")),
                           cwd=Path(__file__).parent, check=False)
        if (out / "polar.json").exists():
            rr = json.loads((out / "polar.json").read_text())["results"]
            rows[lev] = {_akey(r["alpha"]): r for r in rr}
    result = {"polar": str(polar_dir), "levels": sorted(rows), "alphas": {}}
    for a in alphas:
        k = _akey(a)
        pts = [(lev, rows[lev].get(k)) for lev in sorted(rows)]
        if len(pts) < 3 or any(p is None or p.get("Cl") is None for _, p in pts):
            result["alphas"][k] = {"error": "angle missing on one of the meshes"}
            continue
        cells = [p.get("cells") for _, p in pts]
        if any(c is None for c in cells):                # older polars: cells ~ level^2
            cells = [lev ** 2 for lev, _ in pts]
        entry = {"cells": cells, "status": [p.get("status") for _, p in pts]}
        for q in quantities:
            vals = [p.get(q) for _, p in pts]
            if all(v is not None and math.isfinite(v) for v in vals):
                entry[q] = gci_from_cells(vals, cells)
        result["alphas"][k] = entry
    (gdir / "grid_study.json").write_text(json.dumps(result, indent=1))
    return result


def load_grid_study(polar_dir):
    f = Path(polar_dir) / "_grid" / "grid_study.json"
    if not f.exists():
        return None
    g = json.loads(f.read_text())
    return {k: v for k, v in g["alphas"].items() if "error" not in v}


# ----------------------------------------------------------------------
def envelope_table():
    return [{"case": c.id, "title": c.title, "reference": c.reference, "elements": c.elements,
             "transition": "/".join(c.transition), "Re": f"{c.re[0]:.2g}–{c.re[1]:.2g}",
             "Mach": f"{c.mach[0]:g}–{c.mach[1]:g}", "alpha": f"{c.alpha[0]:g}…{c.alpha[1]:g}",
             "Cl error": c.cl_err, "Cd error": c.cd_err, "report": c.report, "note": c.note}
            for c in CASES]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="confidence of a flow condition")
    c.add_argument("--re", type=float, required=True)
    c.add_argument("--mach", type=float, default=0.0)
    c.add_argument("--alpha", type=float, required=True)
    c.add_argument("--transition", default="free", choices=["free", "fixed", "off", "laminar"])
    c.add_argument("--multi", action="store_true")
    c.add_argument("--refine", type=float, default=1.0)
    p = sub.add_parser("polar", help="assess every point of an airfoil.py polar")
    p.add_argument("polar_dir")
    g = sub.add_parser("grid-study", help="rerun angles at mesh levels 0.7/1/1.4 and compute the GCI")
    g.add_argument("polar_dir")
    g.add_argument("--alpha", required=True, help="comma-separated angles")
    g.add_argument("--jobs", type=int, default=1)
    sub.add_parser("envelope", help="list the validated cases")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        r = assess_point(a.re, a.mach, a.alpha, transition=a.transition,
                         elements="multi" if a.multi else "single", refine=a.refine)
        print(f"{r.level.upper()} ({r.label}): Cl ±{100 * r.cl_band:.1f} %, Cd ±{100 * r.cd_band:.1f} %")
        for lev, t in r.reasons:
            print(f"  [{lev}] {t}")
    elif a.cmd == "polar":
        d = json.loads((Path(a.polar_dir) / "polar.json").read_text())
        ass = assess_polar(d, load_grid_study(a.polar_dir))
        print(summary_line(ass))
        for p_ in ass["points"]:
            print(f"  α {p_['alpha']:6g}  {p_['level']:5s}  Cl ±{100 * p_['cl_band']:.1f} %  "
                  f"Cd ±{100 * p_['cd_band']:.1f} %   {p_['reasons'][-1][1][:90]}")
    elif a.cmd == "grid-study":
        res = grid_study(a.polar_dir, [float(x) for x in a.alpha.split(",")], a.jobs)
        for k, v in res["alphas"].items():
            if "error" in v:
                print(f"  α {k}: {v['error']}")
                continue
            print(f"  α {k}: " + "  ".join(f"{q} GCI {100 * v[q]['gci_fine']:.2f} % (p {v[q]['p']:.2f}, "
                                            f"ext {v[q]['phi_ext']:.5g})" for q in ("Cl", "Cd") if q in v))
        print(f"[trust] -> {Path(a.polar_dir) / '_grid' / 'grid_study.json'}")
    else:
        for r in envelope_table():
            print(f"{r['case']:26s} {r['Re']:>14s}  M {r['Mach']:>9s}  α {r['alpha']:>9s}  "
                  f"Cd ±{100 * r['Cd error']:.0f} %  {r['title']}")


if __name__ == "__main__":
    main()
