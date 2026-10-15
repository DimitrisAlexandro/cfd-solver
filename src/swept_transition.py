"""
Transition on swept wings: attachment-line contamination and instability,
crossflow and Tollmien-Schlichting (streamwise) transition of an infinite
swept wing section.

    python swept_transition.py naca0012 --sweep 30 --re 6e6 --alpha 2 --tu 0.1
    python swept_transition.py naca64a010 --sweep 35 --re 2e7 --alpha 0 --contaminated

Infinite swept wing (independence principle): the chordwise boundary layer
develops as a two-dimensional one in the flow normal to the leading edge,
with V_n = V cos(sweep), and the spanwise velocity W = V sin(sweep) is
constant at the edge.

  * Edge velocity: inviscid panel method (panel.py) on the section normal to
    the leading edge at alpha_n = atan(tan alpha / cos sweep).
  * Attachment line (Poll 1979, 1985): Rbar = W / sqrt(nu dU_e/dx) at the
    stagnation line. Turbulence from the wing root or a fuselage spreads
    along it ("contamination") when Rbar > ~245 (Pfenninger 1965, Poll 1979);
    the clean attachment line becomes unstable at Rbar ~ 580 (Poll 1979; linear
    theory, Hall, Malik & Poll 1984: ~583). Either -> turbulent from x = 0.
  * Crossflow: the crossflow velocity profile of the three-dimensional
    laminar layer from Falkner-Skan-Cooke local similarity (fsc.py): at every
    station the member with the station's shape factor H (integral laminar
    boundary layer, so the upstream history enters through H and theta),
    the skew from W/U_e. Crossflow Reynolds number (Beasley 1973; Owen &
    Randall 1952) chi = w_cf,max delta_10 / nu, delta_10 = height where w_cf
    has fallen to 10 % of its maximum. Transition where chi reaches chi_tr
    (default 200; the published values scatter from ~150 to ~250 with
    surface finish and free-stream disturbances: set it with --chi).
    (Matching the local Hartree parameter m = (x/U_e) dU_e/dx instead drove
    the profile to separation in every decelerating region: chi ~ 5000.)
  * Tollmien-Schlichting: e^N envelope method (transition.py) on the
    chordwise layer at the normal Reynolds number Re_c cos(sweep); N_crit from
    the free-stream turbulence level by Mack's relation
    N_crit = -8.43 - 2.4 ln(Tu) when --tu is given.
  * Laminar separation first -> separation-induced transition (bubble length
    as in transition.py).

The first mechanism along each surface sets the transition location. Limits:
local similarity for the crossflow profile (an estimate, not a crossflow
stability analysis), infinite-swept-wing flow (no taper / tip / root
effects), incompressible.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import airfoil_geom as ag
import cst
import fsc
import panel
import transition as T

RBAR_CONTAMINATION = 245.0
RBAR_INSTABILITY = 580.0
CHI_TR = 200.0


def section_outline(spec, n=200):
    """smooth, cosine-spaced open outline (CST fit of the airfoil)"""
    au, al, _ = cst.fit(ag.sharp_outline(spec), 10)
    return cst.outline(au, al, n)


def edge_velocity(spec, alpha_n, n=200):
    """surfaces from the attachment line: {"upper"/"lower": (s, x, ue)},
    ue = U_e / V_n, s, x in chords"""
    o = section_outline(spec, n)
    pr = panel.solve([o], alpha_n)
    e = pr["elements"][0]
    x, y, vt = e["x"], e["y"], e["Vt"]
    i_le = int(np.argmin(x))
    ch = np.where(np.sign(vt[:-1]) != np.sign(vt[1:]))[0]
    i_st = int(ch[np.argmin(np.abs(ch - i_le))]) if len(ch) else i_le
    # stagnation point between panels i_st and i_st+1 (linear in vt)
    f = vt[i_st] / (vt[i_st] - vt[i_st + 1])
    xs = x[i_st] + f * (x[i_st + 1] - x[i_st])
    ys = y[i_st] + f * (y[i_st + 1] - y[i_st])
    out = {"stagnation": (float(xs), float(ys))}
    for name, sel in (("upper", np.arange(i_st, -1, -1)), ("lower", np.arange(i_st + 1, len(x)))):
        P = np.c_[np.r_[xs, x[sel]], np.r_[ys, y[sel]]]
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        ue = np.r_[0.0, np.abs(vt[sel])]
        out[name] = (s, P[:, 0], ue)
    return out, pr


def attachment_line(surf, sweep, re_c):
    """Rbar = W / sqrt(nu dU_e/dx): slope of U_e at the stagnation line from
    the first stations of both surfaces"""
    lam = math.radians(sweep)
    slopes = []
    for side in ("upper", "lower"):
        s, x, ue = surf[side]
        k = min(4, len(s) - 1)
        slopes.append(np.polyfit(s[:k + 1], ue[:k + 1], 1)[0])
    due = float(np.mean(slopes))                     # (dU_e/ds) c / V_n
    if lam == 0:
        return 0.0, due
    rbar = math.sin(lam) * math.sqrt(re_c / (math.cos(lam) * max(due, 1e-9)))
    return rbar, due


def crossflow_chi(H, theta, ue, sweep, re_n):
    """chi = w_cf,max delta_10 / nu along a surface: at every station the
    Falkner-Skan-Cooke member with the station's shape factor H (from the
    integral laminar boundary layer, which carries the upstream history),
    delta_10 scaled with the station's momentum thickness theta:
        chi = (w_max / Q_e) Q_e (eta_10 / theta_eta) theta Re_n
    (speeds in V_n, lengths in chords)"""
    lam = math.radians(sweep)
    n = len(H)
    chi = np.zeros(n)
    beta = np.full(n, np.nan)
    if lam == 0:
        return chi, beta
    t = fsc.table()
    o = np.argsort(t["H"])
    tl = math.tan(lam)
    for j in range(n):
        if not (np.isfinite(H[j]) and np.isfinite(theta[j]) and ue[j] > 1e-3):
            continue
        b = float(np.interp(H[j], t["H"][o], t["beta"][o]))
        b = round(b, 3)
        w, _, e10 = fsc.crossflow(b, tl / ue[j])
        th_eta = fsc.integrals(b)["theta"]
        q = math.sqrt(ue[j] ** 2 + tl**2)
        chi[j] = w * q * e10 / th_eta * theta[j] * re_n
        beta[j] = b
    return chi, beta


def analyse(spec, sweep, re_c, alpha, tu=None, ncrit=None, chi_tr=CHI_TR, contaminated=False,
            rbar_cont=RBAR_CONTAMINATION, rbar_inst=RBAR_INSTABILITY, n=200):
    """Transition on both surfaces of an infinite swept wing section.
    re_c: Reynolds number of the free-stream speed and the chord normal to
    the leading edge; alpha: free-stream angle of attack (deg)."""
    lam = math.radians(sweep)
    if ncrit is None:
        ncrit = T.ncrit_from_turbulence(tu) if tu else T.NCRIT
    alpha_n = math.degrees(math.atan(math.tan(math.radians(alpha)) / math.cos(lam)))
    re_n = re_c * math.cos(lam)
    surf, pr = edge_velocity(spec, alpha_n, n)
    rbar, due = attachment_line(surf, sweep, re_c)
    out = {"airfoil": spec, "sweep": sweep, "re_c": re_c, "re_n": re_n, "alpha": alpha,
           "alpha_n": alpha_n, "tu_percent": tu, "ncrit": ncrit, "chi_tr": chi_tr,
           "attachment_line": {"Rbar": rbar, "dUe_ds": due, "contaminated_input": contaminated,
                               "limit_contamination": rbar_cont, "limit_instability": rbar_inst},
           "Cl_n": pr["Cl"]}
    al_turb = rbar > rbar_inst or (contaminated and rbar > rbar_cont)
    out["attachment_line"]["turbulent"] = bool(al_turb)
    out["attachment_line"]["why"] = ("unstable (Rbar > %g)" % rbar_inst if rbar > rbar_inst else
                                     "contaminated (Rbar > %g)" % rbar_cont if al_turb else
                                     "laminar")
    for side in ("upper", "lower"):
        s, x, ue = surf[side]
        res = {"x": x.tolist(), "s": s.tolist(), "ue": ue.tolist()}
        cands = []
        if al_turb:
            cands.append((0.0, "attachment line " + out["attachment_line"]["why"]))
        # TS: e^N on the chordwise layer
        i0 = T.march_start(ue)
        ss, uu, xx = s[i0:] - s[i0], ue[i0:], x[i0:]
        th, H, ret, sep = T.laminar_bl(ss, uu, re_n)
        end = max(sep if sep is not None else len(th), 2)
        N, _ = T.integrate_N(ss[:end], H[:end], ret[:end], th[:end])
        hit = np.where(N >= ncrit)[0]
        if len(hit):
            j = hit[0]
            fj = (ncrit - N[j - 1]) / max(N[j] - N[j - 1], 1e-12)
            cands.append((float(xx[j - 1] + fj * (xx[j] - xx[j - 1])), "Tollmien-Schlichting (e^N)"))
        elif sep is not None:
            L = T.bubble_length(re_n, uu[sep], th[max(sep - 1, 1)], "mayle")
            cands.append((float(np.interp(ss[sep] + L, ss, xx)), "laminar separation"))
        res["N"] = np.r_[np.zeros(i0), N, np.full(len(s) - i0 - len(N), np.nan)].tolist()
        # crossflow (laminar part, up to separation)
        Hs = np.r_[np.full(i0, np.nan), H, np.full(len(s) - i0 - len(H), np.nan)]
        ths = np.r_[np.full(i0, np.nan), th, np.full(len(s) - i0 - len(th), np.nan)]
        chi, beta = crossflow_chi(Hs, ths, ue, sweep, re_n)
        res["chi"] = chi.tolist()
        hit = np.where(chi >= chi_tr)[0]
        if len(hit):
            j = hit[0]
            if j == 0:
                cands.append((float(x[0]), "crossflow"))
            else:
                fj = (chi_tr - chi[j - 1]) / max(chi[j] - chi[j - 1], 1e-12)
                cands.append((float(x[j - 1] + fj * (x[j] - x[j - 1])), "crossflow"))
        res["chi_max"] = float(chi.max()) if len(chi) else 0.0
        if cands:
            # first along the surface = smallest arc length; x is monotone
            # downstream of the leading edge, use the arc length to compare
            def s_of(xv):
                k = int(np.argmin(np.abs(x - xv)))
                return s[k]
            xt, how = min(cands, key=lambda c: s_of(c[0]))
        else:
            xt, how = 1.0, "laminar to the trailing edge"
        res.update(x_tr=float(xt), mechanism=how,
                   candidates=[{"x": float(c[0]), "mechanism": c[1]} for c in cands])
        out[side] = res
    return out


def plot(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for side, st in (("upper", "-"), ("lower", "--")):
        d = r[side]
        x = np.array(d["x"])
        ax[0].plot(x, np.array(d["ue"]), st, label=side)
        ax[1].plot(x, np.array(d["N"]), st, label=side)
        ax[2].plot(x, np.array(d["chi"]), st, label=side)
        for a in ax:
            a.axvline(d["x_tr"], color="C3" if side == "upper" else "C2", lw=0.8, ls=st)
    ax[1].axhline(r["ncrit"], color="k", lw=0.6)
    ax[2].axhline(r["chi_tr"], color="k", lw=0.6)
    ax[0].set_ylabel("U_e / V_n"); ax[1].set_ylabel("N (Tollmien-Schlichting)")
    ax[2].set_ylabel("crossflow Reynolds number chi")
    for a in ax:
        a.set_xlabel("x/c"); a.grid(alpha=.3); a.legend(fontsize=8)
    al = r["attachment_line"]
    fig.suptitle(f"{r['airfoil']}, sweep {r['sweep']:g} deg, Re {r['re_c']:.3g}, alpha {r['alpha']:g}: "
                 f"attachment line Rbar {al['Rbar']:.0f} ({al['why']}); transition upper "
                 f"{r['upper']['x_tr']:.3f} ({r['upper']['mechanism']}), lower "
                 f"{r['lower']['x_tr']:.3f} ({r['lower']['mechanism']})", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("airfoil")
    ap.add_argument("--sweep", type=float, required=True, help="leading-edge sweep (deg)")
    ap.add_argument("--re", type=float, required=True,
                    help="V c / nu, chord normal to the leading edge")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--tu", type=float, help="free-stream turbulence (%%): N_crit by Mack")
    ap.add_argument("--ncrit", type=float)
    ap.add_argument("--chi", type=float, default=CHI_TR, help="crossflow transition chi")
    ap.add_argument("--contaminated", action="store_true",
                    help="turbulence from the root / fuselage reaches the attachment line")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    r = analyse(a.airfoil, a.sweep, a.re, a.alpha, a.tu, a.ncrit, a.chi, a.contaminated)
    al = r["attachment_line"]
    print(f"attachment line: Rbar {al['Rbar']:.0f} -> {al['why']}  (N_crit {r['ncrit']:.2f})")
    for side in ("upper", "lower"):
        d = r[side]
        print(f"{side}: transition x/c {d['x_tr']:.3f} ({d['mechanism']}); max chi {d['chi_max']:.0f}; "
              + "; ".join(f"{c['mechanism']} {c['x']:.3f}" for c in d["candidates"]))
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "swept_transition.json").write_text(json.dumps(r, indent=1))
        plot(r, out / "swept_transition.png")
    return r


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# along the span of a wing (wing.py)
# ----------------------------------------------------------------------
def wing_transition(wing, sol, re_strip, airfoil, tip_airfoil=None, tu=None, ncrit=None,
                    contaminated=False, chi_tr=CHI_TR, n_eval=8):
    """Transition along the span of a quasi-3D wing solution (wing.solve):
    at n_eval stations of the right half the local leading-edge sweep (from
    the quarter-chord sweep and the taper), the chord normal to the leading
    edge, its Reynolds number and the strip's effective angle (the
    decambered section angle of the vortex-lattice solution) -> analyse().
    contaminated: the root (fuselage junction) feeds turbulence onto the
    attachment line."""
    half = np.where(wing.eta > 0)[0]
    pick = half[np.linspace(0, len(half) - 1, n_eval).round().astype(int)]
    s = 0.5 * wing.b
    tlam = math.tan(math.radians(wing.sweep)) + 0.25 * (wing.cr - wing.ct) / s
    lam_le = math.degrees(math.atan(tlam))
    rows = []
    for i in pick:
        eta = float(wing.eta[i])
        foil = tip_airfoil if (tip_airfoil and eta > 0.5) else airfoil
        cn = float(wing.c[i]) * math.cos(math.radians(lam_le))
        re_n_chord = float(re_strip[i]) * math.cos(math.radians(lam_le))   # V c_n / nu
        a_eff = float(sol["span"]["alpha_eff"][i])
        r = analyse(foil, lam_le, re_n_chord, a_eff, tu, ncrit, chi_tr, contaminated)
        rows.append({"eta": eta, "chord": float(wing.c[i]), "chord_normal": cn,
                     "sweep_le": lam_le, "re": float(re_strip[i]), "alpha_eff": a_eff,
                     "Rbar": r["attachment_line"]["Rbar"],
                     "attachment_line": r["attachment_line"]["why"],
                     "x_tr_upper": r["upper"]["x_tr"], "mech_upper": r["upper"]["mechanism"],
                     "x_tr_lower": r["lower"]["x_tr"], "mech_lower": r["lower"]["mechanism"],
                     "chi_max_upper": r["upper"]["chi_max"], "ncrit": r["ncrit"]})
    # laminar fraction of the wetted area (both surfaces, area-weighted)
    w = np.array([wing.c[i] * abs(wing.dy[i]) for i in pick])
    lam = np.array([0.5 * (q["x_tr_upper"] + q["x_tr_lower"]) for q in rows])
    return {"sweep_le": lam_le, "stations": rows,
            "laminar_fraction": float((w * lam).sum() / w.sum())}


def wing_table(t):
    L = [f"Transition along the span (leading-edge sweep {t['sweep_le']:.1f} deg; "
         f"N_crit {t['stations'][0]['ncrit']:.2f}; laminar wetted fraction ~{100 * t['laminar_fraction']:.0f} %)",
         f"{'eta':>5s} {'Re':>9s} {'a_eff':>6s} {'Rbar':>5s}  {'x_tr up':>7s} {'mechanism':28s} "
         f"{'x_tr lo':>7s} mechanism"]
    for q in t["stations"]:
        L.append(f"{q['eta']:5.2f} {q['re']:9.3g} {q['alpha_eff']:6.2f} {q['Rbar']:5.0f}  "
                 f"{q['x_tr_upper']:7.3f} {q['mech_upper'][:28]:28s} {q['x_tr_lower']:7.3f} "
                 f"{q['mech_lower']}")
    return "\n".join(L)
