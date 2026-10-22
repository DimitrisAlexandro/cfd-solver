"""Flight dynamics of an aircraft (aircraft.py): the linear longitudinal and lateral modes
from the stability derivatives, with the MIL-F-8785C handling-quality checks, and the
linear time responses to control steps."""
import math

import numpy as np

G0 = 9.80665          # m/s², standard gravity (as in aircraft.py)


def modes(ac, an, der, trim):
    """eigenvalues of the longitudinal and lateral-directional small-
    perturbation equations (stability axes, Nelson ch. 4 / 5), thrust =
    drag and independent of speed, Ixz = 0. Returns the modes with natural
    frequency, damping, period / time constant, and the MIL-F-8785C
    (Class I, Category B) Level 1 limits."""
    V, rho = an.V, an.rho
    Q = 0.5 * rho * V**2
    S, b, c, m = ac.S, ac.b, ac.c, ac.mass
    Ix, Iy, Iz = ac.inertias()
    CL0, CD0 = trim["CL"], trim["CD"]
    CLa, CDa, Cma = der["CL_a"], der["CD_a"], der["Cm_a"]
    CLq, Cmq = der["CL_q"], der["Cm_q"]
    CLad, Cmad = der["CL_adot"], der["Cm_adot"]
    Xu = -2 * CD0 * Q * S / (m * V)
    Xw = -(CDa - CL0) * Q * S / (m * V)
    Zu = -2 * CL0 * Q * S / (m * V)
    Zw = -(CLa + CD0) * Q * S / (m * V)
    Zwd = -CLad * c / (2 * V) * Q * S / (V * m)
    Zq = -CLq * c / (2 * V) * Q * S / m
    Mw = Cma * Q * S * c / (V * Iy)
    Mwd = Cmad * c / (2 * V) * Q * S * c / (V * Iy)
    Mq = Cmq * c / (2 * V) * Q * S * c / Iy
    den = 1 - Zwd
    A_lon = np.array([
        [Xu, Xw, 0.0, -G0],
        [Zu / den, Zw / den, (V + Zq) / den, 0.0],
        [Mwd * Zu / den, Mw + Mwd * Zw / den, Mq + Mwd * (V + Zq) / den, 0.0],
        [0.0, 0.0, 1.0, 0.0]])
    Yb = Q * S * der["CY_b"] / m
    Yp = Q * S * b * der["CY_p"] / (2 * m * V)
    Yr = Q * S * b * der["CY_r"] / (2 * m * V)
    Lb = Q * S * b * der["Cl_b"] / Ix
    Lp = Q * S * b**2 * der["Cl_p"] / (2 * Ix * V)
    Lr = Q * S * b**2 * der["Cl_r"] / (2 * Ix * V)
    Nb = Q * S * b * der["Cn_b"] / Iz
    Np = Q * S * b**2 * der["Cn_p"] / (2 * Iz * V)
    Nr = Q * S * b**2 * der["Cn_r"] / (2 * Iz * V)
    A_lat = np.array([
        [Yb / V, Yp / V, -(1 - Yr / V), G0 / V],
        [Lb, Lp, Lr, 0.0],
        [Nb, Np, Nr, 0.0],
        [0.0, 1.0, 0.0, 0.0]])

    def describe(lam):
        lam = complex(lam)
        if abs(lam.imag) > 1e-9:
            wn = abs(lam)
            z = -lam.real / wn
            return {"real": lam.real, "imag": abs(lam.imag), "omega_n": wn, "zeta": z,
                    "period_s": 2 * math.pi / abs(lam.imag),
                    "t_half_s": math.log(2) / -lam.real if lam.real < 0 else None,
                    "t_double_s": math.log(2) / lam.real if lam.real > 0 else None}
        return {"real": lam.real, "imag": 0.0, "time_constant_s": -1 / lam.real if lam.real < 0 else None,
                "t_half_s": math.log(2) / -lam.real if lam.real < 0 else None,
                "t_double_s": math.log(2) / lam.real if lam.real > 0 else None}

    out = {}
    ev = np.linalg.eigvals(A_lon)
    cpx = sorted([e for e in ev if e.imag > 1e-9], key=lambda e: -abs(e))
    real = sorted([e.real for e in ev if abs(e.imag) <= 1e-9], key=lambda e: -abs(e))
    if len(cpx) == 2:
        out["short period"], out["phugoid"] = describe(cpx[0]), describe(cpx[1])
    else:
        if cpx:
            out["phugoid" if abs(cpx[0]) < 1.0 else "short period"] = describe(cpx[0])
        for i, e in enumerate(real):
            out[f"longitudinal real {i + 1}"] = describe(e)
    ev = np.linalg.eigvals(A_lat)
    cpx = [e for e in ev if e.imag > 1e-9]
    real = sorted([e.real for e in ev if abs(e.imag) <= 1e-9], key=lambda e: -abs(e))
    if cpx:
        out["dutch roll"] = describe(cpx[0])
    if len(real) >= 2:
        out["roll"], out["spiral"] = describe(real[0]), describe(real[-1])
    elif real:
        out["roll"] = describe(real[0])
    # MIL-F-8785C Class I (small light aircraft), Category B, Level 1
    chk = {}
    if "short period" in out:
        z = out["short period"]["zeta"]
        chk["short period"] = (0.30 <= z <= 2.0, "damping ratio 0.30 to 2.0")
    if "phugoid" in out:
        chk["phugoid"] = (out["phugoid"]["zeta"] >= 0.04, "damping ratio at least 0.04")
    if "dutch roll" in out:
        d = out["dutch roll"]
        chk["dutch roll"] = (d["zeta"] >= 0.08 and d["omega_n"] >= 0.4 and d["zeta"] * d["omega_n"] >= 0.15,
                             "ζ ≥ 0.08, ω_n ≥ 0.4 rad/s, ζω_n ≥ 0.15 rad/s")
    if "roll" in out:
        tc = out["roll"].get("time_constant_s")
        chk["roll"] = (tc is not None and tc <= 1.4, "time constant at most 1.4 s")
    if "spiral" in out:
        sp = out["spiral"]
        chk["spiral"] = (sp.get("t_double_s") is None or sp["t_double_s"] >= 20.0,
                         "time to double at least 20 s (or stable)")
    for k, (ok, txt) in chk.items():
        out[k]["level1"] = bool(ok)
        out[k]["criterion"] = txt
    out["_A_lon"], out["_A_lat"] = A_lon.tolist(), A_lat.tolist()
    out["_inertia"] = (Ix, Iy, Iz)
    return out


def responses(ac, an, der, md, t_end=(12.0, 12.0), elevator=-1.0, aileron=5.0, rudder=5.0):
    """linear time responses (stability axes) from the matrices of modes():
    elevator step (deg; - = trailing edge up, nose up), aileron step,
    rudder doublet (1 s each way). Returns time histories in deg, deg/s, m/s."""
    from scipy.linalg import expm
    V, rho = an.V, an.rho
    Q = 0.5 * rho * V**2
    S, b, c, m = ac.S, ac.b, ac.c, ac.mass
    Ix, Iy, Iz = md["_inertia"]
    ctr = der["controls"]
    A_lon, A_lat = np.array(md["_A_lon"]), np.array(md["_A_lat"])
    out = {}

    def sim(Amat, Bcol, u_of_t, T, n=600):
        dt = T / n
        M = np.zeros((5, 5))
        M[:4, :4], M[:4, 4] = Amat * dt, Bcol * dt
        E = expm(M)
        Ad, Bd = E[:4, :4], E[:4, 4]
        x = np.zeros(4)
        hist = [x.copy()]
        for k in range(n):
            x = Ad @ x + Bd * u_of_t(k * dt)
            hist.append(x.copy())
        return np.linspace(0, T, n + 1), np.array(hist)
    if "elevator" in ctr:
        d = ctr["elevator"]
        Zd = -d["CL"] * Q * S / m
        Md = d["Cm"] * Q * S * c / Iy
        Zwd = -der["CL_adot"] * c / (2 * V) * Q * S / (V * m)
        Mwd = der["Cm_adot"] * c / (2 * V) * Q * S * c / (V * Iy)
        Bc = np.array([-d["CD"] * Q * S / m, Zd / (1 - Zwd), Md + Mwd * Zd / (1 - Zwd), 0.0])
        de = math.radians(elevator)
        tt, x = sim(A_lon, Bc, lambda tk: de, t_end[0])
        out["elevator"] = {"t": tt.tolist(), "input_deg": elevator, "u": x[:, 0].tolist(),
                           "alpha": np.degrees(x[:, 1] / V).tolist(), "q": np.degrees(x[:, 2]).tolist(),
                           "theta": np.degrees(x[:, 3]).tolist()}
    lat_in = []
    if "aileron" in ctr:
        lat_in.append(("aileron", aileron, lambda tk: math.radians(aileron)))
    if "rudder" in ctr:
        lat_in.append(("rudder", rudder, lambda tk: math.radians(rudder) * (1 if tk < 1 else (-1 if tk < 2 else 0))))
    for name, amp, u in lat_in:
        d = ctr[name]
        Bc = np.array([Q * S * d["CY"] / (m * V), Q * S * b * d["Cl"] / Ix, Q * S * b * d["Cn"] / Iz, 0.0])
        tt, x = sim(A_lat, Bc, u, t_end[1])
        out[name] = {"t": tt.tolist(), "input_deg": amp, "beta": np.degrees(x[:, 0]).tolist(),
                     "p": np.degrees(x[:, 1]).tolist(), "r": np.degrees(x[:, 2]).tolist(),
                     "phi": np.degrees(x[:, 3]).tolist()}
    return out
