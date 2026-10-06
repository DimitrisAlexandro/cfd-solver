"""
Inviscid multi-element panel method (Phase 3, step E5).

Linear-strength vortex panels (Katz & Plotkin, ch. 11) on every element:
zero normal velocity at each panel midpoint plus a Kutta condition per
element (gamma_upper TE + gamma_lower TE = 0). Each element's circulation
comes out of its own Kutta condition, so the slot effects of a multi-element
airfoil (slat dumping velocity, circulation effect, off-surface pressure
recovery, Smith 1975) are all in the solution.

Used as the inviscid reference for the multi-element RANS results and as a
fast preview (`airfoil.py show <spec> --alpha 8`).

    res = solve(outlines, alpha_deg)          # outlines: open TE -> TE, CCW
    res["Cl"], res["Cm"], res["elements"][k]["Cl"], res["elements"][k]["Cp"]

Checked against the exact Joukowski solution (tests/test_multi_element.py):
Cl within 0.1 % with 200 panels.
"""

import math

import numpy as np

TWO_PI = 2.0 * math.pi


def _influence(xp, yp, A, B):
    """Velocity at points (xp, yp) induced by linear vortex panels A->B with
    unit strength at the start (ga) and at the end (gb).
    Returns (ua, va, ub, vb), each (npts, npanels), global components.
    Positive gamma = counter-clockwise circulation."""
    d = B - A
    L = np.linalg.norm(d, axis=1)
    t = d / L[:, None]
    nl = np.c_[-t[:, 1], t[:, 0]]                       # left normal (local +y)
    rx = xp[:, None] - A[None, :, 0]
    ry = yp[:, None] - A[None, :, 1]
    x = rx * t[None, :, 0] + ry * t[None, :, 1]         # local coordinates
    y = rx * nl[None, :, 0] + ry * nl[None, :, 1]
    th0 = np.arctan2(y, x)
    th1 = np.arctan2(y, x - L[None])
    dth = th1 - th0
    r0 = x**2 + y**2
    r1 = (x - L[None]) ** 2 + y**2
    lnr = np.log(np.maximum(r1, 1e-300) / np.maximum(r0, 1e-300))   # ln(r1^2 / r0^2)
    I0u = dth
    I1u = x * dth + 0.5 * y * lnr
    J0 = -0.5 * lnr
    J1 = x * J0 - L[None] + y * dth
    Ll = L[None]
    u_a = -(I0u - I1u / Ll) / TWO_PI
    u_b = -(I1u / Ll) / TWO_PI
    v_a = (J0 - J1 / Ll) / TWO_PI
    v_b = (J1 / Ll) / TWO_PI
    # local (u along t, v along nl) -> global
    gx = lambda u, v: u * t[None, :, 0] + v * nl[None, :, 0]
    gy = lambda u, v: u * t[None, :, 1] + v * nl[None, :, 1]
    return gx(u_a, v_a), gy(u_a, v_a), gx(u_b, v_b), gy(u_b, v_b)


def solve(outlines, alpha_deg=0.0, moment_ref=(0.25, 0.0), ref_chord=1.0, names=None):
    """outlines: list of (N_k+1, 2) open outlines, trailing edge -> around ->
    trailing edge, counter-clockwise (the multi_element 'outline' arrays)."""
    a = math.radians(alpha_deg)
    Uinf = np.array([math.cos(a), math.sin(a)])
    names = names or [f"element{k + 1}" for k in range(len(outlines))]
    A = np.vstack([o[:-1] for o in outlines])
    B = np.vstack([o[1:] for o in outlines])
    npan = [len(o) - 1 for o in outlines]
    # node index of every panel end: element k has nodes off_k .. off_k + N_k
    noff = np.cumsum([0] + [n + 1 for n in npan])
    ia = np.concatenate([noff[k] + np.arange(n) for k, n in enumerate(npan)])
    ib = ia + 1
    nn = noff[-1]
    d = B - A
    L = np.linalg.norm(d, axis=1)
    t = d / L[:, None]
    nout = np.c_[t[:, 1], -t[:, 0]]                     # outward (fluid side)
    mid = 0.5 * (A + B)
    eps = 1e-7 * L
    col = mid + eps[:, None] * nout                     # just outside the body
    ua, va, ub, vb = _influence(col[:, 0], col[:, 1], A, B)
    np_ = len(A)
    # node influence: panel j contributes 'a' to node ia[j], 'b' to node ib[j]
    U = np.zeros((np_, nn)); V = np.zeros((np_, nn))
    for M, ca, cb in ((U, ua, ub), (V, va, vb)):
        np.add.at(M.T, ia, ca.T)
        np.add.at(M.T, ib, cb.T)
    An = U * nout[:, 0:1] + V * nout[:, 1:2]
    rhs = -(nout @ Uinf)
    # system: np_ normal-flow equations + one Kutta condition per element
    K = np.zeros((len(outlines), nn))
    for k in range(len(outlines)):
        K[k, noff[k]] = 1.0
        K[k, noff[k + 1] - 1] = 1.0
    M = np.vstack([An, K])
    r = np.r_[rhs, np.zeros(len(outlines))]
    gam = np.linalg.lstsq(M, r, rcond=None)[0] if M.shape[0] != M.shape[1] else \
        np.linalg.solve(M, r)
    # surface speed: the body interior is at rest, so the tangential velocity
    # just outside equals the local sheet strength (second-order accurate;
    # summing the induced velocities at the offset collocation points is
    # only first order in the panel size)
    Vt = 0.5 * (gam[ia] + gam[ib])
    Cp = 1.0 - Vt**2
    # forces from the pressure on each panel (inviscid: pressure only)
    F = -(Cp * L)[:, None] * nout                       # per unit q
    drag_dir = Uinf
    lift_dir = np.array([-Uinf[1], Uinf[0]])
    rm = mid - np.asarray(moment_ref)
    Mz = rm[:, 0] * F[:, 1] - rm[:, 1] * F[:, 0]
    res = {"alpha": alpha_deg, "elements": []}
    p0 = 0
    for k, n in enumerate(npan):
        sl = slice(p0, p0 + n)
        Fk = F[sl].sum(0)
        res["elements"].append({
            "name": names[k], "x": mid[sl, 0], "y": mid[sl, 1], "Cp": Cp[sl], "Vt": Vt[sl],
            "Cl": float(Fk @ lift_dir / ref_chord), "Cd": float(Fk @ drag_dir / ref_chord),
            "Cm": float(-Mz[sl].sum() / ref_chord**2),
            # circulation (clockwise positive = lifting)
            "Gamma": float(-(0.5 * (gam[ia[sl]] + gam[ib[sl]]) * L[sl]).sum())})
        p0 += n
    Ft = F.sum(0)
    res.update(Cl=float(Ft @ lift_dir / ref_chord), Cd=float(Ft @ drag_dir / ref_chord),
               Cm=float(-Mz.sum() / ref_chord**2),
               Cl_kj=float(2 * sum(e["Gamma"] for e in res["elements"]) / ref_chord))
    return res


def joukowski(xc=-0.1, yc=0.08, n=200):
    """Exact Joukowski airfoil (circle through z = 1, map z + 1/z), returned
    as an open CCW outline from the cusped TE, chord and the exact lift
    curve Cl(alpha) as a function."""
    R = math.hypot(1 - xc, yc)
    beta = math.asin(yc / R)                           # angle of the TE from the centre
    th = np.linspace(0, TWO_PI, n + 1)
    zc = complex(xc, yc)
    th0 = -beta                                        # the TE point zeta = 1
    zeta = zc + R * np.exp(1j * (th0 + th))
    z = zeta + 1 / zeta
    P = np.c_[z.real, z.imag]
    te = P[0].copy()
    le = P[np.argmin(P[:, 0])]
    chord = float(np.linalg.norm(te - le))
    # orientation: theta increasing = CCW in zeta; Joukowski keeps orientation
    def cl_exact(alpha_deg):
        a = math.radians(alpha_deg)
        return 8 * math.pi * R / chord * math.sin(a + beta)
    return P, chord, cl_exact, le
