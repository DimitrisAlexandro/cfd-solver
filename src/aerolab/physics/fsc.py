"""
Falkner-Skan-Cooke similarity solutions: the laminar boundary layer of an
infinite swept wing with chordwise edge velocity U_e = C x^m and constant
spanwise velocity W_e (Cooke 1950).

    f''' + f f'' + beta (1 - f'^2) = 0     f(0) = f'(0) = 0, f'(inf) = 1
    g''  + f g'                    = 0     g(0) = 0,         g(inf) = 1
    beta = 2m/(m+1),  eta = y sqrt((m+1) U_e / (2 nu x))
    u = U_e f'(eta),  w = W_e g(eta)

Used for (i) the crossflow-velocity profile and its Reynolds number in the
swept-wing transition criteria (swept_transition.py) and (ii) the local
closures of the amplification-factor transport model (aft.py): shape factor
and the ratio of the maximum vorticity Reynolds number to Re_theta,
    k(H) = max_y (y^2 |du/dy| / nu) / Re_theta     (Blasius: 2.193),
tabulated over the family.

    python src/run.py fsc            prints the table (H, Re_V,max/Re_theta, crossflow)
"""

import math
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp

BETA_SEP = -0.198838       # separation profile (Hartree)


@lru_cache(maxsize=512)
def _solve(beta, eta_max=None):
    """(eta, f, f', f'', g, g') on a fine grid; cached per beta"""
    beta = float(beta)
    if eta_max is None:
        eta_max = 10.0 if beta >= 0 else 10.0 + 25.0 * min(-beta / 0.2, 1.0)
    eta = np.linspace(0.0, eta_max, 801)

    def ode(e, y):
        f, fp, fpp, g, gp = y
        return np.vstack([fp, fpp, -f * fpp - beta * (1 - fp**2), gp, -f * gp])

    def bc(a, b):
        return np.array([a[0], a[1], b[1] - 1.0, a[3], b[3] - 1.0])
    # initial guess: an error-function-like profile
    s = eta / max(eta_max / 4.0, 1e-9)
    fp0 = np.tanh(s * 1.5)
    y0 = np.vstack([np.cumsum(fp0) * (eta[1] - eta[0]), fp0, np.gradient(fp0, eta),
                    fp0, np.gradient(fp0, eta)])
    sol = None
    # continuation in beta from Blasius for adverse gradients (the BVP has two
    # solutions near separation; the attached branch is the physical one)
    if beta < -0.1:
        prev = _solve(round(beta / 2, 6) if beta / 2 > -0.1 else -0.1)
        y0 = np.vstack([np.interp(eta, prev[0], prev[k]) for k in range(1, 6)])
    sol = solve_bvp(ode, bc, eta, y0, tol=1e-8, max_nodes=200000)
    if not sol.success:
        raise RuntimeError(f"Falkner-Skan-Cooke BVP failed for beta = {beta}: {sol.message}")
    Y = sol.sol(eta)
    return (eta,) + tuple(Y[k] for k in range(5))


def profile(beta):
    """eta, f', f'', g arrays of the similarity solution"""
    eta, f, fp, fpp, g, gp = _solve(round(float(beta), 6))
    return eta, fp, fpp, g


def integrals(beta):
    """displacement / momentum thickness (eta units), H, wall shear f''(0),
    k = max(eta^2 f'') / theta (Re_V,max / Re_theta)"""
    eta, fp, fpp, g = profile(beta)
    dstar = np.trapezoid(1 - fp, eta)
    theta = np.trapezoid(fp * (1 - fp), eta)
    k = float(np.max(eta**2 * np.abs(fpp)) / theta)
    return {"beta": float(beta), "dstar": float(dstar), "theta": float(theta),
            "H": float(dstar / theta), "fpp0": float(fpp[0]), "k": k}


def crossflow(beta, sweep_ratio):
    """Crossflow profile of the FSC layer for W_e/U_e = sweep_ratio:
    w_cf = (U_e W_e / Q_e) (g - f'), normalised by Q_e (external streamline
    speed). Returns (max |w_cf|/Q_e, eta at the max, eta where it falls to
    10 % of the max outside the peak)."""
    eta, fp, fpp, g = profile(beta)
    r = float(sweep_ratio)
    q = math.sqrt(1.0 + r * r)
    wcf = r / (q * q) * (g - fp)                 # (U_e W_e / Q_e^2)(g - f')
    i = int(np.argmax(np.abs(wcf)))
    wmax = abs(wcf[i])
    out = np.where((np.abs(wcf) < 0.1 * wmax) & (np.arange(len(eta)) > i))[0]
    e10 = float(eta[out[0]]) if len(out) else float(eta[-1])
    return wmax, float(eta[i]), e10


@lru_cache(maxsize=1)
def table(n=61):
    """closure table over the attached family (beta from near separation
    to strong acceleration): beta, H, k, theta (eta units)"""
    betas = np.r_[np.linspace(-0.19, -0.02, 18), np.linspace(0.0, 1.0, n - 18)]
    rows = [integrals(round(b, 6)) for b in betas]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def k_of_H(H):
    """Re_V,max / Re_theta as a function of the shape factor (FS family)"""
    t = table()
    o = np.argsort(t["H"])
    return np.interp(H, t["H"][o], t["k"][o])


def main():
    t = table()
    print(f"{'beta':>7s} {'H':>7s} {'k=ReVmax/Reth':>14s} {'f\"(0)':>8s} {'theta_eta':>9s}")
    for i in range(0, len(t["beta"]), 4):
        print(f"{t['beta'][i]:7.3f} {t['H'][i]:7.3f} {t['k'][i]:14.3f} {t['fpp0'][i]:8.4f} "
              f"{t['theta'][i]:9.4f}")
    print("crossflow, W/U = 1 (45 deg sweep):")
    for b in (0.0, 0.2, 0.5, 1.0):
        w, e, e10 = crossflow(b, 1.0)
        print(f"  beta {b:4.1f}: max w_cf/Q_e {w:.4f} at eta {e:.2f}, 10 % height eta {e10:.2f}")


if __name__ == "__main__":
    main()
