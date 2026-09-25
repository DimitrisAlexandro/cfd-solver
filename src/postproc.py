"""
Derived quantities for pre- and post-processing.

mesh_quality(...)  per-cell quality fields (usable BEFORE the simulation)
flow_fields(...)   |U|, Cp, vorticity, continuity error per cell
wall_data(...)     Cp, Cf, y+ on each wall face, plus force coefficients
"""

import numpy as np


# ----------------------------------------------------------------------
# mesh quality (per cell)
# ----------------------------------------------------------------------
def mesh_quality(fvm, cells, n_bl=0):
    P = fvm.points
    n = fvm.ncells
    min_angle = np.empty(n)
    aspect = np.empty(n)
    for i, c in enumerate(cells):
        Q = P[c]
        e = np.roll(Q, -1, axis=0) - Q                    # edge vectors
        L = np.linalg.norm(e, axis=1)
        a, b = -np.roll(e, 1, axis=0), e                   # edges leaving each vertex
        cosang = np.einsum('ij,ij->i', a, b) / (np.roll(L, 1) * L)
        min_angle[i] = np.degrees(np.arccos(np.clip(cosang, -1, 1))).min()
        aspect[i] = L.max() / L.min()

    # non-orthogonality and neighbour size ratio (worst face of each cell)
    d, S = fvm.d, fvm.Sf
    cosno = np.abs(np.einsum('ij,ij->i', d, S)) / (np.linalg.norm(d, axis=1) *
                                                   np.linalg.norm(S, axis=1))
    nonorth_f = np.degrees(np.arccos(np.clip(cosno, -1, 1)))
    ratio_f = np.maximum(fvm.vol[fvm.own], fvm.vol[fvm.nei]) / \
        np.minimum(fvm.vol[fvm.own], fvm.vol[fvm.nei])
    nonorth = np.zeros(n)
    ratio = np.ones(n)
    for arr, fv in ((nonorth, nonorth_f), (ratio, ratio_f)):
        np.maximum.at(arr, fvm.own, fv)
        np.maximum.at(arr, fvm.nei, fv)

    region = np.zeros(n, np.int32)
    region[:n_bl] = 1
    return {"min_angle": min_angle, "nonorthogonality": nonorth, "aspect_ratio": aspect,
            "size_ratio": ratio, "cell_area": fvm.vol.copy(),
            "cell_type": np.array([len(c) for c in cells], np.int32),
            "boundary_layer": region}


# ----------------------------------------------------------------------
# flow fields (per cell)
# ----------------------------------------------------------------------
def flow_fields(solver, U_inf=1.0, p_inf=0.0):
    m = solver.m
    U, p = solver.U, solver.p
    gu, gv = solver.gU
    q = 0.5 * U_inf ** 2
    return {"U": U.copy(), "U_mag": np.linalg.norm(U, axis=1), "p": p.copy(),
            "Cp": (p - p_inf) / q,
            "vorticity": gv[:, 0] - gu[:, 1],
            "continuity_error": m.sum_to_cells(solver.F, solver.Fb) / m.vol}


# ----------------------------------------------------------------------
# wall data (per wall face)
# ----------------------------------------------------------------------
def wall_data(solver, patch, U_inf=(1.0, 0.0), p_inf=0.0, ref_length=1.0):
    m = solver.m
    Uinf = np.asarray(U_inf, float)
    Umag = np.linalg.norm(Uinf)
    q = 0.5 * Umag ** 2
    idx, pb, tau, S, Sm = solver.wall_values(patch)
    n_body = -S / Sm[:, None]                               # outward from the body
    dn = np.abs(np.einsum('ij,ij->i', m.db[idx], n_body))
    tau_mag = np.linalg.norm(tau, axis=1)
    sign = np.sign(tau @ (Uinf / Umag))                     # < 0: reversed flow
    x = m.bxf[idx]
    Fp, Fv = solver.force(patch)
    F = Fp + Fv
    drag_dir = Uinf / Umag
    lift_dir = np.array([-drag_dir[1], drag_dir[0]])
    coeffs = {"Cd": F @ drag_dir / (q * ref_length), "Cl": F @ lift_dir / (q * ref_length),
              "Cd_pressure": Fp @ drag_dir / (q * ref_length),
              "Cd_viscous": Fv @ drag_dir / (q * ref_length)}
    return {"face": idx, "x": x[:, 0], "y": x[:, 1],
            "side": np.where(n_body[:, 1] >= 0, 1, -1),      # +1 upper, -1 lower
            "Cp": (pb - p_inf) / q, "Cf": sign * tau_mag / q,
            "yplus": np.sqrt(tau_mag) * dn / solver.nu, "wall_distance": dn}, coeffs


def force_coefficients(solver, patch, U_inf=(1.0, 0.0), ref_length=1.0):
    return wall_data(solver, patch, U_inf, ref_length=ref_length)[1]
