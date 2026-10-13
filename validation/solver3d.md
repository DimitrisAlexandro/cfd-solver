# 3D RANS solver (3D-1) and wing meshes (3D-2): validation and status

In-house 3D finite volume (no OpenFOAM): `src/fv3d/`. The heavy loops are in C++
(`fv3d/native/aero3d.cpp`, built with g++ on first use, numpy fallback).

## 3D-1: core (`python -m fv3d.validate_core3d`) — all pass
| case | result | reference |
|---|---|---|
| Lid-driven cavity Re 100, one-cell extrusion | max \|U3D − U2D\| = 9e-12 | 2D solver, same grid |
| NACA 0012 Re 6e6 α 10 (SA), extruded | Cl 1.09040 / Cd 0.012970 vs 2D 1.09039 / 0.012969 | CFL3D 1.0909 / 0.01231 |
| Laminar square duct Re_Dh 50 | f·Re 56.1, u_max/u_mean 2.04 | 56.91, 2.096 (Shah & London) |
| Sphere Re 100 (61k cells) | Cd 1.099, Ls/D 0.875 | 1.087–1.09, 0.87–0.88 |

## 3D-2: wings (`python -m fv3d.validate_wing3d`)
Rectangular NACA 0012 wing, AR 6, Re 3e6, SA, 252k cells, compared with AeroLab's vortex
lattice using the 2D SA polar of the section:

| α | CL RANS | CL VLM | ΔCL | CD RANS | CD VLM+section |
|---|---|---|---|---|---|
| 0 | −0.0008 | −0.0006 | — | 0.00960 | 0.00937 |
| 4 | 0.2989 | 0.2988 | 0.0 % | 0.01495 | 0.01444 |
| 8 | 0.6036 | 0.5853 | +3.1 % | 0.03081 | 0.02912 |

Each converged in 136–325 iterations (continuity 2e-6 to 4e-5). Span loading: `wing3d_validation.png`.

What made the wing work:
- **Tip closure.** A flat cap with a triangulated interior. Beyond the tip, the y+ ~ 1 rows are re-spaced over two spanwise levels so that their first row matches the fill cells. With thin rows next to large fill cells, the linear face weight goes to 1 and pressure and velocity decouple, so the run diverged.
- **Extrusion.** Faces are written already oriented, and cell centres sit at the mid-section, so tapered needle cells don't produce 90° non-orthogonality.
- **Gradients.** The Green–Gauss gradient includes a warped-face correction, which makes it exact for linear fields on any cell.
- **Momentum and pressure.** Convection uses the bounded form, and the pressure system is diagonally scaled before the multigrid-preconditioned conjugate gradient.
- **Start-up.** Each run starts from a 2D SA solution of the section mapped onto the 3D cells. A cold start drives the blunt trailing-edge wake through a transient that can diverge.

## Known limits (next work)
- **Taper and sweep.** Tapered or swept wings do not converge yet. Sweep 20° diverges at the tip trailing edge; taper 0.5 diverges at the leading edge, inboard of the tip. This is the next item to fix.
- **Finer mesh.** The finer mesh (level 0.55, 480k cells) diverged at α 4, so there is no grid-convergence result yet.
- **Tip-cap cells.** A few cells above the tip cap carry spurious velocities of about 4–5 U∞; forces are not affected beyond the quoted values.
- **Multiple steady states.** Steady RANS on a blunt trailing edge can have more than one steady state. On the α 10 airfoil, a first-order start leads to Cl 1.052 instead of 1.090. The defaults follow the 2D solver's path.
