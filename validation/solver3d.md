# 3D RANS solver (3D-1) and wing meshes (3D-2): validation and status

In-house 3D finite volume (no OpenFOAM): `src/aerolab/fv3d/`. The heavy loops are in C++
(`fv3d/native/aero3d.cpp`, built with g++ on first use, numpy fallback).

## 3D-1: core (`python src/run.py validate_core3d`) — all pass
| case | result | reference |
|---|---|---|
| Lid-driven cavity Re 100, one-cell extrusion | max \|U3D − U2D\| = 9e-12 | 2D solver, same grid |
| NACA 0012 Re 6e6 α 10 (SA), extruded | Cl 1.09040 / Cd 0.012970 vs 2D 1.09039 / 0.012969 | CFL3D 1.0909 / 0.01231 |
| Laminar square duct Re_Dh 50 | f·Re 56.1, u_max/u_mean 2.04 | 56.91, 2.096 (Shah & London) |
| Sphere Re 100 (61k cells) | Cd 1.099, Ls/D 0.875 | 1.087–1.09, 0.87–0.88 |

## 3D-2: wings (`python src/run.py validate_wing3d`)
Rectangular NACA 0012 wing, AR 6, Re 3e6, SA, 415k cells (wake-band mesh), compared with
AeroLab's vortex lattice using the 2D SA polar of the section:

| α | CL RANS | CL VLM | ΔCL | CD RANS | CD VLM+section |
|---|---|---|---|---|---|
| 0 | −0.0005 | −0.0006 | — | 0.00960 | 0.00937 |
| 4 | 0.2991 | 0.2988 | +0.1 % | 0.01494 | 0.01444 |
| 8 | 0.6026 | 0.5853 | +3.0 % | 0.03056 | 0.02912 |

Each angle converged in 143–450 iterations, with continuity between 2e-6 and 4e-5. The
previous 252k-cell mesh gave the same results within 0.002 in CL. Span loading is in
`wing3d_validation.png`.

What made the wing work:
- **Tip closure.** A flat cap with a triangulated interior. Beyond the tip, the y+ ~ 1 rows are re-spaced over two spanwise levels so that their first row matches the fill cells. With thin rows next to large fill cells, the linear face weight goes to 1 and pressure and velocity decouple, so the run diverged.
- **Extrusion.** Faces are written already oriented, and cell centres sit at the mid-section, so tapered needle cells don't produce 90° non-orthogonality.
- **Gradients.** The Green–Gauss gradient includes a warped-face correction, which makes it exact for linear fields on any cell.
- **Momentum and pressure.** Convection uses the bounded form, and the pressure system is diagonally scaled before the multigrid-preconditioned conjugate gradient.
- **Start-up.** Each run starts from a 2D SA solution of the section mapped onto the 3D cells. A cold start drives the blunt trailing-edge wake through a transient that can diverge.

## Tapered and swept wings

Two defects stopped them; both are fixed.

1. **Side-face centres off the mid-section.** Tapering turns the spanwise side faces into
   trapezoids. Their area centroid sits off the mid-section by about Δz·Δc/(12c), while
   the cell centres are kept on the mid-section by design. In a y+ ~ 1 wall cell
   (about 1e-5 c thick) that offset is tens of cell heights (68 on the test strip). The
   skewness correction and the wall-face non-orthogonal correction then multiplied the
   wall-normal velocity gradient by a spanwise distance. The leading-edge wall cells
   reached 6 U∞ within 20 iterations, then diverged. Swept wings diverged the same way.
   - **Fix:** the extrusion supplies side-face centres at the mid-section (the mean of
     the four nodes). The face integral is corrected about the new centre, so gradients
     stay exact for linear fields.
   - **Regression test:** `test_lateral_face_centres_are_level_with_cell_centres`.
2. **Kink in the mesh deformation at the tip plane.** Inboard, the planform change (taper,
   sweep, twist) shears every spanwise layer; beyond the tip it stopped dead. The thin
   tip-plane layers carried a mode that never settled: forces converged, but continuity
   stalled at about 5e-3.
   - **Fix:** beyond the tip the deformation eases off over half a tip chord
     (`tip_smooth`). Continuity now falls to 1e-5 in 100 iterations, as on the
     rectangular wing.

Neither change touches a rectangular, untwisted wing: its side faces are rectangles, and
it has no deformation.

| wing (NACA 0012, Re 3e6, semi-span 3 c, level 0.4) | α | CL RANS | CL VLM | ΔCL | CD RANS | CD VLM+section | iterations |
|---|---|---|---|---|---|---|---|
| sweep 20° | 4 | 0.2906 | 0.2877 | +1.0 % | 0.01464 | 0.01421 | 425 |
| taper 0.5 | 4 | 0.3287 | 0.3345 | −1.7 % | 0.01467 | 0.01410 | 194 |
| taper 0.5, sweep 20°, twist −2° (AR 8) | 0 | −0.0667 | −0.0676 | — | 0.00936 | 0.00973 | 137 |
| same | 4 | 0.2561 | 0.2574 | −0.5 % | 0.01204 | 0.01223 | 145 |
| same | 8 | 0.5801 | 0.5687 | +2.0 % | 0.02369 | 0.02333 | 475 |

Span loading of the tapered, swept, twisted wing at α 4, against the vortex lattice:

| η | 0.1 | 0.5 | 0.8 | 0.95 |
|---|---|---|---|---|
| RANS | 0.260 | 0.204 | 0.135 | 0.078 |
| VLM | 0.261 | 0.206 | 0.137 | 0.077 |

## Wake mesh

Airfoil meshes carry a graded wake band, 10 chords long, along the free stream. When one
mesh serves several angles (3D wings), the band covers them all. Cell size grades smoothly
across the band and keeps growing smoothly past its end. Before, a 3-chord strip ended
with a 10× jump in cell size.

Cell growth along the band depends on the mesh's use:

- **3D wing section: 0.012 per chord** (× 1/level). This holds the tip vortex for about
  10 chords.
- **2D airfoil meshes: 0.05 per chord.** Finer 2D wake cells resolve the weak shedding of
  the blunt trailing edge, and the steady solvers then sit in a slow limit cycle. On NACA
  0012 at M 0.7, α 2, Cl swings ±1 % at 0.012–0.03 and converges at 0.05.

The 3D mesh also caps the spanwise spacing within 0.3 tip chords of the tip, where the
tip vortex is.

| AR 6 wing, α 4, wake survey | old mesh | new mesh |
|---|---|---|
| CDi just behind the wing (VLM 0.00484) | 0.00452 | 0.00480 |
| CDi at 4 c | 0.00376 | 0.00392 |
| tip-vortex peak vorticity at 4 c | 0.46 | 1.77 |
| tip-vortex core radius at 4 c | 0.29 c | 0.12 c |

Forces are unchanged. The AR 6 wing gives CL 0.2991 / CD 0.01492 (before: 0.2989 / 0.01493).
On the 2D side, NACA 0012 at α 10 (level 1) gives Cl 1.0764 / Cd 0.01202, against
1.0766 / 0.01203 on the old mesh.

Cost: +2.5 % cells in 2D at level 1, and +64 % for the AR 6 wing at level 0.4 (415k cells,
from 252k).

## Known limits (next work)
- **Finer mesh.** The finer mesh (level 0.55) diverged at α 4 before these fixes, so
  there is no grid-convergence result yet.
- **Tip-cap cells.** A few cells above the tip cap can carry spurious velocities. Forces
  are not affected beyond the quoted values.
- **Multiple steady states.** Steady RANS on a blunt trailing edge can have more than one
  steady state. On the α 10 airfoil, a first-order start leads to Cl 1.052 instead of
  1.090. The defaults follow the 2D solver's path.
