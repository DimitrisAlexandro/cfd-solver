# 3D RANS solver

The in-house 3D finite-volume solver for wings (`src/aerolab/fv3d/`).

## 3D RANS solver (in house, no OpenFOAM)

`src/aerolab/fv3d/` is a 3D finite-volume RANS solver with the same discretisation as the
validated 2D one: cell-centred on polyhedral meshes (OpenFOAM-style faces / owner /
neighbour), collocated SIMPLEC with Rhie-Chow, Spalart-Allmaras, least-squares and
Green-Gauss gradients exact for linear fields on any cell shape (skewed, tapered,
warped faces), and the heavy loops in C++ (assembly, limiter, Gauss-Seidel, multigrid
preconditioned CG, SA source) - built automatically with `g++` on first use
(`sudo apt install build-essential` on WSL/Ubuntu; without a compiler it falls back to
numpy, slower).

```bash
cd src
python run.py wing3d --semi-span 3 --taper 0.5 --sweep 20 --twist -2 --re 3e6 \
       --alpha 0,4,8 --level 0.4 --out ../cases/wing3d_demo      # half wing, SA
python run.py wing3d --airfoil naca2412 --tip-airfoil naca0012 --semi-span 3 --taper 0.5 \
       --alpha 0,4 --level 0.4 --out ../cases/wing3d_tip                # root -> tip section
python run.py validate_core3d     # 3D-1: cavity / airfoil consistency with 2D, duct, sphere
python run.py validate_wing3d     # 3D-2: wings against the vortex lattice
```

**In the app:** Wing → **Wing 3D CFD**. The wing comes from a saved aircraft (its main
wing or a tail, including an aircraft imported from STL or AVL) or is entered by hand, in metres with a
speed and altitude. The page converts it to root chords and the root-chord Reynolds number and queues
the run (it keeps going when the browser is closed). The results show CL, CD, Cm and the span
loading next to the vortex lattice of the same wing, the method the aircraft pages use.
A wing with panel breaks is solved as one straight panel from root to tip.

**Root and tip sections.** `--tip-airfoil` changes the section linearly from root to tip. The
tip's wall points are matched to the root's one to one (both are repanelled alike). Each wall-layer
column moves with its wall point and turns with the wall, so it stays normal to the new surface
and the y+ ~ 1 rows keep their heights. The rest of the section mesh follows by Laplace
interpolation. Sections too different to morph (a cell would invert) are refused before the run.

The wing mesher sweeps the validated 2D hybrid airfoil mesh (y+ < 1 boundary
layer) through spanwise stations with taper, sweep, twist and dihedral, closes the
tip with a flat cap and relaxes the wall layers beyond the tip. A graded wake band
10 chords long follows the free stream at the run's angles, and the spanwise spacing
is capped around the tip vortex. Outputs: CL, CD (pressure/friction), Cm, span
loading c_l c(eta).

The tip cap closes a thin (or cusped) trailing edge with quads between facing upper and
lower wall points instead of sliver triangles. The pressure gradient's skewness correction
lags the previous gradient. Where that fixed-point iteration would amplify itself (the wall
rows re-spaced in one layer beyond the tip trailing edge: on the fine mesh the gradient grew
without bound after ~150 iterations), the solver finds the cells once by power iteration and
lags the least-squares gradient there instead. A run that still blows up is stopped and
reported as `diverged` for that angle. The next angle restarts from the 2D section solution.

Rectangular, tapered, swept and twisted wings converge. Against the vortex lattice, CL
is within 2 %:

- taper 0.5, sweep 20°, twist −2°, AR 8: α 4 CL 0.2561 vs 0.2574;
- span loading within about 1 % from root to tip.

Results and limits: `validation/solver3d.md`, `validation/wing3d_report.md`,
`validation/core3d_report.md`.

With `--out`, every angle also saves its full flow field (`fields_a<alpha>.npz`).
