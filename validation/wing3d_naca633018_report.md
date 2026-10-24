# 3D RANS wing: imported NACA 63(3)-018 section

An imported airfoil with a nearly cusped trailing edge, run through the Wing 3D job on the
fine mesh (level 0.5). Before the tip trailing-edge fix (PR #24) this case diverged after
about 150 iterations (`LinAlgError: SVD did not converge` in the multigrid setup); NACA 0012
at the same level did too.

Section: `validation/wing3d_naca633018.dat` (NACA 63(3)-018, 97 points, Selig format).
Wing: rectangular half wing, semi-span 3 root chords (AR 6), no sweep, dihedral or twist,
Re 3e6 on the root chord, far field 12 chords, SA model.

```bash
cd src
python -m aerolab.fv3d.wing3d --airfoil ../validation/wing3d_naca633018.dat --semi-span 3 \
       --re 3e6 --alpha 0,4 --level 0.5 --farfield 12 --out ../cases/wing3d_n633018
python -m aerolab.fv3d.wing3d --airfoil ../validation/wing3d_naca633018.dat --semi-span 3 \
       --re 3e6 --alpha 8 --level 0.5 --farfield 12 --out ../cases/wing3d_n633018_a8
```

Mesh: 676 704 cells (wake band for 0-4 deg) and 712 488 cells (0-8 deg), 56 spanwise layers,
non-orthogonality max 67 deg (75 deg before the trailing-edge quads in the tip cap).

| alpha | status | CL (RANS) | CL (VLM) | CD (RANS) | CD pressure | CD (VLM: CDi + section) | Cm | iterations | continuity | time (4 cores) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | converged | -0.0013 | 0.0000 | 0.01071 | 0.00313 | 0.01045 | +0.0003 | 400 | 2.9e-06 | 12 min |
| 4 | converged | 0.3144 | 0.2855 | 0.01638 | 0.00876 | 0.01591 | -0.0015 | 450 | 5.6e-05 | 16 min |
| 8 | converged | 0.6280 | 0.5697 | 0.03332 | 0.02595 | 0.03217 | -0.0042 | 425 | 7.7e-05 | 22 min |

- No divergence at any angle; CL and CD stayed constant to four digits over the last
  200 iterations. Alpha 8 was run on its own, from the 2D section start (no warm start
  from alpha 4).
- Lift is linear, about 0.078 per degree.
- The vortex-lattice CL is about 10 % lower at 4 and 8 deg. Its section lift slope for
  this airfoil is *estimated* (no cached polar), so the gap reflects that estimate. With
  the computed NACA 0012 polar (`wing3d_report.md`) the two agree within 0.1-3 %.
- Not compared with experiment.
