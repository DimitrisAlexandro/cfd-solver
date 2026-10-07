# Phase 4b — Coupled solver (2026-10-07)

Branch `phase4b-coupled` (on top of `phase4-speed`). 100 tests pass (5 new in `tests/test_coupled.py`).
Optional dependency: `pypardiso` (MKL PARDISO). Without it everything runs as before.

## Result: works and is correct, but only ~1.2× faster on this machine — not the 5–10×
NACA 0012, Re 6e6, SA, polar −4…16° (11 angles, 29k cells, 2-core cloud machine, 1 process):

| run | wall time | iterations / point |
|---|---|---|
| SIMPLEC (Phase 4 numba), warm starts | 555 s | 395–1350 |
| coupled, warm starts | 470 s (1.18×) | 76–229 |
| coupled, cold starts | 525 s | 80–189 |

Hard points gain the most: α 14° 98 → 52 s, α 16° 83 → 69 s; 30P30N α 5.5: 210 → 164 s (1.3×).

**Why not 5–10×:** the coupled solver needs 3–8× fewer outer iterations (as predicted), but each one costs ~4× a SIMPLEC iteration: 0.33 s, of which ~0.2 s is the PARDISO LU factorization of the 87 564 × 87 564 block system (11.4 M LU entries). The factorization did not get faster with any PARDISO setting tested (ordering, matching/scaling, two-level, single precision ~1.4× but needs float32 plumbing) or with 2 threads vs 1 on this machine. On a machine with more real cores PARDISO scales and SIMPLEC does not, so **measure on your own machine**: `python benchmark_coupled.py --polar`.

## Accuracy
- Laminar channel (triangles): coupled = SIMPLEC to **1e-12** (test).
- Turbulent NACA: Cd within 0.3 %. Cl at α ≥ 8° within 1e-4; at |α| ≤ 6° the coupled run settles **6e-4 lower in Cl** (α 0: −0.00078 vs −0.00012). Started from the SIMPLEC solution, the coupled solver stays on it (residual → 6e-9), so this is a second, nearly-degenerate state of the limited 2nd-order scheme that the cold coupled path reaches, not a different discretisation. 6e-4 in Cl is ~0.1–0.3 % of Cl and far below the mesh error, but it is not "identical", so SIMPLEC stays the default.
- Several coupled points end as "converged (mean of a small oscillation)": near convergence the residual grows slowly (×1.03/iteration from ~1e-6) and saturates at ~1e-5. Not caused by the vortex far field, SA sub-iterations or the limiter alone; relaxation 0.95/0.9 slows it but does not remove it.

## Bug found and fixed (affects SIMPLEC too)
The Majumdar under-relaxation correction was applied on interior faces but **not on outlet faces**, so the converged answer depended slightly on alpha_u (channel: 5e-4 change in U between alpha_u 0.9 and 0.7). Fixed in `simple_solver.py` and `coupled.py`; now alpha_u 0.9 vs 0.7 agree to 8e-12. Changes previous SIMPLEC results by ≲1e-4 (only at outlet-influenced cells).

## What was built
- `coupled.py`: `CoupledSolver` (u, v, p in one block system, same discretisation as SIMPLEC: Rhie–Chow + Majumdar, Green-Gauss pressure, skew terms), MKL PARDISO with symbolic analysis once / numeric refactorization each iteration (`PardisoLU`), alpha_u 0.99, SA segregated with 3 sub-iterations, adaptive relaxation (0.99 → 0.95 → 0.9 when the residual rises 4× above its minimum).
- `polar.run_point(solver=...)` / `airfoil.py polar --solver simple|coupled|auto`; iteration-based settings (monitor window, transition loop, vortex update) scaled ×0.1 for coupled. `auto` = `$CFD_SOLVER` or simple.
- `benchmark_coupled.py`: side-by-side timing on your machine.
- Removed again: limiter freezing (moved the solution to Cl −0.0128 at α 0 — wrong), GMRES/AMG and factorization-reuse variants (slower).

## Notes
- Coarse NACA mesh level 0.6 diverges with SIMPLEC at α 4 (pre-existing, also before this branch); coupled converges there.
- pypardiso on Linux may need `libmkl_rt.so`: `sudo ln -sf /usr/local/lib/libmkl_rt.so.3 /usr/local/lib/libmkl_rt.so && sudo ldconfig` (path depends on the pip install).

## Next options
1. Run `benchmark_coupled.py --polar` on your machine; if coupled ≥ 2× there, `export CFD_SOLVER=coupled`.
2. Real 5–10× needs a cheaper linear step than a full LU each iteration: compiled block-AMG (PETSc/hypre via petsc4py) with coupled-solver preconditioning — new dependency, a multi-day job.
3. Or stay with SIMPLEC + `--jobs` (angles in parallel), which already scales with cores.
