# Faster and sturdier: MUMPS for the adjoint, pseudo-time continuation in SIMPLEC

## MUMPS for the adjoint linear solve

`adjoint.factorize()` uses MUMPS (python-mumps, METIS ordering, two steps of iterative refinement) when it is installed, and SuperLU otherwise. Benchmark on NACA 0012, Re 3e6, alpha 4 (`python src/bench_adjoint.py`, 4 cores; data in `adjoint_speed.json`):

| mesh level | cells | unknowns | flow | Jacobian | SuperLU LU | MUMPS LU | adjoint residual SuperLU / MUMPS |
|---|---|---|---|---|---|---|---|
| 0.5 | 7 726 | 60 169 | 9 s | 15 s | 11.7 s | 6.7 s | 1.5e-9 / 5.0e-10 |
| 0.7 | 14 027 | 109 435 | 37 s | 28 s | 32.9 s | 17.2 s | 2.7e-9 / 1.0e-9 |
| 1.0 | 27 564 | 215 335 | 58 s | 63 s | 107.3 s | 54.4 s | 1.4e-8 / 1.5e-9 |

- **Speed**: MUMPS factorises twice as fast at every level.
- **Accuracy**: MUMPS is more accurate on the finest mesh. The two adjoint solutions agree to 1e-7.
- **Cost on the validated mesh (level 1)**: a complete gradient (flow + Jacobian + LU + solves) takes about 3 minutes, so level 1 is practical. The coloured complex-step Jacobian now costs as much as the factorisation; it is the next target.
- **PETSc**: not used. petsc4py has no wheels for Python 3.13, and MUMPS through python-mumps gives the same factorisation without the PETSc stack.

Install:
- Debian/Ubuntu: `apt install libmumps-seq-dev`, then `pip install python-mumps`. pip may need pkg-config files for `dmumps_seq`; conda avoids this.
- Conda: `conda install -c conda-forge python-mumps`.
- macOS: `brew install mumps`.

## Pseudo-time continuation, and the bug it uncovered

**The bug.** Pseudo-time stepping adds V/dtau = a_P / CFL to the momentum and SA diagonals. Testing it showed that this SIMPLEC diverged whenever it was damped more strongly: alpha_u below about 0.8, or CFL below about 3.

The cause was the Majumdar term in the Rhie-Chow flux, (1 - alpha_u)(F_old - Ubar_old). It used the face interpolation of the *predicted* velocity U*. Majumdar's formulation uses the *corrected* velocity of the previous iteration. Both give the same converged solution, so the bug never showed in converged results.

With the corrected velocity:
- SIMPLEC is stable at alpha_u = 0.3 and at CFL 0.1.
- The converged solution does not change: on the cavity, max |dU| < 1e-7 between alpha_u 0.3, 0.5 and 0.9 and between CFL 0.2, 5 and off. On NACA 0012, Cl and Cd agree to every printed digit with the code before the change. The adjoint residual check is unchanged.

**How the pseudo-time terms stay out of the solution.** With pseudo-time, Rhie-Chow uses the damped diagonal, consistent with the cell velocities; using V/a_P alone diverged at small CFL. A local Majumdar factor alpha_f = D_f / interp(V / a_P) absorbs the pseudo-time term, so it drops out of the converged flux exactly.

**The driver, `SimpleSolver.solve_robust`:**
- Switched-evolution-relaxation CFL ramp, never below the starting CFL. Pseudo-time switches off at CFL 200.
- State snapshots every 25 iterations. On a blow-up (NaN, an exception, or a residual above 1 and 1e3 times its best value), it rolls back and cuts the CFL ceiling by 4, at most 6 times.

**`adjoint.solve_primal`, used by the optimiser and FIML (`robust=True`):**
- Cold starts begin with first-order convection.
- Every phase runs plain SIMPLEC first. If it blows up, the run goes back to the phase's start state and continues under pseudo-time. If it stalls well above the tolerance, it continues under pseudo-time and keeps the result only if it got lower.
- It raises an error if no converged flow is found. Before, a diverged state could be returned silently.
- Pseudo-time *first* was tried and rejected. From a cold start, or from a warm start on a deformed mesh, the impulsive pressure (about V/dtau) gave the far-field vortex a lift of about 1e6, which broke runs that plain SIMPLEC converges.

Results:

| case | before | now |
|---|---|---|
| design shapes from the earlier optimisation (evals 52, 56), cold start | diverged | converged (2e-7, 5e-7; the optimiser accepts < 1e-6) |
| NACA 0012, Re 6e6, alpha 4, level 0.5 | diverged | converged (1e-10) |
| NACA 0012, Re 3e6, alpha 15 / 16 / 17 (stalled) | converged | converged, same Cl and Cd |
| NACA 0012, Re 3e6, alpha 4, level 0.3 mesh | diverged | converged (9e-8, Cl 0.4422) |

**Limits.** Some states remain out of reach:
- **Optimiser trial steps that put the first lower-surface CST coefficient on its bound** (+0.03, the sharpest nose the bounds allow). The second-order scheme is unstable in the first wall cells at the lower-nose stagnation point: the velocity grows over about 30 iterations while the residual is still falling, at any CFL down to 0.05. This is a discretisation instability on that cell geometry, not a relaxation problem. Changing the scheme only for some designs would make the objective inconsistent, so these steps stay rejected (the line search steps back, as before).
- **NACA 0012 at Re 6e6 above alpha 7 on the level-0.5 mesh.** No steady flow, even with warm-started alpha continuation.
- **Free-transition low-Re cases (S809, E387).** They converge only to about 1e-5. The laminar-separation-bubble flow is not steady to round-off on these meshes.

Also fixed along the way:
- The roll-back snapshot now includes the outlet pressure set by the far-field vortex. A blow-up had left it NaN and poisoned every roll-back.
- A ZeroDivisionError from the numba Gauss-Seidel kernel on a diverged field is now caught.
- `SimpleSolver.solve()` stops at the first non-finite residual instead of iterating on NaNs.

Tests: `tests/test_robust.py` covers:
- the low alpha_u and pseudo-time solutions against the reference;
- a forced blow-up and its roll-back;
- MUMPS against SuperLU;
- the robust and plain NACA 0012 paths against each other.

## The adjoint design re-run (NACA 0012, minimum Cd at Cl 0.5, Re 3e6, t/c >= 0.119, level 0.5)

Same settings as `adjoint_report.md`, stopped after 77 evaluations (the earlier run stopped at 75).

| | earlier solver | this solver |
|---|---|---|
| converged evaluations | 26 of 75 | 40 of 77 |
| converged evaluations in the window 52-74 | 1 of 23 | 8 of 23 |
| best Cd (design mesh) | 0.0095468 | 0.0095419 |

The two runs follow the same path up to evaluation 52; after that the new solver converges more of the trial shapes. The gain in Cd is negligible: the design sits on a plateau where the remaining improvement needs a sharper nose than the CST bound allows, and the failed trial steps are the bound-limited shapes described above. The optimiser is no longer stopped by the solver on most of its line searches, but this design problem was already converged. A nose-radius constraint, or a finer mesh near the nose, would be needed to go further.
