# Phase 5 — Compressible flow (2026-10-07)

Branch `phase5-compressible` (on top of `phase4b-coupled`). New: `compressible.py`, `validate_bump.py`, `validate_transonic.py`, `tests/test_compressible.py` (8 tests). No new dependencies.

## What it is
A **pressure-based compressible solver** on the same meshes, discretisation and boundary handling as the incompressible SIMPLEC solver:
- ideal gas, **total-enthalpy energy equation** (adiabatic walls, viscous work), Sutherland viscosity, full viscous stress incl. −⅔μ∇·U;
- compressible SIMPLEC: Rhie–Chow + Majumdar mass fluxes, pressure correction with the **ψ p′ convection term** (elliptic at low Mach, hyperbolic in supersonic zones → shocks captured), density at faces by **limited linear upwinding**;
- **compressible Spalart–Allmaras** (ρν̃ form, NASA TMR);
- far field: point vortex with Prandtl–Glauert scaling + isentropic outlet pressure; total-pressure inlets for internal flows;
- Euler mode (`inviscid=True`): slip walls, μ = 0.

In the polar tool: `python src/run.py airfoil polar <airfoil> --re R --mach M ...` now runs this solver (`--prandtl-glauert` keeps the old incompressible + correction). Above M 0.4 transition is fully turbulent or fixed (`--xtr`); free e^N transition only below M 0.4 (incompressible stability theory, now on the isentropic edge velocity). Output fields add Mach, ρ, T and total-pressure loss.

## Validation
| case | quantity | this solver | reference |
|---|---|---|---|
| channel, M 0.001 | velocity vs incompressible solver | 3e-7 (no stress terms) / 1.3e-4 | — (consistency) |
| Ni bump M 0.5 (Euler) | M_max, fore-aft asymmetry, entropy | 0.688, 0.015, 0.4 % | ≈0.69, symmetric, isentropic |
| Ni bump M 0.675 (Euler) | shock position, M_max | **x = 1.722 (72 % of the bump)**, 1.31 | ≈72 %, ≈1.3–1.4 |
| NACA 0012 M 0.5 α 1.25 (Euler) | Cl, Cd | 0.1780, 0.0003 | ≈0.18, 0 (no shock) |
| NACA 0012 M 0.8 α 1.25 (Euler) | Cl, Cd, Cm, shocks | 0.357, 0.0233, −0.039, upper 0.65 / lower 0.36 | ≈0.35–0.36, ≈0.022–0.023, ≈−0.04 (AGARD AR-211, Vassberg & Jameson 2010) |
| **RAE 2822 case 9 (RANS, SA)** M 0.734 α 2.79 Re 6.5e6 | Cl / Cd / Cm | **0.814 / 0.0201 / −0.0997** (29k cells) | **experiment 0.803 / 0.0168 / −0.099** |

RAE 2822 grid dependence (levels 0.7 / 1 / 1.4 = 15k / 30k / 55k cells): Cl 0.808 / 0.814 / 0.818, Cd 0.0211 / 0.0201 / 0.0196, Cm −0.101 / −0.0997 / −0.0995, shock x/c 0.597 / 0.587 / 0.580. The Cp distribution (`transonic_rae2822_cp.png`) follows the experiment closely, including the suction plateau and the lower surface; the shock sits ~2–3 % c aft of the experiment and the drag is high, which is the known behaviour of SA at these conditions (published SA results are typically Cd ≈ 0.018–0.020 with the shock slightly aft).

Plots: `bump_mach.png`, `transonic_naca0012_euler_cp.png`, `transonic_rae2822_cp.png`, `transonic_rae2822_mach.png`; numbers in `bump_report.txt`, `transonic_results.json`.

## Bugs found on the way (fixed)
1. **Transpose viscous stress dropped on outlet faces** (old, affects every incompressible SA run): the laminar channel with the term switched on was off by 4 % instead of 0.01 %. At 100-chord far fields the effect on airfoil forces is negligible, but it was wrong.
2. (Phase 5 internal) isothermal ψ = 1/RT overshoots the density correction by γ when H is fixed → isentropic ψ = 1/a²; the Majumdar term must use the velocity part of the corrected flux only; a velocity inlet must not take its density from the extrapolated pressure (global feedback) → isentropic free-stream density; total-pressure inlets linearised d(ρU)/dp = −(1−M²)/U.

## Known limitations
- Euler, transonic: small Cp wiggles in the supersonic zone and a post-shock overshoot (pressure-based scheme, central pressure); forces converge to 3–4 digits, residuals stall at ~2e-5 because of the limiter.
- A cold start runs 30 iterations with frozen density first; transonic points need ~1000–3000 iterations (RAE 2822: 6.5 min on 30k cells, 2-core machine).
- e^N transition uses incompressible stability theory → only below M 0.4.
- Supersonic free streams (M > 1, bow shocks, supersonic outflow) are not set up: the far-field conditions assume subsonic in- and outflow.

## Next options
- RAE 2822 case 6 / NACA 0012 transonic RANS polar (drag-divergence curve) as an app demo.
- Unsteady stall or quasi-3D wings (the other Phase 5 options).
