# Design tools (2026-10-08)

Branch `app-streamlit`. Four tools, each a command-line script in `src/` and a page in the AeroLab app (`./run_app.sh`). Tests: `tests/test_design.py` (8).

## ✏️ Inverse design — `design_inverse.py`
The target pressure distribution (from an airfoil, edited with smooth ΔCp bumps or point by point, or a CSV) → the CST airfoil (8 Bernstein coefficients per side) whose panel-method Cp matches it (Levenberg–Marquardt), with an optional minimum thickness. **Seconds.** The designed `.dat` goes straight into the CFD analysis for the viscous check.
- Recovery check: the NACA 2412 Cp at α 3° from a NACA 0012 start → shape error 8e-8 c; RAE 2822 from NACA 0012 → 8e-8 c (`design_inverse_rae2822.png`).
- Edited targets need not be exactly realisable: the closest shape is returned (rms ΔCp shown).

## 🎯 Optimizer — `design_optimize.py`
CST fit of the start airfoil + smooth perturbations (4 variables per side), objective Cd at the target Cl(s) (= max L/D there), constraints thickness ≥ t_min and Cm ≥ Cm_min; every candidate is solved with the RANS solver (secant on α for each Cl); radial-basis-function surrogate + trust region; the winner is re-checked on the validated mesh.

Demo (`design_optimize_naca2412.png`): NACA 2412, Re 3e6, Cl 0.5, Cm ≥ −0.07, t/c ≥ 11.7 %, **free transition**, 30 evaluations (57 min, 2 cores):
| | Cd (validated mesh) | α | Cm | t/c | transition upper / lower |
|---|---|---|---|---|---|
| NACA 2412 | 0.00506 | 2.49° | −0.049 | 11.9 % | 0.39 / 0.80 |
| optimized | **0.00429 (−15 %)** | 2.29° | −0.052 | 11.9 % | 0.53 / 0.85 |
The gain comes from moving transition aft (a natural-laminar-flow trend: maximum thickness moved back, flatter upper surface), so it relies on the e^N transition model.
- **Fully turbulent** the same search found nothing (+0.2 % after verification): with turbulent boundary layers the NACA 2412 is already near the optimum and the differences are below the coarse-mesh noise. Optimize with the physics that matters.

## 📏 Wing sizing wizard — `design_wing_sizing.py`
Mission (mass, cruise speed and altitude, stall speed, span limit, sweep) + airfoil → for every aspect ratio × taper × washout: area from the stall speed with that planform's own C_L,max (wing tool, iterated), cruise C_L, α and wing L/D, stall-onset position; ranked by L/D with the span limit and a root-first-stall check (onset ≤ 60 % semi-span). Wing only.

Demo (`design_wing_sizing_demo.png`): 750 kg, 60 m/s at 1500 m, stall 27 m/s, span ≤ 11 m, NACA 2412 CFD polars: 24 planforms in 85 s → AR 10, taper 0.4, no twist: S 10.3 m², b 10.1 m, chords 1.45/0.58 m, cruise L/D 38 (wing), C_L,max 1.61, stall onset 49 % semi-span. Trends as in the textbooks: rectangular wings stall at the root (4 %), strong taper moves the stall outboard, washout moves it back inboard.

## 🌀 Drag-rise explorer — `design_drag_rise.py`
Mach sweep at fixed Cl with the compressible RANS solver (α by secant at every Mach, warm-started): Cd(M), drag-divergence Mach by dCd/dM = 0.1 (Douglas) and +20 counts (Boeing), Korn's equation for comparison, Mach fields with the sonic line.

Demo (`design_drag_rise_naca0012*.png`): NACA 0012, Re 6e6, Cl 0.3, M 0.50 → 0.82, mesh level 0.7, 47 min:
- **M_dd = 0.722 (Douglas), 0.709 (Boeing); Korn's equation (conventional section): 0.720.**
- Supersonic pocket from M 0.66, shock grows from M 0.70; at M 0.82 shock-induced separation — Cl 0.3 cannot be held at a reasonable α any more (flagged, not used for M_dd).
