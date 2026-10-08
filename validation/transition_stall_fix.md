# Free transition near stall: fixed (2026-10-09)

Free-transition (e^N) polars used to diverge near stall (NACA 2412, Re 2.2e6, α ≥ 14°), and the wing tool reran those angles fully turbulent. Three causes, all in the RANS ↔ transition coupling (`src/transition.py`), plus two convergence-test tweaks (`src/polar.py`):

| # | Cause | Effect | Fix |
|---|---|---|---|
| 1 | At high α the stagnation point is on the lower surface. There the RANS wall pressure is a few 1e-3 above the total pressure, so U_e = √(1−2p) is clipped to 0 for 1–2 columns, and the laminar march started from U_e ≈ 0 | immediate spurious "laminar separation" at x 0.034 on the lower side; adding the bubble length (with the U_e floor 0.1: 0.18 c) to the *chordwise* x → transition estimate 0.20 c instead of 0.01 c at α 14 → the leading-edge laminar region burst into stall → NaN | the march starts at the last U_e ≈ 0 station (`march_start`); the bubble length is measured along the surface; the stretch from the stagnation point to the leading edge counts as laminar |
| 2 | Steep adverse gradient behind the suction peak: one RK4 sub-step threw H from 3.55 to −143. The separation test (H ≥ 3.8) never fired | "laminar to the trailing edge" on the suction side at α 10–16 → transition jumped to 0.25 c → NaN | H ≤ 1.02, θ ≤ 0 or non-finite values count as separation (`_separated`) |
| 3 | Cold start deep in stall (Re 4.4e6, α 18–22): with the lower surface laminar to the trailing edge, the trailing-edge corner cells (wall distance 1e-4 c) were set laminar while the separated upper-surface flow passes them, and ν_t ≈ 1e4 ν was switched off in one iteration | \|U\| 2.6 → 246 within 10 iterations at the trailing edge | wake and trailing-edge cells (last 0.3 % of chord) stay turbulent, as the wake in XFOIL; intermittency changes are blended in over the settle iterations (converged solution unchanged) |
| 4 | Slow drift / flip-flop (Cl ±1e-4, Cd ±0.65 %) with the residuals stalled at 3e-6 | α 10 and −6 ran to the 6000-iteration limit, "unsteady" | after 1500 iterations: converged when the forces vary < 5 × force_tol over 500 iterations; the small-oscillation Cd band is now 2 % peak-to-peak (the mean and amplitude are reported) |

## Result (`transition_stall_fix.png`)
NACA 2412, level 1, N_crit 9, −6…22° in steps of 2°:

| | Re 2.2e6 before | Re 2.2e6 after | Re 4.4e6 before | Re 4.4e6 after |
|---|---|---|---|---|
| fully turbulent fallbacks | 5 (α 14–22) | **0** | 2 (α 14, 22) | **0** |
| Cl_max (section) | 1.67 (α 16, turbulent) | **1.74 (α 18)** | 1.74 (α 16) | **1.86 (α 18)** |
| Cd at α 12 | 0.0170 | 0.0157 | 0.0151 | 0.0136 |
| α ≤ 8°, change vs before | — | Cl ≤ 0.7 %, Cd ≤ 1.3 % (largest at α −6: the Cd flip-flop) | — | Cl ≤ 0.2 %, Cd ≤ 0.3 % |

- The free-transition polars before the fix still contained bad points that hadn't diverged: at Re 4.4e6, α 16–18 "converged" with the spurious 0.2 c laminar region (Cd 0.0265 and 0.0443 instead of 0.0222 and 0.0322).
- α 20 (both Re) and α 22 (Re 4.4e6) are genuinely unsteady past the lift peak (fully turbulent too); the mean is reported.
- Wing demo (`wing.py`, span 10 m, taper 0.5): C_L,max 1.627 → **1.724**; cruise unchanged (max L/D 34.8 at α 3°).
- Sizing demo: C_L,max 1.61 → 1.70, so a smaller wing (9.7 m², AR 12) meets the 27 m/s stall speed.
- Caution: SA RANS tends to over-predict C_L,max for trailing-edge stall. Compare the stall region with wind-tunnel data before relying on it for design.

Tests: `tests/test_transition.py` adds two regression tests (synthetic high-α surface: the plain march reproduces the spurious separation at station 2; the fixed estimate equals the one without clipped stations; H never leaves the physical range).
