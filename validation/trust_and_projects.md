# Trust layer and project graph (2026-10-12)

Branch `trust-and-projects` (on top of `wing-structure`). New: `src/aerolab/workflow/trust.py`, `src/aerolab/workflow/project.py`,
`app/tools/23_Projects.py`, `app/sections/confidence.py`, `tests/test_trust.py` (11),
`tests/test_project.py` (6). No new dependencies.

## 1. Trust layer (`src/aerolab/workflow/trust.py`)

Every polar point gets a level (validated range / use with care / low confidence), an expected
uncertainty band for Cl and Cd, the reasons, and the nearest validated case.

- **Validation envelope**: 11 cases with the Re, Mach, α and transition range they cover and the error
  found there: NASA TMR NACA 0012 (fully turbulent), XFOIL tripped NACA 2412, XFOIL free transition
  (NACA 0012 at 1e6/3e6, NACA 2412 at 3e6; attached and 10–12° split), NASA LTPT Eppler 387 (2e5–3e5 and
  1e5 separately), laminar NACA 0012 Re 1000–5000, JAXA 30P30N, RAE 2822 case 9, subsonic compressible.
  Distance: root-sum-square of the excess outside the range, in units of a factor 1.8 in Re, 0.05 in
  Mach and 2° in α. Band = case error × (1 + distance) + terms below.
- **Point rules**: diverged / unsteady (red), transition not converged, past Cl max (SA over-predicts
  it), at Cl max, Prandtl–Glauert above M 0.3 (red above 0.5), supersonic pocket (RAE 2822: Cd +20 %),
  laminar solver above Re 2e4, bubbles below Re 1.5e5, no data for Re 2e4–8e4, coarse mesh, y+ > 1.5,
  t/c outside 7–16 %, multi-element near Cl max, transition point averaged around a bubble.
- **Grid study** (`trust.py grid-study`, app button): reruns chosen angles at mesh levels 0.7 / 1 / 1.4,
  Celik et al. (2008) apparent order, Richardson extrapolation, GCI (Fs 1.25, order capped at the
  formal 2 per Roache) relative and absolute. The GCI replaces the mesh term of the Cd band.
- **Other tools**: evidence and limits for wing, stability, loads, structure, aeroelasticity,
  performance, drag, sizing, weight and balance, drag rise, plus context notes (M > 0.3 on the
  aeroelastic lattice, AR < 3 vortex lift, Raymer flap increments, flutter below 1.2 V_D, ...).
- `airfoil.py` now records everything needed to rerun a polar in `polar.json` settings (airfoil, Re,
  N_crit, trip, regime, transition mode, low-Re mode).

### Checks
- GCI recovers p = 2 and the exact value for φ = 1 + 0.3 h² (also p = 1.6 on unequal ratios).
- Step C flat plate Cf (3264 / 13056 / 52224 cells): extrapolated 0.0027285 vs SA reference 0.002729.
- NACA 0012, Re 3e6, free transition, α 4°, levels 0.7 / 1 / 1.4: Cl 0.4304 / 0.4291 / 0.4257 → GCI 1.1 %;
  Cd 0.005916 / 0.006052 / 0.006038 → GCI 0.33 % (oscillatory). Cm ≈ 0 (symmetric section): relative
  GCI meaningless, absolute GCI reported.
- The same run at α 8° ended unsteady and is rated low confidence (Cd band ±28 %).

## 2. Project graph (`src/aerolab/workflow/project.py`)

A project holds variants; a variant is a graph of analyses with parameters and inputs. Node kinds:
aircraft model (source + field overrides such as `mass`, `surfaces.wing.span`), stability, drag,
weight and balance, loads, structure (also writes the aircraft with the sizing attached),
aeroelasticity, performance, CFD section polar.

- Fingerprint = hash(kind, parameters, content of external files, fingerprints of the inputs).
  Results live in `runs/<kind>-<fingerprint>/`: content-addressed, so identical inputs in two
  variants share one result, and undoing a change finds the old result again.
- Status per node: up to date / stale (with reasons: "mass: 750 → 820", "input changed: Flight loads",
  "an input file changed") / not run / running / failed; "solver code updated since this run" when the
  script changed. A new variant reports its changes against its parent.
- `run` executes only what is not up to date, in dependency order; a failure skips its dependants.
- `compare`: key results of every analysis per variant, change against the first, better/worse
  direction, trust level, and parameter differences.

### Demo (`project.py`, demo trainer, 2-core machine)
- Baseline chain (8 analyses): 55 s. Changing `loads.vc` 60 → 65 marks loads, structure and
  aeroelasticity stale; the rerun is 3 analyses.
- Variant `heavy` (mass 820 kg, span 11 m): root bending +11 %, wing mass +8.6 %, take-off +12.7 %,
  ROC −11 %, 3D flutter −2.3 %.
- Variant `cfrp` (structure material CFRP): only structure and aeroelasticity rerun (≈ 4 s), the
  other 6 results shared: wing mass −41 %, tip deflection +44 %, 3D flutter speed +22 %.

## App
- **Projects** page (top of the sidebar): graph coloured by state, status table with reasons,
  "Run what's out of date (N)" as a background job, key results with trust badges, parameter
  editing (with "changing this makes stale: ..."), add/remove analyses, variants, comparison table
  coloured better/worse with a chart and the parameter differences.
- **Confidence** page: check any condition before running, envelope map (Re–α boxes per case),
  validated-case table, evidence of every tool.
- Airfoil polar results: confidence line, uncertainty bars on the lift curve and drag polar,
  *Confidence and grid check* tab (per-angle reasons, GCI table, grid-check button).
- Polar library: confidence column. Stability, loads, structure, aeroelasticity, performance and
  wing results: a confidence expander with the evidence and context notes.

## Limits / next
- The envelope covers 2D sections; aircraft-level tools carry tool evidence, not a point-by-point
  envelope.
- Fingerprints don't include the section-polar cache (cases/wing_polars): recomputing a cached polar
  doesn't mark the aircraft analyses stale.
- Bands are calibrated on a handful of sections (NACA 00xx/24xx, E387, RAE 2822, 30P30N); unusual
  sections get a thickness penalty only.
