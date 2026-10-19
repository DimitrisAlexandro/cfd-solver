# Pre-run checks, project templates and reports (2026-10-12)

Branch `prerun-templates-reports` (on top of `trust-and-projects`). New: `src/aerolab/workflow/preflight.py`,
`src/aerolab/workflow/templates.py`, `src/aerolab/workflow/report.py`, `tests/test_prerun_templates_reports.py` (13). New dependencies:
`python-docx`, `reportlab` (reports).

## Pre-run checks (`src/aerolab/workflow/preflight.py`)
Levels ok / info / warn / error; an error disables Run (with a "Run anyway" override).

**Airfoil polar** (about 1 s; `--deep` adds the mesh, ~3 s):
- geometry: file and format, t/c outside the validated 7–16 %, blunt trailing edge, flap above 20°;
- Reynolds number: < 1e3 error, multi-element < 1e5 error, laminar < 2e4, no data 2e4–8e4, bubbles < 1.5e5, > 5e7;
- Mach: ≥ 0.95 error, > 0.8 strong shocks, Prandtl–Glauert above 0.3 / 0.5, free transition switched off ≥ 0.4;
- angles: empty or |α| > 25° error, > 18° unsteady, > 30 angles;
- **y+ at the suction peak**: y+max ≈ 0.523 (U_e,max/U)^0.81 / level, U_e,max from the panel method.
  Fitted to 22 converged NACA 0012 / 2412 runs (Re 3e6 and 6e6, α −4…16°), within 5 %
  (NACA 2412 Re 3e6: 0.63 at 0°, 1.52 at 16°, reproduced to 8 % in the test);
- mesh quality (deep): cells, first cell, non-orthogonality max / 99th percentile, clamped faces, largest
  size jump between neighbouring triangles (NACA 2412 level 1: 27.5k cells, 44°, 4.7×);
- expected confidence of every angle (trust layer) and the runtime: per-angle cost model (free
  transition ×1.35, compressible ×1.6–3, separated α, low-Re mode, multi-element) × a machine factor
  calibrated on the converged angles already in cases/.

**Project variant**: aircraft source and changes (invalid paths are errors), c.g. range, section data at
each surface's flight Re against the cached CFD polars (none → thin-airfoil estimate; outside 0.5–2× →
extrapolated) with a one-click "compute section polars at Re X" job, speeds against the stall speed
(stability/drag within 15 %, V_C below 1.3 V_S), propulsion (glider mode, missing BEMT model, electric
without battery energy), spar positions, and the runtime from the medians of earlier project runs.
The design report repeats the warnings under "Setup checks".

## Templates (`src/aerolab/workflow/templates.py`)
| template | method | MTOW | span | key results (baseline run) |
|---|---|---|---|---|
| Two-seat trainer | sized (Raymer Class II), piston | 782 kg | 9.3 m | SM 20 %, L/D 19, V_max 78 m/s |
| Motor-glider | sized (powered-sailplane statistics), electric self-launch | 495 kg | 13.6 m | SM 6.9 %, L/D 29, flutter 118 m/s |
| Sailplane 15 m | layout rules + typical mass fractions, no engine | 400 kg | 14.9 m | SM 14 %, L/D 38, min sink 0.6–0.7 m/s class |
| Fixed-wing UAV | layout rules + typical mass fractions, electric | 15 kg | 3.5 m | SM 17 %; section data extrapolated (flagged) |
| eVTOL lift + cruise | sized, electric, c.g. 12 % MAC | 569 kg | 6.8 m | SM 11 %, L/D 16; hover not modelled (stated) |

Each template sets the flight conditions of every analysis, the loads category, the structure material and
aileron mass balance, and lists its limits in the project. NACA 2412 sections by default (CFD polars ship
with AeroLab).

Supporting changes: `sizing.finalize()` (the end of the sizing loop as a function, results unchanged),
`performance.py --power 0` glider mode (trimmed polar, stall, glide ratio, sink polar), project nodes:
drag `engine`, performance `propulsion` (simple / the aircraft's BEMT model), `prop_diameter`, `v_design`,
structure aileron mass / balance / frequency. The section-polar library is now part of the aircraft
node's fingerprint: computing new polars marks the aircraft analyses out of date (reason shown).

## Reports (`src/aerolab/workflow/report.py`)
Block document model rendered to PDF (reportlab) and Word (python-docx).
- **Design report** (one variant): summary table with confidence, setup checks, template notes, aircraft
  (geometry, surfaces, planform figure), one section per analysis (inputs, results, confidence and
  notes, the tools' figures plus wing-box and p-k flutter plots), assumptions / limits / evidence per
  tool, validated cases, traceability table (result id = input fingerprint).
- **Trade study** (several variants): parameter differences, key findings (largest improvements and
  degradations per variant), caveats (stale results, results to use with care), comparison table coloured
  better/worse, change chart, assumptions and references.
- About 2–3 s per report; Projects → Reports tab, or `python src/run.py report design|trade ...`.

## Checks
- 216 tests pass (13 new here).
- All five templates run end to end (8 analyses each).
- Every changed page loaded headless; report buttons, deep mesh check and template creation driven through
  the app.

## Limits / next
- Template masses for the sailplane and UAV are fractions, not a weight model; the structure sizing uses
  light-aircraft minimum gauges (UAV wing box heavy).
- Runtime estimates are ±2×; they tighten as runs accumulate on the machine.
- The performance range is at the best-range speed without reserve; the sizing mission uses the cruise
  speed with reserves, so the two differ.
