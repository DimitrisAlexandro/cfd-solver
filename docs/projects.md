# Projects, checks, reports and confidence

The workspace tools: projects as dependency graphs, pre-run checks, templates, reports and the confidence of every result.

## Projects: a design as a dependency graph

A project holds variants of a design; each variant is a graph of analyses (aircraft model →
stability, drag, weight and balance, loads → wing structure → aeroelasticity, performance; CFD
section polars). Every result is stored under the fingerprint of its inputs, so:

- change a parameter or an input file and the page shows what is **out of date and why**
  ("mass: 750 → 820", "input changed: Flight loads");
- **Run what's out of date** recomputes only those analyses, in dependency order;
- variants **share** every result whose inputs are identical: a CFRP-wing variant reruns only the
  structure and aeroelasticity (about 4 s), not the whole chain;
- **Compare variants**: key results side by side with the change against the baseline (better / worse
  coloured), the confidence of each result, and the parameter differences.

```bash
cd src
python run.py project new "Trainer study" --template aircraft
python run.py project run ../cases/projects/trainer-study                   # ~1 min
python run.py project variant ../cases/projects/trainer-study cfrp --set "structure.material=CFRP (UD caps, ±45 skins)"
python run.py project status ../cases/projects/trainer-study                # what is stale, and why
python run.py project run ../cases/projects/trainer-study --variant cfrp    # 2 analyses, the rest shared
python run.py project compare ../cases/projects/trainer-study
```

## Pre-run checks, templates and reports

**Pre-run checks** (`src/aerolab/workflow/preflight.py`) run before anything is started, in about a second:

- airfoil polar: geometry (file, thickness, trailing edge, flap angle), Reynolds and Mach sanity, angle
  range, the expected y+ at the suction peak (fitted to 22 converged runs, within 5 %), optional mesh
  quality (builds the mesh, ~3 s: cells, non-orthogonality, size jumps), the expected confidence of every
  angle and the runtime, calibrated on the runs already on this machine;
- project variant: aircraft file and changes, section data at each surface's flight Reynolds number (with
  a one-click "compute section polars" action), speeds against the stall speed, propulsion and battery,
  structure inputs, and the runtime of everything that will run.

Errors block the Run button (with a "Run anyway" override); warnings explain what to change.

**Templates** (`src/aerolab/workflow/templates.py`): a ready-to-run project for each class — two-seat trainer,
motor-glider (electric self-launch), 15 m sailplane (glide polar), 15 kg fixed-wing UAV, eVTOL lift +
cruise (wing-borne flight) — with the geometry, masses, propulsion, flight conditions, loads category and
aileron balance of that class. Trainer, motor-glider and eVTOL are sized from requirements; sailplane and
UAV use typical mass fractions (Raymer's statistics don't hold there). Each template states its limits.

**Reports** (`src/aerolab/workflow/report.py`): one button turns a variant into a design report, or several variants into
a trade study, as PDF or Word: results with inputs and confidence, figures, key findings (better / worse
against the reference), evidence and limits of every tool, validated cases, and a traceability table
(result id = fingerprint of the inputs).

```bash
cd src
python run.py templates list
python run.py templates new glider "My sailplane"
python run.py preflight polar naca2412 --re 3e6 --alpha=-4:16:2 --deep
python run.py preflight project ../cases/projects/my-sailplane
python run.py project run ../cases/projects/my-sailplane
python run.py report design ../cases/projects/my-sailplane --format pdf
python run.py report trade ../cases/projects/trainer-study --variants baseline,heavy,cfrp --format docx
```

## Confidence of every result (trust layer)

`src/aerolab/workflow/trust.py` rates every polar point green / amber / red ("validated range", "use with care",
"low confidence") with an expected uncertainty band for Cl and Cd and the reasons:

- **validation envelope**: 11 validated cases (NASA TMR, XFOIL free and tripped, NASA LTPT E387,
  JAXA 30P30N, RAE 2822, laminar NACA 0012), each with the Re / Mach / α / transition range it covers
  and the error found there; the band grows with the distance from the nearest case;
- **point rules**: convergence, past Cl max (SA over-predicts it), shocks, Prandtl–Glauert above M 0.3,
  laminar bubbles below Re 1.5e5, coarse mesh, y+ above 1, unusual thickness;
- **grid study**: reruns chosen angles at mesh levels 0.7 / 1 / 1.4 and gives Richardson extrapolation
  and the grid convergence index (Celik et al. 2008); it replaces the mesh term of the band.

The polar page shows the bands as error bars and a *Confidence and grid check* tab; the Polar library
and the Projects page show the level of every result; the aircraft pages show the evidence and limits
of each tool; the **Confidence** page checks any flow condition before you run it.

```bash
python src/run.py trust check --re 3e6 --alpha 4 --transition free     # GREEN: Cl ±2 %, Cd ±3 %
python src/run.py trust polar ../cases/n2412                            # every angle
python src/run.py trust grid-study ../cases/n2412 --alpha 4,8 --jobs 2  # GCI
```
