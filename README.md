# AeroLab: CFD solver and aircraft aerodynamics workbench

A from-scratch aerodynamics toolchain: a hybrid mesher and finite-volume RANS
solver for airfoils (Spalart–Allmaras, e^N transition, compressible flow with
shocks, multi-element high-lift), a quasi-3D wing tool, whole-aircraft
stability and control, aero databases, flight loads, performance and design
tools — with a browser front end (AeroLab).

## Install and run

```bash
git clone <this repository> && cd cfd-solver
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./run_app.sh                 # opens http://localhost:8501
```

All results are written to `cases/` (not in git).

**Runs and the worker.** Every analysis started in the app is queued (`cases/_jobs`) and run by
a separate worker process, a few at a time:
- Locally the app starts the worker itself.
- Runs keep going when the browser is closed or the app is restarted.
- A run cut off by a restart of the worker is queued again when the worker comes back.
- The sidebar shows running and queued runs; **Runs** lists all of them.

Settings (environment variables):
- `AEROLAB_WORKER_SLOTS`: parallel runs (default: half the cores).
- `AEROLAB_JOB_THREADS`: threads per run.
- `AEROLAB_JOB_MEM_GB`: memory limit per run.

`python src/run.py jobqueue status` lists the queue; `python src/run.py jobqueue worker` runs a
worker by hand.

**Command line.** Every tool runs as `python src/run.py <tool> [arguments]`, for example
`python src/run.py airfoil polar naca0012 --re 3e6 --alpha=0:10:2 --out cases/n0012`.
`python src/run.py` alone lists the tools.

**Code layout.** `src/aerolab/` holds the code in packages:

| package | contents |
|---|---|
| `core` | finite-volume solvers and numerics |
| `mesh` | mesher, triangulation, mesh checks, viewers |
| `physics` | transition and boundary-layer models |
| `airfoil` | sections, CST, panel method, polars, multi-element |
| `design` | adjoint, optimisers, inverse design, sizing, surrogate |
| `ml` | field inversion and learned corrections |
| `aircraft` | wings, aircraft, loads, performance, structures |
| `workflow` | projects, pre-run checks, confidence, reports, job queue |
| `validation` | validation and benchmark scripts |

The app (Streamlit) is in `app/`.

## Put it online for a team

**Option A: Streamlit Community Cloud (free; aircraft tools and short runs).**
1. Go to https://share.streamlit.io, sign in with GitHub and allow access to this private repository.
2. Click **Create app** and fill in:
   - repository `DimitrisAlexandro/cfd-solver`
   - branch `main`
   - main file `app/AeroLab.py`
   - Advanced settings: Python 3.12
3. The app is private: invite your team by email with **Share**.

`app/requirements.txt` is the dependency list there.

Limits:
- At most 2 CPU cores and 2.7 GB of memory, and one private app per account.
- The app sleeps after 12 h without visitors.
- Results are lost when it restarts, so download what you need. Polar runs work but are slow.

**Option B: your own server (Docker; full speed, results kept).**
```bash
git clone https://github.com/DimitrisAlexandro/cfd-solver.git && cd cfd-solver
cp .env.example .env          # set AEROLAB_PASSWORD (empty = no password)
docker compose up -d --build  # http://<server>:8501
```
- Two services from one image:
  - `aerolab` is the web app.
  - `aerolab-worker` runs the queued analyses.
  
  Updating or restarting the app does not stop runs.
- Results, polars, aircraft and the job queue are kept in the `aerolab-cases` volume across updates.
- Update with `git pull && docker compose up -d --build`.
- For HTTPS, put the server behind a reverse proxy, for example Caddy: `aerolab.example.com { reverse_proxy localhost:8501 }`.
- The coupled solver's MKL library is optional: `docker compose build --build-arg PARDISO=1`.
- MUMPS, the faster adjoint solver, is built into the image. `--build-arg MUMPS=0` leaves it
  out; the adjoint then uses SuperLU and gives the same results, more slowly.
- Worker settings go in `.env`: `AEROLAB_WORKER_SLOTS`, `AEROLAB_JOB_THREADS`, `AEROLAB_JOB_MEM_GB`.
- The 3D solver's C++ kernel is compiled while the image is built, tuned to the CPU of the
  machine that builds it. To build an image that runs on other machines, set
  `AEROLAB_NATIVE_MARCH=x86-64-v2` in `.env`. Without a compiler the 3D solver still runs, on
  its slower numpy fallback; the build log says which one you got.

A fresh install starts with CFD section polars of the NACA 2412 (clean and with 10°/20° flaps)
and the NACA 0012 (`data/wing_polars`), so the aircraft tools use CFD sections from the start.

## What it does

| Group | Page | Command line |
|---|---|---|
| Analysis | Airfoil polar (incl. plain flaps, compressible), Multi-element, Wing | `src/aerolab/airfoil/airfoil.py`, `src/aerolab/aircraft/wing.py` |
| Aircraft | Aircraft model (3D), Weight and balance, Drag buildup, Propeller and engine, Stability and control, Aero database, Flight loads, Performance | `src/aerolab/aircraft/aircraft.py`, `src/aerolab/aircraft/weight_balance.py`, `src/aerolab/aircraft/drag_buildup.py`, `src/aerolab/aircraft/propeller.py`, `src/aerolab/aircraft/aero_database.py`, `src/aerolab/aircraft/loads.py`, `src/aerolab/aircraft/performance.py` |
| Design | Conceptual sizing, Inverse design, Shape optimizer, Wing sizing, Drag rise | `src/aerolab/aircraft/sizing.py`, `src/aerolab/design/design_*.py` |
| Workspace | Projects (templates, pre-run checks, dependency graph, variants, comparison, reports) | `src/aerolab/workflow/project.py`, `src/aerolab/workflow/templates.py`, `src/aerolab/workflow/preflight.py`, `src/aerolab/workflow/report.py` |
| Data | Geometry check, Polar library, Runs, Validation, Confidence | `src/aerolab/workflow/trust.py` |
| Research | Discrete adjoint and adjoint design, local transition transport, swept-wing transition, field inversion / ML, polar surrogate | `src/aerolab/design/adjoint.py`, `src/aerolab/design/design_adjoint.py`, `src/aerolab/physics/aft.py`, `src/aerolab/physics/swept_transition.py`, `src/aerolab/ml/fiml.py`, `src/aerolab/ml/bubble_ml.py`, `src/aerolab/design/surrogate.py` |

Examples:

```bash
cd src
python run.py airfoil polar naca2412 --re 3e6 --alpha=-4:16:2 --out ../cases/n2412
python run.py airfoil polar "naca2412@flap=25c20" --re 3e6 --alpha=-4:12:2 --out ../cases/n2412_flap20
python run.py aircraft demo && python run.py aircraft analyse ../cases/aircraft/demo_trainer.json --speed 50 --alt 1000
python run.py aero_database aircraft ../cases/aircraft/demo_trainer.json --alpha=-6:24:1 --beta=-10:10:2 \
       --mach 0.15 --deflect elevator=-20,-10,10,20 aileron=-15,15 rudder=-20,20 --out ../cases/db_demo
python run.py loads ../cases/aircraft/demo_trainer.json --vc 60 --category utility --fuel 80 --out ../cases/loads_demo
python run.py performance ../cases/aircraft/demo_trainer.json --power 75 --fuel 80 --out ../cases/perf_demo
```

Exports for other tools: JSBSim `<aerodynamics>` XML, AVL geometry + mass
files, XFOIL-format polars (XFLR5), Excel/CSV/JSON tables, Nastran
FORCE/MOMENT load sets, VTK flow fields (ParaView).

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
python run.py validate_core3d     # 3D-1: cavity / airfoil consistency with 2D, duct, sphere
python run.py validate_wing3d     # 3D-2: wings against the vortex lattice
```

The wing mesher sweeps the validated 2D hybrid airfoil mesh (y+ < 1 boundary
layer) through spanwise stations with taper, sweep, twist and dihedral, closes the
tip with a flat cap and relaxes the wall layers beyond the tip. A graded wake band
10 chords long follows the free stream at the run's angles, and the spanwise spacing
is capped around the tip vortex. Outputs: CL, CD (pressure/friction), Cm, span
loading c_l c(eta).

Rectangular, tapered, swept and twisted wings converge. Against the vortex lattice, CL
is within 2 %:

- taper 0.5, sweep 20°, twist −2°, AR 8: α 4 CL 0.2561 vs 0.2574;
- span loading within about 1 % from root to tip.

Results and limits: `validation/solver3d.md`, `validation/wing3d_report.md`,
`validation/core3d_report.md`.

With `--out`, every angle also saves its full flow field (`fields_a<alpha>.npz`).

## Results viewer

The app's **Data → Results viewer** page opens every result under `cases/`, newest first:

- **2D** runs (single cases and each angle of a polar): any field (p, U, ν_t, vorticity,
  mesh quality) with view presets (full, near, leading edge, trailing edge, wake) or a
  custom window, cell edges, streamlines; the mesh alone; wall Cp / Cf / y+;
  convergence; case details.
- **3D** wings: rotatable surface coloured by Cp or Cf, with surface and symmetry-plane
  mesh; a span cut through any layer (field or mesh); a crossflow plane x = const
  (tip vortex: streamwise vorticity with in-plane velocity arrows); span loading and
  CL history.
- **Wake** (2D and 3D): the flow behind the body.
  - 2D: the velocity-deficit field with virtual rake lines, u/U∞ and total-pressure-loss
    profiles at each station, and the drag recovered from the wake (Betz wake-rake
    formula) against the surface-force drag.
  - 3D, the whole wake in one scene: the wing, the turbulence (nested shells of the SA eddy
    viscosity ν_t/ν, or an estimated turbulence intensity), the wake sheet (total-pressure
    loss), the tip vortex (Q-criterion), streamlines winding round it, and the mesh the wake
    passes through (on the wake surface and the symmetry plane). Isosurfaces are computed
    on the server (marching cubes, scikit-image), so only triangles reach the browser.
  - 3D: the wake seen from above (sheet and tip vortex), planes behind the wing in 3D,
    and a Betz / Maskell wake survey: profile drag (total-pressure loss) + induced drag
    (½∫ψξ, crossflow kinetic energy) against the surface CD and CL²/(πAR), shed
    circulation against the root circulation, spanwise profile drag, and the tip-vortex
    path, peak vorticity and core radius downstream.
  - On the AR 6 wing at α 4: wake CD 0.0140 vs surface 0.0149, induced 0.0045 vs vortex
    lattice 0.0048 just behind the wing, shed circulation 0.179 vs root 0.174; further
    downstream numerical diffusion of the coarser wake mesh lowers the induced drag.

The readers (`src/aerolab/mesh/postview.py`, `src/aerolab/mesh/wakeview.py`) need numpy, scipy and matplotlib only.

## Validation

`validation/` holds every comparison with experiments and reference codes
(NASA TMR NACA 0012, JAXA 30P30N, RAE 2822 case 9, Ni bump, transition
experiments, wing theory and DATCOM, AeroSandbox vortex lattice). Reports:
`phase*_report.md`, `aircraft_report.md`, `design_report.md`,
`transition_stall_fix.md`. The app's Validation page shows them all.

Tests: `python -m pytest -q tests`.

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

## Designing a new aircraft

1. **Conceptual sizing** (`src/aerolab/aircraft/sizing.py`): mission, field lengths, climb, ceiling, configuration and
   propulsion in; constraint diagram, mission fuel or battery, Class I/II weights, converged take-off
   mass, geometry, component masses and c.g. positions, landing gear and propulsion out. Save it as an
   aircraft model and press **Verify**: the Performance tool checks every requirement.
2. **Weight and balance** (`src/aerolab/aircraft/weight_balance.py`): edit components, loading diagram, scissor plot for
   the tail size, landing-gear checks.
3. **Drag buildup** (`src/aerolab/aircraft/drag_buildup.py`): component CD0 with laminar runs, roughness, gear, cooling,
   struts; sets the model's extra drag.
4. **Propeller and engine** (`src/aerolab/aircraft/propeller.py`): minimum-induced-loss blade design and BEMT analysis,
   fixed pitch or constant speed, piston / turboprop / electric; the Performance page uses it.
5. Then Stability and control, Flight loads, Aero database and Performance as before.

Validation: `python src/run.py validate_design_tools --verify` (Cessna 172 and Pipistrel Velis Electro
sizing checks, drag buildup, propeller, scissor plot) → `validation/design_tools_report.md`.
## Low-Reynolds-number airfoils (UAV, model aircraft)

From Re 2e4 to 5e5 the airfoil tool runs in **low-Re mode**: RANS (Spalart–Allmaras) with e^N transition
and separation-induced transition for laminar separation bubbles. Transition sits at laminar
separation plus Mayle's long-bubble length (Re_l = 1000 Re_θs^0.7; for leading-edge bubbles not
shorter than Horton's 4×10⁴), and a turbulence trip scaled with the separating layer reattaches the
shear layer. Below Re 2e4 the flow is solved laminar (`--regime laminar` forces the laminar solver,
`--regime rans` the RANS model).

```bash
cd src
python run.py airfoil polar ../validation/reference/e387.dat --re 2e5 --alpha -2:10:2 --xfoil
python run.py validate_lowre            # Eppler 387 at Re 1e5, 2e5, 3e5 vs NASA Langley LTPT and XFOIL (~1.5 h)
```

## Wing structure (first pass)

Aircraft → **Wing structure** (`src/aerolab/aircraft/wing_structure.py`) sizes the wing box for every case of a Flight
loads run: spar caps, webs, skins and ribs (aluminium 2024-T3 / 7075-T6, CFRP or GFRP), the wing mass
against Raymer's Class II estimate, EI and GJ, deflection and twist, divergence and aileron-reversal
speeds and a flutter estimate (typical section with Theodorsen aerodynamics) against CS-23.629's 1.2 V_D.

```bash
cd src
python run.py loads ../cases/aircraft/demo_trainer.json --vc 60 --out ../cases/loads_demo
python run.py wing_structure ../cases/aircraft/demo_trainer.json ../cases/loads_demo/loads.json --out ../cases/struct_demo
```

## Aeroelasticity

Aircraft → **Aeroelasticity** (`src/aerolab/aircraft/aeroelastic.py`), after saving the wing structure with the aircraft:
wing modes (beam finite elements, with the aileron as a rigid surface on a hinge spring), 3D unsteady
aerodynamics (frequency-domain vortex-ring lattice), **p-k flutter** for symmetric and antisymmetric
modes including **aileron flutter** and a mass-balance sweep, and static aeroelasticity on the same
lattice (divergence, aileron reversal, elastic lift). Flight loads has a **flexible wing** option that
trims every case with the elastic twist and bending.

```bash
cd src
python run.py aeroelastic ../cases/aircraft/my_aircraft.json --out ../cases/aeroelastic
python run.py loads ../cases/aircraft/my_aircraft.json --vc 60 --flexible --out ../cases/loads_flex
python run.py validate_aeroelastic      # Theodorsen limit, Goland wing (137 m/s), aileron mass balance (~4 min)
```

## Research tools: adjoint, transition physics, machine learning

**Discrete adjoint** (`src/aerolab/design/adjoint.py`): the exact gradient of Cl, Cd or Cm with respect to every
design variable for one extra linear solve. The converged state is the root of a residual that
mirrors the solver term by term (the quantities the solver lags between iterations — Rhie–Chow
face fluxes, the skew-corrected pressure gradient, outlet fluxes, the far-field vortex — are
unknowns of their own), the Jacobian comes from coloured complex-step differences (exact to
round-off), and the transposed system is solved by sparse LU. Shape sensitivities move the mesh
with a deformation that turns the boundary-layer columns with the wall. Gradients agree with
finite differences of complete flow solutions to 1e-5–1e-3 (`validation/adjoint_report.md`).

```bash
cd src
python run.py adjoint gradient naca0012 --re 3e6 --alpha 4          # dCl, dCd, dCm / d(alpha, Re, 18 CST coefficients)
python run.py adjoint check naca0012 --re 3e6 --alpha 2              # against finite differences
python run.py design_adjoint --start naca0012 --re 3e6 --cl 0.5 --t-min 0.119   # 19-variable SLSQP design
python run.py adjoint_error estimate naca0012 --re 3e6 --alpha 4 --check        # adjoint-weighted residual
```

**Transition physics.** `--transition aft` runs a local amplification-factor transport model
(`src/aerolab/physics/aft.py`): the e^N envelope as transport equations on the RANS mesh (no surface marching),
with Thwaites' integral carried along for the boundary-layer history; within 0.01–0.05 c of XFOIL
on NACA 0012. `--tu` sets N_crit from the free-stream turbulence (Mack). Swept wings
(`src/aerolab/physics/swept_transition.py`, `wing.py --transition-3d ALPHA`): attachment-line contamination and
instability (Poll), crossflow (Falkner–Skan–Cooke profiles, `src/aerolab/physics/fsc.py`) and Tollmien–Schlichting
along the span.

```bash
python src/run.py airfoil polar naca0012 --re 3e6 --alpha 0:6:2 --transition aft --tu 0.07
python src/run.py swept_transition naca0012 --sweep 30 --re 1e7 --alpha 2 --out ../cases/swept
python src/run.py wing --airfoil naca2412 --span 30 --root-chord 6 --tip-chord 1.8 --sweep 30 --re 1e7 --transition-3d 3
```

**Machine learning.** Field inversion (`src/aerolab/ml/fiml.py`): a multiplier on the SA production term found
by adjoint-based optimisation against measured lift (or Cp), learned from local flow features by a
small neural network and used inside SA for new cases. Learned laminar-separation-bubble length
(`src/aerolab/ml/bubble_ml.py`, Eppler 387 LTPT data, Gaussian process with uncertainty). Polar surrogate
(`src/aerolab/design/surrogate.py`): a Gaussian process over every polar computed so far, with calibrated
uncertainty and the next most informative RANS run.

```bash
python src/run.py fiml invert naca0012 --re 3e6 --alpha 15 --out ../cases/fiml       # Gregory's Cl as target
python src/run.py bubble_ml scan && python src/run.py bubble_ml fit
python src/run.py surrogate train && python src/run.py surrogate predict naca2412 --re 3e6 --alpha 0:12:2
python src/run.py validate_research all                                              # all reports
```
