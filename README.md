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

All results are written to `cases/` (not in git). Long analyses run as
background processes; they keep running when the browser is closed.

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
- Results, polars and aircraft are kept in the `aerolab-cases` volume across updates.
- Update with `git pull && docker compose up -d --build`.
- For HTTPS, put the server behind a reverse proxy, for example Caddy: `aerolab.example.com { reverse_proxy localhost:8501 }`.
- The coupled solver's MKL library is optional: `docker compose build --build-arg PARDISO=1`.

A fresh install starts with CFD section polars of the NACA 2412 (clean and with 10°/20° flaps)
and the NACA 0012 (`data/wing_polars`), so the aircraft tools use CFD sections from the start.

## What it does

| Group | Page | Command line |
|---|---|---|
| Analysis | Airfoil polar (incl. plain flaps, compressible), Multi-element, Wing | `src/airfoil.py`, `src/wing.py` |
| Aircraft | Aircraft model (3D), Weight and balance, Drag buildup, Propeller and engine, Stability and control, Aero database, Flight loads, Performance | `src/aircraft.py`, `src/weight_balance.py`, `src/drag_buildup.py`, `src/propeller.py`, `src/aero_database.py`, `src/loads.py`, `src/performance.py` |
| Design | Conceptual sizing, Inverse design, Shape optimizer, Wing sizing, Drag rise | `src/sizing.py`, `src/design_*.py` |
| Workspace | Projects (templates, pre-run checks, dependency graph, variants, comparison, reports) | `src/project.py`, `src/templates.py`, `src/preflight.py`, `src/report.py` |
| Data | Geometry check, Polar library, Runs, Validation, Confidence | `src/trust.py` |
| Research | Discrete adjoint and adjoint design, local transition transport, swept-wing transition, field inversion / ML, polar surrogate | `src/adjoint.py`, `src/design_adjoint.py`, `src/aft.py`, `src/swept_transition.py`, `src/fiml.py`, `src/bubble_ml.py`, `src/surrogate.py` |

Examples:

```bash
cd src
python airfoil.py polar naca2412 --re 3e6 --alpha=-4:16:2 --out ../cases/n2412
python airfoil.py polar "naca2412@flap=25c20" --re 3e6 --alpha=-4:12:2 --out ../cases/n2412_flap20
python aircraft.py demo && python aircraft.py analyse ../cases/aircraft/demo_trainer.json --speed 50 --alt 1000
python aero_database.py aircraft ../cases/aircraft/demo_trainer.json --alpha=-6:24:1 --beta=-10:10:2 \
       --mach 0.15 --deflect elevator=-20,-10,10,20 aileron=-15,15 rudder=-20,20 --out ../cases/db_demo
python loads.py ../cases/aircraft/demo_trainer.json --vc 60 --category utility --fuel 80 --out ../cases/loads_demo
python performance.py ../cases/aircraft/demo_trainer.json --power 75 --fuel 80 --out ../cases/perf_demo
```

Exports for other tools: JSBSim `<aerodynamics>` XML, AVL geometry + mass
files, XFOIL-format polars (XFLR5), Excel/CSV/JSON tables, Nastran
FORCE/MOMENT load sets, VTK flow fields (ParaView).

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
python project.py new "Trainer study" --template aircraft
python project.py run ../cases/projects/trainer-study                   # ~1 min
python project.py variant ../cases/projects/trainer-study cfrp --set "structure.material=CFRP (UD caps, ±45 skins)"
python project.py status ../cases/projects/trainer-study                # what is stale, and why
python project.py run ../cases/projects/trainer-study --variant cfrp    # 2 analyses, the rest shared
python project.py compare ../cases/projects/trainer-study
```

## Pre-run checks, templates and reports

**Pre-run checks** (`src/preflight.py`) run before anything is started, in about a second:

- airfoil polar: geometry (file, thickness, trailing edge, flap angle), Reynolds and Mach sanity, angle
  range, the expected y+ at the suction peak (fitted to 22 converged runs, within 5 %), optional mesh
  quality (builds the mesh, ~3 s: cells, non-orthogonality, size jumps), the expected confidence of every
  angle and the runtime, calibrated on the runs already on this machine;
- project variant: aircraft file and changes, section data at each surface's flight Reynolds number (with
  a one-click "compute section polars" action), speeds against the stall speed, propulsion and battery,
  structure inputs, and the runtime of everything that will run.

Errors block the Run button (with a "Run anyway" override); warnings explain what to change.

**Templates** (`src/templates.py`): a ready-to-run project for each class — two-seat trainer,
motor-glider (electric self-launch), 15 m sailplane (glide polar), 15 kg fixed-wing UAV, eVTOL lift +
cruise (wing-borne flight) — with the geometry, masses, propulsion, flight conditions, loads category and
aileron balance of that class. Trainer, motor-glider and eVTOL are sized from requirements; sailplane and
UAV use typical mass fractions (Raymer's statistics don't hold there). Each template states its limits.

**Reports** (`src/report.py`): one button turns a variant into a design report, or several variants into
a trade study, as PDF or Word: results with inputs and confidence, figures, key findings (better / worse
against the reference), evidence and limits of every tool, validated cases, and a traceability table
(result id = fingerprint of the inputs).

```bash
cd src
python templates.py list
python templates.py new glider "My sailplane"
python preflight.py polar naca2412 --re 3e6 --alpha=-4:16:2 --deep
python preflight.py project ../cases/projects/my-sailplane
python project.py run ../cases/projects/my-sailplane
python report.py design ../cases/projects/my-sailplane --format pdf
python report.py trade ../cases/projects/trainer-study --variants baseline,heavy,cfrp --format docx
```

## Confidence of every result (trust layer)

`src/trust.py` rates every polar point green / amber / red ("validated range", "use with care",
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
python trust.py check --re 3e6 --alpha 4 --transition free     # GREEN: Cl ±2 %, Cd ±3 %
python trust.py polar ../cases/n2412                            # every angle
python trust.py grid-study ../cases/n2412 --alpha 4,8 --jobs 2  # GCI
```

## Designing a new aircraft

1. **Conceptual sizing** (`src/sizing.py`): mission, field lengths, climb, ceiling, configuration and
   propulsion in; constraint diagram, mission fuel or battery, Class I/II weights, converged take-off
   mass, geometry, component masses and c.g. positions, landing gear and propulsion out. Save it as an
   aircraft model and press **Verify**: the Performance tool checks every requirement.
2. **Weight and balance** (`src/weight_balance.py`): edit components, loading diagram, scissor plot for
   the tail size, landing-gear checks.
3. **Drag buildup** (`src/drag_buildup.py`): component CD0 with laminar runs, roughness, gear, cooling,
   struts; sets the model's extra drag.
4. **Propeller and engine** (`src/propeller.py`): minimum-induced-loss blade design and BEMT analysis,
   fixed pitch or constant speed, piston / turboprop / electric; the Performance page uses it.
5. Then Stability and control, Flight loads, Aero database and Performance as before.

Validation: `python src/validate_design_tools.py --verify` (Cessna 172 and Pipistrel Velis Electro
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
python airfoil.py polar ../validation/reference/e387.dat --re 2e5 --alpha -2:10:2 --xfoil
python validate_lowre.py            # Eppler 387 at Re 1e5, 2e5, 3e5 vs NASA Langley LTPT and XFOIL (~1.5 h)
```

## Wing structure (first pass)

Aircraft → **Wing structure** (`src/wing_structure.py`) sizes the wing box for every case of a Flight
loads run: spar caps, webs, skins and ribs (aluminium 2024-T3 / 7075-T6, CFRP or GFRP), the wing mass
against Raymer's Class II estimate, EI and GJ, deflection and twist, divergence and aileron-reversal
speeds and a flutter estimate (typical section with Theodorsen aerodynamics) against CS-23.629's 1.2 V_D.

```bash
cd src
python loads.py ../cases/aircraft/demo_trainer.json --vc 60 --out ../cases/loads_demo
python wing_structure.py ../cases/aircraft/demo_trainer.json ../cases/loads_demo/loads.json --out ../cases/struct_demo
```

## Aeroelasticity

Aircraft → **Aeroelasticity** (`src/aeroelastic.py`), after saving the wing structure with the aircraft:
wing modes (beam finite elements, with the aileron as a rigid surface on a hinge spring), 3D unsteady
aerodynamics (frequency-domain vortex-ring lattice), **p-k flutter** for symmetric and antisymmetric
modes including **aileron flutter** and a mass-balance sweep, and static aeroelasticity on the same
lattice (divergence, aileron reversal, elastic lift). Flight loads has a **flexible wing** option that
trims every case with the elastic twist and bending.

```bash
cd src
python aeroelastic.py ../cases/aircraft/my_aircraft.json --out ../cases/aeroelastic
python loads.py ../cases/aircraft/my_aircraft.json --vc 60 --flexible --out ../cases/loads_flex
python validate_aeroelastic.py      # Theodorsen limit, Goland wing (137 m/s), aileron mass balance (~4 min)
```

## Research tools: adjoint, transition physics, machine learning

**Discrete adjoint** (`src/adjoint.py`): the exact gradient of Cl, Cd or Cm with respect to every
design variable for one extra linear solve. The converged state is the root of a residual that
mirrors the solver term by term (the quantities the solver lags between iterations — Rhie–Chow
face fluxes, the skew-corrected pressure gradient, outlet fluxes, the far-field vortex — are
unknowns of their own), the Jacobian comes from coloured complex-step differences (exact to
round-off), and the transposed system is solved by sparse LU. Shape sensitivities move the mesh
with a deformation that turns the boundary-layer columns with the wall. Gradients agree with
finite differences of complete flow solutions to 1e-5–1e-3 (`validation/adjoint_report.md`).

```bash
cd src
python adjoint.py gradient naca0012 --re 3e6 --alpha 4          # dCl, dCd, dCm / d(alpha, Re, 18 CST coefficients)
python adjoint.py check naca0012 --re 3e6 --alpha 2              # against finite differences
python design_adjoint.py --start naca0012 --re 3e6 --cl 0.5 --t-min 0.119   # 19-variable SLSQP design
python adjoint_error.py estimate naca0012 --re 3e6 --alpha 4 --check        # adjoint-weighted residual
```

**Transition physics.** `--transition aft` runs a local amplification-factor transport model
(`src/aft.py`): the e^N envelope as transport equations on the RANS mesh (no surface marching),
with Thwaites' integral carried along for the boundary-layer history; within 0.01–0.05 c of XFOIL
on NACA 0012. `--tu` sets N_crit from the free-stream turbulence (Mack). Swept wings
(`src/swept_transition.py`, `wing.py --transition-3d ALPHA`): attachment-line contamination and
instability (Poll), crossflow (Falkner–Skan–Cooke profiles, `src/fsc.py`) and Tollmien–Schlichting
along the span.

```bash
python airfoil.py polar naca0012 --re 3e6 --alpha 0:6:2 --transition aft --tu 0.07
python swept_transition.py naca0012 --sweep 30 --re 1e7 --alpha 2 --out ../cases/swept
python wing.py --airfoil naca2412 --span 30 --root-chord 6 --tip-chord 1.8 --sweep 30 --re 1e7 --transition-3d 3
```

**Machine learning.** Field inversion (`src/fiml.py`): a multiplier on the SA production term found
by adjoint-based optimisation against measured lift (or Cp), learned from local flow features by a
small neural network and used inside SA for new cases. Learned laminar-separation-bubble length
(`src/bubble_ml.py`, Eppler 387 LTPT data, Gaussian process with uncertainty). Polar surrogate
(`src/surrogate.py`): a Gaussian process over every polar computed so far, with calibrated
uncertainty and the next most informative RANS run.

```bash
python fiml.py invert naca0012 --re 3e6 --alpha 15 --out ../cases/fiml       # Gregory's Cl as target
python bubble_ml.py scan && python bubble_ml.py fit
python surrogate.py train && python surrogate.py predict naca2412 --re 3e6 --alpha 0:12:2
python validate_research.py all                                              # all reports
```
