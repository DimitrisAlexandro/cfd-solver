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
pip install -r requirements.txt && pip install -e .
./run_app.sh                 # opens http://localhost:8501
```

`requirements.txt` adds the test runner and optional tools to `app/requirements.txt`, the one
pinned list of runtime dependencies (also used by Docker, Streamlit Cloud and `pyproject.toml`).
`pip install -e .` installs the `aerolab` package and command from `src/`, editable.

All results are written to `cases/` (not in git).

**Runs and the worker.** Every analysis started in the app is queued (`cases/_jobs`) and run by
a separate worker process, a few at a time:
- Locally the app starts the worker itself.
- Runs keep going when the browser is closed or the app is restarted.
- A run cut off by a restart of the worker is queued again when the worker comes back.
- The sidebar shows running and queued runs; **Results → Runs** lists all of them.

**App menu**, in the order of a design:
- **Start:** Overview, with a step-by-step guide; Projects.
- **Airfoil:** Airfoil polar (with a geometry and mesh check), Airfoil optimiser (fast adjoint or
  thorough RANS search), Polar estimate.
- **Wing:** Wing (vortex lattice), Wing 3D CFD.
- **Aircraft:** Sizing → Aircraft model → Weight and balance → Drag buildup → Propeller and engine →
  Stability and control → Performance.
- **Structure:** Flight loads → Wing structure → Aeroelasticity. Each aircraft and structure page ends
  with a link to the next step.
- **Results:** runs, polars and flow fields.
- **Advanced:** Research, Drag rise, Multi-element airfoil, Inverse design, Aero database.
- **Trust:** Validation and confidence.

Settings (environment variables):
- `AEROLAB_WORKER_SLOTS`: parallel runs (default: half the cores).
- `AEROLAB_JOB_THREADS`: threads per run.
- `AEROLAB_JOB_MEM_GB`: memory limit per run.

`aerolab jobqueue status` lists the queue; `aerolab jobqueue worker` runs a worker by hand.

**Command line.** Every tool runs as `aerolab <tool> [arguments]`, for example
`aerolab airfoil polar naca0012 --re 3e6 --alpha=0:10:2 --out cases/n0012`.
`aerolab` alone lists the tools. Without `pip install -e .`, `python src/run.py <tool> ...` does
the same, and the examples below use that form.

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

The app (Streamlit) is in `app/`: `AeroLab.py` (the menu), `tools/` (the menu pages) and
`sections/` (the tools that the merged pages, such as Airfoil polar or Results, switch between).

## What it does

| Group | Page | Command line |
|---|---|---|
| Analysis | Airfoil polar (incl. plain flaps, compressible), Multi-element, Wing, Wing 3D CFD | `src/aerolab/airfoil/airfoil.py`, `src/aerolab/aircraft/wing.py`, `src/aerolab/fv3d/wing3d.py` |
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

## Validation

`validation/` holds every comparison with experiments and reference codes
(NASA TMR NACA 0012, JAXA 30P30N, RAE 2822 case 9, Ni bump, transition
experiments, wing theory and DATCOM, AeroSandbox vortex lattice). Reports:
`phase*_report.md`, `aircraft_report.md`, `design_report.md`,
`transition_stall_fix.md`. The app's Validation page shows them all.

Tests: `python -m pytest -q` (about 15 min on 4 cores). GitHub Actions runs them, and builds the
Docker image, on every pull request (`.github/workflows/ci.yml`).

## Documentation

- [Deployment](docs/deployment.md): Streamlit Community Cloud, or your own server with Docker.
- [3D RANS solver](docs/3d-solver.md): wings in 3D, and its validation.
- [Results viewer](docs/results-viewer.md): 2D and 3D flow fields and wakes.
- [Projects, checks, reports and confidence](docs/projects.md).
- [Aircraft design and structure](docs/aircraft-design.md): sizing, wing structure, aeroelasticity.
- [Low-Reynolds-number airfoils](docs/low-reynolds.md).
- [Research tools](docs/research-tools.md): adjoint, transition physics, machine learning.
