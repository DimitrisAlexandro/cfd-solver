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

## What it does

| Group | Page | Command line |
|---|---|---|
| Analysis | Airfoil polar (incl. plain flaps, compressible), Multi-element, Wing | `src/airfoil.py`, `src/wing.py` |
| Aircraft | Aircraft model (3D), Stability and control, Aero database, Flight loads, Performance | `src/aircraft.py`, `src/aero_database.py`, `src/loads.py`, `src/performance.py` |
| Design | Inverse design, Shape optimizer, Wing sizing, Drag rise | `src/design_*.py` |
| Data | Geometry check, Polar library, Runs, Validation | |

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
