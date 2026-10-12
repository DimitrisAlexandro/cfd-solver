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
