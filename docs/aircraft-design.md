# Aircraft design and structure

From requirements to a sized aircraft, then its wing structure and aeroelasticity.

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

## Wing structure (first pass)

Structure → **Wing structure** (`src/aerolab/aircraft/wing_structure.py`) sizes the wing box for every case of a Flight
loads run: spar caps, webs, skins and ribs (aluminium 2024-T3 / 7075-T6, CFRP or GFRP), the wing mass
against Raymer's Class II estimate, EI and GJ, deflection and twist, divergence and aileron-reversal
speeds and a flutter estimate (typical section with Theodorsen aerodynamics) against CS-23.629's 1.2 V_D.

```bash
cd src
python run.py loads ../cases/aircraft/demo_trainer.json --vc 60 --out ../cases/loads_demo
python run.py wing_structure ../cases/aircraft/demo_trainer.json ../cases/loads_demo/loads.json --out ../cases/struct_demo
```

## Aeroelasticity

Structure → **Aeroelasticity** (`src/aerolab/aircraft/aeroelastic.py`), after saving the wing structure with the aircraft:
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
