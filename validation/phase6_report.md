# Phase 6 — Quasi-3D wing analysis (2026-10-08)

Branch `phase6-wing` (on top of `phase5-compressible`). New: `src/wing.py`, `src/validate_wing.py`, `tests/test_wing.py` (5 tests). No new dependencies.

## What it does
`wing.py` turns your CFD section polars into wing aerodynamics in seconds:

```bash
cd src
python wing.py --airfoil naca2412 --span 10 --root-chord 1.6 --tip-chord 0.8 \
               --sweep 5 --dihedral 3 --twist -2 --speed 40 --alpha=-4:20:1
python wing.py --airfoil naca2412 --tip-airfoil naca0012 ...      # different tip section
python wing.py --polar ../cases/myfoil/polar.json ... --re 3e6     # reuse an airfoil.py polar
```
Output folder: `wing.png` (C_L–α, drag polar with the induced part, C_m, span loading vs elliptic, local c_l vs section c_l,max, planform with stalled strips), `report.txt`, `wing_polar.csv`, `wing.json` (all span distributions — for the app).

## Method
- **Vortex lattice** (horseshoe vortices, 4 chordwise × 2×40 spanwise panels, cosine spacing): taper, quarter-chord sweep, dihedral, linear twist, root incidence.
- **Nonlinear sections by decambering** (Mukherjee & Gopalarathnam 2006): each strip gets an extra incidence so that its lattice lift equals the CFD polar at its effective angle (Newton iteration) → nonlinear lift curve, stall onset, C_L,max; profile drag and section C_m from the polar (strip theory).
- **Induced drag** from a sine series of the span loading (exact for the computed loading).
- **Compressibility**: Prandtl–Glauert/Göthert (x stretched by 1/β); section polars from the compressible solver above M 0.3.
- **Section polars**: computed with the CFD solver at the root and tip Reynolds numbers (ISA atmosphere from speed and altitude), interpolated in log Re and along the span, Viterna–Corrigan beyond the computed angles; cached in `cases/wing_polars/` so a second wing with the same airfoil runs in seconds.

## Validation (`validate_wing.py` → `wing_validation.png`, `wing_report.txt`)
| check | result | reference |
|---|---|---|
| elliptic wing, span efficiency | e = 1.000 (AR 4–40) | 1 (exact) |
| elliptic AR 40, lift slope | 5.971 /rad | lifting line 5.984 (−0.2 %) |
| rectangular AR 1 / 2 / 6 / 20 | 1.477 / 2.498 / 4.241 / 5.450 | DATCOM 1.483 / 2.603 / 4.529 / 5.686 (−0.4 … −6.4 %) |
| AR 5, taper 0.5, sweep 0 / 30 / 45 / 60° | 4.094 / 3.844 / 3.388 / 2.605 | DATCOM 4.248 / 3.947 / 3.443 / 2.636 (−1.2 … −3.6 %) |
| compressibility AR 6, 30°, M 0 → 0.8 | 4.055 → 5.199 | DATCOM 4.162 → 5.256 (ratio within 1.5 %) |
| AR 100 wing with the CFD NACA 0012 polar | C_L within 1.5–2 % of the section c_l at α − α_i | 2D limit |

DATCOM's own accuracy is quoted as about ±5 %; the lattice sits 1–6 % below it (lifting-surface theory gives lower slopes than the lifting-line-based formula for unswept, low-aspect-ratio wings).

## Demo: CFD polars → wing (`wing_demo_naca2412.png`)
NACA 2412 wing, b 10 m, chords 1.6/0.8 m (AR 8.3, taper 0.5), 5° sweep, 3° dihedral, 2° washout, 40 m/s at sea level (Re_MAC 3.4e6, M 0.12). Section polars from the CFD solver at Re 2.2e6 and 4.4e6 (free transition, α −6…22°): 2 × ~15 min once, then every wing analysis takes **~7 s for 31 angles**.
- max L/D 34.8 at α 3° (C_D,i + C_D,p), e 0.98–0.99 in cruise
- **C_L,max 1.63 at α 21°** (section c_l,max ≈ 1.68–1.74 → ratio 0.95), stall starting at ~60 % semi-span (red strips)
- past C_L,max the strip-theory solution is not unique (flagged "not converged"): trend only.

## Found on the way
- ~~Free-transition CFD polars diverge near stall~~ **fixed 2026-10-09** (`transition.py`): three causes, see `validation/transition_stall_fix.md`. NACA 2412 polars at Re 2.2e6 and 4.4e6 now run with free transition from −6° to 22° without any fallback (the fallback stays in the code as a safety net). Demo wing: C_L,max 1.627 → 1.724, cruise unchanged.

## Limits
- Attached-flow lattice: no leading-edge vortex lift (delta wings), no fuselage/tail, symmetric flight only.
- Post-stall is strip theory with the section polar: C_L,max and the stall pattern are indicative (trend), not a 3D separated-flow prediction.
- Induced drag uses the planar formula (dihedral neglected there).
