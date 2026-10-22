# Low-Reynolds-number airfoils

Airfoils for UAVs and model aircraft, Re 2e4 to 5e5.

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
