# Research tools

The discrete adjoint, transition physics and machine-learning tools.

## Research tools: adjoint, transition physics, machine learning

**In the app:** Advanced → Research, in four sections; each tool starts with a "What is this?" guide.
Adjoint design is under Airfoil → Airfoil optimiser (fast), the polar surrogate under Airfoil → Polar estimate.
- Adjoint gradients, drawn as arrows on the airfoil.
- **Mesh accuracy:** the real mesh error of one result, from a solve on the 4× finer nested mesh,
  with a map of where the error comes from; or a goal-oriented adaptation study against uniform refinement.
- **Swept-wing transition:** for one section, or along a whole wing from a saved aircraft.
- **Polar surrogate** (Airfoil → Polar estimate): with buttons to retrain it on this machine's polars (the model is saved in
  `cases/_surrogate/`) and to test it, leaving one polar out at a time.
- Results of the learned SA correction and of the learned laminar-bubble model.

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
