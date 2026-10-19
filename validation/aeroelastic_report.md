# Aeroelasticity: validation

Reproduce with `python src/run.py validate_aeroelastic`.

## Unsteady aerodynamics

Vortex-ring lattice with an oscillating wake against Theodorsen's 2-D lift (aspect ratio 40, mid-span), plunge and pitch, k ≤ 0.6: largest error 3.0 % with 16 chordwise panels, 1.5 % with 32 (first-order convergence). The steady limit equals the aircraft vortex lattice (test).

## Goland wing (sea level)

| | flutter speed | flutter frequency |
|---|---|---|
| Goland (1945), strip theory | 137.2 m/s | 70.7 rad/s |
| this tool, strip theory (Theodorsen) + p-k | 137.1 m/s | 69.7 rad/s |
| this tool, 3D vortex-ring lattice + p-k | 154.8 m/s | 69.9 rad/s |

Modes (coupled bending-torsion beam FE): 48.2, 95.7, 243.9, 347.8 rad/s (uncoupled: 49.5 and 87.1 rad/s exactly, test). The 3D lattice includes the tip loss that strip theory ignores: the flutter speed rises; the frequency stays close.

## Demo trainer

- Aileron 15 Hz, c.g. 0.30 aileron chord aft of the hinge: flutter **97 m/s** (antisymmetric, 37.6 Hz) < 1.2 V_D = 100 m/s — the classic unbalanced-aileron flutter.
- Mass-balance sweep (flutter speed against the aileron c.g.):

| aileron c.g. aft of hinge | flutter speed |
|---|---|
| -0.15 | none below 2 V_D |
| -0.10 | none below 2 V_D |
| -0.05 | none below 2 V_D |
| +0.00 | 115.9 m/s |
| +0.05 | 103.7 m/s |
| +0.10 | 95.2 m/s |
| +0.15 | 93.2 m/s |
| +0.20 | 93.7 m/s |
| +0.25 | 95.0 m/s |
| +0.30 | 96.8 m/s |
| +0.35 | 99.0 m/s |
| +0.40 | 101.7 m/s |

- Divergence 217 m/s EAS; aileron reversal above 167 m/s EAS (2 V_D).
- Flexible against rigid wing loads (root bending, kN m): A  manoeuvre 27.29 → 27.55, C  manoeuvre 27.75 → 28.05, D  manoeuvre 28.06 → 28.35, G  negative -10.77 → -10.84, F  negative -11.00 → -11.34, E  V_D 0.22 → -0.36, gust + V_C 31.68 → 32.06, gust - V_C -17.06 → -17.50, gust + V_D 24.64 → 24.82, gust - V_D -9.55 → -10.44, cruise 1 g 7.46 → 7.42.

Limits: incompressible, clamped root, wing only, linear, no structural damping.
