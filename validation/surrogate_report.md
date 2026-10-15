# Gaussian-process polar surrogate (surrogate.py)

86 converged points from 9 CFD polars; inputs: alpha, log Re, t/c and its position, max camber and its position, leading-edge parameter, flap chord x deflection, flap deflection, free / fully turbulent. One GP each for Cl, log Cd, Cm (ARD squared exponential + noise, maximum marginal likelihood).

## Leave one polar out (uncalibrated sd)

| left out | points (α max) | rms Cl, all / α ≤ 12° | rms Cd, all / α ≤ 12° | Cl within 2 sd | Cd within 2 sd |
|---|---|---|---|---|---|
| naca0012 Re 1e+06 free | 4 (6°) | 0.014 / 0.014 | 8.2 % / 8.2 % | 100 % | 75 % |
| naca0012 Re 2.5e+06 free | 13 (18°) | 0.029 / 0.017 | 9.1 % / 7.5 % | 92 % | 62 % |
| naca0012 Re 3e+06 free | 7 (12°) | 0.014 / 0.014 | 6.3 % / 6.3 % | 86 % | 71 % |
| naca2412 Re 2.2e+06 free | 14 (22°) | 0.156 / 0.013 | 19.7 % / 4.8 % | 86 % | 71 % |
| naca2412 Re 3e+06 turbulent | 5 (8°) | 0.396 / 0.396 | 16.2 % / 16.2 % | 100 % | 100 % |
| naca2412 Re 3e+06 free | 7 (10°) | 0.011 / 0.011 | 3.8 % / 3.8 % | 71 % | 100 % |
| naca2412 Re 4.4e+06 free | 13 (18°) | 0.069 / 0.034 | 20.4 % / 9.1 % | 46 % | 77 % |
| naca2412@flap=25c10 Re 2.2e+06 free | 11 (16°) | 0.114 / 0.125 | 21.5 % / 23.5 % | 36 % | 91 % |
| naca2412@flap=25c20 Re 2.2e+06 free | 12 (16°) | 0.197 / 0.215 | 29.1 % / 31.8 % | 50 % | 50 % |

Calibration (sd multipliers so that 95 % of the left-out points fall in +-2 sd): Cl x2.33, logCd x1.83, Cm x1.38.

Reading: below stall, for an airfoil family the library contains, the surrogate is useful (see the α ≤ 12° columns); past stall (the wing polars reach 20°+, where few polars have data), for a geometry class it has seen once (flapped sections) or for the only fully turbulent polar it is poor, and the uncalibrated GP is over-confident there - hence the calibration. The surrogate improves with every polar added to the library (`surrogate.py train`), and `surrogate.py suggest` names the angle where one more RANS run would teach it the most.

