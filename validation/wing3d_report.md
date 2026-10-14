# 3D RANS wing validation (3D-2)

Reference: AeroLab vortex lattice (6 chordwise panels) with the 2D SA polar of the section (NACA 0012, Re 3e+06, level 0.7).

| case | cells | alpha | CL (RANS) | CL (VLM) | dCL | CD (RANS) | CD (VLM: CDi + section) | iterations | continuity |
|---|---|---|---|---|---|---|---|---|---|
| rect | 414836 | 0 | -0.0005 | -0.0006 | +nan% | 0.00960 | 0.00937 | 143 | 2.5e-06 |
| rect | 414836 | 4 | 0.2991 | 0.2988 | +0.1% | 0.01494 | 0.01444 | 425 | 3.3e-06 |
| rect | 414836 | 8 | 0.6026 | 0.5853 | +3.0% | 0.03056 | 0.02912 | 450 | 3.7e-05 |
| taper | 429845 | 4 | 0.2561 | 0.2574 | -0.5% | 0.01204 | 0.01223 | 147 | 2.9e-06 |
