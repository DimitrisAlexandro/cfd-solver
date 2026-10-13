# 3D RANS wing validation (3D-2)

Reference: AeroLab vortex lattice (6 chordwise panels) with the 2D SA polar of the section (NACA 0012, Re 3e+06, level 0.7).

| case | cells | alpha | CL (RANS) | CL (VLM) | dCL | CD (RANS) | CD (VLM: CDi + section) | iterations | continuity |
|---|---|---|---|---|---|---|---|---|---|
| rect | 252365 | 0 | -0.0008 | -0.0006 | +nan% | 0.00960 | 0.00937 | 136 | 1.7e-06 |
| rect | 252365 | 4 | 0.2989 | 0.2988 | +0.0% | 0.01495 | 0.01444 | 275 | 5.3e-06 |
| rect | 252365 | 8 | 0.6036 | 0.5853 | +3.1% | 0.03081 | 0.02912 | 325 | 4.1e-05 |
