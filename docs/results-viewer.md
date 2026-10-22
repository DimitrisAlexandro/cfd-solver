# Results viewer

The app's Results → Flow fields page: 2D and 3D flow fields and wakes.

## Results viewer

The app's **Results → Flow fields** page opens every result under `cases/`, newest first:

- **2D** runs (single cases and each angle of a polar): any field (p, U, ν_t, vorticity,
  mesh quality) with view presets (full, near, leading edge, trailing edge, wake) or a
  custom window, cell edges, streamlines; the mesh alone; wall Cp / Cf / y+;
  convergence; case details.
- **3D** wings: rotatable surface coloured by Cp or Cf, with surface and symmetry-plane
  mesh; a span cut through any layer (field or mesh); a crossflow plane x = const
  (tip vortex: streamwise vorticity with in-plane velocity arrows); span loading and
  CL history.
- **Wake** (2D and 3D): the flow behind the body.
  - 2D: the velocity-deficit field with virtual rake lines, u/U∞ and total-pressure-loss
    profiles at each station, and the drag recovered from the wake (Betz wake-rake
    formula) against the surface-force drag.
  - 3D, the whole wake in one scene: the wing, the turbulence (nested shells of the SA eddy
    viscosity ν_t/ν, or an estimated turbulence intensity), the wake sheet (total-pressure
    loss), the tip vortex (Q-criterion), streamlines winding round it, and the mesh the wake
    passes through (on the wake surface and the symmetry plane). Isosurfaces are computed
    on the server (marching cubes, scikit-image), so only triangles reach the browser.
  - 3D: the wake seen from above (sheet and tip vortex), planes behind the wing in 3D,
    and a Betz / Maskell wake survey: profile drag (total-pressure loss) + induced drag
    (½∫ψξ, crossflow kinetic energy) against the surface CD and CL²/(πAR), shed
    circulation against the root circulation, spanwise profile drag, and the tip-vortex
    path, peak vorticity and core radius downstream.
  - On the AR 6 wing at α 4: wake CD 0.0140 vs surface 0.0149, induced 0.0045 vs vortex
    lattice 0.0048 just behind the wing, shed circulation 0.179 vs root 0.174; further
    downstream numerical diffusion of the coarser wake mesh lowers the induced drag.

The readers (`src/aerolab/mesh/postview.py`, `src/aerolab/mesh/wakeview.py`) need numpy, scipy and matplotlib only.
