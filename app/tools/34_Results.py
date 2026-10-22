"""Results: every run, every polar and every flow field in one place."""
import views

views.subpages({
    "Runs": ("sections/runs.py", "progress, logs and results of every analysis", None),
    "Polars": ("sections/polar_library.py", "overlay and download every polar computed", None),
    "Flow fields": ("sections/flow_fields.py", "meshes, pressure, flow cuts and wakes, 2D and 3D", None),
}, "sub_results")
