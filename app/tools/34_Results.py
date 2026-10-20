"""Results: every run, every polar and every flow field in one place."""
import views

views.subpages({
    "Runs": ("10_Jobs_and_results.py", "progress, logs and results of every analysis", None),
    "Polars": ("9_Polar_library.py", "overlay and download every polar computed", None),
    "Flow fields": ("26_Results_viewer.py", "meshes, pressure, flow cuts and wakes, 2D and 3D", None),
}, "sub_results")
