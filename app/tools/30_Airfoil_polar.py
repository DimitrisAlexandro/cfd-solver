"""Airfoil polar: run a polar, or check the section and its mesh first."""
import views

views.subpages({
    "Run a polar": ("sections/run_polar.py", "Cl, Cd, Cm against α with the RANS solver", None),
    "Check geometry and mesh": ("sections/geometry_check.py", "shape, thickness, inviscid Cp and the CFD mesh, "
                                "before spending CFD time", None),
}, "sub_polar")
