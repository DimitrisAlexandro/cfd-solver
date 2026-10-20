"""Airfoil polar: run a polar, or check the section and its mesh first."""
import views

views.subpages({
    "Run a polar": ("1_Airfoil_analysis.py", "Cl, Cd, Cm against α with the RANS solver", None),
    "Check geometry and mesh": ("4_Geometry_preview.py", "shape, thickness, inviscid Cp and the CFD mesh, "
                                "before spending CFD time", None),
}, "sub_polar")
