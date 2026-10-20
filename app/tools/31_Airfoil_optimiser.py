"""Airfoil optimiser: minimum drag at the design lift, by two methods."""
import views

views.subpages({
    "Fast: adjoint": ("25_Research.py", "gradients from the adjoint steer the shape; 10–30 min",
                      {"ONLY": "Adjoint design"}),
    "Thorough: RANS search": ("6_Optimizer.py", "every candidate solved with RANS, a surrogate steering the "
                              "search; about 1 h, more robust far from the start shape", None),
}, "sub_opt")
