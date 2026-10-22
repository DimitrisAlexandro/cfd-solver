"""Validation and confidence: can I trust this result?"""
import views

views.subpages({
    "Confidence": ("sections/confidence.py", "how far a result or a planned run can be trusted", None),
    "Validation": ("sections/validation.py", "the solver against wind-tunnel data and reference codes", None),
}, "sub_trust")
