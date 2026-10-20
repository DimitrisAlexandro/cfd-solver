"""Sizing: the whole aircraft from the requirements, or the wing only."""
import views

views.subpages({
    "Whole aircraft": ("17_Conceptual_sizing.py", "requirements in, a complete aircraft model out", None),
    "Wing only": ("7_Wing_sizing.py", "mass, speeds and span limit in; the best wing planform out", None),
}, "sub_sizing")
views.workflow_footer("tools/33_Sizing.py")
