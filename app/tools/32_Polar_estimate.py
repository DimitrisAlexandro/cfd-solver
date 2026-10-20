"""Polar estimate: the polar surrogate on its own page."""
import runpy

import views

runpy.run_path(str(views.TOOLS_DIR / "25_Research.py"), init_globals={"ONLY": "Polar surrogate"},
               run_name="__main__")
