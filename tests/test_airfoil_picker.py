"""The app's airfoil picker lists the airfoil files uploaded before, on any page."""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")

APP = Path(__file__).resolve().parents[1] / "app"
os.environ.setdefault("AEROLAB_EMBEDDED_WORKER", "0")
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP.parent / "src"))

import common as C  # noqa: E402
import views  # noqa: E402


def _naca0012_dat(path):
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 40)))
    y = 0.6 * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    pts = np.vstack([np.c_[x, y][::-1], np.c_[x, -y][1:]])
    path.write_text("my foil\n" + "\n".join(f"{a:.6f} {b:.6f}" for a, b in pts) + "\n")


def test_uploaded_airfoils_listed_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CASES", tmp_path)
    assert views.uploaded_airfoils() == []                  # no uploads folder yet
    up = tmp_path / "_uploads"
    up.mkdir()
    _naca0012_dat(up / "old.dat")
    _naca0012_dat(up / "new.txt")
    t = time.time()
    os.utime(up / "old.dat", (t - 60, t - 60))
    (up / "notes.txt").write_text("not an airfoil\n")      # unreadable: left out
    (up / "plane.avl").write_text("x\n")                    # not a coordinate file
    (up / "stl_wing").mkdir()                               # STL import folder
    assert views.uploaded_airfoils() == [str(up / "new.txt"), str(up / "old.dat")]
    assert views._airfoil_label(str(up / "old.dat")) == "Uploaded: old.dat"
    assert views._airfoil_label("naca2412") == "NACA 2412"
    assert views._airfoil_label("tmr0012") == views.NAMES["tmr0012"]
