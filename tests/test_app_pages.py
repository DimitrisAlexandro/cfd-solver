"""The app: every page of the menu, and every tool of the merged pages, opens without an error."""
import os
import re
import sys
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).resolve().parents[1] / "app"
os.environ.setdefault("AEROLAB_EMBEDDED_WORKER", "0")      # no background worker from a test
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP.parent / "src"))

# the merged pages: switch key -> its tools
SUBPAGES = {"tools/30_Airfoil_polar.py": "sub_polar", "tools/31_Airfoil_optimiser.py": "sub_opt",
            "tools/33_Sizing.py": "sub_sizing", "tools/34_Results.py": "sub_results",
            "tools/35_Trust.py": "sub_trust"}


def menu_pages():
    return re.findall(r'P\("(tools/[^"]+\.py)"', (APP / "AeroLab.py").read_text())


@pytest.fixture(scope="module")
def app():
    at = st_testing.AppTest.from_file(str(APP / "AeroLab.py"), default_timeout=600).run()
    assert not at.exception
    return at


def test_menu_pages_exist_and_links_point_to_menu_pages():
    pages = set(menu_pages())
    assert all((APP / p).exists() for p in pages)
    # st.page_link only opens pages of the menu
    for f in [APP / "Home.py", APP / "views.py", APP / "aircraft_views.py", *APP.glob("tools/*.py")]:
        for target in re.findall(r'page_link\(\s*(?:str\(HERE / )?"(tools/[^"]+\.py)"', f.read_text()):
            assert target in pages, f"{f.name} links to {target}, not in the menu"
    import views
    assert all(p in pages for p, _ in views.WORKFLOW)


@pytest.mark.parametrize("page", menu_pages())
def test_page_opens(app, page):
    app.switch_page(page).run()
    assert not app.exception, [e.value for e in app.exception]
    if page in SUBPAGES:
        sw = app.segmented_control(key=SUBPAGES[page])
        for option in sw.options[1:]:
            sw.set_value(option).run()
            assert not app.exception, (option, [e.value for e in app.exception])
            sw = app.segmented_control(key=SUBPAGES[page])
