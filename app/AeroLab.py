"""AeroLab: interactive front end of the cfd-solver project.

    cd ~/cfd-solver && ./run_app.sh        (= streamlit run app/AeroLab.py)
"""
from pathlib import Path

import streamlit as st

HERE = Path(__file__).parent
st.set_page_config(page_title="AeroLab", page_icon=str(HERE / "assets" / "mark.png"),
                   layout="wide", initial_sidebar_state="expanded")

import ui  # noqa: E402

ui.brand(HERE / "assets")


def _password():
    """optional shared password: environment AEROLAB_PASSWORD or secrets 'password'"""
    import os
    pw = os.environ.get("AEROLAB_PASSWORD")
    if not pw:
        try:
            pw = st.secrets.get("password")
        except Exception:                        # noqa: BLE001  (no secrets file)
            pw = None
    return pw


_pw = _password()
if _pw and not st.session_state.get("auth_ok"):
    import hmac
    c = st.columns([1, 1, 1])[1]
    with c:
        st.markdown("### AeroLab")
        entered = st.text_input("Password", type="password")
        if entered:
            if hmac.compare_digest(entered, _pw):
                st.session_state["auth_ok"] = True
                st.rerun()
            st.error("Wrong password.")
    st.stop()


def P(file, title, default=False):
    url = None if default else title.lower().replace(" ", "-")
    return st.Page(str(HERE / file), title=title, default=default, url_path=url)


nav = st.navigation({
    "": [P("Home.py", "Overview", default=True)],
    "Analysis": [P("tools/1_Airfoil_analysis.py", "Airfoil polar"),
                 P("tools/2_Multi_element.py", "Multi-element airfoil"),
                 P("tools/3_Wing.py", "Wing")],
    "Aircraft": [P("tools/12_Aircraft_model.py", "Aircraft model"),
                 P("tools/18_Weight_balance.py", "Weight and balance"),
                 P("tools/19_Drag_buildup.py", "Drag buildup"),
                 P("tools/20_Propeller.py", "Propeller and engine"),
                 P("tools/13_Stability.py", "Stability and control"),
                 P("tools/14_Aero_database.py", "Aero database"),
                 P("tools/15_Loads.py", "Flight loads"),
                 P("tools/16_Performance.py", "Performance")],
    "Design": [P("tools/17_Conceptual_sizing.py", "Conceptual sizing"),
               P("tools/5_Inverse_design.py", "Inverse design"),
               P("tools/6_Optimizer.py", "Shape optimizer"),
               P("tools/7_Wing_sizing.py", "Wing sizing"),
               P("tools/8_Drag_rise.py", "Drag rise")],
    "Data": [P("tools/4_Geometry_preview.py", "Geometry check"),
             P("tools/9_Polar_library.py", "Polar library"),
             P("tools/10_Jobs_and_results.py", "Runs"),
             P("tools/11_Validation.py", "Validation")],
})
nav.run()
