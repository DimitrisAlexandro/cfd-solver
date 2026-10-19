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


# sections follow the workflow; page titles (and so the URLs) are unchanged
nav = st.navigation({
    "": [P("Home.py", "Overview", default=True), P("tools/23_Projects.py", "Projects"),
         P("tools/10_Jobs_and_results.py", "Runs")],
    "Airfoil": [P("tools/1_Airfoil_analysis.py", "Airfoil polar"),
                P("tools/2_Multi_element.py", "Multi-element airfoil"),
                P("tools/5_Inverse_design.py", "Inverse design"),
                P("tools/6_Optimizer.py", "Shape optimizer"),
                P("tools/8_Drag_rise.py", "Drag rise"),
                P("tools/9_Polar_library.py", "Polar library"),
                P("tools/4_Geometry_preview.py", "Geometry check")],
    "Wing and aircraft": [P("tools/3_Wing.py", "Wing"),
                          P("tools/7_Wing_sizing.py", "Wing sizing"),
                          P("tools/17_Conceptual_sizing.py", "Conceptual sizing"),
                          P("tools/12_Aircraft_model.py", "Aircraft model"),
                          P("tools/18_Weight_balance.py", "Weight and balance"),
                          P("tools/19_Drag_buildup.py", "Drag buildup"),
                          P("tools/20_Propeller.py", "Propeller and engine"),
                          P("tools/13_Stability.py", "Stability and control"),
                          P("tools/14_Aero_database.py", "Aero database"),
                          P("tools/15_Loads.py", "Flight loads"),
                          P("tools/21_Wing_structure.py", "Wing structure"),
                          P("tools/22_Aeroelasticity.py", "Aeroelasticity"),
                          P("tools/16_Performance.py", "Performance")],
    "Research": [P("tools/25_Research.py", "Adjoint, transition, ML")],
    "Trust": [P("tools/11_Validation.py", "Validation"),
              P("tools/24_Confidence.py", "Confidence")],
})


@st.fragment(run_every="5s")
def _run_monitor():
    """running / queued runs and the worker, on every page"""
    import common as C
    jobs = C.list_jobs()
    run = sum(j["status"] == "running" for j in jobs)
    que = sum(j["status"] == "queued" for j in jobs)
    alive = C.worker_alive()
    if run or que:
        st.markdown(f"**Runs** :blue-badge[{run} running] :violet-badge[{que} queued]")
        for j in [j for j in jobs if j["status"] in C.ACTIVE][:4]:
            st.caption(f"{'▶' if j['status'] == 'running' else '…'} {j['title'][:38]}")
        st.page_link(str(HERE / "tools/10_Jobs_and_results.py"), label="Open Runs")
    if not alive and (que or os.environ.get("AEROLAB_EMBEDDED_WORKER", "1") == "0"):
        st.caption(":orange-badge[worker not running]")


import os  # noqa: E402

with st.sidebar:
    _run_monitor()
nav.run()
