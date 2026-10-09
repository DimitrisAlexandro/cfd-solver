import time
from pathlib import Path

import streamlit as st

import common as C
import ui
import views

C.page_setup()
ui.header("Runs", "Every analysis started from this app, newest first. Runs continue when the "
                  "browser is closed.")

jobs = C.list_jobs()
tab1, tab2 = st.tabs(["Runs", "Open a result folder"])
with tab1:
    if not jobs:
        st.info("No runs yet. Start one from any analysis page.")
    for j in jobs:
        when = time.strftime("%d %b %H:%M", time.localtime(j["started"]))
        with st.expander(f"{ui.status_md(j['status'])} {j['title']}, {when}",
                         expanded=j["status"] == "running"):
            if j["status"] == "running":
                views.live_job(j["id"], j["kind"])
            else:
                c1, c2, _ = st.columns([1, 1, 4])
                show = c1.button("Open results", key="show" + j["id"], disabled=j["status"] != "finished")
                if c2.button("Show log", key="log" + j["id"]):
                    st.code(C.job_log(j["id"], tail=200), language=None)
                if show:
                    st.session_state["open_result"] = (j["kind"], j["out"])
                st.caption(j["out"])
with tab2:
    folders = sorted([p for p in C.CASES.iterdir() if p.is_dir() and not p.name.startswith("_")
                      and ((p / "polar.json").exists() or (p / "wing.json").exists())],
                     key=lambda p: -p.stat().st_mtime)
    if folders:
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        fo = c1.selectbox("Folder in cases/", folders, format_func=lambda p: p.name)
        if c2.button("Open", width="stretch"):
            st.session_state["open_result"] = ("wing" if (fo / "wing.json").exists() else "polar", str(fo))
    else:
        st.caption("No result folders in cases/ yet.")

res = st.session_state.get("open_result")
if res:
    kind, out = res
    (views.wing_results if kind == "wing" else views.polar_results)(out)
