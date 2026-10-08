import time
from pathlib import Path

import streamlit as st

import common as C
import views

C.page_setup("Jobs & results", "⏱️")
st.title("⏱️ Jobs & results")

jobs = C.list_jobs()
tab1, tab2 = st.tabs(["Jobs", "Open any result folder"])
with tab1:
    if not jobs:
        st.info("no jobs yet")
    icons = {"running": "🔄", "finished": "✅", "failed": "❌", "cancelled": "⏹️"}
    for j in jobs:
        age = (time.time() - j["started"]) / 60
        with st.expander(f"{icons[j['status']]} {j['title']} — {j['status']}, "
                         f"{time.strftime('%d %b %H:%M', time.localtime(j['started']))}",
                         expanded=j["status"] == "running"):
            if j["status"] == "running":
                views.live_job(j["id"], j["kind"])
            else:
                c1, c2 = st.columns([1, 1])
                show = c1.button("show results", key="show" + j["id"], disabled=j["status"] != "finished")
                if c2.button("show log", key="log" + j["id"]):
                    st.code(C.job_log(j["id"], tail=200))
                if show:
                    st.session_state["open_result"] = (j["kind"], j["out"])
with tab2:
    folders = sorted([p for p in C.CASES.iterdir() if p.is_dir() and not p.name.startswith("_")
                      and ((p / "polar.json").exists() or (p / "wing.json").exists())],
                     key=lambda p: -p.stat().st_mtime)
    if folders:
        fo = st.selectbox("result folder", folders, format_func=lambda p: p.name)
        if st.button("open"):
            st.session_state["open_result"] = ("wing" if (fo / "wing.json").exists() else "polar", str(fo))

res = st.session_state.get("open_result")
if res:
    st.divider()
    kind, out = res
    st.subheader(Path(out).name)
    (views.wing_results if kind == "wing" else views.polar_results)(out)
