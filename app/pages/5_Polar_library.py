from pathlib import Path

import pandas as pd
import streamlit as st

import common as C

C.page_setup("Polar library", "📚")
st.title("📚 Polar library")
st.caption("Every polar computed by the airfoil analysis or cached by the wing tool. "
           "Select several to overlay them.")

rows = C.scan_polars()
if not rows:
    st.info("no polars yet — run an airfoil analysis first")
    st.stop()
df = pd.DataFrame(rows)
c1, c2 = st.columns([2, 1])
q = c1.text_input("filter (airfoil name)", "")
src = c2.multiselect("source", sorted(df.source.unique()), default=sorted(df.source.unique()))
view = df[df.name.str.contains(q, case=False) & df.source.isin(src)].reset_index(drop=True)
view.insert(0, "show", False)
ed = st.data_editor(view.drop(columns=["path"]), hide_index=True, width="stretch",
                    disabled=[c for c in view.columns if c != "show"])
sel = view[ed["show"].values]
if len(sel):
    series = []
    for _, r in sel.iterrows():
        lab = f"{r['name']} Re {r['Re']:.3g}" + (f" M {r['Mach']:g}" if r["Mach"] else "") + \
              (f" ({r['transition']})" if r["transition"] else "")
        series.append((lab, C.load_polar_rows(r["path"])))
    figs = C.polar_figures(series)
    c1, c2 = st.columns(2)
    c1.plotly_chart(figs["cl"], width="stretch")
    c2.plotly_chart(figs["polar"], width="stretch")
    c1.plotly_chart(figs["cm"], width="stretch")
    c2.plotly_chart(figs["ld"], width="stretch")
    for _, r in sel.iterrows():
        p = Path(r["path"])
        st.download_button(f"⬇ {p.parent.name}/{p.name}", p.read_bytes(), file_name=p.name,
                           key=str(p))
else:
    st.caption("tick 'show' to plot")
