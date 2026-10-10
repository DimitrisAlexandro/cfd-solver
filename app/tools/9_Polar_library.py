from pathlib import Path

import pandas as pd
import streamlit as st

import common as C
import ui

C.page_setup()
ui.header("Polar library", "Every polar from airfoil analyses and the wing tool's section cache. "
                           "Tick polars to overlay them.")

rows = C.scan_polars()
if not rows:
    st.info("No polars yet. Run an airfoil polar and it appears here.")
    st.stop()
df = pd.DataFrame(rows)
c1, c2 = st.columns([2, 1])
q = c1.text_input("Filter by section name", "", placeholder="naca2412")
src = c2.multiselect("Source", sorted(df.source.unique()), default=sorted(df.source.unique()))
view = df[df.name.str.contains(q, case=False) & df.source.isin(src)].reset_index(drop=True)
view.insert(0, "plot", False)
view["name"] = view["name"].map(C.display_name)
view["Re"] = view["Re"].map(C.fmt_re)
ed = st.data_editor(view.drop(columns=["path"]), hide_index=True, width="stretch",
                    disabled=[c for c in view.columns if c != "plot"],
                    column_config={"plot": st.column_config.CheckboxColumn("Plot", width="small"),
                                   "name": "Section", "Re": "Re",
                                   "Mach": st.column_config.NumberColumn("Mach", format="%.2f"),
                                   "transition": "Transition", "angles": "Angles", "source": "Source"})
sel = view[ed["plot"].values]
if len(sel):
    series = []
    for _, r in sel.iterrows():
        lab = f"{r['name']}, Re {r['Re']}" + (f", M {r['Mach']:g}" if r["Mach"] else "") + \
              (f", {r['transition']}" if r["transition"] else "")
        series.append((lab, C.load_polar_rows(r["path"])))
    figs = C.polar_figures(series)
    c1, c2 = st.columns(2)
    ui.plot(figs["cl"], container=c1)
    ui.plot(figs["polar"], container=c2)
    ui.plot(figs["cm"], container=c1)
    ui.plot(figs["ld"], container=c2)
    cols = st.columns(4)
    for i, (_, r) in enumerate(sel.iterrows()):
        p = Path(r["path"])
        cols[i % 4].download_button(f"Download {p.parent.name if p.name == 'polar.json' else p.stem}",
                                    p.read_bytes(), file_name=p.name, key=str(p))
else:
    st.caption("Tick Plot on one or more rows.")
