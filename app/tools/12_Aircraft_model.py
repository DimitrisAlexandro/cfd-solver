import json
import re

import pandas as pd
import streamlit as st

import aircraft_views as AV
import common as C
import ui
import views

from aerolab.aircraft import aircraft as A

C.page_setup()
ui.header("Aircraft model", "Lifting surfaces, control surfaces, fuselage, mass and centre of "
                            "gravity. The stability, database and loads pages all use the aircraft "
                            "saved here.")

top = st.columns([3, 1, 1], vertical_alignment="bottom")
with top[0]:
    ac, path = AV.aircraft_picker()
if top[1].button("New from demo", width="stretch"):
    p = C.AIRCRAFT / "my_aircraft.json"
    k = 1
    while p.exists():
        k += 1
        p = C.AIRCRAFT / f"my_aircraft_{k}.json"
    d = A.demo_aircraft()
    d.name = f"My aircraft {k}"
    d.save(p)
    AV.select(p)
    st.rerun()
if top[2].button("Delete this file", width="stretch", disabled=len(list(C.AIRCRAFT.glob("*.json"))) < 2):
    st.session_state["confirm_delete"] = str(path)
if st.session_state.get("confirm_delete") == str(path):
    st.warning(f"Delete {path.name}? Results already computed with it are kept.")
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("Delete", type="primary"):
        path.unlink()
        st.session_state.pop("aircraft_file", None)
        st.session_state["aircraft_force"] = True
        st.session_state.pop("confirm_delete", None)
        st.rerun()
    if c2.button("Keep"):
        st.session_state.pop("confirm_delete", None)
        st.rerun()

st.page_link("tools/27_Wing_3D.py", label="3D CFD of this aircraft's wing")
tab_edit, tab_up = st.tabs(["Edit", "Upload your own"])

with tab_up:
    AV.upload_panel()

with tab_edit:
    left, right = st.columns([6, 5], gap="large")
    with left:
        with st.container(border=True):
            ui.step("Name, mass and centre of gravity", first=True)
            c1, c2 = st.columns([2, 1])
            name = c1.text_input("Name", ac.name)
            mass = c2.number_input("Mass (kg)", 0.1, 1e6, float(ac.mass))
            c1, c2, c3 = st.columns(3)
            cgx = c1.number_input("c.g. x (m from the nose)", -100.0, 200.0, float(ac.cg[0]), 0.01)
            cgz = c2.number_input("c.g. z (m)", -50.0, 50.0, float(ac.cg[2]), 0.01)
            cdx = c3.number_input("Extra drag ΔCD (gear, cooling…)", 0.0, 0.2, float(ac.cd_extra), 0.001,
                                  format="%.4f")
            est = ac.inertias() if ac.inertia is None else None
            use_I = st.checkbox("Enter moments of inertia (otherwise estimated)", value=ac.inertia is not None)
            I0 = ac.inertias()
            c1, c2, c3 = st.columns(3)
            Ixx = c1.number_input("Ixx (kg m²)", 0.0, 1e9, float(I0[0]), disabled=not use_I, format="%.4g")
            Iyy = c2.number_input("Iyy (kg m²)", 0.0, 1e9, float(I0[1]), disabled=not use_I, format="%.4g")
            Izz = c3.number_input("Izz (kg m²)", 0.0, 1e9, float(I0[2]), disabled=not use_I, format="%.4g")
            if not use_I:
                st.caption("Estimated from Roskam's radii of gyration for single-engine propeller aircraft.")

            ui.step("Lifting surfaces")
            st.caption("First row is the main wing (reference area, span and chord). Position = root "
                       "leading edge, m from the nose; z up. Sweep at the quarter chord; twist = tip "
                       "incidence relative to the root. A fin: symmetric off, vertical on, span = height.")
            sdf = st.data_editor(AV.surfaces_frame(ac), num_rows="dynamic", width="stretch", key=f"sdf{path}",
                                 column_config={
                                     "name": "Surface", "span": st.column_config.NumberColumn("Span (m)", format="%.3f"),
                                     "root_chord": st.column_config.NumberColumn("Root c (m)", format="%.3f"),
                                     "tip_chord": st.column_config.NumberColumn("Tip c (m)", format="%.3f"),
                                     "x": st.column_config.NumberColumn("x (m)", format="%.3f"),
                                     "y": st.column_config.NumberColumn("y (m)", format="%.3f"),
                                     "z": st.column_config.NumberColumn("z (m)", format="%.3f"),
                                     "sweep": st.column_config.NumberColumn("Sweep (°)", format="%.1f"),
                                     "dihedral": st.column_config.NumberColumn("Dihedral (°)", format="%.1f"),
                                     "twist": st.column_config.NumberColumn("Twist (°)", format="%.1f"),
                                     "incidence": st.column_config.NumberColumn("Incidence (°)", format="%.2f"),
                                     "airfoil": "Section", "tip_airfoil": "Tip section",
                                     "symmetric": "Mirrored", "vertical": "Vertical",
                                     "ns": st.column_config.NumberColumn("Panels span", min_value=2, max_value=60),
                                     "nc": st.column_config.NumberColumn("Panels chord", min_value=1, max_value=20)})
            ui.step("Panel breaks (kinks)")
            st.caption("Optional stations between root and tip for cranked or multi-panel surfaces "
                       "(XFLR5 / AVL style). y = distance from the root along the semi-span; dx, dz = "
                       "leading-edge position relative to the root leading edge; twist relative to the "
                       "root. The row above still sets the tip (tip chord, sweep and dihedral of the "
                       "root-to-tip line, tip twist). Straight lines between stations.")
            kdf = st.data_editor(AV.kinks_frame(ac), num_rows="dynamic", width="stretch", key=f"kdf{path}",
                                 column_config={
                                     "surface": st.column_config.SelectboxColumn(
                                         "Surface", options=[n for n in sdf["name"].astype(str) if n.strip()]),
                                     "y": st.column_config.NumberColumn("y from root (m)", min_value=0.0, format="%.3f"),
                                     "chord": st.column_config.NumberColumn("Chord (m)", min_value=0.0, format="%.3f"),
                                     "dx": st.column_config.NumberColumn("LE dx (m)", format="%.3f"),
                                     "dz": st.column_config.NumberColumn("LE dz (m)", format="%.3f"),
                                     "twist": st.column_config.NumberColumn("Twist (°)", format="%.2f")})
            ui.step("Control surfaces")
            st.caption("Span range as a fraction of the semi-span (0 root, 1 tip); chord fraction aft of "
                       "the hinge. Sign conventions: elevator + = trailing edge down, aileron + = roll "
                       "right, rudder + = trailing edge left.")
            names = [n for n in sdf["name"].astype(str) if n.strip()]
            cdf = st.data_editor(AV.controls_frame(ac), num_rows="dynamic", width="stretch", key=f"cdf{path}",
                                 column_config={
                                     "surface": st.column_config.SelectboxColumn("Surface", options=names),
                                     "name": st.column_config.SelectboxColumn(
                                         "Control", options=["elevator", "aileron", "rudder", "flap"]),
                                     "eta0": st.column_config.NumberColumn("From η", min_value=0.0, max_value=1.0, format="%.2f"),
                                     "eta1": st.column_config.NumberColumn("To η", min_value=0.0, max_value=1.0, format="%.2f"),
                                     "cf": st.column_config.NumberColumn("Chord fraction", min_value=0.05, max_value=0.6, format="%.2f"),
                                     "mode": st.column_config.SelectboxColumn("Deflection",
                                                                              options=["symmetric", "antisymmetric"])})
            ui.step("Fuselage")
            fu = ac.fuselage
            c1, c2, c3 = st.columns(3)
            fl = c1.number_input("Length (m)", 0.0, 200.0, float(fu.length))
            fd = c2.number_input("Max. diameter (m)", 0.0, 20.0, float(fu.diameter))
            fxm = c3.number_input("Max. diameter at (fraction of length)", 0.05, 0.9, float(fu.x_max), 0.05)
            c1, c2, c3 = st.columns(3)
            fx = c1.number_input("Nose x (m)", -50.0, 50.0, float(fu.x))
            fz = c2.number_input("Axis z (m)", -20.0, 20.0, float(fu.z))
            ftr = c3.number_input("Tail-end / max. diameter", 0.0, 1.0, float(fu.tail_ratio), 0.05)
            c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
            fname = c1.text_input("File name", path.stem)
            save = c2.button("Save aircraft", type="primary", width="stretch")

    try:
        new = AV.build_from_frames(
            ac, sdf, cdf, kdf, name=name, mass=mass, cg=(cgx, 0.0, cgz), cd_extra=cdx,
            inertia=(Ixx, Iyy, Izz) if use_I else None,
            fuselage=dict(length=fl, diameter=fd, x=fx, z=fz, x_max=fxm, tail_ratio=ftr))
        err = None
    except Exception as e:                          # noqa: BLE001
        new, err = ac, str(e)

    with right:
        if err:
            st.error(f"The tables can't be turned into an aircraft: {err}")
        AV.view3d(new)
        vh, vv = AV.tail_volumes(new)
        xm, _ = new.wing.mac_position()
        k = st.columns(3)
        k[0].metric("Wing area", f"{new.S:.2f} m²")
        k[1].metric("Span", f"{new.b:.2f} m")
        k[2].metric("Aspect ratio", f"{new.b ** 2 / new.S:.2f}")
        k = st.columns(3)
        k[0].metric("MAC", f"{new.c:.3f} m")
        k[1].metric("Wing loading", f"{new.mass * A.G0 / new.S:.0f} N/m²")
        k[2].metric("c.g.", f"{100 * (new.cg[0] - xm) / new.c:.1f} % MAC")
        k = st.columns(3)
        k[0].metric("Tail volume V_H", f"{vh:.3f}")
        k[1].metric("Tail volume V_V", f"{vv:.4f}")
        stat = k[2]
        st.caption("Typical light aircraft: horizontal tail volume 0.5–0.8, vertical 0.03–0.05.")
        if not err:
            try:
                an = A.Analysis(new, 50.0, 0.0, C.POLAR_CACHE)
                d = an.derivatives(2.0, downwash=False)
                stat.metric("Static margin", f"{100 * d['static_margin']:.1f} % MAC",
                            help=f"Stick fixed, quick check at 50 m/s. Neutral point at x = {d['x_np']:.3f} m. "
                                 "Full analysis on the Stability and control page.")
            except Exception as e:                  # noqa: BLE001
                st.caption(f"Quick stability check not available: {e}")

    if not err:
        with right:
            import tempfile
            from pathlib import Path as _P
            from aerolab.aircraft import aero_database as D
            with tempfile.TemporaryDirectory() as td:
                D.write_avl(new, td)
                avl = (_P(td) / "aircraft.avl").read_text()
                mass_f = (_P(td) / "aircraft.mass").read_text()
            from aerolab.aircraft import stl_aircraft as S
            c1, c2 = st.columns(2)
            c1.download_button("Download aircraft (JSON)", json.dumps(new.to_dict(), indent=1),
                               file_name=f"{path.stem}.json", width="stretch")
            c2.download_button("Download STL (surface mesh, m)", S.stl_bytes(S.aircraft_mesh(new), new.name),
                               file_name=f"{path.stem}.stl", width="stretch",
                               help="Closed surfaces of the lifting surfaces and the fuselage, for CAD, "
                                    "meshing or 3-D printing.")
            c1, c2 = st.columns(2)
            c1.download_button("Download AVL geometry", avl, file_name=f"{path.stem}.avl", width="stretch")
            c2.download_button("Download AVL mass file", mass_f, file_name=f"{path.stem}.mass", width="stretch")

    if save and not err:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", fname.strip()) or "aircraft"
        out = C.AIRCRAFT / f"{slug}.json"
        new.save(out)
        AV.select(out)
        st.success(f"Saved {out.name}.")
        st.rerun()

views.workflow_footer("tools/12_Aircraft_model.py")
