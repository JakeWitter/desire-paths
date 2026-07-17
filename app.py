import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt

from paths.world_manager import WorldManager
from paths.visualise import world_draw
from paths.ui import (
    render_bottom_info,
    render_path_ui,
    render_building_ui,
    render_action_ui,
    render_global_ui,
    render_visualise_ui,
)

st.set_page_config(layout="wide")
t0 = time.perf_counter()

if "grid_w" not in st.session_state or "grid_h" not in st.session_state:
    st.session_state.grid_w = 320
    st.session_state.grid_h = 160

if "manager" not in st.session_state:
    st.session_state.manager = WorldManager.new_default(
        width=st.session_state.grid_w, height=st.session_state.grid_h
    )
if "running" not in st.session_state:
    st.session_state.running = False
if "steps" not in st.session_state:
    st.session_state.steps = 1
if "timings" not in st.session_state:
    st.session_state.timings = {"update": [], "draw": [], "ui": []}

manager = st.session_state.manager

with st.sidebar:
    # control ui
    col1, col2 = st.columns(2)
    with col1:
        st.toggle("Running", key="running")
    with col2:
        if st.button("Reset", use_container_width=True):
            st.session_state.manager = WorldManager.new_default(
                width=st.session_state.grid_w, height=st.session_state.grid_h
            )
            manager = st.session_state.manager

    render_global_ui()

    render_visualise_ui()

    render_path_ui(manager)

    render_building_ui(manager)

    render_action_ui(manager)

# plot main figute
t_ui = time.perf_counter() - t0
fig = world_draw(manager, view=st.session_state.view)

st.pyplot(fig, width="stretch")
plt.close(fig)
t_draw = time.perf_counter() - (t0 + t_ui)

render_bottom_info(manager)

# call simulation, run
if st.session_state.running:
    t1 = time.perf_counter()
    for _ in range(st.session_state.steps):
        manager.update()
    t_update = time.perf_counter() - t1

    history = st.session_state.timings
    history["update"].append(t_update * 1000)
    history["draw"].append(t_draw * 1000)
    history["ui"].append(t_ui * 1000)

    with st.sidebar:
        for label, values in history.items():
            arr = np.array(values)
            recent = arr[-200:] if len(arr) >= 200 else arr
            st.caption(
                f"{label} — overall: {arr.mean():.1f}ms  recent: {recent.mean():.1f}ms"
            )

    time.sleep(0.01)
    st.rerun()
