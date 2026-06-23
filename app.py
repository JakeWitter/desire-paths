import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt

from paths.world_manager import WorldManager
from paths.visualise import world_draw
from paths.cost_function import COST_Fs

st.set_page_config(layout="wide")

GRID_WIDTH = 100
GRID_HEIGHT = 55

if "manager" not in st.session_state:
    st.session_state.manager = WorldManager(
        width=GRID_WIDTH, height=GRID_HEIGHT, default_cost=20
    )

if "running" not in st.session_state:
    st.session_state.running = False

if "timings" not in st.session_state:
    st.session_state.timings = {"update": [], "draw": []}


with st.sidebar:
    if st.button("Start/Stop"):
        st.session_state.running = not st.session_state.running
    if st.button("Reset"):
        st.session_state.manager = WorldManager(width=GRID_WIDTH, height=GRID_HEIGHT)
    scale = st.slider("Cost reduction amount scale", 0.0, 8.0, 1.0)
    default_cost = st.slider("Default cost for unused tile", 0.0, 100.0, 20.0)
    cost_f_selected = st.selectbox("Cost reduction function", list(COST_Fs.keys()))
    recalculate_agents_paths_every = st.slider("Recalculate agents paths", 1, 30, 5)
    steps_per_frame = st.slider("Steps per frame", 1, 50, 1)
    # TODO below
    # slider for uses decay rate
    # slider for building rate factor
    show_spawn_prob = st.checkbox("Show spawn probability")


st.session_state.manager.world.use_to_cost_f = COST_Fs[cost_f_selected]
st.session_state.manager.world.cost_scale = scale
st.session_state.manager.world.default_cost = default_cost
st.session_state.manager.recalculate_agents_paths_every = recalculate_agents_paths_every

t0 = time.perf_counter()
fig = world_draw(st.session_state.manager, show_spawn_prob=show_spawn_prob)
t_draw = time.perf_counter() - t0

st.pyplot(fig, width="stretch")
plt.close(fig)

manager = st.session_state.manager
st.caption(
    f"Time: {manager.time} | Buildings: {len(manager.buildings)} | Agents: {len(manager.agents)} | Next building: {int(manager.next_building_time - manager.time)} steps"
)

if st.session_state.running:
    t0 = time.perf_counter()
    for _ in range(steps_per_frame):
        st.session_state.manager.update()
    t_update = time.perf_counter() - t0

    history = st.session_state.timings
    history["update"].append(t_update * 1000)
    history["draw"].append(t_draw * 1000)

    with st.sidebar:
        for label, values in history.items():
            arr = np.array(values)
            recent = arr[-200:] if len(arr) >= 200 else arr
            st.caption(
                f"{label} — overall: {arr.mean():.1f}ms  recent: {recent.mean():.1f}ms"
            )

    time.sleep(0.02)
    st.rerun()
