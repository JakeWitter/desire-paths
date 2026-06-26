import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt

from paths.world_manager import WorldManager
from paths.visualise import world_draw
from paths.cost_function import COST_Fs
from paths.pathfinder import AStarBackend, FieldFlowBackend

st.set_page_config(layout="wide")

GRID_WIDTH = 150
GRID_HEIGHT = 95
PATHFINDING_OPTIONS = ["AStar", "FlowField"]

if "manager" not in st.session_state:
    st.session_state.manager = WorldManager(
        width=GRID_WIDTH, height=GRID_HEIGHT, default_cost=20
    )

if "running" not in st.session_state:
    st.session_state.running = False

if "pathfinding_type" not in st.session_state:
    st.session_state.pathfinding_type = "AStar"

if "timings" not in st.session_state:
    st.session_state.timings = {"update": [], "draw": []}


manager = st.session_state.manager

with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start/Stop", use_container_width=True):
            st.session_state.running = not st.session_state.running
    with col2:
        if st.button("Reset", use_container_width=True):
            st.session_state.manager = WorldManager(
                width=GRID_WIDTH, height=GRID_HEIGHT
            )
    steps_per_frame = st.slider("Steps per frame", 1, 200, 1)
    show_spawn_prob = st.checkbox("Show spawn probability")

    # st.divider()

    with st.expander("Path & cost"):
        with st.expander("Pathfinding"):
            current_type = (
                "AStar" if isinstance(manager.pathfinder, AStarBackend) else "FlowField"
            )
            pathfinding_type = st.selectbox(
                "Pathfinding backend",
                PATHFINDING_OPTIONS,
                index=PATHFINDING_OPTIONS.index(current_type),
            )
            if pathfinding_type == "AStar" and not isinstance(
                manager.pathfinder, AStarBackend
            ):
                manager.pathfinder = AStarBackend(
                    manager.world, diagonal=True, recalculate_every=5
                )
                manager.agents = []
            elif pathfinding_type == "FlowField" and not isinstance(
                manager.pathfinder, FieldFlowBackend
            ):
                manager.pathfinder = FieldFlowBackend(manager.world, temperature=1.0)
                manager.agents = []
            manager.pathfinder.streamlit_controls()
        scale = st.slider("Cost reduction scale", 0.0, 8.0, 1.0)
        default_cost = st.slider("Default tile cost", 0.0, 100.0, 20.0)
        cost_f_selected = st.selectbox("Cost function", list(COST_Fs.keys()))
        path_degrade_factor = st.selectbox(
            "Path degrade factor",
            [0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0],
            index=3,
        )

    with st.expander("Buildings"):
        building_rate = st.slider("Building rate", 0.0, 0.5, 0.05)
        building_rate_decay = st.slider("Building rate decay", 0.0, 1.0, 0.1)
        attractiveness_scale = st.slider("Attractiveness scale", 0.0, 5.0, 1.0)

    agent_factor = st.slider("Agent rate", 0.0, 10.0, 1.0)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Spawn building", use_container_width=True):
            manager.spawn_building()
    with col2:
        if st.button("Spawn agent", use_container_width=True):
            manager.spawn_agent()
    if st.button("Delete all agents", use_container_width=True):
        manager.agents = []

manager.world.use_to_cost_f = COST_Fs[cost_f_selected]
manager.world.cost_scale = scale
manager.world.default_cost = default_cost
manager.path_degrade_amt = path_degrade_factor
manager.building_rate = building_rate
manager.building_rate_decay = building_rate_decay
manager.attractiveness_scale = attractiveness_scale
if agent_factor != manager.agent_factor:
    manager.agent_factor = agent_factor
    for building in manager.buildings:
        building.set_next_agent_spawn(manager.time, manager.agent_factor)

t0 = time.perf_counter()
fig = world_draw(manager, show_spawn_prob=show_spawn_prob)
t_draw = time.perf_counter() - t0

st.pyplot(fig, width="stretch")
plt.close(fig)

st.caption(
    f"Time: {manager.time} | Buildings: {len(manager.buildings)} | Agents: {len(manager.agents)} | Next building: {int(manager.next_building_time - manager.time)} steps"
)
st.caption(f"Seed positions: {manager.seed_positions}")

if st.session_state.running:
    t0 = time.perf_counter()
    for _ in range(steps_per_frame):
        manager.update()
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
