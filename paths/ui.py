import streamlit as st
import numpy as np

from .pathfinder import PathfinderType


def render_global_ui():
    st.slider("Steps per frame", 1, 200, key="steps")
    st.slider("Grid width", 2, 1000, key="grid_w")
    st.slider("Grid height", 2, 800, key="grid_h")


def render_visualise_ui():
    with st.expander("Visualisation"):
        st.radio(
            "View",
            ["Costs", "Uses", "Elev.", "Elev. grad", "Bld. prob", "Cost breakdown"],
            key="view",
        )
        if st.session_state.view == "Cost breakdown":
            tgl_slope = st.toggle("Slope")
            tgl_costs = st.toggle("Use costs")
            tgl_noise = st.toggle("Agent noise")
            st.session_state.breakdown["slope"] = tgl_slope
            st.session_state.breakdown["use costs"] = tgl_costs
            st.session_state.breakdown["agent noise"] = tgl_noise
        else:
            st.session_state.breakdown = {"use costs": True}


def render_path_ui(manager):
    pf = manager.pathfinder
    with st.expander("Path & cost"):
        # general variables
        pf.temperature = st.slider("Temperature", 0.0, 5.0, 1.0)
        pf.slope_scale = st.slider("Slope scale", 0.0, 200.0, 35.0)
        pf.adventurousness = st.slider("Adventurousness scale", 0.0, 10.0, 0.0001)

        # backend specific
        with st.expander("Pathfinding"):
            chosen_type = st.selectbox(
                "Pathfinding backend",
                list(PathfinderType),
                format_func=lambda t: t.value,
                index=list(PathfinderType).index(pf.pathfinder_type),
            )
            manager.set_pathfinder_type(chosen_type)
            pf = manager.pathfinder

            if pf.pathfinder_type == PathfinderType.FLOW:
                pf.recalculate_every = st.slider(
                    "Recalculate fields every:", 1, 50, pf.recalculate_every
                )
                pf.momentum_weight = st.slider("Momentum weight", 0.0, 5.0, 0.0)
            elif pf.pathfinder_type == PathfinderType.ASTAR:
                pf.recalculate_every = st.slider(
                    "Recalculate paths every:", 1, 50, pf.recalculate_every
                )

        # final few
        manager.world.cost_scale = st.slider("Cost reduction scale", 0.0, 6.0, 0.8)
        manager.world.default_cost = st.slider("Default tile cost", 0.0, 30.0, 6.0)
        manager.world.alpha = st.slider("Cost function alpha", 0.1, 2.0, 1.0)

        manager.path_degrade_amt = st.selectbox(
            "Path degrade factor",
            [0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0],
            index=5,
        )


def render_building_ui(manager):
    with st.expander("Buildings"):
        manager.building_rate = st.slider("Building rate", 0.0, 0.5, 0.05)
        manager.building_rate_decay = st.slider("Building rate decay", 0.0, 1.0, 0.1)
        manager.attractiveness_scale = st.slider("Attractiveness scale", 0.0, 5.0, 1.0)

        agent_factor = st.slider("Agent rate", 0.0, 10.0, 1.0)
        if agent_factor != manager.agent_factor:
            manager.agent_factor = agent_factor
            for building in manager.buildings:
                building.set_next_agent_spawn(manager.time, manager.agent_factor)


def render_action_ui(manager):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Spawn building", use_container_width=True):
            manager.spawn_building()
        if st.button("Delete all agents", use_container_width=True):
            manager.agents = []
    with col2:
        if st.button("Spawn agent", use_container_width=True):
            manager.spawn_agent()
        if st.button("Reset uses", use_container_width=True):
            manager.world.uses = np.full((manager.world.height, manager.world.width), 0)
            manager.world.update_costs(manager.buildings)
            manager.pathfinder.update(
                manager.world.costs, manager.buildings, manager.buildings
            )


def render_bottom_info(manager):
    st.caption(
        f"Time: {manager.time} | Buildings: {len(manager.buildings)} | Agents: {len(manager.agents)} | Next building: {int(manager.next_building_time - manager.time)} steps"
    )
    st.caption(f"Seed positions: {manager.seed_positions}")
    oldest = sorted(manager.agents, key=lambda a: a.age)[-5:]
    st.caption(
        f"Oldest agents: {[(a.age, round(a.adventurousness, 3)) for a in oldest]}"
    )
