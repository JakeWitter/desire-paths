# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
uv run streamlit run app.py
```

## Architecture

This is a desire-path simulation: agents pathfind between buildings on a grid, and repeated traversal wears down tile costs, causing emergent "desire paths" to form over time.

### Key data flow

1. `GridWorld` owns the numpy cost array and the pathfinding `Grid` object. It updates tile costs from `uses` (a wear array) each step.
2. `WorldManager` orchestrates everything: owns `GridWorld`, a list of `Building`s, a list of `Agent`s, and a `PathfinderBackend`. Calls `pathfinder.update()` after each cost update.
3. `Agent` is stateless with respect to pathfinding — it just holds position and target, and calls `pathfinder.next_step()` each move.
4. `PathfinderBackend` (ABC in `pathfinder.py`) has two implementations:
   - `AStarBackend`: wraps `python-pathfinding`, caches paths per agent id, recalculates every N steps
   - `FieldFlowBackend`: uses `scipy.sparse.csgraph.dijkstra` to compute distance fields per building, agents step toward minimum distance neighbour (with optional softmax temperature sampling)
5. `app.py` is the Streamlit entry point. All state lives in `st.session_state`. Each backend implements `streamlit_controls()` to render its own sidebar section.

### Building spawn distribution

`distributions.py` computes a spatial probability map for where new buildings spawn — summing gaussian components (repulsion close, attraction at medium range, broad DoG for longer-range structure) around each existing building.

### Cost model

`uses` array accumulates agent visits. Each step: `costs = default_cost - cost_function(uses) * scale`, clipped to minimum 1. Buildings are set to cost 0 (unwalkable for non-building agents). `uses` decays multiplicatively each step (`path_degrade_amt`).

## Package structure

The simulation code lives in the `paths/` package with relative imports (`from .grid_world import GridWorld`). `pyrightconfig.json` configures Pyright to find the venv and treat the project root as a source root.
