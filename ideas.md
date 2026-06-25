# Ideas to implement

## Large / future
- Directly interactable plot
- Geographic features - height + slope, difficult terrain
- Simulation on a separate process from Streamlit (full decoupling)

## Medium
- Saved test scenarios - to show path cheapness effect
- Buildings spawn near, but not on, roads (extend distributions.py)
- Busy building type — high attractiveness, higher spawn rate
- Momentum / inertia for agents — agent tracks its own last-move direction (dx, dy), updated after each move. Pass as extra args to next_step. FlowField backend blends momentum vector with field gradient before softmax sampling — reduces oscillation (each step being an independent sample causes back-and-forth). A* can ignore it. Temperature alone does not fix oscillation, momentum does.
- FlowField: add diagonal edges (dx,dy in [(1,1),(1,-1),(-1,1),(-1,-1)], weight cost*sqrt(2)) to fix cityblock routing
- Path drawing for A* backend — add get_display_path(agent_id) method, use in visualiser
- Vectorise FlowField graph construction (currently slow double loop)

## Small
- Global colour palette module (colourmaps currently local to visualise.py)
- Use consistent palette throughout
- Buildings spawn up to a limit with decreasing chance

## Done
- Sliders for parameters
- Make pathfinding prefer diagonal
- Consider cost reducing function (linear, sqrt, log with scale slider)
- Distributions for where buildings can spawn — near other buildings, not directly adjacent
- Make plot larger / smaller with webpage
- Buildings own their agent spawn timers (removed world-level agent timer)
- Buildings spawn agents, not world timer
- Minimum agent count (always N agents active)
- Buildings colour by time-to-next-spawn (black → red)
- Agents recalculate paths periodically
- Steps per frame slider (speed vs smoothness tradeoff)
- Start with 2 buildings
- Building attractiveness — weighted destination selection (power scaling via attractiveness_scale)
- Swappable pathfinding backends (AStarBackend, FieldFlowBackend) with per-backend streamlit controls
