# Ideas to implement

## Next
- Fix FlowField agent oscillation near non-target buildings: in `FieldFlowBackend.update()` (`pathfinder.py`), replace cost 0 with `np.inf` before building the graph so building tiles have infinite edge weight and are impassable. A* already handles this via `walkable=False` (`grid_world.py:35`); FlowField does not.

## Large / future
- Directly interactable plot
- Geographic features - height + slope, difficult terrain
- Simulation on a separate process from Streamlit (full decoupling)

## Medium
- Saved test scenarios - to show path cheapness effect
- Buildings spawn near, but not on, roads (extend distributions.py)
- Busy building type — high attractiveness, higher spawn rate
- Per-agent temperature — each agent has its own temperature value (e.g. drawn from a distribution on spawn), so the population has a mix of cautious/predictable and exploratory/random walkers rather than all agents sharing one global setting
- Path drawing for A* backend — add get_display_path(agent_id) method, use in visualiser
- Congestion penalty — add a small cost *increase* on very heavily used tiles so agents seek alternatives. Caps road dominance but may just widen busy roads rather than divert to new ones; probably needs combining with high alpha to work as intended
- Richer destination selection — weight targets by `attractiveness / distance ** beta` so agents prefer nearby buildings. Per-agent beta (drawn on spawn) gives a mix of local and long-range walkers. Seed buildings as natural hubs (high attractiveness) would drive inter-cluster arterial paths while local traffic fills in within clusters
- Cost-ignoring slider — proportion of agents that ignore path costs and walk more directly (or randomly). Creates persistent off-road wear that can seed new paths rather than just widening existing ones

## Small
- Global colour palette module (colourmaps currently local to visualise.py)
- Use consistent palette throughout
- Replace cost function dropdown with a single `alpha` slider — `uses ** alpha` where alpha=1 is linear, 0.5 is sqrt. Higher alpha (>1) concentrates paths (slow-start, only rewards heavy use); lower alpha spreads them (edge tiles cheapened quickly). No cap either way — combine with congestion penalty for that

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
- Buildings spawn up to a limit with decreasing chance. Buildings have a decay factor slowing subsequent buildings, the 'steepness' of this effect can be controlled.
- Momentum / inertia for agents — EMA velocity (vx, vy) per agent, passed to next_step. FlowField blends momentum into cost-space before softmax, reducing oscillation. A* ignores it.
- FlowField: diagonal edges (weight cost*sqrt(2)) to fix cityblock routing and enable genuine corner-cutting
- Vectorise FlowField graph construction (currently slow double loop)
