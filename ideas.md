# Ideas to implement

## Large / future
- Directly interactable plot
- Height map — per-tile elevation, slope-based asymmetric movement cost (uphill expensive, downhill cheap), steep slopes block building placement
- Terrain regions — per-tile base cost layer (forest, marsh, open ground etc.), stacks with wear model, shapes where desire paths naturally form
- Simulation on a separate process from Streamlit

## Medium
- Door accessibility — later-spawned buildings can block doors that were valid at spawn time. Needs a connectivity check or post-spawn validation.
- Saved test scenarios - to show interesting behaviours. Potentially with parameters set to vary over time?
- Buildings spawn near, but not on, roads
- Busy building type — high attractiveness, higher spawn rate
- Congestion penalty — add a small cost *increase* on currently used, or just very heavily used, tiles so agents seek alternatives. Caps road dominance but may just widen busy roads rather than divert to new ones
- Richer destination selection — weight targets by `attractiveness / distance ** beta` so agents prefer nearby buildings. Per-agent beta (drawn on spawn) gives a mix of local and long-range walkers. Seed buildings as natural hubs (high attractiveness) would drive inter-town paths

## Small
- Use 2D prefix sums (cumsum) for building spawn probability weighting — score candidate top-left positions by sum of prob over full footprint rather than just the top-left tile.
- Nicely colours and colourpalette handling
- Keep agents when switching pathfinding backends.

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
- Steps per frame slider
- Start with 2 buildings
- Building attractiveness — weighted destination selection (power scaling via attractiveness_scale)
- Swappable pathfinding backends (AStarBackend, FieldFlowBackend) with per-backend streamlit controls
- Buildings spawn up to a limit with decreasing chance. Buildings have a decay factor slowing subsequent buildings, the 'steepness' of this effect can be controlled.
- Momentum / inertia for agents — EMA velocity (vx, vy) per agent, passed to next_step. FlowField blends momentum into cost before softmax, reducing oscillation. A* ignores it.
- FlowField: diagonal edges (weight cost*sqrt(2)) to fix cityblock routing and enable genuine corner-cutting
- Vectorise FlowField graph construction (currently slow double loop)
- FlowField oscillations around 0 cost squares reduced by using 0.01 instead, and buildings are np.inf.
- Per-agent noise field (standard normal, fixed at spawn) scaled by global temperature and per-agent adventurousness (currently 1.0). Both A* and FlowField backends respect it, giving genuine route diversity rather than step-level zigzag.
- `recent_positions` deque per agent with fixed penalty for revisiting recent tiles, preventing agents cycling indefinitely in noise-created local minima.
- Multi-tile buildings with random dimensions, perimeter door tile, centroid-based spawn distribution. Agents target door; unreachable targets die cleanly.
- `adventurousness` sampled per agent from uniform 0–2 on spawn, giving a population mix of cautious and exploratory walkers.
- Smooth per-agent noise field — replace standard normal per-tile noise with low-frequency smooth noise (Perlin or gaussian-blurred), so agents have broad regional preferences rather than step-level jitter. Pairs well with terrain generation.
- Replace cost function dropdown with a single `alpha` slider — `uses ** alpha` where alpha=1 is linear, 0.5 is sqrt. Higher alpha (>1) concentrates paths and lower alpha spreads them.
- Door facing and directed entry — door tile has a facing direction, agents can only enter from the entry tile (one directed edge in the pathfinding graph from entry→door). Requires directed edge support in FlowField graph construction and careful handling in A*.
