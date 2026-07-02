# Ideas to implement

## Large / future
- Directly interactable plot
- Geographic features - height + slope, difficult terrain
- Simulation on a separate process from Streamlit

## Medium
- More interesting buildings — buildings larger than a single tile. Buildings may touch as long as they don't violate placement rules:
  - Whole footprint is impassable - can't overlap another building
  - Entrances — each building gets one door tile on its perimeter. This placement requires the new door has at least one free cardinal neighbour, and the new footprint doesn't cover any existing door. Later buildings can collectively enclose a door into an unreachable courtyard — a full connectivity check would catch this?
- Saved test scenarios - to show interesting behaviours. Potentially with parameters set to vary over time?
- Buildings spawn near, but not on, roads
- Busy building type — high attractiveness, higher spawn rate
- Per-agent temperature. Each agent could have its own temperature value (e.g. drawn from a distribution on spawn), so the population has a mix of cautious/predictable and exploratory/random walkers
- Congestion penalty — add a small cost *increase* on currently used, or just very heavily used, tiles so agents seek alternatives. Caps road dominance but may just widen busy roads rather than divert to new ones
- Richer destination selection — weight targets by `attractiveness / distance ** beta` so agents prefer nearby buildings. Per-agent beta (drawn on spawn) gives a mix of local and long-range walkers. Seed buildings as natural hubs (high attractiveness) would drive inter-town paths

## Small
- Nicely colours and colourpalette handling
- Replace cost function dropdown with a single `alpha` slider — `uses ** alpha` where alpha=1 is linear, 0.5 is sqrt. Higher alpha (>1) concentrates paths and lower alpha spreads them.

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
