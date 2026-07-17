# Ideas to implement

## Large / future
- FlowField graph construction performance — the per-direction loop creates ~24 temporary numpy arrays (~47KB each) every update, which fragments glibc's heap over time. Fix: pre-compute `rows` and `cols` (constant — depend only on grid geometry, not costs) once in `__init__`; pre-allocate a single `data` buffer filled in-place each update via `np.take(..., out=slice)`. Eliminates all `np.concatenate` calls and per-direction allocations. Pair with batched dijkstra (`csg.dijkstra(graph, indices=all_srcs)`) and pre-allocated bool buffers for diagonal validity checks for a fully allocation-free hot path.
- Directly interactable plot
- Terrain regions — per-tile base cost layer (forest, marsh, open ground etc.), stacks with wear model, shapes where desire paths naturally form
- Procedural height generation — multiple hills, Perlin noise terrain; currently a single hardcoded Gaussian hill. Asymmetric or more interesting hills would be good too.
- Simulation on a separate process from Streamlit
- Parallelise per-building dijkstra recompute across a thread pool — scipy's dijkstra is Cython and releases the GIL during computation, so genuine multi-core speedup is possible since each call only reads the shared graph and writes its own `_fields` entry. Agent movement (`move_agents()`) is plain Python/GIL-bound and wouldn't benefit from this; would need vectorising separately if it ever becomes the bottleneck.

## Medium
- Expand visualisations to include most/all cost calculations. Add support on the plot for showing elevation_grad_mag, raw uses, then different views from an agent's perspective - slope cost, noise cost, uses cost. FlowField ele cost will need to have a direction toggle too? Maybe, overall, some buttons you can select multiple of, then it adds up and displays that combination of costs? 
- Smooth per-agent noise field — replace standard normal per-tile noise with low-frequency smooth noise (Perlin or gaussian-blurred), so agents have broad regional preferences rather than step-level jitter. Required for true route diversity; pairs well with terrain generation.
- Door accessibility — later-spawned buildings can block doors that were valid at spawn time. Needs a connectivity check or post-spawn validation.
- Saved test scenarios - to show interesting behaviours. Potentially with parameters set to vary over time?
- Buildings spawn near, but not on, roads
- Busy building type — high attractiveness, higher spawn rate
- Congestion penalty — add a small cost *increase* on currently used, or just very heavily used, tiles so agents seek alternatives. Caps road dominance but may just widen busy roads rather than divert to new ones
- Richer destination selection — weight targets by `attractiveness / distance ** beta` so agents prefer nearby buildings. Per-agent beta (drawn on spawn) gives a mix of local and long-range walkers. Seed buildings as natural hubs (high attractiveness) would drive inter-town paths
- Batch the periodic FlowField dijkstra calls — `csg.dijkstra` accepts `indices` as a list, so all building entry sources can be computed in one call instead of one Python/scipy round trip per building. Likely the dominant cost of `WorldManager.update()` at scale (~1700ms observed at 105 buildings, 511 agents, 430x250 grid).
- Reconsider `adventurousness` mechanic/naming — it flattens the cost landscape (suppresses the field-distance term's influence as `combined_adv` rises above 1), so high-adventurousness agents don't take a deliberate alternate route, they become dominated by noise/momentum/recent-position penalty instead. Behaviourally closer to aimless drift than boldness/directness — worth a rename or redesign to match the intent.
- Write docstrings and comments throughout.

## Small
- Use 2D prefix sums (cumsum) for building spawn probability weighting — score candidate top-left positions by sum of prob over full footprint rather than just the top-left tile.
- Nicely colours and colourpalette handling
- Keep agents when switching pathfinding backends.
- Add Spawn building buttons for spawning 'rural' or 'town' buildings


## Done
- Sliders for parameters
- Make pathfinding prefer diagonal
- Consider cost reducing function (linear, sqrt, log with scale slider)
- Distributions for where buildings can spawn — near other buildings, not directly adjacent
- Make plot larger / smaller with webpage
- Buildings own their agent spawn timers (removed world-level agent timer)
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
- Per-agent noise field (standard normal, fixed at spawn) scaled by global temperature and per-agent adventurousness (currently 1.0). Both backends use it.
- `recent_positions` deque per agent with fixed penalty for revisiting recent tiles, preventing agents cycling indefinitely in noise-created local minima.
- Multi-tile buildings with random dimensions, perimeter door tile, centroid-based spawn distribution. Agents target door; unreachable targets die cleanly.
- `adventurousness` sampled per agent from uniform 0–2 on spawn, giving a population mix of cautious and exploratory walkers.
- Per-agent `temperature` split from `adventurousness` — temperature scales noise/wiggle, adventurousness flattens the cost landscape (lerp toward uniform) so high-adventurousness agents are more influenced by noise and momentum. This is almost more of an aimlessness and should be reconsidered.
- Replace cost function dropdown with a single `alpha` slider — `uses ** alpha` where alpha=1 is linear, 0.5 is sqrt. Higher alpha (>1) concentrates paths and lower alpha spreads them.
- Door facing and directed entry — door tile has a facing direction, agents can only enter from the entry tile (one directed edge in the pathfinding graph from entry→door). Requires directed edge support in FlowField graph construction and careful handling in A*.
- Height map — elevation with slope-based asymmetric movement cost (uphill expensive, downhill cheap). Visualisation toggle in sidebar. Single Gaussian hill currently hardcoded.
