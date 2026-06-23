# Ideas to implement

## Large / future
- Directly interactable plot
- Geographic features - height + slope, difficult terrain
- Simulation on a separate process from Streamlit (full decoupling)

## Medium
- Other pathfinding algs
- Saved test scenarios - to show path cheapness effect
- 'Adventurousness' for agents - disregarding path cheapening
- Related to above, some noise or something to smooth tight corners?
- Buildings spawn near, but not on, roads (extend distributions.py)
- Building attractiveness — weighted destination selection
- Busy building type — high attractiveness, higher spawn rate

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
