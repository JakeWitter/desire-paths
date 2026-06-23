import numpy as np
from random import sample

from .grid_world import GridWorld
from .agent import Agent
from .building import Building
from .distributions import building_spawn_prob


class WorldManager:
    def __init__(
        self,
        width,
        height,
        building_rate=0.05,
        building_agent_rate=0.02,
        diagonal=True,
        default_cost=10.0,
        path_degrade_every=1,
        path_degrade_amt=0.9925,
        recalculate_agents_paths_every=5,
        min_agents=2,
        building_rate_decay=0.1,
    ):
        self.world = GridWorld(width, height, default_cost=default_cost)
        self.agents = []
        self.buildings = []
        self.building_rate = building_rate
        self.building_agent_rate = building_agent_rate
        self.diagonal = diagonal
        self.time = 0
        self.next_building_time = np.random.exponential(1 / self.building_rate)
        # self.next_agent_time = np.random.exponential(1 / self.agent_rate)
        self.path_degrade_every = path_degrade_every
        self.path_degrade_amt = path_degrade_amt
        self.recalculate_agents_paths_every = recalculate_agents_paths_every
        self.min_agents = min_agents
        self.building_rate_decay = building_rate_decay
        self.spawn_building()
        self.spawn_building()

    def spawn_building(self):
        prob = building_spawn_prob(self.world.width, self.world.height, self.buildings)
        flat = prob.flatten()
        flat /= flat.sum()
        idx = np.random.choice(len(flat), p=flat)
        y, x = divmod(idx, self.world.width)
        building = Building(int(x), int(y), self.building_agent_rate, self.time)
        self.buildings.append(building)
        self.world.update_costs(self.buildings)
        self.spawn_agent(source=building)

    def spawn_agent(self, source=None, target=None):
        if len(self.buildings) >= 2:
            if source is None:
                source = sample(self.buildings, 1)[0]
            if target is None:
                targets = [b for b in self.buildings if b != source]
                target = sample(targets, 1)[0]
            agent = Agent.from_buildings(
                self.world,
                source,
                target,
                diagonal=self.diagonal,
                recalculate_every=self.recalculate_agents_paths_every,
            )
            self.agents.append(agent)

    def update(self):
        """Updates the simulation by advancing time, spawning buildings and agents, and moving agents."""
        self.time += 1

        # Spawn buildings and agents if it's time
        if self.time >= self.next_building_time:
            self.spawn_building()
            self.next_building_time += np.random.exponential(1 / self.building_rate) * (1 + len(self.buildings) * self.building_rate_decay)

        while self.min_agents > len(self.agents):
            self.spawn_agent()

        for b in self.buildings:
            if b.next_agent_time <= self.time:
                self.spawn_agent(b)
                b.set_next_agent_spawn(self.time)
        # Move agents and update world usage
        self.move_agents()
        if (self.time % self.path_degrade_every) == 0:
            self.world.uses = self.world.uses * self.path_degrade_amt
            self.world.uses = np.clip(self.world.uses, 0, None)
        self.world.update_costs(self.buildings)  # Update tile costs

    def move_agents(self):
        """Moves each agent along their path and updates the world usage."""
        for agent in self.agents:
            self.world.uses[agent.y, agent.x] += 1
            agent.move()
        self.cull_agents()

    def cull_agents(self):
        """Removes agents that have reached their destination."""
        self.agents = [agent for agent in self.agents if agent.alive]
