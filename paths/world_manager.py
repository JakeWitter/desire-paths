import numpy as np
from random import sample, choice
from itertools import product

from .grid_world import GridWorld
from .agent import Agent
from .building import Building
from .distributions import building_spawn_prob, seed_positions
from .pathfinder import PathfinderBackend, AStarBackend, CARDINAL


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
        agent_factor=1.0,
        attractiveness_scale=1.0,
        n_seeds=3,
    ):
        self.world: GridWorld = GridWorld(width, height, default_cost=default_cost)
        self.pathfinder: PathfinderBackend = AStarBackend(
            self.world,
            diagonal=diagonal,
            recalculate_every=recalculate_agents_paths_every,
        )
        self.agents = []
        self.buildings = []
        self.building_rate = building_rate
        self.building_agent_rate = building_agent_rate
        self.time = 0
        self.next_building_time = np.random.exponential(1 / self.building_rate)
        # self.next_agent_time = np.random.exponential(1 / self.agent_rate)
        self.path_degrade_every = path_degrade_every
        self.path_degrade_amt = path_degrade_amt
        self.min_agents = min_agents
        self.building_rate_decay = building_rate_decay
        self.agent_factor = agent_factor
        self.attractiveness_scale = attractiveness_scale

        self.seed_positions = seed_positions(n_seeds, width, height)
        for x, y in self.seed_positions:
            self.spawn_building(x, y, "seed")

        for b in self.buildings:
            self.spawn_agent(source=b)

    def spawn_building(
        self, x: int | None = None, y: int | None = None, building_type: str = "norm"
    ):
        occupied = {tile for b in self.buildings for tile in b.tiles}
        if x is None or y is None:
            prob = building_spawn_prob(
                self.world.width, self.world.height, self.buildings
            )
            ys, xs = np.mgrid[0 : self.world.height, 0 : self.world.width]
            while True:
                bw, bh = (np.random.randint(1, 5), np.random.randint(1, 5))
                valid_mask = (xs <= self.world.width - bw) & (
                    ys <= self.world.height - bh
                )
                valid = prob * valid_mask
                flat = valid.flatten()
                flat /= flat.sum()
                idx = np.random.choice(len(flat), p=flat)
                y, x = divmod(idx, self.world.width)
                tiles = [(x + w, y + h) for w, h in product(range(bw), range(bh))]
                if not occupied.intersection(tiles):
                    break
        else:
            while True:
                bw, bh = (np.random.randint(1, 5), np.random.randint(1, 5))
                tiles = [(x + w, y + h) for w, h in product(range(bw), range(bh))]
                if not occupied.intersection(tiles):
                    break
        door_candidates = [
            tile
            for tile in tiles
            if ((tile[0] in [x, x + bw - 1]) or tile[1] in [y, y + bh - 1])
            and any(
                (tile[0] + dx, tile[1] + dy) not in occupied
                and (tile[0] + dx, tile[1] + dy) not in tiles
                and 0 <= tile[0] + dx < self.world.width
                and 0 <= tile[1] + dy < self.world.height
                for dx, dy in CARDINAL
            )
        ]
        if not door_candidates:
            return
        door_x, door_y = choice(door_candidates)
        attractiveness = 1.0 + np.random.exponential(1.0)
        building = Building(
            bw,
            bh,
            tiles,
            door_x,
            door_y,
            self.building_agent_rate,
            self.time,
            self.agent_factor,
            attractiveness=attractiveness,
            building_type=building_type,
        )
        self.buildings.append(building)
        self.world.update_costs(self.buildings)
        self.pathfinder.update(self.world.costs, self.buildings)
        if building_type == "norm":
            self.spawn_agent(source=building)

    def spawn_agent(self, source=None, target=None):
        if len(self.buildings) >= 2:
            if source is None:
                source = sample(self.buildings, 1)[0]
            if target is None:
                targets = [b for b in self.buildings if b != source]
                weights = np.array([b.attractiveness for b in targets], dtype=float)
                weights = weights**self.attractiveness_scale
                weights /= weights.sum()
                target = targets[np.random.choice(len(targets), p=weights)]
            agent = Agent.from_buildings(
                self.world,
                self.pathfinder,
                source,
                target,
            )
            self.agents.append(agent)

    def update(self):
        """Updates the simulation by advancing time, spawning buildings and agents, and moving agents."""
        self.time += 1

        # Spawn buildings and agents if it's time
        if self.time >= self.next_building_time:
            self.spawn_building()
            self.next_building_time += np.random.exponential(1 / self.building_rate) * (
                1 + len(self.buildings) * self.building_rate_decay
            )

        while self.min_agents > len(self.agents):
            self.spawn_agent()

        for b in self.buildings:
            if b.next_agent_time <= self.time:
                self.spawn_agent(b)
                b.set_next_agent_spawn(self.time, self.agent_factor)
        # Move agents and update world usage
        self.move_agents()
        if (self.time % self.path_degrade_every) == 0:
            self.world.uses = self.world.uses * self.path_degrade_amt
            self.world.uses = np.clip(self.world.uses, 0, None)
        self.world.update_costs(self.buildings)  # Update tile costs
        self.pathfinder.update(self.world.costs, self.buildings)

    def move_agents(self):
        """Moves each agent along their path and updates the world usage."""
        for agent in self.agents:
            self.world.uses[agent.y, agent.x] += 1
            agent.move()
        self.cull_agents()

    def cull_agents(self):
        """Removes agents that have reached their destination."""
        for agent in self.agents:
            if not agent.alive:
                self.pathfinder.remove_agent(agent.id)
        self.agents = [agent for agent in self.agents if agent.alive]
