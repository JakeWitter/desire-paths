import numpy as np
from random import sample
from itertools import product

from .grid_world import GridWorld
from .agent import Agent
from .building import Building, BuildingType, POSS_BUILD_WIDTH, POSS_BUILD_HEIGHT
from .distributions import building_spawn_field, seed_positions
from .pathfinder import PathfinderBackend, PathfinderType
from .pathfinder_astar import AStarBackend
from .pathfinder_flowfield import FlowFieldBackend


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
        self.pathfinder: PathfinderBackend = FlowFieldBackend(
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
        self.path_degrade_every = path_degrade_every
        self.path_degrade_amt = path_degrade_amt
        self.min_agents = min_agents
        self.building_rate_decay = building_rate_decay
        self.agent_factor = agent_factor
        self.attractiveness_scale = attractiveness_scale

        # spawn seed buildings then assign them initial agents
        self.seed_positions = seed_positions(n_seeds, width, height)
        for x, y in self.seed_positions:
            self.spawn_building(x, y, BuildingType.SEED)

        for b in self.buildings:
            self.spawn_agent(source=b)

    @classmethod
    def new_default(cls, width, height):
        return cls(width=width, height=height)

    def set_pathfinder_type(self, pf_type: PathfinderType):
        if self.pathfinder.pathfinder_type == pf_type:
            return
        print(f"time={self.time} RECONSTRUCTING PATHFINDER")
        print(f"was {self.pathfinder} new={pf_type}")
        print(f"buildings = {len(self.buildings)}")
        if pf_type == PathfinderType.ASTAR:
            self.pathfinder = AStarBackend(
                self.world,
                self.pathfinder.diagonal,
                self.pathfinder.recalculate_every,
            )
            self.agents = []
        if pf_type == PathfinderType.FLOW:
            self.pathfinder = FlowFieldBackend(
                self.world,
                1.0,
                self.pathfinder.diagonal,
                self.pathfinder.recalculate_every,
            )
            self.agents = []
            self.pathfinder.update(self.world.costs, self.buildings, self.buildings)

    def _find_building_tiles(self, occupied, x=None, y=None):
        """Finds a valid width, height, x, y and tileset for a new building.
        Looks at globals from `building.py` for ranges. Uses `occupied` to
        determine invalid tiles.

        Args:
            occupied (set): Tiles that are considered invalid/full
            x (int | None): x_coord for new building, None means generate
            y (int | None): y_coord for new building, None means generate

        Returns:
            tuple: (
            bw (int): width of new building
            bh (int): height of new building
            x (int): x_coord for new building
            y (int): y_coord for new building
            tiles (list): list of tiles new building occupies
            )
        """
        random_placement = x is None or y is None
        if random_placement:
            field = building_spawn_field(
                self.world.width, self.world.height, self.buildings
            )
        while True:
            bw, bh = (
                np.random.randint(*POSS_BUILD_WIDTH),
                np.random.randint(*POSS_BUILD_HEIGHT),
            )
            if random_placement:
                flat = field.flatten()
                flat /= flat.sum()
                idx = np.random.choice(len(flat), p=flat)
                y, x = divmod(idx, self.world.width)
            if x + bw > self.world.width or y + bh > self.world.height:
                continue
            tiles = [(x + w, y + h) for w, h in product(range(bw), range(bh))]
            if not occupied.intersection(tiles):
                break
        # TODO combine in to size (bw, bh), and pos (x,y)
        return bw, bh, x, y, tiles

    def spawn_building(
        self,
        x: int | None = None,
        y: int | None = None,
        building_type: BuildingType = BuildingType.NORM,
    ):
        """Spawns a building at the location provided, sampling size, or samples for
        size and location if `x`, `y` are None. This is added to buildings, world and
        the pathfinder as appropriate

        Args:
            x (int | None): The x coord for this building. Sampled using
                _find_building_tiles if not passed.
            y (int | None): The y coord for this building. Sampled using
                _find_building_tiles if not passed.
            building_type (BuildingType): Enum denoting the type of this building.
        """
        occupied = {tile for b in self.buildings for tile in b.tiles}
        occupied.update(b.entry for b in self.buildings)
        bw, bh, x, y, tiles = self._find_building_tiles(occupied, x, y)
        door, entry = Building.find_door(
            tiles, occupied, (x, y), (bw, bh), (self.world.width, self.world.height)
        )
        if door is None or entry is None:
            return

        attractiveness = 1.0 + np.random.exponential(1.0)
        building = Building(
            bw,
            bh,
            tiles,
            door,
            entry,
            self.building_agent_rate,
            self.time,
            self.agent_factor,
            attractiveness=attractiveness,
            building_type=building_type,
        )
        self.buildings.append(building)
        self.world.update_costs(self.buildings)
        self.pathfinder.update(self.world.costs, self.buildings, [building])
        if building_type == BuildingType.NORM:
            self.spawn_agent(source=building)

    def spawn_agent(self, source=None, target=None):
        """Creates an agent, sampling source and target if not passed.

        Args:
            source (Building): Start building for the agent's path
            target (Building): Destination building for the agent's path
        """
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
        """Orchestrates one whole update. Advances time, spawns buildings, spawns agents,
        moves agents, updates costs and pathfinding"""
        self.time += 1
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

        self.move_agents()
        if (self.time % self.path_degrade_every) == 0:
            self.world.uses = self.world.uses * self.path_degrade_amt
            self.world.uses = np.clip(self.world.uses, 0, None)
        self.world.update_costs(self.buildings)

        if self.time % self.pathfinder.recalculate_every == 0:
            self.pathfinder.update(self.world.costs, self.buildings, self.buildings)

    def move_agents(self):
        """Updates the current world uses, then tells each agent to move along
        its path. Then culls."""
        for a in self.agents:
            self.world.uses[a.y, a.x] += 1
            try:
                a.move()
            except KeyError:
                print(f"time={self.time} a={a.id} target={a.target} pos=({a.x}, {a.y})")
                print(
                    f"agent.pathfinder is self.pathfinder: {a.pathfinder is self.pathfinder}"
                )
                print(
                    f"building with this entry: {[b for b in self.buildings if b.entry == a.target]}"
                )
                print(f"building total {len(self.buildings)}")
                print(
                    f"fields keys ({len(self.pathfinder._fields)}): {sorted(self.pathfinder._fields.keys())}"
                )
                raise
        self.cull_agents()

    def cull_agents(self):
        """Removes agents that are marked as not alive."""
        for agent in self.agents:
            if not agent.alive:
                self.pathfinder.remove_agent(agent.id)
        self.agents = [agent for agent in self.agents if agent.alive]
