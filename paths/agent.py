# paths/agent.py

from .grid_world import GridWorld
from .building import Building
from .pathfinder import PathfinderBackend

from random import randint
import numpy as np
from collections import deque


class Agent:
    def __init__(
        self,
        world: GridWorld,
        pathfinder: PathfinderBackend,
        x=0,
        y=0,
        target_x=5,
        target_y=5,
        alpha=1,
    ):
        self.world = world
        self.x, self.y = x, y
        self.target_x, self.target_y = target_x, target_y
        self.alive = True
        self.id = randint(0, 100000)
        self.pathfinder = pathfinder
        self.vx = 0
        self.vy = 0
        self.alpha = alpha
        self.noise_field = np.random.default_rng().standard_normal(
            (world.height, world.width)
        )
        self.adventurousness = 1.0
        self.recent_positions = deque(maxlen=5)

    @classmethod
    def from_buildings(
        cls,
        world: GridWorld,
        pathfinder: PathfinderBackend,
        b1: Building,
        b2: Building,
    ):
        """Alternative constructor that creates an Agent starting from b1 and targeting b2."""
        agent = cls(
            world,
            pathfinder,
            x=b1.x,
            y=b1.y,
            target_x=b2.x,
            target_y=b2.y,
        )
        return agent

    def move(self):
        result = self.pathfinder.next_step(self)
        if result is None:
            self.alive = False
        else:
            dx = result[0] - self.x
            dy = result[1] - self.y
            self.vx = self.alpha * dx + (1 - self.alpha) * self.vx
            self.vy = self.alpha * dy + (1 - self.alpha) * self.vy
            self.x, self.y = result
            self.recent_positions.append((self.x, self.y))
