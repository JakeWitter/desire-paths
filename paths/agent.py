# paths/agent.py

from .grid_world import GridWorld
from .building import Building
from .pathfinder import PathfinderBackend

from random import randint


class Agent:
    def __init__(
        self,
        world: GridWorld,
        pathfinder: PathfinderBackend,
        x=0,
        y=0,
        target_x=5,
        target_y=5,
    ):
        self.world = world
        self.x, self.y = x, y
        self.target_x, self.target_y = target_x, target_y
        self.alive = True
        self.id = randint(0, 100000)
        self.pathfinder = pathfinder

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
        result = self.pathfinder.next_step(
            self.id, self.x, self.y, self.target_x, self.target_y
        )
        if result is None:
            self.alive = False
        else:
            self.x, self.y = result
