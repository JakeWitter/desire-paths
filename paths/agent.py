# paths/agent.py

from .grid_world import GridWorld
from .building import Building
from .pathfinder import PathfinderBackend

from random import uniform
import numpy as np
from collections import deque
from itertools import count


class Agent:
    _id_counter = count()

    def __init__(
        self,
        world: GridWorld,
        pathfinder: PathfinderBackend,
        x,
        y,
        target,
        src_entry,
        target_door,
        mom_alpha=1,
    ):
        self.world = world
        self.x, self.y = x, y
        self.target = target
        self.src_entry = src_entry
        self.target_door = target_door
        self.alive = True
        self.id = next(Agent._id_counter)
        self.pathfinder = pathfinder
        self.vx = 0
        self.vy = 0
        self.mom_alpha = mom_alpha
        self.noise_field = np.abs(
            np.random.default_rng().standard_normal((world.height, world.width))
        )
        self.adventurousness = uniform(0.5, 4)
        self.temperature = uniform(0.5, 1.5)
        self.recent_positions = deque(maxlen=100)
        self.age = 0
        self._has_exited = False

    @classmethod
    def from_buildings(
        cls,
        world: GridWorld,
        pathfinder: PathfinderBackend,
        b1: Building,
        b2: Building,
    ):
        """Alternative constructor that creates an Agent starting from b1 and
        targeting b2."""
        agent = cls(
            world,
            pathfinder,
            b1.door[0],
            b1.door[1],
            b2.entry,
            b1.entry,
            b2.door,
        )
        return agent

    def _move_to(self, pos: tuple):
        """Moves an agent to `pos`, and updates momentum.

        Args:
            pos: The position to move to.
        """
        nx, ny = pos
        dx, dy = nx - self.x, ny - self.y
        self.vx = self.mom_alpha * dx + (1 - self.mom_alpha) * self.vx
        self.vy = self.mom_alpha * dy + (1 - self.mom_alpha) * self.vy
        self.x, self.y = pos

    def _expire_if_stale(self):
        """Checks if an agent is old, and low temperature. This should identify agents
        that have been repeating themselves.

        Returns:
            bool: Notes if this agent has been found wanting.
        """
        if self.age > 400 and self.temperature <= 0.1:
            self.alive = False
            return True
        return False

    def _calm_if_repeating(self):
        """Reduces an agent's temperature and adventurousness, more so if it is
        repeating positions."""
        if (self.x, self.y) in self.recent_positions:
            self.temperature *= 0.98
            self.adventurousness *= 0.98
        else:
            self.temperature *= 0.999
            self.adventurousness *= 0.999

    def _arrive_if_reached(self):
        """Checks if an agent has reached its destination entry, and then causes it
        to step in to the door of its destination. Marks it as not alive.

        Returns:
            bool: Notes if the agent has reached its final destination
        """
        if (self.x, self.y) == self.target:
            (self.x, self.y) = self.target_door
            self.alive = False
            return True
        return False

    def _exit_if_appropriate(self):
        """Causes the agent to 'exit' its start building. This should only fire
        on an agent's first step, and causes it to step to the entry of it's
        building.

        Returns:
            bool: Notes if the agent has exited
        """
        if not self._has_exited:
            self._move_to(self.src_entry)
            self._has_exited = True
            return True
        return False

    def move(self):
        """Orchestrates an agent moving itself. Performs checks to return early,
        calls the pathfinder."""
        self.age += 1

        if self._expire_if_stale():
            return
        if self._exit_if_appropriate():
            return
        if self._arrive_if_reached():
            return

        result = self.pathfinder.next_step(self)
        if result is None:
            self.alive = False
        else:
            self._move_to(result)
            self._calm_if_repeating()
            self.recent_positions.append((self.x, self.y))
