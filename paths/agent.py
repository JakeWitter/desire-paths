# paths/agent.py

from .grid_world import GridWorld
from .building import Building

from random import randint
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.heuristic import octile, euclidean
from pathfinding.core.diagonal_movement import DiagonalMovement


class Agent:
    def __init__(
        self,
        world: "GridWorld",
        x=0,
        y=0,
        target_x=5,
        target_y=5,
        diagonal=True,
        recalculate_every=5,
    ):
        self.world = world
        self.x, self.y = x, y
        self.target_x, self.target_y = target_x, target_y
        self.diagonal = diagonal
        self.path = self.calculate_path()
        self.step = 0
        self.alive = True
        self.id = randint(0, 100000)
        self.recalculate_every = recalculate_every

    @classmethod
    def from_buildings(
        cls,
        world: "GridWorld",
        b1: Building,
        b2: Building,
        diagonal=True,
        recalculate_every=5,
    ):
        """Alternative constructor that creates an Agent starting from b1 and targeting b2."""
        agent = cls(
            world,
            x=b1.x,
            y=b1.y,
            target_x=b2.x,
            target_y=b2.y,
            diagonal=diagonal,
            recalculate_every=recalculate_every,
        )
        return agent

    def calculate_path(self):
        self.world.grid.cleanup()
        # Retrieve the start and target nodes from the world grid
        start_node = self.world.grid.node(self.x, self.y)
        target_node = self.world.grid.node(self.target_x, self.target_y)

        # Store original weights for potential buildings at start/end nodes
        original_start_walkable = start_node.walkable
        original_target_walkable = target_node.walkable

        start_node.walkable = True
        target_node.walkable = True

        # Initialize the A* Finder with diagonal movement enabled
        if self.diagonal:
            finder = AStarFinder(
                heuristic=octile,
                diagonal_movement=DiagonalMovement.only_when_no_obstacle,
            )
        else:
            finder = AStarFinder(heuristic=euclidean)

        path, _ = finder.find_path(start_node, target_node, self.world.grid)
        # print(f"Path found: {path}")

        # Restore the original weights for start and end nodes
        start_node.walkable = original_start_walkable
        target_node.walkable = original_target_walkable

        return path

    def move(self):
        self.step += 1
        if self.step >= len(self.path):
            self.alive = False
            return

        node = self.path[self.step]
        self.x, self.y = node.x, node.y

        if self.recalculate_every and self.step % self.recalculate_every == 0:
            self.path = self.calculate_path()
            self.step = 0
