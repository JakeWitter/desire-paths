from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.heuristic import octile, euclidean
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
import numpy as np

from .pathfinder import PathfinderBackend, PathfinderType
from .grid_world import GridWorld


class AStarBackend(PathfinderBackend):
    def __init__(self, world, diagonal, recalculate_every) -> None:
        self.pathfinder_type = PathfinderType.ASTAR
        self.recalculate_every = recalculate_every
        self.world: GridWorld = world
        self.diagonal = diagonal
        self._cache = dict()
        self.temperature = 1.0

    def next_step(self, agent) -> tuple[int, int] | None:
        if agent.id not in self._cache:
            path = self.calculate_path(agent)
            if not path or len(path) < 2:
                return None
            self._cache[agent.id] = (path, 1)
            return (path[1].x, path[1].y)

        path, step = self._cache[agent.id]
        if step + 1 >= len(path):
            return None
        loc = path[step + 1]

        if step > 0 and step % self.recalculate_every == 0:
            path = self.calculate_path(agent)
            if not path:
                return None
            self._cache[agent.id] = (path, 1)
            return (path[1].x, path[1].y) if len(path) > 1 else None

        self._cache[agent.id] = [path, step + 1]
        return (loc.x, loc.y)

    def slope_field(self, direction: tuple[int, int]) -> np.ndarray:
        return self.elevation_cost()

    def elevation_cost(self):
        return self.slope_scale * self.world.elevation_grad_mag

    def calculate_path(self, agent):
        t = self.adventurousness_blend(agent)
        flat_costs = (1 - t) * self.world.costs + t
        noisy_costs = flat_costs + self.noise_cost(agent)
        noisy_costs += self.elevation_cost()
        noisy_costs[self.world.costs == 0] = 0
        noisy_grid = Grid(matrix=noisy_costs)

        start_node = noisy_grid.node(agent.x, agent.y)
        target_node = noisy_grid.node(*agent.target)

        # Initialize the A* Finder with diagonal movement enabled
        if self.diagonal:
            finder = AStarFinder(
                heuristic=octile,
                diagonal_movement=DiagonalMovement.only_when_no_obstacle,
            )
        else:
            finder = AStarFinder(heuristic=euclidean)

        try:
            path, _ = finder.find_path(start_node, target_node, noisy_grid)
        except AttributeError:
            path = []

        return path

    def remove_agent(self, agent_id):
        self._cache.pop(agent_id, None)

    def update(self, costs, buildings, update_buildings) -> None:
        pass
