import numpy as np
from pathfinding.core.grid import Grid


class GridWorld:
    def __init__(
        self,
        width,
        height,
        default_cost=10.0,
        cost_scale=1.0,
        alpha=1.0,
    ):
        self.width = width
        self.height = height
        self.default_cost = default_cost
        self.cost_scale = cost_scale
        self.uses = np.full((height, width), 0)
        self.grid = Grid(width=width, height=height)
        self.alpha = alpha
        self._prev_costs = np.full((self.height, self.width), np.inf)
        self.update_costs([])

    def update_costs(self, buildings):
        self.costs = np.full((self.height, self.width), float(self.default_cost))
        self.costs -= self.uses**self.alpha * self.cost_scale
        self.costs = np.clip(self.costs, 1, None)
        for building in buildings:
            for tile_x, tile_y in building.tiles:
                self.costs[tile_y, tile_x] = 0
        self.grid.cleanup()

        changed_ys, changed_xs = np.where(self.costs != self._prev_costs)
        for y, x in zip(changed_ys, changed_xs):
            cost = self.costs[y, x]
            self.grid.update_node(x, y, weight=cost, walkable=cost > 0)
        self._prev_costs = self.costs.copy()
