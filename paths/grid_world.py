import numpy as np
from .distributions import generate_hill


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
        self.elevation = np.full((height, width), 0.0)
        self.alpha = alpha
        self.update_costs([])
        hill_sigma = 10
        self.elevation += generate_hill(
            (int(width / 2), int(height / 2)), (width, height), hill_sigma
        )
        grad_y, grad_x = np.gradient(self.elevation)
        self.elevation_grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    def update_costs(self, buildings):
        """Recalculates `.costs`, using .uses, .default_cost, .alpha,
        .cost_scale and making buildings impassable.

        Args:
            buildings (list): Current buildings
        """
        self.costs = np.full((self.height, self.width), float(self.default_cost))
        self.costs -= self.uses**self.alpha * self.cost_scale
        self.costs = np.clip(self.costs, 1, None)
        for building in buildings:
            for tile_x, tile_y in building.tiles:
                self.costs[tile_y, tile_x] = 0
