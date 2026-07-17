from .grid_world import GridWorld
from .pathfinder import PathfinderBackend, PathfinderType, CARDINAL, DIAGONAL, DIA_COST

import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import numpy as np


class FlowFieldBackend(PathfinderBackend):
    def __init__(self, world, temperature=1.0, diagonal=True, recalculate_every=5):
        self.pathfinder_type = PathfinderType.FLOW
        self.world: GridWorld = world
        self.temperature: float = temperature
        self.recalculate_every = recalculate_every
        self._fields = {}
        self.momentum_weight = 1
        self.diagonal = diagonal
        self.update_diagonals()
        self.ys, self.xs = np.mgrid[0 : self.world.height, 0 : self.world.width]

    def update_diagonals(self):
        self.dxys = CARDINAL + DIAGONAL if self.diagonal else CARDINAL

    def _dslice(self, d):
        if d > 0:
            return slice(None, -d)
        if d < 0:
            return slice(-d, None)
        if d == 0:
            return slice(None, None)

    def slope_edge_cost(self, src, dst):
        dst_x, dst_y = dst
        src_x, src_y = src
        return self.slope_scale * (
            self.world.elevation[dst_y, dst_x] - self.world.elevation[src_y, src_x]
        )

    def update(self, costs, buildings, update_buildings) -> None:
        height, width = costs.shape
        n = height * width
        costs = np.maximum(costs, 0.01).copy()
        for b in buildings:
            for tile_x, tile_y in b.tiles:
                costs[tile_y, tile_x] = np.inf
        rows = []
        cols = []
        data = []

        for dx, dy in self.dxys:
            x_slice = self._dslice(dx)
            y_slice = self._dslice(dy)

            src_y = self.ys[y_slice, x_slice]
            src_x = self.xs[y_slice, x_slice]
            dst_x = src_x + dx
            dst_y = src_y + dy

            scale = DIA_COST if dx != 0 and dy != 0 else 1.0
            src = src_y * width + src_x
            dst = dst_y * width + dst_x
            elevation_cost = self.slope_edge_cost((src_x, src_y), (dst_x, dst_y))
            weights = costs[dst_y, dst_x] * scale + elevation_cost
            weights = np.maximum(weights, 0.01)
            if dx != 0 and dy != 0:
                valid = (costs[src_y, src_x + dx] < np.inf) & (
                    costs[src_y + dy, src_x] < np.inf
                )
                weights[~valid] = np.inf

            rows.append(src.flatten())
            cols.append(dst.flatten())
            data.append(weights.flatten())

        graph = sp.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        )

        new_fields = {}
        for b in update_buildings:
            entry_x, entry_y = b.entry
            src = entry_y * width + entry_x
            dist = csg.dijkstra(graph, indices=src)
            new_fields[b.entry] = dist.reshape(height, width)

        if buildings == update_buildings:
            self._fields = new_fields
        else:
            self._fields.update(new_fields)

    def next_step(self, agent) -> tuple[int, int] | None:
        width, height = self.world.width, self.world.height

        # check target has a flowfield calculated
        if agent.target not in self._fields:
            return None
        # nothing to do if the agent is at its target
        if (agent.x, agent.y) == agent.target:
            return None

        neighbours = []
        movement_costs = []

        t = self.adventurousness_blend(agent)
        noise = self.noise_cost(agent)

        # loop through possible movements, check plausibility and calculate cost,
        # influenced by adventurousness, momentum, temperature scaled noise
        for dx, dy in self.dxys:
            nx, ny = agent.x + dx, agent.y + dy
            if 0 <= nx < width and 0 <= ny < height:
                # immediately step to target if available
                if (nx, ny) == agent.target:
                    return (nx, ny)
                # if diagonal, check if it's cutting a building corner. If so, continue
                if dx != 0 and dy != 0:
                    if (
                        self.world.costs[agent.y, nx] == 0
                        or self.world.costs[ny, agent.x] == 0
                    ):
                        continue
                # see t equation above
                field = self._fields.get(agent.target)
                if field is None:
                    return None
                cost = (1 - t) * field[ny, nx] + t
                if not np.isfinite(cost):
                    continue
                if cost == 0:
                    return (nx, ny)
                cost += noise[ny, nx]
                # disincentivise recent_positions. Hacky, reconsider sometime
                if (nx, ny) in agent.recent_positions:
                    cost += 5000.0
                neighbours.append((nx, ny))

                # apply momentum formula
                dot = dx * agent.vx + dy * agent.vy
                movement_costs.append(cost - self.momentum_weight * dot)

        if not neighbours:
            return None

        return neighbours[np.argmin(movement_costs)]
