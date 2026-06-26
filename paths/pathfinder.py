from abc import ABC, abstractmethod

from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.heuristic import octile, euclidean
from pathfinding.core.diagonal_movement import DiagonalMovement

import streamlit as st

from .grid_world import GridWorld

import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import numpy as np

CARDINAL = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAGONAL = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
DIA_COST = np.sqrt(2)


class PathfinderBackend(ABC):
    @abstractmethod
    def next_step(
        self, agent_id, x, y, target_x, target_y, vx, vy
    ) -> tuple[int, int] | None: ...

    @abstractmethod
    def update(self, costs, buildings) -> None: ...

    @abstractmethod
    def streamlit_controls(self) -> None: ...

    def remove_agent(self, agent_id):
        pass


class AStarBackend(PathfinderBackend):
    def __init__(self, world, diagonal, recalculate_every) -> None:
        self.world = world
        self.diagonal = diagonal
        self.recalculate_every = recalculate_every
        self._cache = dict()

    def next_step(
        self, agent_id, x, y, target_x, target_y, vx, vy
    ) -> tuple[int, int] | None:
        if agent_id not in self._cache.keys():
            path = self.calculate_path(x, y, target_x, target_y)
            if not path or len(path) < 2:
                return None
            self._cache[agent_id] = (path, 1)
            return (path[1].x, path[1].y)

        path, step = self._cache[agent_id]
        if step + 1 >= len(path):
            return None
        loc = path[step + 1]

        if step > 0 and step % self.recalculate_every == 0:
            path = self.calculate_path(x, y, target_x, target_y)
            if not path:
                return None
            self._cache[agent_id] = (path, 1)
            return (path[1].x, path[1].y) if len(path) > 1 else None

        self._cache[agent_id] = [path, step + 1]
        return (loc.x, loc.y)

    def calculate_path(self, x, y, target_x, target_y):
        self.world.grid.cleanup()

        start_node = self.world.grid.node(x, y)
        target_node = self.world.grid.node(target_x, target_y)

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

        try:
            path, _ = finder.find_path(start_node, target_node, self.world.grid)
        except AttributeError:
            path = []

        start_node.walkable = original_start_walkable
        target_node.walkable = original_target_walkable

        return path

    def remove_agent(self, agent_id):
        self._cache.pop(agent_id, None)

    def streamlit_controls(self) -> None:
        self.recalculate_every = st.slider(
            "Recalculate paths every:", 1, 50, self.recalculate_every
        )

    def update(self, costs, buildings) -> None:
        pass


class FieldFlowBackend(PathfinderBackend):
    def __init__(self, world, temperature, diagonal=True, recalculate_every=5):
        self.world: GridWorld = world
        self.temperature: float = temperature
        self.recalculate_every = recalculate_every
        self._fields = {}
        self._steps_since_update = -1
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

    def update(self, costs, buildings) -> None:
        self._steps_since_update += 1
        if self._steps_since_update % self.recalculate_every == 0:
            self._steps_since_update = 0
            height, width = costs.shape
            n = height * width
            costs = np.maximum(costs, 0.01).copy()
            for b in buildings:
                costs[b.y, b.x] = np.inf
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
                weights = costs[dst_y, dst_x] * scale
                if dx != 0 and dy != 0:
                    valid = (costs[src_y, src_x + dx] < np.inf) & (
                        costs[src_y + dy, src_x] < np.inf
                    )
                    weights = weights * valid

                rows.append(src.flatten())
                cols.append(dst.flatten())
                data.append(weights.flatten())

            graph = sp.csr_matrix(
                (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                shape=(n, n),
            )

            self._fields = {}
            for b in buildings:
                src = b.y * width + b.x
                dist = csg.dijkstra(graph, indices=src)
                self._fields[(b.x, b.y)] = dist.reshape(height, width)

    def next_step(
        self, agent_id, x, y, target_x, target_y, vx, vy
    ) -> tuple[int, int] | None:
        height, width = self.world.height, self.world.width

        if (target_x, target_y) not in self._fields.keys():
            return None

        if (x, y) == (target_x, target_y):
            return None

        neighbours = []

        for dx, dy in self.dxys:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                cost = self._fields[(target_x, target_y)][ny, nx]
                neighbours.append(((nx, ny), cost))
        if not neighbours:
            return None

        positions, _ = zip(*neighbours)
        mom_costs = []
        for pos, cost in neighbours:
            dx, dy = pos[0] - x, pos[1] - y
            dot = dx * vx + dy * vy
            mom_costs.append(cost - self.momentum_weight * dot)

        costs = np.array(mom_costs)
        if self.temperature == 0:
            return positions[np.argmin(costs)]
        else:
            costs -= costs.min()
            weights = np.exp(-costs / self.temperature)
            weights = np.where(np.isfinite(weights), weights, 0.0)
            if weights.sum() == 0:
                return positions[np.argmin(costs)]
            weights /= weights.sum()
            idx = np.random.choice(len(positions), p=weights)
            return positions[idx]

    def streamlit_controls(self) -> None:
        self.recalculate_every = st.slider(
            "Recalculate fields every:", 1, 50, self.recalculate_every
        )
        self.temperature = st.slider("Temperature", 0.0, 5.0, self.temperature)
        self.momentum_weight = st.slider("Momentum weight", 0.0, 5.0, 0.5)
