from abc import ABC, abstractmethod

from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.heuristic import octile, euclidean
from pathfinding.core.diagonal_movement import DiagonalMovement

import streamlit as st

from .grid_world import GridWorld

import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import numpy as np


class PathfinderBackend(ABC):
    @abstractmethod
    def next_step(
        self, agent_id, x, y, target_x, target_y
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

    def next_step(self, agent_id, x, y, target_x, target_y) -> tuple[int, int] | None:
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
    def __init__(self, world, temperature, recalculate_every=5):
        self.world: GridWorld = world
        self.temperature: float = temperature
        self.recalculate_every = recalculate_every
        self._fields = {}
        self._steps_since_update = 0

    def update(self, costs, buildings) -> None:
        self._steps_since_update += 1
        if self._steps_since_update % self.recalculate_every == 0:
            self._steps_since_update = 0
            height, width = costs.shape
            n = height * width

            graph = sp.lil_matrix((n, n), dtype=float)

            for y in range(height):
                for x in range(width):
                    src = y * width + x
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            dst = ny * width + nx
                            graph[src, dst] = costs[ny, nx]

            graph = graph.tocsr()

            self._fields = {}
            for b in buildings:
                src = b.y * width + b.x
                dist = csg.dijkstra(graph, indices=src)
                self._fields[(b.x, b.y)] = dist.reshape(height, width)

    def next_step(self, agent_id, x, y, target_x, target_y) -> tuple[int, int] | None:
        height, width = self.world.height, self.world.width

        if (target_x, target_y) not in self._fields.keys():
            return None

        if (x, y) == (target_x, target_y):
            return None

        neighbours = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                cost = self._fields[(target_x, target_y)][ny, nx]
                neighbours.append(((nx, ny), cost))

        if not neighbours:
            return None
        positions, costs = zip(*neighbours)
        costs = np.array(costs)
        if self.temperature == 0:
            return positions[np.argmin(costs)]
        else:
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
