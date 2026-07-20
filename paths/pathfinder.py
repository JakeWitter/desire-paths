from abc import ABC, abstractmethod

import numpy as np
from enum import Enum

CARDINAL = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAGONAL = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
DIA_COST = np.sqrt(2)


class PathfinderType(Enum):
    ASTAR = "AStar"
    FLOW = "FlowField"


class PathfinderBackend(ABC):
    pathfinder_type: PathfinderType
    temperature: float = 1.0
    slope_scale: float = 1.0
    adventurousness: float = 1.0
    diagonal = True
    recalculate_every = 5

    @abstractmethod
    def next_step(self, agent) -> tuple[int, int] | None: ...

    @abstractmethod
    def update(self, costs, buildings, update_buildings) -> None: ...

    @abstractmethod
    def slope_field(self, direction: tuple[int, int]) -> np.ndarray: ...

    def remove_agent(self, agent_id):
        pass

    # combined_adv = 1 → t = 0 → pure field behaviour, same for adv ≤ 1
    # combined_adv > 1 → t > 0 → field influence shrinks, noise and momentum matter
    #     more
    def adventurousness_blend(self, agent) -> float:
        return np.clip(1 - 1 / (agent.adventurousness * self.adventurousness), 0, 1)

    def noise_cost(self, agent) -> np.ndarray:
        return self.temperature * agent.temperature * agent.noise_field
