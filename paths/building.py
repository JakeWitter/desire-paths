# src/building.py
import numpy as np
from enum import Enum
from .pathfinder import CARDINAL
from random import choice

POSS_BUILD_WIDTH = (1, 7)
POSS_BUILD_HEIGHT = (1, 7)


class BuildingType(Enum):
    NORM = "norm"
    SEED = "seed"


class Building:
    def __init__(
        self,
        width,
        height,
        tiles,
        door,
        entry,
        agent_spawn_rate,
        current_time,
        global_agent_factor,
        attractiveness=1.0,
        building_type: BuildingType = BuildingType.NORM,
    ):
        self.width = width
        self.height = height
        self.tiles = tiles
        self.door = door
        self.entry = entry
        self.agent_spawn_rate = agent_spawn_rate
        self.attractiveness = attractiveness
        self.set_next_agent_spawn(current_time, global_agent_factor)
        self.building_type = building_type
        self.centroid = (
            sum(x for x, _ in tiles) / len(tiles),
            sum(y for _, y in tiles) / len(tiles),
        )

    def set_next_agent_spawn(self, current_time: int, global_agent_factor: float):
        """Samples an exponential for time until next agent based on
        self.agent_spawn_rate, scaled according to global_agent_factor.

        Args:
            current_time (int): The time now
            global_agent_factor (float64): Scales the exponential variable
        """
        self.next_agent_time = (
            current_time
            + global_agent_factor * np.random.exponential(1 / self.agent_spawn_rate)
        )

    @staticmethod
    def find_door(tiles, occupied, pos, building_size, world_size):
        x, y = pos
        bw, bh = building_size
        ww, wh = world_size

        door_candidates = []
        for tile in tiles:
            if (tile[0] in [x, x + bw - 1]) or tile[1] in [y, y + bh - 1]:
                for dx, dy in CARDINAL:
                    if (
                        ((tile[0] + dx, tile[1] + dy) not in occupied)
                        and (tile[0] + dx, tile[1] + dy) not in tiles
                        and 0 <= tile[0] + dx < ww
                        and 0 <= tile[1] + dy < wh
                    ):
                        door_candidates.append((tile, (tile[0] + dx, tile[1] + dy)))

        if not door_candidates:
            return None, None
        return choice(door_candidates)
