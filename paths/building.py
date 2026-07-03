# src/building.py
import numpy as np


class Building:
    def __init__(
        self,
        width,
        height,
        tiles,
        door_x,
        door_y,
        agent_spawn_rate,
        current_time,
        global_agent_factor,
        attractiveness=1.0,
        building_type="norm",
    ):
        self.width = width
        self.height = height
        self.tiles = tiles
        self.door_x = door_x
        self.door_y = door_y
        self.agent_spawn_rate = agent_spawn_rate
        self.attractiveness = attractiveness
        self.set_next_agent_spawn(current_time, global_agent_factor)
        self.building_type = building_type
        self.centroid_x = sum(x for x, _ in tiles) / len(tiles)
        self.centroid_y = sum(y for _, y in tiles) / len(tiles)

    def set_next_agent_spawn(self, current_time, global_agent_factor):
        self.next_agent_time = (
            current_time
            + global_agent_factor * np.random.exponential(1 / self.agent_spawn_rate)
        )
