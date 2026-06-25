# src/building.py
import numpy as np


class Building:
    def __init__(self, x, y, agent_spawn_rate, current_time, global_agent_factor, attractiveness=1.0):
        self.x = x
        self.y = y
        self.pos = (x, y)
        self.agent_spawn_rate = agent_spawn_rate
        self.attractiveness = attractiveness
        self.set_next_agent_spawn(current_time, global_agent_factor)

    def set_next_agent_spawn(self, current_time, global_agent_factor):
        self.next_agent_time = (
            current_time
            + global_agent_factor * np.random.exponential(1 / self.agent_spawn_rate)
        )
