# src/building.py
import numpy as np


class Building:
    def __init__(self, x, y, agent_spawn_rate, current_time):
        self.x = x
        self.y = y
        self.pos = (x, y)
        self.agent_spawn_rate = agent_spawn_rate
        self.set_next_agent_spawn(current_time)

    def set_next_agent_spawn(self, current_time):
        self.next_agent_time = current_time + np.random.exponential(
            1 / self.agent_spawn_rate
        )
