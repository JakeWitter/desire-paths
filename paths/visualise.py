import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from .world_manager import WorldManager
from .distributions import building_spawn_prob

# TODO: move to a global palette module
_building_cmap = LinearSegmentedColormap.from_list("building", ["black", "red"])


def world_draw(manager: WorldManager, show_spawn_prob: bool = False):
    fig, ax = plt.subplots()
    if show_spawn_prob:
        data = building_spawn_prob(
            manager.world.width, manager.world.height, manager.buildings
        )
    else:
        data = manager.world.costs
    ax.imshow(data, cmap="YlGn", origin="upper")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    max_interval = 1 / manager.building_agent_rate
    max_attractiveness = max((b.attractiveness for b in manager.buildings), default=1.0)
    for building in manager.buildings:
        t_remaining = building.next_agent_time - manager.time
        intensity = max(0.0, min(1.0, t_remaining / max_interval))
        colour = _building_cmap(1 - intensity)
        normalised = min(building.attractiveness / max_attractiveness, 1.0)
        t = min(manager.attractiveness_scale / 5.0, 1.0)
        lw = 0.2 + 2.3 * normalised * t
        ax.add_patch(
            patches.Rectangle((building.x - 0.5, building.y - 0.5), 1, 1,
                               facecolor=colour, edgecolor="white", linewidth=lw)
        )
    # ax.grid(visible=True, color="black", linestyle="-", linewidth=0.5)

    cmap = plt.get_cmap("tab20")
    for agent in manager.agents:
        colour = cmap(agent.id % 20)
        ax.plot(agent.x, agent.y, "o", markersize=6, color=colour)

        if len(manager.agents) <= 15:
            if hasattr(agent, "path") and agent.path[agent.step :]:
                path_x, path_y = zip(*agent.path[agent.step :])
                ax.plot(path_x, path_y, "--", linewidth=1, color=colour)
    return fig
