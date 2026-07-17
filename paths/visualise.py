import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

from .pathfinder_astar import AStarBackend
from .world_manager import WorldManager
from .distributions import building_spawn_field

_building_cmap = LinearSegmentedColormap.from_list("building", ["black", "red"])


def world_draw(manager: WorldManager, view: str = "Paths"):
    fig, ax = plt.subplots()
    if view == "Paths":
        data = manager.world.costs
    elif view == "Bld. prob":
        data = building_spawn_field(
            manager.world.width, manager.world.height, manager.buildings
        )
    else:
        data = manager.world.elevation
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

        min_x = min(x for x, _ in building.tiles)
        min_y = min(y for _, y in building.tiles)

        ax.add_patch(
            patches.Rectangle(
                (min_x - 0.5, min_y - 0.5),
                building.width,
                building.height,
                facecolor=colour,
                edgecolor="white",
                linewidth=lw,
            )
        )
        door_x, door_y = building.door
        entry_x, entry_y = building.entry
        marker_x = door_x + 0.5 * (entry_x - door_x)
        marker_y = door_y + 0.5 * (entry_y - door_y)
        ax.plot(marker_x, marker_y, "o", markersize=3, color="white")
    # ax.grid(visible=True, color="black", linestyle="-", linewidth=0.5)
    if len(manager.agents) <= 60:
        cmap = plt.get_cmap("tab20")
        for agent in manager.agents:
            colour = cmap(agent.id % 20)
            ax.plot(agent.x, agent.y, "o", markersize=6, color=colour)

            if len(manager.agents) <= 15:
                if (
                    isinstance(manager.pathfinder, AStarBackend)
                    and agent.id in manager.pathfinder._cache
                ):
                    path, step = manager.pathfinder._cache[agent.id]
                    path_x = [node.x for node in path[step:]]
                    path_y = [node.y for node in path[step:]]
                    ax.plot(path_x, path_y, "--", linewidth=1, color=colour)
    return fig
