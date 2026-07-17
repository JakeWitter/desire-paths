import numpy as np


def distance_grid(width, height, x, y):
    ys, xs = np.mgrid[0:height, 0:width]
    return np.sqrt((xs - x) ** 2 + (ys - y) ** 2)


def seed_positions(n, width, height):
    """Generates `n` seed positions on the grid, for initial buildings.

    Args:
        n (int): Number of seeds to place
        width (int): Width of the grid
        height (int): Height of the grid

    Returns:
        list: List of (x,y) tuples for each seed position.
    """
    ys, xs = np.mgrid[0:height, 0:width]

    x_margin = int(width * 0.08)
    y_margin = int(height * 0.08)
    x_mask = np.minimum(xs, width - 1 - xs) >= x_margin
    y_mask = np.minimum(ys, height - 1 - ys) >= y_margin
    edge_mask = (x_mask & y_mask).astype(float)

    positions = []
    for s in range(n):
        if s == 0:
            min_dist = np.ones((height, width))
        else:
            min_dist = np.full((height, width), np.inf)
            for sx, sy in positions:
                d = np.sqrt((xs - sx) ** 2 + (ys - sy) ** 2)
                np.minimum(min_dist, d, out=min_dist)
        weights = (min_dist**10 * edge_mask).flatten()
        weights /= weights.sum()
        idx = np.random.choice(len(weights), p=weights)
        y, x = divmod(idx, width)
        positions.append((int(x), int(y)))
    return positions


def gaussian(x, y, strength, sigma):
    """Generates a closure for a Gaussian using parameters, and scaled.

    Args:
        x (int): x_coord for Gaussian peak
        y (int): y_coord for Gaussian peak
        strength (float): scaling factor for Gaussian
        sigma (float): Gaussian variance

    Returns:
        Callable[[int, int], np.ndarray]: Function that evaluates the Gaussian
    """

    def evaluate(width, height):
        dist = distance_grid(width, height, x, y)
        return strength * np.exp(-(dist**2) / (2 * sigma**2))

    return evaluate


def uniform():
    """Generates a closure for a uniform distribution. Used for assigning flat
    weight to all cells

    Returns:
        Callable[[int, int], np.ndarray]: Function that evaluates this distribution
    """

    def evaluate(width, height):
        return np.ones((height, width))

    return evaluate


def building_spawn_field(
    width,
    height,
    buildings,
    repulsion=10.0,
    repulsion_sigma=0.75,
    attraction=30.0,
    attraction_sigma=3.5,
):
    """Creates a 2d array for spawn weight, or unnormalised probability.
    Works by creating a list of distributions, based on buildings among other things,
    that are evaluated per cell.

    Args:
        width (int): Width of the grid
        height (int): Height of the grid
        buildings (list): Buildings attract and repel, marked as not spawnable cells
        repulsion (float): Building repulsion effect
        repulsion_sigma (float): Width of building repulsion effect
        attraction (float): Building attraction effect
        attraction_sigma (float): Width of building attraction effect

    Returns:
        np.ndarray: Spawn weight field
    """
    distrs = [uniform()]
    for building in buildings:
        cx, cy = building.centroid
        distrs.append(gaussian(cx, cy, -repulsion, repulsion_sigma))
        distrs.append(gaussian(cx, cy, attraction, attraction_sigma))

    field = sum(d(width, height) for d in distrs)
    field = np.clip(field, 0, None)
    for building in buildings:
        for tile in building.tiles:
            field[tile[1], tile[0]] = 0
    return field


def generate_hill(cpos, world_size, sigma):
    """

    Args:
        cpos ():
        world_size ():
        sigma ():

    Returns:

    """
    cx, cy = cpos
    width, height = world_size
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    return np.exp(-dist_sq / (2 * sigma**2))
