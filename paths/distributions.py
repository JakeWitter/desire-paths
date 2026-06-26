import numpy as np


def distance_grid(width, height, x, y):
    ys, xs = np.mgrid[0:height, 0:width]
    return np.sqrt((xs - x) ** 2 + (ys - y) ** 2)


def seed_positions(n, width, height):
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
    def evaluate(width, height):
        dist = distance_grid(width, height, x, y)
        return strength * np.exp(-(dist**2) / (2 * sigma**2))

    return evaluate


def uniform():
    def evaluate(width, height):
        return np.ones((height, width))

    return evaluate


def building_spawn_prob(
    width,
    height,
    buildings,
    repulsion=10.0,
    repulsion_sigma=0.75,
    attraction=30.0,
    attraction_sigma=3.5,
):
    distrs = [uniform()]
    for building in buildings:
        distrs.append(gaussian(building.x, building.y, -repulsion, repulsion_sigma))
        distrs.append(gaussian(building.x, building.y, attraction, attraction_sigma))

    prob = sum(d(width, height) for d in distrs)
    prob = np.clip(prob, 0, None)
    for building in buildings:
        prob[building.y, building.x] = 0
    return prob
