import numpy as np


def distance_grid(width, height, x, y):
    ys, xs = np.mgrid[0:height, 0:width]
    return np.sqrt((xs - x) ** 2 + (ys - y) ** 2)


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
    repulsion=60.0,
    repulsion_sigma=1.0,
    attraction=9.0,
    attraction_sigma=8.0,
    broad_repulsion=4.0,
    broad_attraction=2.0,
):
    distrs = [uniform()]
    for building in buildings:
        distrs.append(gaussian(building.x, building.y, -repulsion, repulsion_sigma))
        distrs.append(gaussian(building.x, building.y, attraction, attraction_sigma))
        distrs.append(gaussian(building.x, building.y, -broad_repulsion, height / 3))
        distrs.append(gaussian(building.x, building.y, broad_attraction, height / 1.5))

    prob = sum(d(width, height) for d in distrs)
    prob = np.clip(prob, 0, None)
    for building in buildings:
        prob[building.y, building.x] = 0
    return prob
