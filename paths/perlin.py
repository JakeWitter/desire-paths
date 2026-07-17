import numpy as np
import math


def build_gradients(cells, seed):
    rng = np.random.default_rng(seed)
    cells_w, cells_h = cells
    angles = rng.uniform(0, 2 * math.pi, size=(cells_h + 1, cells_w + 1))
    grid = np.stack([np.cos(angles), np.sin(angles)], axis=2)
    return grid


def perlin_value_at(p, world_size, grid, cells):
    def dot_for_corner(corner_offs, i, j, u, v, grid):
        corner_offs_x, corner_offs_y = corner_offs
        grad = grid[j + corner_offs_y, i + corner_offs_x]
        dist_vec = (u - corner_offs_x, v - corner_offs_y)
        return grad @ dist_vec

    world_w, world_h = world_size
    px, py = p
    cells_w, cells_h = cells
    cell_w, cell_h = world_w / cells_w, world_h / cells_h

    i, j = math.floor(px / cell_w), math.floor(py / cell_h)
    u, v = (px / cell_w) - i, (py / cell_h) - j

    v00 = dot_for_corner((0, 0), i, j, u, v, grid)
    v01 = dot_for_corner((0, 1), i, j, u, v, grid)
    v10 = dot_for_corner((1, 0), i, j, u, v, grid)
    v11 = dot_for_corner((1, 1), i, j, u, v, grid)

    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def lerp(a, b, t):
        return a + t * (b - a)

    fu, fv = fade(u), fade(v)
    top = lerp(v00, v10, fu)
    bottom = lerp(v01, v11, fu)
    value = lerp(top, bottom, fv)
    return value


def one_perlin(world_size, wavelength, seed):
    world_w, world_h = world_size

    cells_w = max(1, round(world_w / wavelength))
    cells_h = max(1, round(world_h / wavelength))
    cells = cells_w, cells_h

    grid = build_gradients(cells, seed)
    world = np.zeros((world_h, world_w))

    for py in range(0, world_h):
        for px in range(0, world_w):
            world[py, px] = perlin_value_at((px, py), world_size, grid, cells)

    return world


def octaves(world_size, wavelengths, amps, seed=None):
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**32)
    world_w, world_h = world_size
    world = np.zeros((world_h, world_w))
    for i, (cwh, amp) in enumerate(zip(wavelengths, amps)):
        world += one_perlin(world_size, cwh, seed + i) * amp
    return world
