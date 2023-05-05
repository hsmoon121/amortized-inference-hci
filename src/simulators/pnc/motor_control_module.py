"""
An implementation of "Point-and-click model"
from Seungwon et al., "A simulation model of intermittently controlled point-and-click behaviour." CHI 2021.

Original code: https://github.com/dodoseung/point-and-click-simulation-model
"""

import os
from pathlib import Path
import numpy as np
import pickle
from tqdm import tqdm

from .model_config import CONFIG
from collections import defaultdict

G = np.array([[1, CONFIG["INTERVAL"]], [0, 1]])
H = np.array([[CONFIG["INTERVAL"] * CONFIG["INTERVAL"] / 2], [CONFIG["INTERVAL"]]])


def get_Gs():
    Gs, GTs = [np.array([[1, 0], [0, 1]])], [np.array([[1, 0], [0, 1]])]
    for i in range(1, 64):
        Gs.append(Gs[-1] @ G)
        GTs.append(GTs[-1] @ G.T)
    return Gs, GTs


Gs, GTs = get_Gs()


def get_gram(n, k):
    gram = np.zeros((2, 2))
    for j in range(0, k):
        gram += Gs[j] @ H @ H.T @ GTs[n - k + j]
    return gram


def get_grammians():
    Grams, Gram_invs = defaultdict(dict), dict()
    for n in tqdm(range(64)):
        for k in range(n + 1):
            Grams[n][k] = get_gram(n, k)
        if n > 1:
            Gram_invs[n] = np.linalg.inv(get_gram(n, n))
    return Grams, Gram_invs

gram_fpath = os.path.join(Path(__file__).parent, "materials", "grams.pkl")
if not os.path.exists(gram_fpath):
    Grams, Gram_invs = get_grammians()
    with open(gram_fpath, "wb") as fp:
        pickle.dump((Grams, Gram_invs), fp)
else:
    with open(gram_fpath, "rb") as fp:
        Grams, Gram_invs = pickle.load(fp)

# Optimal trajectory generation
# Pos0 and Vel0 are the position and velocity of initial state.
# PosN and VelN are the position and velocity of final state.
def otg_new(pos_0, vel_0, pos_n, vel_n, n, num):
    x_0 = np.array([[pos_0, vel_0]]).T
    x_n = np.array([[pos_n, vel_n]]).T

    g_n = Gs[n]
    inv_gram_n = Gram_invs[n]
    span = inv_gram_n @ (x_n - g_n @ x_0)

    xs = []
    for k in range(0, num + 1):
        g_k = Gs[k]
        gram_k = Grams[n][k]
        xs.append((g_k @ x_0) + gram_k @ span)

    xs = np.concatenate(xs, axis=1)
    return xs


def __boundary(size, pos, vel, radius):
    if size == 0:
        return pos, vel

    width, height = CONFIG["WINDOW_WIDTH"], CONFIG["WINDOW_HEIGHT"]

    pos_x, pos_y = pos[0], pos[1]
    vel_x, vel_y = vel[0], vel[1]

    # x
    bound = width

    upper_bound = bound - radius[0]
    lower_bound = radius[0]

    for i in range(size):
        next_pos = pos_x + (CONFIG["INTERVAL"] * vel_x)
        if upper_bound >= next_pos >= lower_bound:
            pos_x = next_pos
        elif next_pos > upper_bound:
            pos_x = (2 * upper_bound) - next_pos
            vel_x *= -1
        elif next_pos < lower_bound:
            pos_x = (2 * lower_bound) - next_pos
            vel_x *= -1
    # y
    bound = height

    upper_bound = bound - radius[0]
    lower_bound = radius[0]

    for i in range(size):
        next_pos = pos_y + (CONFIG["INTERVAL"] * vel_y)
        if upper_bound >= next_pos >= lower_bound:
            pos_y = next_pos
        elif next_pos > upper_bound:
            pos_y = (2 * upper_bound) - next_pos
            vel_y *= -1
        elif next_pos < lower_bound:
            pos_y = (2 * lower_bound) - next_pos
            vel_y *= -1

    return np.array([pos_x, pos_y]), np.array([vel_x, vel_y])


def boundary(idx, pos, vel, radius):
    if idx == 0:
        return pos, vel

    width, height = CONFIG["WINDOW_WIDTH"], CONFIG["WINDOW_HEIGHT"]

    pos_x, pos_y = pos[0], pos[1]
    vel_x, vel_y = vel[0], vel[1]

    final_pos = pos + vel * idx * CONFIG["INTERVAL"]
    final_vel = vel.copy()

    for i, bound in enumerate([width, height]):
        upper_bound = bound - radius[0]
        lower_bound = radius[0]

        while not (lower_bound <= final_pos[i] <= upper_bound):
            if final_pos[i] < lower_bound:
                final_pos[i] = (2 * lower_bound) - final_pos[i]
            else:
                final_pos[i] = (2 * upper_bound) - final_pos[i]
            final_vel[i] *= -1

    return final_pos, final_vel
