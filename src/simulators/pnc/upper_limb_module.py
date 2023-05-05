"""
An implementation of "Point-and-click model"
from Seungwon et al., "A simulation model of intermittently controlled point-and-click behaviour." CHI 2021.

Original code: https://github.com/dodoseung/point-and-click-simulation-model
"""

from . import mouse_module as mouse
import numpy as np
from .model_config import CONFIG


### Simulating mouse rotation
def mouse_noise(vel_x, vel_y, h_pos_x, h_pos_y, mouseGain):
    dx = (vel_x[1:] + vel_x[:-1]) / 2 * CONFIG["INTERVAL"]
    dy = (vel_y[1:] + vel_y[:-1]) / 2 * CONFIG["INTERVAL"]
    hand_ori_prev = mouse.get_hand_orientation(
        np.array((h_pos_x, h_pos_y)), CONFIG["forearm"]
    )
    for idx in range(len(dx)):
        if dx[idx] != 0 or dy[idx] != 0:
            hand_pos_prev = np.array((h_pos_x, h_pos_y))
            hand_dx, hand_dy = mouse.rot_mat(dx[idx], dy[idx], -hand_ori_prev)
            h_pos_x += hand_dx
            h_pos_y += hand_dy
            mouse_dx, mouse_dy = mouse.get_cursor_displacement(
                hand_pos_prev,
                np.array((h_pos_x, h_pos_y)),
                CONFIG["forearm"],
                mouseGain,
                #mouseGain[idx],
            )
            dx[idx] = mouse_dx
            dy[idx] = mouse_dy

    return dx, dy, h_pos_x, h_pos_y


### Signal-dependent motor noise
def motor_noise(vel_x, vel_y, time, mean_x, mean_y, nc):
    for i in range(1, time + 1):
        v = np.array([vel_x[i], vel_y[i]])
        if np.linalg.norm(v) == 0:
            vel_dir = np.array([0, 0])
        else:
            vel_dir = v / np.linalg.norm(v)
        vel_per = np.array([-vel_dir[1], vel_dir[0]])

        vel = (v[0] ** 2 + v[1] ** 2) ** 0.5
        noise_dir = nc[0] * abs(vel) * np.random.normal(0, 1)
        noise_per = nc[1] * abs(vel) * np.random.normal(0, 1)

        vel_noisy = v + noise_dir * vel_dir + noise_per * vel_per
        vel_x[i] = vel_noisy[0] + mean_x
        vel_y[i] = vel_noisy[1] + mean_y

    pos_x_delta = (vel_x[1:] + vel_x[:-1]) / 2 * CONFIG["INTERVAL"]
    pos_y_delta = (vel_y[1:] + vel_y[:-1]) / 2 * CONFIG["INTERVAL"]

    return np.array(pos_x_delta), np.array(pos_y_delta), vel_x, vel_y
