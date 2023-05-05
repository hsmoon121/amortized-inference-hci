"""
An implementation of "Point-and-click model"
from Seungwon et al., "A simulation model of intermittently controlled point-and-click behaviour." CHI 2021.

Original code: https://github.com/dodoseung/point-and-click-simulation-model
"""

import math
import numpy as np
from .motor_control_module import boundary


### Perceiving target position and velocity
dist_head_to_monitor = 0.63
sample = 1


def visual_speed_noise(_vel, sigma):
    vel = (_vel ** 2).sum() ** 0.5
    if vel <= 0:
        vel = np.finfo(float).tiny

    v_set = []
    for i in range(sample):
        v = 2 * math.degrees(math.atan(vel / (2 * dist_head_to_monitor)))
        v_0 = 0.3
        v_hat = math.log(1 + v / v_0)
        # sigma = 0.15 * 1
        v_prime = np.random.lognormal(v_hat, sigma)
        v_final = (v_prime - 1) * v_0
        v_final = dist_head_to_monitor * 2 * math.tan(math.radians(v_final) / 2)
        if v_final < 0:
            v_final = 0
        v_set.append(v_final)

    v_final = sum(v_set) / sample
    ratio = _vel / vel
    vel = v_final * ratio
    return vel


def add_noise_past(time, pos, vel, radius, sigma):
    pos, vel = boundary(time, pos, -vel, radius)
    vel = visual_speed_noise(-vel, sigma)
    pos, vel = boundary(time, pos, vel, radius)
    return pos, vel
